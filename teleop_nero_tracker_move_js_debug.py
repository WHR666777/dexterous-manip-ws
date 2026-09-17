"""Vive Tracker → 绝对法兰位姿 → 七轴 IK → move_js，独立调试入口。

离线：python teleop_nero_tracker_move_js_debug.py --ik-test
真机：python teleop_nero_tracker_move_js_debug.py （需输入 MOVE）
R 重新对齐；Ctrl+C 仅退出、不失能、不自动急停。跳过发送/断开不保证停止或防坠落。
"""
import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
from robot_control import NeroArm
from robot_control.nero_ik import (
    NeroIK, URDF_PATH, MAX_IK_JOINT_JUMP_DEG, MAX_FK_POSITION_ERROR_M,
    MAX_FK_ROTATION_ERROR_DEG, pose_error, valid_transform, self_test,
)
import teleop_nero_tracker_move_pose_debug as reference

MAX_JOINT_STEP = np.deg2rad(5)  # 保持旧 JS 参数；并非已验证的机械安全速度。
CONTROL_HZ = reference.CONTROL_HZ  # 当前 10Hz，不追赶超时周期、不连发补帧。
LOG_INTERVAL = 0.5
MAX_TRACKER_TRANSLATION_JUMP_M = reference.MAX_POSITION_STEP / reference.POSITION_SCALE
MAX_TRACKER_ROTATION_JUMP_DEG = np.rad2deg(reference.MAX_ROTATION_STEP)
MAX_FEEDBACK_AGE_S = 0.25
MAX_FEEDBACK_SKEW_S = 0.10
MAX_TRACKER_READ_S = 0.10
MAX_TARGET_AGE_S = 0.25  # 本地取得 Tracker 后到发送前的时间；不是源样本年龄。
INITIAL_ALIGNMENT_TIMEOUT_S = 5.0
STARTUP_FK_POSITION_M = 0.01
STARTUP_FK_ROTATION_DEG = 5.0
STATIONARY_TARGET_M = 0.0001
STATIONARY_TARGET_DEG = 0.1
MAX_STATIONARY_REDUNDANCY_DRIFT_DEG = 1.0
LOG = logging.getLogger('nero.js_debug')


def state_ready(feedback):
    status = feedback.get('status') or {}
    enabled = feedback.get('enabled') or []
    return (feedback.get('state_fresh') is True and status.get('arm_status') == 0
            and status.get('ctrl_mode') == 1 and len(enabled) == 7 and all(enabled)
            and not any((status.get('err_status') or {}).values()))


def clip_command(ik, q_ik, previous, actual):
    """有限关节走直接合法路径；不能把跨限位的短角当作可执行路径。"""
    if not all(ik.in_limits(q) for q in (q_ik, previous, actual)):
        raise ValueError('JOINT_LIMIT: IK/上一目标/反馈不是合法七轴角')
    q_cmd = previous + np.clip(np.asarray(q_ik)-previous, -MAX_JOINT_STEP, MAX_JOINT_STEP)
    if (not ik.in_limits(q_cmd) or np.any(np.abs(q_cmd-previous) > MAX_JOINT_STEP+1e-12)
            or np.any(np.abs(q_cmd-actual) > MAX_JOINT_STEP+1e-12)):
        raise ValueError('COMMAND_JUMP: 限幅后的目标与实际关节仍相差超过5°')
    return q_cmd


def pose_columns(row, prefix, T):
    if T is None or not valid_transform(T):
        return
    for label, value in zip(('x', 'y', 'z'), T[:3, 3]):
        row[f'{prefix}_{label}'] = float(value)
    # 欧拉角只用于输出；奇异位姿下表示不唯一，不用其差值控制。
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    for label, value in zip(('roll', 'pitch', 'yaw'), angles):
        row[f'{prefix}_{label}'] = float(value)


def joint_columns(row, prefix, q):
    if q is not None:
        for index, value in enumerate(np.rad2deg(q), 1):
            row[f'{prefix}_{index}'] = float(value)


class RedundancyWatch:
    """近似固定目标下的累计姿态漂移告警；只是疑似冗余变化，不能诊断物理故障。"""
    def __init__(self):
        self.target = self.q = None

    def update(self, target, q):
        if self.target is not None:
            p, r = pose_error(target, self.target)
            if p <= STATIONARY_TARGET_M and r <= np.deg2rad(STATIONARY_TARGET_DEG):
                return bool(np.max(np.abs(q-self.q)) > np.deg2rad(MAX_STATIONARY_REDUNDANCY_DRIFT_DEG))
        self.target, self.q = target.copy(), q.copy()
        return False


class TeleopSession:
    """不拥有硬件；send 只由真机入口注入，也方便离线验证不发送路径。"""
    def __init__(self, ik, tracker_pose, flange_pose, q_actual):
        if not valid_transform(tracker_pose) or not valid_transform(flange_pose) or not ik.in_limits(q_actual):
            raise ValueError('初始 Tracker、法兰或关节反馈非法')
        p, r = pose_error(ik.forward(q_actual), flange_pose)
        if p > STARTUP_FK_POSITION_M or r > np.deg2rad(STARTUP_FK_ROTATION_DEG):
            raise ValueError(f'URDF FK 与反馈不一致：{p*1000:.2f}mm/{np.rad2deg(r):.2f}deg')
        self.ik = ik
        self.gate = reference.FlangeTargetGate(tracker_pose, flange_pose)
        self.previous_tracker = tracker_pose.copy()
        self.q_cmd_prev = np.asarray(q_actual).copy()
        self.previous_ik = None
        self.robot_fault_latched = False
        self.q_anchor = np.asarray(q_actual).copy()
        self.sent_count = 0
        self.redundancy_watch = RedundancyWatch()

    def step(self, tracker_pose, feedback, send, reanchor=False):
        row = dict(timestamp=time.time(), tracker_valid=valid_transform(tracker_pose),
                   tracker_age=None, tracker_age_source='unavailable_in_tracker_API',
                   ik_success=False, tracker_jump=False, target_jump=False, ik_jump=False,
                   joint_limit=False, sent=False, code='SKIP', seed_source='none',
                   robot_state=feedback.get('status'), enabled=feedback.get('enabled'),
                   error_code=(feedback.get('status') or {}).get('err_status'),
                   feedback_errors=feedback.get('errors'), sent_count=self.sent_count)
        status = feedback.get('status') or {}
        row['emergency_stop'] = status.get('arm_status') == 1 if 'arm_status' in status else None
        row['motion_state'] = status.get('motion_status')
        q_actual = feedback.get('q') if feedback.get('joint_fresh') else None
        actual_limit = (q_actual is not None and np.shape(q_actual) == (7,)
                        and np.isfinite(q_actual).all() and not self.ik.in_limits(q_actual))
        if not self.ik.in_limits(q_actual):
            q_actual = None
        joint_columns(row, 'q_actual', feedback.get('q'))
        for name in ('joint_age_s', 'state_age_s', 'flange_age_s', 'joint_skew_s'):
            row[name] = feedback.get(name)
        pose_columns(row, 'tracker', tracker_pose)
        pose_columns(row, 'actual_flange', feedback.get('flange'))
        if not state_ready(feedback):
            self.robot_fault_latched = True
            self.gate.paused = True
            row['code'] = 'ROBOT_ERROR'
            return row
        if self.robot_fault_latched:
            row['code'] = 'ROBOT_ERROR_LATCHED'  # R 不清除设备故障/发送异常。
            return row
        if actual_limit:
            self.gate.paused = True
            row.update(code='ACTUAL_JOINT_LIMIT', joint_limit=True)
            return row
        if reanchor:
            flange = feedback.get('flange')
            if (q_actual is not None and feedback.get('flange_fresh') and valid_transform(flange)
                    and row['tracker_valid']):
                p, r = pose_error(self.ik.forward(q_actual), flange)
                if (p <= STARTUP_FK_POSITION_M and r <= np.deg2rad(STARTUP_FK_ROTATION_DEG)
                        and self.gate.reanchor(tracker_pose, flange, status.get('motion_status'))):
                    self.q_cmd_prev = q_actual.copy()
                    self.q_anchor = q_actual.copy()
                    self.previous_ik = None
                    self.redundancy_watch = RedundancyWatch()
                    self.previous_tracker = tracker_pose.copy()
                    row['code'] = 'REANCHORED'
                    return row  # R 本周期不发送。
            row['code'] = 'REANCHOR_REJECTED'
            return row
        if not row['tracker_valid']:
            self.gate.paused = True
            row['code'] = 'TRACKER_INVALID'
            return row
        dp, dr = pose_error(tracker_pose, self.previous_tracker)
        row.update(delta_tracker_position_mm=dp*1000, delta_tracker_rotation_deg=np.rad2deg(dr))
        self.previous_tracker = tracker_pose.copy()
        if dp > MAX_TRACKER_TRANSLATION_JUMP_M or dr > np.deg2rad(MAX_TRACKER_ROTATION_JUMP_DEG):
            self.gate.paused = True
            row.update(code='TRACKER_JUMP', tracker_jump=True)
            return row
        # 完全复用原版世界系平移/旋转增量映射，robot anchor 明确为法兰。
        target = reference.tracker_target(self.gate.tracker_anchor, tracker_pose, self.gate.flange_anchor)
        pose_columns(row, 'target', target)
        if not valid_transform(target):
            self.gate.paused = True
            row['code'] = 'TARGET_INVALID'
            return row
        dp, dr = pose_error(target, self.gate.previous)
        row.update(delta_target_position_mm=dp*1000, delta_target_rotation_deg=np.rad2deg(dr))
        if not reference.step_is_valid(target, self.gate.previous):
            self.gate.paused = True
            row.update(code='TARGET_JUMP', target_jump=True)
            return row
        seed = q_actual if q_actual is not None else self.previous_ik
        row['seed_source'] = 'actual' if q_actual is not None else 'previous_ik_diagnostic_only'
        if q_actual is None:
            self.gate.paused = True  # 恢复反馈后须 R，不会自动追赶积累目标。
        if seed is None:
            row['code'] = 'FEEDBACK_MISSING_NO_SEED'
            return row
        solution = self.ik.solve(target, q_seed=seed)
        row['ik_detail'] = self.ik.last_diagnostics.copy()
        row['solve_ms'] = self.ik.last_diagnostics.get('solve_ms')
        for name in ('fk_position_error_mm', 'fk_rotation_error_deg'):
            row[name] = self.ik.last_diagnostics.get(name)
        if solution is None:
            row['code'] = self.ik.last_diagnostics.get('code', 'IK_FAILED')
            row['ik_jump'] = row['code'] == 'IK_JUMP'
            row['joint_limit'] = row['code'] == 'IK_JOINT_LIMIT'
            row['possible_redundancy_flip'] = row['ik_jump']
            return row
        joint_columns(row, 'q_ik', solution)
        if not self.ik.in_limits(solution):
            row.update(code='IK_JOINT_LIMIT', joint_limit=True)
            return row
        p, r = pose_error(self.ik.forward(solution), target)
        row.update(fk_position_error_mm=p*1000, fk_rotation_error_deg=np.rad2deg(r))
        if p >= MAX_FK_POSITION_ERROR_M or r >= np.deg2rad(MAX_FK_ROTATION_ERROR_DEG):
            row['code'] = 'IK_FK_ERROR'
            return row
        dq = solution-seed
        joint_columns(row, 'dq_ik', dq)
        row['dq_ik_reference'] = row['seed_source']
        row['dq_ik_max_deg'] = float(np.max(np.abs(np.rad2deg(dq))))
        if (not self.ik.continuous(solution, seed)
                or (self.previous_ik is not None and not self.ik.continuous(solution, self.previous_ik))):
            row.update(code='IK_JUMP', ik_jump=True, possible_redundancy_flip=True)
            return row
        row['ik_success'] = True
        if q_actual is not None and self.redundancy_watch.update(target, solution):
            self.gate.paused = True
            row.update(code='IK_REDUNDANCY_FLIP', possible_redundancy_flip=True)
            return row
        self.previous_ik = solution.copy()
        joint_columns(row, 'dq_anchor', solution-self.q_anchor)  # 累计冗余漂移诊断，不把正常大范围运动当故障。
        if q_actual is None:
            row['code'] = 'FEEDBACK_MISSING_DIAGNOSTIC_ONLY'
            return row
        if self.gate.paused:
            row['code'] = 'PAUSED_WAIT_R'
            return row
        try:
            q_cmd = clip_command(self.ik, solution, self.q_cmd_prev, q_actual)
        except ValueError as exc:
            row.update(code='COMMAND_JUMP', error=str(exc))
            return row
        joint_columns(row, 'q_proposed', q_cmd)
        dq_cmd = q_cmd-self.q_cmd_prev
        row['command_clipped'] = bool(np.any(np.abs(q_cmd-solution) > 1e-12))
        p, r = pose_error(self.ik.forward(q_cmd), target)
        row.update(command_fk_position_error_mm=p*1000, command_fk_rotation_error_deg=np.rad2deg(r))
        # q_proposed 是待发送值，q_cmd 只在调用成功返回后填写。返回也不等于到位。
        try:
            send(q_cmd.copy())
        except Exception as exc:
            self.robot_fault_latched = True
            self.gate.paused = True
            row.update(code='SDK_SEND_EXCEPTION', error=repr(exc))
            return row
        self.q_cmd_prev = q_cmd.copy()
        self.gate.previous = target.copy()
        self.sent_count += 1
        joint_columns(row, 'q_cmd', q_cmd)
        joint_columns(row, 'dq_cmd', dq_cmd)
        row.update(code='SENT', sent=True, sent_count=self.sent_count,
                   dq_cmd_max_deg=float(np.max(np.abs(np.rad2deg(dq_cmd)))))
        return row


CSV_FIELDS = [
    'timestamp', 'code', 'sent', 'sent_count', 'tracker_valid', 'tracker_age', 'tracker_age_source',
    'tracker_read_ms', 'tracker_local_age_s', 'joint_age_s', 'joint_skew_s', 'state_age_s', 'flange_age_s',
    *[f'{prefix}_{axis}' for prefix in ('tracker', 'target', 'actual_flange')
      for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')],
    'delta_tracker_position_mm', 'delta_tracker_rotation_deg',
    'delta_target_position_mm', 'delta_target_rotation_deg',
    *[f'{prefix}_{i}' for prefix in ('q_actual', 'q_ik', 'q_cmd', 'dq_ik', 'dq_cmd', 'q_proposed', 'dq_anchor')
      for i in range(1, 8)],
    'seed_source', 'dq_ik_reference', 'dq_ik_max_deg', 'dq_cmd_max_deg', 'command_clipped',
    'fk_position_error_mm', 'fk_rotation_error_deg', 'command_fk_position_error_mm',
    'command_fk_rotation_error_deg', 'ik_success', 'tracker_jump', 'target_jump', 'ik_jump',
    'possible_redundancy_flip', 'joint_limit', 'robot_state', 'error_code', 'enabled',
    'emergency_stop', 'motion_state', 'feedback_errors', 'error', 'ik_detail', 'solve_ms',
    'cycle_ms', 'post_send_state', 'pre_send_feedback',
]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class CSVLog:
    """所有 q/dq 列为 deg；xyz 为 m；RPY 为 rad；缺失值留空。"""
    def __init__(self, stream):
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction='raise')
        self.writer.writeheader()
        stream.flush()

    def write(self, row):
        data = {k: json.dumps(v, ensure_ascii=False, default=json_default)
                if isinstance(v, (dict, list, np.ndarray)) else v for k, v in row.items()}
        self.writer.writerow(data)
        self.stream.flush()


def message_age(messages, now):
    stamps = np.asarray([float(m.timestamp) for m in messages])
    ages = now-stamps
    if not np.isfinite(stamps).all() or np.any(stamps <= 0) or np.any(ages < -0.01):
        raise ValueError('反馈时间戳缺失或不是当前 Unix 秒时间域')
    return float(max(0, np.max(ages))), float(np.max(stamps)-np.min(stamps))


def read_feedback(arm):
    """只读已有封装。独立检查七个 CAN 电机帧，防止缓存看似有效但已停更。"""
    result = dict(q=None, flange=None, status=None, enabled=[], state_fresh=False,
                  joint_fresh=False, flange_fresh=False, errors={})
    try:
        # 从同批原始消息复制位置与时间戳；与 get_joint_positions 相同的1..7顺序。
        messages = arm.get_raw_motor_states()
        q = np.array([m.msg.position for m in messages], dtype=float)
        age, skew = message_age(messages, time.time())
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('关节反馈不是有限的 shape=(7,)')
        result.update(q=q, joint_age_s=age, joint_skew_s=skew,
                      joint_fresh=age <= MAX_FEEDBACK_AGE_S and skew <= MAX_FEEDBACK_SKEW_S)
    except Exception as exc:
        result['errors']['joints'] = repr(exc)
    try:
        result['status'] = arm.get_arm_status()
        age, _ = message_age([arm.get_raw_arm_status()], time.time())
        result['enabled'] = [arm._driver.get_joint_enable_status(i) for i in range(1, 8)]
        # enable 状态来自独立 low-speed 帧，不能只靠高频位置帧判断它新鲜。
        enable_age, _ = message_age([arm._driver.get_driver_states(i) for i in range(1, 8)], time.time())
        age = max(age, enable_age)
        result.update(state_age_s=age, state_fresh=age <= MAX_FEEDBACK_AGE_S)
    except Exception as exc:
        result['errors']['state'] = repr(exc)
    try:
        result['flange'] = reference.pose_to_matrix(arm.get_flange_pose())
        # V111 aggregate.timestamp 只来自最后一帧，故必须检查三个组成帧。
        parser = arm._driver._parser
        frames = [getattr(parser, name) for name in ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz')]
        age, skew = message_age(frames, time.time())
        result.update(flange_age_s=age, flange_fresh=valid_transform(result['flange'])
                      and age <= MAX_FEEDBACK_AGE_S and skew <= MAX_FEEDBACK_SKEW_S)
    except Exception as exc:
        result['errors']['flange'] = repr(exc)
    return result


def checked_send(arm, ik, command, previous, tracker_acquired):
    """IK 后再读一次真实状态；不把计算开始时的反馈当作永久有效。"""
    fb = read_feedback(arm)
    if (time.monotonic()-tracker_acquired > MAX_TARGET_AGE_S
            or not state_ready(fb) or not fb['joint_fresh'] or not ik.in_limits(fb['q'])):
        raise RuntimeError('PRE_SEND_GUARD: Tracker 本地目标过期/反馈缺失/设备状态异常')
    verified = clip_command(ik, command, previous, fb['q'])
    if not np.allclose(verified, command, atol=1e-12, rtol=0):
        raise ValueError('PRE_SEND_GUARD: 命令变化超过单步门限')
    arm.validate_joint_command(command, max_joint_delta=MAX_JOINT_STEP)
    arm.move_js(command)  # 唯一运动接口；单位 rad，七轴，无 speed_percent。
    return fb


def apply_sdk_limits(ik, config):
    """核对实际 SDK 顺序并取限位交集，SDK 值为 [lower, upper]。"""
    if list(config.get('joint_names', [])) != ik.joint_names:
        raise ValueError('SDK/URDF joint_names 顺序不一致，禁止发送')
    limits = config.get('joint_limits', {})
    try:
        sdk_limits = np.asarray([limits[name] for name in ik.joint_names], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('SDK joint_limits 缺失或不是七组上下限') from exc
    if (sdk_limits.shape != (7, 2) or not np.isfinite(sdk_limits).all()
            or np.any(sdk_limits[:, 0] >= sdk_limits[:, 1])):
        raise ValueError('SDK joint_limits 非法')
    intersection = np.column_stack((np.maximum(ik.limits[:, 0], sdk_limits[:, 0]),
                                    np.minimum(ik.limits[:, 1], sdk_limits[:, 1])))
    if np.any(intersection[:, 0] >= intersection[:, 1]):
        raise ValueError('SDK/URDF 无合法限位交集')
    ik.limits = intersection


def run(urdf_path):
    ik = NeroIK(urdf_path)
    R = reference.R_base_tracker
    if (not reference.BASE_MAPPING_CALIBRATED or not np.isfinite(R).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1)):
        raise ValueError('请在 pose_debug 中核验并设置有效的 R_base_tracker 标定矩阵')
    folder = Path(__file__).resolve().parent / 'teleop_debug_logs'
    folder.mkdir(exist_ok=True)
    stem = 'nero_tracker_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(folder/(stem+'.log'), encoding='utf-8')],
                        force=True)
    LOG.info('CSV=%s；q/dq=deg，xyz=m，RPY=rad', folder/(stem+'.csv'))
    LOG.warning('JS没有轨迹规划；退出/丢帧只停止新命令，不保证取消在途运动、保持位置或防坠落。'
                '先清空工作区、做好机械支撑，保持硬件急停可用。R不是复位。')
    if input('确认以上边界并允许启动使能和跟随，输入 MOVE：').strip() != 'MOVE':
        return
    arm = tracker_module = None
    try:
        # 仅真机入口加载 SDK/SteamVR。不得把两个嵌套 pyAgxArm 目录插入 sys.path。
        sys.path.insert(0, str(Path(__file__).resolve().parent/'dp_real_0309'))
        from track import ViveTrackerModule
        import pyAgxArm
        LOG.info('python=%s SDK=%s URDF=%s CAN=%s/%s hz=%s R=%s scale=%s',
                 sys.executable, pyAgxArm.__file__, ik.urdf_path, NERO_CAN_INTERFACE,
                 NERO_CAN_CHANNEL, CONTROL_HZ, R.tolist(), reference.POSITION_SCALE)
        tracker_module = ViveTrackerModule()
        tracker_module.print_discovered_objects()
        devices = tracker_module.return_selected_devices('tracker')
        if len(devices) != 1:
            raise RuntimeError('请只连接一个 Tracker')
        tracker = next(iter(devices.values()))
        arm = NeroArm(can_interface=NERO_CAN_INTERFACE, can_channel=NERO_CAN_CHANNEL,
                      max_joint_delta=MAX_JOINT_STEP)
        config = arm._driver.get_config()
        # 取 URDF 与 SDK 限位交集；仍保留封装发送时的官方限位验证。
        apply_sdk_limits(ik, config)
        arm.connect()
        time.sleep(.5)
        fb = read_feedback(arm)
        LOG.info('before_enable=%s', json.dumps(fb, default=json_default, ensure_ascii=False))
        if not fb['state_fresh'] or (fb['status'] or {}).get('arm_status') != 0:
            raise RuntimeError('启动已有故障或状态过期；请先人工检查，不自动复位')
        arm.enable()
        arm._driver.set_normal_mode()
        deadline = time.monotonic()+INITIAL_ALIGNMENT_TIMEOUT_S
        session = None
        with reference.ReanchorKeyboard() as keyboard, open(folder/(stem+'.csv'), 'x', newline='', encoding='utf-8') as stream:
            csv_log = CSVLog(stream)
            LOG.info('保持 Tracker 与机械臂静止，等待初始对齐；超时最多5秒。')
            while time.monotonic() < deadline:
                T = reference.read_tracker_pose(tracker)
                fb = read_feedback(arm)
                if T is not None and state_ready(fb) and fb['joint_fresh'] and fb['flange_fresh']:
                    session = TeleopSession(ik, T, fb['flange'], fb['q'])
                    break
                if (fb['status'] or {}).get('arm_status', 0) != 0:
                    raise RuntimeError(f'启动控制器故障：{fb}')
                time.sleep(1/CONTROL_HZ)
            if session is None:
                raise TimeoutError(f'初始对齐超时，未发送运动：{fb}')
            LOG.info('初始对齐完成；R重新对齐，Ctrl+C退出。设备故障锁定后须退出人工排查。')
            report_at = 0.0
            while True:
                started = time.monotonic()
                row = None
                try:
                    T = reference.read_tracker_pose(tracker)
                    acquired = time.monotonic()
                    read_ms = (acquired-started)*1000
                    if acquired-started > MAX_TRACKER_READ_S:
                        T = None
                    fb = read_feedback(arm)
                    pre_send = {}
                    def send(q):
                        pre_send.update(checked_send(arm, ik, q, session.q_cmd_prev, acquired))
                    row = session.step(T, fb, send, reanchor=keyboard.read())
                    row.update(tracker_read_ms=read_ms, tracker_local_age_s=time.monotonic()-acquired)
                    if pre_send:
                        row['pre_send_feedback'] = pre_send
                    if row['sent']:
                        after = read_feedback(arm)
                        row['post_send_state'] = after
                        if not state_ready(after):
                            session.robot_fault_latched = True
                            session.gate.paused = True
                            row['code'] = 'ROBOT_ERROR_AFTER_SEND'
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session.robot_fault_latched = True
                    session.gate.paused = True
                    row = dict(timestamp=time.time(), code='SDK_OR_LOOP_EXCEPTION', error=repr(exc), sent=False)
                finally:
                    if row is not None:
                        row['cycle_ms'] = (time.monotonic()-started)*1000
                        csv_log.write(row)
                        if row['code'] not in ('SENT', 'PAUSED_WAIT_R'):
                            LOG.warning('[%s] %s', row['code'], json.dumps(row, ensure_ascii=False, default=json_default))
                        elif time.monotonic() >= report_at:
                            LOG.info('%s', json.dumps(row, ensure_ascii=False, default=json_default))
                        if time.monotonic() >= report_at:
                            report_at = time.monotonic()+LOG_INTERVAL
                    time.sleep(max(0, 1/CONTROL_HZ-(time.monotonic()-started)))
    except KeyboardInterrupt:
        LOG.warning('操作者 Ctrl+C：停止下发新目标。')
    finally:
        if arm is not None:
            LOG.warning('退出仅断开通信：未发送 disable/reset/急停/回零；不能保证实机已停稳或保持使能。')
            arm.disconnect()
        del tracker_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ik-test', action='store_true', help='只运行离线FK/IK，不初始化CAN或SteamVR')
    parser.add_argument('--urdf', type=Path, default=URDF_PATH)
    args = parser.parse_args()
    if args.ik_test:
        self_test(args.urdf)
    else:
        run(args.urdf)


if __name__ == '__main__':
    main()
