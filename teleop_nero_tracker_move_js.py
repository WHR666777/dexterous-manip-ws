"""Vive Tracker 相对遥操 NERO，使用 move_js 下发七轴关节角。

运行：python3 teleop_nero_tracker_move_js.py
标定仍在 teleop_nero_tracker.py 中设置；CAN 配置沿用 config.py。
依赖与原脚本相同（含 SciPy）；IK 使用下方 URDF 的 base_link→link7 链。
move_js 不做轨迹规划；NERO_SPEED_PERCENT 不用于本脚本。
"""

import sys
import argparse
import pprint
import warnings
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
from robot_control import NeroArm
import teleop_nero_tracker as reference
from teleop_nero_tracker import (
    CONTROL_HZ, pose_to_matrix, read_tracker_pose, tracker_target, step_is_valid,
)

URDF_PATH = '/home/lab404/ZJX/Nero_L20/pyAgxArm/assets/nero_description.urdf'
MAX_JOINT_STEP = np.deg2rad(5)  # 同时约束相对反馈和上一已发送目标的变化。
IK_POSITION_TOL = 0.001  # m
IK_ROTATION_TOL = np.deg2rad(0.5)


def pose_diagnostic(matrix):
    """欧拉角仅显示；距离判断使用相对旋转。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UserWarning)
        angles = Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz')
    return {'矩阵': matrix.tolist(), '绝对位姿 m/rad': np.r_[matrix[:3, 3], angles].tolist(),
            'xyz mm': (matrix[:3, 3] * 1000).tolist(), 'rpy deg': np.rad2deg(angles).tolist(),
            '欧拉角奇异（仅显示，不用于误差判断）': bool(caught)}


def pose_delta(target, actual):
    return {'平移 mm': ((target[:3, 3] - actual[:3, 3]) * 1000).tolist(),
            '位置距离 mm': float(np.linalg.norm(target[:3, 3] - actual[:3, 3]) * 1000),
            '旋转 deg': float(np.rad2deg(Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).magnitude()))}


def arm_diagnostic(arm):
    """只读状态，诊断读取失败不能掩盖原始异常。"""
    result = {}
    for name, reader in [('状态', arm.get_arm_status), ('健康', arm.is_ok),
                         ('使能', arm.is_enabled)]:
        try:
            result[name] = reader()
        except Exception as exc:
            result[name] = f'读取失败：{exc}'
    return result


def feedback_timing(arm):
    """额外读取缓存时间戳用于排查旧反馈，不声称与 seed 是原子快照。"""
    try:
        stamps = [getattr(msg, 'timestamp', None) for msg in arm.get_raw_motor_states()]
        return {'电机消息时间戳': stamps,
                '法兰消息时间戳': getattr(arm.get_raw_flange_pose(), 'timestamp', None),
                '说明': '诊断时刻再次读取缓存；七轴与法兰不是同步采样'}
    except Exception as exc:
        return {'说明': f'消息时间戳不可用：{exc}'}


class TrackerDiagnostics:
    """计数为本进程调用频率，不冒充传感器新帧或 CAN 实际发送频率。"""
    def __init__(self):
        self.last_print = 0.0
        self.started = time.monotonic()
        self.counts = dict(tracker=0, IK=0, move_js=0, feedback=0)
        self.was_sent = False
        self.first_drop = False

    def finish(self, report):
        now = time.monotonic()
        sent = report.get('move_js_called', False)
        if self.was_sent and not sent and not self.first_drop:
            print('首次停止发送 True→False，原因：', report.get('reason'), flush=True)
            self.first_drop = True
        self.was_sent = sent
        if not sent or now - self.last_print >= .2:
            print('========== TRACKER DEBUG（法兰数据链）==========', flush=True)
            pprint.pprint(report, sort_dicts=False, width=150, compact=True)
            self.last_print = now
        elapsed = now - self.started
        if elapsed >= 1:
            print('[HZ] 读取/求解/调用频率（不保证新帧或控制器收到）：',
                  {key: round(value / elapsed, 2) for key, value in self.counts.items()}, flush=True)
            self.started = now
            self.counts = dict.fromkeys(self.counts, 0)


def pose_matches(actual, target, pos_tol=IK_POSITION_TOL, rot_tol=IK_ROTATION_TOL):
    return (np.isfinite(actual).all() and np.isfinite(target).all()
            and np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= pos_tol
            and Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).magnitude() <= rot_tol)


class NeroIK:
    """从随仓库提供的七轴 URDF 求局部 IK，不加载网格或启动仿真。"""

    def __init__(self, urdf_path=URDF_PATH):
        root = ET.parse(urdf_path).getroot()
        self.origins, self.axes, limits = [], [], []
        parent = 'base_link'
        for i in range(1, 8):
            joint = root.find(f"joint[@name='joint{i}']")
            if (joint is None or joint.get('type') != 'revolute'
                    or joint.find('parent').get('link') != parent
                    or joint.find('child').get('link') != f'link{i}'):
                raise ValueError('URDF 必须提供 base_link→link7 的 joint1..7 旋转关节链。')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            self.origins.append(pose_to_matrix(np.r_[xyz, rpy]))
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find('limit')
            limits.append([float(limit.get('lower')), float(limit.get('upper'))])
            parent = f'link{i}'
        self.limits = np.asarray(limits)

    def forward(self, q):
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, q):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ rotation
        return transform

    def solve(self, target, current_q, debug=False):
        debug = debug or getattr(self, 'debug', False)
        report = {'坐标约定': '目标 Base→Flange；FK 为 base_link→link7；数值不能证明 frame 语义',
                  '候选策略': '一个初始值、一个候选、无 restart', 'reason': ''}
        self.last_report = report

        def finish(reason, candidate=None):
            report['reason'] = reason
            report['成功'] = candidate is not None
            if reason or debug:
                print('========== IK DEBUG（逆解诊断）==========')
                pprint.pprint(report, sort_dicts=False, width=150, compact=True)
                print('FINAL RESULT = NONE' if candidate is None else '最终结果：成功', flush=True)
            return candidate

        try:
            q = np.asarray(current_q, dtype=float)
        except (TypeError, ValueError):
            return finish('seed 异常：无法转换关节角')
        report['当前关节 rad'] = q.tolist()
        report['当前关节 deg'] = np.rad2deg(q).tolist()
        if q.shape != (7,) or not np.isfinite(q).all():
            return finish('seed 异常：必须为七个有限关节角')
        try:
            target = np.asarray(target, dtype=float)
            valid = (target.shape == (4, 4) and np.isfinite(target).all()
                     and np.allclose(target[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                     and np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-3)
                     and np.isclose(np.linalg.det(target[:3, :3]), 1, atol=1e-3))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            report['输入目标'] = repr(target)
            return finish('目标矩阵非法：需要有限的 4×4 刚体变换')
        report['目标法兰'] = pose_diagnostic(target)
        current_fk = self.forward(q)
        report['当前 FK'] = pose_diagnostic(current_fk)
        report['目标相对当前 FK'] = pose_delta(target, current_fk)
        report['seed 限位检查'] = ((q >= self.limits[:, 0]) & (q <= self.limits[:, 1])).tolist()
        lower = np.maximum(self.limits[:, 0], q - MAX_JOINT_STEP)
        upper = np.minimum(self.limits[:, 1], q + MAX_JOINT_STEP)
        report['局部搜索范围 deg'] = np.rad2deg(np.c_[lower, upper]).tolist()
        if np.any(lower >= upper):
            return finish('局部搜索区间为空：seed 附近 5° 与关节限位无有效交集')
        seed = np.clip(q, lower, upper)
        report['优化初始值 rad'] = seed.tolist()

        def residual(candidate):
            actual = self.forward(candidate)
            translation = (actual[:3, 3] - target[:3, 3]) / IK_POSITION_TOL
            rotation = Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).as_rotvec() / IK_ROTATION_TOL
            # 七轴冗余以当前实测姿态为偏好，避免无意义的零空间跳变。
            return np.r_[translation, rotation, 0.01 * (candidate - q)]

        try:
            result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=40)
        except (ValueError, FloatingPointError) as exc:
            return finish(f'优化器异常：{exc}')
        report['优化器'] = {'成功': bool(result.success), '消息': str(result.message),
                         '评估次数': result.nfev, '原始解 rad': np.asarray(result.x).tolist()}
        candidate = np.asarray(result.x)
        reasons = [] if result.success else ['优化器失败']
        if candidate.shape != (7,) or not np.isfinite(candidate).all():
            return finish('；'.join(reasons + ['候选形状错误或含非有限数']))
        report['候选关节 deg'] = np.rad2deg(candidate).tolist()
        report['候选减当前 deg'] = np.rad2deg(candidate - q).tolist()
        report['候选减seed deg'] = np.rad2deg(candidate - seed).tolist()
        report['逐轴检查'] = []
        for i, value in enumerate(candidate):
            limit_ok = self.limits[i, 0] <= value <= self.limits[i, 1]
            current_ok = abs(value - q[i]) <= MAX_JOINT_STEP
            seed_ok = abs(value - seed[i]) <= MAX_JOINT_STEP
            report['逐轴检查'].append({'关节': i + 1, '限位 deg': np.rad2deg(self.limits[i]).tolist(),
                                      '限位通过': bool(limit_ok), '当前5°通过': bool(current_ok),
                                      'seed5°通过': bool(seed_ok)})
            if not limit_ok:
                reasons.append(f'J{i+1} 关节限位超限')
            if not current_ok:
                reasons.append(f'J{i+1} 相对当前关节超过5°')
            if not seed_ok:
                reasons.append(f'J{i+1} 相对seed超过5°')
        candidate_fk = self.forward(candidate)
        report['候选 FK'] = pose_diagnostic(candidate_fk)
        errors = pose_delta(target, candidate_fk)
        report['候选 FK 对目标误差'] = errors
        if errors['位置距离 mm'] > IK_POSITION_TOL * 1000:
            reasons.append('FK 位置误差超限')
        if errors['旋转 deg'] > np.rad2deg(IK_ROTATION_TOL):
            reasons.append('FK 姿态误差超限')
        return finish('；'.join(reasons), None if reasons else candidate)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Vive Tracker→绝对法兰→局部IK→move_js')
    parser.add_argument('--debug-ik', action='store_true', help='成功求解也打印完整逆解诊断')
    args = parser.parse_args(argv)
    R_base_tracker = reference.R_base_tracker
    if not reference.BASE_MAPPING_CALIBRATED:
        raise RuntimeError("请先标定 R_base_tracker，再设置 BASE_MAPPING_CALIBRATED=True。")
    if (not np.isfinite(R_base_tracker).all()
            or not np.allclose(R_base_tracker.T @ R_base_tracker, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R_base_tracker), 1.0, atol=1e-6)):
        raise ValueError("R_base_tracker 必须是正交且行列式为 +1 的旋转矩阵。")

    ik = NeroIK()  # 在连接硬件前检查模型能否加载。

    # 原模块使用同目录绝对导入；只在运行时添加路径、初始化 SteamVR。
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'dp_real_0309'))
    from track import ViveTrackerModule

    tracker_module = None
    arm = None
    motion_started = False
    try:
        tracker_module = ViveTrackerModule()
        tracker_module.print_discovered_objects()
        devices = tracker_module.return_selected_devices('tracker')
        if len(devices) != 1:
            raise RuntimeError("请只连接原项目使用的那一个 Tracker。")
        tracker = next(iter(devices.values()))

        arm = NeroArm(can_interface=NERO_CAN_INTERFACE, can_channel=NERO_CAN_CHANNEL,
                      max_joint_delta=MAX_JOINT_STEP)
        arm.connect()
        arm.enable(timeout=5.0)  # 封装参数是超时秒数，不是 SDK 关节编号。
        dt = 1.0 / CONTROL_HZ
        print("保持 Tracker 和机械臂静止，等待有效初始位姿……")
        deadline = time.monotonic() + 5.0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('初始对齐超过5秒：检查 Tracker 有效性与法兰反馈')
            T_tracker_0 = read_tracker_pose(tracker)
            try:
                # 初始锚点直接使用 Base 下实测法兰，不读取或转换 TCP。
                T_flange_0 = pose_to_matrix(arm.get_flange_pose())
                initial_joints = arm.get_joint_positions()
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(dt)
                continue
            if T_tracker_0 is not None:
                break
            time.sleep(dt)

        # 防止 URDF 末端坐标定义与实际法兰反馈不一致时发送错误关节角。
        if not pose_matches(ik.forward(initial_joints), T_flange_0,
                            pos_tol=0.01, rot_tol=np.deg2rad(5)):
            raise RuntimeError("URDF FK 与法兰反馈不一致，请核对模型、零位及法兰坐标系。")
        print('初始 FK 对实际法兰误差：', pose_delta(T_flange_0, ik.forward(initial_joints)))
        previous_joints = initial_joints.copy()
        previous_flange = T_flange_0.copy()
        previous_generated = T_flange_0.copy()
        previous_candidate = initial_joints.copy()
        previous_actual = initial_joints.copy()
        diag = TrackerDiagnostics()
        print("初始对齐完成，开始 move_js 相对遥操；Ctrl+C 停止。")
        while True:
            cycle_start = time.monotonic()
            report = {'时间': time.time(), 'Tracker有效': False, 'move_js_called': False,
                      'reason': '', 'IK成功': None, '法兰步长通过': None,
                      '关节校验通过': None, '目标法兰': None, '候选关节 deg': None}
            try:
                diag.counts['tracker'] += 1
                T_tracker = read_tracker_pose(tracker)
                report['机械臂'] = arm_diagnostic(arm)
                actual_flange = pose_to_matrix(arm.get_flange_pose())
                current_joints = arm.get_joint_positions()
                diag.counts['feedback'] += 1
                report['实际法兰'] = pose_diagnostic(actual_flange)
                report['当前关节 deg'] = np.rad2deg(current_joints).tolist()
                report['反馈时间'] = feedback_timing(arm)
                report['实际关节相对上周期变化 deg'] = np.rad2deg(current_joints - previous_actual).tolist()
                previous_actual = current_joints.copy()
                report['实际法兰对当前FK误差'] = pose_delta(actual_flange, ik.forward(current_joints))
                if T_tracker is None:
                    report['reason'] = 'Tracker 丢帧或位姿非法：未生成目标、不发送旧目标'
                    continue  # 丢失时不发送旧位姿；恢复后仍检查与最后目标的距离。
                report['Tracker有效'] = True
                report['Tracker原始位姿'] = pose_diagnostic(T_tracker)
                report['Tracker四元数 xyzw（由同帧矩阵转换）'] = Rotation.from_matrix(T_tracker[:3, :3]).as_quat().tolist()
                report['Tracker初始锚点'] = T_tracker_0.tolist()
                calibrated = np.eye(4)
                calibrated[:3, :3] = R_base_tracker @ T_tracker[:3, :3]
                calibrated[:3, 3] = R_base_tracker @ T_tracker[:3, 3]
                report['仅旋转标定后的Tracker（未标定绝对原点）'] = calibrated.tolist()
                report['Tracker局部相对变换 inv(T0)@T'] = (np.linalg.inv(T_tracker_0) @ T_tracker).tolist()
                report['Tracker世界系相对增量（映射实际使用）'] = pose_delta(T_tracker, T_tracker_0)
                report['初始机器人法兰'] = pose_diagnostic(T_flange_0)
                target_flange = tracker_target(T_tracker_0, T_tracker, T_flange_0)
                report['目标法兰'] = pose_diagnostic(target_flange)
                report['目标相对上一生成目标'] = pose_delta(target_flange, previous_generated)
                previous_generated = target_flange.copy()
                report['目标相对上一已发送目标'] = pose_delta(target_flange, previous_flange)
                # 检查实际发送的法兰目标；跳过后不推进 previous。
                # 大幅漂移恢复后须回到上一目标附近，或退出后重新对齐。
                report['法兰步长通过'] = bool(step_is_valid(target_flange, previous_flange))
                if not report['法兰步长通过']:
                    report['reason'] = '法兰步长超过1cm/5°；须回到上次发送目标附近，或退出重新对齐'
                    continue
                diag.counts['IK'] += 1
                joints = ik.solve(target_flange, current_joints, debug=args.debug_ik)
                report['IK成功'] = joints is not None
                if joints is None:
                    report['reason'] = ik.last_report['reason']
                    continue
                report['候选关节 deg'] = np.rad2deg(joints).tolist()
                report['候选减实际 deg'] = np.rad2deg(joints - current_joints).tolist()
                report['候选减上一候选 deg'] = np.rad2deg(joints - previous_candidate).tolist()
                report['候选减上一已发送 deg'] = np.rad2deg(joints - previous_joints).tolist()
                previous_candidate = joints.copy()
                if np.any(np.abs(joints - previous_joints) > MAX_JOINT_STEP):
                    report['reason'] = '候选相对上一已发送关节目标超过5°'
                    continue  # IK 不收敛或关节跳变时不发送，也不推进目标基准。
                try:
                    arm.validate_joint_command(joints)
                    report['关节校验通过'] = True
                except ValueError as exc:
                    report['关节校验通过'] = False
                    report['reason'] = f'发送前关节校验失败：{exc}'
                    raise
                # 唯一运动发送点：官方 JS/Follower 模式，七轴角度单位 rad。
                motion_started = True
                arm.move_js(joints)
                report['move_js_called'] = True
                report['发送后机械臂缓存状态'] = arm_diagnostic(arm)
                report['发送语义'] = '封装正常返回，不代表控制器确认收到或到位'
                diag.counts['move_js'] += 1
                previous_joints = joints.copy()
                previous_flange = target_flange
            except BaseException as exc:
                report['reason'] = report['reason'] or f'周期异常/中断：{type(exc).__name__}: {exc}'
                raise
            finally:
                diag.finish(report)
                time.sleep(max(0.0, dt - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        print("\n停止遥操。")
    finally:
        try:
            if arm is not None:
                try:
                    # 无普通 stop/stop_servo；仅断开不能取消在途目标。
                    # 已发送运动时调用公开电子急停，不自动 reset 或失能。
                    if motion_started and arm.is_connected():
                        arm.emergency_stop()
                finally:
                    arm.disconnect()
        finally:
            # 原 ViveTrackerModule.__del__ 负责 openvr.shutdown()。
            del tracker_module


if __name__ == '__main__':
    main()
