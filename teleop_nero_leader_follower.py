"""同型号 NERO 主从遥操：can1主臂零力拖动，can0先对齐再绝对关节跟随。

运行：python teleop_nero_leader_follower.py
确认START后两臂初始化CAN，使能主臂并切换leader模式；主臂不发送位置目标。
任何运动后的退出/故障请求从臂电子急停，不自动复位或失能。急停不保证防坠落。
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from robot_control import NeroArm
from teleop_nero_tracker_move_js_debug import (
    read_feedback, json_default, message_age, MAX_FEEDBACK_AGE_S, MAX_FEEDBACK_SKEW_S,
)

LEADER_CHANNEL = 'can1'  # 主臂CAN通道：初始化后进入leader零力拖动模式。
FOLLOWER_CHANNEL = 'can0'  # 从臂CAN通道：先对齐，再接收绝对关节目标。
CAN_INTERFACE = 'socketcan'  # Linux CAN后端；此参数不会自动配置网卡波特率。
CONTROL_HZ = 20  # 目标循环频率Hz；10Hz约100ms一帧，实际频率还受读取/发送耗时影响。
ALIGN_SPEED_PERCENT = 5  # 仅控制启动move_joints对齐速度；不控制后续move_js跟随速度。
ALIGN_TIMEOUT_S = 30  # 对齐到位等待上限（秒）；超时中止，不进入跟随。
ALIGN_TOLERANCE_RAD = np.deg2rad(.5)  # 对齐误差阈值：逐轴比较固定目标及主从差值，单位rad。
ALIGN_SETTLED_SAMPLES = 3  # 连续满足对齐误差且从臂报告停止的次数；10Hz时采样间隔约0.1秒。
MASTER_HOLD_TOLERANCE_RAD = np.deg2rad(1)  # 对齐期间主臂偏离固定启动姿态的最大角度；不是跟随速度。
MAX_MASTER_STEP_RAD = np.deg2rad(10)  # 跟随时主臂相邻周期最大角度变化；超过即拒绝，不是命令限速。
MAX_COMMAND_STEP_RAD = np.deg2rad(2)  # 从臂每轴目标每周期最多推进1°；配合10Hz约限制目标斜率为10°/s，不保证实机速度。
MAX_FOLLOW_ERROR_RAD = np.deg2rad(15)  # 主从差值及命令-反馈差值上限；增大只容忍更多滞后，不会加速，禁止当速度旋钮。
LOG_INTERVAL_S = .5  # 正常跟随日志输出间隔（秒）；不改变运动发送频率。
STARTUP_TIMEOUT_S = 5.0  # CAN初始化及leader专用反馈等待上限（秒），不用于对齐运动计时。
STARTUP_POLL_S = .01  # 初始化重试/反馈轮询间隔（秒）；不控制运行阶段跟随频率。
LOG = logging.getLogger('nero.leader_follower')


def valid_joints(q, limits):
    q = np.asarray(q, dtype=float)
    return (q.shape == (7,) and np.isfinite(q).all()
            and np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))


def require_feedback(fb, limits, *, follower):
    """主臂专用流允许普通状态未知；从臂仍强制完整状态，已知故障均拒绝。"""
    status = fb.get('status') or {}
    leader_stream = not follower and fb.get('source') == 'leader_joint_angles'
    known_fault = (status.get('arm_status') not in (None, 0)
                   or any((status.get('err_status') or {}).values()))
    if (not fb.get('joint_fresh') or known_fault
            or (not leader_stream and (not fb.get('state_fresh') or status.get('arm_status') != 0))
            or not valid_joints(fb.get('q'), limits)):
        raise RuntimeError(f'反馈过期、限位或设备故障：{fb}')
    if follower and (status.get('ctrl_mode') != 1 or len(fb.get('enabled', [])) != 7
                     or not all(fb['enabled'])):
        raise RuntimeError(f'从臂不是CAN已使能状态：{fb}')
    return fb['q'].copy()


def read_leader_feedback(arm, since=0.0):
    """V111继承的leader专用七帧；不能把SDK部分聚合中的默认零值当真实角度。"""
    fb = dict(source='leader_joint_angles', q=None, joint_fresh=False,
              state_fresh=False, status=None, enabled=[], errors={},
              state_observability='UNKNOWN: leader模式关闭普通CAN推送')
    try:
        driver = arm._driver
        # 先检查全部组成帧，聚合时间戳只代表最后一个组成帧，不能单独用于判新鲜。
        frames = [getattr(driver._parser, f'leader_joint_{i}') for i in range(1, 8)]
        copies = [SimpleNamespace(timestamp=float(frame.timestamp))
                  for frame in frames]
        message = driver.get_leader_joint_angles()
        q = np.asarray(message.msg, dtype=float).copy()
        age, skew = message_age(copies, time.time())
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('leader角度必须是有限的七轴rad数组')
        fb.update(q=q, joint_age_s=age, joint_skew_s=skew,
                  joint_fresh=age <= MAX_FEEDBACK_AGE_S and skew <= MAX_FEEDBACK_SKEW_S
                  and all(frame.timestamp >= since for frame in copies))
    except Exception as exc:
        fb['errors']['leader_joints'] = repr(exc)
    try:
        fb['status'] = arm.get_arm_status()  # 可能是旧缓存，仅记录，绝不伪装为正常。
        age, _ = message_age([arm.get_raw_arm_status()], time.time())
        fb.update(state_age_s=age, state_fresh=age <= MAX_FEEDBACK_AGE_S)
        if fb['state_fresh']:
            fb['state_observability'] = 'fresh_arm_status_only'
    except Exception as exc:
        fb['errors']['ordinary_status'] = repr(exc)
    return fb


def start_can_mode(arm, limits, label, timeout=STARTUP_TIMEOUT_S):
    """使能与normal交替以启动推送；使用SDK布尔返回，不误用NeroArm.enable的None返回。"""
    deadline = time.monotonic()+timeout
    while True:
        fb = read_feedback(arm)
        status = fb.get('status') or {}
        if status.get('arm_status') not in (None, 0) or any((status.get('err_status') or {}).values()):
            raise RuntimeError(f'{label}已有故障，禁止自动恢复：{fb}')
        enabled = arm._driver.enable()
        arm._driver.set_normal_mode()  # 即使已使能也需启动普通CAN推送。
        if enabled:
            fb = read_feedback(arm)
            try:
                require_feedback(fb, limits, follower=True)
                LOG.info('%s CAN mode enabled，七轴及状态反馈就绪', label)
                return
            except RuntimeError:
                pass  # 等待异步新反馈；下一轮仍检查故障，不自动reset。
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{label} CAN模式初始化超时：{fb}')
        time.sleep(STARTUP_POLL_S)


def wait_leader_feedback(arm, limits, since):
    deadline = time.monotonic()+STARTUP_TIMEOUT_S
    while True:
        fb = read_leader_feedback(arm, since)
        status = fb.get('status') or {}
        if status.get('arm_status') not in (None, 0) or any((status.get('err_status') or {}).values()):
            raise RuntimeError(f'主臂已知故障：{fb}')
        if fb['joint_fresh']:
            require_feedback(fb, limits, follower=False)
            return fb
        if time.monotonic() >= deadline:
            raise TimeoutError(f'leader模式七轴消息未就绪：{fb}')
        time.sleep(STARTUP_POLL_S)


class Alignment:
    """目标为确认前捕获的主臂姿态，不能在对齐运动中追逐活动主臂。"""
    def __init__(self, target):
        self.target = np.asarray(target).copy()
        self.settled = 0

    def update(self, leader, follower):
        if np.max(np.abs(leader['q']-self.target)) > MASTER_HOLD_TOLERANCE_RAD:
            raise RuntimeError('对齐过程中主臂移动超过1°，中止；请保持主臂静止重新启动')
        reached = (np.max(np.abs(follower['q']-self.target)) < ALIGN_TOLERANCE_RAD
                   and np.max(np.abs(follower['q']-leader['q'])) < ALIGN_TOLERANCE_RAD
                   and follower['status'].get('motion_status') == 0)
        self.settled = self.settled+1 if reached else 0
        return self.settled >= ALIGN_SETTLED_SAMPLES


def follow_command(master, actual, previous_command, previous_master, limits):
    for q in (master, actual, previous_command, previous_master):
        if not valid_joints(q, limits):
            raise ValueError('非法七轴数值或关节超限')
    # 均是有限转角关节；不通过wrap短角绕过机械限位。
    if np.max(np.abs(master-previous_master)) > MAX_MASTER_STEP_RAD:
        raise ValueError('主臂单周期跳变超过5°')
    if np.max(np.abs(master-actual)) > MAX_FOLLOW_ERROR_RAD:
        raise ValueError('主从误差超过5°，停止跟随；不自动追赶')
    command = previous_command + np.clip(master-previous_command,
                                         -MAX_COMMAND_STEP_RAD, MAX_COMMAND_STEP_RAD)
    if not valid_joints(command, limits) or np.max(np.abs(command-actual)) > MAX_FOLLOW_ERROR_RAD:
        raise ValueError('从臂命令超限或跟随滞后')
    return command


def common_limits(leader, follower):
    names = [f'joint{i}' for i in range(1, 8)]
    bounds = []
    for arm in (leader, follower):
        config = arm._driver.get_config()
        if list(config.get('joint_names', [])) != names:
            raise ValueError('SDK关节名称/顺序不是joint1..7')
        limits = np.asarray([config['joint_limits'][name] for name in names], dtype=float)
        if limits.shape != (7, 2) or not np.isfinite(limits).all():
            raise ValueError('SDK限位非法')
        bounds.append(limits)
    limits = np.column_stack((np.maximum(bounds[0][:, 0], bounds[1][:, 0]),
                              np.minimum(bounds[0][:, 1], bounds[1][:, 1])))
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('两臂限位无合法交集')
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leader', default=LEADER_CHANNEL)
    parser.add_argument('--follower', default=FOLLOWER_CHANNEL)
    args = parser.parse_args()
    if args.leader == args.follower:
        parser.error('主从不能使用同一CAN接口')
    folder = Path(__file__).resolve().parent/'teleop_debug_logs'
    folder.mkdir(exist_ok=True)
    path = folder/('nero_leader_follower_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding='utf-8')], force=True)
    LOG.info('主臂=%s 从臂=%s 日志=%s', args.leader, args.follower, path)
    LOG.warning('两臂必须同型号、同零位标定。确认从臂整条对齐路径无碰撞，并落实机械支撑/硬件急停。'
                '启动会使能两臂并切主臂leader模式；运动后Ctrl+C/故障请求从臂急停，不能保证防坠落。')
    leader = follower = None
    motion_started = False
    phase = 'connect'
    try:
        leader = NeroArm(can_interface=CAN_INTERFACE, can_channel=args.leader)
        follower = NeroArm(can_interface=CAN_INTERFACE, can_channel=args.follower)
        limits = common_limits(leader, follower)
        if not callable(getattr(leader._driver, 'set_leader_mode', None)):
            raise RuntimeError('当前SDK不支持set_leader_mode')
        LOG.info('驱动 主臂=%s 从臂=%s（沿用NeroArm固定V111，不自动切换DEFAULT）',
                 type(leader._driver).__module__, type(follower._driver).__module__)
        leader.connect()
        follower.connect()
        phase = 'start_can_modes'
        start_can_mode(leader, limits, '主臂')
        start_can_mode(follower, limits, '从臂')
        phase = 'leader_mode'
        leader_since = time.time()
        leader._driver.set_leader_mode()
        master_fb = wait_leader_feedback(leader, limits, leader_since)
        LOG.warning('leader mode七轴反馈已就绪；普通状态可能停止更新，后续按专用七帧检查，非设备无故障保证。')
        slave_fb = read_feedback(follower)
        target = require_feedback(master_fb, limits, follower=False)
        current = require_feedback(slave_fb, limits, follower=True)
        follower.validate_joint_command(target)
        LOG.info('主臂目标deg=%s 从臂当前deg=%s 对齐变化deg=%s',
                 np.rad2deg(target), np.rad2deg(current), np.rad2deg(target-current))
        if input(f'保持主臂静止；确认允许从臂以{ALIGN_SPEED_PERCENT}%速度运动到以上主臂姿态，输入 A：').strip() != 'A':
            LOG.info('取消对齐：两臂已初始化使能，主臂保持leader模式；未发送位置目标')
            return
        # 输入可能等待很久：重新验证两臂及固定目标没有漂移，不再重发模式切换。
        master_fb, slave_fb = read_leader_feedback(leader, leader_since), read_feedback(follower)
        master = require_feedback(master_fb, limits, follower=False)
        require_feedback(slave_fb, limits, follower=True)
        if np.max(np.abs(master-target)) >= ALIGN_TOLERANCE_RAD:
            raise RuntimeError('确认期间主臂已移动，请重新启动并保持静止')
        master_fb = read_leader_feedback(leader, leader_since)
        master = require_feedback(master_fb, limits, follower=False)
        require_feedback(read_feedback(follower), limits, follower=True)
        if np.max(np.abs(master-target)) >= ALIGN_TOLERANCE_RAD:
            raise RuntimeError('使能期间主臂移动，禁止启动对齐')
        phase = 'alignment'
        motion_started = True  # 即使发送部分失败，仍需尝试取消在途运动。
        follower.move_joints(target, speed_percent=ALIGN_SPEED_PERCENT)
        alignment = Alignment(target)
        deadline = time.monotonic()+ALIGN_TIMEOUT_S
        report_at = 0.0
        while True:
            master_fb, slave_fb = read_leader_feedback(leader, leader_since), read_feedback(follower)
            master = require_feedback(master_fb, limits, follower=False)
            actual = require_feedback(slave_fb, limits, follower=True)
            if time.monotonic() >= report_at:
                LOG.info('ALIGN max_error_deg=%.3f settled=%d',
                         np.rad2deg(np.max(np.abs(actual-target))), alignment.settled)
                report_at = time.monotonic()+LOG_INTERVAL_S
            if alignment.update(master_fb, slave_fb):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('30秒未确认对齐，不进入跟随')
            time.sleep(1/CONTROL_HZ)
        phase = 'following'
        previous_command, previous_master = actual.copy(), master.copy()
        LOG.info('连续3次确认两臂对齐，开始绝对关节跟随；请缓慢拖动主臂。Ctrl+C停止。')
        while True:
            started = time.monotonic()
            master_fb, slave_fb = read_leader_feedback(leader, leader_since), read_feedback(follower)
            master = require_feedback(master_fb, limits, follower=False)
            actual = require_feedback(slave_fb, limits, follower=True)
            command = follow_command(master, actual, previous_command, previous_master, limits)
            # 发送前再次检查从臂当前状态/实际跟随误差，SDK再次验证官方限位。
            latest = require_feedback(read_feedback(follower), limits, follower=True)
            if np.max(np.abs(command-latest)) > MAX_FOLLOW_ERROR_RAD:
                raise RuntimeError('发送前从臂反馈发生较大变化')
            follower.validate_joint_command(command, max_joint_delta=MAX_FOLLOW_ERROR_RAD)
            follower.move_js(command)
            previous_command, previous_master = command.copy(), master.copy()
            if time.monotonic() >= report_at:
                LOG.info('FOLLOW %s', json.dumps(dict(master_deg=np.rad2deg(master),
                         follower_deg=np.rad2deg(actual), command_deg=np.rad2deg(command),
                         master_state=master_fb['status'], master_state_fresh=master_fb['state_fresh'],
                         master_state_observability=master_fb['state_observability'],
                         master_joint_age_s=master_fb.get('joint_age_s'),
                         master_joint_skew_s=master_fb.get('joint_skew_s'),
                         follower_state=slave_fb['status']),
                         default=json_default, ensure_ascii=False))
                report_at = time.monotonic()+LOG_INTERVAL_S
            time.sleep(max(0, 1/CONTROL_HZ-(time.monotonic()-started)))
    except KeyboardInterrupt:
        LOG.warning('操作者中断 phase=%s', phase)
    except Exception:
        LOG.exception('停止发送，禁止自动恢复 phase=%s', phase)
        raise
    finally:
        try:
            if follower is not None and motion_started:
                LOG.warning('请求从臂电子急停以取消在途运动；不自动reset/disable，不能保证防坠落。')
                try:
                    follower.emergency_stop()
                except Exception:
                    LOG.exception('从臂急停请求失败，请使用现场硬件安全措施')
        finally:
            try:
                if follower is not None:
                    follower.disconnect()
            finally:
                if leader is not None:
                    LOG.warning('主臂仅断开：不自动退出leader模式、不失能；请现场确认两臂状态。')
                    leader.disconnect()


if __name__ == '__main__':
    main()
