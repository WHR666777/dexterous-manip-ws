"""NERO move_pose 遥操作诊断版：在项目根目录运行本文件。
日志自动保存到本文件旁 teleop_debug_logs/，每条 DIAG 都有事件名与单调时间。
保留原位置映射、10 Hz、步长门限和退出急停；不自动复位故障。
没有驱动状态码定义与 IK 模型，因此不会把跟踪误差武断判定为不可达。
"""

import sys
import json
from datetime import datetime
import logging
import time
import os
import select
import termios
import tty
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE, NERO_SPEED_PERCENT
from robot_control import NeroArm


POSITION_SCALE = 0.6
CONTROL_HZ = 10  # 沿用 dp_real_0309/teleoperation.py 的目标更新频率。
MAX_POSITION_STEP = 0.03  # 每次已发送目标之间最多 1 cm，超限锁定暂停。
MAX_ROTATION_STEP = np.deg2rad(10)

# 原 teleoperation.py 的位置映射为 D @ A @ D @ R_tracker_0.T；
# D=diag(1,-1,-1)，A=Rx(135°)，前面的 R_tracker_0.T 来自原点标定。
# D @ A @ D = A。下面仅保留 A 作为标定起点，不冒充实际外参：
# R_base_tracker 必须替换为「SteamVR Standing 世界系 → NERO Base」旋转，
# 包含原 Tracker 参考朝向及 UR5 Base → NERO Base 的实际安装旋转。
# R_base_tracker = np.array([
#     [0.9680744142, 0.0480036369, -0.2460235344],
#     [-0.2466273395, 0.0069966955, -0.9690851364],
#     [-0.0447982593, 0.9988226555, 0.0186123311],
# ])
R_base_tracker = np.array([
     [1,0,0],
     [0,0,-1],
    [0,1,0],
 ])
BASE_MAPPING_CALIBRATED = True  # 人工标定并更新矩阵后改为 True。

LOG = logging.getLogger('nero.teleop')



# 以下仅为诊断告警阈值，不改变原来的目标门限或发送策略。
TRACKING_WARN_M = 0.02
TRACKING_WARN_RAD = np.deg2rad(10)
TRACKING_WARN_SECONDS = 1.0


def valid_transform(T):
    """检查矩阵，不把 NaN/Inf 送进旋转库。"""
    T = np.asarray(T)
    return (T.shape == (4, 4) and np.isfinite(T).all()
            and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
            and np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-3))


def pose_difference(target, reference):
    """旋转误差使用相对旋转角；不直接拿欧拉角差当真实转角。"""
    if not valid_transform(target) or not valid_transform(reference):
        return None
    dp = target[:3, 3] - reference[:3, 3]
    return {
        'delta_xyz_mm': (dp * 1000).tolist(),
        'position_mm': float(np.linalg.norm(dp) * 1000),
        'rotation_deg': float(np.rad2deg(Rotation.from_matrix(
            target[:3, :3] @ reference[:3, :3].T).magnitude())),
    }


def target_diagnostics(target, previous, actual):
    """分别量化命令跳变和实际跟踪误差；两者不能混为一谈。"""
    step = pose_difference(target, previous)
    error = pose_difference(target, actual)
    return {
        'valid': bool(valid_transform(target)),
        'step': step,
        'tracking_error': error,
        'target_jump': bool(step is None
                            or step['position_mm'] > MAX_POSITION_STEP * 1000
                            or step['rotation_deg'] > np.rad2deg(MAX_ROTATION_STEP)),
        'reachability': 'UNKNOWN_NO_IK_OR_VERIFIED_STATUS_CODE_MAP',
    }


class TeleopDiagnostics:
    """原始状态全量记录；只说明观测事实，不猜 SDK 故障码含义。"""
    def __init__(self):
        self.started = time.monotonic()
        self.first_failure = None
        self.previous_state = None
        self.lag_since = None
        self.last_lag_report = -float('inf')

    def event(self, event, level=logging.INFO, **data):
        payload = dict(event=event, monotonic_s=time.monotonic(),
                       elapsed_s=time.monotonic() - self.started, **data)
        LOG.log(level, 'DIAG %s', json.dumps(
            payload, ensure_ascii=False,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))

    def failure(self, event, **data):
        if self.first_failure is None:
            self.first_failure = dict(event=event, **data)
        self.event(event, logging.ERROR, **data)

    def state(self, status, enabled, context):
        # 只比较控制状态和使能，不让连续变化的关节角产生无意义的状态变化告警。
        current = json.dumps(dict(status=status, enabled=enabled), sort_keys=True, default=str)
        if current != self.previous_state:
            self.event('STATE_CHANGE', context=context, status=status, enabled=enabled,
                       previous_state=self.previous_state)
            self.previous_state = current

    def tracking(self, info):
        error = info['tracking_error']
        now = time.monotonic()
        if error is None:
            self.event('FEEDBACK_UNAVAILABLE', logging.WARNING)
            self.lag_since = None
            return
        large = (error['position_mm'] > TRACKING_WARN_M * 1000
                 or error['rotation_deg'] > np.rad2deg(TRACKING_WARN_RAD))
        if not large:
            self.lag_since = None
            return
        if self.lag_since is None:
            self.lag_since = now
        if now - self.lag_since >= TRACKING_WARN_SECONDS and now - self.last_lag_report >= 1:
            self.event('TRACKING_LAG', logging.WARNING, duration_s=now-self.lag_since,
                       error=error,
                       meaning='持续跟踪误差；可能速度不足、反馈滞后、命令未执行或不可达，不能单独定因')
            self.last_lag_report = now


def setup_debug_log():
    """同时输出终端和文件；FileHandler 每条记录会 flush。"""
    folder = Path(__file__).resolve().parent / 'teleop_debug_logs'
    folder.mkdir(exist_ok=True)
    path = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding='utf-8')],
                        force=True)
    LOG.info('诊断日志=%s', path)
    return path


def snapshot(arm):
    """只读诊断；单项失败也保留其他反馈。"""
    result = {}
    readers = {
        'status': arm.get_arm_status,
        'enabled': lambda: [arm._driver.get_joint_enable_status(i) for i in range(1, 8)],
        'joints_rad': arm.get_joint_positions,
        'flange_m_rad': arm.get_flange_pose,
    }
    for name, reader in readers.items():
        try:
            value = reader()
            result[name] = value.tolist() if isinstance(value, np.ndarray) else value
        except Exception as exc:
            result[name] = f'{type(exc).__name__}: {exc}'
    return result


def check_ready(arm, diag=None, context='state_check'):
    try:
        status = arm.get_arm_status()
        enabled = [arm._driver.get_joint_enable_status(i) for i in range(1, 8)]
        reasons = []
        if status.get('arm_status') != 0:
            reasons.append(f"arm_status={status.get('arm_status')}，原程序要求0")
        if status.get('ctrl_mode') != 1:
            reasons.append(f"ctrl_mode={status.get('ctrl_mode')}，原程序要求1")
        if not all(enabled):
            reasons.append(f"未使能关节={[i for i, value in enumerate(enabled, 1) if not value]}")
    except Exception as exc:
        if diag is not None:
            diag.failure('STATE_READ_EXCEPTION', context=context, error=repr(exc))
        raise
    if diag is not None:
        diag.state(status, enabled, context)
    if reasons:
        if diag is not None:
            diag.failure('STATE_NOT_READY', context=context, reasons=reasons,
                         status=status, enabled=enabled,
                         meaning='此记录发生于脚本退出急停之前；状态数字含义需对照本机SDK')
        raise RuntimeError(f'禁止发送目标: reasons={reasons}, status={status}, enabled={enabled}')
    return status


def pose_to_matrix(pose):
    """NERO 的 m、rad RPY → 齐次矩阵；R=Rz(yaw)Ry(pitch)Rx(roll)。"""
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    return result


def read_tracker_pose(tracker):
    # 复用原底层读取；get_T() 会在无效帧返回旧缓存，因此不用该方法。
    # get_pose_matrix() 返回 OpenVR 的 3×4 device→Standing world 矩阵，
    # 原代码直接使用平移（m），没有毫米换算；也不改写其底层读取方式。
    pose = tracker.get_pose_matrix()
    if pose is None:
        return None
    result = np.eye(4)
    result[:3, :] = np.array(pose)['m']
    rotation = result[:3, :3]
    if (not np.isfinite(result).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
        return None
    return result


def tracker_target(T_tracker_0, T_tracker, T_robot_0):
    # 初始 Tracker 对应当前 TCP；固定初始锚点，避免绝对坐标引起跳动或积分漂移。
    target = T_robot_0.copy()
    delta_p = T_tracker[:3, 3] - T_tracker_0[:3, 3]
    target[:3, 3] += POSITION_SCALE * (R_base_tracker @ delta_p)

    # 平移是在世界系中相减，旋转也采用世界系增量并左乘初始 TCP 姿态。
    # R @ R0.T = R0 @ (R0.T @ R) @ R0.T，不能把局部增量当作世界系增量。
    delta_R = T_tracker[:3, :3] @ T_tracker_0[:3, :3].T
    target[:3, :3] = (
        R_base_tracker @ delta_R @ R_base_tracker.T @ T_robot_0[:3, :3]
    )
    return target


def step_is_valid(target, previous):
    distance = np.linalg.norm(target[:3, 3] - previous[:3, 3])
    angle = Rotation.from_matrix(
        target[:3, :3] @ previous[:3, :3].T
    ).magnitude()
    return (np.isfinite(target).all()
            and distance <= MAX_POSITION_STEP and angle <= MAX_ROTATION_STEP)


class FlangeTargetGate:
    """法兰目标超限后锁定，不因 Tracker 返回附近而自动恢复。"""
    def __init__(self, tracker_pose, flange_pose):
        self.tracker_anchor = tracker_pose.copy()
        self.flange_anchor = flange_pose.copy()
        self.previous = flange_pose.copy()
        self.paused = False

    def target(self, tracker_pose):
        if self.paused:
            return None
        target = tracker_target(self.tracker_anchor, tracker_pose, self.flange_anchor)
        if not step_is_valid(target, self.previous):
            self.paused = True
            LOG.warning('法兰目标超限：%.2fmm / %.2fdeg，已暂停发送；保持静止，按 R 重新对齐',
                        np.linalg.norm(target[:3, 3] - self.previous[:3, 3]) * 1000,
                        np.rad2deg(Rotation.from_matrix(
                            target[:3, :3] @ self.previous[:3, :3].T).magnitude()))
            return None
        return target

    def reanchor(self, tracker_pose, flange_pose, motion_status):
        if tracker_pose is None or motion_status != 0:
            return False
        self.tracker_anchor = tracker_pose.copy()
        self.flange_anchor = flange_pose.copy()
        self.previous = flange_pose.copy()
        self.paused = False
        return True


class ReanchorKeyboard:
    """非阻塞读取 R；保留 Ctrl+C 信号，并在退出时恢复终端。"""
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('请在交互式终端运行，以便按 R 重新对齐；输出可以重定向。')
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        if select.select([self.fd], [], [], 0)[0]:
            return 'r' in os.read(self.fd, 1024).decode('ascii', errors='ignore').lower()
        return False

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)


def main(keyboard):
    setup_debug_log()
    diag = TeleopDiagnostics()
    diag.event('RUN_START', position_scale=POSITION_SCALE, control_hz=CONTROL_HZ,
               max_step_mm=MAX_POSITION_STEP * 1000, max_step_deg=np.rad2deg(MAX_ROTATION_STEP),
               reachability='UNKNOWN: 没有IK和已核验状态码表，保留原始错误字段')
    LOG.info('script=%s python=%s CAN=%s/%s speed=%s%%',
             __file__, sys.executable, NERO_CAN_INTERFACE, NERO_CAN_CHANNEL, NERO_SPEED_PERCENT)
    if not BASE_MAPPING_CALIBRATED:
        raise RuntimeError("请先标定 R_base_tracker，再设置 BASE_MAPPING_CALIBRATED=True。")
    if (not np.isfinite(R_base_tracker).all()
            or not np.allclose(R_base_tracker.T @ R_base_tracker, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R_base_tracker), 1.0, atol=1e-6)):
        raise ValueError("R_base_tracker 必须是正交且行列式为 +1 的旋转矩阵。")

    # 原模块使用同目录绝对导入；只在运行时添加路径、初始化 SteamVR。
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'dp_real_0309'))
    from track import ViveTrackerModule

    tracker_module = None
    arm = None
    motion_started = False
    phase = 'tracker_init'
    last_target = None
    last_sent_pose = None
    previous_tracker = None
    invalid_since = None
    exit_reason = 'not_exiting'
    sent = invalid = rejected = 0
    report_at = time.monotonic()
    try:
        tracker_module = ViveTrackerModule()
        tracker_module.print_discovered_objects()
        devices = tracker_module.return_selected_devices('tracker')
        if len(devices) != 1:
            raise RuntimeError("请只连接原项目使用的那一个 Tracker。")
        tracker = next(iter(devices.values()))

        phase = 'connect'
        arm = NeroArm(can_interface=NERO_CAN_INTERFACE, can_channel=NERO_CAN_CHANNEL)
        LOG.info('driver=%s', type(arm._driver).__module__)
        arm.connect()
        # 复位由操作者单独处理，启动时不自动清除控制器故障。
        time.sleep(0.5)
        LOG.info('before_enable=%s', snapshot(arm))
        phase = 'enable'
        arm.enable()  # 封装内部已重试，成功返回，超时抛异常
        LOG.info('after_enable=%s', snapshot(arm))
        phase = 'normal_mode'
        arm._driver.set_normal_mode()
        time.sleep(0.01)
        LOG.info('after_normal_mode=%s', snapshot(arm))
        check_ready(arm, diag, phase)
        phase = 'alignment'
        dt = 1.0 / CONTROL_HZ
        print("保持 Tracker 和机械臂静止，等待有效初始位姿……")
        deadline = time.monotonic() + 5.0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('初始对齐超过5秒：检查Tracker有效性及机械臂位姿反馈')
            T_tracker_0 = read_tracker_pose(tracker)
            try:
                # 直接以实际法兰建立初始锚点，不读取 TCP。
                T_flange_0 = pose_to_matrix(arm.get_flange_pose())
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(dt)
                continue
            if T_tracker_0 is not None:
                break
            time.sleep(dt)

        if not valid_transform(T_flange_0):
            raise ValueError('初始法兰反馈不是有效位姿')
        gate = FlangeTargetGate(T_tracker_0, T_flange_0)
        previous_tracker = T_tracker_0.copy()
        diag.event('ALIGNED', tracker=T_tracker_0, flange=T_flange_0)
        termios.tcflush(keyboard.fd, termios.TCIFLUSH)
        LOG.info('aligned tracker=%s flange=%s', T_tracker_0.tolist(), T_flange_0.tolist())
        print("法兰初始对齐完成；超限后暂停发送，保持静止按 R 重新对齐；Ctrl+C 急停退出。")
        while True:
            cycle_start = time.monotonic()
            try:
                phase = 'state_check'
                check_ready(arm, diag, phase)
                phase = 'tracker_read'
                T_tracker = read_tracker_pose(tracker)
                if keyboard.read():
                    phase = 'reanchor'
                    if gate.reanchor(T_tracker, pose_to_matrix(arm.get_flange_pose()),
                                     arm.get_arm_status().get('motion_status')):
                        diag.event('REANCHORED', tracker=gate.tracker_anchor, flange=gate.flange_anchor)
                        last_sent_pose = None
                        previous_tracker = T_tracker.copy()
                        diag.lag_since = None
                        LOG.info('R 重新对齐完成；本周期不发送运动，下周期恢复法兰跟随')
                    else:
                        LOG.warning('拒绝重新对齐：Tracker 无有效帧或机械臂尚未报告停止；静止后重新按 R')
                    continue
                if T_tracker is None:
                    invalid += 1
                    if invalid_since is None:
                        invalid_since = time.monotonic()
                        diag.event('TRACKER_INVALID_START', logging.WARNING, last_target=last_target)
                    continue  # 丢失时不发送旧位姿；恢复后仍检查与最后目标的距离。
                if invalid_since is not None:
                    diag.event('TRACKER_RECOVERED', lost_seconds=time.monotonic() - invalid_since)
                    invalid_since = None
                if gate.paused:
                    phase = 'paused_wait_R'
                    rejected += 1
                    continue
                phase = 'target_diagnosis'
                candidate = tracker_target(gate.tracker_anchor, T_tracker, gate.flange_anchor)
                feedback_before = snapshot(arm)
                actual = None
                try:
                    actual = pose_to_matrix(feedback_before['flange_m_rad'])
                except (ValueError, TypeError, KeyError):
                    pass  # 读不到时仍记录原始反馈；不将缺失误判为零误差。
                info = target_diagnostics(candidate, gate.previous, actual)
                tracker_step = pose_difference(T_tracker, previous_tracker)
                previous_tracker = T_tracker.copy()
                diag.event('TARGET_CHECK', target=candidate, previous_target=gate.previous,
                           tracker_step=tracker_step, feedback=feedback_before, **info)
                diag.tracking(info)
                if not info['valid']:
                    diag.failure('INVALID_TARGET', target=candidate)
                    raise ValueError('目标含无效数值或旋转矩阵非法，禁止发送')
                if info['target_jump']:
                    diag.event('TARGET_JUMP', logging.WARNING, **info,
                               meaning='目标超限，本条不发送；锁定等R。暂停不取消已下发运动')
                target_flange = gate.target(T_tracker)
                if target_flange is None:
                    phase = 'paused_wait_R'
                    rejected += 1
                    continue
                pose = np.concatenate((target_flange[:3, 3], Rotation.from_matrix(
                    target_flange[:3, :3]).as_euler('xyz')))
                # 唯一运动发送点：封装的非阻塞 move_pose，目标是法兰、单位 m/rad。
                # 当前无 servo 接口；move_pose 也不是有实时保证的伺服流。
                motion_started = True
                phase = 'move_pose'
                last_target = pose.tolist()
                if sent == 0:
                    LOG.info('first_target_m_rad=%s feedback=%s', last_target, snapshot(arm))
                # 保存两种角度变化，识别 +/-pi 附近欧拉角表示跳变。
                # 不在诊断版中自动改写角度，以免改变待排查的控制行为。
                rpy_delta = None if last_sent_pose is None else np.rad2deg(pose[3:] - last_sent_pose[3:])
                if rpy_delta is not None and np.max(np.abs(rpy_delta)) > 180:
                    diag.event('RPY_REPRESENTATION_JUMP', logging.WARNING,
                               raw_rpy_delta_deg=rpy_delta, physical_step=info['step'],
                               meaning='欧拉角表示跨界；真实转角看physical_step，接口如何处理需确认')
                diag.event('SEND_BEGIN', command_index=sent + 1, pose_m_rad=pose,
                           speed_percent=NERO_SPEED_PERCENT, feedback_before=feedback_before)
                send_started = time.monotonic()
                try:
                    result = arm.move_pose(pose, speed_percent=NERO_SPEED_PERCENT)
                except Exception as exc:
                    diag.failure('MOVE_POSE_EXCEPTION', command_index=sent + 1,
                                 error=repr(exc), target=pose, feedback=snapshot(arm))
                    raise
                send_ms = (time.monotonic() - send_started) * 1000
                sent += 1
                last_sent_pose = pose.copy()
                diag.event('SEND_RETURN', command_index=sent, call_ms=send_ms, return_value=result,
                           feedback_after=snapshot(arm),
                           meaning='调用返回不代表控制器接受、可达或已经到位')
                phase = 'post_send_state_check'
                # 即时反馈可能仍是缓存；下一周期还会继续检查，保留异步故障证据。
                check_ready(arm, diag, phase)
                gate.previous = target_flange.copy()
            finally:
                if time.monotonic() >= report_at:
                    LOG.info('phase=%s sent=%d tracker_invalid=%d rejected=%d last_target=%s feedback=%s',
                             phase, sent, invalid, rejected, last_target, snapshot(arm))
                    report_at = time.monotonic() + 1.0
                cycle_seconds = time.monotonic() - cycle_start
                if cycle_seconds > dt:
                    diag.event('CYCLE_OVERRUN', logging.WARNING, cycle_ms=cycle_seconds * 1000, budget_ms=dt * 1000)
                time.sleep(max(0.0, dt - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        exit_reason = 'USER_CTRL_C'
        diag.event('USER_CTRL_C', phase=phase)
        print("\n停止遥操。")
    except Exception as exc:
        exit_reason = f'{phase}: {type(exc).__name__}: {exc}'
        diag.failure('EXIT_EXCEPTION', phase=phase, error=repr(exc), sent=sent, last_target=last_target)
        LOG.exception('故障 phase=%s sent=%d last_target=%s feedback_before_stop=%s',
                      phase, sent, last_target, snapshot(arm) if arm is not None else None)
        raise
    finally:
        try:
            if arm is not None:
                try:
                    # 无普通 stop/stop_servo；仅断开不能取消在途目标。
                    # 已发送运动时调用公开电子急停，不自动 reset 或失能。
                    if motion_started and arm.is_connected():
                        try:
                            LOG.warning('退出：尝试发送电子急停；下次运行前需检查控制器状态')
                            diag.event('SCRIPT_ESTOP_REQUEST', logging.WARNING,
                                       exit_reason=exit_reason, first_failure=diag.first_failure,
                                       feedback_before=snapshot(arm),
                                       meaning='从此条开始才是本脚本主动请求电子急停，先看它之前的异常')
                            arm.emergency_stop()
                            diag.event('SCRIPT_ESTOP_RETURN', feedback_after=snapshot(arm),
                                       meaning='急停调用返回，不能仅凭此条确认已停稳')
                        except Exception:
                            diag.event('SCRIPT_ESTOP_FAILED', logging.ERROR)
                            LOG.exception('急停请求失败，无法确认机械臂已停止')
                finally:
                    arm.disconnect()
                    diag.event('DISCONNECTED', exit_reason=exit_reason, first_failure=diag.first_failure)
        finally:
            # 原 ViveTrackerModule.__del__ 负责 openvr.shutdown()。
            del tracker_module


if __name__ == '__main__':
    with ReanchorKeyboard() as keyboard:
        main(keyboard)
