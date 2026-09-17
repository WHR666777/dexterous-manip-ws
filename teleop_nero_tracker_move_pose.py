"""单个 Vive Tracker 相对遥操作 NERO；标定下方映射后运行本文件。"""

import sys
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


POSITION_SCALE = 0.5
CONTROL_HZ = 10  # 沿用 dp_real_0309/teleoperation.py 的目标更新频率。
MAX_POSITION_STEP = 0.01  # 每次已发送目标之间最多 1 cm，超限锁定暂停。
MAX_ROTATION_STEP = np.deg2rad(5)

# 原 teleoperation.py 的位置映射为 D @ A @ D @ R_tracker_0.T；
# D=diag(1,-1,-1)，A=Rx(135°)，前面的 R_tracker_0.T 来自原点标定。
# D @ A @ D = A。下面仅保留 A 作为标定起点，不冒充实际外参：
# R_base_tracker 必须替换为「SteamVR Standing 世界系 → NERO Base」旋转，
# 包含原 Tracker 参考朝向及 UR5 Base → NERO Base 的实际安装旋转。
R_base_tracker = np.array([
    [0.9680744142, 0.0480036369, -0.2460235344],
    [-0.2466273395, 0.0069966955, -0.9690851364],
    [-0.0447982593, 0.9988226555, 0.0186123311],
])
BASE_MAPPING_CALIBRATED = True  # 人工标定并更新矩阵后改为 True。

LOG = logging.getLogger('nero.teleop')


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


def check_ready(arm):
    status = arm.get_arm_status()
    enabled = [arm._driver.get_joint_enable_status(i) for i in range(1, 8)]
    if status['arm_status'] != 0 or status['ctrl_mode'] != 1 or not all(enabled):
        raise RuntimeError(f'禁止发送目标: status={status}, enabled={enabled}')


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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
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
        check_ready(arm)
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

        gate = FlangeTargetGate(T_tracker_0, T_flange_0)
        termios.tcflush(keyboard.fd, termios.TCIFLUSH)
        LOG.info('aligned tracker=%s flange=%s', T_tracker_0.tolist(), T_flange_0.tolist())
        print("法兰初始对齐完成；超限后暂停发送，保持静止按 R 重新对齐；Ctrl+C 急停退出。")
        while True:
            cycle_start = time.monotonic()
            try:
                phase = 'state_check'
                check_ready(arm)
                phase = 'tracker_read'
                T_tracker = read_tracker_pose(tracker)
                if keyboard.read():
                    phase = 'reanchor'
                    if gate.reanchor(T_tracker, pose_to_matrix(arm.get_flange_pose()),
                                     arm.get_arm_status().get('motion_status')):
                        LOG.info('R 重新对齐完成；本周期不发送运动，下周期恢复法兰跟随')
                    else:
                        LOG.warning('拒绝重新对齐：Tracker 无有效帧或机械臂尚未报告停止；静止后重新按 R')
                    continue
                if T_tracker is None:
                    invalid += 1
                    continue  # 丢失时不发送旧位姿；恢复后仍检查与最后目标的距离。
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
                arm.move_pose(pose, speed_percent=NERO_SPEED_PERCENT)
                sent += 1
                gate.previous = target_flange.copy()
            finally:
                if time.monotonic() >= report_at:
                    LOG.info('phase=%s sent=%d tracker_invalid=%d rejected=%d last_target=%s feedback=%s',
                             phase, sent, invalid, rejected, last_target, snapshot(arm))
                    report_at = time.monotonic() + 1.0
                time.sleep(max(0.0, dt - (time.monotonic() - cycle_start)))
    except KeyboardInterrupt:
        print("\n停止遥操。")
    except Exception:
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
                            arm.emergency_stop()
                        except Exception:
                            LOG.exception('急停请求失败，无法确认机械臂已停止')
                finally:
                    arm.disconnect()
        finally:
            # 原 ViveTrackerModule.__del__ 负责 openvr.shutdown()。
            del tracker_module


if __name__ == '__main__':
    with ReanchorKeyboard() as keyboard:
        main(keyboard)
