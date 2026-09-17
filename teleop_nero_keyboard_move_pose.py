"""键盘法兰点动：每次按键发送一次 move_pose，不执行本地 IK。

W/S、A/D、R/F：Base X/Y/Z 正负平移（默认 2 mm）。
I/K、J/L、U/O：绕 Base X/Y/Z 正负旋转（默认 1°）。
P 读取反馈；Q 正常退出保持使能；空格/Esc/Ctrl+C 请求急停退出。
长按按键可能产生系统重复按键；目标始终从当前实际法兰计算。
Q 不取消已下发目标，机械臂可能继续完成最后一次运动。
"""

import argparse
import termios
import time

import numpy as np
from scipy.spatial.transform import Rotation

from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
from robot_control import NeroArm
from teleop_nero_tracker import pose_to_matrix, step_is_valid
from teleop_nero_keyboard_move_js import TerminalKeyboard, keyboard_target, checked_pose


def require_ready(arm):
    status = arm.get_arm_status()
    if (not arm.is_connected() or not arm.is_ok() or not arm.is_enabled()
            or status.get('arm_status') != 0 or status.get('ctrl_mode') != 1):
        raise RuntimeError(f'机械臂未就绪，禁止发送目标：{status}')


def prepare_arm(arm):
    """沿用成功的 move_pose 示例初始化，不自动复位故障。"""
    arm.enable()
    arm._driver.set_normal_mode()
    deadline = time.monotonic() + 5
    while True:
        try:
            require_ready(arm)
            checked_pose(pose_to_matrix(arm.get_flange_pose()))
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def command_key(arm, key, translation_step, rotation_step, speed_percent):
    """仅对运动按键发送一次；使用实测法兰，不累计旧目标。"""
    if key.lower() not in 'wsadrfikjluo':
        return False
    require_ready(arm)
    actual = checked_pose(pose_to_matrix(arm.get_flange_pose()))
    target = keyboard_target(key, actual, translation_step, rotation_step)
    if not step_is_valid(target, actual):
        raise ValueError('法兰目标变化超过 1 cm / 5°，拒绝发送')
    pose = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_euler('xyz')]
    require_ready(arm)
    arm.move_pose(pose.tolist(), speed_percent=speed_percent)
    print(f'按键 {key.upper()}：move_pose 已调用；目标法兰 m/rad：',
          np.round(pose, 6), flush=True)
    print('目标 xyz mm：', np.round(pose[:3] * 1000, 3),
          '姿态 deg：', np.round(np.rad2deg(pose[3:]), 3), flush=True)
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--channel', default=NERO_CAN_CHANNEL)
    parser.add_argument('--interface', default=NERO_CAN_INTERFACE)
    parser.add_argument('--step-mm', type=float, default=2, help='平移步长，最大10 mm')
    parser.add_argument('--step-deg', type=float, default=1, help='旋转步长，最大5°')
    parser.add_argument('--speed', type=int, default=10, help='速度百分比，默认10，范围1–100')
    args = parser.parse_args(argv)
    for name, value, maximum in [('平移步长', args.step_mm, 10),
                                  ('旋转步长', args.step_deg, 5), ('速度百分比', args.speed, 100)]:
        if not np.isfinite(value) or not 0 < value <= maximum:
            parser.error(f'{name} 必须在 (0, {maximum}] 内')
    return args


def main(argv=None):
    args = parse_args(argv)
    arm = None
    normal_exit = False
    try:
        with TerminalKeyboard() as keyboard:
            arm = NeroArm(can_interface=args.interface, can_channel=args.channel)
            arm.connect()
            prepare_arm(arm)
            termios.tcflush(keyboard.fd, termios.TCIFLUSH)
            print(__doc__, flush=True)
            print(f'V111；步长 {args.step_mm}mm / {args.step_deg}°；速度 {args.speed}%', flush=True)
            while True:
                key = keyboard.read()
                if key == 'quit':
                    normal_exit = True
                    print('正常退出：不急停、不失能；最后目标仍可能继续执行。', flush=True)
                    break
                if key == 'estop':
                    print('请求电子急停并退出。', flush=True)
                    break
                require_ready(arm)
                if key == 'p':
                    print('实际法兰 m/rad：', arm.get_flange_pose(),
                          '状态：', arm.get_arm_status(), flush=True)
                elif key is not None:
                    command_key(arm, key, args.step_mm / 1000,
                                np.deg2rad(args.step_deg), args.speed)
                time.sleep(.02)
    except KeyboardInterrupt:
        print('Ctrl+C：请求电子急停。', flush=True)
    finally:
        if arm is not None:
            try:
                if not normal_exit and arm.is_connected():
                    try:
                        arm.emergency_stop()
                    except Exception as exc:
                        print(f'急停请求失败，无法确认停止，请使用现场急停：{exc}', flush=True)
            finally:
                arm.disconnect()


if __name__ == '__main__':
    main()
