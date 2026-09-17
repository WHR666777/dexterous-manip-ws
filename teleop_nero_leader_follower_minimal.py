"""最小 NERO 主从遥操：主臂零力拖动，从臂对齐后直接跟随七轴角度。

运行：python3 teleop_nero_leader_follower_minimal.py [--leader can1 --follower can0]
启动时保持主臂静止，输入 A 允许从臂低速对齐；O 张开手，C 闭合手，Q/Ctrl+C 退出。
B 开始录制，S 停止保存；Q/Ctrl+C 退出也保存。--output 指定 10 Hz 数据目录。
两臂需同型号、同零位。Q 退出保持使能，仅断开通信；运动后异常/Ctrl+C 退出仍请求从臂急停。
"""
import argparse
from contextlib import ExitStack
import time

import numpy as np

# from config import L20_HAND_TYPE, L20_CAN_CHANNEL, L20_SPEED
from robot_control import NeroArm
# from robot_control import LinkerHandL20
from teleop_nero_tracker_minimal import Keyboard
# from teleop_recording import EpisodeRecorder

LEADER_CHANNEL = 'can1'
FOLLOWER_CHANNEL = 'can0'
CONTROL_HZ = 20  # 直接跟随频率；不额外裁剪目标步长。
STARTUP_TIMEOUT_S = 5.0
ALIGN_SPEED_PERCENT = 5  # 仅用于初始对齐，不控制后续 move_js 的速度。
ALIGN_TIMEOUT_S = 30.0
ALIGN_TOLERANCE_RAD = np.deg2rad(.5)


def start_arm(arm):
    """启动 CAN 推送，确认无已知故障、七轴已使能且反馈完整。"""
    since = time.time()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        try:
            status = arm.get_arm_status()
        except RuntimeError:
            status = {}  # CAN 刚连接时可能尚无状态帧，继续使能并等待。
        if status.get('arm_status') not in (None, 0) or any((status.get('err_status') or {}).values()):
            raise RuntimeError(f'机械臂启动故障：{status}')
        enabled = arm._driver.enable()
        arm._driver.set_normal_mode()
        # SDK 缓存初始值不代表真实反馈：启动后每轴和状态帧都要收到。
        try:
            messages = arm.get_raw_motor_states() + [arm.get_raw_arm_status()]
        except RuntimeError:
            messages = []
        if (enabled and len(messages) == 8 and all(message.timestamp >= since for message in messages)
                and status.get('arm_status') == 0 and status.get('ctrl_mode') == 1
                and all(arm._driver.get_joint_enable_status(i) for i in range(1, 8))):
            arm.validate_joint_command(arm.get_joint_positions())
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('机械臂初始化反馈超时')
        time.sleep(.01)


def leader_joints(leader):
    """leader 模式使用专用角度流，普通电机状态流此时可能停止更新。"""
    return np.asarray(leader._driver.get_leader_joint_angles().msg, dtype=float).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leader', default=LEADER_CHANNEL)
    parser.add_argument('--follower', default=FOLLOWER_CHANNEL)
    parser.add_argument('--output', '-o', default='data/teleop_nero_leader_follower_minimal', help='录制数据目录')
    args = parser.parse_args()
    if args.leader == args.follower:
        parser.error('主从不能使用同一 CAN 接口')

    try:
        with ExitStack() as resources:
            # recorder = EpisodeRecorder(args.output)
            # resources.callback(recorder.stop)  # 退出先执行设备清理，再保存当前录制片段。
            leader = NeroArm(can_interface='socketcan', can_channel=args.leader)
            resources.callback(leader.disconnect)
            follower = NeroArm(can_interface='socketcan', can_channel=args.follower)
            resources.callback(follower.disconnect)
            leader.connect()
            follower.connect()
            start_arm(leader)
            start_arm(follower)
            since = time.time()
            leader._driver.set_leader_mode()
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            # 切换后先收齐七帧，避免把聚合缓存中尚未更新的零值当目标。
            while True:
                # SDK 收到对应 CAN 帧后才创建属性；尚未收到时继续等待。
                messages = [getattr(leader._driver._parser, f'leader_joint_{i}', None)
                            for i in range(1, 8)]
                if all(message is not None and message.timestamp >= since for message in messages):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('主臂 leader 七轴反馈超时')
                time.sleep(.01)
            target = leader_joints(leader)
            follower.validate_joint_command(target)
            print('对齐目标（度）：', np.rad2deg(target).round(2))
            if input('保持主臂静止，输入 A 开始对齐：').strip() != 'A':
                return

            # hand = LinkerHandL20(hand_type=L20_HAND_TYPE, can_channel=L20_CAN_CHANNEL)
            # resources.callback(hand.disconnect)
            # hand.connect()  # 手型/通道在 config.py 中设置；仅按键时发送姿态。
            # hand.set_speed(L20_SPEED)  # 读取 config.py 的五指速度，张开和闭合共用。
            keyboard = resources.enter_context(Keyboard())
            if np.max(np.abs(leader_joints(leader) - target)) >= ALIGN_TOLERANCE_RAD:
                raise RuntimeError('确认期间主臂移动，请重新启动')
            # Q 正常退出保持使能；运动发送失败等异常仍执行急停。
            keep_enabled = False

            def cleanup_follower():
                if not keep_enabled:
                    follower.emergency_stop()

            resources.callback(cleanup_follower)
            follower.move_joints(target, speed_percent=ALIGN_SPEED_PERCENT)
            deadline = time.monotonic() + ALIGN_TIMEOUT_S
            settled = 0
            while settled < 3:
                if keyboard.read() == 'q':
                    keep_enabled = True
                    return
                if np.max(np.abs(leader_joints(leader) - target)) >= ALIGN_TOLERANCE_RAD:
                    raise RuntimeError('对齐期间请保持主臂静止')
                reached = (np.max(np.abs(follower.get_joint_positions() - target)) < ALIGN_TOLERANCE_RAD
                           and follower.get_arm_status().get('motion_status') == 0)
                settled = settled + 1 if reached else 0
                if time.monotonic() >= deadline:
                    raise TimeoutError('从臂对齐超时')
                time.sleep(1 / CONTROL_HZ)

            print('开始跟随：O 张开，C 闭合，B 录制，S 保存，Q/Ctrl+C 退出并保存。')
            while True:
                started = time.monotonic()
                key = keyboard.read()
                if key == 'q':
                    keep_enabled = True
                    break
                # recorder.key(key)
                # if key in ('o', 'c'):
                #     hand.set_gripper(key == 'o')
                #     recorder.gripper(key == 'o')
                command = leader_joints(leader)
                follower.move_js(command)  # 保留封装/SDK 的数值及关节限位校验。
                # recorder.command(command)  # 只记录发送成功的动作，观测读取从臂反馈。
                # recorder.capture(follower, hand)
                time.sleep(max(0., 1 / CONTROL_HZ - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print('已中断，退出。')


if __name__ == '__main__':
    main()
