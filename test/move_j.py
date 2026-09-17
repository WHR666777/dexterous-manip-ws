"""绝对关节角 move_j 最小测试；请先填写下方七轴目标，单位为度。

正常结束不失能、不返回；异常/中断请求电子急停。
使能或急停均不能保证故障时不下落，请先落实机械支撑和现场安全措施。
"""
import time

import numpy as np

from _common import connected_arm, wait_feedback


TARGET_DEG = [
    0,  # J1：填写绝对角度（度）
    0,  # J2
    20,  # J3
    20,  # J4
    0,  # J5
    0,  # J6
    20,  # J7
]
SPEED_PERCENT = 10


def read_target(values):
    if len(values) != 7 or any(value is None for value in values):
        raise ValueError('请先将 TARGET_DEG 中七个 None 全部替换为目标角度，单位为度。')
    degrees = np.asarray(values, dtype=float)
    if degrees.shape != (7,) or not np.isfinite(degrees).all():
        raise ValueError('目标必须是七个有限角度。')
    return np.deg2rad(degrees)


def require_ready(arm):
    status = arm.get_arm_status()
    if status.get('arm_status') != 0 or status.get('ctrl_mode') != 1 or not arm.is_enabled():
        raise RuntimeError(f'机械臂未就绪：{status}；本脚本不自动复位。')


def main():
    target = read_target(TARGET_DEG)  # 未填写时在连接硬件前退出。
    with connected_arm() as arm:
        current = wait_feedback(arm.get_joint_positions)
        arm.validate_joint_command(target)
        print('当前关节 deg：', np.round(np.rad2deg(current), 3))
        print('绝对目标 deg：', np.round(np.rad2deg(target), 3))
        print('各轴变化 deg：', np.round(np.rad2deg(target - current), 3))
        print('速度百分比：', SPEED_PERCENT)
        if input('确认目标、运动路径及周围安全后输入 MOVE：').strip() != 'MOVE':
            print('已取消，不发送运动。')
            return
        if arm.get_arm_status().get('arm_status') != 0:
            raise RuntimeError('控制器存在异常，请先检查；不自动复位或使能。')
        completed = False
        try:
            arm.enable()
            arm._driver.set_normal_mode()
            wait_feedback(lambda: require_ready(arm), timeout=3)
            # 封装执行官方限位检查，随后只调用一次 SDK move_j。
            arm.move_joints(target, speed_percent=SPEED_PERCENT)
            deadline = time.monotonic() + 10
            settled = 0
            while time.monotonic() < deadline:
                time.sleep(.1)
                require_ready(arm)
                actual = arm.get_joint_positions()
                error = np.rad2deg(target - actual)
                print('实际 deg：', np.round(np.rad2deg(actual), 3),
                      '目标减实际 deg：', np.round(error, 3), flush=True)
                reached = np.max(np.abs(error)) < .5 and arm.get_arm_status().get('motion_status') == 0
                settled = settled + 1 if reached else 0
                if settled >= 3:
                    completed = True
                    print('已连续确认到位；保持使能，不自动返回。')
                    return
            raise TimeoutError('10秒内未确认到位。')
        finally:
            if not completed and arm.is_connected():
                print('异常或中断：请求电子急停；不能保证故障时不下落。', flush=True)
                arm.emergency_stop()


if __name__ == '__main__':
    main()
