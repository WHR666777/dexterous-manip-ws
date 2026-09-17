"""绝对法兰位姿示例：Base 坐标系，xyz 米、xyz-RPY 弧度。"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from _common import connected_arm, wait_feedback

TARGET_POSE = [-0.27,      -0.02349,     0.448,      -1.57079633,  0,          3.14155775] # 可填写 [x, y, z, roll, pitch, yaw]；None 时运行后输入六个数。
SPEED_PERCENT = 20


def read_target(configured):
    if configured is None:
        configured = input('输入绝对法兰位姿 x y z roll pitch yaw（m/rad，空格分隔）：').split()
    target = np.asarray(configured, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError('目标必须是六个有限数值 [x,y,z,roll,pitch,yaw]')
    if abs(target[3]) > np.pi or abs(target[4]) > np.pi / 2 or abs(target[5]) > np.pi:
        raise ValueError('RPY 使用弧度：roll/yaw 在 ±pi，pitch 在 ±pi/2 内')
    return target.copy()


def require_ready(arm):
    status = arm.get_arm_status()
    if status['arm_status'] != 0 or status['ctrl_mode'] != 1 or not arm.is_enabled():
        raise RuntimeError(f'运动条件不满足：{status}, enabled={arm.is_enabled()}')


def main():
    target = read_target(TARGET_POSE)
    print('绝对目标 m/rad：', target, flush=True)
    with connected_arm() as arm:
        print('启动状态：', wait_feedback(arm.get_arm_status), flush=True)
        if input('将使能并移动到上述绝对位姿，确认后输入 MOVE：').strip() != 'MOVE':
            print('已取消')
            return
        if arm.get_arm_status()['arm_status'] != 0:
            raise RuntimeError('控制器异常，请先单独处理；本例不自动复位。')
        arm.enable()
        arm._driver.set_normal_mode()  # V111 模式/CAN push。
        deadline = time.monotonic() + 3
        while True:
            try:
                require_ready(arm)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        print('起点 m/rad：', wait_feedback(arm.get_flange_pose), flush=True)
        print('目标 m/rad：', target, flush=True)
        attempted = False
        try:
            require_ready(arm)
            attempted = True
            arm.move_pose(target, speed_percent=SPEED_PERCENT)
            deadline = time.monotonic() + 10
            settled = 0
            while time.monotonic() < deadline:
                time.sleep(0.1)
                require_ready(arm)
                actual = arm.get_flange_pose()
                distance = np.linalg.norm(actual[:3] - target[:3])
                angle = (Rotation.from_euler('xyz', actual[3:]) *
                         Rotation.from_euler('xyz', target[3:]).inv()).magnitude()
                reached = (distance < 0.001 and angle < np.deg2rad(0.5)
                           and arm.get_arm_status()['motion_status'] == 0)
                settled = settled + 1 if reached else 0
                print(f'位置误差={distance * 1000:.2f}mm 姿态误差={np.rad2deg(angle):.2f}deg', flush=True)
                if settled >= 3:
                    print('反馈已到目标附近；保持使能，不自动返回。')
                    attempted = False
                    return
            raise TimeoutError('10秒内未确认到位')
        finally:
            if attempted:
                print('运动中断/异常，尝试发送电子急停。', flush=True)
                arm.emergency_stop()


if __name__ == '__main__':
    main()
