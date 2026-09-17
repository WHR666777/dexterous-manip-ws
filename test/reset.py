"""参考 agilex_teleop 的急停→复位→正常模式→失能→使能流程。

使用 config.py 中的单臂通道和 V111 封装，不发送位置目标。
失能会失去力矩，执行前必须可靠支撑机械臂。
"""
import time
from _common import connected_arm, wait_feedback


def reset_arm(arm):
    print('[1/6] 发送电子急停', flush=True)
    arm.emergency_stop()
    print('[2/6] 发送控制器复位', flush=True)
    arm.reset()
    time.sleep(0.2)
    print('[3/6] 切换正常模式', flush=True)
    arm._driver.set_normal_mode()
    time.sleep(0.3)
    print('[4/6] 失能，最多等待5秒', flush=True)
    # NeroArm 成功返回 None、失败抛异常，不能写 while not arm.disable()。
    arm.disable(timeout=5.0, poll_interval=0.1)
    print('[5/6] 重新使能，最多等待5秒', flush=True)
    arm.enable(timeout=5.0, poll_interval=0.1)
    print('[6/6] 验证状态，最多等待5秒', flush=True)
    deadline = time.monotonic() + 5.0
    consecutive = 0
    while time.monotonic() < deadline:
        status = arm.get_arm_status()
        enabled = [arm._driver.get_joint_enable_status(i) for i in range(1, 8)]
        print(f'状态：{status}\n七轴使能：{enabled}', flush=True)
        ready = (status['arm_status'] == 0 and status['ctrl_mode'] == 1 and all(enabled))
        consecutive = consecutive + 1 if ready else 0
        if consecutive >= 3:
            print('重置成功：连续三次报告正常、CAN控制、七轴使能；保持使能。', flush=True)
            return
        time.sleep(0.25)
    raise TimeoutError('重置后未稳定恢复正常，请查看上方控制器状态和七轴使能反馈')


def main():
    with connected_arm() as arm:
        print('复位前：', wait_feedback(arm.get_arm_status), flush=True)
        if input('将急停、复位、失能并重新使能；会暂时失去力矩！确认已可靠支撑后输入 RESET：').strip().upper() != 'RESET':
            print('已取消')
            return
        try:
            reset_arm(arm)
        except Exception:
            print('重置失败；不继续发送复位或运动指令。', flush=True)
            try:
                print('失败时状态：', arm.get_arm_status(), flush=True)
                print('七轴使能：', [arm._driver.get_joint_enable_status(i) for i in range(1, 8)], flush=True)
            except Exception as exc:
                print('读取失败状态时异常：', exc, flush=True)
            raise


if __name__ == '__main__':
    main()
