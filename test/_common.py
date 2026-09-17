"""独立硬件示例共用入口；导入时不连接机械臂。"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
from robot_control import NeroArm


@contextmanager
def connected_arm():
    arm = NeroArm(can_interface=NERO_CAN_INTERFACE, can_channel=NERO_CAN_CHANNEL)
    try:
        arm.connect()
        print(f'已打开 {NERO_CAN_INTERFACE}/{NERO_CAN_CHANNEL}，驱动固定 V111', flush=True)
        yield arm
    finally:
        arm.disconnect()  # 不自动失能，避免失去支撑。


def wait_feedback(reader, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return reader()
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def print_feedback(method):
    try:
        with connected_arm() as arm:
            reader = getattr(arm, method)
            print(wait_feedback(reader), flush=True)
            for _ in range(19):
                time.sleep(0.25)
                print(reader(), flush=True)
    except KeyboardInterrupt:
        print('结束读取')
