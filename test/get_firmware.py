"""查询实际固件版本；不使能。"""
from _common import connected_arm

if __name__ == '__main__':
    with connected_arm() as arm:
        print(arm.get_firmware(), flush=True)
