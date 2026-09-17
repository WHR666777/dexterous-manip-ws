"""读取状态：arm_status 0正常/1急停/2无解，ctrl_mode 1为CAN。"""
from _common import print_feedback

if __name__ == '__main__':
    print_feedback('get_arm_status')
