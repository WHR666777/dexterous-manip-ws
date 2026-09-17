"""读取 TCP [x,y,z,roll,pitch,yaw]，单位 m/rad；不使能。

本脚本新建 SDK 实例；默认 TCP offset 为零，TCP 与法兰重合。
其他进程设置的 SDK TCP offset 不会自动传入本脚本。
"""
from _common import print_feedback

if __name__ == '__main__':
    print_feedback('get_tcp_pose')
