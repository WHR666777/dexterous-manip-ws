"""Vive Tracker → 法兰目标 → 原 NeroIK → move_js，独立最小入口。

运行：python3 -B teleop_nero_tracker_minimal.py（直接使能、对齐并跟随，无 MOVE 确认）
R：以当前 Tracker/真实法兰重新对齐；Q/Ctrl+C：退出并断开，不失能、不急停。
O：灵巧手张开；C：闭合（无需回车）；手型和 CAN 通道使用 config.py 的 L20 配置。
B：开始录制，S：停止保存，Q/Ctrl+C 退出也保存；--output 指定数据目录，记录频率 10 Hz。
调试仅输出终端；仅录制时创建数据文件。退出不等于取消机械臂在途运动。
依赖 numpy/scipy/openvr/pyAgxArm、robot_control 及原 IK 的模型/求解器；不依赖其他 teleop。
"""
import argparse
from collections import deque
import logging
from contextlib import ExitStack
import os
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation

from config import L20_HAND_TYPE, L20_CAN_CHANNEL, L20_SPEED
from robot_control import NeroArm, LinkerHandL20
from robot_control.nero_ik import NeroIK, valid_transform


class Keyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('请在交互终端运行，以便使用 R/Q/O/C/B/S；输出可以重定向。')
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.pending = deque()
        return self

    def read(self):
        if select.select([self.fd], [], [], 0)[0]:
            keys = os.read(self.fd, 1024).decode('ascii', errors='ignore').lower()
            # Q 立即退出；其余按键保序，避免同批 O/B 或 S/B 丢失录制操作。
            if 'q' in keys:
                self.pending.clear()
                return 'q'
            self.pending.extend(key for key in keys if key in 'rocbs')
        return self.pending.popleft() if self.pending else ''

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
        