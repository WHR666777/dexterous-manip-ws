"""LinkerHand L20 单主动槽位、小增量的默认只读示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import L20_CAN_CHANNEL, L20_HAND_MODEL, L20_HAND_TYPE  # noqa: E402
from robot_control import L20_ACTIVE_POSITION_INDICES, LinkerHandL20  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建只读默认值的命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="默认只读的 LinkerHand {0} 状态与小动作示例。".format(L20_HAND_MODEL),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="请求进入二次确认后的单次小动作分支。",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        choices=L20_ACTIVE_POSITION_INDICES,
        default=L20_ACTIVE_POSITION_INDICES[0],
        help="零基 L20 主动位置槽位；保留槽位不可选。",
    )
    parser.add_argument(
        "--delta-raw",
        type=int,
        choices=(1, 2, 5, 10),
        default=1,
        help="相对当前反馈增加的保守原始位置增量，单位为 L20 raw value。",
    )
    return parser


def confirm_execution() -> bool:
    """要求操作者精确输入 EXECUTE 后才允许发送动作。"""
    return input("Type EXECUTE to send a small robot command: ").strip() == "EXECUTE"


def run(
    args: argparse.Namespace,
    hand_factory: Callable[..., LinkerHandL20] = LinkerHandL20,
    confirm: Callable[[], bool] = confirm_execution,
) -> int:
    """连接 L20、打印已验证反馈，并在双重确认后改变一个主动槽位。

    Parameters
    ----------
    args : argparse.Namespace
        ``build_parser`` 解析的参数；``joint_index`` 为零基主动位置索引，
        ``delta_raw`` 为整数 raw 增量。
    hand_factory : callable, optional
        创建公开 :class:`LinkerHandL20` 接口的工厂，默认创建真实封装；可注入
        无硬件测试替身。
    confirm : callable, optional
        返回精确执行确认结果的函数。

    Returns
    -------
    int
        正常或未确认时为 ``0``；候选 raw 位置不在 ``[0, 255]`` 时为 ``2``。

    Notes
    -----
    默认路径仅连接和读取 SDK 版本、20 槽位置以及五电机状态，不发送位置命令。
    执行路径只在 ``--execute`` 与精确确认均成立后，从当前反馈复制 20 槽 raw
    位置并改变一个主动槽位；不使用张开、握拳等预设。
    """
    hand = None
    try:
        hand = hand_factory(hand_type=L20_HAND_TYPE, can_channel=L20_CAN_CHANNEL)
        hand.connect()
        print("L20 SDK version:", hand.get_sdk_version())
        print("L20 position (raw):", hand.get_joint_positions_raw())
        print("L20 speed:", hand.get_speed())
        print("L20 current:", hand.get_current())
        print("L20 temperature:", hand.get_temperature())
        print("L20 fault:", hand.get_fault())
        if not args.execute:
            print("Read-only mode: no L20 position command was sent.")
            return 0
        if not confirm():
            print("Execution was not confirmed: no L20 position command was sent.")
            return 0

        target = hand.get_joint_positions_raw().copy()
        target[args.joint_index] += args.delta_raw
        if target[args.joint_index] > 255:
            print("Candidate L20 raw target exceeds 255: no position command was sent.")
            return 2
        hand.set_joint_positions_raw(target)
        return 0
    except KeyboardInterrupt:
        print("Interrupted: no further L20 commands will be sent.")
        return 130
    finally:
        if hand is not None:
            try:
                hand.disconnect()
            except Exception as error:
                print("L20 disconnect failed:", error)


def main() -> int:
    """运行示例并返回进程退出码。"""
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
