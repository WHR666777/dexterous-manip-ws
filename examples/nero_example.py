"""Nero 单关节、小增量的默认只读示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    NERO_CAN_CHANNEL,
    NERO_CAN_INTERFACE,
    NERO_MAX_JOINT_DELTA,
    NERO_SPEED_PERCENT,
)
from examples._safety import _disconnect_or_report  # noqa: E402
from robot_control import NeroArm  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建只读默认值的命令行解析器。"""
    parser = argparse.ArgumentParser(description="默认只读的 Nero 状态与小动作示例。")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="请求进入二次确认后的单次小动作分支。",
    )
    parser.add_argument(
        "--joint-index",
        type=int,
        choices=range(7),
        default=0,
        help="零基 Nero 关节索引；仅在执行模式使用。",
    )
    parser.add_argument(
        "--delta-rad",
        type=float,
        choices=(0.005, 0.01, 0.02),
        default=0.005,
        help="相对当前反馈增加的保守关节增量，单位 rad。",
    )
    return parser


def confirm_execution() -> bool:
    """要求操作者精确输入 EXECUTE 后才允许发送动作。"""
    return input("Type EXECUTE to send a small robot command: ").strip() == "EXECUTE"


def run(
    args: argparse.Namespace,
    arm_factory: Callable[..., NeroArm] = NeroArm,
    confirm: Callable[[], bool] = confirm_execution,
) -> int:
    """连接 Nero、读取状态，并在双重确认后发送一次相对小动作。

    Parameters
    ----------
    args : argparse.Namespace
        ``build_parser`` 解析的参数；``joint_index`` 是零基索引，``delta_rad``
        单位为 rad。
    arm_factory : callable, optional
        创建公开 :class:`NeroArm` 接口的工厂，默认创建真实封装；可注入无硬件
        测试替身。
    confirm : callable, optional
        返回精确执行确认结果的函数。

    Returns
    -------
    int
        正常或未确认时为 ``0``；仅断开失败时为 ``1``；当前反馈加增量被 Wrapper
        拒绝时为 ``2``；收到 ``KeyboardInterrupt`` 时为 ``130``。

    Notes
    -----
    默认路径只连接和读取，不使能、不发送运动。执行路径只在 ``--execute`` 与
    精确确认均成立后使能，并以当前七维 rad 反馈为基准改变一个关节。finally
    始终尝试断开；断开失败报告到标准错误流且不会掩盖已有主异常。不自动 disable，
    避免可能的机械臂下落。
    """
    arm = None
    exit_code = 0
    try:
        arm = arm_factory(
            can_interface=NERO_CAN_INTERFACE,
            can_channel=NERO_CAN_CHANNEL,
            max_joint_delta=NERO_MAX_JOINT_DELTA,
        )
        arm.connect()
        print("Nero healthy:", arm.is_ok())
        print("Nero status:", arm.get_arm_status())
        print("Nero joint position (rad):", arm.get_joint_positions())
        if not args.execute:
            print("Read-only mode: no Nero enable or motion command was sent.")
        elif not confirm():
            print("Execution was not confirmed: no Nero enable or motion command was sent.")
        else:
            arm.enable()
            target = arm.get_joint_positions().copy()
            target[args.joint_index] += args.delta_rad
            try:
                arm.move_joints(target, speed_percent=NERO_SPEED_PERCENT)
            except ValueError as error:
                print("Nero command rejected:", error, file=sys.stderr)
                exit_code = 2
            else:
                print("Nero resulting joint position (rad):", arm.get_joint_positions())
    except KeyboardInterrupt:
        print("Interrupted: no further Nero commands will be sent.")
        exit_code = 130
    finally:
        if arm is not None:
            cleanup_succeeded = _disconnect_or_report(arm, "Nero")
            if not cleanup_succeeded and sys.exc_info()[0] is None and exit_code == 0:
                exit_code = 1
    return exit_code


def main() -> int:
    """运行示例并返回进程退出码。"""
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
