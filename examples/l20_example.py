"""LinkerHand L20 单主动槽位、小增量的默认只读示例。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._safety import _disconnect_or_report  # noqa: E402
from robot_control import L20_ACTIVE_POSITION_INDICES, LinkerHandL20  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建默认只读的 L20 命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        配置好执行门、主动槽位和有限 raw 增量选项的解析器。

    Notes
    -----
    仅创建内存中的解析器，不构造 Wrapper、不访问 CAN，也不发送硬件命令。
    """
    parser = argparse.ArgumentParser(
        description="默认只读的 LinkerHand L20 状态与小动作示例。",
    )
    parser.add_argument("--can-channel", required=True, help="现场确认的 L20 CAN 通道，例如 can1。")
    parser.add_argument("--hand-type", choices=("left", "right"), default="right")
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
        choices=(-10, -5, -2, -1, 1, 2, 5, 10),
        default=1,
        help="相对当前反馈增减的保守原始位置量，单位为 L20 raw value。",
    )
    parser.add_argument(
        "--speed-raw",
        type=int,
        choices=(10, 20, 30, 50),
        default=30,
        help="动作前设置的 G20 五指保守原始速度；默认 30。",
    )
    return parser


def confirm_execution() -> bool:
    """阻塞等待操作者输入精确的执行确认词。

    Returns
    -------
    bool
        去除首尾空白后的输入恰为 ``"EXECUTE"`` 时为 ``True``。

    Raises
    ------
    EOFError
        标准输入在读取前关闭时抛出。

    Notes
    -----
    本函数只读取标准输入且可能无限等待；它本身不访问 CAN，也不发送 L20
    位置命令。返回 ``True`` 只是 :func:`run` 动作分支的第二道门。
    """
    return input("Type EXECUTE to send a small robot command: ").strip() == "EXECUTE"


def run(
    args: argparse.Namespace,
    hand_factory: Callable[..., LinkerHandL20] = LinkerHandL20,
    confirm: Callable[[], bool] = confirm_execution,
    wait: Callable[[float], None] = time.sleep,
) -> int:
    """连接 L20、打印已验证反馈，并在双重确认后改变一个主动槽位。

    Parameters
    ----------
    args : argparse.Namespace
        ``build_parser`` 解析的参数；``joint_index`` 为零基主动位置索引，
        ``delta_raw`` 为整数 raw 增量，``speed_raw`` 为 G20 五指统一原始速度。
    hand_factory : callable, optional
        创建公开 :class:`LinkerHandL20` 接口的工厂，默认创建真实封装；可注入
        无硬件测试替身。
    confirm : callable, optional
        返回精确执行确认结果的函数。
    wait : callable, optional
        动作发送后等待稳定时间的函数；默认映射 :func:`time.sleep`，测试可注入
        无等待替身。

    Returns
    -------
    int
        正常或未确认时为 ``0``；仅断开失败时为 ``1``；候选 raw 位置不在
        ``[0, 255]`` 时为 ``2``；收到 ``KeyboardInterrupt`` 时为 ``130``。

    Notes
    -----
    默认路径仅连接并通过 SDK 3.1.1 的 G20 协议读取版本、20 槽位置、速度、
    最大扭矩、温度与故障，不发送位置命令。执行路径只在 ``--execute`` 与精确确认
    均成立后，先为五指设置统一的保守 raw 速度，再从当前反馈复制 20 槽 raw 位置并
    改变一个主动槽位；不使用张开、握拳等预设。发送后再次请求位置反馈供操作者比较，
    但固定 SDK 不提供响应 generation，回读不单独证明该次 CAN 命令已经执行。
    finally 始终尝试断开；断开失败报告到标准错误流且不会掩盖已有主异常。
    """
    hand = None
    exit_code = 0
    try:
        hand = hand_factory(hand_type=args.hand_type, can_channel=args.can_channel)
        hand.connect()
        print("L20 SDK version:", hand.get_sdk_version())
        print("L20 position (raw):", hand.get_joint_positions_raw())
        print("L20 speed:", hand.get_speed())
        print("L20 torque:", hand.get_torque())
        print("L20 temperature:", hand.get_temperature())
        print("L20 fault:", hand.get_fault())
        if not args.execute:
            print("Read-only mode: no L20 position command was sent.")
        elif not confirm():
            print("Execution was not confirmed: no L20 position command was sent.")
        else:
            target = hand.get_joint_positions_raw().copy()
            target[args.joint_index] += args.delta_raw
            if not 0 <= target[args.joint_index] <= 255:
                print("Candidate L20 raw target is outside [0, 255]: "
                      "no position command was sent.")
                exit_code = 2
            else:
                speed = np.full(5, args.speed_raw, dtype=np.int64)
                print("L20 command speed (raw):", speed)
                hand.set_speed(speed)
                print("L20 command target (raw):", target)
                hand.set_joint_positions_raw(target)
                wait(0.5)
                print("L20 position after command (raw):", hand.get_joint_positions_raw())
    except KeyboardInterrupt:
        print("Interrupted: no further L20 commands will be sent.")
        exit_code = 130
    finally:
        if hand is not None:
            cleanup_succeeded = _disconnect_or_report(hand, "L20")
            if not cleanup_succeeded and sys.exc_info()[0] is None and exit_code == 0:
                exit_code = 1
    return exit_code


def main() -> int:
    """解析进程参数并运行 L20 示例。

    Returns
    -------
    int
        :func:`run` 定义的进程退出码。

    Raises
    ------
    SystemExit
        ``argparse`` 处理 ``--help`` 或无效参数时抛出。
    ImportError
        默认真实 Wrapper 的 SDK 或依赖不可导入时抛出。
    RuntimeError
        L20 连接或反馈等 Wrapper 操作失败且未转换为退出码时抛出。

    Notes
    -----
    本函数进入 :func:`run` 后会连接真实 CAN，相关 SDK 操作可能阻塞。默认只读；
    只有 ``--execute`` 和交互式 ``EXECUTE`` 双门均通过后才会发送一次单槽位小
    动作。SDK 正常返回只表示调用结束，不保证每个底层 CAN 帧均已发送成功。
    """
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
