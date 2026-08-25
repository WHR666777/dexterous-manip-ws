"""Nero 与 LinkerHand L20 的默认只读联合示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CONTROL_HZ,
    L20_CAN_CHANNEL,
    L20_HAND_TYPE,
    NERO_CAN_CHANNEL,
    NERO_CAN_INTERFACE,
    NERO_MAX_JOINT_DELTA,
)
from examples._safety import _disconnect_or_report  # noqa: E402
from robot_control import (  # noqa: E402
    L20_ACTIVE_POSITION_INDICES,
    LinkerHandL20,
    NeroArm,
    RobotSystem,
)


def build_parser() -> argparse.ArgumentParser:
    """构建默认只读的 Nero + L20 命令行解析器。

    Returns
    -------
    argparse.ArgumentParser
        配置好执行门、L20 主动槽位、raw 增量和有限观测次数的解析器。

    Notes
    -----
    仅创建内存中的解析器，不构造组合 Wrapper、不访问 CAN，也不发送硬件命令。
    """
    parser = argparse.ArgumentParser(description="默认只读的 Nero + L20 联合示例。")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="请求进入二次确认后的单次组合动作分支。",
    )
    parser.add_argument(
        "--hand-joint-index",
        type=int,
        choices=L20_ACTIVE_POSITION_INDICES,
        default=L20_ACTIVE_POSITION_INDICES[0],
        help="零基 L20 主动位置槽位；保留槽位不可选。",
    )
    parser.add_argument(
        "--hand-delta-raw",
        type=int,
        choices=(1, 2, 5, 10),
        default=1,
        help="相对当前 L20 raw 反馈增加的单槽位保守增量。",
    )
    parser.add_argument(
        "--observation-cycles",
        type=int,
        choices=range(1, CONTROL_HZ + 1),
        default=CONTROL_HZ,
        help="单次动作后只读 policy 接入循环的有限观测次数。",
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
    本函数只读取标准输入且可能无限等待；它本身不访问 CAN、不使能 Nero，也不
    发送组合动作。返回 ``True`` 只是 :func:`run` 动作分支的第二道门。
    """
    return input("Type EXECUTE to send a small robot command: ").strip() == "EXECUTE"


def _build_system(
    *,
    arm_factory: Callable[..., NeroArm] = NeroArm,
    hand_factory: Callable[..., LinkerHandL20] = LinkerHandL20,
) -> RobotSystem:
    """以共享 CAN 配置创建尚未连接的组合 Wrapper。"""
    arm = arm_factory(
        can_interface=NERO_CAN_INTERFACE,
        can_channel=NERO_CAN_CHANNEL,
        max_joint_delta=NERO_MAX_JOINT_DELTA,
    )
    hand = hand_factory(hand_type=L20_HAND_TYPE, can_channel=L20_CAN_CHANNEL)
    return RobotSystem(arm=arm, hand=hand)


def run(
    args: argparse.Namespace,
    system_factory: Callable[..., RobotSystem] = _build_system,
    confirm: Callable[[], bool] = confirm_execution,
) -> int:
    """连接组合 Wrapper，并在双重确认后发送至多一次小的组合动作。

    Parameters
    ----------
    args : argparse.Namespace
        ``build_parser`` 解析的参数。L20 索引为零基主动槽位，raw 增量为整数。
    system_factory : callable, optional
        创建尚未连接 :class:`RobotSystem` 的工厂；默认将共享配置传给两个公开
        子 Wrapper，可注入无硬件测试替身。
    confirm : callable, optional
        返回精确执行确认结果的函数。

    Returns
    -------
    int
        正常或未确认时为 ``0``；仅断开失败时为 ``1``；候选 L20 raw 位置不在
        ``[0, 255]`` 时为 ``2``；收到 ``KeyboardInterrupt`` 时为 ``130``。

    Notes
    -----
    默认路径只连接、self-check 和读取 observation，不使能 Nero，也不调用
    ``step``。执行路径在两个门均通过后，以当前反馈构造 action：Nero 保持当前
    七维 rad 目标，L20 只改变一个主动槽位并转换为 ``[-1, 1]`` normalized action。
    ``step`` 至多调用一次；finally 始终尝试断开，不自动 disable Nero，避免下落。
    断开失败报告到标准错误流且不会掩盖已有主异常。
    """
    system = None
    exit_code = 0
    try:
        system = system_factory()
        system.connect()
        print("Combined self-check:", system.self_check())
        observation = system.get_observation()
        print("Combined observation:", observation)
        if not args.execute:
            print("Read-only mode: no Nero enable or combined motion command was sent.")
        elif not confirm():
            print("Execution was not confirmed: no Nero enable or combined motion command was sent.")
        else:
            system.enable()
            observation = system.get_observation()
            arm_target = np.asarray(observation["arm"]["joint_position"], dtype=np.float64).copy()
            hand_raw = np.asarray(
                observation["hand"]["joint_position_raw"], dtype=np.float64,
            ).copy()
            hand_raw[args.hand_joint_index] += args.hand_delta_raw
            if hand_raw[args.hand_joint_index] > 255:
                print("Candidate L20 raw target exceeds 255: no combined command was sent.")
                exit_code = 2
            else:
                hand_target = hand_raw / 255.0 * 2.0 - 1.0
                system.step({
                    "arm_joint_position": arm_target,
                    "hand_joint_position": hand_target,
                })
                for _ in range(args.observation_cycles):
                    observation = system.get_observation()
                    # Future integration point: action = policy(observation)
                    print("Policy integration observation:", observation)
    except KeyboardInterrupt:
        print("Interrupted: no further combined commands will be sent.")
        exit_code = 130
    finally:
        if system is not None:
            cleanup_succeeded = _disconnect_or_report(system, "Combined system")
            if not cleanup_succeeded and sys.exc_info()[0] is None and exit_code == 0:
                exit_code = 1
    return exit_code


def main() -> int:
    """解析进程参数并运行 Nero + L20 联合示例。

    Returns
    -------
    int
        :func:`run` 定义的进程退出码。

    Raises
    ------
    SystemExit
        ``argparse`` 处理 ``--help`` 或无效参数时抛出。
    ImportError
        任一默认真实 Wrapper 的 SDK 或依赖不可导入时抛出。
    RuntimeError
        连接、诊断、观测或动作等 Wrapper 操作失败且未转换为退出码时抛出。

    Notes
    -----
    本函数进入 :func:`run` 后会连接两条真实 CAN，SDK 反馈与收尾操作可能阻塞。
    默认只读且 Nero 保持失能；只有 ``--execute`` 和交互式 ``EXECUTE`` 双门均
    通过后才会使能 Nero 并调用一次组合动作。L20 正常返回不保证每个底层 CAN
    帧均已发送成功，操作者仍须负责机械工作区、承载和急停安全。
    """
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
