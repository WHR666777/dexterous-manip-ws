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
from robot_control import (  # noqa: E402
    L20_ACTIVE_POSITION_INDICES,
    LinkerHandL20,
    NeroArm,
    RobotSystem,
)


def build_parser() -> argparse.ArgumentParser:
    """构建只读默认值的命令行解析器。"""
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
    """要求操作者精确输入 EXECUTE 后才允许发送动作。"""
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
        正常或未确认时为 ``0``；候选 L20 raw 位置不在 ``[0, 255]`` 时为 ``2``。

    Notes
    -----
    默认路径只连接、self-check 和读取 observation，不使能 Nero，也不调用
    ``step``。执行路径在两个门均通过后，以当前反馈构造 action：Nero 保持当前
    七维 rad 目标，L20 只改变一个主动槽位并转换为 ``[-1, 1]`` normalized action。
    ``step`` 至多调用一次；finally 始终尝试断开，不自动 disable Nero，避免下落。
    """
    system = None
    try:
        system = system_factory()
        system.connect()
        print("Combined self-check:", system.self_check())
        observation = system.get_observation()
        print("Combined observation:", observation)
        if not args.execute:
            print("Read-only mode: no Nero enable or combined motion command was sent.")
            return 0
        if not confirm():
            print("Execution was not confirmed: no Nero enable or combined motion command was sent.")
            return 0

        system.enable()
        observation = system.get_observation()
        arm_target = np.asarray(observation["arm"]["joint_position"], dtype=np.float64).copy()
        hand_raw = np.asarray(
            observation["hand"]["joint_position_raw"], dtype=np.float64,
        ).copy()
        hand_raw[args.hand_joint_index] += args.hand_delta_raw
        if hand_raw[args.hand_joint_index] > 255:
            print("Candidate L20 raw target exceeds 255: no combined command was sent.")
            return 2
        hand_target = hand_raw / 255.0 * 2.0 - 1.0
        system.step({
            "arm_joint_position": arm_target,
            "hand_joint_position": hand_target,
        })
        for _ in range(args.observation_cycles):
            observation = system.get_observation()
            # Future integration point: action = policy(observation)
            print("Policy integration observation:", observation)
        return 0
    except KeyboardInterrupt:
        print("Interrupted: no further combined commands will be sent.")
        return 130
    finally:
        if system is not None:
            try:
                system.disconnect()
            except Exception as error:
                print("Combined system disconnect failed:", error)


def main() -> int:
    """运行示例并返回进程退出码。"""
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
