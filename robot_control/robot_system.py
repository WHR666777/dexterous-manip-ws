"""Nero 机械臂与 LinkerHand L20 的组合研究控制接口。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from .l20 import LinkerHandL20
from .nero import NeroArm


class RobotSystem:
    """组合 Nero 七轴机械臂和 LinkerHand L20 的控制边界。

    Parameters
    ----------
    arm : NeroArm or None, optional
        已构造的 Nero 封装；省略时创建默认 :class:`NeroArm`。
    hand : LinkerHandL20 or None, optional
        已构造的 L20 封装；省略时创建默认 :class:`LinkerHandL20`。

    Notes
    -----
    本类固定适用于 Nero ``V111`` 七自由度机械臂和 LinkerHand ``L20``，不
    推断其他固件、手型或自由度。设备对象可在无硬件单元测试中注入。构造本身
    不连接任一设备；调用方须先调用 :meth:`connect`，再按需调用 :meth:`enable`。
    """

    def __init__(
        self,
        arm: Optional[NeroArm] = None,
        hand: Optional[LinkerHandL20] = None,
    ) -> None:
        """创建组合控制器，不连接硬件。

        Parameters
        ----------
        arm : NeroArm or None, optional
            Nero 控制封装；为 ``None`` 时创建默认实例。
        hand : LinkerHandL20 or None, optional
            L20 控制封装；为 ``None`` 时创建默认实例。

        Returns
        -------
        None
            构造完成后返回。
        """
        self.arm = NeroArm() if arm is None else arm
        self.hand = LinkerHandL20() if hand is None else hand

    def connect(self) -> None:
        """依次连接 Nero 与 L20，并在后者失败时回滚 Nero。

        Returns
        -------
        None
            两个设备都已连接后返回。

        Raises
        ------
        RuntimeError
            Nero 或 L20 连接失败时抛出。L20 失败时此前已连接的 Nero 会先断开；若该回滚也
            失败，异常文本同时包含 L20 和 Nero 的失败信息，并以 L20 失败为
            异常链原因。

        Notes
        -----
        映射 :meth:`NeroArm.connect` 后 :meth:`LinkerHandL20.connect`，调用顺序
        固定为 Nero 后 L20；不会自动使能机械臂。
        """
        self.arm.connect()
        try:
            self.hand.connect()
        except Exception as hand_error:
            try:
                self.arm.disconnect()
            except Exception as rollback_error:
                raise RuntimeError(
                    "Failed to connect L20 after Nero connected: {0}; Nero "
                    "rollback failed: {1}".format(hand_error, rollback_error),
                ) from hand_error
            raise RuntimeError(
                "Failed to connect L20 after Nero connected: {0}".format(hand_error),
            ) from hand_error

    def enable(self, timeout: float = 5.0) -> None:
        """使能 Nero 机械臂，不向 L20 发送命令。

        Parameters
        ----------
        timeout : float, default=5.0
            传给 Nero 使能操作的最长等待时间，单位 s。

        Returns
        -------
        None
            Nero 已使能后返回。

        Raises
        ------
        ValueError
            ``timeout`` 不是 Nero 接受的有限非负秒数时抛出。
        RuntimeError
            Nero 未连接时抛出。
        TimeoutError
            Nero 未在 ``timeout`` 内完成使能时抛出。

        Notes
        -----
        直接映射 :meth:`NeroArm.enable`；L20 没有独立使能接口，因此不向其
        发送任何命令。
        """
        self.arm.enable(timeout=timeout)

    def disable(self, timeout: float = 5.0) -> None:
        """失能 Nero 机械臂，不向 L20 发送命令。

        Parameters
        ----------
        timeout : float, default=5.0
            传给 Nero 失能操作的最长等待时间，单位 s。

        Returns
        -------
        None
            Nero 已失能后返回。

        Raises
        ------
        ValueError
            ``timeout`` 不是 Nero 接受的有限非负秒数时抛出。
        RuntimeError
            Nero 未连接时抛出。
        TimeoutError
            Nero 未在 ``timeout`` 内完成失能时抛出。

        Notes
        -----
        直接映射 :meth:`NeroArm.disable`；L20 不接收命令。失能可能导致机械臂
        下落，调用者应先确保机械安全。
        """
        self.arm.disable(timeout=timeout)

    def disconnect(self) -> None:
        """按 L20、Nero 顺序尝试断开，并聚合所有失败信息。

        Returns
        -------
        None
            两个设备都已成功断开时返回。

        Raises
        ------
        RuntimeError
            任一设备断开失败时抛出；错误文本包含每个失败设备的原始信息。

        Notes
        -----
        映射 :meth:`LinkerHandL20.disconnect` 后 :meth:`NeroArm.disconnect`。
        即便前一个设备断开失败，也始终尝试另一个设备。子封装负责各自的
        幂等性，因此重复调用本方法安全。
        """
        errors = []
        for name, device in (("L20", self.hand), ("Nero", self.arm)):
            try:
                device.disconnect()
            except Exception as exc:
                errors.append("{0} disconnect failed: {1}".format(name, exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    @staticmethod
    def _feedback_array(
        value: Any,
        shape: tuple[int, ...],
        label: str,
        *,
        integer: bool = False,
    ) -> np.ndarray:
        """将子设备反馈转换为受限的独立数组。"""
        try:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError
            if integer and (
                np.any(array < 0.0)
                or np.any(array > 255.0)
                or np.any(array != np.floor(array))
            ):
                raise ValueError
            if integer:
                return array.astype(np.int64, copy=True)
            return array.copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("{0} feedback is malformed.".format(label)) from exc

    @staticmethod
    def _feedback_mapping(value: Any, label: str) -> Mapping[str, Any]:
        """验证子设备返回的字典式反馈。"""
        if not isinstance(value, Mapping):
            raise RuntimeError("{0} feedback is malformed.".format(label))
        return value

    @staticmethod
    def _plain_value(value: Any, label: str) -> Any:
        """递归转换诊断反馈，确保其只含普通 Python 值。"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [RobotSystem._plain_value(item, label) for item in value.tolist()]
        if isinstance(value, Mapping):
            try:
                return {
                    str(key): RobotSystem._plain_value(item, label)
                    for key, item in value.items()
                }
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("{0} feedback is malformed.".format(label)) from exc
        if isinstance(value, (list, tuple)):
            return [RobotSystem._plain_value(item, label) for item in value]
        raise RuntimeError("{0} feedback is malformed.".format(label))

    @staticmethod
    def _action_keys_error() -> ValueError:
        """创建不依赖调用方键类型排序的动作键错误。"""
        return ValueError(
            "Action keys must be exactly ['arm_joint_position', "
            "'hand_joint_position'].",
        )

    def get_observation(self) -> dict[str, Any]:
        """读取 Nero 与 L20 的规范嵌套观测。

        Returns
        -------
        dict[str, object]
            包含 ``arm``、``hand`` 与唯一顶层 ``timestamp``。``arm`` 含
            ``joint_position`` ``(7,)``、``joint_torque`` ``(7,)``、
            ``tcp_pose`` ``(6,)``；``hand`` 含新鲜的 ``joint_position_raw``
            ``(20,)``。时间戳为两个设备读取完成后的 Unix wall-clock seconds。

        Raises
        ------
        RuntimeError
            任一子设备观测不可读取、不是规范映射，或其反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 :meth:`NeroArm.get_observation` 和
        :meth:`LinkerHandL20.get_joint_positions_raw`。丢弃 Nero 子观测自身的
        时间戳，避免多个不一致的完成时间；L20 始终以 ``fresh=True`` 请求位置，
        不读取缓存。
        """
        try:
            arm_observation = self._feedback_mapping(
                self.arm.get_observation(), "Nero observation",
            )
            arm = {
                "joint_position": self._feedback_array(
                    arm_observation["joint_position"], (7,), "Nero observation",
                ),
                "joint_torque": self._feedback_array(
                    arm_observation["joint_torque"], (7,), "Nero observation",
                ),
                "tcp_pose": self._feedback_array(
                    arm_observation["tcp_pose"], (6,), "Nero observation",
                ),
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Nero observation feedback is malformed.") from exc

        try:
            hand = {
                "joint_position_raw": self._feedback_array(
                    self.hand.get_joint_positions_raw(fresh=True),
                    (20,), "L20 position", integer=True,
                ),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("L20 position feedback is malformed.") from exc
        return {"arm": arm, "hand": hand, "timestamp": float(time.time())}

    def step(self, action: Mapping[str, Any]) -> None:
        """完整预验证后依次发送 Nero 与 L20 的位置目标。

        Parameters
        ----------
        action : mapping
            键必须恰为 ``arm_joint_position`` 与 ``hand_joint_position``。
            前者为有限 ``(7,)`` rad Nero 目标，后者为有限 ``(20,)`` 且位于
            ``[-1, 1]`` 的 L20 归一化目标。

        Returns
        -------
        None
            两个命令都发送后立即返回，不等待运动完成。

        Raises
        ------
        ValueError
            ``action`` 不是映射、键不完整，或任一动作在发送前校验失败时抛出。
        RuntimeError
            Nero 验证所需反馈不可用，或手部发送失败时抛出；后者明确说明 Nero
            命令可能已经发送。

        Notes
        -----
        映射 :meth:`NeroArm.validate_joint_command`、
        :meth:`NeroArm.command_joint_positions` 和
        :meth:`LinkerHandL20.set_joint_positions_normalized`。两个动作在任何设备
        发送前均完成验证。发送顺序固定为 Nero 后 L20；若 Nero 发送失败，不会
        尝试手部发送。
        """
        if not isinstance(action, Mapping):
            raise ValueError("Action must be a mapping.")
        expected_keys = {"arm_joint_position", "hand_joint_position"}
        try:
            action_keys = set(action.keys())
        except Exception as exc:
            raise self._action_keys_error() from exc
        if action_keys != expected_keys:
            raise self._action_keys_error()
        try:
            arm_action = action["arm_joint_position"]
            hand_action = action["hand_joint_position"]
        except Exception as exc:
            raise ValueError("Action values are unavailable.") from exc
        try:
            arm_value = self.arm.validate_joint_command(arm_action)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("arm_joint_position is invalid.") from exc
        try:
            hand_value = np.asarray(hand_action, dtype=np.float64)
            if hand_value.shape != (20,) or not np.all(np.isfinite(hand_value)):
                raise ValueError
            if np.any(hand_value < -1.0) or np.any(hand_value > 1.0):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "hand_joint_position must be a finite array with shape (20,) "
                "and values in [-1, 1].",
            ) from exc
        self.arm.command_joint_positions(arm_value)
        try:
            self.hand.set_joint_positions_normalized(hand_value)
        except Exception as exc:
            raise RuntimeError(
                "L20 command failed; Nero command may already have been sent: "
                "{0}".format(exc),
            ) from exc

    def self_check(self) -> dict[str, Any]:
        """只读地汇总 Nero 与 L20 的连接、状态和故障诊断。

        Returns
        -------
        dict[str, object]
            仅含普通 Python 值的 ``nero`` 与 ``l20`` 诊断。Nero 项含固定
            ``firmware_config``、SDK 报告的 ``firmware_reported`` 与机械臂状态；
            L20 项含新鲜 ``joint_position_raw`` 的普通 Python 列表（位置读取
            失败时为 ``None``）、``position_ok`` 以及五个电机的 ``fault`` 列表。

        Raises
        ------
        RuntimeError
            子设备状态、固件、机械臂状态或故障反馈不可读取或格式不合法时抛出。

        Notes
        -----
        映射 :meth:`NeroArm.is_connected`、:meth:`NeroArm.is_enabled`、
        :meth:`NeroArm.is_ok`、:meth:`NeroArm.get_firmware`、
        :meth:`NeroArm.get_arm_status`、:meth:`LinkerHandL20.is_connected`、
        :meth:`LinkerHandL20.get_joint_positions_raw` 与
        :meth:`LinkerHandL20.get_fault`。本方法只调用状态与反馈读取接口，绝不
        发送运动、预设或清故障命令。每个已检查类别都会打印一行 ``[OK]`` 或
        ``[FAIL]`` 摘要。L20 位置读取失败会记录 ``position_ok=False`` 并打印
        ``[FAIL]``，不会发送运动命令。
        """
        position = None
        position_ok = True
        try:
            position = self._feedback_array(
                self.hand.get_joint_positions_raw(fresh=True),
                (20,), "L20 position", integer=True,
            ).tolist()
        except Exception:
            position_ok = False

        try:
            firmware = self._feedback_mapping(
                self.arm.get_firmware(), "Nero firmware",
            )
            status = self._feedback_mapping(
                self.arm.get_arm_status(), "Nero arm status",
            )
            fault = self._feedback_array(
                self.hand.get_fault(), (5,), "L20 fault", integer=True,
            ).tolist()
            nero = {
                "connected": bool(self.arm.is_connected()),
                "enabled": bool(self.arm.is_enabled()),
                "ok": bool(self.arm.is_ok()),
                "firmware_config": "V111",
                "firmware_reported": self._plain_value(
                    firmware.get("software_version"), "Nero firmware",
                ),
                "arm_status": self._plain_value(status, "Nero arm status"),
            }
            l20 = {
                "connected": bool(self.hand.is_connected()),
                "joint_position_raw": position,
                "position_ok": position_ok,
                "fault": fault,
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Robot self-check feedback is malformed.") from exc

        checks = (
            ("Nero connection", nero["connected"]),
            ("Nero enabled", nero["enabled"]),
            ("Nero health", nero["ok"]),
            ("Nero firmware", nero["firmware_reported"] == "1.11"),
            ("Nero arm status", nero["arm_status"].get("arm_status") == 0),
            ("L20 connection", l20["connected"]),
            ("L20 position", l20["position_ok"]),
            ("L20 fault", not any(l20["fault"])),
        )
        for label, passed in checks:
            print("[{0}] {1}".format("OK" if passed else "FAIL", label))
        return {"nero": nero, "l20": l20}
