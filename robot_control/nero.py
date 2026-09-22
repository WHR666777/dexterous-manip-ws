"""Nero v1.11 官方 SDK 的轻量研究控制封装。"""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Callable, List, Optional

import numpy as np


NERO_DOF = 7


def _load_nero_sdk():
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    return type("NeroSdk", (), {
        "AgxArmFactory": AgxArmFactory,
        "ArmModel": ArmModel,
        "NeroFW": NeroFW,
        "create_agx_arm_config": staticmethod(create_agx_arm_config),
    })


class NeroArm:
    """以固定 v1.11 Driver 控制七自由度 AgileX Nero 机械臂。

    该封装仅映射已核查的官方 ``pyAgxArm`` v1.11 接口，不提供逆
    运动学、底层 MIT/CPV 控制或未经验证的关节速度；支持 JS 关节跟随。
    通过 ``driver``
    注入的对象仅用于测试隔离，不改变真实 SDK 的调用语义。
    """

    def __init__(
        self,
        can_interface: str = "socketcan",
        can_channel: str = "can0",
        max_joint_delta: Optional[float] = None,
        driver: Optional[Any] = None,
    ) -> None:
        """创建 Nero v1.11 封装并启用官方软件关节限位。

        Parameters
        ----------
        can_interface : str, default="socketcan"
            传给官方 Driver 配置的 CAN 接口名称。
        can_channel : str, default="can0"
            传给官方 Driver 配置的 CAN 通道名称。
        max_joint_delta : float or None, optional
            额外单次关节目标变化上限，单位 rad；``None`` 不启用。
        driver : object or None, optional
            已创建的 v1.11 Driver。此注入缝仅供无硬件单元测试；为
            ``None`` 时使用 ``NeroFW.V111`` 创建官方 Driver。

        Raises
        ------
        ValueError
            ``max_joint_delta`` 不是有限非负数时抛出。
        RuntimeError
            SDK 配置未提供完整七轴官方限位时抛出。

        Notes
        -----
        映射 ``create_agx_arm_config(..., firmeware_version=NeroFW.V111)``
        与 ``set_joint_limits_enabled(True)``；不会连接硬件或阻塞。
        """
        if driver is None:
            sdk = _load_nero_sdk()
            config = sdk.create_agx_arm_config(
                robot=sdk.ArmModel.NERO,
                firmeware_version=sdk.NeroFW.V111,
                interface=can_interface,
                channel=can_channel,
            )
            driver = sdk.AgxArmFactory.create_arm(config)
        self._driver = driver
        self._max_joint_delta = self._validate_delta_setting(max_joint_delta)
        self._driver.set_joint_limits_enabled(True)
        config = self._driver.get_config()
        limits = config.get("joint_limits", {})
        names = [f"joint{i}" for i in range(1, NERO_DOF + 1)]
        if any(name not in limits for name in names):
            raise RuntimeError("Nero SDK config does not contain all 7 joint limits.")
        self._joint_limits = np.asarray(
            [limits[name] for name in names], dtype=np.float64,
        )

    @staticmethod
    def _validate_delta_setting(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise ValueError("max_joint_delta must be a finite nonnegative value.")
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_joint_delta must be a finite nonnegative value.") from exc
        if not np.isfinite(normalized) or normalized < 0:
            raise ValueError("max_joint_delta must be a finite nonnegative value.")
        return normalized

    def connect(self) -> None:
        """连接 Nero CAN Driver；重复调用不会重复连接。

        Returns
        -------
        None
            连接完成后返回。

        Notes
        -----
        映射官方 ``Driver.connect()``，可能阻塞于 CAN 初始化；不自动使能。
        """
        if not self.is_connected():
            self._driver.connect()

    def disconnect(self) -> None:
        """断开 Nero CAN Driver；不自动失能以避免机械臂下落。

        Returns
        -------
        None
            断开完成后返回。

        Notes
        -----
        始终映射官方幂等 ``Driver.disconnect()``，可能阻塞于线程收尾；即使
        ``is_connected()`` 已为假也让 Driver 清理其内部资源。
        """
        self._driver.disconnect()

    def is_connected(self) -> bool:
        """返回官方 Driver 报告的 CAN 连接状态。

        Returns
        -------
        bool
            ``True`` 表示 Driver 已连接。

        Notes
        -----
        映射 ``Driver.is_connected()``；不阻塞且不访问机械臂状态。
        """
        return bool(self._driver.is_connected())

    def is_enabled(self) -> bool:
        """返回七个 Nero 关节是否均已使能。

        Returns
        -------
        bool
            七轴均报告 enabled 时为 ``True``。

        Notes
        -----
        映射七次 ``Driver.get_joint_enable_status(1..7)``；不阻塞。
        """
        return all(
            bool(self._driver.get_joint_enable_status(index))
            for index in range(1, NERO_DOF + 1)
        )

    def is_ok(self) -> bool:
        """返回官方 Driver 的健康状态。

        Returns
        -------
        bool
            官方 ``is_ok`` 的布尔结果。

        Notes
        -----
        映射 ``Driver.is_ok()``；不阻塞，不制造额外健康结论。
        """
        return bool(self._driver.is_ok())

    def _require_connected(self) -> None:
        if not self.is_connected():
            raise RuntimeError("Nero arm is not connected.")

    def _require_enabled(self) -> None:
        if not self.is_enabled():
            raise RuntimeError("Nero arm is not enabled.")

    def enable_normal_mode(self) -> None:
        while not self._driver.enable():
            self._driver.set_normal_mode()
            time.sleep(0.01)
    
    def enable_leader_mode(self) -> None:
        while not self._driver.enable():
            self._driver.set_leader_mode()
            time.sleep(0.01)

    @staticmethod
    def _validate_retry_settings(timeout: float, poll_interval: float) -> tuple[float, float]:
        valid_number_types = (int, float, np.integer, np.floating)
        if isinstance(timeout, bool) or not isinstance(timeout, valid_number_types):
            raise ValueError("timeout must be nonnegative.")
        if isinstance(poll_interval, bool) or not isinstance(poll_interval, valid_number_types):
            raise ValueError("poll_interval must be positive.")
        try:
            timeout_value = float(timeout)
        except OverflowError as exc:
            raise ValueError("timeout must be nonnegative.") from exc
        try:
            poll_interval_value = float(poll_interval)
        except OverflowError as exc:
            raise ValueError("poll_interval must be positive.") from exc
        if not np.isfinite(timeout_value) or timeout_value < 0:
            raise ValueError("timeout must be nonnegative.")
        if not np.isfinite(poll_interval_value) or poll_interval_value <= 0:
            raise ValueError("poll_interval must be positive.")
        return timeout_value, poll_interval_value

    def _retry_until_true(
        self,
        operation: Callable[[], Any],
        *,
        timeout: float,
        poll_interval: float,
        message: str,
    ) -> None:
        timeout_value, poll_interval_value = self._validate_retry_settings(
            timeout, poll_interval,
        )
        deadline = time.monotonic() + timeout_value
        while True:
            if bool(operation()):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            time.sleep(poll_interval_value)

    def enable(self, timeout: float = 5.0, poll_interval: float = 0.05) -> None:
        """在有限时间内重试使能 Nero 全部七轴。

        Parameters
        ----------
        timeout : float, default=5.0
            最大等待时间，单位 s，必须为有限非负数。
        poll_interval : float, default=0.05
            两次官方调用之间的等待时间，单位 s，必须为有限正数。

        Raises
        ------
        RuntimeError
            未连接时抛出。
        ValueError
            时间参数非法时抛出。
        TimeoutError
            在 ``timeout`` 内未全部使能时抛出。

        Notes
        -----
        重复映射不带 timeout 的 v1.11 ``Driver.enable()``；可能阻塞至超时。
        """
        self._require_connected()
        self._retry_until_true(
            self._driver.enable,
            timeout=timeout,
            poll_interval=poll_interval,
            message="Nero enable timed out.",
        )

    def disable(self, timeout: float = 5.0, poll_interval: float = 0.05) -> None:
        """在有限时间内重试失能 Nero 全部七轴。

        Parameters
        ----------
        timeout : float, default=5.0
            最大等待时间，单位 s，必须为有限非负数。
        poll_interval : float, default=0.05
            两次官方调用之间的等待时间，单位 s，必须为有限正数。

        Raises
        ------
        RuntimeError
            未连接时抛出。
        ValueError
            时间参数非法时抛出。
        TimeoutError
            在 ``timeout`` 内未失能时抛出。

        Notes
        -----
        重复映射不带 timeout 的 v1.11 ``Driver.disable()``；可能阻塞至超时，
        且失能可能造成机械臂下落。
        """
        self._require_connected()
        self._retry_until_true(
            self._driver.disable,
            timeout=timeout,
            poll_interval=poll_interval,
            message="Nero disable timed out.",
        )

    def command_joint_positions(self, joints: Any) -> None:
        """非阻塞地发送七轴关节位置目标。

        Parameters
        ----------
        joints : array-like, shape (7,)
            七轴目标角度，单位 rad。完整限位校验由 ``validate_joint_command``
            提供。

        Returns
        -------
        None
            目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接、七轴未全部使能，或启用变化量限制时反馈不可用时抛出。
        ValueError
            目标的形状、数值、官方限位或变化量不合法时抛出。

        Notes
        -----
        最终映射 v1.11 ``Driver.move_j()``，不会等待运动完成。
        """
        self._require_connected()
        self._require_enabled()
        self._driver.move_j(self.validate_joint_command(joints).tolist())

    @staticmethod
    def _array_from_input(values: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
        if values is None:
            raise ValueError("Values must not be None.")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Values must be numeric.") from exc
        if array.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Values must all be finite.")
        return array.copy()

    @staticmethod
    def _array_from_feedback(values: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
        try:
            return NeroArm._array_from_input(values, expected_shape)
        except ValueError as exc:
            raise RuntimeError("Nero SDK feedback is unavailable or malformed.") from exc

    @staticmethod
    def _message_or_raise(message: Any) -> Any:
        if message is None or getattr(message, "msg", None) is None:
            raise RuntimeError("Nero SDK feedback is unavailable: " + str(message))
        return message

    def get_raw_joint_positions(self) -> Any:
        """返回官方原始关节角度消息，供底层调试使用。

        Returns
        -------
        object
            SDK ``get_joint_angles()`` 返回的聚合消息；其 ``msg`` 为七轴 rad。

        Raises
        ------
        RuntimeError
            未连接或反馈尚不可用时抛出。

        Notes
        -----
        直接映射 ``Driver.get_joint_angles()``；不复制消息且不阻塞等待反馈。固定
        SDK 会从零初始化聚合缓存，并在收到任一组成帧后返回，因此本调试接口
        可能暴露混合了缺失零值或旧值的部分聚合；规范位置读取不会使用它。
        """
        self._require_connected()
        return self._message_or_raise(self._driver.get_joint_angles())

    def get_raw_motor_states(self) -> list[Any]:
        """返回七个官方原始电机状态消息，供底层调试使用。

        Returns
        -------
        list of object, length 7
            索引 0--6 分别对应关节 1--7 的 SDK 消息。

        Raises
        ------
        RuntimeError
            未连接或任一关节反馈尚不可用时抛出。

        Notes
        -----
        映射七次 ``Driver.get_motor_states(1..7)``；不生成速度或其他反馈。
        """
        self._require_connected()
        return [
            self._message_or_raise(self._driver.get_motor_states(index))
            for index in range(1, NERO_DOF + 1)
        ]

    def get_raw_flange_pose(self) -> Any:
        """返回官方原始法兰位姿消息，供底层调试使用。

        Returns
        -------
        object
            SDK 消息，其 ``msg`` 顺序为 ``[x, y, z, roll, pitch, yaw]``，
            单位为 ``[m, m, m, rad, rad, rad]``。

        Raises
        ------
        RuntimeError
            未连接或反馈尚不可用时抛出。

        Notes
        -----
        真实固定 v1.11 Driver 路径先要求三个 parser pose 组成帧全部存在，再映射
        ``Driver.get_flange_pose()``；不复制消息也不阻塞。无该私有 parser 形状的
        注入 fake 保持支持。
        """
        self._require_connected()
        self._require_complete_v111_pose_frames()
        return self._message_or_raise(self._driver.get_flange_pose())

    def _require_complete_v111_pose_frames(self) -> None:
        """拒绝固定 v1.11 Driver 的部分三帧位姿聚合。

        此窄适配器绑定 pyAgxArm commit
        ``8cd90f9106219a156c3c0d7e58ee36d838a89baf``。仅当注入对象暴露固定
        Driver 的完整 ``_parser`` 三字段形状时执行；不具备该私有形状的无硬件
        fake 保持兼容。
        """
        parser = getattr(self._driver, "_parser", None)
        frame_names = ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
        if parser is None or not all(hasattr(parser, name) for name in frame_names):
            return
        if any(getattr(parser, name) is None for name in frame_names):
            raise RuntimeError("Nero SDK pose feedback is incomplete.")

    def get_raw_arm_status(self) -> Any:
        """返回官方原始机械臂状态消息，供底层调试使用。

        Returns
        -------
        object
            SDK ``get_arm_status()`` 返回的消息。

        Raises
        ------
        RuntimeError
            未连接或反馈尚不可用时抛出。

        Notes
        -----
        直接映射 ``Driver.get_arm_status()``；不复制消息也不阻塞。
        """
        self._require_connected()
        return self._message_or_raise(self._driver.get_arm_status())

    def get_joint_positions(self) -> np.ndarray:
        """读取七轴关节位置。

        Returns
        -------
        numpy.ndarray, shape (7,), dtype float64
            关节 1--7 的位置，单位 rad；返回独立副本。

        Raises
        ------
        RuntimeError
            未连接、官方反馈不可用，或反馈形状/数值不合法时抛出。

        Notes
        -----
        由七次 ``Driver.get_motor_states(1..7).msg.position`` 组成；仅当七个
        组成消息均存在且位置均为有限标量时返回，不使用可能部分填充的官方
        ``get_joint_angles()`` 聚合缓存。
        """
        messages = self.get_raw_motor_states()
        try:
            positions = [message.msg.position for message in messages]
        except AttributeError as exc:
            raise RuntimeError("Nero SDK feedback is unavailable.") from exc
        return self._array_from_feedback(positions, (NERO_DOF,))

    def get_joint_torques(self) -> np.ndarray:
        """读取七轴电机扭矩反馈。

        Returns
        -------
        numpy.ndarray, shape (7,), dtype float64
            关节 1--7 的扭矩，单位 N*m；返回独立副本。

        Raises
        ------
        RuntimeError
            未连接、任一反馈不可用，或扭矩反馈形状/数值不合法时抛出。

        Notes
        -----
        由七次 ``Driver.get_motor_states(i).msg.torque`` 转换；不提供速度。
        """
        messages = self.get_raw_motor_states()
        try:
            torques = [message.msg.torque for message in messages]
        except AttributeError as exc:
            raise RuntimeError("Nero SDK feedback is unavailable.") from exc
        return self._array_from_feedback(torques, (NERO_DOF,))

    def get_flange_pose(self) -> np.ndarray:
        """读取法兰在世界坐标中的六维位姿。

        Returns
        -------
        numpy.ndarray, shape (6,), dtype float64
            ``[x, y, z, roll, pitch, yaw]``，单位 ``[m, m, m, rad, rad, rad]``；
            返回独立副本。

        Raises
        ------
        RuntimeError
            未连接、反馈不可用，或反馈形状/数值不合法时抛出。

        Notes
        -----
        由 ``Driver.get_flange_pose().msg`` 转换；真实固定 v1.11 Driver 必须先有
        全部三个 parser pose 组成帧，不从部分聚合返回状态。
        """
        return self._array_from_feedback(self.get_raw_flange_pose().msg, (6,))

    def get_tcp_pose(self) -> np.ndarray:
        """读取 TCP 在世界坐标中的六维位姿。

        Returns
        -------
        numpy.ndarray, shape (6,), dtype float64
            ``[x, y, z, roll, pitch, yaw]``，单位 ``[m, m, m, rad, rad, rad]``；
            返回独立副本。

        Raises
        ------
        RuntimeError
            未连接、反馈不可用，或反馈形状/数值不合法时抛出。

        Notes
        -----
        由 ``Driver.get_tcp_pose().msg`` 转换；真实固定 v1.11 Driver 必须先有
        全部三个 parser pose 组成帧。不阻塞，不推算替代位姿。
        """
        self._require_connected()
        self._require_complete_v111_pose_frames()
        message = self._message_or_raise(self._driver.get_tcp_pose())
        return self._array_from_feedback(message.msg, (6,))

    @staticmethod
    def _to_plain_data(value: Any) -> Any:
        if isinstance(value, Enum):
            return NeroArm._to_plain_data(value.value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: NeroArm._to_plain_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [NeroArm._to_plain_data(item) for item in value]
        fields = getattr(value, "_fields_", None)
        if fields is not None:
            try:
                return {
                    field: NeroArm._to_plain_data(getattr(value, field))
                    for field in fields
                    if isinstance(field, str) and not field.startswith("_")
                }
            except AttributeError as exc:
                raise RuntimeError("Nero SDK feedback cannot be serialized.") from exc
        try:
            attributes = vars(value)
        except TypeError as exc:
            raise RuntimeError("Nero SDK feedback cannot be serialized.") from exc
        return {
            key: NeroArm._to_plain_data(item)
            for key, item in attributes.items()
            if not key.startswith("_")
        }

    def get_firmware(self) -> dict[str, Any]:
        """读取 SDK 报告的 Nero 固件信息。

        Returns
        -------
        dict[str, object]
            由 SDK 固件反馈深复制得到的普通 Python 数据，例如
            ``{"software_version": "1.11"}``。

        Raises
        ------
        RuntimeError
            未连接、反馈不可用或返回值无法表示为字典时抛出。

        Notes
        -----
        映射 ``Driver.get_firmware()``，官方请求/响应调用可能阻塞；不会根据报告
        结果切换固定的 v1.11 实现。
        """
        self._require_connected()
        firmware = self._driver.get_firmware()
        if firmware is None:
            raise RuntimeError("Nero SDK firmware feedback is unavailable.")
        result = self._to_plain_data(firmware)
        if not isinstance(result, dict):
            raise RuntimeError("Nero SDK firmware feedback must be a mapping.")
        return result

    def get_arm_status(self) -> dict[str, Any]:
        """读取并序列化官方机械臂状态。

        Returns
        -------
        dict[str, object]
            包含 SDK 实际公开字段，如 ``ctrl_mode``、``arm_status``、
            ``mode_feedback``、``teach_status``、``motion_status``、
            ``trajectory_num`` 与嵌套 ``err_status`` 的新字典。

        Raises
        ------
        RuntimeError
            未连接、反馈不可用或状态无法序列化为字典时抛出。

        Notes
        -----
        由 ``Driver.get_arm_status().msg`` 深复制；不泄露 SDK 消息且不阻塞。
        """
        result = self._to_plain_data(self.get_raw_arm_status().msg)
        if not isinstance(result, dict):
            raise RuntimeError("Nero SDK arm status must be an object.")
        return result

    def get_observation(self) -> dict[str, Any]:
        """读取用于研究策略的最小 Nero 观测字典。

        Returns
        -------
        dict[str, object]
            ``joint_position`` 为 ``(7,)`` rad，``joint_torque`` 为 ``(7,)`` N*m，
            ``tcp_pose`` 为 ``(6,)`` ``[m, m, m, rad, rad, rad]``，``timestamp``
            为本次读取完成时的 Unix wall-clock seconds。

        Raises
        ------
        RuntimeError
            未连接、任一底层反馈不可用，或反馈形状/数值不合法时抛出。

        Notes
        -----
        顺序读取官方反馈，不提供 v1.11 不可信的 ``joint_velocity``；可能短暂阻塞。
        """
        observation = {
            "joint_position": self.get_joint_positions(),
            "joint_torque": self.get_joint_torques(),
            "tcp_pose": self.get_tcp_pose(),
        }
        observation["timestamp"] = float(time.time())
        return observation

    def get_state_vector(self) -> np.ndarray:
        """读取仅含关节位置的 Nero 状态向量。

        Returns
        -------
        numpy.ndarray, shape (7,), dtype float64
            关节 1--7 的位置，单位 rad；不包含虚构的速度。

        Raises
        ------
        RuntimeError
            未连接、反馈不可用，或反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 ``get_joint_positions()``；可能短暂阻塞于反馈读取。
        """
        return self.get_joint_positions()

    def validate_joint_command(
        self, joints: Any, *, max_joint_delta: Optional[float] = None,
    ) -> np.ndarray:
        """验证七轴 rad 目标与官方限位及可选单步变化限制。

        Parameters
        ----------
        joints : array-like, shape (7,)
            关节 1--7 目标位置，单位 rad，必须有限。
        max_joint_delta : float or None, optional
            本调用的有限非负变化上限，单位 rad。传入 ``None`` 时使用构造时
            配置；若构造时同为 ``None`` 则不检查变化量。

        Returns
        -------
        numpy.ndarray, shape (7,), dtype float64
            已验证的独立目标副本，单位 rad。

        Raises
        ------
        ValueError
            形状、有限性、官方限位或变化量限制不满足时抛出。
        RuntimeError
            需要变化量检查但当前 SDK 关节反馈不可用时抛出。

        Notes
        -----
        限位只来自 ``Driver.get_config()["joint_limits"]``；不会截断目标，
        变化量检查会读取 ``Driver.get_joint_angles()``，可能短暂阻塞。
        """
        command = self._array_from_input(joints, (NERO_DOF,))
        for index, (lower, upper) in enumerate(self._joint_limits, start=1):
            if command[index - 1] < lower or command[index - 1] > upper:
                raise ValueError(f"joint {index} violates official joint limits: limit {lower}~{upper}, actually {command[index - 1]}")
        delta = self._max_joint_delta if max_joint_delta is None else self._validate_delta_setting(max_joint_delta)
        if delta is not None:
            current = self.get_joint_positions()
            if np.any(np.abs(command - current) > delta):
                raise ValueError("Joint command exceeds max_joint_delta.")
        return command

    @staticmethod
    def _validate_speed_percent(speed_percent: Any) -> int:
        if (
            isinstance(speed_percent, bool)
            or not isinstance(speed_percent, (int, np.integer))
            or speed_percent < 0
            or speed_percent > 100
        ):
            raise ValueError("speed_percent must be an integer in [0, 100].")
        return int(speed_percent)

    @staticmethod
    def _validate_pose(pose: Any) -> np.ndarray:
        result = NeroArm._array_from_input(pose, (6,))
        if result[3] < -np.pi or result[3] > np.pi:
            raise ValueError("roll must be in [-pi, pi].")
        if result[4] < -np.pi / 2 or result[4] > np.pi / 2:
            raise ValueError("pitch must be in [-pi/2, pi/2].")
        if result[5] < -np.pi or result[5] > np.pi:
            raise ValueError("yaw must be in [-pi, pi].")
        return result

    def _prepare_motion(self, speed_percent: Optional[int]) -> Optional[int]:
        self._require_connected()
        self._require_enabled()
        if speed_percent is None:
            return None
        return self._validate_speed_percent(speed_percent)

    def move_joints(self, joints: Any, *, speed_percent: Optional[int] = None) -> None:
        """验证后非阻塞地发送七轴关节目标。

        Parameters
        ----------
        joints : array-like, shape (7,)
            关节 1--7 目标位置，单位 rad，受官方限位与配置变化量限制。
        speed_percent : int or None, optional
            官方速度百分比，范围 ``[0, 100]``；不是 rad/s，``None`` 保持 SDK 设置。

        Returns
        -------
        None
            可选速度设置与目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接、未全部使能或变化量反馈不可用时抛出。
        ValueError
            目标或速度百分比不合法时抛出。

        Notes
        -----
        可选映射 ``set_speed_percent()`` 后映射 ``move_j()`` 一次，不等待完成；
        连续目标频率必须经真机逐级验证。
        """
        speed = self._prepare_motion(speed_percent)
        command = self.validate_joint_command(joints)
        if speed is not None:
            self._driver.set_speed_percent(speed)
        self._driver.move_j(command.tolist())

    def move_js(self, joints: Any) -> None:
        """验证后以 JS（Follower）模式非阻塞地发送七轴关节目标。

        Parameters
        ----------
        joints : array-like, shape (7,)
            关节 1--7 目标位置，单位 rad；受官方限位与构造时配置的
            ``max_joint_delta`` 限制。变化量相对于当前关节反馈检查。

        Returns
        -------
        None
            目标发送后立即返回，不等待到达。

        Raises
        ------
        RuntimeError
            未连接、未全部使能或变化量检查所需反馈不可用时抛出。
        ValueError
            目标形状、有限性、关节限位或变化量检查不通过时抛出。

        Notes
        -----
        映射官方 ``Driver.move_js()``，由 SDK 切换至 JS 模式。
        不做平滑或轨迹规划，不设置速度百分比；调用方负责控制更新频率。
        """
        self._prepare_motion(None)
        command = self.validate_joint_command(joints)
        self._driver.move_js(command.tolist())

    def move_pose(self, pose: Any, *, speed_percent: Optional[int] = None) -> None:
        """验证后非阻塞地发送法兰笛卡尔位姿目标。

        Parameters
        ----------
        pose : array-like, shape (6,)
            ``[x, y, z, roll, pitch, yaw]``，单位 ``[m, m, m, rad, rad, rad]``；
            roll/yaw 在 ``[-pi, pi]``，pitch 在 ``[-pi/2, pi/2]``。
        speed_percent : int or None, optional
            官方速度百分比整数 ``[0, 100]``；``None`` 保持 SDK 设置。

        Returns
        -------
        None
            可选速度设置与目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接或未全部使能时抛出。
        ValueError
            位姿或速度百分比不合法时抛出。

        Notes
        -----
        可选映射 ``set_speed_percent()`` 后映射 ``move_p()``；不等待完成。
        """
        speed = self._prepare_motion(speed_percent)
        command = self._validate_pose(pose)
        if speed is not None:
            self._driver.set_speed_percent(speed)
        self._driver.move_p(command.tolist())

    def move_linear(self, pose: Any, *, speed_percent: Optional[int] = None) -> None:
        """验证后非阻塞地发送线性法兰位姿目标。

        Parameters
        ----------
        pose : array-like, shape (6,)
            ``[x, y, z, roll, pitch, yaw]``，单位 ``[m, m, m, rad, rad, rad]``；
            roll/yaw 在 ``[-pi, pi]``，pitch 在 ``[-pi/2, pi/2]``。
        speed_percent : int or None, optional
            官方速度百分比整数 ``[0, 100]``；``None`` 保持 SDK 设置。

        Returns
        -------
        None
            可选速度设置与目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接或未全部使能时抛出。
        ValueError
            位姿或速度百分比不合法时抛出。

        Notes
        -----
        可选映射 ``set_speed_percent()`` 后映射 ``move_l()``；不等待完成，官方不
        承诺其可用作连续目标流。
        """
        speed = self._prepare_motion(speed_percent)
        command = self._validate_pose(pose)
        if speed is not None:
            self._driver.set_speed_percent(speed)
        self._driver.move_l(command.tolist())

    def set_tcp_offset(self, pose: List[float]) -> None:
        self._driver.set_tcp_offset(pose)

    def emergency_stop(self) -> None:
        """立即请求 Nero 电子急停。

        Returns
        -------
        None
            请求发送后返回。

        Raises
        ------
        RuntimeError
            未连接时抛出。

        Notes
        -----
        映射 ``Driver.electronic_emergency_stop()``；不等待，急停后 reset 可能
        造成机械臂下落。
        """
        self._require_connected()
        self._driver.electronic_emergency_stop()

    def reset(self) -> None:
        """请求官方 Nero 运动状态复位。

        Returns
        -------
        None
            请求发送后返回。

        Raises
        ------
        RuntimeError
            未连接时抛出。

        Notes
        -----
        映射 ``Driver.reset()``；不会自动急停或使能，也不等待，急停后的 reset
        可能造成机械臂下落。
        """
        self._require_connected()
        self._driver.reset()
