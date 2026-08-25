"""LinkerHand L20 官方 SDK 的轻量研究控制封装。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np


L20_JOINT_NAMES = (
    "thumb_base", "index_base", "middle_base", "ring_base", "little_base",
    "thumb_abduction", "index_abduction", "middle_abduction",
    "ring_abduction", "little_abduction", "thumb_roll",
    "reserved_11", "reserved_12", "reserved_13", "reserved_14",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
)
"""L20 官方 20 槽位的固定英文名称，含四个保留槽位。"""

L20_ACTIVE_POSITION_INDICES = tuple(range(11)) + tuple(range(15, 20))
"""可控制 L20 位置槽位在 20 元原始动作中的索引。"""

_L20_POSITION_SHAPE = (20,)
_L20_MOTOR_SHAPE = (5,)

# Official source: example/gui_control/config/constants.py, L20 ``张开`` action,
# LinkerHand commit 0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8.
_L20_OPEN_PRESET = (
    255, 255, 255, 255, 255, 255, 10, 100, 180, 240,
    245, 255, 255, 255, 255, 255, 255, 255, 255, 255,
)

# Official source: example/gui_control/config/constants.py, L20 ``握拳`` action,
# LinkerHand commit 0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8.
_L20_CLOSE_PRESET = (
    40, 0, 0, 0, 0, 131, 10, 100, 180, 240,
    19, 255, 255, 255, 255, 135, 0, 0, 0, 0,
)


def _load_linker_api() -> Callable[..., Any]:
    """加载官方 ``LinkerHandApi``，必要时使用项目内固定 SDK 副本。"""
    module_name = "LinkerHand.linker_hand_api"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as first_error:
        project_root = Path(__file__).resolve().parents[1]
        sdk_root = project_root / "linkerhand-python-sdk"
        api_file = sdk_root / "LinkerHand" / "linker_hand_api.py"
        if not sdk_root.is_dir() or not api_file.is_file():
            raise ImportError(
                "LinkerHand SDK is unavailable. Install it or place the pinned "
                "checkout at linkerhand-python-sdk/."
            ) from first_error
        sdk_path = str(sdk_root)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as retry_error:
            raise ImportError(
                "LinkerHand SDK could not be imported. Install its dependencies "
                "and verify linkerhand-python-sdk/."
            ) from retry_error
    try:
        return module.LinkerHandApi
    except AttributeError as exc:
        raise ImportError("LinkerHand SDK does not expose LinkerHandApi.") from exc


class LinkerHandL20:
    """以固定官方 L20 映射控制 LinkerHand 左手或右手。

    Parameters
    ----------
    hand_type : {"left", "right"}, default="right"
        官方 SDK 的手型标识。
    can_channel : str, default="can1"
        官方 SDK 使用的 CAN 通道名称。
    join_timeout : float, default=1.0
        每次断开时有限等待 SDK 接收线程退出的秒数。
    api_factory : callable or None, optional
        无硬件测试的官方 API 构造注入缝；``None`` 时在连接阶段懒加载 SDK。

    Raises
    ------
    ValueError
        构造配置不合法时抛出。

    Notes
    -----
    构造时不导入 SDK、不创建 CAN 对象；仅在 :meth:`connect` 创建官方
    ``LinkerHandApi``。位置始终以官方 20 槽位顺序表示，速度、电流和故障
    仅暴露 SDK 已支持的五电机数据。注入缝不改变真实 SDK 调用顺序。
    """

    def __init__(
        self,
        hand_type: str = "right",
        can_channel: str = "can1",
        join_timeout: float = 1.0,
        api_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        """创建惰性 L20 封装，不连接 CAN。

        Parameters
        ----------
        hand_type : str, default="right"
            官方 SDK 的手型标识；必须为 ``"left"`` 或 ``"right"``。
        can_channel : str, default="can1"
            传给官方 SDK 的 CAN 通道名称。
        join_timeout : float, default=1.0
            ``disconnect`` 每次等待 SDK 接收线程退出的有限非负秒数。
        api_factory : callable or None, optional
            创建官方 ``LinkerHandApi`` 的工厂。仅供无硬件测试注入；``None``
            时在连接阶段懒加载官方 SDK。

        Raises
        ------
        ValueError
            手型、通道或线程等待时间不是合法配置时抛出。

        Notes
        -----
        本方法不访问 CAN、不会创建线程，也不会调用 SDK。
        """
        if hand_type not in ("left", "right"):
            raise ValueError("hand_type must be 'left' or 'right'.")
        if not isinstance(can_channel, str) or not can_channel:
            raise ValueError("can_channel must be a nonempty string.")
        self._hand_type = hand_type
        self._can_channel = can_channel
        self._join_timeout = self._validate_join_timeout(join_timeout)
        self._api_factory = api_factory
        if self._api_factory is not None and not callable(self._api_factory):
            raise ValueError("api_factory must be callable or None.")
        self._api: Optional[Any] = None
        self._cleanup_api: Optional[Any] = None

    @staticmethod
    def _validate_join_timeout(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating),
        ):
            raise ValueError("join_timeout must be a finite nonnegative number.")
        try:
            normalized = float(value)
        except OverflowError as exc:
            raise ValueError("join_timeout must be a finite nonnegative number.") from exc
        if not np.isfinite(normalized) or normalized < 0:
            raise ValueError("join_timeout must be a finite nonnegative number.")
        return normalized

    def connect(self) -> None:
        """创建一次官方 L20 API 并开始其 CAN 生命周期。

        Returns
        -------
        None
            API 创建完成后返回。

        Raises
        ------
        ImportError
            官方 SDK 或其依赖不可导入时抛出。
        RuntimeError
            上一次断开仍在等待接收线程退出时抛出。

        Notes
        -----
        ``api_factory=None`` 时先在本调用内懒加载 ``LinkerHandApi`` 类，再精确
        调用 ``LinkerHandApi(hand_type=..., hand_joint="L20", modbus="None",
        can=...)``。SDK 构造会访问 CAN，可能阻塞；重复调用已连接实例不会重复
        创建 API。
        """
        if self._api is not None:
            return
        if self._cleanup_api is not None:
            raise RuntimeError("L20 receive-thread cleanup is still pending.")
        factory = self._api_factory
        if factory is None:
            factory = _load_linker_api()
        self._api = factory(
            hand_type=self._hand_type,
            hand_joint="L20",
            modbus="None",
            can=self._can_channel,
        )

    def disconnect(self) -> None:
        """停止 SDK 接收循环、关闭其 CAN 总线并有限等待线程退出。

        Returns
        -------
        None
            SDK 总线关闭且接收线程已退出后返回。

        Raises
        ------
        TimeoutError
            接收线程在 ``join_timeout`` 内仍存活时抛出；随后可再次调用以重试。
        RuntimeError
            SDK 关闭、线程访问或线程 join 发生错误时抛出。

        Notes
        -----
        断开一开始 :meth:`is_connected` 即返回 ``False``。超时时仅保留 SDK
        引用用于下次有限 join；在确认线程结束后立即清除。绝不调用操作系统
        ``ip link down``，只调用 SDK 的 ``close_can_interface()``。
        """
        is_cleanup_retry = self._api is None and self._cleanup_api is not None
        api = self._api if self._api is not None else self._cleanup_api
        if api is None:
            return
        self._api = None
        self._cleanup_api = api
        errors = []
        low_level = None
        try:
            low_level = api.hand
        except (AttributeError, TypeError) as exc:
            errors.append(exc)

        receive_thread = None
        if low_level is not None and not is_cleanup_retry:
            try:
                low_level.running = False
            except (AttributeError, TypeError) as exc:
                errors.append(exc)
            try:
                low_level.close_can_interface()
            except Exception as exc:
                errors.append(exc)
        if low_level is not None:
            try:
                receive_thread = low_level.receive_thread
                receive_thread.join(timeout=self._join_timeout)
            except Exception as exc:
                errors.append(exc)

        if receive_thread is None:
            self._cleanup_api = None
            self._raise_teardown_errors(errors)
            return

        try:
            still_alive = bool(receive_thread.is_alive())
        except Exception as exc:
            errors.append(exc)
            self._raise_teardown_errors(errors)
            return
        if still_alive:
            detail = self._format_teardown_errors(errors)
            if detail:
                raise TimeoutError(
                    "L20 receive thread did not stop before join_timeout; " + detail,
                )
            raise TimeoutError("L20 receive thread did not stop before join_timeout.")

        self._cleanup_api = None
        self._raise_teardown_errors(errors)

    @staticmethod
    def _format_teardown_errors(errors: list) -> str:
        return "; ".join(
            "{}: {}".format(type(error).__name__, error) for error in errors
        )

    @classmethod
    def _raise_teardown_errors(cls, errors: list) -> None:
        if errors:
            raise RuntimeError(
                "L20 disconnect teardown failed: " + cls._format_teardown_errors(errors),
            ) from errors[0]

    def is_connected(self) -> bool:
        """返回封装的 L20 连接状态。

        Returns
        -------
        bool
            官方 API 已创建且断开尚未开始时为 ``True``。

        Notes
        -----
        不查询硬件；断开开始后即使为线程清理暂留 SDK 引用也返回 ``False``。
        """
        return self._api is not None

    def _require_connected(self) -> Any:
        if self._api is None:
            raise RuntimeError("L20 hand is not connected.")
        return self._api

    @staticmethod
    def _array_from_input(values: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("L20 {} values must be numeric.".format(name)) from exc
        if array.shape != shape:
            raise ValueError(
                "L20 {} must have shape {}, got {}.".format(name, shape, array.shape),
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("L20 {} values must all be finite.".format(name))
        return array.copy()

    @classmethod
    def _validate_raw_values(
        cls, values: Any, shape: Tuple[int, ...], name: str,
    ) -> np.ndarray:
        array = cls._array_from_input(values, shape, name)
        if not np.all(array == np.floor(array)):
            raise ValueError("L20 {} values must be integral.".format(name))
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("L20 {} values must be in [0, 255].".format(name))
        return array.astype(np.uint8).astype(np.int64)

    @classmethod
    def _array_from_feedback(
        cls, values: Any, shape: Tuple[int, ...], name: str, raw: bool = False,
    ) -> np.ndarray:
        try:
            if raw:
                return cls._validate_raw_values(values, shape, name)
            return cls._array_from_input(values, shape, name)
        except ValueError as exc:
            raise RuntimeError(
                "L20 {} feedback is unavailable or malformed.".format(name),
            ) from exc

    @staticmethod
    def _validate_fresh(fresh: Any) -> bool:
        if not isinstance(fresh, bool):
            raise ValueError("fresh must be a bool.")
        return fresh

    def get_active_joint_indices(self) -> Tuple[int, ...]:
        """返回 L20 可控制位置槽位的固定索引。

        Returns
        -------
        tuple of int, length 16
            20 槽位中的可控制索引；11--14 为保留槽位而被排除。

        Notes
        -----
        返回不可变常量，不读取 SDK 或硬件。
        """
        return L20_ACTIVE_POSITION_INDICES

    def set_joint_positions_raw(self, positions: Any) -> None:
        """验证后发送官方 20 槽位原始位置目标。

        Parameters
        ----------
        positions : array-like, shape (20,)
            官方 L20 顺序的整数原始位置，每项在 ``[0, 255]``。

        Returns
        -------
        None
            调用 SDK 发送目标后立即返回。

        Raises
        ------
        ValueError
            输入形状、数值、整数性或范围不合法时抛出。
        RuntimeError
            未连接时抛出。

        Notes
        -----
        映射 ``LinkerHandApi.finger_move(pose=...)``；不等待手部运动完成。
        """
        raw = self._validate_raw_values(positions, _L20_POSITION_SHAPE, "position")
        self._require_connected().finger_move(pose=raw.tolist())

    def set_joint_positions_normalized(self, action: Any) -> None:
        """将归一化 20 槽位目标半向上转换后发送。

        Parameters
        ----------
        action : array-like, shape (20,)
            每项为有限值 ``[-1, 1]``；``-1`` 映射 0，``0`` 映射 128，``1``
            映射 255。

        Returns
        -------
        None
            转换并调用 SDK 后立即返回。

        Raises
        ------
        ValueError
            输入形状、数值或范围不合法时抛出。
        RuntimeError
            未连接时抛出。

        Notes
        -----
        使用 ``floor((x + 1) * 127.5 + 0.5)`` 半向上量化，再映射
        ``finger_move(pose=...)``；不等待运动完成。
        """
        values = self._array_from_input(action, _L20_POSITION_SHAPE, "position")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("L20 normalized position values must be in [-1, 1].")
        raw = np.floor((values + 1.0) * 127.5 + 0.5).astype(np.int64)
        self._require_connected().finger_move(pose=raw.tolist())

    def _get_joint_positions_raw(self, fresh: bool) -> np.ndarray:
        api = self._require_connected()
        feedback = api.get_state() if fresh else api.get_state_for_pub()
        return self._array_from_feedback(
            feedback, _L20_POSITION_SHAPE, "position", raw=True,
        )

    def get_joint_positions_raw(self, *, fresh: bool = True) -> np.ndarray:
        """读取新鲜或已缓存的 20 槽位原始位置。

        Parameters
        ----------
        fresh : bool, default=True
            ``True`` 调用官方 ``get_state()`` 请求状态；``False`` 调用
            ``get_state_for_pub()`` 读取 SDK 缓存。

        Returns
        -------
        numpy.ndarray, shape (20,), dtype int64
            官方顺序的独立原始位置副本，范围 ``[0, 255]``。

        Raises
        ------
        ValueError
            ``fresh`` 不是布尔值时抛出。
        RuntimeError
            未连接，或 SDK 反馈形状/数值不合法时抛出。

        Notes
        -----
        官方 ``get_state()`` 依次请求四类位置帧，典型源端等待约 40 ms；封装
        不承诺新鲜读取可超过 20 Hz。缓存读取不请求 CAN，也不制造缺失数据。
        """
        return self._get_joint_positions_raw(self._validate_fresh(fresh))

    def get_cached_joint_positions_raw(self) -> np.ndarray:
        """读取 SDK 发布缓存中的 20 槽位原始位置。

        Returns
        -------
        numpy.ndarray, shape (20,), dtype int64
            官方顺序的独立缓存位置副本，范围 ``[0, 255]``。

        Raises
        ------
        RuntimeError
            未连接，或 SDK 缓存反馈形状/数值不合法时抛出。

        Notes
        -----
        精确映射 ``LinkerHandApi.get_state_for_pub()``；不请求 CAN 更新。
        """
        return self._get_joint_positions_raw(False)

    def get_joint_positions_normalized(self, *, fresh: bool = True) -> np.ndarray:
        """读取新鲜或缓存位置并转换为 ``[-1, 1]``。

        Parameters
        ----------
        fresh : bool, default=True
            ``True`` 请求官方新鲜状态，``False`` 读取 SDK 发布缓存。

        Returns
        -------
        numpy.ndarray, shape (20,), dtype float64
            由原始位置 ``raw / 127.5 - 1`` 得到的独立归一化副本。

        Raises
        ------
        ValueError
            ``fresh`` 不是布尔值时抛出。
        RuntimeError
            未连接或 SDK 原始位置反馈不合法时抛出。

        Notes
        -----
        ``fresh=True`` 时的源端读取典型等待约 40 ms，封装不承诺可超过 20 Hz；
        不推断保留槽位，它们仍按官方 20 槽位原样转换。
        """
        return self.get_joint_positions_raw(fresh=fresh).astype(np.float64) / 127.5 - 1.0

    def get_cached_joint_positions_normalized(self) -> np.ndarray:
        """读取 SDK 缓存位置并转换为 ``[-1, 1]``。

        Returns
        -------
        numpy.ndarray, shape (20,), dtype float64
            独立归一化缓存位置副本。

        Raises
        ------
        RuntimeError
            未连接或 SDK 缓存位置反馈不合法时抛出。

        Notes
        -----
        映射缓存原始位置读取后使用 ``raw / 127.5 - 1``；不请求 CAN 更新。
        """
        return self.get_cached_joint_positions_raw().astype(np.float64) / 127.5 - 1.0

    def _validate_five_raw(self, values: Any, name: str) -> np.ndarray:
        return self._validate_raw_values(values, _L20_MOTOR_SHAPE, name)

    def set_speed(self, speed: Any) -> None:
        """验证后设置 L20 支持的五电机速度原始值。

        Parameters
        ----------
        speed : array-like, shape (5,)
            五电机整数速度原始值，每项在 ``[0, 255]``；SDK 未声明物理单位。

        Returns
        -------
        None
            设置请求发送后返回。

        Raises
        ------
        ValueError
            输入形状、数值、整数性或范围不合法时抛出。
        RuntimeError
            未连接时抛出。

        Notes
        -----
        精确映射 ``LinkerHandApi.set_speed(speed=...)``；SDK 请求可能短暂阻塞。
        """
        validated = self._validate_five_raw(speed, "speed")
        self._require_connected().set_speed(speed=validated.tolist())

    def get_speed(self) -> np.ndarray:
        """读取 L20 支持的五电机速度原始值。

        Returns
        -------
        numpy.ndarray, shape (5,), dtype int64
            五电机的独立整数速度副本，范围 ``[0, 255]``；无物理单位声明。

        Raises
        ------
        RuntimeError
            未连接或 SDK 速度反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 ``LinkerHandApi.get_speed()``，该 SDK 调用可能发起 CAN 请求。
        """
        return self._array_from_feedback(
            self._require_connected().get_speed(), _L20_MOTOR_SHAPE, "speed", raw=True,
        )

    def set_current(self, current: Any) -> None:
        """验证后设置 L20 支持的五电机电流原始值。

        Parameters
        ----------
        current : array-like, shape (5,)
            五电机整数电流原始值，每项在 ``[0, 255]``；SDK 未声明物理单位。

        Returns
        -------
        None
            设置请求发送后返回。

        Raises
        ------
        ValueError
            输入形状、数值、整数性或范围不合法时抛出。
        RuntimeError
            未连接时抛出。

        Notes
        -----
        精确映射 ``LinkerHandApi.set_current(current=...)``；不提供伪造扭矩接口。
        """
        validated = self._validate_five_raw(current, "current")
        self._require_connected().set_current(current=validated.tolist())

    def get_current(self) -> np.ndarray:
        """读取 L20 支持的五电机电流原始值。

        Returns
        -------
        numpy.ndarray, shape (5,), dtype int64
            五电机的独立整数电流副本，范围 ``[0, 255]``；无物理单位声明。

        Raises
        ------
        RuntimeError
            未连接或 SDK 电流反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 ``LinkerHandApi.get_current()``；不会将其命名或转换为扭矩。
        """
        return self._array_from_feedback(
            self._require_connected().get_current(), _L20_MOTOR_SHAPE, "current", raw=True,
        )

    def get_fault(self) -> np.ndarray:
        """读取 L20 支持的五电机故障码。

        Returns
        -------
        numpy.ndarray, shape (5,), dtype int64
            五电机独立整数故障码副本，范围 ``[0, 255]``。
            ``0`` 为正常，``1`` 为电流过载，``2`` 为过温，``3`` 为编码器错误，
            ``4`` 为过压或欠压。

        Raises
        ------
        RuntimeError
            未连接或 SDK 故障反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 ``LinkerHandApi.get_fault()``；SDK 调用可能发起 CAN 请求。
        """
        return self._array_from_feedback(
            self._require_connected().get_fault(), _L20_MOTOR_SHAPE, "fault", raw=True,
        )

    def clear_faults(self) -> None:
        """请求清除 L20 的五电机故障码。

        Returns
        -------
        None
            SDK 清故障请求返回后返回；不会伪造清除后的反馈数组。

        Raises
        ------
        RuntimeError
            未连接时抛出。

        Notes
        -----
        精确映射 ``LinkerHandApi.clear_faults()``；随后可用 :meth:`get_fault`
        读取实际 SDK 反馈。
        """
        self._require_connected().clear_faults()

    def get_temperature(self) -> np.ndarray:
        """读取官方 20 槽位温度反馈。

        Returns
        -------
        numpy.ndarray, shape (20,), dtype float64
            官方顺序的独立有限温度数据；SDK 未声明物理单位，封装不臆测单位。

        Raises
        ------
        RuntimeError
            未连接或 SDK 温度反馈形状/数值不合法时抛出。

        Notes
        -----
        映射 ``LinkerHandApi.get_temperature()``，该调用可能发起多个 CAN 请求。
        """
        return self._array_from_feedback(
            self._require_connected().get_temperature(), _L20_POSITION_SHAPE, "temperature",
        )

    def get_sdk_version(self) -> str:
        """读取 LinkerHand Python SDK 自身报告的版本字符串。

        Returns
        -------
        str
            官方 ``LinkerHandApi.version`` 的非空版本字符串。

        Raises
        ------
        RuntimeError
            未连接或 SDK 版本字段缺失、非字符串或为空时抛出。

        Notes
        -----
        不调用 L20 嵌入式 ``get_version()``，因为该官方实现返回占位数据。
        """
        api = self._require_connected()
        version = getattr(api, "version", None)
        if not isinstance(version, str) or not version:
            raise RuntimeError("L20 SDK version is unavailable or malformed.")
        return version

    def open_hand(self) -> None:
        """发送官方 L20 ``张开`` 预设动作。

        Returns
        -------
        None
            预设目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接时抛出。

        Notes
        -----
        使用固定官方 20 槽位 ``张开`` 值并映射 ``finger_move``；不等待运动完成。
        """
        self._require_connected().finger_move(pose=list(_L20_OPEN_PRESET))

    def close_hand(self) -> None:
        """发送官方 L20 ``握拳`` 预设动作。

        Returns
        -------
        None
            预设目标发送后立即返回。

        Raises
        ------
        RuntimeError
            未连接时抛出。

        Notes
        -----
        使用固定官方 20 槽位 ``握拳`` 值并映射 ``finger_move``；不等待运动完成。
        """
        self._require_connected().finger_move(pose=list(_L20_CLOSE_PRESET))
