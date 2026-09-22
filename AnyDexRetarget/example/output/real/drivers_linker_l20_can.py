"""SocketCAN output driver for the physical LinkerHand L20/G20 hand."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from anydexretarget.linker_l20 import (
    l20_qpos_to_can_slots as _public_l20_qpos_to_can_slots,
)

from .base import HandOutput
from .drivers_linker_l20 import (
    _L20_ACTIVE_JOINT_ORDER,
    _clamp_uint8,
)


_L20_CAN_MIN_COMMAND_HZ = 1.0
_L20_CAN_MAX_COMMAND_HZ = 200.0
_L20_CAN_DEFAULT_COMMAND_HZ = 60.0
_L20_CAN_DEFAULT_MAX_STEP = 10
_L20_CAN_RESERVED_SLICE = slice(11, 15)
_L20_CAN_DEG_TO_RAD = np.pi / 180.0

_L20_CAN_JOINT_LIMITS = {
    "THUMB_CMC_PITCH": (0.0, 47.56 * _L20_CAN_DEG_TO_RAD),
    "INDEX_MCP_PITCH": (0.0, 69.90 * _L20_CAN_DEG_TO_RAD),
    "MIDDLE_MCP_PITCH": (0.0, 69.90 * _L20_CAN_DEG_TO_RAD),
    "RING_MCP_PITCH": (0.0, 69.90 * _L20_CAN_DEG_TO_RAD),
    "PINKY_MCP_PITCH": (0.0, 69.90 * _L20_CAN_DEG_TO_RAD),
    "THUMB_CMC_ROLL": (0.0, 79.64 * _L20_CAN_DEG_TO_RAD),
    "INDEX_MCP_ROLL": (-13.18 * _L20_CAN_DEG_TO_RAD, 13.18 * _L20_CAN_DEG_TO_RAD),
    "MIDDLE_MCP_ROLL": (-13.18 * _L20_CAN_DEG_TO_RAD, 13.18 * _L20_CAN_DEG_TO_RAD),
    "RING_MCP_ROLL": (-13.18 * _L20_CAN_DEG_TO_RAD, 13.18 * _L20_CAN_DEG_TO_RAD),
    "PINKY_MCP_ROLL": (-13.18 * _L20_CAN_DEG_TO_RAD, 13.18 * _L20_CAN_DEG_TO_RAD),
    "THUMB_CMC_YAW": (0.0, 89.95 * _L20_CAN_DEG_TO_RAD),
    "THUMB_MCP": (0.0, 71.62 * _L20_CAN_DEG_TO_RAD),
    "INDEX_PIP": (0.0, 100.27 * _L20_CAN_DEG_TO_RAD),
    "MIDDLE_PIP": (0.0, 100.27 * _L20_CAN_DEG_TO_RAD),
    "RING_PIP": (0.0, 100.27 * _L20_CAN_DEG_TO_RAD),
    "PINKY_PIP": (0.0, 100.27 * _L20_CAN_DEG_TO_RAD),
}

_L20_CAN_SLOT_MAP = (
    (0, "THUMB_CMC_PITCH", "decreasing"),
    (1, "INDEX_MCP_PITCH", "decreasing"),
    (2, "MIDDLE_MCP_PITCH", "decreasing"),
    (3, "RING_MCP_PITCH", "decreasing"),
    (4, "PINKY_MCP_PITCH", "decreasing"),
    (5, "THUMB_CMC_ROLL", "decreasing"),
    (6, "INDEX_MCP_ROLL", "increasing"),
    (7, "MIDDLE_MCP_ROLL", "increasing"),
    (8, "RING_MCP_ROLL", "increasing"),
    (9, "PINKY_MCP_ROLL", "increasing"),
    (10, "THUMB_CMC_YAW", "decreasing"),
    (15, "THUMB_MCP", "decreasing"),
    (16, "INDEX_PIP", "decreasing"),
    (17, "MIDDLE_PIP", "decreasing"),
    (18, "RING_PIP", "decreasing"),
    (19, "PINKY_PIP", "decreasing"),
)


def _joint_indices(qpos: np.ndarray, joint_names) -> dict[str, int]:
    names = tuple(str(name) for name in joint_names)
    if qpos.ndim != 1:
        raise ValueError(f"Expected 1-D Linker L20 qpos, got shape {qpos.shape}")
    if len(qpos) != len(names):
        raise ValueError(
            "Linker L20 qpos/joint-name length mismatch: "
            f"{len(qpos)} positions vs {len(names)} names"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Invalid Linker L20 retarget output: contains NaN/Inf")

    upper_names = [name.upper() for name in names]
    result: dict[str, int] = {}
    for suffix in _L20_ACTIVE_JOINT_ORDER:
        matches = [index for index, name in enumerate(upper_names) if name.endswith(suffix)]
        if not matches:
            raise ValueError(f"Missing Linker L20 joint in retarget output: *{suffix}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous Linker L20 joint suffix *{suffix}: {matches}")
        result[suffix] = matches[0]
    return result


def _reserved_values(reserved) -> np.ndarray:
    if reserved is None:
        return np.full(4, 255, dtype=np.int64)
    values = np.asarray(reserved)
    if values.shape != (4,):
        raise ValueError(f"Linker L20 reserved slots must have shape (4,), got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("Linker L20 reserved slots must be integer values")
    values = values.astype(np.int64)
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("Linker L20 reserved slots must be in [0, 255]")
    return values


def _normalized_can_joint(qpos: np.ndarray, indices: dict[str, int], suffix: str) -> float:
    lower, upper = _L20_CAN_JOINT_LIMITS[suffix]
    value = float(qpos[indices[suffix]])
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _raw_can_joint(
    qpos: np.ndarray,
    indices: dict[str, int],
    suffix: str,
    direction: str,
) -> int:
    ratio = _normalized_can_joint(qpos, indices, suffix)
    if direction == "decreasing":
        ratio = 1.0 - ratio
    elif direction != "increasing":
        raise ValueError(f"Unknown Linker L20 CAN direction: {direction}")
    return _clamp_uint8(ratio * 255.0)


def l20_qpos_to_can_slots(
    qpos,
    joint_names,
    reserved=None,
    pinch_context: tuple[str, float] | None = None,
) -> np.ndarray:
    """Map retargeted L20 radians into the documented 20-slot CAN raw order."""
    _ = pinch_context
    qpos_array = np.asarray(qpos, dtype=np.float64)
    indices = _joint_indices(qpos_array, joint_names)

    command = np.zeros(20, dtype=np.int64)
    for slot, suffix, direction in _L20_CAN_SLOT_MAP:
        command[slot] = _raw_can_joint(qpos_array, indices, suffix, direction)
    command[_L20_CAN_RESERVED_SLICE] = _reserved_values(reserved)
    return command


# Backwards-compatible example import; the installable package is now the
# single public entry point used by new integrations.
l20_qpos_to_can_slots = _public_l20_qpos_to_can_slots


def _load_hand_factory():
    try:
        from robot_control import LinkerHandL20
    except ModuleNotFoundError:
        workspace_root = Path(__file__).resolve().parents[4]
        workspace_path = str(workspace_root)
        if workspace_path not in sys.path:
            sys.path.insert(0, workspace_path)
        try:
            from robot_control import LinkerHandL20
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Linker L20 CAN mode requires the official LinkerHand/G20 "
                "Python SDK module `robot_control` with `LinkerHandL20`. "
                "Install/copy that SDK into this Python environment or the "
                "workspace root, or run with `--l20-dry-run` to test mapping "
                "without hardware output."
            ) from exc
    return LinkerHandL20


class LinkerL20CANOutput(HandOutput):
    """Send rate- and slew-limited L20 targets through the official G20 CAN API."""

    def __init__(
        self,
        hand_side: str = "right",
        can_channel: str = "can0",
        speed: int = 30,
        command_hz: float = _L20_CAN_DEFAULT_COMMAND_HZ,
        max_step: int = _L20_CAN_DEFAULT_MAX_STEP,
        clear_faults: bool = False,
        print_registers: bool = False,
        dry_run: bool = False,
        hand_factory: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if hand_side not in ("left", "right"):
            raise ValueError("hand_side must be 'left' or 'right'")
        if not isinstance(can_channel, str) or not can_channel:
            raise ValueError("can_channel must be a nonempty string")
        if isinstance(speed, bool) or not isinstance(speed, (int, np.integer)) or not 0 <= int(speed) <= 255:
            raise ValueError("speed must be an integer in [0, 255]")
        if isinstance(command_hz, bool) or not isinstance(
            command_hz, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"command_hz must be in [1, {_L20_CAN_MAX_COMMAND_HZ:g}]")
        normalized_command_hz = float(command_hz)
        if not np.isfinite(normalized_command_hz) or not (
            _L20_CAN_MIN_COMMAND_HZ
            <= normalized_command_hz
            <= _L20_CAN_MAX_COMMAND_HZ
        ):
            raise ValueError(f"command_hz must be in [1, {_L20_CAN_MAX_COMMAND_HZ:g}]")
        if isinstance(max_step, bool) or not isinstance(max_step, (int, np.integer)) or not 1 <= int(max_step) <= 255:
            raise ValueError("max_step must be an integer in [1, 255]")
        if not callable(clock):
            raise ValueError("clock must be callable")

        self._command_interval = 1.0 / normalized_command_hz
        self._max_step = int(max_step)
        self._print_registers = bool(print_registers)
        self._dry_run = bool(dry_run)
        self._clock = clock
        self._last_send_time = float("-inf")
        self._current_raw: np.ndarray | None = None
        self._hand = None
        self.send_count = 0

        if self._dry_run:
            print("Linker L20 CAN dry-run enabled: CAN output is disabled.")
            return

        factory = hand_factory if hand_factory is not None else _load_hand_factory()
        hand = factory(hand_type=hand_side, can_channel=can_channel)
        try:
            hand.connect()
            if clear_faults:
                hand.clear_faults()
            initial = np.asarray(hand.get_joint_positions_raw(), dtype=np.int64)
            if initial.shape != (20,) or np.any(initial < 0) or np.any(initial > 255):
                raise RuntimeError("L20 initial position feedback must be 20 raw values in [0, 255]")
            hand.set_speed(np.full(5, int(speed), dtype=np.int64))
        except BaseException as startup_error:
            try:
                hand.disconnect()
            except BaseException as cleanup_error:
                raise startup_error from cleanup_error
            raise

        self._hand = hand
        self._current_raw = initial.copy()
        if np.all(self._current_raw[_L20_CAN_RESERVED_SLICE] == 0):
            print(
                "Note: the official G20 API reports reserved slots 11-14 as zero; "
                "commands will keep those reserved slots at 255."
            )
        print(
            f"Connected to Linker L20 CAN on {can_channel} "
            f"(speed={int(speed)}, command_hz={normalized_command_hz:g}, max_step={int(max_step)})."
        )

    def send(self, qpos, joint_names) -> None:
        now = float(self._clock())
        if now - self._last_send_time + 1e-12 < self._command_interval:
            return

        target = l20_qpos_to_can_slots(qpos, joint_names)

        if self._current_raw is None:
            command = target
        else:
            command = np.clip(
                target,
                self._current_raw - self._max_step,
                self._current_raw + self._max_step,
            ).astype(np.int64)
            command[_L20_CAN_RESERVED_SLICE] = target[_L20_CAN_RESERVED_SLICE]

        if self._current_raw is not None and np.array_equal(command, self._current_raw):
            self._last_send_time = now
            return

        if self._print_registers or self._dry_run:
            print(f"L20 CAN raw slots: {command.tolist()}")
        if self._hand is not None:
            self._hand.set_joint_positions_raw(command)

        self._current_raw = command.copy()
        self._last_send_time = now
        self.send_count += 1

    def close(self) -> None:
        if self._hand is None:
            return
        self._hand.disconnect()
        self._hand = None


__all__ = [
    "LinkerL20CANOutput",
    "l20_qpos_to_can_slots",
]
