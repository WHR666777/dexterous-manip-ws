"""Public Linker L20/G20 joint-to-register mapping.

This module intentionally has no hardware dependency.  Retargeting and data
collection code can therefore produce the exact 20-slot G20 command without
constructing the example CAN driver.
"""

from __future__ import annotations

import numpy as np


L20_RESERVED_SLICE = slice(11, 15)
L20_ACTIVE_JOINT_ORDER = (
    "THUMB_CMC_YAW",
    "THUMB_CMC_ROLL",
    "THUMB_CMC_PITCH",
    "THUMB_MCP",
    "INDEX_MCP_ROLL",
    "INDEX_MCP_PITCH",
    "INDEX_PIP",
    "MIDDLE_MCP_ROLL",
    "MIDDLE_MCP_PITCH",
    "MIDDLE_PIP",
    "RING_MCP_ROLL",
    "RING_MCP_PITCH",
    "RING_PIP",
    "PINKY_MCP_ROLL",
    "PINKY_MCP_PITCH",
    "PINKY_PIP",
)

_DEG_TO_RAD = np.pi / 180.0

# Calibrated physical ranges.  Keep this table as the single source of truth
# for both the example hardware driver and dataset action generation.
L20_JOINT_LIMITS = {
    "THUMB_CMC_PITCH": (0.0, 47.56 * _DEG_TO_RAD),
    "INDEX_MCP_PITCH": (0.0, 69.90 * _DEG_TO_RAD),
    "MIDDLE_MCP_PITCH": (0.0, 69.90 * _DEG_TO_RAD),
    "RING_MCP_PITCH": (0.0, 69.90 * _DEG_TO_RAD),
    "PINKY_MCP_PITCH": (0.0, 69.90 * _DEG_TO_RAD),
    "THUMB_CMC_ROLL": (0.0, 79.64 * _DEG_TO_RAD),
    "INDEX_MCP_ROLL": (-13.18 * _DEG_TO_RAD, 13.18 * _DEG_TO_RAD),
    "MIDDLE_MCP_ROLL": (-13.18 * _DEG_TO_RAD, 13.18 * _DEG_TO_RAD),
    "RING_MCP_ROLL": (-13.18 * _DEG_TO_RAD, 13.18 * _DEG_TO_RAD),
    "PINKY_MCP_ROLL": (-13.18 * _DEG_TO_RAD, 13.18 * _DEG_TO_RAD),
    "THUMB_CMC_YAW": (0.0, 89.95 * _DEG_TO_RAD),
    "THUMB_MCP": (0.0, 71.62 * _DEG_TO_RAD),
    "INDEX_PIP": (0.0, 100.27 * _DEG_TO_RAD),
    "MIDDLE_PIP": (0.0, 100.27 * _DEG_TO_RAD),
    "RING_PIP": (0.0, 100.27 * _DEG_TO_RAD),
    "PINKY_PIP": (0.0, 100.27 * _DEG_TO_RAD),
}

L20_SLOT_MAP = (
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


def _clamp_uint8(value: float) -> int:
    return int(np.clip(np.rint(value), 0, 255))


def _joint_indices(qpos: np.ndarray, joint_names) -> dict[str, int]:
    names = tuple(str(name) for name in joint_names)
    if qpos.ndim != 1 or len(qpos) != len(names):
        raise ValueError(
            "Linker L20 qpos and joint_names must be matching 1-D sequences"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Linker L20 qpos contains NaN/Inf")
    upper_names = [name.upper() for name in names]
    result: dict[str, int] = {}
    for suffix in L20_ACTIVE_JOINT_ORDER:
        matches = [i for i, name in enumerate(upper_names) if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Linker L20 joint matching *{suffix}, got {matches}"
            )
        result[suffix] = matches[0]
    return result


def _reserved_values(reserved) -> np.ndarray:
    values = np.full(4, 255, dtype=np.int64) if reserved is None else np.asarray(reserved)
    if values.shape != (4,):
        raise ValueError("Linker L20 reserved slots must have shape (4,)")
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("Linker L20 reserved slots must be finite integers")
    values = values.astype(np.int64)
    if np.any((values < 0) | (values > 255)):
        raise ValueError("Linker L20 reserved slots must be in [0, 255]")
    return values


def l20_qpos_to_can_slots(qpos, joint_names, reserved=None) -> np.ndarray:
    """Map retargeted Linker L20 radians to the official 20-slot G20 order."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    indices = _joint_indices(qpos_array, joint_names)
    command = np.zeros(20, dtype=np.int64)
    for slot, suffix, direction in L20_SLOT_MAP:
        lower, upper = L20_JOINT_LIMITS[suffix]
        ratio = float(np.clip((qpos_array[indices[suffix]] - lower) / (upper - lower), 0, 1))
        if direction == "decreasing":
            ratio = 1.0 - ratio
        command[slot] = _clamp_uint8(ratio * 255.0)
    command[L20_RESERVED_SLICE] = _reserved_values(reserved)
    return command


def slew_l20_command(target, previous, max_step: int = 10) -> np.ndarray:
    """Limit active G20 slots while preserving target reserved values."""
    target_array = np.asarray(target, dtype=np.int64)
    previous_array = np.asarray(previous, dtype=np.int64)
    if target_array.shape != (20,) or previous_array.shape != (20,):
        raise ValueError("Linker L20 target and previous commands must have shape (20,)")
    if isinstance(max_step, bool) or not isinstance(max_step, (int, np.integer)) or max_step < 1:
        raise ValueError("max_step must be a positive integer")
    result = np.clip(target_array, previous_array - max_step, previous_array + max_step)
    result[L20_RESERVED_SLICE] = target_array[L20_RESERVED_SLICE]
    return result.astype(np.int64)


__all__ = [
    "L20_ACTIVE_JOINT_ORDER",
    "L20_JOINT_LIMITS",
    "L20_RESERVED_SLICE",
    "L20_SLOT_MAP",
    "l20_qpos_to_can_slots",
    "slew_l20_command",
]
