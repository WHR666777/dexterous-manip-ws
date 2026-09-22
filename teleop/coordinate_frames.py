"""AnyDex Quest coordinates and Nero-base pose conversions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def anydex_wrist_to_base(
    wrist_position: np.ndarray,
    wrist_quaternion: np.ndarray,
    r_base_quest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only site calibration to AnyDex's already converted RH wrist."""
    position_rh = np.asarray(wrist_position, dtype=np.float64)
    quaternion_rh = np.asarray(wrist_quaternion, dtype=np.float64)
    if position_rh.shape != (3,) or quaternion_rh.shape != (4,):
        raise ValueError("AnyDex wrist must be position (3,) plus xyzw quaternion (4,).")
    calibration = np.asarray(r_base_quest, dtype=np.float64)
    position = calibration @ position_rh
    base_rotation = calibration @ Rotation.from_quat(quaternion_rh).as_matrix()
    return position, Rotation.from_matrix(base_rotation).as_quat()


def pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("Nero pose must have shape (6,).")
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return transform


def position_quaternion_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return transform
