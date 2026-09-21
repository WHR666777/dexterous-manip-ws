"""Rigid-transform conventions shared by all Quest teleoperation modes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# Unity: +X right, +Y up, +Z forward (left handed).
# Canonical teleop frame: +X forward, +Y left, +Z up (right handed).
UNITY_TO_FLU = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def valid_rotation(value, atol: float = 1e-5) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
        return (
            matrix.shape == (3, 3)
            and np.isfinite(matrix).all()
            and np.allclose(matrix.T @ matrix, np.eye(3), atol=atol)
            and np.isclose(np.linalg.det(matrix), 1.0, atol=atol)
        )
    except (TypeError, ValueError):
        return False


def valid_transform(value, atol: float = 1e-5) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
        return (
            matrix.shape == (4, 4)
            and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1], atol=atol)
            and valid_rotation(matrix[:3, :3], atol=atol)
        )
    except (TypeError, ValueError):
        return False


def unity_pose_to_flu(position, quaternion_xyzw) -> np.ndarray:
    """Convert one Unity world pose to an FLU homogeneous transform."""
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(
        np.r_[position, quaternion]
    ).all():
        raise ValueError("position/quaternion must be finite shape (3,)/(4,)")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-9:
        raise ValueError("zero quaternion")
    unity_rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    result = np.eye(4)
    result[:3, 3] = UNITY_TO_FLU @ position
    result[:3, :3] = UNITY_TO_FLU @ unity_rotation @ UNITY_TO_FLU.T
    return result


def unity_landmarks_to_flu(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("HTS landmarks must be finite shape (21, 3)")
    return (UNITY_TO_FLU @ points.T).T


def matrix_to_pose7(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if not valid_transform(transform):
        raise ValueError("expected a valid 4x4 rigid transform")
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]


def pose6_to_matrix(pose) -> np.ndarray:
    """NERO xyz + extrinsic xyz RPY feedback to a homogeneous matrix."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("NERO pose must be finite shape (6,)")
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return result


def pose_distance(first, second) -> tuple[float, float]:
    if not valid_transform(first) or not valid_transform(second):
        raise ValueError("pose_distance requires valid transforms")
    position = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = float(
        Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()
    )
    return position, rotation


def relative_flange_target(
    wrist_anchor,
    wrist_current,
    flange_anchor,
    base_from_quest_rotation,
    position_scale: float = 0.5,
) -> np.ndarray:
    """Map a world-space Quest wrist delta onto a robot flange anchor."""
    if not all(valid_transform(item) for item in (wrist_anchor, wrist_current, flange_anchor)):
        raise ValueError("wrist/flange anchors must be valid transforms")
    mapping = np.asarray(base_from_quest_rotation, dtype=np.float64)
    if not valid_rotation(mapping):
        raise ValueError("base_from_quest_rotation must be SO(3)")
    if not np.isfinite(position_scale) or position_scale <= 0:
        raise ValueError("position_scale must be positive")
    target = np.asarray(flange_anchor, dtype=np.float64).copy()
    delta_position = wrist_current[:3, 3] - wrist_anchor[:3, 3]
    target[:3, 3] += float(position_scale) * (mapping @ delta_position)
    delta_rotation = wrist_current[:3, :3] @ wrist_anchor[:3, :3].T
    target[:3, :3] = (
        mapping @ delta_rotation @ mapping.T @ flange_anchor[:3, :3]
    )
    return target


def calibrate_base_rotation(x_motion, y_motion, minimum_motion: float = 0.05) -> np.ndarray:
    """Build Quest-FLU to robot-base rotation from labelled +X/+Y motions."""
    x_motion = np.asarray(x_motion, dtype=np.float64)
    y_motion = np.asarray(y_motion, dtype=np.float64)
    if x_motion.shape != (3,) or y_motion.shape != (3,):
        raise ValueError("calibration motions must have shape (3,)")
    if not np.isfinite(np.r_[x_motion, y_motion]).all():
        raise ValueError("calibration motions must be finite")
    if np.linalg.norm(x_motion) < minimum_motion or np.linalg.norm(y_motion) < minimum_motion:
        raise ValueError("calibration motion is too short")
    x_axis = x_motion / np.linalg.norm(x_motion)
    y_orthogonal = y_motion - np.dot(y_motion, x_axis) * x_axis
    if np.linalg.norm(y_orthogonal) < minimum_motion * 0.35:
        raise ValueError("calibration +X/+Y motions are nearly collinear")
    y_axis = y_orthogonal / np.linalg.norm(y_orthogonal)
    z_axis = np.cross(x_axis, y_axis)
    quest_basis = np.column_stack((x_axis, y_axis, z_axis))
    mapping = quest_basis.T
    if not valid_rotation(mapping):
        raise ValueError("calibration did not produce an SO(3) rotation")
    return mapping
