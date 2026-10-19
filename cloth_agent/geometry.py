"""Small, explicit geometry helpers. Positions are metres; angles are radians."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def transform(value) -> np.ndarray:
    candidate = np.asarray(value, dtype=float)
    if candidate.shape == (6,):
        result = np.eye(4)
        result[:3, 3] = candidate[:3]
        result[:3, :3] = Rotation.from_euler("xyz", candidate[3:]).as_matrix()
    else:
        result = candidate.copy()
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Expected a finite SE(3) matrix or xyz/roll-pitch-yaw pose.")
    return result


def transform_points(matrix, points) -> np.ndarray:
    pose = transform(matrix)
    points = np.asarray(points, dtype=float)
    if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must have shape (3,) or (N,3).")
    return points @ pose[:3, :3].T + pose[:3, 3]


def tcp_to_flange(task_tcp, task_from_base, flange_from_tcp) -> np.ndarray:
    return (np.linalg.inv(transform(task_from_base)) @ transform(task_tcp)
            @ np.linalg.inv(transform(flange_from_tcp)))


def interpolate_poses(start, end, position_step=0.005, rotation_step=np.deg2rad(2)):
    start, end = transform(start), transform(end)
    delta = Rotation.from_matrix(end[:3, :3] @ start[:3, :3].T).as_rotvec()
    count = max(1, int(np.ceil(max(
        np.linalg.norm(end[:3, 3] - start[:3, 3]) / position_step,
        np.linalg.norm(delta) / rotation_step,
    ))))
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        pose = np.eye(4)
        pose[:3, 3] = start[:3, 3] * (1 - fraction) + end[:3, 3] * fraction
        pose[:3, :3] = Rotation.from_rotvec(delta * fraction).as_matrix() @ start[:3, :3]
        yield pose


def calculate(operation: str, **values):
    if operation == "transform_points":
        return transform_points(values["matrix"], values["points"]).tolist()
    if operation == "tcp_to_flange":
        return tcp_to_flange(**values).tolist()
    if operation == "distance":
        return float(np.linalg.norm(np.asarray(values["a"]) - np.asarray(values["b"])))
    if operation == "offset":
        return (np.asarray(values["point"], dtype=float)
                + np.asarray(values["offset"], dtype=float)).tolist()
    if operation in ("camera_to_task", "pixel_to_task"):
        if values.get("task_from_camera") is None:
            raise ValueError("Camera-to-task calibration has not been configured.")
        if operation == "camera_to_task":
            points = values["points"]
        else:
            points = calculate("pixel_to_point", pixel=values["pixel"],
                               depth_m=values["depth_m"], intrinsics=values["intrinsics"])
        return transform_points(values["task_from_camera"], points).tolist()
    if operation == "pixel_to_point":
        u, v = values["pixel"]
        z, intr = float(values["depth_m"]), values["intrinsics"]
        if not np.isfinite(z) or z <= 0:
            raise ValueError("depth_m must be a positive valid depth.")
        coefficients = intr.get("coeffs", [])
        if any(abs(float(c)) > 0 for c in coefficients):
            if str(intr.get("model", "")).split(".")[-1] != "brown_conrady":
                raise ValueError("Projection requires a supported Brown-Conrady model or undistorted pixels.")
            import cv2
            matrix = np.array([[intr["fx"], 0, intr["ppx"]],
                               [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=float)
            xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), matrix,
                                     np.asarray(coefficients, dtype=float))[0, 0]
            return [float(xy[0] * z), float(xy[1] * z), z]
        return [(u - intr["ppx"]) * z / intr["fx"],
                (v - intr["ppy"]) * z / intr["fy"], z]
    raise ValueError(f"Unknown geometry operation: {operation}")
