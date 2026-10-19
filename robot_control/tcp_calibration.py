"""Joint RGB-D camera/base and fixed contact-point calibration, without hardware.

``a_from_b`` maps coordinates expressed in frame b into frame a.
Positions are metres. The camera and robot base must remain fixed during capture.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cloth_agent.geometry import transform


class CalibrationError(ValueError):
    pass


def tcp_rotation_from_reference(base_from_flange) -> np.ndarray:
    """Tool-fixed rotation: reference Z down, X projected flange +Z, right-handed Y.

    Evaluate this once at the replay reference, not at every tilted sample.
    Nero base +Z is assumed to be physically upward.
    """
    rotation = transform(base_from_flange)[:3, :3]
    z = np.array([0., 0., -1.])
    x = rotation[:, 2] - np.dot(rotation[:, 2], z) * z
    if np.linalg.norm(x) < 1e-6:
        raise CalibrationError("Reference flange axis is vertical; its horizontal projection is undefined.")
    x /= np.linalg.norm(x)
    base_from_tcp_rotation = np.column_stack((x, np.cross(z, x), z))
    return rotation.T @ base_from_tcp_rotation


@dataclass
class CalibrationResult:
    base_from_camera: np.ndarray
    tcp_offset_flange_m: np.ndarray
    residuals_m: np.ndarray
    jacobian_singular_values: np.ndarray
    condition_number: float

    def as_dict(self, flange_from_tcp_rotation) -> dict:
        flange_from_tcp = np.eye(4)
        flange_from_tcp[:3, :3] = flange_from_tcp_rotation
        flange_from_tcp[:3, 3] = self.tcp_offset_flange_m
        transform(flange_from_tcp)
        distances = np.linalg.norm(self.residuals_m, axis=1)
        return {
            "units": {"translation": "m", "residual": "m"},
            "transform_convention": "a_from_b maps coordinates from b to a",
            "camera_frame": "color optical (aligned RGB-D)",
            "base_frame": "Nero base_link",
            "flange_frame": "Nero link7",
            "base_from_camera": self.base_from_camera.tolist(),
            "camera_from_base": np.linalg.inv(self.base_from_camera).tolist(),
            "flange_from_tcp": flange_from_tcp.tolist(),
            "tcp_from_flange": np.linalg.inv(flange_from_tcp).tolist(),
            "fit": {
                "sample_count": len(distances),
                "rms_m": float(np.sqrt(np.mean(distances ** 2))),
                "max_m": float(distances.max()),
                "per_sample_m": distances.tolist(),
                "condition_number": self.condition_number,
                "jacobian_singular_values": self.jacobian_singular_values.tolist(),
            },
        }


def solve_camera_tcp(points_camera, base_from_flange) -> CalibrationResult:
    """Fit R_BC c_i + t_BC = R_BF,i p_F + t_BF,i (9 unknowns).

    The TCP orientation is defined separately; observing one point cannot measure it.
    No samples are silently removed. Residuals retain the input sample order.
    """
    points = np.asarray(points_camera, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise CalibrationError("Camera points must be a finite (N, 3) array.")
    if len(points) < 6:
        raise CalibrationError("Capture at least 6 samples; 15-25 varied poses are recommended.")
    poses = np.asarray([transform(pose) for pose in base_from_flange])
    if poses.shape != (len(points), 4, 4):
        raise CalibrationError("Each camera point needs one matching flange FK pose.")
    rotations, positions = poses[:, :3, :3], poses[:, :3, 3]
    # A common unchanged direction makes tool and camera translations inseparable.
    motion_singular = np.linalg.svd(np.vstack(rotations - rotations[0]), compute_uv=False)
    if motion_singular[-1] < max(1e-5, motion_singular[0] * 1e-3):
        raise CalibrationError(
            "Unobservable TCP/camera translation: fixed orientation or yaw-only samples. "
            "Keep the fingers fixed and add roll/pitch tilts about different directions."
        )

    # Rigid registration to flange origins gives a direct starting estimate (p_F=0).
    centered_camera, centered_base = points - points.mean(0), positions - positions.mean(0)
    u, _, vt = np.linalg.svd(centered_camera.T @ centered_base)
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    initial_rotation = vt.T @ correction @ u.T
    initial_translation = positions.mean(0) - initial_rotation @ points.mean(0)
    initial = np.r_[Rotation.from_matrix(initial_rotation).as_rotvec(), initial_translation, np.zeros(3)]

    def residual(parameters):
        camera_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        from_camera = points @ camera_rotation.T + parameters[3:6]
        from_robot = np.einsum("nij,j->ni", rotations, parameters[6:9]) + positions
        return (from_camera - from_robot).ravel()

    fitted = least_squares(residual, initial, jac="3-point", x_scale="jac",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
    if not fitted.success:
        raise CalibrationError(f"Calibration did not converge: {fitted.message}")
    # Normalize columns so the observability check is not tied to metres vs radians.
    norms = np.linalg.norm(fitted.jac, axis=0)
    normalized = fitted.jac / np.maximum(norms, 1e-15)
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    if singular[-1] < 1e-4 * singular[0]:
        raise CalibrationError(
            "Calibration geometry is ill-conditioned; spread positions and use independent wrist tilts."
        )
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = Rotation.from_rotvec(fitted.x[:3]).as_matrix()
    base_from_camera[:3, 3] = fitted.x[3:6]
    return CalibrationResult(base_from_camera, fitted.x[6:9].copy(),
                             fitted.fun.reshape(-1, 3), singular, condition)
