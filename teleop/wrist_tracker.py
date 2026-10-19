"""Relative wrist anchoring migrated from the reference vr_teleop project."""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation


class WristTracker:
    """Map Quest wrist residuals onto a robot flange anchor with translation EMA."""

    def __init__(
        self,
        initial_site_pos: np.ndarray,
        initial_site_quat: np.ndarray,
        *,
        position_scale: float = 1.0,
        ema_alpha: float = 0.8,
        negate_rot_xy: bool = False,
        base_xmat: np.ndarray | None = None,
        position_deadband: float = 0.0,
        rotation_deadband_deg: float = 0.0,
        fixed_roll_pitch: np.ndarray | None = None,
        roll_pitch_step_deg: float = 1.0,
    ) -> None:
        self.position_scale = float(position_scale)
        self.ema_alpha = float(ema_alpha)
        self.negate_rot_xy = bool(negate_rot_xy)
        self.base_xmat = None if base_xmat is None else np.asarray(base_xmat, dtype=float).copy()
        self.position_deadband = float(position_deadband)
        self.rotation_deadband_deg = float(rotation_deadband_deg)
        self.fixed_roll_pitch = (None if fixed_roll_pitch is None
                                 else np.asarray(fixed_roll_pitch, dtype=float).copy())
        self.roll_pitch_step_rad = math.radians(roll_pitch_step_deg)
        if self.fixed_roll_pitch is not None:
            if self.fixed_roll_pitch.shape != (2,) or not np.isfinite(self.fixed_roll_pitch).all():
                raise ValueError("fixed_roll_pitch must contain two finite angles in radians.")
            if not np.isfinite(self.roll_pitch_step_rad) or self.roll_pitch_step_rad <= 0:
                raise ValueError("roll_pitch_step_deg must be finite and positive.")
        self.reanchor(initial_site_pos, initial_site_quat)

    def reanchor(self, site_pos: np.ndarray, site_quat: np.ndarray) -> None:
        self.initial_site_pos = np.asarray(site_pos, dtype=np.float64).copy()
        self.initial_site_quat = np.asarray(site_quat, dtype=np.float64).copy()
        self.initial_wrist_position: np.ndarray | None = None
        self.initial_wrist_quaternion: np.ndarray | None = None
        self.smoothed_residual: np.ndarray | None = None
        self.target_position = self.initial_site_pos.copy()
        self.target_quaternion = self.initial_site_quat.copy()
        if self.fixed_roll_pitch is not None:
            initial_rpy = Rotation.from_quat(self.initial_site_quat).as_euler("xyz")
            self._roll_pitch = initial_rpy[:2].copy()
            self._site_yaw = initial_rpy[2]

    @property
    def initialized(self) -> bool:
        return self.initial_wrist_position is not None

    def update(self, wrist_position: np.ndarray, wrist_quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(wrist_position, dtype=np.float64)
        quaternion = np.asarray(wrist_quaternion, dtype=np.float64)
        if self.initial_wrist_position is None:
            self.initial_wrist_position = position.copy()
            self.initial_wrist_quaternion = quaternion.copy()
            if self.fixed_roll_pitch is not None:
                wrist_rotation = Rotation.from_quat(quaternion).as_matrix()
                self._wrist_yaw = math.atan2(wrist_rotation[1, 0], wrist_rotation[0, 0])
            return self.target_position.copy(), self.target_quaternion.copy()

        residual = position - self.initial_wrist_position
        if self.base_xmat is not None:
            residual = self.base_xmat @ residual
        if self.smoothed_residual is None:
            self.smoothed_residual = residual
        else:
            self.smoothed_residual = (
                self.ema_alpha * residual + (1.0 - self.ema_alpha) * self.smoothed_residual
            )
        if self.position_deadband > 0 and np.linalg.norm(self.smoothed_residual) < self.position_deadband:
            self.smoothed_residual = np.zeros(3)
        self.target_position = self.initial_site_pos + self.position_scale * self.smoothed_residual

        if self.fixed_roll_pitch is not None:
            # Extract base-frame yaw directly so Pico roll/pitch cannot leak
            # into it through the full relative-rotation composition below.
            wrist_rotation = Rotation.from_quat(quaternion).as_matrix()
            yaw = math.atan2(wrist_rotation[1, 0], wrist_rotation[0, 0])
            yaw_delta = (yaw - self._wrist_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_delta) < math.radians(self.rotation_deadband_deg):
                yaw_delta = 0.0
            # Approach the demonstrated tilt after anchoring, rather than
            # requesting an instantaneous rotation from the startup pose.
            error = (self.fixed_roll_pitch - self._roll_pitch + math.pi) % (2 * math.pi) - math.pi
            distance = float(np.linalg.norm(error))
            if distance > 0:
                self._roll_pitch += error * min(1.0, self.roll_pitch_step_rad / distance)
            self.target_quaternion = Rotation.from_euler(
                "xyz", [*self._roll_pitch, self._site_yaw + yaw_delta]
            ).as_quat()
            return self.target_position.copy(), self.target_quaternion.copy()

        relative = Rotation.from_quat(quaternion) * Rotation.from_quat(self.initial_wrist_quaternion).inv()
        relative_quaternion = relative.as_quat()
        if self.negate_rot_xy:
            relative_quaternion[:2] *= -1.0
            relative = Rotation.from_quat(relative_quaternion)
        if self.rotation_deadband_deg > 0 and relative.magnitude() < math.radians(self.rotation_deadband_deg):
            relative = Rotation.identity()
        self.target_quaternion = (relative * Rotation.from_quat(self.initial_site_quat)).as_quat()
        return self.target_position.copy(), self.target_quaternion.copy()

