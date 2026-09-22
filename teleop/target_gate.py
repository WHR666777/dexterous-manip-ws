"""Latched Cartesian target step gate."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class TargetGate:
    def __init__(
        self,
        max_position_step_m: float,
        max_rotation_step_deg: float,
        tcp_workspace_cube_side_m: float,
    ) -> None:
        self.max_position_step_m = float(max_position_step_m)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.tcp_workspace_cube_side_m = float(tcp_workspace_cube_side_m)
        if (
            not np.isfinite(self.tcp_workspace_cube_side_m)
            or self.tcp_workspace_cube_side_m <= 0
        ):
            raise ValueError("tcp_workspace_cube_side_m must be finite and positive.")
        self.paused = True
        self.reason = "initial anchor required"
        self._accepted: np.ndarray | None = None
        self._tcp_workspace_center: np.ndarray | None = None

    def set_tcp_workspace_center(self, center: np.ndarray) -> None:
        candidate = np.asarray(center, dtype=np.float64)
        if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
            raise ValueError("TCP workspace center must contain three finite values.")
        self._tcp_workspace_center = candidate.copy()

    def reanchor(self, transform: np.ndarray) -> None:
        self._accepted = np.asarray(transform, dtype=np.float64).copy()
        self.paused = False
        self.reason = ""

    def pause(self, reason: str) -> None:
        self.paused = True
        self.reason = reason

    def accept(self, target: np.ndarray, tcp_position: np.ndarray) -> bool:
        if self.paused or self._accepted is None:
            return False
        candidate = np.asarray(target, dtype=np.float64)
        tcp_candidate = np.asarray(tcp_position, dtype=np.float64)
        if tcp_candidate.shape != (3,) or not np.all(np.isfinite(tcp_candidate)):
            self.pause("TCP target is invalid")
            return False
        if self._tcp_workspace_center is None:
            self.pause("TCP workspace center is not initialized")
            return False

        half_side = self.tcp_workspace_cube_side_m / 2.0
        tcp_offset = np.abs(tcp_candidate - self._tcp_workspace_center)
        if np.any(tcp_offset > half_side):
            axis = "xyz"[int(np.argmax(tcp_offset))]
            self.pause(
                f"TCP workspace cube exceeded on {axis}: "
                f"offset {float(np.max(tcp_offset)):.4f} m > {half_side:.4f} m"
            )
            return False
        position_step = np.linalg.norm(candidate[:3, 3] - self._accepted[:3, 3])
        rotation_step = Rotation.from_matrix(
            candidate[:3, :3] @ self._accepted[:3, :3].T
        ).magnitude()
        if position_step > self.max_position_step_m:
            self.pause(f"position step {position_step:.4f} m exceeded limit")
            return False
        if np.degrees(rotation_step) > self.max_rotation_step_deg:
            self.pause(f"rotation step {np.degrees(rotation_step):.2f} deg exceeded limit")
            return False
        self._accepted = candidate.copy()
        return True
