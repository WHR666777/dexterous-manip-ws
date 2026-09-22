"""Latched Cartesian target step gate."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class TargetGate:
    def __init__(self, max_position_step_m: float, max_rotation_step_deg: float) -> None:
        self.max_position_step_m = float(max_position_step_m)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.paused = True
        self.reason = "initial anchor required"
        self._accepted: np.ndarray | None = None

    def reanchor(self, transform: np.ndarray) -> None:
        self._accepted = np.asarray(transform, dtype=np.float64).copy()
        self.paused = False
        self.reason = ""

    def pause(self, reason: str) -> None:
        self.paused = True
        self.reason = reason

    def accept(self, target: np.ndarray) -> bool:
        if self.paused or self._accepted is None:
            return False
        candidate = np.asarray(target, dtype=np.float64)
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

