"""Lazy AnyDex Linker L20 retargeting adapter."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


class AnyDexL20Mapper:
    def __init__(self, side: str, max_raw_step: int = 10, config_path=None):
        workspace = Path(__file__).resolve().parents[2]
        anydex_root = workspace / "AnyDexRetarget"
        if str(anydex_root) not in sys.path:
            sys.path.insert(0, str(anydex_root))
        try:
            from anydexretarget import Retargeter, l20_qpos_to_can_slots, slew_l20_command
        except ImportError as exc:
            raise ImportError(
                "AnyDex dependencies are unavailable; use the anydex Python environment"
            ) from exc
        config = Path(config_path) if config_path else (
            anydex_root / "example/config/vector/quest3/quest3_linker_l20.yaml"
        )
        self.side = side
        self.retargeter = Retargeter.from_yaml(str(config), hand_side=side)
        self.joint_names = self.retargeter.optimizer.robot.dof_joint_names
        self._map_slots = l20_qpos_to_can_slots
        self._slew = slew_l20_command
        self.max_raw_step = int(max_raw_step)
        self.previous_raw = None

    def seed(self, raw_feedback):
        raw = np.asarray(raw_feedback, dtype=np.int64)
        if raw.shape != (20,) or np.any((raw < 0) | (raw > 255)):
            raise ValueError("L20 feedback must be shape (20,) in [0,255]")
        self.previous_raw = raw.copy()

    def map(self, landmarks_local) -> tuple[np.ndarray, np.ndarray]:
        landmarks = np.asarray(landmarks_local, dtype=np.float64)
        if landmarks.shape != (21, 3) or not np.isfinite(landmarks).all():
            raise ValueError("hand landmarks must be finite shape (21,3)")
        qpos = np.asarray(self.retargeter.retarget(landmarks), dtype=np.float64)
        target = self._map_slots(qpos, self.joint_names)
        command = target if self.previous_raw is None else self._slew(
            target, self.previous_raw, self.max_raw_step
        )
        return qpos, command

    def commit(self, raw_command):
        self.previous_raw = np.asarray(raw_command, dtype=np.int64).copy()


class VirtualHand:
    def __init__(self):
        self.raw = np.full(20, 255, dtype=np.int64)
        self.sent = []

    def get_joint_positions_raw(self, fresh=True):
        return self.raw.copy()

    def set_joint_positions_raw(self, command):
        command = np.asarray(command, dtype=np.int64)
        if command.shape != (20,) or np.any((command < 0) | (command > 255)):
            raise ValueError("invalid virtual L20 command")
        self.raw = command.copy()
        self.sent.append(command.copy())
