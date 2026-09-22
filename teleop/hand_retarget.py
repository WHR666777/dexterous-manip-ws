"""Adapter around the authoritative root AnyDexRetarget checkout."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


class L20Retargeter:
    def __init__(self, config_path: str, hand_side: str, max_raw_step: int) -> None:
        root = Path(__file__).resolve().parents[1]
        anydex_root = root / "AnyDexRetarget"
        if str(anydex_root) not in sys.path:
            sys.path.insert(0, str(anydex_root))
        from anydexretarget import Retargeter, l20_qpos_to_can_slots, slew_l20_command

        self._retargeter = Retargeter.from_yaml(config_path, hand_side=hand_side)
        self._map = l20_qpos_to_can_slots
        self._slew = slew_l20_command
        self._joint_names = self._retargeter.optimizer.robot.dof_joint_names
        self._max_raw_step = int(max_raw_step)
        self._previous: np.ndarray | None = None

    def reset(self, previous_raw: np.ndarray | None = None) -> None:
        self._retargeter.reset()
        self._previous = None if previous_raw is None else np.asarray(previous_raw, dtype=np.int64).copy()

    def retarget(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._retargeter.retarget(np.asarray(landmarks, dtype=np.float64))
        raw = self._map(qpos, self._joint_names, reserved=[255, 255, 255, 255])
        if self._previous is not None:
            raw = self._slew(raw, self._previous, self._max_raw_step)
        self._previous = raw.copy()
        return np.asarray(qpos, dtype=np.float64), raw

