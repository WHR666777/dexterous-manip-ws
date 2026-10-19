"""Non-blocking thumb/index clip playback, with the other fingers held open."""

import csv
from pathlib import Path

import numpy as np

from robot_control.l20 import L20_JOINT_NAMES, _L20_OPEN_PRESET


THUMB_INDEX_INDICES = (0, 5, 10, 15, 1, 6, 16)


class HandReplay:
    def __init__(self, clip_file: str | Path):
        self.clip_file = Path(clip_file)
        with self.clip_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        times = np.asarray([float(row["video_time_s"]) for row in rows])
        values = np.asarray([
            [float(row[L20_JOINT_NAMES[i]]) for i in THUMB_INDEX_INDICES]
            for row in rows
        ])
        if (len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Hand replay clip needs at least two increasing video timestamps.")
        if (values.shape != (len(times), 7) or not np.isfinite(values).all()
                or np.any(values != np.floor(values))
                or np.any((values < 0) | (values > 255))):
            raise ValueError("Hand replay targets must be integers in [0, 255].")
        self.times = times - times[0]
        self.duration = float(self.times[-1] + np.median(np.diff(times)))
        self.actions = np.tile(_L20_OPEN_PRESET, (len(times), 1))
        self.actions[:, THUMB_INDEX_INDICES] = values.astype(np.int64)
        self.reset()

    def reset(self) -> None:
        self._started_at = None
        self._current = self.actions[0].copy()

    def start(self, now: float) -> None:
        self.reset()
        self._started_at = now

    def stop(self) -> None:
        """Cancel playback while retaining the last requested finger pose."""
        self._started_at = None

    def target(self, now: float) -> np.ndarray:
        if self._started_at is not None:
            elapsed = now - self._started_at
            index = np.searchsorted(self.times, elapsed + 1e-9, side="right") - 1
            self._current = self.actions[int(np.clip(index, 0, len(self.actions) - 1))].copy()
            if elapsed >= self.duration:
                self.stop()
        return self._current.copy()
