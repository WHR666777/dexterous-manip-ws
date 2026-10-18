"""Periodic L20 queries outside the control loop; host timestamps only."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread
import time

import numpy as np


@dataclass(frozen=True)
class HandFeedbackSnapshot:
    positions_raw: np.ndarray
    query_started_at: float
    query_finished_at: float
    query_sequence: int


class HandFeedbackPoller:
    def __init__(self, hand, poll_hz: float = 20.0, max_age_s: float = .25):
        if not np.isfinite(poll_hz) or poll_hz <= 0:
            raise ValueError('hand feedback poll_hz must be finite and positive')
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('hand feedback max_age_s must be finite and positive')
        self.hand = hand
        self.poll_hz = float(poll_hz)
        self.max_age_s = float(max_age_s)
        self._lock = Lock()
        self._stop = Event()
        self._ready = Event()
        self._thread: Thread | None = None
        self._latest: HandFeedbackSnapshot | None = None
        self._error: Exception | None = None
        self._durations = deque(maxlen=256)
        self._first_query_at = None

    def start(self, timeout_s: float = 3.0) -> None:
        if self._stop.is_set():
            raise RuntimeError('A stopped hand feedback poller cannot be restarted')
        if self._thread is not None:
            raise RuntimeError('Hand feedback poller already started')
        self._thread = Thread(target=self._poll, name='l20-feedback', daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError('No complete L20 feedback query within startup timeout')
        self.get_latest()

    def _poll(self) -> None:
        period = 1.0 / self.poll_hz
        next_query = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                positions = self.hand.get_joint_positions_raw(fresh=True)
                positions = np.asarray(positions, dtype=float)
                if (positions.shape != (20,) or not np.isfinite(positions).all()
                        or np.any(positions != np.floor(positions))
                        or np.any(positions < 0) or np.any(positions > 255)):
                    raise RuntimeError('Incomplete L20 feedback query')
                positions = positions.astype(np.int64)
                finished = time.monotonic()
                with self._lock:
                    sequence = 1 if self._latest is None else self._latest.query_sequence + 1
                    self._latest = HandFeedbackSnapshot(positions, started, finished, sequence)
                    self._durations.append(finished - started)
                    if self._first_query_at is None:
                        self._first_query_at = started
                self._ready.set()
                next_query += period
                now = time.monotonic()
                if next_query < now:
                    # Skip missed slots; do not burst queries after a slow SDK call.
                    next_query += (int((now - next_query) / period) + 1) * period
                self._stop.wait(max(0.0, next_query - time.monotonic()))
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()

    def get_latest(self) -> HandFeedbackSnapshot:
        with self._lock:
            if self._error is not None:
                raise RuntimeError('L20 feedback query failed') from self._error
            snapshot = self._latest
            if snapshot is None:
                raise RuntimeError('L20 feedback query unavailable')
            if time.monotonic() - snapshot.query_finished_at > self.max_age_s:
                raise RuntimeError('L20 feedback query snapshot is stale')
            return replace(snapshot, positions_raw=snapshot.positions_raw.copy())

    def report(self) -> None:
        with self._lock:
            snapshot = self._latest
            durations = np.asarray(self._durations) * 1000.0
            first = self._first_query_at
        if snapshot is None:
            return
        span = snapshot.query_started_at - first
        hz = (snapshot.query_sequence - 1) / span if span > 0 else 0.0
        print(
            f'[hand feedback] queries={snapshot.query_sequence} target_hz={self.poll_hz:.1f} '
            f'actual_hz={hz:.2f} query_age_ms='
            f'{(time.monotonic() - snapshot.query_finished_at) * 1000:.1f} '
            f'query_ms(mean/p95/max)={durations.mean():.1f}/'
            f'{np.percentile(durations, 95):.1f}/{durations.max():.1f}'
        )

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError('L20 feedback query still running; keep SDK connected')
            self._thread = None
