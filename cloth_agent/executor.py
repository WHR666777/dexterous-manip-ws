"""Single command owner. Programs supervise motions through their handles."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition, Event, RLock, Thread
import time
import uuid

import numpy as np

from .backends import hand_vector
from .planning import MotionPlan


class MotionInterrupted(RuntimeError):
    pass


class MotionFailed(RuntimeError):
    pass


@dataclass
class Motion:
    plan: MotionPlan
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    elapsed: float = 0.
    reason: str = ""
    done: Event = field(default_factory=Event)
    finished_at: float | None = None

    def snapshot(self):
        return {"motion_id": self.id, "arm": self.plan.arm, "status": self.status,
                "elapsed": self.elapsed, "duration": self.plan.duration, "reason": self.reason}


class Executor:
    def __init__(self, backend, config, on_event=None):
        self.backend, self.config = backend, config
        self.period = 1 / config["control_hz"]
        self.on_event = on_event or (lambda event: None)
        self._condition = Condition(RLock())
        self._motions = {}
        self._active = None
        self._holds = {arm: backend.get_joints(arm) for arm in config["arms"]}
        self._pause_requested = False
        self._pause_ack = Event()
        self._hand_requests = []
        self._closing = False
        self._fatal = None
        self._thread = Thread(target=self._run, name="cloth-executor", daemon=True)
        self._thread.start()

    def start(self, plan: MotionPlan):
        with self._condition:
            if self._closing or self._fatal:
                raise MotionFailed(self._fatal or "Executor is closed.")
            if self._active is not None:
                raise RuntimeError("Only one arm motion may run at a time.")
            q = self.backend.get_joints(plan.arm)
            if np.max(np.abs(q - plan.positions[0])) > self.config["motion"]["joint_tolerance_rad"]:
                raise ValueError("Trajectory start differs from current feedback; replan from current state.")
            motion = Motion(plan)
            self._motions[motion.id] = motion
            self._active = motion
            self._pause_requested = False
            self.on_event({"type": "motion_started", **motion.snapshot()})
            return motion.snapshot()

    def status(self, motion_id):
        with self._condition:
            return self._motions[motion_id].snapshot()

    def active(self):
        with self._condition:
            return self._active.snapshot() if self._active else None

    def latest(self):
        with self._condition:
            return next(reversed(self._motions.values())).snapshot() if self._motions else None

    def wait(self, motion_id, timeout=None):
        with self._condition:
            motion = self._motions[motion_id]
        if not motion.done.wait(timeout):
            raise TimeoutError("Timed out waiting for motion result.")
        if motion.status == "paused":
            raise MotionInterrupted(motion.reason)
        if motion.status != "completed":
            raise MotionFailed(motion.reason)
        return motion.snapshot()

    def pause_and_hold(self, motion_id=None, reason="program requested pause"):
        with self._condition:
            if motion_id is not None:
                motion = self._motions[motion_id]
                if motion.status != "running":
                    return motion.snapshot()
            if self._fatal:
                raise MotionFailed(self._fatal)
            if self._closing:
                raise MotionFailed("Executor is closing.")
            self._pause_ack.clear()
            self._pause_requested = reason
        if not self._pause_ack.wait(max(2., self.period * 4)):
            raise MotionFailed("Executor did not acknowledge hold.")
        with self._condition:
            if self._fatal:
                raise MotionFailed(self._fatal)
            return self._motions[motion_id].snapshot() if motion_id else {"status": "holding"}

    def set_hand(self, arm, pose):
        pose = hand_vector(pose)
        request = {"arm": arm, "pose": pose, "done": Event(), "error": None}
        with self._condition:
            if self._closing or self._fatal:
                raise MotionFailed(self._fatal or "Executor is closed.")
            if self._active and self._active.plan.arm != arm:
                raise RuntimeError("The other arm must retain its grasp during motion.")
            self._hand_requests.append(request)
        if not request["done"].wait(max(2., self.period * 4)):
            raise MotionFailed("Executor did not acknowledge hand command.")
        if request["error"]:
            raise MotionFailed(request["error"])

    def _finish(self, motion, status, reason=""):
        motion.status, motion.reason = status, reason
        motion.finished_at = time.monotonic()
        motion.done.set()
        self._active = None
        self.on_event({"type": "motion_" + status, **motion.snapshot()})

    def _tick(self, delta):
        with self._condition:
            if self._pause_requested:
                reason = str(self._pause_requested)
                for arm in self._holds:
                    self._holds[arm] = self.backend.get_joints(arm)
                if self._active:
                    self._finish(self._active, "paused", reason)
                self._pause_requested = False
            motion = self._active
            if motion:
                motion.elapsed = min(motion.elapsed + delta,
                                     motion.plan.duration + self.config["motion"]["completion_timeout_s"] + delta)
                q = motion.plan.sample(motion.elapsed)
                self.backend.send_joints(motion.plan.arm, q)
                self._holds[motion.plan.arm] = q.copy()
                if motion.elapsed >= motion.plan.duration:
                    error = np.max(np.abs(self.backend.get_joints(motion.plan.arm)
                                          - motion.plan.positions[-1]))
                    if error <= self.config["motion"]["joint_tolerance_rad"]:
                        self._finish(motion, "completed")
                    elif motion.elapsed > motion.plan.duration + self.config["motion"]["completion_timeout_s"]:
                        self._holds[motion.plan.arm] = self.backend.get_joints(motion.plan.arm)
                        self._finish(motion, "failed", "Motion arrival timed out.")
            for arm, target in self._holds.items():
                if self._active is None or self._active.plan.arm != arm:
                    self.backend.send_joints(arm, target)
            for request in self._hand_requests:
                try:
                    self.backend.send_hand(request["arm"], request["pose"])
                except Exception as exc:
                    request["error"] = str(exc)
                request["done"].set()
            self._hand_requests.clear()
            self._pause_ack.set()

    def _run(self):
        previous = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                # Delayed ticks extend the motion; they never burst to catch up.
                self._tick(min(self.period, max(0., now - previous)))
                previous = now
                with self._condition:
                    if self._closing:
                        return
                time.sleep(max(0., self.period - (time.monotonic() - now)))
        except Exception as exc:
            with self._condition:
                self._fatal = f"Executor failed: {exc}"
                # Best effort holds use feedback, never the remaining path.
                hold_errors = []
                for arm in self._holds:
                    try:
                        self.backend.send_joints(arm, self.backend.get_joints(arm))
                    except Exception as hold_error:
                        hold_errors.append(f"{arm}: {hold_error}")
                if hold_errors:
                    self._fatal += "; hold failed: " + "; ".join(hold_errors)
                if self._active:
                    self._finish(self._active, "failed", self._fatal)
                for request in self._hand_requests:
                    request["error"] = self._fatal
                    request["done"].set()
                self._pause_ack.set()
                self.on_event({"type": "executor_failed", "reason": self._fatal})

    def close(self):
        if self._thread.is_alive():
            self.pause_and_hold(reason="executor closing")
            with self._condition:
                self._closing = True
            self._thread.join(timeout=2.)
        if self._thread.is_alive():
            raise RuntimeError("Executor did not stop.")
