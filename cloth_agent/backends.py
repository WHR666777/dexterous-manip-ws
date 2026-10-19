"""Fake devices and adapters to the existing hardware wrappers."""

from __future__ import annotations

from collections import deque
from threading import Lock, current_thread
import time

import numpy as np

from robot_control.nero_ik import NeroIK, pose_error
from .config import joint_vector, validate_hardware_config
from .geometry import transform


def hand_vector(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (20,) or not np.isfinite(result).all()
            or np.any(result != np.floor(result)) or np.any(result < 0) or np.any(result > 255)):
        raise ValueError("Hand pose must contain 20 integer SDK raw values in [0,255].")
    return result.astype(int)


class MockBackend:
    def __init__(self, config):
        self._lock = Lock()
        initial = joint_vector(config["mock"]["initial_joints"], "mock.initial_joints")
        self.positions = {name: initial.copy() for name in config["arms"]}
        self.hands = {name: np.full(20, 255, dtype=int) for name in config["arms"]}
        self.commands = deque(maxlen=10000)
        self.frozen = set()  # Test seam: a device can stop following its targets.
        self.closed = False

    def get_joints(self, arm):
        with self._lock:
            return self.positions[arm].copy()

    def get_hand(self, arm):
        with self._lock:
            return self.hands[arm].copy()

    def send_joints(self, arm, positions):
        q = joint_vector(positions, "command")
        with self._lock:
            self.commands.append((time.monotonic(), arm, q.copy(), current_thread().name))
            if arm not in self.frozen:
                self.positions[arm] = q.copy()

    def send_hand(self, arm, pose):
        with self._lock:
            self.hands[arm] = hand_vector(pose)

    def close(self):
        self.closed = True


class HardwareBackend:
    """Construction is explicit and happens only after the CLI execution gate."""

    def __init__(self, config):
        validate_hardware_config(config)
        from robot_control.nero import NeroArm
        from robot_control.l20 import LinkerHandL20

        self.arms, self.hands = {}, {}
        try:
            for name, settings in config["arms"].items():
                arm = NeroArm(can_channel=settings["can_channel"],
                              max_joint_delta=config["motion"]["max_ik_step_rad"])
                self.arms[name] = arm
                arm.connect()
                if not arm.is_enabled():
                    raise RuntimeError(f"{name} Nero is disabled; enable through the existing workflow.")
                deadline = time.monotonic() + 3
                while True:
                    try:
                        q = arm.get_joint_positions()
                        sdk_flange = transform(arm.get_flange_pose())
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.05)
                p_error, r_error = pose_error(NeroIK().forward(q), sdk_flange)
                if p_error > .01 or r_error > np.deg2rad(5):
                    raise RuntimeError(f"{name} FK/SDK preflight mismatch: {p_error} m, {r_error} rad.")
                hand = LinkerHandL20(hand_type=name, can_channel=settings["hand_can_channel"])
                self.hands[name] = hand
                hand.connect()
                if settings.get("hand_speed") is not None:
                    hand.set_speed(settings["hand_speed"])
        except BaseException:
            self.close()
            raise

    def get_joints(self, arm):
        return self.arms[arm].get_joint_positions()

    def get_hand(self, arm):
        return self.hands[arm].get_joint_positions_raw(fresh=False)

    def send_joints(self, arm, positions):
        self.arms[arm].move_js(positions)

    def send_hand(self, arm, pose):
        self.hands[arm].set_joint_positions_raw(hand_vector(pose))

    def close(self):
        errors = []
        for device in [*self.hands.values(), *self.arms.values()]:
            try:
                device.disconnect()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Device disconnect failed: " + "; ".join(errors))
