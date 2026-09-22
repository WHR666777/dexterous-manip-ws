import numpy as np
import pytest

from teleop.coordinate_frames import anydex_wrist_to_base
from teleop import controller
from teleop.target_gate import TargetGate
from teleop.wrist_tracker import WristTracker


def test_anydex_wrist_receives_only_site_calibration():
    position = np.array([1.0, 2.0, 3.0])
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    converted_position, converted_quaternion = anydex_wrist_to_base(
        position, quaternion, np.eye(3)
    )
    np.testing.assert_allclose(converted_position, position)
    np.testing.assert_allclose(converted_quaternion, quaternion)


def test_wrist_tracker_preserves_reference_ema_semantics():
    tracker = WristTracker(np.zeros(3), np.array([0, 0, 0, 1.0]), ema_alpha=0.8)
    tracker.update(np.zeros(3), np.array([0, 0, 0, 1.0]))
    first, _ = tracker.update(np.array([1.0, 0, 0]), np.array([0, 0, 0, 1.0]))
    second, _ = tracker.update(np.array([0.0, 0, 0]), np.array([0, 0, 0, 1.0]))
    np.testing.assert_allclose(first, [1.0, 0, 0])
    np.testing.assert_allclose(second, [0.2, 0, 0])


def test_target_gate_latches_until_reanchor():
    gate = TargetGate(0.01, 5.0, 0.30)
    gate.set_tcp_workspace_center(np.zeros(3))
    origin = np.eye(4)
    gate.reanchor(origin)
    small = origin.copy()
    small[0, 3] = 0.005
    assert gate.accept(small, small[:3, 3])
    large = small.copy()
    large[0, 3] += 0.02
    assert not gate.accept(large, large[:3, 3])
    assert gate.paused
    assert not gate.accept(small, small[:3, 3])
    gate.reanchor(small)
    assert not gate.paused


def test_target_gate_pauses_outside_tcp_workspace_cube():
    gate = TargetGate(0.20, 5.0, 0.30)
    origin = np.eye(4)
    gate.set_tcp_workspace_center(np.array([0.4, 0.0, 0.2]))
    gate.reanchor(origin)

    inside = np.array([0.549, 0.0, 0.2])
    assert gate.accept(origin, inside)

    outside = np.array([0.551, 0.0, 0.2])
    assert not gate.accept(origin, outside)
    assert gate.paused
    assert "TCP workspace cube exceeded on x" in gate.reason


class _ResetArm:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.calls: list[str] = []

    def connect(self) -> None:
        self.calls.append("connect")

    def is_enabled(self) -> bool:
        self.calls.append("is_enabled")
        return self.enabled

    def disable(self) -> None:
        self.calls.append("disable")
        self.enabled = False

    def reset(self) -> None:
        self.calls.append("reset")

    def disconnect(self) -> None:
        self.calls.append("disconnect")


@pytest.mark.parametrize("initially_enabled", (True, False))
def test_reset_arm_disables_before_reset_and_disconnects(monkeypatch, initially_enabled):
    arm = _ResetArm(enabled=initially_enabled)
    monkeypatch.setattr(controller, "_make_arm", lambda config: arm)

    controller.reset_arm({})

    assert arm.calls[0] == "connect"
    assert arm.calls[-1] == "disconnect"
    assert "reset" in arm.calls
    if initially_enabled:
        assert arm.calls.index("disable") < arm.calls.index("reset")
    else:
        assert "disable" not in arm.calls
