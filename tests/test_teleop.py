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


class _DelayedFeedbackArm:
    def __init__(self, unavailable_reads: int) -> None:
        self.unavailable_reads = unavailable_reads
        self.read_calls = 0
        self.connected = False
        self.disconnected = False

    def connect(self) -> None:
        self.connected = True

    def get_joint_positions(self) -> np.ndarray:
        self.read_calls += 1
        if self.read_calls <= self.unavailable_reads:
            raise RuntimeError("Nero SDK feedback is unavailable: None")
        return np.arange(7, dtype=np.float64)

    def disconnect(self) -> None:
        self.disconnected = True


def test_record_start_pose_waits_for_complete_feedback(monkeypatch, tmp_path):
    arm = _DelayedFeedbackArm(unavailable_reads=2)
    output = tmp_path / "nero_start_pose.yaml"
    monkeypatch.setattr(controller, "_make_arm", lambda config: arm)
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    controller.record_start_pose({}, str(output))

    assert arm.read_calls == 3
    assert arm.disconnected
    assert "joint_positions_rad:\n- 0.0" in output.read_text(encoding="utf-8")


def test_record_start_pose_times_out_without_complete_feedback(monkeypatch, tmp_path):
    arm = _DelayedFeedbackArm(unavailable_reads=99)
    ticks = iter((0.0, 3.0))
    monkeypatch.setattr(controller, "_make_arm", lambda config: arm)
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="未在 3.0 s 内收到完整的七轴关节反馈"):
        controller.record_start_pose({}, str(tmp_path / "nero_start_pose.yaml"))

    assert arm.disconnected
