import numpy as np
import pytest

from teleop.coordinate_frames import anydex_wrist_to_base
from teleop import controller
from teleop.target_gate import TargetGate
from teleop.wrist_tracker import WristTracker
from teleop.hand_feedback import HandFeedbackSnapshot
from teleop.hand_retarget import HAND_TIMING_FIELDS


class FakeHandFeedbackPoller:
    """Deterministic snapshot service; worker lifecycle is tested separately."""
    def __init__(self, hand, poll_hz, max_age_s):
        self.hand, self.poll_hz, self.max_age_s = hand, poll_hz, max_age_s
        self.sequence = 0

    def start(self):
        pass

    def get_latest(self):
        self.sequence += 1
        positions = self.hand.get_joint_positions_raw(fresh=False)
        now = controller.time.monotonic()
        return HandFeedbackSnapshot(positions, now - .02, now - .003, self.sequence)

    def stop(self):
        pass

    def report(self):
        pass


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


@pytest.mark.parametrize("camera_enabled", [True, False])
@pytest.mark.parametrize("interrupt_recording", [True, False])
def test_run_records_camera_before_action_and_cleans_up(
    monkeypatch, tmp_path, camera_enabled, interrupt_recording,
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from teleop.config import load_config
    from teleop.realsense_camera import RGBDFrame

    if camera_enabled:
        pytest.importorskip("cv2")
        pytest.importorskip("h5py")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/quest3_nero_l20.yaml")
    config["camera"].update(enabled=camera_enabled, width=64, height=48)
    events = []
    quest_frame = SimpleNamespace(wrist_position=np.zeros(3), wrist_quat=np.array([0, 0, 0, 1.0]),
                                  landmarks=np.zeros((21, 3)), received_at=0.0)
    image = RGBDFrame(np.zeros((48, 64, 3), dtype=np.uint8),
                      np.full((48, 64), 1234, dtype=np.uint16),
                      10.0, 100.0, 100.0, 1, 1, "hardware_clock", "hardware_clock")

    class Camera:
        def __init__(self, cfg):
            self.metadata = {"model": "L515", "width": 64, "height": 48}

        def start(self):
            events.append("camera_start")

        def get_latest(self):
            events.append("image")
            return image

        def stop(self):
            events.append("camera_stop")

    hand = SimpleNamespace(
        connect=lambda: events.append("hand_connect"),
        disconnect=lambda: events.append("hand_disconnect"),
        set_speed=lambda _: None,
        get_joint_positions_raw=lambda **kwargs: np.zeros(20, dtype=np.int64),
        set_joint_positions_raw=lambda _: events.append("action"),
    )
    retarget_calls = 0

    def retarget(_):
        nonlocal retarget_calls
        retarget_calls += 1
        if interrupt_recording and retarget_calls == 2:
            raise RuntimeError("simulated control error")
        return None, np.zeros(20, dtype=np.int64)

    keys = iter(["r", "b", None, "s", "q"])

    class Keyboard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return next(keys)

    monkeypatch.setattr(controller, "RealSenseCamera", Camera)
    monkeypatch.setattr(controller, 'HandFeedbackPoller', FakeHandFeedbackPoller)
    monkeypatch.setattr(controller, "Keyboard", Keyboard)
    monkeypatch.setattr(controller, "_make_hand", lambda _: hand)
    monkeypatch.setattr(controller, "_retargeter", lambda _: SimpleNamespace(
        reset=lambda _: None, retarget=retarget))
    monkeypatch.setattr(controller, "_quest_input", lambda _: SimpleNamespace(
        stop=lambda: events.append("quest_stop")))
    monkeypatch.setattr(controller, "_latest_frame", lambda *_: quest_frame)
    if interrupt_recording:
        with pytest.raises(RuntimeError, match="simulated control error"):
            controller.run(config, "hand", str(tmp_path))
    else:
        controller.run(config, "hand", str(tmp_path))
    path = next(tmp_path.glob("*/trajectory.npz"))
    expected_samples = 1 if interrupt_recording else 2
    with np.load(path) as data:
        assert len(data["hand_action_raw"]) == expected_samples
        if camera_enabled:
            assert data["camera_frame_index"].tolist() == list(range(expected_samples))
        else:
            assert "camera_frame_index" not in data.files
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["sample_hz"] == 20
    assert "hand_disconnect" in events and "quest_stop" in events
    if camera_enabled:
        assert events.index("camera_start") < events.index("hand_connect")
        assert events.index("image") < events.index("action")
        assert events.count("camera_stop") == 1
        assert metadata["camera"]["frames"] == expected_samples
    else:
        assert not any(event.startswith("camera") for event in events)
        assert not (path.parent / "camera").exists()


@pytest.mark.parametrize("control", ["arm", "hand", "both"])
@pytest.mark.parametrize("delay_scale", [1.0, 0.1])
@pytest.mark.parametrize("input_device", ["quest3", "pico4"])
def test_control_timing_measures_stages_without_changing_dispatch_or_sleep(
    monkeypatch, tmp_path, capsys, control, delay_scale, input_device,
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from teleop.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / f"configs/{input_device}_nero_l20.yaml")
    config["camera"]["enabled"] = False
    clock = [0.0]
    loop_calls = []
    current_calls = None
    sleeps = []
    delays = {
        "arm_feedback": 0.002, "ik": 0.075, "retarget": 0.100,
        "arm_send": 0.015, "hand_send": 0.016,
        "arm_torque": 0.003, "arm_tcp": 0.004, "hand_feedback": 0.005,
    }

    def call(name, result=None):
        if current_calls is not None:
            current_calls.append(name)
        clock[0] += delays.get(name, 0.0) * delay_scale
        return result

    def sleep(duration):
        sleeps.append(duration)
        clock[0] += duration

    arm = SimpleNamespace(
        connect=lambda: None, enable_normal_mode=lambda: None, is_enabled=lambda: True,
        disconnect=lambda: None,
        get_joint_positions=lambda: call("arm_feedback", np.zeros(7)),
        get_joint_torques=lambda: call("arm_torque", np.zeros(7)),
        get_tcp_pose=lambda: call("arm_tcp", np.zeros(6)),
        get_flange_pose=lambda: np.zeros(6),
        move_js=lambda _: call("arm_send"),
    )
    hand = SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, set_speed=lambda _: None,
        get_joint_positions_raw=lambda **_: call("hand_feedback", np.zeros(20, dtype=int)),
        set_joint_positions_raw=lambda _: call("hand_send"),
    )
    ik = SimpleNamespace(
        forward=lambda _: np.eye(4), solve=lambda *_: call("ik", np.zeros(7)),
    )
    retargeter = SimpleNamespace(
        reset=lambda _: None,
        retarget=lambda _: call("retarget", (None, np.full(20, 42, dtype=int))),
        last_diagnostics={**dict.fromkeys(HAND_TIMING_FIELDS, .001),
                          'hand_retarget_numevals': 17},
    )
    frame = SimpleNamespace(
        wrist_position=np.zeros(3), wrist_quat=np.array([0, 0, 0, 1.0]),
        landmarks=np.zeros((21, 3)), received_at=0.0,
    )
    # One unrecorded cycle and two recorded cycles, preceded by a long re-anchor.
    keys = iter(["r", None, "b", None, "q"])

    class Keyboard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            nonlocal current_calls
            key = next(keys)
            current_calls = None
            if key == "r":
                clock[0] += 2.8
            elif key in (None, "b"):
                current_calls = []
                loop_calls.append(current_calls)
            return key

    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(controller.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(controller.time, "sleep", sleep)
    monkeypatch.setattr(controller, "Keyboard", Keyboard)
    monkeypatch.setattr(controller, 'HandFeedbackPoller', FakeHandFeedbackPoller)
    monkeypatch.setattr(controller, "_make_arm", lambda _: arm)
    monkeypatch.setattr(controller, "_make_hand", lambda _: hand)
    monkeypatch.setattr(controller, "NeroIK", lambda **_: ik)
    monkeypatch.setattr(controller, "_move_arm_to_start_pose", lambda *_: None)
    monkeypatch.setattr(controller, "_retargeter", lambda _: retargeter)
    monkeypatch.setattr(controller, "_input_device", lambda _: SimpleNamespace(
        stop=lambda: None, is_available=lambda: True))
    monkeypatch.setattr(controller, "_latest_frame", lambda *_: frame)

    controller.run(config, control, str(tmp_path))

    has_arm = control in ("arm", "both")
    has_hand = control in ("hand", "both")
    expected_order = (["arm_feedback", "ik"] if has_arm else [])
    expected_order += (["retarget"] if has_hand else [])
    expected_order += (["arm_send"] if has_arm else [])
    expected_order += (["hand_send"] if has_hand else [])
    expected_order += (["arm_feedback", "arm_torque", "arm_tcp"] if has_arm else [])
    expected_order += (["hand_feedback"] if has_hand else [])
    assert loop_calls == [expected_order] * 3

    expected = {
        "timing_ik_s": 0.075 * has_arm * delay_scale,
        "timing_hand_retarget_s": 0.100 * has_hand * delay_scale,
        "timing_arm_send_s": 0.015 * has_arm * delay_scale,
        "timing_hand_send_s": 0.016 * has_hand * delay_scale,
        "timing_seed_feedback_s": 0.002 * has_arm * delay_scale,
        "timing_feedback_s": (0.009 * has_arm + 0.005 * has_hand) * delay_scale,
    }
    with np.load(next(tmp_path.glob("*/trajectory.npz"))) as data:
        for key, duration in expected.items():
            np.testing.assert_allclose(data[key], [duration] * 2, atol=1e-12)
        if has_hand:
            np.testing.assert_array_equal(data["hand_action_raw"], np.full((2, 20), 42))
            np.testing.assert_array_equal(data['hand_retarget_numevals'], [17, 17])
            for key in HAND_TIMING_FIELDS:
                np.testing.assert_allclose(data[key], [.001, .001])
            np.testing.assert_allclose(data['hand_feedback_query_age_s'], [.003, .003])
            np.testing.assert_allclose(
                data['hand_feedback_snapshot_read_at'] - data['hand_feedback_query_finished_at'],
                data['hand_feedback_query_age_s'],
            )
        else:
            np.testing.assert_array_equal(data['hand_retarget_numevals'], [-1, -1])
            assert np.isnan(data['hand_feedback_query_finished_at']).all()
    metadata = json.loads(next(tmp_path.glob("*/metadata.json")).read_text())
    assert metadata["input_device"] == input_device
    assert metadata["input_side"] == metadata["quest_side"] == "right"
    work_s = sum(expected.values())
    if work_s < 0.05:
        np.testing.assert_allclose(sleeps, [0.05 - work_s] * 3, atol=1e-12)
    else:
        assert sleeps == []
    output = capsys.readouterr().out
    assert "[control timing] cycles=3" in output
    assert f"active_hz={1.0 / max(work_s, 0.05):.2f}" in output
    assert f"overruns={3 if work_s > 0.05 else 0}/3" in output
    assert "ms(mean/p95/max):" in output
    if has_hand:
        assert '[hand retarget] cycles=3 numevals(mean/p95/max)=17.0/17.0/17' in output
