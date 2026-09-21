from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quest_teleop.configuration import load_config, validate_hardware_config
from quest_teleop.control import (
    ArmControlSession,
    ArmFeedback,
    VirtualArm,
    apply_sdk_joint_limits,
    dispatch_arm_proposals,
    read_arm_feedback,
)
from quest_teleop.geometry import (
    UNITY_TO_FLU,
    calibrate_base_rotation,
    relative_flange_target,
    unity_pose_to_flu,
    valid_rotation,
)
from quest_teleop.camera import RGBDFrame
from quest_teleop.cli import _common_parser
from quest_teleop.recording import EpisodeRecorder, NullRecorder
from quest_teleop.tracking import (
    QuestTrackingReceiver,
    TrackedHand,
    TrackedHead,
    TrackingSnapshot,
    WorldResetDetector,
)
from robot_control.nero_ik import NeroIK


def sdk_hand(side="Right", sequence=1, recv=None):
    recv = recv or 1_000_000_000
    pose = SimpleNamespace(x=1, y=2, z=3, qx=0, qy=0, qz=0, qw=1)
    landmarks = SimpleNamespace(points=np.zeros((21, 3)))
    return SimpleNamespace(
        side=SimpleNamespace(value=side), wrist=pose, landmarks=landmarks,
        recv_ts_ns=recv, recv_time_unix_ns=2_000_000_000,
        source_ts_ns=3, sequence_id=sequence, source_frame_seq=4,
    )


def tracked(side, transform, sequence=1, recv=1_000_000_000):
    return TrackedHand(side, transform, np.zeros((21, 3)), recv, None, None, sequence, sequence)


def test_unity_to_flu_axes_and_pose():
    np.testing.assert_array_equal(UNITY_TO_FLU @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_array_equal(UNITY_TO_FLU @ [1, 0, 0], [0, -1, 0])
    pose = unity_pose_to_flu([1, 2, 3], [0, 0, 0, 1])
    np.testing.assert_allclose(pose[:3, 3], [3, -1, 2])
    np.testing.assert_allclose(pose[:3, :3], np.eye(3))


def test_relative_mapping_zero_and_world_rotation():
    anchor = np.eye(4)
    anchor[:3, 3] = [0.1, 0.2, 0.3]
    flange = np.eye(4)
    flange[:3, 3] = [0.4, 0.5, 0.6]
    np.testing.assert_allclose(relative_flange_target(anchor, anchor, flange, np.eye(3)), flange)
    moved = anchor.copy(); moved[0, 3] += 0.1
    target = relative_flange_target(anchor, moved, flange, np.eye(3), 0.5)
    np.testing.assert_allclose(target[:3, 3], [0.45, 0.5, 0.6])


def test_guided_calibration_and_degenerate_rejection():
    rotation = calibrate_base_rotation([0, -0.1, 0], [0, 0, 0.1])
    assert valid_rotation(rotation)
    np.testing.assert_allclose(rotation @ [0, -1, 0], [1, 0, 0], atol=1e-8)
    with pytest.raises(ValueError):
        calibrate_base_rotation([0.1, 0, 0], [0.2, 0, 0])


def test_tracking_adapter_staleness_and_sequence_reset():
    receiver = QuestTrackingReceiver(autostart=False)
    receiver.ingest(sdk_hand(sequence=5, recv=1_000_000_000))
    snapshot = receiver.snapshot()
    np.testing.assert_allclose(snapshot.right.wrist_world[:3, 3], [3, -1, 2])
    assert not snapshot.valid(("right",), max_age_s=0.01)
    receiver.ingest(sdk_hand(sequence=0, recv=2_000_000_000))
    assert receiver.snapshot().sequence_reset
    receiver.clear_sequence_reset()
    assert not receiver.snapshot().sequence_reset


def test_common_head_and_wrist_jump_detected_but_head_only_is_not():
    detector = WorldResetDetector()
    identity = np.eye(4)
    first = TrackingSnapshot(tracked("left", identity), tracked("right", identity), TrackedHead(identity, 1, None, None, 1, 1))
    assert not detector.update(first)
    jump = np.eye(4); jump[0, 3] = 0.1
    common = TrackingSnapshot(tracked("left", jump, 2), tracked("right", jump, 2), TrackedHead(jump, 2, None, None, 2, 2))
    assert detector.update(common)
    detector = WorldResetDetector(); detector.update(first)
    head_only = TrackingSnapshot(tracked("left", identity, 2), tracked("right", identity, 2), TrackedHead(jump, 2, None, None, 2, 2))
    assert not detector.update(head_only)


def feedback_for(ik, q):
    flange = ik.forward(q)
    return ArmFeedback(
        q.copy(), flange, flange.copy(), np.zeros(7),
        {"arm_status": 0, "ctrl_mode": 1, "motion_status": 0, "err_status": {}},
        (True,) * 7, True,
    )


def test_arm_session_requires_anchor_and_commits_only_after_dispatch():
    ik = NeroIK()
    q = np.array([0.2, -0.3, 0.15, 1.0, -0.1, 0.2, -0.4])
    arm = VirtualArm(ik, q)
    session = ArmControlSession("right", np.eye(3), ik=ik)
    assert not session.toggle()
    assert session.reanchor(np.eye(4), feedback_for(ik, q))
    assert session.toggle()
    proposal = session.propose(np.eye(4), feedback_for(ik, q))
    assert proposal.ready
    before = session.previous_command.copy()
    dispatch_arm_proposals([(session, arm, proposal)])
    assert len(arm.sent) == 1
    np.testing.assert_allclose(session.previous_command, proposal.command_joints)
    np.testing.assert_allclose(before, proposal.command_joints)


def test_partial_dispatch_latches_every_side():
    ik = NeroIK()
    q = np.array([0.2, -0.3, 0.15, 1.0, -0.1, 0.2, -0.4])
    sessions, arms, proposals = [], [], []
    for side in ("left", "right"):
        arm = VirtualArm(ik, q)
        session = ArmControlSession(side, np.eye(3), ik=ik)
        session.reanchor(np.eye(4), feedback_for(ik, q)); session.toggle()
        sessions.append(session); arms.append(arm)
        proposals.append(session.propose(np.eye(4), feedback_for(ik, q)))
    def fail(_): raise RuntimeError("send failed")
    arms[1].move_js = fail
    with pytest.raises(RuntimeError, match="left"):
        dispatch_arm_proposals(zip(sessions, arms, proposals))
    assert all(session.fault_latched for session in sessions)


def test_no_record_has_no_filesystem_side_effect(tmp_path):
    target = tmp_path / "must_not_exist"
    recorder = NullRecorder()
    recorder.start(); recorder.append({"x": 1}); recorder.stop()
    assert not target.exists()


def test_record_flag_defaults_off():
    args = _common_parser("test", recording=True).parse_args([])
    assert args.record is False
    assert args.output is None


def test_camera_index_is_episode_local_and_never_uses_future_frame(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "unused", {}, camera_metadata={"serial": "fake"})
    recorder.recording = True  # Exercise append without importing Zarr or writing files.
    frame = lambda global_index, timestamp: RGBDFrame(
        global_index, np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.uint16),
        timestamp, timestamp + 100, float(global_index),
    )
    recorder.append({"monotonic_ns": 150}, [frame(800, 100), frame(801, 200)])
    recorder.append({"monotonic_ns": 250}, [])
    assert int(recorder.samples[0]["camera/frame_index"]) == 0
    assert int(recorder.samples[1]["camera/frame_index"]) == 1


def test_recording_and_export_end_to_end(tmp_path):
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("h5py")
    from quest_export_dataset import export_act, export_diffusion

    source = tmp_path / "canonical"
    recorder = EpisodeRecorder(source, {"test": True}, camera_metadata={"depth_scale": 0.001}, video_fps=20)
    assert recorder.start()
    base_time = 10_000
    for step in range(2):
        sample = {"monotonic_ns": np.asarray(base_time + step * 100, dtype=np.int64)}
        for side in ("left", "right"):
            sample[f"observation/{side}/arm_joint"] = np.full(7, step, dtype=np.float64)
            sample[f"observation/{side}/hand_raw"] = np.full(20, 128 + step, dtype=np.int64)
            sample[f"action/{side}/arm_joint"] = np.full(7, step + 0.5, dtype=np.float64)
            sample[f"action/{side}/hand_raw"] = np.full(20, 129 + step, dtype=np.int64)
        frame = RGBDFrame(
            900 + step,
            np.full((16, 16, 3), 30 + step, dtype=np.uint8),
            np.full((16, 16), 1000 + step, dtype=np.uint16),
            base_time + step * 100 - 1,
            1_000_000 + step,
            float(step),
        )
        recorder.append(sample, [frame])
    assert recorder.stop("test")

    root = zarr.open_group(str(source / "dataset.zarr"), mode="r")
    episode = root["episodes/000000"]
    assert episode.attrs["complete"] is True
    np.testing.assert_array_equal(episode["camera/frame_index"], [0, 1])
    assert episode["camera/depth_u16"].shape == (2, 16, 16)

    dp, act = tmp_path / "dp", tmp_path / "act"
    export_diffusion(source, dp, include_images=True)
    replay = zarr.open_group(str(dp / "replay_buffer.zarr"), mode="r")
    assert replay["data/state"].shape == (2, 54)
    assert replay["data/action"].shape == (2, 54)
    assert replay["data/realsense_depth"].shape == (2, 16, 16)
    export_act(source, act, include_images=True)
    assert (act / "episode_000000.hdf5").is_file()


def test_hardware_config_rejects_placeholders():
    config = load_config()
    with pytest.raises(ValueError, match="arm_can"):
        validate_hardware_config(config, ("right",), include_hands=False)


def test_existing_calibration_can_be_verified_without_starting_receiver(tmp_path):
    import quest_teleop.cli as cli
    path = tmp_path / "right.yaml"
    path.write_text(
        "sides:\n  right:\n    base_from_quest_rotation:\n"
        "      - [1, 0, 0]\n      - [0, 1, 0]\n      - [0, 0, 1]\n",
        encoding="utf-8",
    )
    assert cli.calibrate_main(["--side", "right", "--verify", str(path)]) == 0


def test_arm_feedback_requires_fresh_joint_state_and_flange_components():
    class FakeArm:
        def __init__(self):
            now = __import__("time").time()
            self.motors = [SimpleNamespace(timestamp=now, msg=SimpleNamespace(position=0.0, torque=0.0)) for _ in range(7)]
            self.raw_status = SimpleNamespace(timestamp=now)
            self.driver_states = [SimpleNamespace(timestamp=now) for _ in range(7)]
            self._parser = SimpleNamespace(**{
                name: SimpleNamespace(timestamp=now)
                for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
            })
            self._driver = self
        def get_raw_motor_states(self): return self.motors
        def get_raw_arm_status(self): return self.raw_status
        def get_arm_status(self): return {"arm_status": 0, "ctrl_mode": 1, "motion_status": 0, "err_status": {}}
        def get_driver_states(self, index): return self.driver_states[index - 1]
        def get_joint_enable_status(self, _index): return True
        def get_flange_pose(self): return np.zeros(6)
        def get_tcp_pose(self): return np.zeros(6)

    arm = FakeArm()
    assert read_arm_feedback(arm).fresh
    arm.driver_states[-1].timestamp -= 1.0
    assert not read_arm_feedback(arm).fresh
    arm.driver_states[-1].timestamp += 1.0
    arm._parser.end_pose_xy.timestamp -= 1.0
    assert not read_arm_feedback(arm).fresh


def test_sdk_and_urdf_joint_limits_are_intersected():
    ik = NeroIK()
    original = ik.limits.copy()
    narrowed = {
        name: [original[i, 0] + 0.01, original[i, 1] - 0.01]
        for i, name in enumerate(ik.joint_names)
    }
    apply_sdk_joint_limits(ik, {"joint_names": ik.joint_names, "joint_limits": narrowed})
    np.testing.assert_allclose(ik.limits[:, 0], original[:, 0] + 0.01)
    np.testing.assert_allclose(ik.limits[:, 1], original[:, 1] - 0.01)


@pytest.mark.parametrize("mode", ["arm", "arm_hand", "bimanual"])
def test_dry_run_entry_modes_end_to_end_without_hardware_or_files(monkeypatch, tmp_path, mode):
    import quest_teleop.cli as cli

    class FakeReceiver:
        def __init__(self): self.sequence = 0
        def snapshot(self):
            self.sequence += 1
            now = __import__("time").monotonic_ns()
            pose = np.eye(4)
            left = TrackedHand("left", pose, np.zeros((21, 3)), now, None, now, self.sequence, self.sequence)
            right = TrackedHand("right", pose, np.zeros((21, 3)), now, None, now, self.sequence, self.sequence)
            head = TrackedHead(pose, now, None, now, self.sequence, self.sequence)
            return TrackingSnapshot(left, right, head)
        def clear_sequence_reset(self): pass
        def stop(self): pass

    class FakeKeyboard:
        def __init__(self): self.keys = iter(("r", "space", "", "q"))
        def __enter__(self): return self
        def read(self): return next(self.keys)
        def __exit__(self, *_): pass

    arms, hands = [], []
    RealVirtualArm = VirtualArm
    class CaptureArm(RealVirtualArm):
        def __init__(self, ik): super().__init__(ik); arms.append(self)
    class CaptureHand:
        def __init__(self): self.raw = np.full(20, 255, dtype=np.int64); self.sent = []; hands.append(self)
        def get_joint_positions_raw(self): return self.raw.copy()
        def set_joint_positions_raw(self, command): self.raw = np.asarray(command).copy(); self.sent.append(self.raw.copy())
    class FakeMapper:
        def __init__(self, *_): self.previous = None
        def seed(self, raw): self.previous = np.asarray(raw).copy()
        def map(self, _landmarks): return np.zeros(16), np.full(20, 200, dtype=np.int64)
        def commit(self, raw): self.previous = np.asarray(raw).copy()

    monkeypatch.setattr(cli, "_tracking_from_args", lambda *_: FakeReceiver())
    monkeypatch.setattr(cli, "Keyboard", FakeKeyboard)
    monkeypatch.setattr(cli, "VirtualArm", CaptureArm)
    monkeypatch.setattr(cli, "VirtualHand", CaptureHand)
    monkeypatch.setattr(cli, "AnyDexL20Mapper", FakeMapper)
    monkeypatch.setattr(cli.time, "sleep", lambda *_: None)
    monkeypatch.chdir(tmp_path)
    args = ["--side", "right"] if mode != "bimanual" else []
    assert cli.run_teleop(mode, args) == 0
    assert len(arms) == (2 if mode == "bimanual" else 1)
    assert all(arm.sent for arm in arms)
    assert len(hands) == (0 if mode == "arm" else (2 if mode == "bimanual" else 1))
    assert all(hand.sent for hand in hands)
    assert list(tmp_path.iterdir()) == []


def test_tracking_check_entry_end_to_end(monkeypatch):
    import quest_teleop.cli as cli

    class FakeReceiver:
        def snapshot(self): return TrackingSnapshot(None, None, None)
        def stop(self): pass

    monkeypatch.setattr(cli, "_tracking_from_args", lambda *_: FakeReceiver())
    assert cli.tracking_check_main(["--duration", "0.001"]) == 0
