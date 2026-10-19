from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_control.l20 import _L20_OPEN_PRESET
from teleop import controller
from teleop.cli import build_parser
from teleop.config import load_config
from teleop.hand_feedback import HandFeedbackSnapshot
from teleop.hand_replay import HandReplay, THUMB_INDEX_INDICES


CLIP = Path(__file__).resolve().parents[1] / "configs/l20_thumb_index_3_4s.csv"


def test_clip_matches_video_frames_and_holds_other_fingers_open():
    replay = HandReplay(CLIP)
    assert len(replay.actions) == 20
    assert replay.duration == pytest.approx(1.0)
    np.testing.assert_array_equal(replay.actions[0, THUMB_INDEX_INDICES],
                                  [253, 156, 221, 160, 26, 255, 207])
    other = [i for i in range(20) if i not in THUMB_INDEX_INDICES]
    np.testing.assert_array_equal(replay.actions[:, other],
                                  np.tile(_L20_OPEN_PRESET, (20, 1))[:, other])


def test_playback_uses_elapsed_time_holds_last_pose_and_resets():
    replay = HandReplay(CLIP)
    np.testing.assert_array_equal(replay.target(99), replay.actions[0])
    replay.start(100)
    np.testing.assert_array_equal(replay.target(100.05), replay.actions[1])
    np.testing.assert_array_equal(replay.target(100.35), replay.actions[7])
    np.testing.assert_array_equal(replay.target(101.1), replay.actions[-1])
    np.testing.assert_array_equal(replay.target(105), replay.actions[-1])
    replay.start(110)
    np.testing.assert_array_equal(replay.target(110), replay.actions[0])
    replay.target(110.2)
    replay.stop()
    np.testing.assert_array_equal(replay.target(120), replay.actions[4])
    replay.reset()
    np.testing.assert_array_equal(replay.target(130), replay.actions[0])


@pytest.mark.parametrize("replacement", ["256", "-1", "1.5", "nan"])
def test_clip_rejects_invalid_position_targets(tmp_path, replacement):
    clip = tmp_path / "clip.csv"
    clip.write_text(CLIP.read_text().replace("3.000,253,", f"3.000,{replacement},"))
    with pytest.raises(ValueError, match="integers"):
        HandReplay(clip)


def test_cli_exposes_opt_in_mode():
    parser = build_parser()
    assert parser.parse_args(["run", "--control", "both"]).hand_mode == "tracking"
    assert parser.parse_args([
        "run", "--control", "both", "--hand-mode", "wrist-only",
    ]).hand_mode == "wrist-only"


@pytest.fixture
def wrist_run(monkeypatch, tmp_path):
    config = load_config(CLIP.parent / "pico4_nero_l20.yaml")
    config["camera"]["enabled"] = False
    config["recording"].pop("lerobot")
    clock, cycle = [0.0], [0]
    events = SimpleNamespace(hand=[], arm=[], targets=[], rows=[], opened=[], captures=[])
    last_hand = np.zeros(20, dtype=int)
    failure = [None]
    base_flange = np.eye(4)
    base_flange[:3, :3] = Rotation.from_euler(
        "xyz", [*config["arm"]["wrist_only_roll_pitch_deg"], 0], degrees=True,
    ).as_matrix()

    def send_hand(target):
        last_hand[:] = target
        events.hand.append((cycle[0], target.copy()))

    def solve(target, seed):
        events.targets.append(target.copy())
        if cycle[0] == 4 and failure[0] == "ik":
            return None
        return np.zeros(7)

    def frame(*_):
        if cycle[0] == 4 and failure[0] == "tracking":
            return None
        position = 2.0 if cycle[0] == 4 and failure[0] == "gate" else cycle[0] * .001
        return SimpleNamespace(wrist_position=np.array([position, 0, 0]),
                               wrist_quat=Rotation.from_euler(
                                   "xyz", [cycle[0] * 2, cycle[0], cycle[0] * 3], degrees=True,
                               ).as_quat(),
                               landmarks=np.full((21, 3), cycle[0]), received_at=clock[0])

    hand = SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, set_speed=lambda _: None,
        open_hand=lambda: events.opened.append(cycle[0]),
        set_thumb_index_positions_raw=send_hand,
    )
    arm = SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, enable_normal_mode=lambda: None,
        is_enabled=lambda: True, get_joint_positions=lambda: np.zeros(7),
        get_joint_torques=lambda: np.zeros(7), get_flange_pose=lambda: np.zeros(6),
        get_tcp_pose=lambda: np.zeros(6),
        move_js=lambda _: events.arm.append(cycle[0]),
    )

    class Recorder:
        def __init__(self, root, metadata, **_):
            self.metadata, self.active = metadata.copy(), False
            events.recorder = self

        def start(self): self.active = True

        def append(self, **row):
            if self.active:
                events.rows.append(row)

        def stop_and_save(self): self.active = False

    monkeypatch.setattr(controller, "_make_hand", lambda _: hand)
    monkeypatch.setattr(controller, "_make_arm", lambda _: arm)
    monkeypatch.setattr(controller, "_move_arm_to_start_pose", lambda *_: None)
    monkeypatch.setattr(controller, "NeroIK", lambda **_: SimpleNamespace(
        forward=lambda _: base_flange.copy(), solve=solve, last_diagnostics={"code": "IK_FAILED"}))
    monkeypatch.setattr(controller, "_retargeter", lambda _: pytest.fail("Finger tracking was constructed"))
    monkeypatch.setattr(controller, "EpisodeRecorder", Recorder)
    monkeypatch.setattr(controller, "HandFeedbackPoller", lambda *_, **__: SimpleNamespace(
        poll_hz=20, max_age_s=.25, start=lambda: None, stop=lambda: None, report=lambda: None,
        get_latest=lambda: HandFeedbackSnapshot(last_hand.copy(), clock[0], clock[0], 1)))
    monkeypatch.setattr(controller, "_input_device", lambda _: SimpleNamespace(
        is_available=lambda: True, stop=lambda: None))
    monkeypatch.setattr(controller, "_latest_frame", frame)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(controller.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(controller.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    def run(keys, fault=None, calibration=False, capture_failure=False):
        failure[0] = fault
        key_iter = iter(keys)

        class Keyboard:
            def __enter__(self): return self
            def __exit__(self, *_): pass

            def read(self):
                cycle[0] += 1
                clock[0] += .05
                return next(key_iter)

        monkeypatch.setattr(controller, "Keyboard", Keyboard)
        options = {"hand_mode": "wrist-only"}
        if calibration:
            from teleop import tcp_calibration

            config["camera"]["enabled"] = True
            monkeypatch.setattr(controller, "RealSenseCamera", lambda _: SimpleNamespace(
                start=lambda: None, stop=lambda: None, metadata={"model": "L515"}))

            class Session:
                def __init__(self, directory, cfg, pose, index, metadata):
                    np.testing.assert_array_equal(pose, HandReplay(CLIP).actions[-1])
                    assert index == 19

                def capture(self, *_):
                    events.captures.append(cycle[0])
                    assert cycle[0] not in events.arm
                    if capture_failure:
                        raise TimeoutError("Still moving")
                    return tmp_path / "sample.npz"

            monkeypatch.setattr(tcp_calibration, "CalibrationSession", Session)
            options = {"hand_mode": "calibration", "calibration_dir": str(tmp_path / "session")}
        controller.run(config, "both", str(tmp_path), **options)
        return events

    return run


def test_wrist_continues_during_replay_and_reset_and_recording_matches_sends(wrist_run):
    events = wrist_run([None, "r", "b", "p", None, None, "o", None, "q"])
    replay = HandReplay(CLIP)
    assert events.opened == [0]
    assert events.arm == [3, 4, 5, 6, 7, 8]
    assert [cycle for cycle, _ in events.hand] == [0, 3, 4, 5, 6, 7, 8]
    for _, target in events.hand[:3]:
        np.testing.assert_array_equal(target, replay.actions[0])
    assert not np.array_equal(events.hand[3][1], replay.actions[0])
    for _, target in events.hand[-2:]:
        np.testing.assert_array_equal(target, replay.actions[0])
    assert not np.array_equal(events.targets[0], events.targets[-1])
    rpy = Rotation.from_matrix(np.array(events.targets)[:, :3, :3]).as_euler("xyz", degrees=True)
    reference = events.recorder.metadata["fixed_roll_pitch_deg"]
    np.testing.assert_allclose(rpy[:, :2], np.tile(reference, (len(rpy), 1)), atol=1e-9)
    np.testing.assert_allclose(rpy[:, 2], [3, 6, 9, 12, 15, 18], atol=1e-9)
    assert events.recorder.metadata["hand_mode"] == "wrist-only"
    assert len(events.rows) == 6
    for row, (_, sent) in zip(events.rows, events.hand[1:]):
        np.testing.assert_array_equal(row["hand_action_raw"], sent)
        assert row["hand_retarget_numevals"] == -1


@pytest.mark.parametrize("fault", ["tracking", "ik", "gate"])
def test_pause_cancels_replay_and_reanchor_preserves_last_finger_pose(wrist_run, fault):
    events = wrist_run(["r", "p", None, None, None, "r", None, "p", None, "q"], fault)
    assert events.arm == [2, 3, 7, 8, 9]
    by_cycle = dict(events.hand)
    assert set(by_cycle) == {0, 2, 3, 7, 8, 9}
    np.testing.assert_array_equal(by_cycle[7], by_cycle[3])
    np.testing.assert_array_equal(by_cycle[8], HandReplay(CLIP).actions[0])


@pytest.mark.parametrize("capture_failure", [False, True])
def test_calibration_freezes_fingers_allows_tilts_and_stays_paused_after_capture(wrist_run, capture_failure):
    events = wrist_run(["r", None, "p", "o", "c", None, "r", None, "q"],
                       calibration=True, capture_failure=capture_failure)
    assert events.arm == [2, 3, 4, 8]
    assert events.captures == [5]
    for _, target in events.hand:
        np.testing.assert_array_equal(target, HandReplay(CLIP).actions[-1])
    rpy = Rotation.from_matrix(np.array(events.targets)[:, :3, :3]).as_euler("xyz")
    assert np.linalg.norm(rpy[2, :2] - rpy[0, :2]) > 1e-4
    assert "fixed_roll_pitch_deg" not in events.recorder.metadata
