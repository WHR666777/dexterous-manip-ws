from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")
spec = importlib.util.spec_from_file_location(
    "replay_episode", Path(__file__).resolve().parents[1] / "scripts/replay_episode.py",
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def episode_path(tmp_path):
    camera = tmp_path / "camera"
    camera.mkdir()
    metadata = {"camera": {"depth_scale_m": 0.001, "depth_file": "camera/depth.h5",
                            "color_file": "camera/color.mp4", "depth_dataset": "depth"}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    np.savez(tmp_path / "trajectory.npz", timestamp_monotonic=[10, 10.1, 10.45, 10.5],
             camera_frame_index=[0, -1, 1, 0], camera_valid=[True, False, True, True],
             quest_landmarks=np.arange(4 * 21 * 3).reshape(4, 21, 3))
    video = cv2.VideoWriter(str(camera / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    assert video.isOpened()
    try:
        for bgr in ([230, 0, 0], [0, 230, 0]):
            video.write(np.full((48, 64, 3), bgr, dtype=np.uint8))
    finally:
        video.release()
    with h5py.File(camera / "depth.h5", "w") as handle:
        depth = np.full((2, 48, 64), 1000, dtype=np.uint16)
        depth[:, 0, 0] = 0
        depth[1, 1, 1] = 2000
        handle["depth"] = depth
    return tmp_path


def test_rgbd_indexing_backward_seek_missing_frame_and_field_pages(episode_path):
    with ExitStack() as resources:
        episode = replay.Episode(episode_path, resources)
        np.testing.assert_allclose(episode.times, [0, 0.1, 0.45, 0.5])
        images, _ = episode.images(0)
        assert images[0][10, 10, 0] > 200
        assert images[1][10, 10] == pytest.approx(1.0)
        assert episode.images(1)[0] is None
        images, _ = episode.images(2)
        assert images[0][10, 10, 1] > 200
        assert images[1][1, 1] == pytest.approx(2.0)
        assert episode.images(3)[0][0][10, 10, 0] > 200
        panel, pages = replay.render(episode, 0, 0, True, 2, 1)
        assert pages > 1
        assert panel.shape == (860, 1280, 3)
        assert "62" in "\n".join(episode.field_lines(0))  # Last of all 21 landmark triples.
        for page in range(pages):
            replay.render(episode, 0, page, True, 2, 1)
        video = episode.video
        depth_file = episode.depth.file
    assert not video.isOpened()
    assert not depth_file.id.valid


def test_numeric_only_and_empty_episodes(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    np.savez(tmp_path / "trajectory.npz", timestamp_monotonic=[1], arm_action=np.zeros((1, 7)))
    with ExitStack() as resources:
        episode = replay.Episode(tmp_path / "trajectory.npz", resources)
        assert episode.images(0) == (None, "No camera recorded")
        replay.render(episode, 0, 0, True, 2, 1)
    np.savez(tmp_path / "trajectory.npz")
    with ExitStack() as resources, pytest.raises(ValueError, match="no timestamped"):
        replay.Episode(tmp_path, resources)


def test_playback_timing_pause_step_and_window_cleanup(monkeypatch):
    clock = [0.0]
    shown = []
    keys = iter([ord(" "), ord("d"), ord("l"), ord("a"), ord(" ")])
    destroyed = []

    def render(episode, row, page, paused, *args):
        shown.append((clock[0], row, page, paused))
        return np.zeros((1, 1, 3), dtype=np.uint8), 3

    def wait_key(delay):
        clock[0] += delay / 1000
        if clock[0] > 0.65:
            return ord("q")
        return next(keys, -1)

    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay, "render", render)
    for name in ("namedWindow", "resizeWindow", "imshow"):
        monkeypatch.setattr(replay.cv2, name, lambda *args: None)
    monkeypatch.setattr(replay.cv2, "waitKey", wait_key)
    monkeypatch.setattr(replay.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(replay.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    episode = SimpleNamespace(times=np.array([0, 0.1, 0.45, 0.5]))
    replay.replay(episode, 2, 1)
    assert shown[0][1:] == (0, 0, False)
    assert any(row == 1 and paused for _, row, _, paused in shown)
    assert any(page == 1 for _, _, page, _ in shown)
    later = [(t, row) for t, row, _, paused in shown if not paused and row == 2]
    assert later[0][0] == pytest.approx(0.025 + 0.45, abs=0.01)
    assert shown[-1][1:] == (3, 1, True)
    assert destroyed == [True]


@pytest.fixture
def fake_arm(monkeypatch):
    from robot_control import nero

    class Arm:
        def __init__(self):
            self.current = np.zeros(7)
            self.enabled = True
            self.calls = []

        def connect(self):
            self.calls.append(("connect",))

        def disconnect(self):
            self.calls.append(("disconnect",))

        def get_joint_positions(self):
            return self.current.copy()

        def is_enabled(self):
            return self.enabled

        def validate_joint_command(self, target, *, max_joint_delta=None):
            assert np.shape(target) == (7,)
            if np.any(np.abs(target) > 2):
                raise ValueError("joint violates official joint limits")
            if max_joint_delta is not None and np.any(np.abs(target - self.current) > max_joint_delta):
                raise ValueError("Joint command exceeds max_joint_delta")
            self.calls.append(("validate", np.array(target), max_joint_delta))

        def move_joints(self, target, *, speed_percent):
            self.calls.append(("move_joints", np.array(target), speed_percent))
            self.current = np.array(target)

        def move_js(self, target):
            self.calls.append(("move_js", np.array(target)))
            self.current = np.array(target)

    arm = Arm()
    monkeypatch.setattr(nero, "NeroArm", lambda **kwargs: arm)
    monkeypatch.setattr(replay.time, "sleep", lambda delay: None)
    return arm


@pytest.fixture
def arm_config():
    return dict(can_interface="socketcan", can_channel="test-can", max_joint_delta_rad=0.3,
                start_speed_percent=10, start_tolerance_rad=0.02, start_timeout_s=1)


def joint_episode():
    return SimpleNamespace(times=np.array([0, 0.1, 0.45]), data={
        "arm_joint_position": np.array([[0.2] * 7, [0.3] * 7, [0.4] * 7]),
        "arm_action": np.full((3, 7), -1.5),
    })


def test_arm_preparation_uses_feedback_first_sample_and_configured_speed(fake_arm, arm_config):
    episode = joint_episode()
    with ExitStack() as resources:
        assert replay.prepare_arm(episode, arm_config, resources) is fake_arm
    calls = fake_arm.calls
    assert [call[0] for call in calls] == ["validate"] * 3 + ["connect", "move_joints", "disconnect"]
    np.testing.assert_array_equal(calls[-2][1], episode.data["arm_joint_position"][0])
    assert calls[-2][2] == 10


@pytest.mark.parametrize("invalid", ["missing", "nan", "shape", "time", "step", "limits"])
def test_invalid_joint_trajectory_never_connects_or_moves(fake_arm, arm_config, invalid):
    episode = joint_episode()
    if invalid == "missing":
        del episode.data["arm_joint_position"]
    elif invalid == "nan":
        episode.data["arm_joint_position"][1, 0] = np.nan
    elif invalid == "shape":
        episode.data["arm_joint_position"] = np.zeros((3, 6))
    elif invalid == "time":
        episode.times[1] = 0
    elif invalid == "step":
        episode.data["arm_joint_position"][1, 0] = 0.8
    elif invalid == "limits":
        episode.data["arm_joint_position"][:] = 2.1
    with ExitStack() as resources, pytest.raises(ValueError):
        replay.prepare_arm(episode, arm_config, resources)
    assert all(call[0] in ("validate", "disconnect") for call in fake_arm.calls)


def test_disabled_arm_never_moves_and_is_disconnected(fake_arm, arm_config):
    fake_arm.enabled = False
    with ExitStack() as resources, pytest.raises(RuntimeError, match="disabled"):
        replay.prepare_arm(joint_episode(), arm_config, resources)
    assert not any(call[0].startswith("move") for call in fake_arm.calls)
    assert fake_arm.calls[-1][0] == "disconnect"


@pytest.mark.parametrize("fail_send", [False, True])
def test_hardware_replay_feedback_only_no_seek_loop_or_catchup(monkeypatch, fake_arm, arm_config, fail_send):
    clock = [0.0]
    commands = []
    destroyed = []
    # Start, pause, try seeking, flip fields, resume; extra spaces at EOF must not restart.
    keys = iter([ord(" "), ord(" "), ord("d"), ord("a"), ord("l"), ord(" ")])

    def wait_key(delay):
        clock[0] += delay / 1000
        if clock[0] > 3.0:
            return ord("q")
        return next(keys, ord(" ") if len(commands) == 3 else -1)

    def render(*args):
        clock[0] += 0.16  # Simulate video decoding slower than the first 100 ms sample interval.
        return np.zeros((1, 1, 3), dtype=np.uint8), 3

    original_send = fake_arm.move_js

    def send(target):
        if fail_send:
            raise RuntimeError("simulated CAN failure")
        commands.append((clock[0], np.array(target)))
        original_send(target)

    monkeypatch.setattr(fake_arm, "move_js", send)
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay, "render", render)
    for name in ("namedWindow", "resizeWindow", "imshow"):
        monkeypatch.setattr(replay.cv2, name, lambda *args: None)
    monkeypatch.setattr(replay.cv2, "waitKey", wait_key)
    monkeypatch.setattr(replay.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(replay.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    episode = joint_episode()
    if fail_send:
        with pytest.raises(RuntimeError, match="simulated CAN failure"):
            replay.replay(episode, 2, 1, arm_config=arm_config)
    else:
        replay.replay(episode, 2, 1, arm_config=arm_config)
        np.testing.assert_array_equal([target for _, target in commands], episode.data["arm_joint_position"])
        assert np.all(np.diff([t for t, _ in commands]) >= np.diff(episode.times))
        streaming_checks = [c for c in fake_arm.calls if c[0] == "validate" and c[2] is not None]
        assert len(streaming_checks) == 3
        assert all(c[2] == arm_config["max_joint_delta_rad"] for c in streaming_checks)
    assert fake_arm.calls[-1][0] == "disconnect"
    assert destroyed == [True]


@pytest.mark.parametrize("stage", ["feedback", "start_move"])
def test_arm_startup_timeouts_disconnect(monkeypatch, fake_arm, arm_config, stage):
    ticks = iter(np.arange(0, 10, 0.2))
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(ticks))
    if stage == "feedback":
        def unavailable():
            raise RuntimeError("Nero SDK feedback is unavailable: None")
        monkeypatch.setattr(fake_arm, "get_joint_positions", unavailable)
    else:
        monkeypatch.setattr(fake_arm, "move_joints", lambda *args, **kwargs: None)
    with ExitStack() as resources, pytest.raises(TimeoutError):
        replay.prepare_arm(joint_episode(), arm_config, resources)
    assert not any(call[0] == "move_js" for call in fake_arm.calls)
    assert fake_arm.calls[-1][0] == "disconnect"


@pytest.mark.parametrize("execute", [False, True])
def test_cli_requires_execute_and_matching_confirmation(monkeypatch, episode_path, execute):
    calls = []
    args = ["replay_episode.py", str(episode_path)] + (["--execute"] if execute else [])
    monkeypatch.setattr(replay.sys, "argv", args)
    monkeypatch.setattr("builtins.input", lambda message: "NO")
    monkeypatch.setattr(replay, "replay", lambda *args, **kwargs: calls.append(kwargs))
    if execute:
        with pytest.raises(SystemExit) as error:
            replay.main()
        assert error.value.code == 1
        assert not calls
    else:
        replay.main()
        assert calls == [{"arm_config": None, "hand_config": None}]


@pytest.fixture
def hand_config():
    return dict(type="right", can_channel="test-hand-can", speed=[100, 110, 120, 130, 140])


@pytest.fixture
def fake_hand(monkeypatch):
    from robot_control import l20

    class Hand:
        def __init__(self):
            self.calls = []

        def connect(self):
            self.calls.append(("connect",))

        def disconnect(self):
            self.calls.append(("disconnect",))

        def set_speed(self, speed):
            self.calls.append(("speed", list(speed)))

        def set_joint_positions_raw(self, target):
            self.calls.append(("position", np.array(target)))

    hand = Hand()

    def factory(**kwargs):
        hand.calls.append(("construct", kwargs))
        return hand

    monkeypatch.setattr(l20, "LinkerHandL20", factory)
    return hand


def both_episode():
    episode = joint_episode()
    episode.metadata = {"quest_side": "right", "control": "both"}
    episode.data["hand_position_raw"] = np.arange(60).reshape(3, 20) + 20
    episode.data["hand_position_raw"][:, 11:15] = 255  # Preserve reserved slots as recorded.
    episode.data["hand_action_raw"] = np.zeros((3, 20), dtype=int)
    return episode


def test_hand_preparation_uses_config_and_does_not_move(fake_hand, hand_config):
    with ExitStack() as resources:
        assert replay.prepare_hand(both_episode(), hand_config, resources) is fake_hand
    assert [c[0] for c in fake_hand.calls] == ["construct", "connect", "speed", "disconnect"]
    assert fake_hand.calls[0][1] == {"hand_type": "right", "can_channel": "test-hand-can"}
    assert fake_hand.calls[2][1] == hand_config["speed"]


@pytest.mark.parametrize("invalid", ["missing", "placeholder", "nan", "fraction", "range", "shape", "side", "time", "speed"])
def test_bad_hand_recording_rejected_before_arm_motion(monkeypatch, fake_hand, arm_config, hand_config, invalid):
    episode = both_episode()
    episode.data["hand_position_raw"] = episode.data["hand_position_raw"].astype(float)
    if invalid == "missing":
        del episode.data["hand_position_raw"]
    elif invalid == "placeholder":
        episode.data["hand_position_raw"][:] = -1
    elif invalid == "nan":
        episode.data["hand_position_raw"][1, 0] = np.nan
    elif invalid == "fraction":
        episode.data["hand_position_raw"][1, 0] = 1.5
    elif invalid == "range":
        episode.data["hand_position_raw"][1, 0] = 256
    elif invalid == "shape":
        episode.data["hand_position_raw"] = np.zeros((3, 16))
    elif invalid == "side":
        episode.metadata["quest_side"] = "left"
    elif invalid == "time":
        episode.times[1] = 0
    elif invalid == "speed":
        hand_config["speed"] = [100] * 20
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(replay.cv2, name, lambda *args: None)
    monkeypatch.setattr(replay, "prepare_arm", lambda *args: pytest.fail("Arm must not be prepared"))
    with pytest.raises(ValueError):
        replay.replay(episode, 2, 1, arm_config=arm_config, hand_config=hand_config)
    assert not fake_hand.calls


@pytest.mark.parametrize("control", ["hand", "both"])
@pytest.mark.parametrize("fail_send", [False, True])
def test_hand_and_both_replay_use_feedback_and_cleanup(monkeypatch, fake_arm, fake_hand,
                                                      arm_config, hand_config, control, fail_send):
    clock = [0.0]
    keys = iter([ord(" "), ord("d"), ord("a"), ord("l")])
    destroyed = []
    original_send = fake_hand.set_joint_positions_raw
    send_count = [0]

    def send(target):
        send_count[0] += 1
        if fail_send and send_count[0] == 3:
            raise RuntimeError("simulated hand failure")
        original_send(target)

    def wait_key(delay):
        clock[0] += delay / 1000
        if clock[0] > 1:
            return ord("q")
        return next(keys, ord(" ") if clock[0] > 0.7 else -1)

    monkeypatch.setattr(fake_hand, "set_joint_positions_raw", send)
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(replay, "render", lambda *args: (np.zeros((1, 1, 3), dtype=np.uint8), 3))
    for name in ("namedWindow", "resizeWindow", "imshow"):
        monkeypatch.setattr(replay.cv2, name, lambda *args: None)
    monkeypatch.setattr(replay.cv2, "waitKey", wait_key)
    monkeypatch.setattr(replay.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(replay.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    episode = both_episode()
    selected_arm = arm_config if control == "both" else None
    if control == "hand":
        del episode.data["arm_joint_position"]  # No Nero data needed for hand-only replay.
    if fail_send:
        with pytest.raises(RuntimeError, match="simulated hand failure"):
            replay.replay(episode, 2, 1, arm_config=selected_arm, hand_config=hand_config)
    else:
        replay.replay(episode, 2, 1, arm_config=selected_arm, hand_config=hand_config)
        sent = [c[1] for c in fake_hand.calls if c[0] == "position"]
        np.testing.assert_array_equal(sent[0], episode.data["hand_position_raw"][0])
        np.testing.assert_array_equal(sent[1:], episode.data["hand_position_raw"])
        if control == "both":
            sent_arm = [c[1] for c in fake_arm.calls if c[0] == "move_js"]
            np.testing.assert_array_equal(sent_arm, episode.data["arm_joint_position"])
    assert fake_hand.calls[-1][0] == "disconnect"
    if control == "hand":
        assert not fake_arm.calls
    else:
        assert fake_arm.calls[-1][0] == "disconnect"
    assert destroyed == [True]


@pytest.mark.parametrize("control", ["arm", "hand", "both"])
def test_cli_routes_selected_hardware(monkeypatch, episode_path, control):
    calls = []
    monkeypatch.setattr(replay.sys, "argv", ["replay_episode.py", str(episode_path),
                                           "--control", control, "--execute"])
    monkeypatch.setattr("builtins.input", lambda message: "EXECUTE")
    monkeypatch.setattr(replay, "replay", lambda *args, **kwargs: calls.append(kwargs))
    replay.main()
    assert (calls[0]["arm_config"] is not None) == (control in ("arm", "both"))
    assert (calls[0]["hand_config"] is not None) == (control in ("hand", "both"))


def test_arm_start_failure_disconnects_hand_without_sending_hand_pose(monkeypatch, fake_arm, fake_hand,
                                                                     arm_config, hand_config):
    fake_arm.enabled = False
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(replay.cv2, name, lambda *args: None)
    with pytest.raises(RuntimeError, match="disabled"):
        replay.replay(both_episode(), 2, 1, arm_config=arm_config, hand_config=hand_config)
    assert fake_arm.calls[-1][0] == "disconnect"
    assert fake_hand.calls[-1][0] == "disconnect"
    assert not any(c[0] == "position" for c in fake_hand.calls)
