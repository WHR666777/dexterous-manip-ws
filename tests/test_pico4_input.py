"""Offline Pico protocol and freshness checks; no headset or CAN required."""

import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop import controller
from teleop.config import load_config

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "AnyDexRetarget" / "example"
sys.path.insert(0, str(EXAMPLE_ROOT))
from input import pico4  # noqa: E402


def offline_pico():
    source = pico4.Pico4.__new__(pico4.Pico4)
    source._lock = threading.Lock()
    source._stop = threading.Event()
    source._left_hand = source._right_hand = None
    source._left_update = source._right_update = source._last_update = 0.0
    source._hand_frames = {"left": None, "right": None}
    source._source_timestamps = {}
    return source


def hand(timestamp=123):
    return {
        "isActive": 1, "timeStampNs": timestamp,
        "HandJointLocations": [
            {"p": f"{index * 0.01},1,2,0,0,0,1", "s": 15} for index in range(26)
        ],
    }


def payload(right, left=None, wrapped=False):
    data = {"Hand": {"rightHand": right, "leftHand": left or {"isActive": 0}}}
    if wrapped:
        data = {"functionName": "Tracking", "value": json.dumps(data)}
    return json.dumps(data).encode()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_complete_frame_preserves_pico_fingers_and_converts_wrist(side, wrapped):
    source = offline_pico()
    tracked = hand()
    # A positive 90-degree rotation about PXR y/up becomes RH z/up.
    tracked["HandJointLocations"][1]["p"] = "0.01,1,2,0,0.707106781,0,0.707106781"
    message = payload(tracked if side == "right" else {"isActive": 0},
                      tracked if side == "left" else None, wrapped)
    source._update_tracking(message)
    frame = source.get_hand_frame(side)
    assert frame.side == side
    np.testing.assert_allclose(frame.wrist_position, [-2.0, -0.01, 1.0])
    np.testing.assert_allclose(Rotation.from_quat(frame.wrist_quat).as_matrix(),
                               Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-8)
    expected = pico4._parse_tracking_json(message)[0 if side == "left" else 1]
    np.testing.assert_array_equal(frame.landmarks, expected)
    assert frame.landmarks.shape == (21, 3)
    assert frame.pair_skew == 0.0
    assert source.get_fingers_sample(side)[1] == frame.received_at
    frame.wrist_position.fill(99)
    frame.wrist_quat.fill(99)
    frame.landmarks.fill(99)
    assert not np.any(source.get_hand_frame(side).landmarks == 99)
    np.testing.assert_allclose(source.get_hand_frame(side).wrist_position, [-2.0, -0.01, 1.0])


@pytest.mark.parametrize("fault", ["inactive", "quaternion", "short_wrist", "missing_point",
                                   "nan", "bad_number", "wrist_status", "finger_status"])
def test_bad_tracking_invalidates_previous_complete_frame(fault):
    source = offline_pico()
    tracked = hand()
    source._update_tracking(payload(tracked))
    assert source.get_hand_frame("right") is not None
    if fault == "inactive":
        tracked["isActive"] = 0
    elif fault == "quaternion":
        tracked["HandJointLocations"][1]["p"] = "0,0,0,0,0,0,0"
    elif fault == "short_wrist":
        tracked["HandJointLocations"][1]["p"] = "0,0,0"
    elif fault == "missing_point":
        tracked["HandJointLocations"][10].pop("p")
    elif fault == "nan":
        tracked["HandJointLocations"][10]["p"] = "nan,0,0,0,0,0,1"
    elif fault == "bad_number":
        tracked["HandJointLocations"][10]["p"] = "oops,0,0,0,0,0,1"
    elif fault == "wrist_status":
        tracked["HandJointLocations"][1]["s"] = 2
    elif fault == "finger_status":
        tracked["HandJointLocations"][10]["s"] = 1
    source._update_tracking(payload(tracked))
    assert source.get_hand_frame("right") is None


def test_other_hand_and_repeated_samples_do_not_refresh_selected_hand(monkeypatch):
    config = load_config(EXAMPLE_ROOT.parents[1] / "configs/pico4_nero_l20.yaml")
    source = offline_pico()
    now = [10.0]
    monkeypatch.setattr(pico4.time, "monotonic", lambda: now[0])
    source._update_tracking(payload(hand(123), hand(456)))
    assert controller._latest_frame(source, config) is not None
    now[0] += config["pico4"]["max_age_s"] + 0.01
    source._update_tracking(payload(hand(123), hand(457)))
    assert source.get_hand_frame("right").received_at == 10.0
    assert source.get_hand_frame("left").received_at == now[0]
    assert controller._latest_frame(source, config) is None
    source._update_tracking(payload(hand(124), hand(458)))
    assert controller._latest_frame(source, config) is not None
    source._clear_hand_frames()
    assert source.get_hand_frame("right") is None


def test_both_hand_snapshot_is_complete_and_owns_its_arrays():
    source = offline_pico()
    source._update_tracking(payload(hand(123), hand(456)))
    frames = source.get_hand_frames()
    assert set(frames) == {"left", "right"}
    assert frames["left"].received_at == frames["right"].received_at
    for side, frame in frames.items():
        frame.wrist_position.fill(99)
        frame.wrist_quat.fill(99)
        frame.landmarks.fill(99)
        assert not np.any(source.get_hand_frame(side).landmarks == 99)
    source._update_tracking(payload(hand(124)))
    assert source.get_hand_frames()["left"] is None


@pytest.mark.parametrize("mode", ["relay", "direct"])
def test_receive_loops_accept_fragmented_protocol_frames(mode):
    source = offline_pico()
    message = payload(hand(), wrapped=True)
    if mode == "relay":
        packet = struct.pack("<I", 5) + b"pico4" + struct.pack("<I", len(message)) + message
        receive = source._relay_loop
    else:
        packet = (bytes([0x3F, pico4._CMD_DEVICE_STATE_JSON]) + struct.pack("<I", len(message))
                  + message + struct.pack("<Q", 123) + bytes([0xA5]))
        receive = source._direct_client_loop
    chunks = iter([packet[:8], packet[8:10], packet[10:17], packet[17:]])
    captured = []

    class Socket:
        def settimeout(self, _):
            pass

        def recv(self, _):
            chunk = next(chunks, None)
            if chunk is None:
                captured.append(source.get_hand_frame("right"))
                return b""
            return chunk

        def close(self):
            pass

    receive(Socket())
    assert captured[0] is not None
    np.testing.assert_allclose(captured[0].wrist_position, [-2, -0.01, 1])


def test_factory_selects_existing_pico_relay(monkeypatch):
    config = load_config(EXAMPLE_ROOT.parents[1] / "configs/pico4_nero_l20.yaml")
    calls = []
    monkeypatch.setattr(pico4, "Pico4", lambda **kwargs: calls.append(kwargs))
    controller._input_device(config)
    assert calls == [{"mode": "relay", "relay_host": "127.0.0.1", "relay_port": 63902,
                      "port": 63901, "broadcast_port": 29888}]


@pytest.mark.parametrize("control", ["hand", "both"])
def test_pico_tracking_loss_pauses_all_commands_until_reanchor(monkeypatch, tmp_path, control, inline_compute):
    config = load_config(EXAMPLE_ROOT.parents[1] / "configs/pico4_nero_l20.yaml")
    config["camera"]["enabled"] = False
    source = offline_pico()
    source.is_available = lambda: True
    source.stop = lambda: None
    sends, anchors = [], []
    clock = [10.0]
    cycle = [0]
    # Frames recover on cycle 3; streaming must wait for R on cycle 4.
    steps = iter([
        ("r", hand(123)), (None, hand(124)), (None, {"isActive": 0}),
        (None, hand(125)), ("r", hand(126)), (None, hand(127)), ("q", hand(128)),
    ])

    class Keyboard:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            key, tracked = next(steps)
            source._update_tracking(payload(tracked))
            cycle[0] += 1
            return key

    arm = SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, enable_normal_mode=lambda: None,
        is_enabled=lambda: True, get_joint_positions=lambda: np.zeros(7),
        get_joint_torques=lambda: np.zeros(7), get_flange_pose=lambda: np.zeros(6),
        get_tcp_pose=lambda: np.zeros(6), move_js=lambda _: sends.append((cycle[0], "arm")),
    )
    output = SimpleNamespace(
        connect=lambda: None, disconnect=lambda: None, set_speed=lambda _: None,
        get_joint_positions_raw=lambda **_: np.zeros(20, dtype=int),
        set_joint_positions_raw=lambda _: sends.append((cycle[0], "hand")),
    )
    retargeter = SimpleNamespace(
        reset=lambda _: anchors.append(cycle[0]),
        retarget=lambda _: (None, np.zeros(20, dtype=int)),
    )
    monkeypatch.setattr(controller, "_input_device", lambda _: source)
    monkeypatch.setattr(controller, "_make_arm", lambda _: arm)
    monkeypatch.setattr(controller, "_make_hand", lambda _: output)
    monkeypatch.setattr(controller, "_retargeter", lambda _: retargeter)
    monkeypatch.setattr(controller, "_move_arm_to_start_pose", lambda *_: None)
    monkeypatch.setattr(controller, "NeroIK", lambda **_: SimpleNamespace(
        forward=lambda _: np.eye(4), solve=lambda *_: np.zeros(7)))
    monkeypatch.setattr(controller, "Keyboard", Keyboard)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(controller.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    controller.run(config, control, str(tmp_path))
    expected = [(2, "arm"), (2, "hand"), (6, "arm"), (6, "hand")] if control == "both" else [
        (2, "hand"), (6, "hand")]
    assert sends == expected
    assert anchors == [1, 5]
