import numpy as np
import pytest

from robot_control.l20 import (
    L20_ACTIVE_POSITION_INDICES,
    L20_JOINT_NAMES,
    LinkerHandL20,
)


class FakeReceiveThread:
    def __init__(self, alive=False):
        self.join_timeout = None
        self.alive = alive

    def join(self, timeout=None):
        self.join_timeout = timeout

    def is_alive(self):
        return self.alive


class FakeLowLevelHand:
    def __init__(self):
        self.running = True
        self.closed = False
        self.close_calls = 0
        self.receive_thread = FakeReceiveThread()

    def close_can_interface(self):
        self.close_calls += 1
        self.closed = True


class FakeLinkerApi:
    def __init__(self):
        self.hand = FakeLowLevelHand()
        self.version = "3.1.1"
        self.positions = list(range(20))
        self.commands = []
        self.speed = [10, 20, 30, 40, 50]
        self.current = [50, 40, 30, 20, 10]
        self.fault = [0, 1, 2, 3, 4]
        self.temperature = list(range(20, 40))

    def finger_move(self, pose):
        self.commands.append(list(pose))

    def get_state(self):
        return list(self.positions)

    def get_state_for_pub(self):
        return list(self.positions)

    def set_speed(self, speed):
        self.speed = list(speed)

    def get_speed(self):
        return list(self.speed)

    def set_current(self, current):
        self.current = list(current)

    def get_current(self):
        return list(self.current)

    def get_fault(self):
        return list(self.fault)

    def clear_faults(self):
        self.fault = [0] * 5

    def get_temperature(self):
        return list(self.temperature)


def test_l20_joint_order_and_reserved_indices_are_fixed():
    assert len(L20_JOINT_NAMES) == 20
    assert L20_JOINT_NAMES[10] == "thumb_roll"
    assert L20_JOINT_NAMES[11:15] == (
        "reserved_11", "reserved_12", "reserved_13", "reserved_14"
    )
    assert L20_ACTIVE_POSITION_INDICES == tuple(range(11)) + tuple(range(15, 20))


def test_connect_is_lazy_and_disconnect_stops_only_the_sdk_bus():
    created = []
    api = FakeLinkerApi()
    hand = LinkerHandL20(api_factory=lambda **kwargs: created.append(kwargs) or api)
    assert not hand.is_connected()
    hand.connect()
    assert created == [{
        "hand_type": "right", "hand_joint": "L20",
        "modbus": "None", "can": "can1",
    }]
    hand.disconnect()
    assert api.hand.running is False
    assert api.hand.closed is True
    assert not hand.is_connected()


def test_disconnect_thread_timeout_is_bounded_and_retryable():
    api = FakeLinkerApi()
    api.hand.receive_thread.alive = True
    hand = LinkerHandL20(api_factory=lambda **kwargs: api, join_timeout=0.01)
    hand.connect()
    with pytest.raises(TimeoutError, match="receive thread"):
        hand.disconnect()
    assert api.hand.receive_thread.join_timeout == 0.01
    api.hand.receive_thread.alive = False
    hand.disconnect()
    assert not hand.is_connected()


def connected_hand():
    api = FakeLinkerApi()
    hand = LinkerHandL20(api_factory=lambda **kwargs: api)
    hand.connect()
    return hand, api


@pytest.mark.parametrize("bad", [
    [0] * 19,
    [0] * 21,
    [0] * 19 + [256],
    [0] * 19 + [-1],
    [0] * 19 + [1.5],
    [0] * 19 + [np.nan],
])
def test_raw_position_rejects_bad_shape_range_or_fraction(bad):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        hand.set_joint_positions_raw(bad)


@pytest.mark.parametrize("bad", [
    [0.0] * 19,
    [0.0] * 19 + [1.01],
    [0.0] * 19 + [np.inf],
])
def test_normalized_position_rejects_invalid_values(bad):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        hand.set_joint_positions_normalized(bad)


def test_normalized_position_maps_endpoints_and_midpoint_half_up():
    hand, api = connected_hand()
    action = [-1.0, 0.0, 1.0] + [0.0] * 17
    hand.set_joint_positions_normalized(action)
    assert api.commands[-1] == [0, 128, 255] + [128] * 17


def test_normalized_position_accepts_spec_action_keyword():
    hand, api = connected_hand()
    hand.set_joint_positions_normalized(action=[-1.0, 0.0, 1.0] + [0.0] * 17)
    assert api.commands[-1] == [0, 128, 255] + [128] * 17


def test_fresh_and_cached_positions_are_explicit_copies():
    hand, api = connected_hand()
    fresh = hand.get_joint_positions_raw(fresh=True)
    cached = hand.get_cached_joint_positions_raw()
    assert fresh.shape == (20,)
    assert cached.shape == (20,)
    fresh[0] = 999
    assert api.positions[0] == 0
    assert hand.get_active_joint_indices() == L20_ACTIVE_POSITION_INDICES


def test_invalid_sdk_position_feedback_is_not_padded_or_fabricated():
    hand, api = connected_hand()
    api.positions = []
    with pytest.raises(RuntimeError, match="position feedback"):
        hand.get_joint_positions_raw(fresh=True)


def test_supported_five_motor_data_and_temperature_shapes():
    hand, api = connected_hand()
    hand.set_speed([1, 2, 3, 4, 5])
    hand.set_current([5, 4, 3, 2, 1])
    assert hand.get_speed().tolist() == [1, 2, 3, 4, 5]
    assert hand.get_current().tolist() == [5, 4, 3, 2, 1]
    assert hand.get_fault().shape == (5,)
    assert hand.get_temperature().shape == (20,)
    hand.clear_faults()
    assert api.fault == [0] * 5
    assert hand.get_sdk_version() == "3.1.1"


@pytest.mark.parametrize("method,value", [
    ("set_speed", [1, 2, 3, 4]),
    ("set_current", [1, 2, 3, 4, 256]),
])
def test_five_motor_setters_validate_exact_shape_and_range(method, value):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        getattr(hand, method)(value)


def test_official_open_and_close_presets_are_used_exactly():
    hand, api = connected_hand()
    hand.open_hand()
    assert api.commands[-1] == [
        255, 255, 255, 255, 255, 255, 10, 100, 180, 240,
        245, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ]
    hand.close_hand()
    assert api.commands[-1] == [
        40, 0, 0, 0, 0, 131, 10, 100, 180, 240,
        19, 255, 255, 255, 255, 135, 0, 0, 0, 0,
    ]


def test_fake_torque_and_embedded_version_are_not_public_wrapper_methods():
    assert not hasattr(LinkerHandL20, "get_torque")
    assert not hasattr(LinkerHandL20, "set_torque")
    assert not hasattr(LinkerHandL20, "get_version")


def test_pending_cleanup_is_disconnected_rejects_connect_and_retries_only_join():
    api = FakeLinkerApi()
    api.hand.receive_thread.alive = True
    hand = LinkerHandL20(api_factory=lambda **kwargs: api, join_timeout=0.01)
    hand.connect()
    with pytest.raises(TimeoutError):
        hand.disconnect()
    assert not hand.is_connected()
    with pytest.raises(RuntimeError, match="cleanup"):
        hand.connect()
    api.hand.receive_thread.alive = False
    hand.disconnect()
    assert api.hand.close_calls == 1


@pytest.mark.parametrize("method,value", [
    ("set_speed", None),
    ("set_current", [10 ** 1000] * 5),
])
def test_malformed_five_motor_inputs_are_value_error_before_connection(method, value):
    hand = LinkerHandL20(api_factory=lambda **kwargs: FakeLinkerApi())
    with pytest.raises(ValueError):
        getattr(hand, method)(value)


def test_default_factory_lazily_loads_api_class_then_constructs_with_l20_config(monkeypatch):
    loader_calls = []
    constructor_calls = []
    api = FakeLinkerApi()

    def official_api_class(**kwargs):
        constructor_calls.append(kwargs)
        return api

    def load_official_api_class():
        loader_calls.append(True)
        return official_api_class

    monkeypatch.setattr("robot_control.l20._load_linker_api", load_official_api_class)
    hand = LinkerHandL20(hand_type="left", can_channel="can7")
    assert loader_calls == []
    hand.connect()
    hand.connect()
    assert loader_calls == [True]
    assert constructor_calls == [{
        "hand_type": "left", "hand_joint": "L20",
        "modbus": "None", "can": "can7",
    }]


def test_fresh_position_selection_is_keyword_only():
    hand, _ = connected_hand()
    with pytest.raises(TypeError):
        hand.get_joint_positions_raw(False)
