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
    def __init__(self, api=None):
        self.api = api
        self.running = True
        self.closed = False
        self.close_calls = 0
        self.close_failures_remaining = 0
        self.receive_thread = FakeReceiveThread()
        self.sent_commands = []
        self.x41, self.x42, self.x43, self.x44, self.x45 = (
            [0] * 6, [1] * 6, [2] * 6, [3] * 6, [4] * 6,
        )
        self.cached_mapping_calls = []

    def send_command(self, frame_property, data_list):
        copied = list(data_list)
        self.sent_commands.append((frame_property, copied))

    def joint_state_to_cmd_state(self, state):
        self.cached_mapping_calls.append([list(row) for row in state])
        return list(self.api.positions)

    def close_can_interface(self):
        self.close_calls += 1
        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            raise RuntimeError("close failed")
        self.closed = True


class FakeLinkerApi:
    def __init__(self):
        self.hand = FakeLowLevelHand(self)
        self.version = "3.1.1"
        self.positions = list(range(20))
        self.state_request_calls = 0
        self.cache_read_calls = 0
        self.cache_returns_none = False
        self.commands = []
        self.speed_commands = []
        self.speed = list(range(20))
        self.torque_commands = []
        self.torque = list(range(20, 40))
        self.fault = [0, 1, 2, 3, 4] * 4
        self.temperature = list(range(20, 40))

    def finger_move(self, pose):
        self.commands.append(list(pose))

    def get_state(self):
        self.state_request_calls += 1
        return list(self.positions)

    def get_state_for_pub(self):
        self.cache_read_calls += 1
        if self.cache_returns_none:
            return None
        return list(self.positions)

    def set_speed(self, speed):
        self.speed_commands.append(list(speed))

    def get_speed(self):
        return list(self.speed)

    def set_torque(self, torque):
        self.torque_commands.append(list(torque))

    def get_torque(self):
        return list(self.torque)

    def get_fault(self):
        return list(self.fault)

    def clear_faults(self):
        self.fault = [0] * 20

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
        "hand_type": "right", "hand_joint": "G20",
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


def test_disconnect_retries_failed_bus_close_after_thread_has_stopped():
    api = FakeLinkerApi()
    api.hand.close_failures_remaining = 1
    hand = LinkerHandL20(api_factory=lambda **kwargs: api, join_timeout=0.01)
    hand.connect()

    with pytest.raises(RuntimeError, match="close failed"):
        hand.disconnect()
    assert api.hand.receive_thread.is_alive() is False
    assert api.hand.close_calls == 1
    assert not hand.is_connected()
    with pytest.raises(RuntimeError, match="cleanup"):
        hand.connect()

    hand.disconnect()
    assert api.hand.close_calls == 2
    assert api.hand.closed is True
    hand.connect()
    assert hand.is_connected()


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


def test_g20_cached_position_recovers_from_sdk_missing_return_without_request():
    """直接相信 G20 的 None 返回会让缓存观测永久不可用。"""
    hand, api = connected_hand()
    api.cache_returns_none = True

    result = hand.get_cached_joint_positions_raw()

    assert result.tolist() == list(range(20))
    assert api.cache_read_calls == 1
    assert api.state_request_calls == 0
    assert api.hand.cached_mapping_calls == [[
        [0] * 6, [1] * 6, [2] * 6, [3] * 6, [4] * 6,
    ]]


def test_fresh_true_selects_request_path_without_claiming_a_new_generation():
    hand, api = connected_hand()
    expected_cached_values = list(api.positions)

    result = hand.get_joint_positions_raw(fresh=True)

    assert result.tolist() == expected_cached_values
    assert api.state_request_calls == 1
    assert api.cache_read_calls == 0


def test_invalid_sdk_position_feedback_is_not_padded_or_fabricated():
    hand, api = connected_hand()
    api.positions = []
    with pytest.raises(RuntimeError, match="position feedback"):
        hand.get_joint_positions_raw(fresh=True)


def test_g20_control_and_diagnostic_shapes_are_preserved_for_physical_l20():
    hand, api = connected_hand()
    hand.set_speed([1, 2, 3, 4, 5])
    hand.set_torque([5, 4, 3, 2, 1])
    assert hand.get_speed().tolist() == list(range(20))
    assert hand.get_torque().tolist() == list(range(20, 40))
    assert hand.get_fault().shape == (20,)
    assert hand.get_temperature().shape == (20,)
    hand.clear_faults()
    assert api.fault == [0] * 20
    assert api.speed_commands == [[1, 2, 3, 4, 5]]
    assert api.torque_commands == [[5, 4, 3, 2, 1]]
    assert hand.get_sdk_version() == "3.1.1"


def test_l20_speed_uses_g20_public_five_finger_api():
    """绕过 G20 公共 API 会退回不适用于当前硬件的旧 L20 速度帧。"""
    hand, api = connected_hand()

    hand.set_speed([1, 2, 3, 4, 5])

    assert api.speed_commands == [[1, 2, 3, 4, 5]]
    assert api.hand.sent_commands == []


def test_temperature_rejects_upstream_missing_data_sentinel():
    hand, api = connected_hand()
    api.temperature[3] = -1
    with pytest.raises(RuntimeError, match="temperature feedback"):
        hand.get_temperature()


@pytest.mark.parametrize("method,value", [
    ("set_speed", [1, 2, 3, 4]),
    ("set_torque", [1, 2, 3, 4, 256]),
])
def test_five_finger_setters_validate_exact_shape_and_range(method, value):
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


def test_unsupported_g20_current_and_embedded_version_are_not_public_methods():
    assert not hasattr(LinkerHandL20, "get_current")
    assert not hasattr(LinkerHandL20, "set_current")
    assert not hasattr(LinkerHandL20, "get_version")


def test_unapproved_normalized_feedback_helpers_are_not_public_wrapper_methods():
    assert not hasattr(LinkerHandL20, "get_joint_positions_normalized")
    assert not hasattr(LinkerHandL20, "get_cached_joint_positions_normalized")


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
    ("set_torque", [10 ** 1000] * 5),
])
def test_malformed_five_finger_inputs_are_value_error_before_connection(method, value):
    hand = LinkerHandL20(api_factory=lambda **kwargs: FakeLinkerApi())
    with pytest.raises(ValueError):
        getattr(hand, method)(value)


def test_default_factory_lazily_constructs_g20_driver_for_physical_l20(monkeypatch):
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
        "hand_type": "left", "hand_joint": "G20",
        "modbus": "None", "can": "can7",
    }]


def test_injected_factory_system_exit_is_normalized_to_device_runtime_error():
    def terminate(**kwargs):
        raise SystemExit(1)

    hand = LinkerHandL20(api_factory=terminate)
    with pytest.raises(RuntimeError, match="L20"):
        hand.connect()
    assert not hand.is_connected()


def test_default_official_partial_object_is_cleaned_when_init_exits(monkeypatch):
    class ExitingOfficialApi:
        instance = None

        def __new__(cls):
            instance = super().__new__(cls)
            cls.instance = instance
            return instance

        def __init__(self, **kwargs):
            self.hand = FakeLowLevelHand()
            raise SystemExit(1)

    monkeypatch.setattr("robot_control.l20._load_linker_api", lambda: ExitingOfficialApi)
    hand = LinkerHandL20(join_timeout=0.02)

    with pytest.raises(RuntimeError, match="L20"):
        hand.connect()

    partial = ExitingOfficialApi.instance
    assert partial is not None
    assert partial.hand.running is False
    assert partial.hand.closed is True
    assert partial.hand.receive_thread.join_timeout == 0.02
    assert not hand.is_connected()


def test_default_official_partial_object_is_cleaned_when_init_raises(monkeypatch):
    class FailingOfficialApi:
        instance = None

        def __new__(cls):
            instance = super().__new__(cls)
            cls.instance = instance
            return instance

        def __init__(self, **kwargs):
            self.hand = FakeLowLevelHand()
            raise OSError("CAN initialization failed")

    monkeypatch.setattr("robot_control.l20._load_linker_api", lambda: FailingOfficialApi)
    hand = LinkerHandL20(join_timeout=0.02)

    with pytest.raises(OSError, match="CAN initialization failed"):
        hand.connect()

    partial = FailingOfficialApi.instance
    assert partial is not None
    assert partial.hand.running is False
    assert partial.hand.closed is True
    assert partial.hand.receive_thread.join_timeout == 0.02
    assert not hand.is_connected()


def test_default_official_partial_object_is_cleaned_when_init_is_interrupted(monkeypatch):
    class InterruptedOfficialApi:
        instance = None

        def __new__(cls):
            instance = super().__new__(cls)
            cls.instance = instance
            return instance

        def __init__(self, **kwargs):
            self.hand = FakeLowLevelHand()
            raise KeyboardInterrupt

    monkeypatch.setattr("robot_control.l20._load_linker_api", lambda: InterruptedOfficialApi)
    hand = LinkerHandL20(join_timeout=0.02)

    with pytest.raises(KeyboardInterrupt):
        hand.connect()

    partial = InterruptedOfficialApi.instance
    assert partial is not None
    assert partial.hand.running is False
    assert partial.hand.closed is True
    assert partial.hand.receive_thread.join_timeout == 0.02
    assert not hand.is_connected()


def test_fresh_position_selection_is_keyword_only():
    hand, _ = connected_hand()
    with pytest.raises(TypeError):
        hand.get_joint_positions_raw(False)
