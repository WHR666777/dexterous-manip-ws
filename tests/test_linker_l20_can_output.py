from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "AnyDexRetarget" / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from output.real.drivers_linker_l20_can import (  # noqa: E402
    LinkerL20CANOutput,
    l20_qpos_to_can_slots,
)


ACTIVE_JOINT_NAMES = [
    "right_thumb_cmc_yaw",
    "right_thumb_cmc_roll",
    "right_thumb_cmc_pitch",
    "right_thumb_mcp",
    "right_index_mcp_roll",
    "right_index_mcp_pitch",
    "right_index_pip",
    "right_middle_mcp_roll",
    "right_middle_mcp_pitch",
    "right_middle_pip",
    "right_ring_mcp_roll",
    "right_ring_mcp_pitch",
    "right_ring_pip",
    "right_pinky_mcp_roll",
    "right_pinky_mcp_pitch",
    "right_pinky_pip",
]


LOWER_ALIGNED_QPOS = np.array(
    [
        0.0,
        0.13962634015954636,
        0.05235987755982989,
        0.13962634015954636,
        -0.23,
        0.15235988,
        0.0,
        -0.23,
        0.15235988,
        0.0,
        -0.23,
        0.15235988,
        0.0,
        -0.23,
        0.15235988,
        0.0,
    ],
    dtype=np.float64,
)


UPPER_ALIGNED_QPOS = np.array(
    [
        1.57,
        1.4423598775598299,
        0.8823598775598299,
        1.25,
        0.23,
        1.37235988,
        1.7415730337078652,
        0.23,
        1.37235988,
        1.7415730337078652,
        0.23,
        1.37235988,
        1.7415730337078652,
        0.23,
        1.37235988,
        1.7415730337078652,
    ],
    dtype=np.float64,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHand:
    def __init__(self, initial: np.ndarray) -> None:
        self.initial = np.asarray(initial, dtype=np.int64).copy()
        self.connected = False
        self.disconnected = False
        self.speed_commands: list[np.ndarray] = []
        self.position_commands: list[np.ndarray] = []
        self.clear_count = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_joint_positions_raw(self) -> np.ndarray:
        return self.initial.copy()

    def set_speed(self, speed) -> None:
        self.speed_commands.append(np.asarray(speed, dtype=np.int64).copy())

    def set_joint_positions_raw(self, positions) -> None:
        self.position_commands.append(np.asarray(positions, dtype=np.int64).copy())

    def clear_faults(self) -> None:
        self.clear_count += 1


class RetryDisconnectHand(FakeHand):
    def __init__(self, initial: np.ndarray) -> None:
        super().__init__(initial)
        self.disconnect_attempts = 0

    def disconnect(self) -> None:
        self.disconnect_attempts += 1
        if self.disconnect_attempts == 1:
            raise TimeoutError("receiver thread is still stopping")
        super().disconnect()


class FeedbackAndCleanupFailHand(FakeHand):
    def get_joint_positions_raw(self) -> np.ndarray:
        raise RuntimeError("feedback failed")

    def disconnect(self) -> None:
        raise TimeoutError("cleanup failed")


class InterruptingFeedbackHand(FakeHand):
    def get_joint_positions_raw(self) -> np.ndarray:
        raise KeyboardInterrupt


def test_qpos_mapping_places_each_joint_group_in_g20_slots() -> None:
    lower = l20_qpos_to_can_slots(
        LOWER_ALIGNED_QPOS,
        ACTIVE_JOINT_NAMES,
        reserved=[11, 22, 33, 44],
    )
    upper = l20_qpos_to_can_slots(
        UPPER_ALIGNED_QPOS,
        ACTIVE_JOINT_NAMES,
        reserved=[55, 66, 77, 88],
    )

    assert lower.tolist() == [
        255, 255, 255, 255, 255,
        255, 0, 0, 0, 0,
        255, 11, 22, 33, 44,
        255, 255, 255, 255, 255,
    ]
    assert upper.tolist() == [
        0, 0, 0, 0, 0,
        0, 255, 255, 255, 255,
        0, 55, 66, 77, 88,
        0, 0, 0, 0, 0,
    ]


def test_first_can_command_is_slew_limited_from_feedback() -> None:
    initial = np.full(20, 100, dtype=np.int64)
    initial[11:15] = [201, 202, 203, 204]
    hand = FakeHand(initial)
    clock = FakeClock()
    output = LinkerL20CANOutput(
        hand_factory=lambda **_: hand,
        speed=30,
        command_hz=20.0,
        max_step=2,
        clock=clock,
    )

    output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)

    assert hand.connected
    assert len(hand.speed_commands) == 1
    assert hand.speed_commands[0].tolist() == [30, 30, 30, 30, 30]
    assert len(hand.position_commands) == 1
    command = hand.position_commands[0]
    assert np.max(np.abs(command[:11] - initial[:11])) <= 2
    assert np.max(np.abs(command[15:] - initial[15:])) <= 2
    assert command[11:15].tolist() == [201, 202, 203, 204]


def test_can_output_respects_configured_command_frequency() -> None:
    hand = FakeHand(np.full(20, 100, dtype=np.int64))
    clock = FakeClock()
    output = LinkerL20CANOutput(
        hand_factory=lambda **_: hand,
        command_hz=50.0,
        max_step=2,
        max_acceleration=10000.0,
        clock=clock,
    )

    output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
    output.send(LOWER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
    assert len(hand.position_commands) == 1

    clock.advance(0.019)
    output.send(LOWER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
    assert len(hand.position_commands) == 1

    clock.advance(0.001)
    output.send(LOWER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
    assert len(hand.position_commands) == 2


def test_can_trajectory_accelerates_without_exceeding_hard_step() -> None:
    initial = np.full(20, 100, dtype=np.int64)
    hand = FakeHand(initial)
    clock = FakeClock()
    output = LinkerL20CANOutput(
        hand_factory=lambda **_: hand,
        command_hz=30.0,
        max_step=4,
        max_velocity=120.0,
        max_acceleration=600.0,
        deadband=0.0,
        clock=clock,
    )

    for _ in range(8):
        output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
        clock.advance(1.0 / 30.0)

    commands = np.asarray(hand.position_commands)
    positive_steps = np.diff(np.r_[100, commands[:, 6]])
    assert positive_steps[:4].tolist() == [1, 1, 2, 3]
    assert np.all(positive_steps >= 0)
    assert np.max(positive_steps) <= 4


def test_can_trajectory_brakes_before_reversing() -> None:
    initial = np.full(20, 100, dtype=np.int64)
    hand = FakeHand(initial)
    clock = FakeClock()
    output = LinkerL20CANOutput(
        hand_factory=lambda **_: hand,
        command_hz=30.0,
        max_step=4,
        max_velocity=120.0,
        max_acceleration=600.0,
        deadband=0.0,
        clock=clock,
    )

    for _ in range(6):
        output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
        clock.advance(1.0 / 30.0)
    reversal_start = len(hand.position_commands)
    for _ in range(10):
        output.send(LOWER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
        clock.advance(1.0 / 30.0)

    values = np.asarray(hand.position_commands)[:, 6]
    reversal_values = values[reversal_start:]
    assert reversal_values.size > 2
    # 目标反向后不会立即把 +4 跳成 -4，而是先减速，再平滑反向。
    steps = np.diff(np.r_[values[reversal_start - 1], reversal_values])
    assert steps[0] >= 0
    assert np.any(steps < 0)
    assert np.max(np.abs(steps)) <= 4


def test_can_hold_cancels_pending_trajectory_velocity() -> None:
    initial = np.full(20, 100, dtype=np.int64)
    hand = FakeHand(initial)
    clock = FakeClock()
    output = LinkerL20CANOutput(
        hand_factory=lambda **_: hand,
        command_hz=30.0,
        max_step=4,
        clock=clock,
    )

    for _ in range(4):
        output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)
        clock.advance(1.0 / 30.0)
    held = hand.position_commands[-1].copy()
    output.hold()
    output.send(UPPER_ALIGNED_QPOS, ACTIVE_JOINT_NAMES)

    assert np.max(np.abs(hand.position_commands[-1] - held)) <= 1


@pytest.mark.parametrize("command_hz", [0.0, -1.0, 0.5, 60.1])
def test_can_output_rejects_unsafe_command_frequency(command_hz: float) -> None:
    with pytest.raises(ValueError, match="command_hz"):
        LinkerL20CANOutput(dry_run=True, command_hz=command_hz)


@pytest.mark.parametrize("command_hz", [True, None, "20"])
def test_can_output_rejects_non_numeric_command_frequency(command_hz) -> None:
    with pytest.raises(ValueError, match="command_hz"):
        LinkerL20CANOutput(dry_run=True, command_hz=command_hz)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_velocity", 0.0),
        ("max_velocity", np.inf),
        ("max_acceleration", 0.0),
        ("max_acceleration", np.nan),
        ("deadband", -0.1),
        ("deadband", 10.1),
    ],
)
def test_can_output_rejects_invalid_trajectory_limits(keyword: str, value) -> None:
    with pytest.raises(ValueError, match=keyword):
        LinkerL20CANOutput(dry_run=True, **{keyword: value})


def test_can_output_disconnects_owned_hand() -> None:
    hand = FakeHand(np.full(20, 100, dtype=np.int64))
    output = LinkerL20CANOutput(hand_factory=lambda **_: hand)

    output.close()

    assert hand.disconnected


def test_can_output_allows_disconnect_retry_after_cleanup_timeout() -> None:
    hand = RetryDisconnectHand(np.full(20, 100, dtype=np.int64))
    output = LinkerL20CANOutput(hand_factory=lambda **_: hand)

    with pytest.raises(TimeoutError, match="still stopping"):
        output.close()
    output.close()

    assert hand.disconnect_attempts == 2
    assert hand.disconnected


def test_startup_cleanup_error_does_not_mask_feedback_error() -> None:
    hand = FeedbackAndCleanupFailHand(np.full(20, 100, dtype=np.int64))

    with pytest.raises(RuntimeError, match="feedback failed") as exc_info:
        LinkerL20CANOutput(hand_factory=lambda **_: hand)

    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "cleanup failed" in str(exc_info.value.__cause__)


def test_keyboard_interrupt_during_startup_disconnects_hand() -> None:
    hand = InterruptingFeedbackHand(np.full(20, 100, dtype=np.int64))

    with pytest.raises(KeyboardInterrupt):
        LinkerL20CANOutput(hand_factory=lambda **_: hand)

    assert hand.disconnected


def test_teleop_cli_exposes_l20_can_controls(capsys) -> None:
    sys.modules.setdefault(
        "anydexretarget",
        types.SimpleNamespace(Retargeter=object),
    )
    import teleop_real

    with pytest.raises(SystemExit) as exc_info:
        teleop_real.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--l20-transport {rs485,can}" in help_text
    assert "--l20-can-channel L20_CAN_CHANNEL" in help_text
    assert "--l20-can-command-hz L20_CAN_COMMAND_HZ" in help_text
    assert "--l20-can-max-step L20_CAN_MAX_STEP" in help_text
    assert "--l20-can-max-velocity L20_CAN_MAX_VELOCITY" in help_text
    assert "--l20-can-max-acceleration L20_CAN_MAX_ACCELERATION" in help_text
    assert "--l20-can-deadband L20_CAN_DEADBAND" in help_text
    assert "--l20-can-input-timeout L20_CAN_INPUT_TIMEOUT" in help_text
