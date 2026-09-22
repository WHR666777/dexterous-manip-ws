import numpy as np

from anydexretarget.linker_l20 import (
    L20_ACTIVE_JOINT_ORDER,
    L20_JOINT_LIMITS,
    L20_SLOT_MAP,
    l20_qpos_to_can_slots,
    slew_l20_command,
)


def test_public_mapping_shape_reserved_and_directions():
    names = [f"right_{name.lower()}" for name in L20_ACTIVE_JOINT_ORDER]
    command = l20_qpos_to_can_slots(np.zeros(len(names)), names)
    assert command.shape == (20,)
    assert command.dtype == np.int64
    np.testing.assert_array_equal(command[11:15], 255)
    assert np.all((command >= 0) & (command <= 255))


def test_public_slew_preserves_reserved_slots():
    previous = np.zeros(20, dtype=np.int64)
    target = np.full(20, 255, dtype=np.int64)
    result = slew_l20_command(target, previous, max_step=10)
    np.testing.assert_array_equal(result[:11], 10)
    np.testing.assert_array_equal(result[11:15], 255)
    np.testing.assert_array_equal(result[15:], 10)


def test_public_mapping_endpoints_follow_can_slot_directions():
    names = [f"left_{name.lower()}" for name in L20_ACTIVE_JOINT_ORDER]
    lower = np.asarray([L20_JOINT_LIMITS[name][0] for name in L20_ACTIVE_JOINT_ORDER])
    upper = np.asarray([L20_JOINT_LIMITS[name][1] for name in L20_ACTIVE_JOINT_ORDER])
    lower_command = l20_qpos_to_can_slots(lower, names)
    upper_command = l20_qpos_to_can_slots(upper, names)
    for slot, _name, direction in L20_SLOT_MAP:
        expected = (255, 0) if direction == "decreasing" else (0, 255)
        assert (lower_command[slot], upper_command[slot]) == expected
