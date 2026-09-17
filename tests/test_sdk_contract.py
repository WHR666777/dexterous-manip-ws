import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKER_SDK = ROOT / "linkerhand-python-sdk"


def test_nero_v111_public_contract_exists():
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    assert ArmModel.NERO == "nero"
    assert NeroFW.V111 == "v111"
    assert callable(create_agx_arm_config)
    assert callable(AgxArmFactory.create_arm)

    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v111.driver import Driver

    expected = {
        "connect", "disconnect", "enable", "disable", "reset",
        "electronic_emergency_stop", "get_joint_angles", "get_motor_states",
        "get_flange_pose", "get_tcp_pose", "get_firmware", "get_arm_status", "move_j",
        "move_p", "move_l", "set_speed_percent", "set_joint_limits_enabled",
    }
    assert expected <= set(dir(Driver))
    assert "timeout" not in inspect.signature(Driver.enable).parameters
    assert "get_ik_joint_angles" not in Driver.__dict__


def test_physical_l20_uses_g20_public_contract():
    sys.path.insert(0, str(LINKER_SDK))
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
        from LinkerHand.core.can.linker_hand_g20_can import LinkerHandG20Can
    finally:
        sys.path.remove(str(LINKER_SDK))

    expected_api = {
        "finger_move", "get_state", "get_state_for_pub", "set_speed",
        "get_speed", "set_torque", "get_torque", "get_temperature",
        "get_fault", "clear_faults",
    }
    assert expected_api <= set(dir(LinkerHandApi))
    assert hasattr(LinkerHandG20Can, "close_can_interface")


def test_g20_driver_maps_twenty_slots_to_five_finger_position_frames():
    sys.path.insert(0, str(LINKER_SDK))
    try:
        from LinkerHand.core.can.linker_hand_g20_can import LinkerHandG20Can
    finally:
        sys.path.remove(str(LINKER_SDK))

    driver = LinkerHandG20Can.__new__(LinkerHandG20Can)
    sent = []
    driver.send_command = lambda frame, data: sent.append((frame.value, list(data)))
    driver.set_joint_positions(list(range(20)))

    assert sent == [
        (0x41, [10, 5, 0, 11, 12, 15]),
        (0x42, [6, 11, 1, 13, 14, 16]),
        (0x43, [7, 12, 2, 13, 14, 17]),
        (0x44, [8, 13, 3, 14, 15, 18]),
        (0x45, [9, 14, 4, 15, 16, 19]),
    ]


def test_g20_driver_maps_five_speed_values_to_finger_frames():
    sys.path.insert(0, str(LINKER_SDK))
    try:
        from LinkerHand.core.can.linker_hand_g20_can import LinkerHandG20Can
    finally:
        sys.path.remove(str(LINKER_SDK))

    driver = LinkerHandG20Can.__new__(LinkerHandG20Can)
    sent = []
    driver.send_command = lambda frame, data: sent.append((frame.value, list(data)))
    driver.set_speed([1, 2, 3, 4, 5])

    assert sent == [
        (0x49, [1] * 6),
        (0x4A, [2] * 6),
        (0x4B, [3] * 6),
        (0x4C, [4] * 6),
        (0x4D, [5] * 6),
    ]


def test_l20_send_command_swallows_can_error_without_resending():
    """固定 SDK 的发送失败边界必须保持为显式的上游限制。"""
    package_root = LINKER_SDK / "LinkerHand"
    sys.path[:0] = [str(LINKER_SDK), str(package_root)]
    try:
        from LinkerHand.core.can.linker_hand_g20_can import LinkerHandG20Can
    finally:
        sys.path.remove(str(package_root))
        sys.path.remove(str(LINKER_SDK))

    source = inspect.getsource(LinkerHandG20Can.send_command)
    assert "except can.CanError" in source
    assert "raise" not in source
    assert source.count("self.bus.send(msg)") == 1
