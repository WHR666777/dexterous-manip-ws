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


def test_l20_public_contract_and_source_limitations_exist():
    sys.path.insert(0, str(LINKER_SDK))
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
        from LinkerHand.core.can.linker_hand_l20_can import LinkerHandL20Can
    finally:
        sys.path.remove(str(LINKER_SDK))

    expected_api = {
        "finger_move", "get_state", "get_state_for_pub", "set_speed",
        "get_speed", "set_current", "get_current", "get_temperature",
        "get_fault", "clear_faults",
    }
    assert expected_api <= set(dir(LinkerHandApi))
    assert hasattr(LinkerHandL20Can, "close_can_interface")

    torque_source = inspect.getsource(LinkerHandL20Can.get_torque)
    version_source = inspect.getsource(LinkerHandL20Can.get_version)
    assert "[0] * 5" in torque_source
    assert "[0] * 5" in version_source
