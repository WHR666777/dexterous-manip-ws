from types import SimpleNamespace

import numpy as np
import pytest

from robot_control.nero import NeroArm


class Message:
    def __init__(self, msg, timestamp=123.0, hz=100.0):
        self.msg = msg
        self.timestamp = timestamp
        self.hz = hz


class FakeNeroDriver:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.enable_after = 1
        self.enable_calls = 0
        self.disable_calls = 0
        self.sent_joints = []
        self.sent_poses = []
        self.sent_linear = []
        self.speed = None
        self.limits_enabled = False
        self.joint_limits = {
            "joint1": [-2.705261, 2.705261],
            "joint2": [-1.745330, 1.745330],
            "joint3": [-2.757621, 2.757621],
            "joint4": [-1.012291, 2.146755],
            "joint5": [-2.757621, 2.757621],
            "joint6": [-0.733039, 0.959932],
            "joint7": [-1.570797, 1.570797],
        }
        self.q = [0.0] * 7
        self.tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.flange = [0.1, 0.2, 0.3, 0.0, 0.1, 0.2]
        self.status = SimpleNamespace(
            ctrl_mode=1, arm_status=0, mode_feedback=1, teach_status=0,
            motion_status=0, trajectory_num=0,
            err_status=SimpleNamespace(joint_1_angle_limit=False),
        )

    def get_config(self):
        return {"joint_limits": self.joint_limits}

    def set_joint_limits_enabled(self, enabled):
        self.limits_enabled = enabled

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def is_ok(self):
        return self.connected

    def enable(self):
        self.enable_calls += 1
        if self.enable_calls >= self.enable_after:
            self.enabled = True
        return self.enabled

    def disable(self):
        self.disable_calls += 1
        self.enabled = False
        return True

    def get_joint_enable_status(self, index):
        return self.enabled

    def get_joint_angles(self):
        return Message(list(self.q))

    def get_motor_states(self, index):
        return Message(SimpleNamespace(
            position=self.q[index - 1], velocity=0.0,
            current=0.0, torque=self.tau[index - 1],
        ))

    def get_flange_pose(self):
        return Message(list(self.flange))

    def get_tcp_pose(self):
        return Message(list(self.flange))

    def get_arm_status(self):
        return Message(self.status)

    def get_firmware(self):
        return {"software_version": "1.11"}

    def set_speed_percent(self, speed):
        self.speed = speed

    def move_j(self, joints):
        self.sent_joints.append(list(joints))

    def move_p(self, pose):
        self.sent_poses.append(list(pose))

    def move_l(self, pose):
        self.sent_linear.append(list(pose))

    def electronic_emergency_stop(self):
        self.estopped = True

    def reset(self):
        self.reset_called = True


def make_connected_arm(enabled=True, **kwargs):
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver, **kwargs)
    arm.connect()
    driver.enabled = enabled
    return arm, driver


def test_constructor_enables_official_software_joint_limits():
    driver = FakeNeroDriver()
    NeroArm(driver=driver)
    assert driver.limits_enabled is True


def test_real_factory_is_called_with_v111(monkeypatch):
    calls = {}
    driver = FakeNeroDriver()
    symbols = SimpleNamespace(
        ArmModel=SimpleNamespace(NERO="nero"),
        NeroFW=SimpleNamespace(V111="v111"),
        create_agx_arm_config=lambda **kwargs: calls.setdefault("config", kwargs),
        AgxArmFactory=SimpleNamespace(create_arm=lambda config: driver),
    )
    monkeypatch.setattr("robot_control.nero._load_nero_sdk", lambda: symbols)
    NeroArm(can_interface="socketcan", can_channel="can7")
    assert calls["config"] == {
        "robot": "nero", "firmeware_version": "v111",
        "interface": "socketcan", "channel": "can7",
    }


def test_connect_disconnect_are_idempotent():
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver)
    arm.connect()
    arm.connect()
    assert arm.is_connected()
    arm.disconnect()
    arm.disconnect()
    assert not arm.is_connected()


def test_enable_retries_until_success():
    arm, driver = make_connected_arm(enabled=False)
    driver.enable_after = 3
    arm.enable(timeout=0.2, poll_interval=0.001)
    assert driver.enable_calls == 3
    assert arm.is_enabled()


def test_enable_timeout_is_finite():
    arm, driver = make_connected_arm(enabled=False)
    driver.enable_after = 10_000
    with pytest.raises(TimeoutError, match="Nero enable timed out"):
        arm.enable(timeout=0.01, poll_interval=0.001)


def test_disable_timeout_is_finite():
    arm, driver = make_connected_arm(enabled=True)
    driver.disable = lambda: False
    with pytest.raises(TimeoutError, match="Nero disable timed out"):
        arm.disable(timeout=0.01, poll_interval=0.001)


@pytest.mark.parametrize("timeout, poll_interval", [
    (np.nan, 0.01),
    (0.01, np.inf),
    ("soon", 0.01),
])
def test_lifecycle_rejects_nonfinite_or_nonnumeric_retry_settings(timeout, poll_interval):
    arm, _ = make_connected_arm(enabled=False)
    with pytest.raises(ValueError):
        arm.enable(timeout=timeout, poll_interval=poll_interval)


def test_commands_require_connection_and_enable():
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver)
    with pytest.raises(RuntimeError, match="not connected"):
        arm.command_joint_positions([0.0] * 7)
    arm.connect()
    with pytest.raises(RuntimeError, match="not enabled"):
        arm.command_joint_positions([0.0] * 7)


@pytest.mark.parametrize("bad", [
    [0.0] * 6,
    [[0.0] * 7],
    [0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
])
def test_joint_validation_rejects_bad_shape_or_nonfinite(bad):
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError):
        arm.validate_joint_command(bad)


def test_joint_validation_rejects_official_limit_violation():
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError, match="joint 1"):
        arm.validate_joint_command([2.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_joint_validation_rejects_excessive_delta():
    arm, _ = make_connected_arm(max_joint_delta=0.05)
    with pytest.raises(ValueError, match="max_joint_delta"):
        arm.validate_joint_command([0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_explicit_delta_override_replaces_instance_delta_for_one_command():
    arm, _ = make_connected_arm(max_joint_delta=0.05)
    result = arm.validate_joint_command(
        [0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], max_joint_delta=0.1,
    )
    assert np.array_equal(result, np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_state_arrays_have_verified_shapes_units_and_copies(monkeypatch):
    arm, driver = make_connected_arm()
    assert arm.get_joint_positions().shape == (7,)
    assert arm.get_joint_torques().shape == (7,)
    assert arm.get_flange_pose().shape == (6,)
    assert arm.get_tcp_pose().shape == (6,)
    monkeypatch.setattr("robot_control.nero.time.time", lambda: 456.0)
    observation = arm.get_observation()
    assert set(observation) == {"joint_position", "joint_torque", "tcp_pose", "timestamp"}
    assert observation["timestamp"] == 456.0
    assert arm.get_state_vector().shape == (7,)
    assert "joint_velocity" not in observation
    q = arm.get_joint_positions()
    q[0] = 99.0
    assert driver.q[0] == 0.0


def test_unavailable_feedback_raises_instead_of_returning_fabricated_arrays():
    arm, driver = make_connected_arm()
    driver.get_joint_angles = lambda: None
    with pytest.raises(RuntimeError, match="unavailable"):
        arm.get_joint_positions()


def test_reported_firmware_is_plain_data():
    arm, _ = make_connected_arm()
    firmware = arm.get_firmware()
    assert firmware == {"software_version": "1.11"}
    assert isinstance(firmware, dict)


def test_unavailable_firmware_feedback_raises_runtime_error():
    arm, driver = make_connected_arm()
    driver.get_firmware = lambda: None
    with pytest.raises(RuntimeError, match="unavailable"):
        arm.get_firmware()


def test_raw_methods_return_sdk_messages():
    arm, _ = make_connected_arm()
    assert isinstance(arm.get_raw_joint_positions(), Message)
    assert len(arm.get_raw_motor_states()) == 7
    assert isinstance(arm.get_raw_flange_pose(), Message)
    assert isinstance(arm.get_raw_arm_status(), Message)


def test_motion_methods_map_to_verified_sdk_calls():
    arm, driver = make_connected_arm()
    arm.command_joint_positions([0.01] * 7)
    arm.move_joints([0.02] * 7, speed_percent=10)
    arm.move_pose([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], speed_percent=9)
    arm.move_linear([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], speed_percent=8)
    assert driver.sent_joints == [[0.01] * 7, [0.02] * 7]
    assert driver.sent_poses == [[0.1, 0.2, 0.3, 0.0, 0.1, 0.2]]
    assert driver.sent_linear == [[0.1, 0.2, 0.3, 0.0, 0.1, 0.2]]
    assert driver.speed == 8


def test_invalid_motion_target_does_not_change_the_sdk_speed_setting():
    arm, driver = make_connected_arm()
    with pytest.raises(ValueError, match="joint 1"):
        arm.move_joints([2.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], speed_percent=10)
    assert driver.speed is None


@pytest.mark.parametrize("bad_speed", [True, -1, 101, 1.5])
def test_motion_rejects_noninteger_or_out_of_range_speed(bad_speed):
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError, match="speed_percent"):
        arm.move_joints([0.01] * 7, speed_percent=bad_speed)


@pytest.mark.parametrize("bad_pose", [
    [0.0] * 5,
    [0.0, 0.0, 0.0, np.pi + 0.01, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, np.pi / 2 + 0.01, 0.0],
])
def test_pose_motion_rejects_invalid_shape_or_orientation(bad_pose):
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError):
        arm.move_pose(bad_pose)


def test_status_and_safety_methods_map_without_message_leakage():
    arm, driver = make_connected_arm()
    status = arm.get_arm_status()
    assert status["arm_status"] == 0
    assert status["err_status"]["joint_1_angle_limit"] is False
    arm.emergency_stop()
    arm.reset()
    assert driver.estopped is True
    assert driver.reset_called is True
