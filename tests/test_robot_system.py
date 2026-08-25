from collections.abc import Mapping

import numpy as np
import pytest

from robot_control.robot_system import RobotSystem


class FakeArm:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.commands = []
        self.disconnected = 0
        self.fail_disconnect = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.disconnected += 1
        if self.fail_disconnect:
            raise RuntimeError("arm disconnect failed")

    def enable(self, timeout=5.0):
        self.enabled = True

    def disable(self, timeout=5.0):
        self.enabled = False

    def is_connected(self):
        return self.connected

    def is_enabled(self):
        return self.enabled

    def is_ok(self):
        return self.connected

    def validate_joint_command(self, value):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError("bad arm action")
        return array

    def command_joint_positions(self, value):
        self.commands.append(list(value))

    def get_observation(self):
        return {
            "joint_position": np.zeros(7), "joint_torque": np.ones(7),
            "tcp_pose": np.zeros(6), "timestamp": 1.0,
        }

    def get_firmware(self):
        return {"software_version": "1.11"}

    def get_arm_status(self):
        return {"arm_status": 0}


class FakeHand:
    def __init__(self, fail_connect=False):
        self.connected = False
        self.fail_connect = fail_connect
        self.commands = []
        self.fault = np.zeros(5, dtype=np.int64)
        self.disconnected = 0
        self.fail_disconnect = False
        self.position_read_fresh_values = []

    def connect(self):
        if self.fail_connect:
            raise RuntimeError("hand failed")
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.disconnected += 1
        if self.fail_disconnect:
            raise RuntimeError("hand disconnect failed")

    def is_connected(self):
        return self.connected

    def get_joint_positions_raw(self, fresh=True):
        self.position_read_fresh_values.append(fresh)
        return np.arange(20)

    def get_fault(self):
        return self.fault.copy()

    def set_joint_positions_normalized(self, value):
        self.commands.append(list(value))


def test_connect_rolls_back_arm_when_hand_fails():
    """手连接失败时，已连接的机械臂必须回滚。"""
    arm = FakeArm()
    robot = RobotSystem(arm=arm, hand=FakeHand(fail_connect=True))
    with pytest.raises(RuntimeError, match="L20"):
        robot.connect()
    assert arm.disconnected == 1


def test_connect_reports_hand_failure_when_arm_rollback_also_fails():
    """L20 连接与 Nero 回滚均失败时，两个设备错误都必须保留。"""
    arm = FakeArm()
    arm.fail_disconnect = True
    robot = RobotSystem(arm=arm, hand=FakeHand(fail_connect=True))
    with pytest.raises(RuntimeError) as caught:
        robot.connect()
    assert "L20" in str(caught.value)
    assert "hand failed" in str(caught.value)
    assert "Nero" in str(caught.value)
    assert "arm disconnect failed" in str(caught.value)
    assert "hand failed" in str(caught.value.__cause__)
    assert arm.disconnected == 1


def test_enable_and_disable_apply_only_to_arm():
    """系统使能操作仅委托给 Nero。"""
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.enable(timeout=0.2)
    assert arm.enabled
    robot.disable(timeout=0.2)
    assert not arm.enabled


def test_disconnect_is_idempotent_for_both_devices():
    """重复断开后两个设备都保持断开。"""
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.disconnect()
    robot.disconnect()
    assert not arm.connected
    assert not hand.connected


def test_disconnect_attempts_both_and_aggregates_failures():
    """手臂均断开失败时，两个错误均应保留。"""
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    arm.fail_disconnect = True
    hand.fail_disconnect = True
    with pytest.raises(RuntimeError) as caught:
        robot.disconnect()
    assert "hand disconnect failed" in str(caught.value)
    assert "arm disconnect failed" in str(caught.value)
    assert arm.disconnected == 1
    assert hand.disconnected == 1


def connected_enabled_robot():
    """创建已连接、已使能的组合测试系统。"""
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.enable()
    return robot, arm, hand


def test_observation_has_fixed_nested_shapes(monkeypatch):
    """组合观测只保留一个获取完成时间戳和新鲜手部位置。"""
    robot, _, _ = connected_enabled_robot()
    monkeypatch.setattr("robot_control.robot_system.time.time", lambda: 789.0)
    observation = robot.get_observation()
    assert set(observation) == {"arm", "hand", "timestamp"}
    assert observation["arm"]["joint_position"].shape == (7,)
    assert observation["arm"]["joint_torque"].shape == (7,)
    assert observation["arm"]["tcp_pose"].shape == (6,)
    assert "timestamp" not in observation["arm"]
    assert observation["hand"]["joint_position_raw"].shape == (20,)
    assert observation["timestamp"] == 789.0


def test_observation_rejects_malformed_child_feedback():
    """子设备返回畸形观测时，系统边界必须给出 RuntimeError。"""
    robot, arm, _ = connected_enabled_robot()
    arm.get_observation = lambda: {"joint_position": np.zeros(7)}
    with pytest.raises(RuntimeError, match="Nero observation"):
        robot.get_observation()


def test_step_validates_both_actions_before_sending():
    """无效手部动作不得在机械臂上产生部分发送。"""
    robot, arm, hand = connected_enabled_robot()
    with pytest.raises(ValueError, match="hand_joint_position"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(19),
        })
    assert arm.commands == []
    assert hand.commands == []


def test_step_rejects_missing_or_unknown_keys():
    """动作键必须恰好为两个规范设备动作键。"""
    robot, _, _ = connected_enabled_robot()
    with pytest.raises(ValueError, match="keys"):
        robot.step({"arm_joint_position": np.zeros(7)})
    with pytest.raises(ValueError, match="keys"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
            "typo": 1,
        })


@pytest.mark.parametrize("action", [None, ["arm_joint_position"], 1])
def test_step_rejects_non_mapping_actions_as_value_error(action):
    """非映射动作不得泄漏集合或下标操作的 TypeError。"""
    robot, _, _ = connected_enabled_robot()
    with pytest.raises(ValueError, match="mapping"):
        robot.step(action)


def test_step_rejects_unorderable_unknown_keys_as_value_error():
    """异常键类型不得使错误格式化泄漏 TypeError。"""
    robot, _, _ = connected_enabled_robot()
    with pytest.raises(ValueError, match="keys"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
            object(): 1,
        })


class EnumeratesButCannotReadAction(Mapping):
    """模拟键集合正确、但读取项目失败的畸形映射。"""

    def __iter__(self):
        return iter(("arm_joint_position", "hand_joint_position"))

    def __len__(self):
        return 2

    def __getitem__(self, key):
        raise KeyError(key)


def test_step_normalizes_action_lookup_failure_to_value_error_without_send():
    """通过键检查却无法读取项目的映射不得泄漏 KeyError 或发送命令。"""
    robot, arm, hand = connected_enabled_robot()
    with pytest.raises(ValueError, match="Action"):
        robot.step(EnumeratesButCannotReadAction())
    assert arm.commands == []
    assert hand.commands == []


def test_step_normalizes_numeric_conversion_overflow_to_value_error():
    """超大数转换失败必须在发送前归一为 ValueError。"""
    robot, arm, hand = connected_enabled_robot()
    with pytest.raises(ValueError, match="hand_joint_position"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": [10 ** 10000] * 20,
        })
    assert arm.commands == []
    assert hand.commands == []


def test_step_sends_once_to_arm_then_hand():
    """验证后的完整动作必须按机械臂再手部的顺序各发送一次。"""
    robot, arm, hand = connected_enabled_robot()
    robot.step({
        "arm_joint_position": np.full(7, 0.01),
        "hand_joint_position": np.zeros(20),
    })
    assert arm.commands == [[0.01] * 7]
    assert hand.commands == [[0.0] * 20]


def test_hand_failure_reports_possible_partial_arm_send():
    """手部发送失败必须明确提示机械臂目标可能已发送。"""
    robot, arm, hand = connected_enabled_robot()

    def fail(_):
        raise RuntimeError("CAN send failed")

    hand.set_joint_positions_normalized = fail
    with pytest.raises(RuntimeError, match="may already have been sent"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
        })
    assert len(arm.commands) == 1


def test_self_check_is_read_only_structured_and_prints_summary(capsys):
    """自检只能读取状态，并返回可序列化的诊断结果。"""
    robot, arm, hand = connected_enabled_robot()
    result = robot.self_check()
    assert result["nero"]["firmware_config"] == "V111"
    assert result["nero"]["firmware_reported"] == "1.11"
    assert result["l20"]["fault"] == [0, 0, 0, 0, 0]
    assert "[OK]" in capsys.readouterr().out
    assert arm.commands == []
    assert hand.commands == []


def test_self_check_reads_fresh_l20_position_and_returns_plain_values(capsys):
    """自检必须读取新鲜 L20 位置，并把它作为普通 Python 列表返回。"""
    robot, _, hand = connected_enabled_robot()
    result = robot.self_check()
    output = capsys.readouterr().out
    assert hand.position_read_fresh_values == [True]
    assert result["l20"]["joint_position_raw"] == list(range(20))
    assert "[OK] L20 position" in output
    assert hand.commands == []


def test_self_check_reports_failed_l20_position_without_motion(capsys):
    """畸形 L20 位置反馈必须报告 FAIL，且不能导致任何运动命令。"""
    robot, arm, hand = connected_enabled_robot()
    hand.get_joint_positions_raw = lambda fresh=True: [0] * 19
    result = robot.self_check()
    output = capsys.readouterr().out
    assert result["l20"]["joint_position_raw"] is None
    assert "[FAIL] L20 position" in output
    assert arm.commands == []
    assert hand.commands == []


def test_self_check_normalizes_malformed_child_feedback_to_runtime_error():
    """自检中的畸形故障反馈必须成为语义明确的 RuntimeError。"""
    robot, _, hand = connected_enabled_robot()
    hand.get_fault = lambda: [0, 1]
    with pytest.raises(RuntimeError, match="L20 fault"):
        robot.self_check()
