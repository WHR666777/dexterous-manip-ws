"""示例命令行的无硬件安全行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

import config
from examples import l20_example, nero_example, nero_l20_example


def test_example_configuration_is_v111_and_uses_separate_buses():
    """配置消费者获得固定固件和隔离的 CAN 总线。"""
    assert config.NERO_FIRMWARE == "1.11"
    assert config.NERO_CAN_INTERFACE == "socketcan"
    assert config.NERO_CAN_CHANNEL == "can0"
    assert config.L20_CAN_CHANNEL == "can1"
    assert config.NERO_CAN_CHANNEL != config.L20_CAN_CHANNEL
    assert config.CONTROL_HZ == 20


@pytest.mark.parametrize("module", [nero_example, l20_example, nero_l20_example])
def test_parser_defaults_to_read_only(module):
    """移除 --execute 后运行路径保持只读。"""
    args = module.build_parser().parse_args([])
    assert args.execute is False


@pytest.mark.parametrize("module", [nero_example, l20_example, nero_l20_example])
def test_execute_requires_exact_confirmation(module, monkeypatch):
    """非精确确认词不能打开动作分支。"""
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert module.confirm_execution() is False
    monkeypatch.setattr("builtins.input", lambda prompt: "EXECUTE")
    assert module.confirm_execution() is True


def test_nero_delta_has_conservative_cli_bound():
    """过大的 Nero 增量在解析阶段被拒绝。"""
    parser = nero_example.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--execute", "--delta-rad", "0.2"])


def test_l20_delta_has_conservative_cli_bound():
    """过大的 L20 原始增量在解析阶段被拒绝。"""
    parser = l20_example.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--execute", "--delta-raw", "100"])


class FakeNeroArm:
    """记录 Nero 示例所触发的公开 Wrapper 边界调用。"""

    def __init__(self, disconnect_error=None, status_error=None, **kwargs):
        self.kwargs = kwargs
        self.connected = False
        self.enable_calls = 0
        self.move_calls = []
        self.disconnect_calls = 0
        self.position = np.arange(7, dtype=np.float64) / 10.0
        self.disconnect_error = disconnect_error
        self.status_error = status_error

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def is_ok(self):
        return True

    def get_arm_status(self):
        if self.status_error is not None:
            raise self.status_error
        return {"arm_status": "ready"}

    def get_joint_positions(self):
        return self.position.copy()

    def enable(self):
        self.enable_calls += 1

    def move_joints(self, target, *, speed_percent):
        self.move_calls.append((np.asarray(target).copy(), speed_percent))
        self.position = np.asarray(target, dtype=np.float64).copy()


def test_nero_run_without_execute_only_reads_and_disconnects():
    """删除默认只读分支会导致 Nero 在未授权时使能或发送目标。"""
    arm = FakeNeroArm()
    args = nero_example.build_parser().parse_args([])

    assert nero_example.run(args, arm_factory=lambda **kwargs: arm) == 0
    assert arm.connected is False
    assert arm.enable_calls == 0
    assert arm.move_calls == []
    assert arm.disconnect_calls == 1


def test_nero_execute_moves_one_current_relative_joint_after_confirmation():
    """移除确认门或改成预设目标会改变一次 Nero 公开运动边界调用。"""
    arm = FakeNeroArm()
    args = nero_example.build_parser().parse_args(
        ["--execute", "--joint-index", "3", "--delta-rad", "0.01"],
    )

    assert nero_example.run(
        args,
        arm_factory=lambda **kwargs: arm,
        confirm=lambda: True,
    ) == 0
    assert arm.enable_calls == 1
    assert len(arm.move_calls) == 1
    target, speed = arm.move_calls[0]
    assert np.array_equal(target, np.array([0.0, 0.1, 0.2, 0.31, 0.4, 0.5, 0.6]))
    assert speed == 10
    assert arm.disconnect_calls == 1


def test_nero_execute_rejected_confirmation_never_enables_or_moves():
    """绕过 Nero 确认分支会在被拒绝的执行请求中使能或发送运动。"""
    arm = FakeNeroArm()
    args = nero_example.build_parser().parse_args(["--execute"])

    assert nero_example.run(
        args,
        arm_factory=lambda **kwargs: arm,
        confirm=lambda: False,
    ) == 0
    assert arm.enable_calls == 0
    assert arm.move_calls == []
    assert arm.disconnect_calls == 1


def test_nero_rejected_wrapper_target_returns_2_and_disconnects(capsys):
    """未处理 Nero 目标验证错误会使承诺的退出码 2 逃逸为异常。"""
    class RejectingNeroArm(FakeNeroArm):
        def move_joints(self, target, *, speed_percent):
            self.move_calls.append((np.asarray(target).copy(), speed_percent))
            raise ValueError("official joint limit rejected target")

    arm = RejectingNeroArm()
    args = nero_example.build_parser().parse_args(["--execute"])

    assert nero_example.run(
        args,
        arm_factory=lambda **kwargs: arm,
        confirm=lambda: True,
    ) == 2
    assert len(arm.move_calls) == 1
    assert arm.disconnect_calls == 1
    assert "Nero command rejected" in capsys.readouterr().err


def test_nero_cleanup_failure_returns_nonzero_and_reports_to_stderr(capsys):
    """吞掉唯一的 Nero 断开失败会误报只读运行成功。"""
    arm = FakeNeroArm(disconnect_error=RuntimeError("Nero bus teardown failed"))
    args = nero_example.build_parser().parse_args([])

    assert nero_example.run(args, arm_factory=lambda **kwargs: arm) == 1
    assert arm.disconnect_calls == 1
    assert "Nero disconnect failed" in capsys.readouterr().err


def test_nero_primary_error_is_not_masked_by_cleanup_failure(capsys):
    """Nero 读取主错误必须保留，即使随后断开也失败。"""
    arm = FakeNeroArm(
        status_error=RuntimeError("primary Nero status failure"),
        disconnect_error=RuntimeError("Nero bus teardown failed"),
    )
    args = nero_example.build_parser().parse_args([])

    with pytest.raises(RuntimeError, match="primary Nero status failure"):
        nero_example.run(args, arm_factory=lambda **kwargs: arm)
    assert arm.disconnect_calls == 1
    assert "Nero disconnect failed" in capsys.readouterr().err


class FakeL20Hand:
    """记录 L20 示例所触发的公开 Wrapper 边界调用。"""

    def __init__(self, disconnect_error=None, version_error=None, **kwargs):
        self.kwargs = kwargs
        self.connected = False
        self.position = np.arange(20, dtype=np.int64) + 20
        self.raw_commands = []
        self.disconnect_calls = 0
        self.disconnect_error = disconnect_error
        self.version_error = version_error

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def get_sdk_version(self):
        if self.version_error is not None:
            raise self.version_error
        return "fake-sdk"

    def get_joint_positions_raw(self):
        return self.position.copy()

    def get_speed(self):
        return np.zeros(5, dtype=np.int64)

    def get_current(self):
        return np.zeros(5, dtype=np.int64)

    def get_temperature(self):
        return np.zeros(5, dtype=np.int64)

    def get_fault(self):
        return np.zeros(5, dtype=np.int64)

    def set_joint_positions_raw(self, target):
        copied = np.asarray(target).copy()
        self.raw_commands.append(copied)
        self.position = copied


def test_l20_run_without_execute_only_reads_and_disconnects():
    """删除默认只读分支会导致 L20 在未授权时发送原始位置。"""
    hand = FakeL20Hand()
    args = l20_example.build_parser().parse_args([])

    assert l20_example.run(args, hand_factory=lambda **kwargs: hand) == 0
    assert hand.connected is False
    assert hand.raw_commands == []
    assert hand.disconnect_calls == 1


def test_l20_execute_changes_one_active_position_from_feedback():
    """把 L20 当前反馈替换为预设或改动多个槽位会改变发送位置。"""
    hand = FakeL20Hand()
    args = l20_example.build_parser().parse_args(
        ["--execute", "--joint-index", "15", "--delta-raw", "5"],
    )

    assert l20_example.run(
        args,
        hand_factory=lambda **kwargs: hand,
        confirm=lambda: True,
    ) == 0
    assert len(hand.raw_commands) == 1
    assert np.array_equal(
        hand.raw_commands[0],
        np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                  30, 31, 32, 33, 34, 40, 36, 37, 38, 39]),
    )
    assert hand.disconnect_calls == 1


def test_l20_execute_rejected_confirmation_never_sends_position():
    """绕过 L20 确认分支会在被拒绝的执行请求中发送位置。"""
    hand = FakeL20Hand()
    args = l20_example.build_parser().parse_args(["--execute"])

    assert l20_example.run(
        args,
        hand_factory=lambda **kwargs: hand,
        confirm=lambda: False,
    ) == 0
    assert hand.raw_commands == []
    assert hand.disconnect_calls == 1


def test_l20_cleanup_failure_returns_nonzero_and_reports_to_stderr(capsys):
    """吞掉唯一的 L20 断开失败会误报只读运行成功。"""
    hand = FakeL20Hand(disconnect_error=RuntimeError("L20 bus teardown failed"))
    args = l20_example.build_parser().parse_args([])

    assert l20_example.run(args, hand_factory=lambda **kwargs: hand) == 1
    assert hand.disconnect_calls == 1
    assert "L20 disconnect failed" in capsys.readouterr().err


def test_l20_primary_error_is_not_masked_by_cleanup_failure(capsys):
    """L20 读取主错误必须保留，即使随后断开也失败。"""
    hand = FakeL20Hand(
        version_error=RuntimeError("primary L20 version failure"),
        disconnect_error=RuntimeError("L20 bus teardown failed"),
    )
    args = l20_example.build_parser().parse_args([])

    with pytest.raises(RuntimeError, match="primary L20 version failure"):
        l20_example.run(args, hand_factory=lambda **kwargs: hand)
    assert hand.disconnect_calls == 1
    assert "L20 disconnect failed" in capsys.readouterr().err


class FakeRobotSystem:
    """记录联合示例调用的组合控制公开边界。"""

    def __init__(self, disconnect_error=None, self_check_error=None, **kwargs):
        self.kwargs = kwargs
        self.connected = False
        self.enable_calls = 0
        self.step_calls = []
        self.disconnect_calls = 0
        self.disconnect_error = disconnect_error
        self.self_check_error = self_check_error

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False
        if self.disconnect_error is not None:
            raise self.disconnect_error

    def self_check(self):
        if self.self_check_error is not None:
            raise self.self_check_error
        return {"nero": {"ok": True}, "l20": {"ok": True}}

    def get_observation(self):
        return {
            "arm": {
                "joint_position": np.arange(7, dtype=np.float64) / 10.0,
                "joint_torque": np.zeros(7, dtype=np.float64),
                "tcp_pose": np.zeros(6, dtype=np.float64),
            },
            "hand": {"joint_position_raw": np.arange(20, dtype=np.int64) + 20},
            "timestamp": 1.0,
        }

    def enable(self):
        self.enable_calls += 1

    def step(self, action):
        self.step_calls.append({key: np.asarray(value).copy() for key, value in action.items()})


def test_combined_run_without_execute_does_not_enable_or_step():
    """默认联合示例不能使能 Nero 或发送组合动作。"""
    system = FakeRobotSystem()
    args = nero_l20_example.build_parser().parse_args([])

    assert nero_l20_example.run(args, system_factory=lambda **kwargs: system) == 0
    assert system.enable_calls == 0
    assert system.step_calls == []
    assert system.disconnect_calls == 1


def test_combined_execute_steps_once_with_one_feedback_relative_hand_change():
    """重复 step、预设手势或多槽位改变都会破坏联合示例的单动作边界。"""
    system = FakeRobotSystem()
    args = nero_l20_example.build_parser().parse_args(
        ["--execute", "--hand-joint-index", "15", "--hand-delta-raw", "2",
         "--observation-cycles", "1"],
    )

    assert nero_l20_example.run(
        args,
        system_factory=lambda **kwargs: system,
        confirm=lambda: True,
    ) == 0
    assert system.enable_calls == 1
    assert len(system.step_calls) == 1
    action = system.step_calls[0]
    assert set(action) == {"arm_joint_position", "hand_joint_position"}
    assert action["arm_joint_position"].shape == (7,)
    assert action["hand_joint_position"].shape == (20,)
    assert np.array_equal(
        action["arm_joint_position"],
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    expected_hand = (np.arange(20, dtype=np.float64) + 20.0) / 255.0 * 2.0 - 1.0
    expected_hand[15] += 2.0 / 255.0 * 2.0
    assert np.allclose(action["hand_joint_position"], expected_hand)
    assert system.disconnect_calls == 1


def test_combined_execute_rejected_confirmation_never_enables_or_steps():
    """绕过联合示例确认分支会在被拒绝的执行请求中使能或 step。"""
    system = FakeRobotSystem()
    args = nero_l20_example.build_parser().parse_args(["--execute"])

    assert nero_l20_example.run(
        args,
        system_factory=lambda **kwargs: system,
        confirm=lambda: False,
    ) == 0
    assert system.enable_calls == 0
    assert system.step_calls == []
    assert system.disconnect_calls == 1


def test_combined_cleanup_failure_returns_nonzero_and_reports_to_stderr(capsys):
    """吞掉唯一的联合系统断开失败会误报只读运行成功。"""
    system = FakeRobotSystem(disconnect_error=RuntimeError("combined teardown failed"))
    args = nero_l20_example.build_parser().parse_args([])

    assert nero_l20_example.run(args, system_factory=lambda **kwargs: system) == 1
    assert system.disconnect_calls == 1
    assert "Combined system disconnect failed" in capsys.readouterr().err


def test_combined_primary_error_is_not_masked_by_cleanup_failure(capsys):
    """联合 self-check 主错误必须保留，即使随后断开也失败。"""
    system = FakeRobotSystem(
        self_check_error=RuntimeError("primary self-check failure"),
        disconnect_error=RuntimeError("combined teardown failed"),
    )
    args = nero_l20_example.build_parser().parse_args([])

    with pytest.raises(RuntimeError, match="primary self-check failure"):
        nero_l20_example.run(args, system_factory=lambda **kwargs: system)
    assert system.disconnect_calls == 1
    assert "Combined system disconnect failed" in capsys.readouterr().err
