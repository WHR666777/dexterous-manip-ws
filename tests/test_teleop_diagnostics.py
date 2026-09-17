"""离线验证诊断与法兰目标，不连接硬件。"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import io
import sys
import unittest
from contextlib import redirect_stdout

import teleop_nero_tracker_move_js as tracker
from teleop_nero_keyboard_move_js import EndEffectorController, keyboard_target


def test_ik_failure_explains_optimizer_and_fk():
    ik = tracker.NeroIK()
    q = np.zeros(7)
    target = ik.forward(q)
    target[0, 3] += 0.1
    with patch.object(tracker, 'least_squares', return_value=SimpleNamespace(
            success=False, x=q, message='评估次数耗尽', nfev=40)):
        assert ik.solve(target, q) is None
    assert '优化器失败' in ik.last_report['reason']
    assert '位置误差' in ik.last_report['reason']


def test_invalid_target_has_reason():
    ik = tracker.NeroIK()
    assert ik.solve(np.zeros((4, 4)), np.zeros(7)) is None
    assert '目标矩阵非法' in ik.last_report['reason']


class Arm:
    """只替代外部硬件边界，TCP 读取故意不提供。"""
    def __init__(self):
        self.sent = []
    def get_joint_positions(self):
        return np.zeros(7)
    def get_flange_pose(self):
        return np.array([0.1, 0.2, 0.3, 0., 0., 0.])
    def is_ok(self):
        return True
    is_enabled = is_connected = is_ok
    def validate_joint_command(self, q):
        return q
    def move_js(self, q):
        self.sent.append(np.asarray(q))


def test_keyboard_targets_flange_without_tcp_and_reports_send():
    arm = Arm()
    targets = []
    def solve(target, current):
        targets.append(target.copy())
        return np.full(7, 0.01)
    fk = np.eye(4)
    fk[:3, 3] = [0.1, 0.2, 0.3]
    ik = SimpleNamespace(forward=lambda q: fk, solve=solve)
    controller = EndEffectorController(arm, ik)
    target = keyboard_target('w', controller.current_flange(), .002, .01)
    assert controller.command_pose(target)
    np.testing.assert_allclose(targets[0][:3, 3], [.102, .2, .3])
    assert not arm.sent
    controller.control_tick()
    assert controller.last_sent
    assert len(arm.sent) == 1


def test_tracker_reports_first_send_drop():
    diag = tracker.TrackerDiagnostics()
    output = io.StringIO()
    with redirect_stdout(output):
        diag.finish({'move_js_called': True, 'reason': ''})
        diag.finish({'move_js_called': False, 'reason': '法兰步长超限'})
    assert '首次停止发送' in output.getvalue()


def test_real_ik_current_pose_succeeds_without_debug_output():
    ik = tracker.NeroIK()
    q = np.array([.1, .2, -.1, .5, .1, .1, .2])
    output = io.StringIO()
    with redirect_stdout(output):
        result = ik.solve(ik.forward(q), q)
    np.testing.assert_allclose(result, q, atol=1e-8)
    assert output.getvalue() == ''
    assert ik.last_report['reason'] == ''


def test_candidate_limit_and_delta_are_reported():
    ik = tracker.NeroIK()
    q = np.zeros(7)
    candidate = q.copy()
    candidate[0] = 4.
    with patch.object(tracker, 'least_squares', return_value=SimpleNamespace(
            success=True, x=candidate, message='模拟候选', nfev=1)):
        assert ik.solve(ik.forward(q), q) is None
    assert '关节限位超限' in ik.last_report['reason']
    assert '相对当前关节超过5°' in ik.last_report['reason']


def test_nonfinite_candidate_is_rejected():
    ik = tracker.NeroIK()
    q = np.zeros(7)
    with patch.object(tracker, 'least_squares', return_value=SimpleNamespace(
            success=True, x=np.full(7, np.nan), message='模拟候选', nfev=1)):
        assert ik.solve(ik.forward(q), q) is None
    assert '非有限数' in ik.last_report['reason']


def test_rejected_keyboard_target_keeps_previous_command():
    arm = Arm()
    fk = np.eye(4)
    fk[:3, 3] = [.1, .2, .3]
    ik = SimpleNamespace(forward=lambda q: fk, solve=lambda target, q: np.full(7, .01))
    controller = EndEffectorController(arm, ik)
    assert controller.command_pose(fk)
    far = fk.copy()
    far[0, 3] += .02
    assert not controller.command_pose(far)
    assert not controller.diagnostic['新目标已接受']
    controller.control_tick()
    assert controller.last_sent
    np.testing.assert_allclose(arm.sent[-1], np.full(7, .01))


def test_tracker_flange_only_sends_then_reports_loss():
    arm = Arm()
    arm.connect = arm.disconnect = arm.emergency_stop = lambda: None
    enabled_timeouts = []
    arm.enable = lambda *, timeout: enabled_timeouts.append(timeout)
    arm.get_arm_status = lambda: {'arm_status': 0}
    fk = np.eye(4)
    fk[:3, 3] = [.1, .2, .3]
    targets = []
    def solve(target, q, **kwargs):
        targets.append(target.copy())
        return np.full(7, .01)
    ik = SimpleNamespace(forward=lambda q: fk, solve=solve)
    module = SimpleNamespace(print_discovered_objects=lambda: None,
                             return_selected_devices=lambda kind: {'tracker': object()})
    output = io.StringIO()
    with patch.dict(sys.modules, {'track': SimpleNamespace(ViveTrackerModule=lambda: module)}), \
            patch.object(tracker, 'NeroArm', return_value=arm), \
            patch.object(tracker, 'NeroIK', return_value=ik), \
            patch.object(tracker, 'read_tracker_pose', side_effect=[np.eye(4), np.eye(4), None, KeyboardInterrupt]), \
            patch.object(tracker.time, 'sleep'), redirect_stdout(output):
        tracker.main([])
    assert enabled_timeouts == [5.0]
    assert len(arm.sent) == 1
    np.testing.assert_allclose(targets[0][:3, 3], [.1, .2, .3])
    assert '首次停止发送' in output.getvalue()
    assert 'Tracker 丢帧' in output.getvalue()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))


def test_pose_gate_latches_until_explicit_reanchor():
    from teleop_nero_tracker import FlangeTargetGate
    gate = FlangeTargetGate(np.eye(4), np.eye(4))
    moved = np.eye(4)
    moved[0, 3] = .2
    assert gate.target(moved) is None
    assert gate.paused
    assert gate.target(np.eye(4)) is None
    actual = np.eye(4)
    actual[0, 3] = .3
    assert not gate.reanchor(moved, actual, motion_status=1)
    assert gate.paused
    assert gate.reanchor(moved, actual, motion_status=0)
    np.testing.assert_allclose(gate.target(moved), actual, atol=1e-9)


def test_pose_gate_rotation_and_invalid_reanchor():
    from teleop_nero_tracker import FlangeTargetGate, pose_to_matrix
    gate = FlangeTargetGate(np.eye(4), np.eye(4))
    rotated = pose_to_matrix([0, 0, 0, 0, 0, np.deg2rad(10.5)])
    assert gate.target(rotated) is None
    assert gate.paused
    assert not gate.reanchor(None, np.eye(4), motion_status=0)
    assert gate.paused


def test_move_pose_keyboard_sends_once_from_actual_flange():
    from teleop_nero_keyboard_move_pose import command_key
    arm = Arm()
    arm.get_arm_status = lambda: {'arm_status': 0, 'ctrl_mode': 1}
    arm.move_pose = lambda pose, *, speed_percent: arm.sent.append((pose.copy(), speed_percent))
    assert command_key(arm, 'w', .002, np.deg2rad(1), 10)
    assert len(arm.sent) == 1
    np.testing.assert_allclose(arm.sent[0][0], [.102, .2, .3, 0, 0, 0])
    assert not command_key(arm, 'x', .002, .01, 10)
    assert len(arm.sent) == 1


def test_move_pose_keyboard_q_preserves_enable_but_exception_stops():
    import teleop_nero_keyboard_move_pose as module
    for key, expected_stop in [('quit', False), ('estop', True)]:
        arm = Arm()
        actions = []
        arm.connect = lambda: actions.append('连接')
        arm.disconnect = lambda: actions.append('断开')
        arm.emergency_stop = lambda: actions.append('急停')
        keyboard = SimpleNamespace(read=lambda: key, fd=0)
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = keyboard
        with patch.object(module, 'NeroArm', return_value=arm), \
                patch.object(module, 'TerminalKeyboard', return_value=context), \
                patch.object(module, 'prepare_arm'), patch.object(module.termios, 'tcflush'):
            module.main([])
        assert ('急停' in actions) == expected_stop
        assert actions[-1] == '断开'
