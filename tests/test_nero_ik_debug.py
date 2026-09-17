"""纯离线测试；FakeArm 仅替代 CAN 边界，不启动 SteamVR。"""
import importlib
import io
import csv
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class IKTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('robot_control.nero_ik')
        self.ik = self.mod.NeroIK()
        self.q = np.array([.2, -.3, .15, 1.0, -.1, .2, -.4])

    def test_zero_fk_matches_hand_checked_urdf(self):
        T = self.ik.forward(np.zeros(7))
        np.testing.assert_allclose(T[:3, 3], [0, -.0235, .71801], atol=1e-6)
        np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [0, 0, -1], [1, 0, 0]], atol=1e-6)

    def test_identical_nonzero_seed_does_not_drift_including_q7(self):
        target = self.ik.forward(self.q)
        for _ in range(5):
            result = self.ik.solve(target, q_seed=self.q)
            np.testing.assert_allclose(result, self.q, atol=1e-10)

    def test_nearby_pose_converges_and_remains_continuous(self):
        other = self.q + np.deg2rad([1, -.5, .5, -1, .5, .3, 1])
        target = self.ik.forward(other)
        q = self.ik.solve(target, q_seed=self.q)
        self.assertIsNotNone(q)
        p, r = self.mod.pose_error(self.ik.forward(q), target)
        self.assertLess(p, .002)
        self.assertLess(r, np.deg2rad(2))
        self.assertLess(np.max(np.abs(q - self.q)), np.deg2rad(10))

    def test_invalid_seed_and_target_never_produce_command(self):
        target = self.ik.forward(self.q)
        for seed in [np.zeros(6), np.full(7, np.nan), np.full(7, np.inf), np.full(7, 9), ['bad']*7]:
            self.assertIsNone(self.ik.solve(target, q_seed=seed))
        for bad in [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)]:
            self.assertIsNone(self.ik.solve(bad, q_seed=self.q))

    def test_far_target_rejected_not_zero_fallback(self):
        target = self.ik.forward(self.q)
        target[:3, 3] = [10, 10, 10]
        self.assertIsNone(self.ik.solve(target, q_seed=self.q))

    def test_wrap_does_not_authorize_crossing_limited_joint(self):
        self.assertAlmostEqual(float(np.rad2deg(self.mod.wrap_angle(np.deg2rad(-358)))), 2)
        # 数学短角不能替代 Nero 有限旋转轴上的实际可执行路径。
        q = self.q.copy()
        q[0] = 2.7
        other = q.copy()
        other[0] = -2.7
        self.assertFalse(self.ik.continuous(other, q))

    def test_nonconverged_optimizer_result_is_rejected_even_if_fk_matches(self):
        target = self.ik.forward(self.q + .001)
        result = SimpleNamespace(x=self.q+.001, success=False, nfev=80)
        with patch.object(self.mod, 'least_squares', return_value=result):
            self.assertIsNone(self.ik.solve(target, self.q))

    def test_converged_wrong_branch_is_rejected_including_q7(self):
        other = self.q.copy()
        other[6] += np.deg2rad(25)
        result = SimpleNamespace(x=other, success=True, nfev=5)
        with patch.object(self.mod, 'least_squares', return_value=result):
            self.assertIsNone(self.ik.solve(self.ik.forward(other), self.q))
        self.assertEqual(self.ik.last_diagnostics['code'], 'IK_JUMP')

    def test_converged_bad_fk_is_rejected(self):
        target = self.ik.forward(self.q)
        target[0, 3] += .01
        result = SimpleNamespace(x=self.q, success=True, nfev=5)
        with patch.object(self.mod, 'least_squares', return_value=result):
            self.assertIsNone(self.ik.solve(target, self.q))
        self.assertEqual(self.ik.last_diagnostics['code'], 'IK_FK_ERROR')

    def test_multiple_candidates_select_nearest_actual_not_first(self):
        target = self.ik.forward(self.q + .004)
        candidates = [SimpleNamespace(x=self.q+offset, success=True, nfev=5)
                      for offset in (.004, .003, .002)]
        with patch.object(self.mod, 'least_squares', side_effect=candidates):
            result = self.ik.solve(target, self.q)
        np.testing.assert_allclose(result, self.q+.002, atol=1e-12)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('teleop_nero_tracker_move_js_debug')
        self.ik = self.mod.NeroIK()
        self.q = np.array([.2, -.3, .15, 1, -.1, .2, -.4])
        self.tracker = np.eye(4)
        self.flange = self.ik.forward(self.q)
        self.session = self.mod.TeleopSession(self.ik, self.tracker, self.flange, self.q)
        self.sent = []

    def feedback(self, q=True, status=None):
        return dict(q=self.q.copy() if q else None, flange=self.flange.copy(),
                    status=status or dict(ctrl_mode=1, arm_status=0, motion_status=0, err_status={}),
                    enabled=[True]*7, state_fresh=True, joint_fresh=q,
                    flange_fresh=True, errors={})

    def test_valid_cycle_sends_seven_finite_radians(self):
        row = self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertTrue(row['sent'])
        np.testing.assert_allclose(self.sent[0], self.q, atol=1e-10)

    def test_missing_feedback_solves_only_and_requires_manual_reanchor(self):
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        row = self.session.step(self.tracker, self.feedback(q=False), self.sent.append)
        self.assertTrue(row['ik_success'])
        self.assertEqual(row['seed_source'], 'previous_ik_diagnostic_only')
        self.assertEqual(len(self.sent), 1)
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(len(self.sent), 1)
        self.session.step(self.tracker, self.feedback(), self.sent.append, reanchor=True)
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(len(self.sent), 2)

    def test_no_seed_no_ik_on_feedback_loss(self):
        row = self.session.step(self.tracker, self.feedback(q=False), self.sent.append)
        self.assertFalse(row['ik_success'])
        self.assertEqual(self.sent, [])

    def test_tracker_jump_latches_and_never_replays(self):
        jumped = self.tracker.copy()
        jumped[0, 3] = .5
        row = self.session.step(jumped, self.feedback(), self.sent.append)
        self.assertTrue(row['tracker_jump'])
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(self.sent, [])

    def test_robot_fault_cannot_be_cleared_with_r(self):
        bad = dict(ctrl_mode=1, arm_status=1, motion_status=0, err_status={})
        self.session.step(self.tracker, self.feedback(status=bad), self.sent.append)
        self.session.step(self.tracker, self.feedback(), self.sent.append, reanchor=True)
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(self.sent, [])

    def test_send_exception_does_not_commit_target(self):
        before = self.session.q_cmd_prev.copy()
        def failed_send(q):
            raise RuntimeError('CAN transmit failed')
        row = self.session.step(self.tracker, self.feedback(), failed_send)
        self.assertFalse(row['sent'])
        np.testing.assert_array_equal(self.session.q_cmd_prev, before)
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(self.sent, [])

    def test_all_seven_commands_clipped_without_crossing_limits(self):
        desired = self.q + np.deg2rad([8, -8, 8, -8, 8, -8, 8])
        result = self.mod.clip_command(self.ik, desired, self.q, self.q)
        np.testing.assert_allclose(np.rad2deg(result-self.q), [5, -5, 5, -5, 5, -5, 5], atol=1e-9)
        invalid = self.q.copy()
        invalid[6] = 9
        with self.assertRaises(ValueError):
            self.mod.clip_command(self.ik, invalid, self.q, self.q)

    def test_csv_keeps_all_seven_axes_and_raw_status(self):
        stream = io.StringIO()
        log = self.mod.CSVLog(stream)
        row = self.session.step(self.tracker, self.feedback(), self.sent.append)
        log.write(row)
        data = list(csv.DictReader(io.StringIO(stream.getvalue())))[0]
        self.assertAlmostEqual(float(data['q_cmd_7']), np.rad2deg(-.4))
        self.assertIn('arm_status', data['robot_state'])
        self.assertEqual(data['tracker_age'], '')  # API 无源样本时间戳，不能伪造0。

    def test_accumulated_target_jump_checked_even_without_tracker_jump(self):
        for distance in [.04, .08]:  # 单次Tracker 40mm <50mm；累计法兰48mm >30mm。
            T = self.tracker.copy()
            T[0, 3] = distance
            row = self.session.step(T, self.feedback(q=False), self.sent.append)
        self.assertTrue(row['target_jump'])
        self.assertEqual(self.sent, [])

    def test_tracker_loss_requires_r(self):
        self.session.step(None, self.feedback(), self.sent.append)
        self.session.step(self.tracker, self.feedback(), self.sent.append)
        self.assertEqual(self.sent, [])

    def test_ik_failure_is_not_sent_and_previous_target_not_advanced(self):
        target = self.tracker.copy()
        target[0, 3] = .005
        previous = self.session.gate.previous.copy()
        with patch.object(self.ik, 'solve', return_value=None):
            row = self.session.step(target, self.feedback(), self.sent.append)
        self.assertFalse(row['sent'])
        self.assertEqual(self.sent, [])
        np.testing.assert_array_equal(previous, self.session.gate.previous)

    def test_sdk_limit_pairs_intersection_and_order_validation(self):
        # 与实际 SDK 一致：joint_limits 值是 [lower, upper]，不是 min/max 字典。
        config = dict(joint_names=[f'joint{i}' for i in range(1, 8)],
                      joint_limits={f'joint{i}': [-1, 1] for i in range(1, 8)})
        self.mod.apply_sdk_limits(self.ik, config)
        self.assertEqual(self.ik.limits[0, 0], -1)
        self.assertEqual(self.ik.limits[5, 0], -.73)
        config['joint_names'].reverse()
        with self.assertRaises(ValueError):
            self.mod.apply_sdk_limits(self.ik, config)

    def test_actual_out_of_urdf_limit_is_diagnosed_as_limit_not_missing(self):
        fb = self.feedback()
        fb['q'][6] = 2
        row = self.session.step(self.tracker, fb, self.sent.append)
        self.assertTrue(row['joint_limit'])
        self.assertEqual(row['code'], 'ACTUAL_JOINT_LIMIT')
        self.assertEqual(self.sent, [])

    def test_stationary_pose_redundancy_drift_latches(self):
        # 有限关节下同一末端位姿，但七轴姿态累积变化：诊断门独立于数值求解器。
        gate = self.mod.RedundancyWatch()
        self.assertFalse(gate.update(self.flange, self.q))
        shifted = self.q.copy()
        shifted[6] += np.deg2rad(1.1)
        self.assertTrue(gate.update(self.flange, shifted))
        moving_target = self.flange.copy()
        moving_target[0, 3] += .002
        self.assertFalse(gate.update(moving_target, shifted))


class FakeArm:
    """仅记录硬件边界；实际分类、时间戳处理、限幅都由生产代码执行。"""
    def __init__(self):
        self.q = np.array([.2, -.3, .15, 1, -.1, .2, -.4])
        now = time.time()
        self.motors = [SimpleNamespace(timestamp=now, msg=SimpleNamespace(position=q)) for q in self.q]
        self.drivers = [SimpleNamespace(timestamp=now) for _ in range(7)]
        self.raw_status = SimpleNamespace(timestamp=now)
        self.status = dict(ctrl_mode=1, arm_status=0, motion_status=0, err_status={})
        self._driver = self
        self._parser = SimpleNamespace(**{n: SimpleNamespace(timestamp=now)
            for n in ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz')})
        self.sent = []
        self.validated = []

    def get_raw_motor_states(self): return self.motors
    def get_driver_states(self, i): return self.drivers[i-1]
    def get_arm_status(self): return self.status.copy()
    def get_raw_arm_status(self): return self.raw_status
    def get_joint_enable_status(self, i): return True
    def get_flange_pose(self): return np.array([0, 0, .4, 0, 0, 0])
    def validate_joint_command(self, q, **kwargs): self.validated.append(q.copy())
    def move_js(self, q): self.sent.append(q.copy())


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('teleop_nero_tracker_move_js_debug')
        self.arm = FakeArm()

    def test_all_joint_samples_must_be_fresh(self):
        self.assertTrue(self.mod.read_feedback(self.arm)['joint_fresh'])
        self.arm.motors[6].timestamp -= 1
        fb = self.mod.read_feedback(self.arm)
        self.assertFalse(fb['joint_fresh'])
        self.assertGreater(fb['joint_age_s'], .9)

    def test_one_old_flange_component_invalidates_aggregate(self):
        self.arm._parser.end_pose_xy.timestamp -= 1
        self.assertFalse(self.mod.read_feedback(self.arm)['flange_fresh'])

    def test_stale_enable_frame_blocks_even_with_fresh_status(self):
        self.arm.drivers[6].timestamp -= 1
        self.assertFalse(self.mod.state_ready(self.mod.read_feedback(self.arm)))

    def test_presend_rechecks_fault_after_ik(self):
        self.arm.status['arm_status'] = 2
        with self.assertRaises(RuntimeError):
            self.mod.checked_send(self.arm, self.mod.NeroIK(), self.arm.q, self.arm.q, time.monotonic())
        self.assertEqual(self.arm.sent, [])

    def test_presend_expired_target_blocks(self):
        with self.assertRaises(RuntimeError):
            self.mod.checked_send(self.arm, self.mod.NeroIK(), self.arm.q, self.arm.q, time.monotonic()-1)
        self.assertEqual(self.arm.sent, [])

    def test_presend_valid_command_reaches_only_js(self):
        self.mod.checked_send(self.arm, self.mod.NeroIK(), self.arm.q, self.arm.q, time.monotonic())
        self.assertEqual(len(self.arm.sent), 1)
        np.testing.assert_array_equal(self.arm.sent[0], self.arm.q)


if __name__ == '__main__':
    unittest.main()
