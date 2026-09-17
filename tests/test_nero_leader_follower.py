"""主从门控离线测试，不连接CAN。"""
import importlib
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np


def feedback(q, enabled=True):
    return dict(q=np.asarray(q, dtype=float), joint_fresh=True, state_fresh=True,
                enabled=[enabled]*7,
                status=dict(arm_status=0, ctrl_mode=1, motion_status=0, err_status={}))


class LeaderFollowerTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('teleop_nero_leader_follower')
        self.limits = np.array([[-2., 2.]]*7)
        self.q = np.zeros(7)

    def test_disabled_master_can_be_read_but_disabled_follower_rejected(self):
        self.m.require_feedback(feedback(self.q, False), self.limits, follower=False)
        with self.assertRaises(RuntimeError):
            self.m.require_feedback(feedback(self.q, False), self.limits, follower=True)

    def test_stale_or_faulted_feedback_rejected(self):
        for field in ('joint_fresh', 'state_fresh'):
            fb = feedback(self.q)
            fb[field] = False
            with self.assertRaises(RuntimeError):
                self.m.require_feedback(fb, self.limits, follower=False)

    def test_alignment_requires_three_samples_and_both_arms_aligned(self):
        gate = self.m.Alignment(self.q)
        far = self.q + .1
        self.assertFalse(gate.update(feedback(self.q), feedback(far)))
        self.assertFalse(gate.update(feedback(self.q), feedback(self.q)))
        self.assertFalse(gate.update(feedback(self.q), feedback(self.q)))
        self.assertTrue(gate.update(feedback(self.q), feedback(self.q)))

    def test_alignment_rejects_moving_master(self):
        gate = self.m.Alignment(self.q)
        with self.assertRaises(RuntimeError):
            gate.update(feedback(self.q+.1), feedback(self.q))

    def test_absolute_follow_and_rate_limit_all_seven_axes(self):
        target = np.deg2rad([2, -2, 2, -2, 2, -2, 2])
        cmd = self.m.follow_command(target, self.q, self.q, self.q, self.limits)
        np.testing.assert_allclose(np.rad2deg(cmd), [1, -1, 1, -1, 1, -1, 1])

    def test_jump_lag_and_limit_rejected(self):
        for target in (self.q+.3, self.q+3, np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.m.follow_command(target, self.q, self.q, self.q, self.limits)
        with self.assertRaises(ValueError):
            self.m.follow_command(self.q+.1, self.q-.1, self.q, self.q+.1, self.limits)

    def test_leader_stream_accepts_unknown_status_but_not_follower(self):
        fb = feedback(self.q)
        fb.update(source='leader_joint_angles', state_fresh=False, status=None)
        self.m.require_feedback(fb, self.limits, follower=False)
        with self.assertRaises(RuntimeError):
            self.m.require_feedback(fb, self.limits, follower=True)

    def test_leader_known_fault_not_ignored(self):
        fb = feedback(self.q)
        fb.update(source='leader_joint_angles', state_fresh=False)
        fb['status']['arm_status'] = 1
        with self.assertRaises(RuntimeError):
            self.m.require_feedback(fb, self.limits, follower=False)

    def test_leader_frames_complete_fresh_and_after_switch(self):
        now = time.time()
        frames = {f'leader_joint_{i}': SimpleNamespace(timestamp=now,
                  msg=SimpleNamespace(**{f'joint_{i}': i*.01})) for i in range(1, 8)}
        driver = SimpleNamespace(_parser=SimpleNamespace(**frames),
                                 get_leader_joint_angles=lambda: SimpleNamespace(msg=[i*.01 for i in range(1, 8)]))
        arm = SimpleNamespace(_driver=driver, get_arm_status=lambda: None,
                              get_raw_arm_status=lambda: None)
        fb = self.m.read_leader_feedback(arm, since=now-.1)
        self.assertTrue(fb['joint_fresh'])
        self.assertFalse(fb['state_fresh'])
        np.testing.assert_allclose(fb['q'], [.01,.02,.03,.04,.05,.06,.07])
        self.assertFalse(self.m.read_leader_feedback(arm, since=now+.1)['joint_fresh'])
        frames['leader_joint_7'].timestamp = now-1
        self.assertFalse(self.m.read_leader_feedback(arm)['joint_fresh'])
        driver._parser.leader_joint_7 = None
        self.assertFalse(self.m.read_leader_feedback(arm)['joint_fresh'])

    def test_normal_start_retries_enable_and_push_before_feedback(self):
        calls = []
        answers = iter([False, True])
        def enable():
            calls.append('enable')
            return next(answers)
        arm = SimpleNamespace(_driver=SimpleNamespace(enable=enable,
                         set_normal_mode=lambda: calls.append('normal')))
        with patch.object(self.m, 'read_feedback', return_value=feedback(self.q)), patch.object(self.m.time, 'sleep'):
            self.m.start_can_mode(arm, self.limits, 'test')
        self.assertEqual(calls, ['enable', 'normal', 'enable', 'normal'])

    def test_start_known_fault_never_enables(self):
        fb = feedback(self.q)
        fb['status']['arm_status'] = 1
        arm = SimpleNamespace(_driver=SimpleNamespace(enable=lambda: self.fail('must not enable')))
        with patch.object(self.m, 'read_feedback', return_value=fb):
            with self.assertRaises(RuntimeError):
                self.m.start_can_mode(arm, self.limits, 'test')

    def test_start_timeout_is_bounded(self):
        arm = SimpleNamespace(_driver=SimpleNamespace(enable=lambda: False, set_normal_mode=lambda: None))
        with patch.object(self.m, 'read_feedback', return_value=feedback(self.q)):
            with self.assertRaises(TimeoutError):
                self.m.start_can_mode(arm, self.limits, 'test', timeout=0)


if __name__ == '__main__':
    unittest.main()
