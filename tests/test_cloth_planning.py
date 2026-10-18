from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cloth_agent.config import load_config, validate_hardware_config
from cloth_agent.geometry import interpolate_poses, tcp_to_flange, transform, transform_points
from cloth_agent.planning import MotionPlan, Planner, PlanningError


def test_tcp_conversion_preserves_rotated_mount_and_task_frame():
    task_base = transform([.2, -.4, .1, .3, -.2, .4])
    flange_tcp = transform([.01, .02, .15, -.3, .1, -.4])
    goal = transform([.5, .1, .4, 1., .2, -.3])
    actual = task_base @ tcp_to_flange(goal, task_base, flange_tcp) @ flange_tcp
    np.testing.assert_allclose(actual, goal, atol=1e-12)
    np.testing.assert_allclose(transform_points(task_base, [0, 0, 0]), task_base[:3, 3])


def test_pose_interpolation_follows_line_and_short_rotation():
    start, end = transform([0, 0, 0, 0, 0, 3.1]), transform([.02, 0, 0, 0, 0, -3.1])
    poses = list(interpolate_poses(start, end))
    assert len(poses) < 10
    for pose in poses:
        np.testing.assert_allclose(pose[1:3, 3], 0)
    np.testing.assert_allclose(poses[-1], end, atol=1e-12)


@pytest.fixture
def planner():
    return Planner(load_config())


def test_joint_plan_preserves_endpoints_and_exact_stored_rate_limits(planner):
    seed = np.asarray(planner.config["mock"]["initial_joints"])
    target = seed + [.06, -.02, .01, 0, 0, 0, 0]
    plan = planner.plan("left", seed, target_joints=target)
    np.testing.assert_allclose(plan.sample(0), seed, atol=1e-10)
    np.testing.assert_allclose(plan.sample(plan.duration), target, atol=1e-10)
    peak_v, peak_a = plan.peak_rates()
    assert np.all(peak_v <= np.asarray(planner.arms["left"]["velocity_limits"]) * 1.00001)
    assert np.all(peak_a <= np.asarray(planner.arms["left"]["acceleration_limits"]) * 1.00001)
    restored = MotionPlan.from_dict(plan.as_dict())
    np.testing.assert_allclose(restored.sample(.1), plan.sample(.1))


@pytest.mark.parametrize("mode", ["joint", "cartesian"])
def test_both_waypoint_modes_end_at_the_requested_tcp(planner, mode):
    seed = np.asarray(planner.config["mock"]["initial_joints"])
    target_q = seed + [.015, -.01, .01, 0, 0, 0, 0]
    solver = planner.ik["left"]
    goal = solver.forward(target_q)
    plan = planner.plan("left", seed, waypoints=[goal], mode=mode)
    final = solver.forward(plan.positions[-1])
    assert np.linalg.norm(final[:3, 3] - goal[:3, 3]) < .002
    assert Rotation.from_matrix(final[:3, :3] @ goal[:3, :3].T).magnitude() < np.deg2rad(2)


def test_planning_failures_report_waypoint_index_and_ik_diagnostics(planner, monkeypatch):
    solver = planner.ik["left"]
    seed = planner.config["mock"]["initial_joints"]
    goal = solver.forward(seed)
    monkeypatch.setattr(solver, "solve", lambda *_: None)
    solver.last_diagnostics = {"code": "IK_JUMP"}
    with pytest.raises(PlanningError) as raised:
        planner.plan("left", seed, waypoints=[goal], mode="joint")
    assert raised.value.as_dict()["index"] == 0
    assert raised.value.as_dict()["diagnostics"]["code"] == "IK_JUMP"


def test_limits_workspace_and_empty_targets_are_rejected(planner):
    seed = np.asarray(planner.config["mock"]["initial_joints"])
    invalid = seed.copy()
    invalid[0] = 100
    with pytest.raises(PlanningError, match="limits"):
        planner.plan("left", seed, target_joints=invalid)
    with pytest.raises(PlanningError, match="waypoints"):
        planner.plan("left", seed, waypoints=[])
    planner.arms["left"]["workspace"] = [[10, 10, 10], [11, 11, 11]]
    with pytest.raises(PlanningError, match="workspace"):
        planner.plan("left", seed, target_joints=seed)


def test_mock_defaults_do_not_fill_hardware_calibration():
    config = deepcopy(load_config())
    config["backend"] = "hardware"
    with pytest.raises(ValueError, match="flange_from_tcp"):
        validate_hardware_config(config)
    assert config["arms"]["left"]["hand_poses"]["closed"] is None
