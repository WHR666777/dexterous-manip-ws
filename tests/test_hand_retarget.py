from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from teleop.config import load_config, select_side_configs
from teleop.hand_retarget import HAND_TIMING_FIELDS, L20Retargeter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AnyDexRetarget'))


def make_retargeter():
    return L20Retargeter(
        str(ROOT / 'AnyDexRetarget/example/config/adaptive/pico4/pico4_linker_l20.yaml'),
        'right', 10,
    )


def legacy_jacobians(robot, q, links, offsets):
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    result = []
    for link, offset in zip(links, offsets):
        local = pin.getFrameJacobian(robot.model, robot.data, link, pin.LOCAL)
        rotation = robot.data.oMf[link].rotation
        if np.allclose(offset, 0):
            result.append(rotation @ local[:3])
        else:
            x, y, z = offset
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            result.append(rotation @ (local[:3] - cross @ local[3:]))
    return np.array(result)


@pytest.mark.parametrize('offset_scale', [0., 1e-9, .02])
def test_batch_jacobians_preserve_legacy_offsets_and_match_finite_difference(offset_scale):
    optimizer = make_retargeter()._retargeter.optimizer
    robot, links = optimizer.robot, optimizer.computed_link_indices
    rng = np.random.default_rng(14)
    q = rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    offsets = rng.normal(size=(len(links), 3)) * offset_scale
    actual = robot.compute_all_jacobians_batch_with_offsets(q, links, offsets)
    np.testing.assert_allclose(actual, legacy_jacobians(robot, q, links, offsets), atol=2e-14)
    eps = 1e-7
    numeric = np.stack([
        (robot.compute_points_batch(q + delta, links, offsets)
         - robot.compute_points_batch(q - delta, links, offsets)) / (2 * eps)
        for delta in eps * np.eye(len(q))
    ], axis=-1)
    np.testing.assert_allclose(actual, numeric, atol=2e-8, rtol=2e-5)
    # The same mutable offset array must invalidate the cached preparation.
    offsets[-1] += [.02, -.01, .03]
    changed = robot.compute_all_jacobians_batch_with_offsets(q, links, offsets)
    np.testing.assert_allclose(changed, legacy_jacobians(robot, q, links, offsets), atol=2e-14)


def synthetic_landmarks():
    points = [[0., 0., 0.]]
    for finger in range(5):
        for segment in range(1, 5):
            points.append([.025 * segment, .02 * (finger - 2), .003 * segment ** 2])
    return np.asarray(points)


@pytest.mark.parametrize('bend', [0., .3, 1.])
def test_bimanual_configs_align_mirrored_hands_in_model_frame(bend):
    from anydexretarget.mediapipe import apply_mediapipe_transformations

    configs = select_side_configs(load_config(ROOT / 'configs/pico4_bimanual_nero_l20.yaml'))
    retargeters = {
        side: L20Retargeter(settings['hand']['retarget_config'], side, 10)._retargeter
        for side, settings in configs.items()
    }
    points = synthetic_landmarks()
    points[:, 2] *= bend
    # The palm correction must work independently of the hand's world pose.
    rotation = Rotation.from_euler('xyz', [23., -41., 68.], degrees=True).as_matrix()
    points = points @ rotation.T + [.12, -.18, .27]
    mirror = np.array([1., -1., 1.])
    right = retargeters['right']._apply_rotation(apply_mediapipe_transformations(points, 'right'))
    left = retargeters['left']._apply_rotation(apply_mediapipe_transformations(points * mirror, 'left'))
    np.testing.assert_allclose(left, right * mirror, atol=1e-10)

    # Confirm that this is the reflection used by the actual left/right URDFs.
    model_points = {}
    for side, retargeter in retargeters.items():
        optimizer = retargeter.optimizer
        limits = optimizer.robot.joint_limits
        q = limits[:, 0] + .4 * (limits[:, 1] - limits[:, 0])
        model_points[side] = optimizer.robot.compute_points_batch(
            q, optimizer.computed_link_indices, optimizer.computed_link_offsets)
    np.testing.assert_allclose(model_points['left'], model_points['right'] * mirror, atol=1e-9)


def test_retarget_diagnostics_are_per_solve_and_keep_limits_and_raw_slew():
    retargeter = make_retargeter()
    retargeter.reset(np.zeros(20, dtype=int))
    q, raw = retargeter.retarget(synthetic_landmarks())
    diag = retargeter.last_diagnostics.copy()
    assert 0 < diag['hand_retarget_numevals'] <= 50
    assert all(np.isfinite(diag[key]) and diag[key] >= 0 for key in HAND_TIMING_FIELDS)
    limits = retargeter._retargeter.optimizer.robot.joint_limits
    # Existing optimizer output is float32; allow its bound-rounding error.
    assert np.all(q >= limits[:, 0] - 2e-7) and np.all(q <= limits[:, 1] + 2e-7)
    assert np.max(raw[:11]) <= 10 and np.max(raw[15:]) <= 10
    np.testing.assert_array_equal(raw[11:15], 255)
    retargeter.retarget(synthetic_landmarks())
    assert retargeter.last_diagnostics['hand_retarget_numevals'] <= 50
    stats = retargeter._retargeter.optimizer.get_timing_stats()
    np.testing.assert_allclose(
        diag['timing_hand_solver_s'] + retargeter.last_diagnostics['timing_hand_solver_s'],
        stats.total_ms / 1000,
    )
