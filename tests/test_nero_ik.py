from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import robot_control.nero_ik as nero_ik
from robot_control.nero_ik import NeroIK, URDF_PATH


def test_nero_urdf_is_project_asset():
    expected = Path(__file__).resolve().parents[1] / "assets" / "nero" / "nero_description.urdf"
    assert URDF_PATH.resolve() == expected.resolve()
    assert expected.is_file()


def test_nero_fk_identity_shape_and_limits():
    solver = NeroIK()
    assert solver.forward(np.zeros(7)).shape == (4, 4)
    assert solver.limits.shape == (7, 2)


def test_ik_joint_continuity_uses_configured_delta():
    seed = np.zeros(7)
    candidate = seed.copy()
    candidate[0] = np.deg2rad(11.0)

    assert not NeroIK().continuous(candidate, seed)
    assert NeroIK(max_joint_delta_rad=np.deg2rad(15.0)).continuous(candidate, seed)


def test_ik_uses_one_seed_and_returns_first_valid_solution(monkeypatch):
    solver = NeroIK()
    seed = np.zeros(7)
    expected = seed.copy()
    expected[0] = 0.01
    target = solver.forward(expected)
    calls = []

    def fake_least_squares(residual, initial, *, jac, bounds, max_nfev):
        calls.append(np.asarray(initial).copy())
        assert jac(initial).shape == (13, 7)
        return SimpleNamespace(x=expected.copy(), success=True, nfev=1)

    monkeypatch.setattr(nero_ik, "least_squares", fake_least_squares)

    solution = solver.solve(target, seed)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], seed)
    np.testing.assert_allclose(solution, expected)
    assert solver.last_diagnostics["code"] == "IK_SUCCESS"
    assert len(solver.last_diagnostics["candidates"]) == 1


def test_ik_timeout_returns_safe_best_intermediate(monkeypatch):
    solver = NeroIK()
    seed = np.zeros(7)
    target_q = seed.copy()
    target_q[0] = 0.1
    target = solver.forward(target_q)

    def fake_least_squares(residual, initial, *, jac, bounds, max_nfev):
        residual(initial)
        residual(target_q)
        raise TimeoutError("test timeout")

    monkeypatch.setattr(nero_ik, "least_squares", fake_least_squares)

    solution = solver.solve(target, seed)

    np.testing.assert_allclose(solution, target_q)
    diagnostics = solver.last_diagnostics
    assert diagnostics["budget_exhausted"]
    assert diagnostics["deadline_fallback"]
    assert diagnostics["ik_success"]
    assert diagnostics["code"] == "IK_SUCCESS"
    assert diagnostics["residual_evaluations"] == 2
    best = diagnostics["best_intermediate"]
    assert best["fk_position_error_mm"] < 1e-6
    assert best["fk_rotation_error_deg"] < 1e-6
    np.testing.assert_allclose(best["dq_ik_deg"], np.rad2deg(target_q - seed))


def test_ik_timeout_rejects_best_intermediate_with_joint_jump(monkeypatch):
    solver = NeroIK()
    seed = np.zeros(7)
    target_q = seed.copy()
    target_q[0] = np.deg2rad(11.0)
    target = solver.forward(target_q)

    def fake_least_squares(residual, initial, *, jac, bounds, max_nfev):
        residual(initial)
        residual(target_q)
        raise TimeoutError("test timeout")

    monkeypatch.setattr(nero_ik, "least_squares", fake_least_squares)

    solution = solver.solve(target, seed)

    assert solution is None
    diagnostics = solver.last_diagnostics
    assert diagnostics["budget_exhausted"]
    assert diagnostics["deadline_fallback"]
    assert not diagnostics["ik_success"]
    assert diagnostics["code"] == "IK_JUMP"


def manual_forward(solver, q):
    """独立保留旧版逐关节 FK，核验 Pinocchio 的坐标、轴和关节顺序。"""
    transform = np.eye(4)
    for origin, axis, angle in zip(solver.origins, solver.axes, q):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        transform = transform @ origin @ rotation
    return transform


def test_pinocchio_fk_matches_original_chain_over_joint_range():
    solver = NeroIK()
    rng = np.random.default_rng(23)
    configurations = rng.uniform(solver.limits[:, 0], solver.limits[:, 1], (50, 7))
    for q in configurations:
        np.testing.assert_allclose(solver.forward(q), manual_forward(solver, q), atol=2e-14)


def test_fk_and_jacobian_ignore_fixed_world_offset(tmp_path):
    root = ET.parse(URDF_PATH)
    fixed_base = next(j for j in root.findall('joint')
                      if j.find('child').get('link') == 'base_link')
    origin = fixed_base.find('origin')
    origin.set('xyz', '.3 -.2 .1')
    origin.set('rpy', '.4 -.3 .2')
    path = tmp_path / 'offset.urdf'
    root.write(path)
    reference, shifted = NeroIK(), NeroIK(path)
    q = np.array([.2, -.3, .15, 1., -.1, .2, -.4])
    expected, expected_jac = reference._forward_and_jacobian(q)
    actual, actual_jac = shifted._forward_and_jacobian(q)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(actual_jac, expected_jac, atol=2e-14)
    np.testing.assert_allclose(shifted.forward(q), expected, atol=2e-14)


@pytest.mark.parametrize('angle', [0., 1e-6, 1.2, np.pi - .001])
def test_residual_analytic_jacobian_matches_central_difference(monkeypatch, angle):
    solver = NeroIK()
    seed = np.array([.2, -.3, .15, 1., -.1, .2, -.4])
    candidate = seed + np.array([.02, -.01, .01, -.02, .01, .015, -.01])
    target = solver.forward(candidate)
    axis = np.array([.2, -.3, .4]); axis /= np.linalg.norm(axis)
    target[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix() @ target[:3, :3]
    target[:3, 3] += [.001, -.002, .003]

    def check_jacobian(residual, initial, *, jac, bounds, max_nfev):
        analytic = jac(candidate)
        step = 1e-7 * np.eye(7)
        numeric = np.column_stack([
            (residual(candidate + delta) - residual(candidate - delta)) / 2e-7
            for delta in step
        ])
        np.testing.assert_allclose(analytic, numeric, atol=3e-5, rtol=3e-6)
        return SimpleNamespace(x=candidate, success=True, nfev=1)

    monkeypatch.setattr(nero_ik, 'least_squares', check_jacobian)
    solver.solve(target, seed)
    assert solver.last_diagnostics['jacobian_evaluations'] == 1


def test_real_ik_accepts_small_move_and_preserves_error_checks():
    solver = NeroIK()
    seed = np.array([.2, -.3, .15, 1., -.1, .2, -.4])
    target_q = seed + np.array([.02, -.01, .01, -.02, .01, .015, -.01])
    target = manual_forward(solver, target_q)
    solution = solver.solve(target, seed)
    assert solution is not None, solver.last_diagnostics
    assert solver.in_limits(solution)
    assert solver.continuous(solution, seed)
    p, r = nero_ik.pose_error(manual_forward(solver, solution), target)
    assert p < nero_ik.MAX_FK_POSITION_ERROR_M
    assert np.rad2deg(r) < nero_ik.MAX_FK_ROTATION_ERROR_DEG
    assert solver.last_diagnostics['jacobian_evaluations'] > 0


@pytest.mark.parametrize('error, expected_code', [
    ('pose', 'IK_NUMERICAL_ERROR'), ('nan_seed', 'IK_NUMERICAL_ERROR'),
    ('limit', 'IK_JOINT_LIMIT'),
])
def test_invalid_inputs_are_rejected(error, expected_code):
    solver = NeroIK()
    seed = np.zeros(7)
    target = solver.forward(seed)
    if error == 'pose':
        target[3, 0] = 1.
    elif error == 'nan_seed':
        seed[0] = np.nan
    else:
        seed[0] = solver.limits[0, 1] + .01
    assert solver.solve(target, seed) is None
    assert solver.last_diagnostics['code'] == expected_code


def test_analytic_jacobian_deadline_keeps_existing_fallback(monkeypatch):
    solver = NeroIK()
    seed = np.zeros(7)
    target_q = seed.copy(); target_q[0] = .1
    target = solver.forward(target_q)
    clock = [0.]
    monkeypatch.setattr(nero_ik.time, 'monotonic', lambda: clock[0])

    def timeout_on_jacobian(residual, initial, *, jac, bounds, max_nfev):
        residual(target_q)
        clock[0] = nero_ik.MAX_SOLVE_SECONDS + .001
        jac(target_q)
        pytest.fail('Jacobian 必须检查同一时间预算')

    monkeypatch.setattr(nero_ik, 'least_squares', timeout_on_jacobian)
    np.testing.assert_allclose(solver.solve(target, seed), target_q)
    assert solver.last_diagnostics['deadline_fallback']
    assert solver.last_diagnostics['jacobian_evaluations'] == 0
