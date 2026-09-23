from pathlib import Path
from types import SimpleNamespace

import numpy as np

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

    def fake_least_squares(residual, initial, *, bounds, max_nfev):
        calls.append(np.asarray(initial).copy())
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

    def fake_least_squares(residual, initial, *, bounds, max_nfev):
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

    def fake_least_squares(residual, initial, *, bounds, max_nfev):
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
