"""Real spawn tests for compute isolation; all hardware is absent or fake."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from teleop import bimanual_controller as dual, compute, controller
from teleop.config import load_config, select_side_configs
from teleop.hand_feedback import HandFeedbackSnapshot
from teleop.hand_retarget import L20Retargeter
from robot_control.nero_ik import NeroIK


ROOT = Path(__file__).resolve().parents[1]


def config_for(side="left"):
    return select_side_configs(load_config(ROOT / "configs/pico4_bimanual_nero_l20.yaml"), side)[side]


class FakeIK:
    def __init__(self, **kwargs):
        if MODE == "startup_failure":
            raise ValueError("model initialization failed")
        self.last_diagnostics = {}

    def forward(self, _): return np.eye(4)

    def solve(self, target, seed):
        if BARRIER is not None:
            BARRIER.wait(5.)
        self.last_diagnostics = {"code": "IK_FAILED" if MODE == "ik_failure" else "IK_SUCCESS",
                                 "pid": os.getpid()}
        return None if MODE == "ik_failure" else seed + .01


class FakeHand:
    def __init__(self, *_):
        self.previous = np.zeros(20, dtype=int)
        self.last_diagnostics = {}

    def reset(self, previous=None):
        self.previous = np.zeros(20, dtype=int) if previous is None else previous.copy()

    def retarget(self, _):
        if MODE in ("timeout", "expired"):
            time.sleep(.15)
        elif MODE == "crash":
            os._exit(17)
        elif MODE == "error":
            raise ValueError("retarget failed")
        self.previous += 1
        self.last_diagnostics = {"hand_retarget_numevals": 3, "pid": os.getpid()}
        return np.zeros(21), self.previous.copy()


def fake_math_worker(connection, config, use_arm, use_hand):
    global MODE, BARRIER
    MODE, BARRIER = config.get("_test_mode"), config.get("_test_barrier")
    if "_test_pids" in config:
        config["_test_pids"][0 if config["pico4"]["side"] == "left" else 1] = os.getpid()
    compute.NeroIK, compute.L20Retargeter = FakeIK, FakeHand
    compute._compute_main(connection, config, use_arm, use_hand)


def wrong_reply_worker(connection, *_):
    connection.send(dict(id=0, value=None))
    request = connection.recv()
    connection.send(dict(id=request["id"] - 1, value=None))
    connection.recv()
    connection.close()


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_models_match_inline_ik_retarget_slew_and_reset(side):
    config = config_for(side)
    local_ik = NeroIK(max_joint_delta_rad=config["arm"]["max_joint_delta_rad"])
    local_hand = L20Retargeter(config["hand"]["retarget_config"], side, 10)
    worker = compute.ComputeProcess(config, "both")
    try:
        worker.start()
        process = worker._process
        assert process.pid != os.getpid()
        seed = np.array([0., .7, 0., 1.5, 0., 0., -1.])
        target = local_ik.forward(seed + [.004, -.003, .002, 0., .001, -.002, .001])
        np.testing.assert_allclose(worker.ik.forward(seed), local_ik.forward(seed), atol=1e-14)
        np.testing.assert_allclose(worker.ik.solve(target, seed), local_ik.solve(target, seed), atol=1e-10)
        assert worker.ik.last_diagnostics["code"] == "IK_SUCCESS"
        points = np.array([[0., 0., 0.]] + [
            [.025 * j, .02 * (i - 2), .003 * j * j]
            for i in range(5) for j in range(1, 5)
        ]) * [1, -1 if side == "left" else 1, 1]
        for initial in (np.zeros(20, dtype=int), np.full(20, 200, dtype=int)):
            local_hand.reset(initial)
            worker.retargeter.reset(initial)
            for scale in (1., .6):
                worker.begin_cycle()
                q, raw = worker.retargeter.retarget(points * scale)
                expected_q, expected_raw = local_hand.retarget(points * scale)
                np.testing.assert_allclose(q, expected_q, atol=1e-12)
                np.testing.assert_array_equal(raw, expected_raw)
                assert worker.retargeter.last_diagnostics["hand_retarget_numevals"] == local_hand.last_diagnostics["hand_retarget_numevals"]
                assert worker.timings["timing_hand_retarget_s"] > 0
                assert worker.timings["timing_hand_retarget_cpu_s"] > 0
                assert worker.timings["timing_compute_ipc_s"] >= 0
    finally:
        worker.close()
        worker.close()
    assert not process.is_alive()


def test_two_compute_processes_solve_in_parallel():
    barrier = mp.get_context("spawn").Barrier(2)
    workers = []
    try:
        for side in ("left", "right"):
            config = config_for(side)
            config["_test_barrier"] = barrier
            worker = compute.ComputeProcess(config, "arm", worker_target=fake_math_worker)
            workers.append(worker)
            worker.start()
            worker.timeout_s = 8.
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda w: w.ik.solve(np.eye(4), np.zeros(7)), workers))
        for result in results:
            np.testing.assert_allclose(result, np.full(7, .01))
        assert len({w.ik.last_diagnostics["pid"] for w in workers}) == 2
    finally:
        processes = [w._process for w in workers]
        for worker in workers: worker.close()
    assert all(not p.is_alive() for p in processes)


def test_timeout_poisoning_rejects_late_results_and_cleans_up():
    config = config_for()
    config["_test_mode"] = "timeout"
    worker = compute.ComputeProcess(config, "hand", worker_target=fake_math_worker)
    try:
        worker.start()
        process = worker._process
        worker.timeout_s = .02
        with pytest.raises(TimeoutError):
            worker.retargeter.retarget(np.zeros((21, 3)))
        time.sleep(.2)  # The old reply has now arrived; it must never be reused.
        with pytest.raises(RuntimeError, match="failed request"):
            worker.retargeter.reset(np.zeros(20))
    finally:
        worker.close()
    assert not process.is_alive()


def test_mismatched_response_is_fatal():
    worker = compute.ComputeProcess(config_for(), "arm", worker_target=wrong_reply_worker)
    try:
        worker.start()
        with pytest.raises(RuntimeError, match="current request"):
            worker.ik.solve(np.eye(4), np.zeros(7))
        with pytest.raises(RuntimeError, match="failed request"):
            worker.ik.solve(np.eye(4), np.zeros(7))
    finally:
        worker.close()


def test_model_startup_failure_closes_child():
    config = config_for()
    config["_test_mode"] = "startup_failure"
    worker = compute.ComputeProcess(config, "arm", worker_target=fake_math_worker)
    with pytest.raises(RuntimeError, match="model initialization failed"):
        worker.start()
    assert worker._process is None
    worker.close()


def fake_io_worker(side, config, control, connection, paused, stopping):
    """Exercise the production side runtime and its nested compute process."""
    index = dual.SIDES.index(side)
    config["_test_io_pids"][index] = os.getpid()
    positions = {"arm": np.zeros(7), "hand": np.full(20, 30, dtype=int)}

    def send(name, value):
        config["_test_sends"][index * 2 + (name == "hand")] += 1
        positions[name] = value.copy()

    arm = SimpleNamespace(connect=lambda: None, enable_normal_mode=lambda: None,
                          is_enabled=lambda: True, disconnect=lambda: None,
                          get_joint_positions=lambda: positions["arm"].copy(),
                          get_joint_torques=lambda: np.zeros(7), get_flange_pose=lambda: np.zeros(6),
                          get_tcp_pose=lambda: np.zeros(6), move_js=lambda q: send("arm", q))
    hand = SimpleNamespace(connect=lambda: None, disconnect=lambda: None,
                           set_speed=lambda _: None, set_joint_positions_raw=lambda q: send("hand", q))
    controller._make_arm, controller._make_hand = lambda _: arm, lambda _: hand
    controller._move_arm_to_start_pose = lambda *_: None
    controller._make_compute = lambda cfg, ctrl: compute.ComputeProcess(
        cfg, ctrl, worker_target=fake_math_worker)
    dual.HandFeedbackPoller = lambda *a, **kw: SimpleNamespace(
        start=lambda: None, stop=lambda: None, report=lambda: None,
        get_latest=lambda: HandFeedbackSnapshot(positions["hand"].copy(), 0., 0., 1))
    dual._worker_main(side, config, control, connection, paused, stopping)


@pytest.mark.parametrize("mode", [None, "crash", "ik_failure"])
def test_dual_nested_compute_commit_failure_and_cleanup(mode):
    ctx = mp.get_context("spawn")
    configs = select_side_configs(load_config(ROOT / "configs/pico4_bimanual_nero_l20.yaml"))
    pids, io_pids, sends = ctx.Array("i", [0, 0]), ctx.Array("i", [0, 0]), ctx.Array("i", [0, 0, 0, 0])
    for side, config in configs.items():
        config.update(_test_pids=pids, _test_io_pids=io_pids, _test_sends=sends,
                      _test_mode=mode if side == "left" else None)
        config["pico4"]["max_age_s"] = 5.
    group = dual._WorkerGroup(configs, "both", context=ctx, worker_target=fake_io_worker)

    def frames():
        return {side: SimpleNamespace(side=side, received_at=time.monotonic(), pair_skew=0.,
                                      wrist_position=np.zeros(3), wrist_quat=[0., 0., 0., 1.],
                                      landmarks=np.zeros((21, 3))) for side in dual.SIDES}

    try:
        group.start()
        group.collect("ready", 0, 10.)
        assert len(set(pids[:]) | set(io_pids[:]) | {os.getpid()}) == 5
        group.request("anchor", 1, frames())
        group.collect("anchored", 1, 5.)
        group.paused.clear()
        group.request("prepare", 2, frames())
        if mode:
            with pytest.raises(RuntimeError if mode == "crash" else dual._Pause):
                group.collect("prepared", 2, 5.)
            assert group.paused.is_set()
            assert sends[:] == [0, 0, 0, 0]
        else:
            group.collect("prepared", 2, 5.)
            assert sends[:] == [0, 0, 0, 0]
            group.request("commit", 2)
            results = group.collect("result", 2, 5.)
            assert sends[:] == [1, 1, 1, 1]
            for reply in results.values():
                row = reply["row"]
                np.testing.assert_array_equal(row["observation_hand_position_raw"], np.full(20, 30))
                np.testing.assert_array_equal(row["hand_action_raw"], np.full(20, 31))
                assert row["hand_retarget_numevals"] == 3
                assert row["timing_compute_ipc_s"] >= 0
    finally:
        group.close()
    for pid in pids[:]:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("mode", [None, "timeout", "crash", "error", "expired", "ik_failure", "paused_crash"])
def test_single_controller_uses_spawn_and_never_sends_failed_or_expired_work(monkeypatch, tmp_path, mode):
    config = config_for()
    config["camera"]["enabled"] = False
    config["_test_mode"] = mode
    worker = compute.ComputeProcess(config, "both", worker_target=fake_math_worker)
    if mode == "timeout": worker.timeout_s = .02
    if mode == "expired":
        config["pico4"]["max_age_s"] = .03
        worker.timeout_s = .5
    sends, closed, rows, saved = [], [], [], []
    positions = {"arm": np.zeros(7), "hand": np.full(20, 30, dtype=int)}

    def send(name, value):
        sends.append(name)
        positions[name] = value.copy()

    arm = SimpleNamespace(connect=lambda: None, enable_normal_mode=lambda: None,
                          is_enabled=lambda: True, disconnect=lambda: closed.append("arm"),
                          get_joint_positions=lambda: positions["arm"].copy(),
                          get_joint_torques=lambda: np.zeros(7), get_flange_pose=lambda: np.zeros(6),
                          get_tcp_pose=lambda: np.zeros(6), move_js=lambda q: send("arm", q))
    hand = SimpleNamespace(connect=lambda: None, disconnect=lambda: closed.append("hand"),
                           set_speed=lambda _: None, set_joint_positions_raw=lambda q: send("hand", q))
    poller = SimpleNamespace(poll_hz=20., max_age_s=.25, start=lambda: None,
                             stop=lambda: closed.append("poller"), report=lambda: None,
                             get_latest=lambda: HandFeedbackSnapshot(positions["hand"].copy(), 0., 0., 1))

    class Recorder:
        def __init__(self, *a, **kw): self.metadata, self.active = {}, False
        def start(self): self.active = True
        def append(self, **row):
            if self.active: rows.append(row)
        def stop_and_save(self): saved.append(True); self.active = False

    keys = iter(["r", "b", "q"] if mode != "paused_crash" else [None])

    class Keyboard:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self):
            if mode == "paused_crash":
                worker._process.terminate()
                worker._process.join(2.)
            return next(keys)

    monkeypatch.setattr(controller, "_make_compute", lambda *_: worker)
    monkeypatch.setattr(controller, "_make_arm", lambda _: arm)
    monkeypatch.setattr(controller, "_make_hand", lambda _: hand)
    monkeypatch.setattr(controller, "HandFeedbackPoller", lambda *a, **kw: poller)
    monkeypatch.setattr(controller, "EpisodeRecorder", Recorder)
    monkeypatch.setattr(controller, "Keyboard", Keyboard)
    monkeypatch.setattr(controller, "_move_arm_to_start_pose", lambda *_: None)
    monkeypatch.setattr(controller, "_input_device", lambda _: SimpleNamespace(is_available=lambda: True, stop=lambda: closed.append("input")))
    monkeypatch.setattr(controller, "_latest_frame", lambda *_: SimpleNamespace(
        received_at=time.monotonic(), wrist_position=np.zeros(3), wrist_quat=[0., 0., 0., 1.],
        landmarks=np.zeros((21, 3))))
    if mode in ("timeout", "crash", "error", "paused_crash"):
        with pytest.raises((RuntimeError, TimeoutError)):
            controller.run(config, "both", str(tmp_path))
    else:
        controller.run(config, "both", str(tmp_path))
    assert worker._process is None
    assert closed == ["poller", "input", "hand", "arm"]
    assert saved
    if mode is None:
        assert sends == ["arm", "hand"]
        assert len(rows) == 1
        np.testing.assert_array_equal(rows[0]["observation_arm_joint_position"], np.zeros(7))
        np.testing.assert_array_equal(rows[0]["observation_hand_position_raw"], np.full(20, 30))
        np.testing.assert_array_equal(rows[0]["hand_action_raw"], np.full(20, 31))
        assert rows[0]["timing_compute_ipc_s"] >= 0
    else:
        assert sends == rows == []
