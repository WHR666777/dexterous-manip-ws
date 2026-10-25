"""Offline config, spawn/commit and paired-recording checks; never opens CAN."""

import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from teleop import bimanual_controller as dual, cli, controller
from teleop.config import load_config, require_site_configuration, select_side_configs
from teleop.hand_feedback import HandFeedbackSnapshot


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pico4_bimanual_nero_l20.yaml"

pytestmark = pytest.mark.usefixtures("inline_compute")


def frame(side, **kwargs):
    return SimpleNamespace(side=side, received_at=time.monotonic(), pair_skew=0.,
                           wrist_position=np.zeros(3), wrist_quat=np.array([0., 0., 0., 1.]),
                           landmarks=np.zeros((21, 3)), **kwargs)


def test_side_configs_are_independent_and_preserve_single_defaults(tmp_path):
    config = load_config(CONFIG)
    selected = select_side_configs(config)
    assert tuple(selected) == dual.SIDES
    left, right = selected.values()
    assert (left["arm"]["can_channel"], left["hand"]["can_channel"]) == ("can_arm_b", "can_hand_b")
    assert (right["arm"]["can_channel"], right["hand"]["can_channel"]) == ("can_arm_a", "can_hand_a")
    assert left["hand"]["type"] == left["pico4"]["side"] == "left"
    assert left["hand"]["retarget_config"].endswith("pico4_linker_l20_left.yaml")
    assert left["arm"]["start_pose_file"].endswith("nero_left_start_pose.yaml")
    np.testing.assert_array_equal(left["calibration"]["R_base_pico4"], right["calibration"]["R_base_pico4"])
    left["tracking"]["ema_alpha"] = .99
    left["calibration"]["R_base_pico4"][0][0] = 99
    assert right["tracking"]["ema_alpha"] == .4
    assert config["sides"]["left"]["calibration"]["R_base_pico4"][0][0] == 0
    # A normalized config remains loadable, including resolved paths.
    path = tmp_path / "dual.yaml"
    path.write_text(yaml.safe_dump(config))
    assert tuple(select_side_configs(load_config(path), "left")) == ("left",)
    original = load_config(ROOT / "configs/pico4_nero_l20.yaml")
    assert tuple(select_side_configs(original)) == ("right",)
    with pytest.raises(ValueError, match="only defines the right"):
        select_side_configs(original, "both")


@pytest.mark.parametrize("fault", ["quest", "same_bus", "wrong_hand", "rotation", "missing_side", "missing_input"])
def test_invalid_dual_configs_are_rejected(tmp_path, fault):
    config = yaml.safe_load(CONFIG.read_text())
    if fault == "quest":
        config["input_device"] = "quest3"
    elif fault == "same_bus":
        config["sides"]["left"]["hand"]["can_channel"] = "can_hand_a"
    elif fault == "wrong_hand":
        config["sides"]["left"]["hand"]["type"] = "right"
    elif fault == "rotation":
        config["sides"]["left"]["calibration"]["R_base_pico4"] = [[0]]
    elif fault == "missing_side":
        del config["sides"]["left"]
    else:
        del config["pico4"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_left_start_pose_is_named_before_any_hardware(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    config["sides"]["left"]["arm"]["start_pose_file"] = str(tmp_path / "missing.yaml")
    monkeypatch.setattr(controller, "_make_arm", lambda _: pytest.fail("Hardware was constructed"))
    with pytest.raises(RuntimeError, match="left.*arm.start_pose_file"):
        dual.run(config, "both")
    require_site_configuration(config, control="hand", require_start_pose=True)


def test_cli_side_selection_and_dual_start_pose_guard(monkeypatch):
    monkeypatch.setattr("sys.argv", ["teleop", "--config", str(CONFIG), "record-start-pose"])
    with pytest.raises(SystemExit, match="requires --side"):
        cli.main()
    seen = []
    monkeypatch.setattr(cli, "record_start_pose", lambda config, *a, **kw: seen.append(config))
    monkeypatch.setattr("sys.argv", ["teleop", "--config", str(CONFIG), "--side", "left", "record-start-pose"])
    cli.main()
    assert seen[0]["arm"]["can_channel"] == "can_arm_b"
    monkeypatch.setattr("builtins.input", lambda _: "EXECUTE")
    monkeypatch.setattr(dual, "run", lambda *a: seen.append(a))
    monkeypatch.setattr("sys.argv", ["teleop", "--config", str(CONFIG), "run", "--control", "hand", "--execute"])
    cli.main()
    assert seen[-1][1:] == ("hand", None)


def fake_row(side):
    marker = 1 if side == "left" else 2
    return dict(
        observation_timestamp_monotonic=time.monotonic(),
        observation_arm_joint_position=np.full(7, marker * .01),
        observation_hand_position_raw=np.full(20, marker * 10),
        arm_action=np.full(7, marker * .1), hand_action_raw=np.full(20, marker * 20),
        timing_prepare_s=.003, timing_dispatch_s=.001,
        **dict.fromkeys(controller._ControlTiming.FIELDS, 0.0),
    )


class FakeSideRuntime:
    """Only the SDK-facing runtime is replaced; real spawn/worker/IPC code runs."""

    def __init__(self, side, config, control):
        self.side, self.config = side, config
        self.index = dual.SIDES.index(side)

    def start(self): pass
    def check_health(self): pass
    def reanchor(self, sample): pass

    def prepare(self, sample):
        if getattr(sample, "hold", False):
            raise dual._IKHold("IK_FK_ERROR")
        if getattr(sample, "fail", False):
            raise dual._Pause("IK failed")
        if getattr(sample, "crash", False):
            raise RuntimeError("SDK failed")
        if getattr(sample, "block", False):
            self.config["stop"].wait(10.)
        barrier = self.config.get("barrier")
        if barrier is not None:
            barrier.wait(timeout=5.)
        self.config["prepared"][self.index] += 1
        return fake_row(self.side)

    def dispatch(self, row, may_send):
        if not may_send():
            raise dual._Pause("Peer paused")
        self.config["sends"][self.index] += 1
        return row

    def close(self):
        self.config["closed"][self.index] = 1

    def cancel_prepared(self, row):
        self.config["cancelled"][self.index] += 1


def fake_sdk_worker(side, config, control, connection, paused, stopping):
    dual._SideController = FakeSideRuntime
    config["stop"] = stopping
    dual._worker_main(side, config, control, connection, paused, stopping)


def worker_group(parallel_barrier=False):
    context = mp.get_context("spawn")
    counters = {name: context.Array("i", [0, 0]) for name in ("sends", "prepared", "closed", "cancelled")}
    shared = {**counters, "barrier": context.Barrier(2) if parallel_barrier else None}
    configs = {side: {"input_device": "pico4", "pico4": {
        "side": side, "max_age_s": 10., "max_pair_skew_s": .05}, **shared} for side in dual.SIDES}
    group = dual._WorkerGroup(configs, "both", context=context, worker_target=fake_sdk_worker)
    return group, counters


def anchor(group, cycle=1):
    group.paused.set()
    group.request("anchor", cycle, {side: frame(side) for side in dual.SIDES})
    group.collect("anchored", cycle, 8.)
    group.paused.clear()


def test_real_spawn_computes_in_parallel_and_waits_for_commit():
    import sys

    sys.path.insert(0, str(ROOT / "AnyDexRetarget/example"))
    from input.pico4 import Pico4HandFrame

    group, counters = worker_group(parallel_barrier=True)
    try:
        group.start()
        group.collect("ready", 0, 8.)
        anchor(group)
        group.request("prepare", 2, {side: Pico4HandFrame(**vars(frame(side))) for side in dual.SIDES})
        group.collect("prepared", 2, 8.)
        # The two-party barrier inside prepare would fail under serial dispatch.
        assert list(counters["prepared"]) == [1, 1]
        assert list(counters["sends"]) == [0, 0]
        group.request("commit", 2)
        results = group.collect("result", 2, 8.)
        assert {reply["cycle"] for reply in results.values()} == {2}
        assert list(counters["sends"]) == [1, 1]
    finally:
        group.close()
    assert list(counters["closed"]) == [1, 1]
    assert all(not process.is_alive() for process in group.processes.values())


def test_failure_cancels_both_and_late_replies_cannot_enter_next_cycle():
    group, counters = worker_group()
    try:
        group.start()
        group.collect("ready", 0, 8.)
        anchor(group)
        group.request("prepare", 2, {side: frame(side, fail=side == "left") for side in dual.SIDES})
        with pytest.raises(dual._Pause):
            group.collect("prepared", 2, 8.)
        assert group.paused.is_set()
        assert list(counters["sends"]) == [0, 0]
        group.request("cancel", 2)
        anchor(group, 3)  # Discards any pending reply from cycle 2.
        group.request("prepare", 4, {side: frame(side) for side in dual.SIDES})
        group.collect("prepared", 4, 8.)
        group.request("commit", 4)
        assert all(reply["cycle"] == 4 for reply in group.collect("result", 4, 8.).values())
        assert list(counters["sends"]) == [1, 1]
    finally:
        group.close()


def test_ik_hold_retries_without_anchor_and_cancels_every_unsent_peer_target():
    group, counters = worker_group()
    try:
        group.start()
        group.collect("ready", 0, 8.)
        anchor(group)
        for cycle in (2, 3, 4):
            group.request("prepare", cycle, {side: frame(side, hold=side == "left") for side in dual.SIDES})
            with pytest.raises(dual._IKHold, match="left: IK_FK_ERROR"):
                group.collect("prepared", cycle, 8.)
            assert not group.paused.is_set()
            assert list(counters["sends"]) == [0, 0]
            group.request("cancel", cycle)
        group.request("prepare", 5, {side: frame(side) for side in dual.SIDES})
        group.collect("prepared", 5, 8.)
        assert list(counters["cancelled"]) == [0, 3]
        group.request("commit", 5)
        group.collect("result", 5, 8.)
        assert list(counters["sends"]) == [1, 1]
    finally:
        group.close()


def test_hard_pause_takes_precedence_over_peer_ik_hold():
    group, counters = worker_group()
    try:
        group.start()
        group.collect("ready", 0, 8.)
        anchor(group)
        group.request("prepare", 2, {side: frame(side, hold=side == "left", fail=side == "right")
                                     for side in dual.SIDES})
        with pytest.raises(dual._Pause):
            group.collect("prepared", 2, 8.)
        assert group.paused.is_set()
        assert list(counters["sends"]) == [0, 0]
    finally:
        group.close()


@pytest.mark.parametrize("fault", ["expired", "crash", "timeout"])
def test_expired_results_worker_crash_and_timeout_never_send(fault):
    group, counters = worker_group()
    try:
        group.start()
        group.collect("ready", 0, 8.)
        anchor(group)
        samples = {side: frame(side, crash=fault == "crash", block=fault == "timeout")
                   for side in dual.SIDES}
        if fault == "expired":
            for sample in samples.values():
                sample.received_at -= 20.
        group.request("prepare", 2, samples)
        if fault == "expired":
            group.collect("prepared", 2, 8.)
            group.request("commit", 2)
            with pytest.raises(dual._Pause, match="expired"):
                group.collect("result", 2, 8.)
        elif fault == "crash":
            with pytest.raises(RuntimeError, match="SDK failed|disconnected|exited"):
                group.collect("prepared", 2, 8.)
        else:
            with pytest.raises(TimeoutError):
                group.collect("prepared", 2, .1)
        assert list(counters["sends"]) == [0, 0]
    finally:
        group.close()
    assert list(counters["closed"]) == [1, 1]


def test_side_runtime_keeps_pre_command_feedback_and_existing_send_order(monkeypatch):
    config = select_side_configs(load_config(CONFIG), "right")["right"]
    positions = {"arm": np.zeros(7), "hand": np.full(20, 20)}
    sends, disconnected = [], []
    arm = SimpleNamespace(
        connect=lambda: None, enable_normal_mode=lambda: None, is_enabled=lambda: True,
        get_joint_positions=lambda: positions["arm"].copy(),
        get_flange_pose=lambda: np.zeros(6), get_tcp_pose=lambda: np.zeros(6),
        get_joint_torques=lambda: np.zeros(7), disconnect=lambda: disconnected.append("arm"),
    )
    def arm_send(target):
        sends.append("arm")
        positions["arm"] = target.copy()
    arm.move_js = arm_send
    def hand_send(target):
        sends.append("hand")
        positions["hand"] = target.copy()
    hand = SimpleNamespace(connect=lambda: None, disconnect=lambda: disconnected.append("hand"),
                           set_speed=lambda _: None, set_joint_positions_raw=hand_send)
    feedback = SimpleNamespace(start=lambda: None, stop=lambda: None, report=lambda: None,
                               get_latest=lambda: HandFeedbackSnapshot(positions["hand"].copy(), 1., 1., 1))
    resets = []
    retargeter = SimpleNamespace(reset=lambda raw: resets.append(raw.copy()),
                                retarget=lambda _: (np.zeros(16), np.full(20, 40)))
    ik = SimpleNamespace(forward=lambda _: np.eye(4), solve=lambda *_: np.full(7, .1))
    monkeypatch.setattr(controller, "_make_arm", lambda _: arm)
    monkeypatch.setattr(controller, "_make_hand", lambda _: hand)
    monkeypatch.setattr(controller, "_retargeter", lambda _: retargeter)
    monkeypatch.setattr(controller, "NeroIK", lambda **_: ik)
    monkeypatch.setattr(controller, "_move_arm_to_start_pose", lambda *_: None)
    monkeypatch.setattr(dual, "HandFeedbackPoller", lambda *a, **k: feedback)
    runtime = dual._SideController("right", config, "both")
    try:
        runtime.start()
        sample = frame("right")
        runtime.reanchor(sample)
        sample.wrist_position[:] = [.02, 0., 0.]
        row = runtime.prepare(sample)
        assert sends == []
        np.testing.assert_array_equal(runtime.last_target, np.eye(4))
        np.testing.assert_array_equal(runtime.gate._accepted, np.eye(4))
        runtime.cancel_prepared(row)
        np.testing.assert_array_equal(resets[-1], row["observation_hand_position_raw"])
        row = runtime.prepare(sample)
        runtime.dispatch(row, lambda: True)
        np.testing.assert_array_equal(runtime.last_target, row["target_base_flange"])
        np.testing.assert_array_equal(runtime.gate._accepted, runtime.last_target)
        assert sends == ["arm", "hand"]
        np.testing.assert_array_equal(row["observation_arm_joint_position"], np.zeros(7))
        np.testing.assert_array_equal(row["observation_hand_position_raw"], np.full(20, 20))
        np.testing.assert_array_equal(row["arm_joint_position"], np.full(7, .1))
        np.testing.assert_array_equal(row["hand_position_raw"], np.full(20, 40))
    finally:
        runtime.close()
    assert disconnected == ["hand", "arm"]


@pytest.mark.parametrize("soft_hold", [False, True])
def test_coordinator_records_only_complete_pairs_and_requires_reanchor(tmp_path, monkeypatch, soft_hold):
    config = load_config(CONFIG)
    config["sides"]["left"]["arm"]["start_pose_file"] = str(ROOT / "configs/nero_start_pose.yaml")
    keys = iter(["r", "b", None, None if soft_hold else "r", None, "q"])
    rows, commands, stopped, cameras = [], [], [], []
    image = object()
    source = SimpleNamespace(is_available=lambda: True,
                             get_hand_frames=lambda: {side: frame(side) for side in dual.SIDES},
                             stop=lambda: stopped.append("input"))
    class Camera:
        metadata = {"width": 640, "height": 480}
        def __init__(self, _): cameras.append(self)
        def start(self): pass
        def get_latest(self): return image
        def stop(self): stopped.append("camera")
    class Recorder:
        active = False
        def __init__(self, *args, **kwargs): self.metadata = args[1]
        def start(self): self.active = True
        def append(self, **row):
            if self.active: rows.append(row)
        def stop_and_save(self): self.active = False
    class Keyboard:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return next(keys)
    class Workers:
        def __init__(self, *args):
            import threading
            self.paused = threading.Event()
            self.paused.set()
        def start(self): pass
        def check_health(self): pass
        def request(self, kind, cycle, frames=None): commands.append((kind, cycle))
        def collect(self, kind, cycle, *args):
            if kind == "prepared" and cycle == 2:
                if soft_hold:
                    raise dual._IKHold("left: IK_FK_ERROR")
                self.paused.set()
                raise dual._Pause("left IK failed")
            return {side: {"cycle": cycle, "row": fake_row(side)} for side in dual.SIDES}
        def close(self): stopped.append("workers")
    monkeypatch.setattr(controller, "_input_device", lambda _: source)
    monkeypatch.setattr(dual, "RealSenseCamera", Camera)
    monkeypatch.setattr(dual, "EpisodeRecorder", Recorder)
    monkeypatch.setattr(dual, "Keyboard", Keyboard)
    monkeypatch.setattr(dual, "_WorkerGroup", Workers)
    monkeypatch.setattr(dual.time, "sleep", lambda _: None)
    dual.run(config, "both", str(tmp_path))
    assert len(cameras) == 1
    assert [row["cycle_id"] for row in rows] == ([3, 4, 5] if soft_hold else [5])
    assert rows[0]["camera_frame"] is image
    assert [cycle for kind, cycle in commands if kind == "commit"] == ([3, 4, 5] if soft_hold else [5])
    assert [cycle for kind, cycle in commands if kind == "anchor"] == ([1] if soft_hold else [1, 4])
    assert stopped == ["workers", "camera", "input"]
