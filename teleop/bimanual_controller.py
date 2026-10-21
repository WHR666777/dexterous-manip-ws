"""One Pico/camera coordinator and two independently owned arm/hand workers.

Only PREPARE computes targets. COMMIT is issued after both sides pass, and only
matching completed cycles reach the recorder. CAN calls are not atomic across
devices: an exception during COMMIT can still leave a partially sent cycle.
"""

from __future__ import annotations

from collections import deque
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation

from . import controller as single
from .config import calibration_rotation, input_config, require_site_configuration, select_side_configs
from .coordinate_frames import anydex_wrist_to_base, pose6_to_matrix, position_quaternion_to_matrix
from .hand_feedback import HandFeedbackPoller
from .hand_retarget import HAND_TIMING_FIELDS
from .keyboard import Keyboard
from .realsense_camera import RealSenseCamera
from .recording import EpisodeRecorder
from .target_gate import TargetGate
from .wrist_tracker import WristTracker


SIDES = ("left", "right")


class _Pause(Exception):
    """A recoverable failure that requires a new anchor for both sides."""


class _Quit(Exception):
    pass


def _fresh(frame, config) -> bool:
    source = input_config(config)
    return (frame is not None and frame.side == source["side"]
            and 0 <= time.monotonic() - frame.received_at <= float(source["max_age_s"])
            and 0 <= frame.pair_skew <= float(source["max_pair_skew_s"]))


def _frames(source, configs):
    frames = source.get_hand_frames()
    return {side: frame if _fresh(frame, configs[side]) else None
            for side, frame in frames.items() if side in configs}


def _check_relay(source, config):
    if input_config(config).get("input_mode") == "relay" and not source.is_available():
        raise RuntimeError("Pico 4 relay is unavailable. Start "
                           "`python3 AnyDexRetarget/example/input/pico4_daemon.py` first.")


def input_check(config) -> None:
    """Inspect both hands through one input client, without creating CAN objects."""
    configs = select_side_configs(config)
    retargeters = {side: single._retargeter(settings) for side, settings in configs.items()}
    source = single._input_device(config)
    seen = dict.fromkeys(SIDES, None)
    try:
        _check_relay(source, config)
        print("Pico4 left/right input check; Ctrl+C to stop. CAN is not opened.")
        while True:
            for side, frame in _frames(source, configs).items():
                if frame is not None and frame.received_at != seen[side]:
                    seen[side] = frame.received_at
                    qpos, raw = retargeters[side].retarget(frame.landmarks)
                    print(f"[{side}] frame age={(time.monotonic()-frame.received_at)*1000:.1f}ms "
                          f"wrist={np.round(frame.wrist_position, 3)} "
                          f"qpos={qpos.shape} raw={raw.tolist()}")
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()


class _SideController:
    """A worker's local state; all SDK objects are constructed inside the worker."""

    def __init__(self, side, config, control):
        self.side, self.config, self.control = side, config, control
        self.arm = self.hand = self.feedback = self.ik = self.retargeter = None
        self.tracker = self.last_target = self.flange_to_tcp = None

    def start(self):
        config = self.config
        self.calibration = calibration_rotation(config)
        self.gate = TargetGate(**{key: config["safety"][key] for key in (
            "max_position_step_m", "max_rotation_step_deg", "tcp_workspace_cube_side_m")})
        if self.control in ("arm", "both"):
            self.arm = single._make_arm(config)
            self.ik = single.NeroIK(max_joint_delta_rad=config["arm"]["max_joint_delta_rad"])
            print(f"[{self.side}] Nero connecting: {config['arm']['can_channel']}", flush=True)
            self.arm.connect()
            print(f"[{self.side}] Nero enabling normal mode (timeout=5.0 s).", flush=True)
            self.arm.enable_normal_mode()
            if not self.arm.is_enabled():
                raise RuntimeError("Nero is not enabled; enable it before run.")
            single._move_arm_to_start_pose(self.arm, config)
            flange, tcp = single._read_complete_arm_startup_poses(self.arm)
            self.flange_to_tcp = np.linalg.inv(pose6_to_matrix(flange)) @ pose6_to_matrix(tcp)
            model_tcp = self.ik.forward(self.arm.get_joint_positions()) @ self.flange_to_tcp
            self.gate.set_tcp_workspace_center(model_tcp[:3, 3])
            print(f"[{self.side}] TCP workspace center={model_tcp[:3, 3].tolist()}", flush=True)
        if self.control in ("hand", "both"):
            self.hand = single._make_hand(config)
            self.retargeter = single._retargeter(config)
            self.feedback = HandFeedbackPoller(
                self.hand, poll_hz=config["hand"]["feedback_poll_hz"],
                max_age_s=config["hand"]["feedback_max_age_s"],
            )
            self.hand.connect()
            self.hand.set_speed(config["hand"]["speed"])
            self.feedback.start()

    def reanchor(self, frame):
        if not _fresh(frame, self.config):
            raise _Pause("Pico4 frame timeout while re-anchoring")
        if self.arm is not None:
            flange = self.ik.forward(self.arm.get_joint_positions())
            position, quaternion = anydex_wrist_to_base(
                frame.wrist_position, frame.wrist_quat, self.calibration)
            tracking = self.config["tracking"]
            self.tracker = WristTracker(
                flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_quat(),
                position_scale=tracking["position_scale"], ema_alpha=tracking["ema_alpha"],
                negate_rot_xy=tracking["negate_rot_xy"],
                position_deadband=tracking["position_deadband_m"],
                rotation_deadband_deg=tracking["rotation_deadband_deg"],
            )
            self.tracker.update(position, quaternion)
            self.gate.reanchor(flange)
            self.last_target = flange.copy()
        if self.retargeter is not None:
            self.retargeter.reset(self.feedback.get_latest().positions_raw)

    def prepare(self, frame):
        if not _fresh(frame, self.config):
            raise _Pause("Pico4 frame timeout")
        started = time.perf_counter()
        stages = dict.fromkeys(single._ControlTiming.FIELDS, 0.0)
        diagnostics = dict.fromkeys(HAND_TIMING_FIELDS, np.nan)
        diagnostics["hand_retarget_numevals"] = -1
        observed_at = time.monotonic()
        observed_hand = self.feedback.get_latest() if self.feedback is not None else None
        position, quaternion = anydex_wrist_to_base(
            frame.wrist_position, frame.wrist_quat, self.calibration)
        seed = action = np.full(7, np.nan)
        hand_action = np.full(20, -1, dtype=np.int64)
        target = np.full((4, 4), np.nan)
        if self.arm is not None:
            if self.tracker is None:
                raise _Pause("Anchor required")
            target_position, target_quaternion = self.tracker.update(position, quaternion)
            target = position_quaternion_to_matrix(target_position, target_quaternion)
            stage = time.perf_counter()
            seed = self.arm.get_joint_positions().copy()
            stages["timing_seed_feedback_s"] = time.perf_counter() - stage
            stage = time.perf_counter()
            action = self.ik.solve(target, seed)
            stages["timing_ik_s"] = time.perf_counter() - stage
            if action is None:
                single._print_ik_failure_diagnostics(self.ik, seed, target, self.last_target)
                raise _Pause(self.ik.last_diagnostics.get("code", "IK failed"))
            if not self.gate.accept(target, (target @ self.flange_to_tcp)[:3, 3]):
                raise _Pause(self.gate.reason)
            self.last_target = target.copy()
        if self.retargeter is not None:
            stage = time.perf_counter()
            _, hand_action = self.retargeter.retarget(frame.landmarks)
            stages["timing_hand_retarget_s"] = time.perf_counter() - stage
            stats = getattr(self.retargeter, "last_diagnostics", {})
            diagnostics.update({key: stats[key] for key in diagnostics if key in stats})
        return dict(
            observation_timestamp_monotonic=observed_at,
            observation_arm_joint_position=seed,
            observation_hand_position_raw=(observed_hand.positions_raw if observed_hand else np.full(20, -1)),
            observation_hand_feedback_query_started_at=(observed_hand.query_started_at if observed_hand else np.nan),
            observation_hand_feedback_query_finished_at=(observed_hand.query_finished_at if observed_hand else np.nan),
            observation_hand_feedback_query_sequence=(observed_hand.query_sequence if observed_hand else -1),
            quest_wrist=np.concatenate((frame.wrist_position, frame.wrist_quat)),
            quest_landmarks=frame.landmarks, quest_received_at=frame.received_at,
            target_base_flange=target, arm_action=action, hand_action_raw=hand_action,
            timing_prepare_s=time.perf_counter() - started, **stages, **diagnostics,
        )

    def dispatch(self, row, may_send):
        started = time.perf_counter()
        row.update(arm_sent_at=np.nan, hand_sent_at=np.nan)
        for name, device, send in (
            ("arm", self.arm, lambda: self.arm.move_js(row["arm_action"])),
            ("hand", self.hand, lambda: self.hand.set_joint_positions_raw(row["hand_action_raw"])),
        ):
            if device is None:
                continue
            if not may_send():
                raise _Pause("Peer paused or shutdown requested")
            stage = time.perf_counter()
            send()
            row[f"{name}_sent_at"] = time.monotonic()
            row[f"timing_{name}_send_s"] = time.perf_counter() - stage
        row["timing_hand_send_wait_s"] = getattr(self.hand, "last_position_send_wait_s", 0.0)
        stage = time.perf_counter()
        row.update(
            arm_joint_position=self.arm.get_joint_positions() if self.arm else np.full(7, np.nan),
            arm_joint_torque=self.arm.get_joint_torques() if self.arm else np.full(7, np.nan),
            arm_tcp_pose=self.arm.get_tcp_pose() if self.arm else np.full(6, np.nan),
        )
        feedback = self.feedback.get_latest() if self.feedback else None
        read_at = time.monotonic() if feedback else np.nan
        row.update(
            hand_position_raw=feedback.positions_raw if feedback else np.full(20, -1),
            hand_feedback_query_started_at=feedback.query_started_at if feedback else np.nan,
            hand_feedback_query_finished_at=feedback.query_finished_at if feedback else np.nan,
            hand_feedback_query_sequence=feedback.query_sequence if feedback else -1,
            hand_feedback_snapshot_read_at=read_at,
            hand_feedback_query_age_s=read_at - feedback.query_finished_at if feedback else np.nan,
            timing_feedback_s=time.perf_counter() - stage,
        )
        row["timing_dispatch_s"] = time.perf_counter() - started
        return row

    def close(self):
        stopped = True
        if self.feedback is not None:
            try:
                self.feedback.stop()
            except Exception as exc:
                stopped = False
                print(f"[{self.side}] feedback stop failed: {exc}", flush=True)
        errors = single._disconnect_devices(self.hand if stopped else None, self.arm)
        if errors:
            print(f"[{self.side}] disconnect errors: {'; '.join(errors)}", flush=True)
        if self.feedback is not None:
            print(f"[{self.side}] feedback summary:", flush=True)
            self.feedback.report()


def _worker_main(side, config, control, connection, paused, stopping):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runtime = _SideController(side, config, control)
    cycle = 0
    pending = None
    try:
        runtime.start()
        connection.send(dict(kind="ready", cycle=0))
        while not stopping.is_set():
            if not connection.poll(.02):
                continue
            command = connection.recv()
            cycle, kind = command["cycle"], command["kind"]
            try:
                if kind == "cancel":
                    pending = None
                    continue
                if kind == "anchor":
                    pending = None
                    runtime.reanchor(command["frame"])
                    reply = dict(kind="anchored", cycle=cycle)
                elif kind == "prepare":
                    pending = None
                    if paused.is_set():
                        raise _Pause("Anchor required")
                    row = runtime.prepare(command["frame"])
                    pending = (cycle, command["frame"], row)
                    reply = dict(kind="prepared", cycle=cycle)
                elif kind == "commit":
                    if pending is None or pending[0] != cycle:
                        raise RuntimeError("COMMIT does not match the prepared cycle")
                    _, frame, row = pending
                    pending = None
                    if not _fresh(frame, config):
                        raise _Pause("Prepared input expired before COMMIT")
                    row = runtime.dispatch(row, lambda: not paused.is_set() and not stopping.is_set())
                    reply = dict(kind="result", cycle=cycle, row=row)
                else:
                    raise RuntimeError(f"Unknown worker command: {kind}")
            except _Pause as exc:
                paused.set()
                pending = None
                reply = dict(kind="paused", cycle=cycle, reason=str(exc))
            connection.send(reply)
    except (EOFError, BrokenPipeError):
        paused.set()
    except BaseException as exc:
        paused.set()
        try:
            connection.send(dict(kind="error", cycle=cycle,
                                 reason=f"{type(exc).__name__}: {exc}"))
        except (EOFError, OSError):
            pass
        raise
    finally:
        try:
            runtime.close()
        finally:
            connection.close()


class _WorkerGroup:
    def __init__(self, configs, control, *, context=None, worker_target=_worker_main):
        self.configs, self.control = configs, control
        self.context = context or mp.get_context("spawn")
        self.target = worker_target
        self.paused, self.stopping = self.context.Event(), self.context.Event()
        self.paused.set()
        self.connections, self.processes = {}, {}

    def start(self):
        for side, config in self.configs.items():
            parent, child = self.context.Pipe()
            process = self.context.Process(
                name=f"teleop-{side}", target=self.target,
                args=(side, config, self.control, child, self.paused, self.stopping),
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self.connections[side], self.processes[side] = parent, process

    def request(self, kind, cycle, frames=None):
        for side, connection in self.connections.items():
            message = dict(kind=kind, cycle=cycle)
            if frames is not None:
                message["frame"] = frames[side]
            connection.send(message)

    def collect(self, kind, cycle, timeout_s, check_keys=lambda: None):
        deadline = time.monotonic() + timeout_s
        pending = {connection: side for side, connection in self.connections.items()}
        replies = {}
        while pending:
            check_keys()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.paused.set()
                raise TimeoutError(f"Workers timed out waiting for {kind}: {list(pending.values())}")
            for connection in wait(list(pending), timeout=min(.02, remaining)):
                side = pending[connection]
                try:
                    reply = connection.recv()
                except (EOFError, OSError) as exc:
                    self.paused.set()
                    raise RuntimeError(f"{side} worker disconnected") from exc
                if reply["kind"] == "error":
                    self.paused.set()
                    raise RuntimeError(f"{side} worker: {reply['reason']}")
                if reply["cycle"] != cycle:
                    continue  # Cancelled/late results cannot enter a new cycle.
                if reply["kind"] == "paused":
                    self.paused.set()
                    raise _Pause(f"{side}: {reply['reason']}")
                if reply["kind"] != kind:
                    raise RuntimeError(f"{side}: expected {kind}, got {reply['kind']}")
                replies[side] = reply
                del pending[connection]
            for connection, side in pending.items():
                if not self.processes[side].is_alive() and not connection.poll():
                    self.paused.set()
                    raise RuntimeError(f"{side} worker exited ({self.processes[side].exitcode})")
        check_keys()
        return replies

    def check_health(self):
        """Also detect a worker dying while the coordinator is paused."""
        for side, connection in self.connections.items():
            while connection.poll():
                try:
                    reply = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"{side} worker disconnected") from exc
                if reply["kind"] == "error":
                    raise RuntimeError(f"{side} worker: {reply['reason']}")
                # No round is in flight here; only cancelled replies may remain.
            if not self.processes[side].is_alive():
                raise RuntimeError(f"{side} worker exited ({self.processes[side].exitcode})")

    def close(self):
        self.paused.set()
        self.stopping.set()
        deadline = time.monotonic() + 5.0
        for process in self.processes.values():
            process.join(max(0.0, deadline - time.monotonic()))
        for side, process in self.processes.items():
            if process.is_alive():
                print(f"[{side}] worker did not exit within 5 s; terminating.")
                process.terminate()
                process.join(1.0)
        for connection in self.connections.values():
            connection.close()


class _BimanualTiming:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.rows = []

    def append(self, work_s, cycle_s, prepare_wait_s, dispatch_wait_s):
        self.rows.append((work_s, cycle_s, prepare_wait_s, dispatch_wait_s))
        if sum(row[1] for row in self.rows) >= 2.0:
            self.report()

    def report(self):
        if not self.rows:
            return
        values = np.asarray(self.rows)
        details = " ".join(
            f"{name}={column.mean()*1000:.1f}/{np.percentile(column,95)*1000:.1f}/{column.max()*1000:.1f}"
            for name, column in zip(("work", "cycle", "prepare_wait", "dispatch_wait"), values.T))
        print(f"[bimanual timing] cycles={len(values)} active_hz={len(values)/values[:,1].sum():.2f} "
              f"target_hz={1/self.period:g} overruns={np.count_nonzero(values[:,0]>self.period)}/{len(values)} "
              f"ms(mean/p95/max): {details}")
        self.rows.clear()


def run(config, control, record_dir=None):
    configs = select_side_configs(config)
    if tuple(configs) != SIDES or config.get("input_device") != "pico4":
        raise ValueError("Bimanual run requires left and right Pico4 configurations.")
    if control not in ("arm", "hand", "both"):
        raise ValueError("control must be arm, hand or both.")
    require_site_configuration(config, control=control, require_start_pose=True)
    if control in ("arm", "both"):
        for settings in configs.values():
            single._load_start_pose(settings)  # Validate both files before either arm moves.
    root = Path(record_dir or config["recording"]["output_dir"])
    lerobot = config["recording"].get("lerobot")
    if control != "both" and lerobot is not None:
        root = root.parent / f"{root.name}_diagnostics"
        lerobot = None
    recorder = EpisodeRecorder(
        root, {"config": config["_path"], "control": control, "hand_mode": "tracking",
               "input_device": "pico4", "input_sides": list(SIDES),
               "hand_feedback": {side: {"poll_hz": settings["hand"]["feedback_poll_hz"],
                                         "max_age_s": settings["hand"]["feedback_max_age_s"]}
                                 for side, settings in configs.items()}},
        sample_hz=float(config["control_hz"]), lerobot=lerobot,
    )
    workers = _WorkerGroup(configs, control)
    source = camera = None
    period = 1.0 / float(config["control_hz"])
    timeout = float(input_config(config)["max_age_s"])
    timings = {side: single._ControlTiming(float(config["control_hz"]), side) for side in SIDES}
    timing = _BimanualTiming(float(config["control_hz"]))
    pending_keys = deque()
    cycle = 0
    try:
        source = single._input_device(config)
        _check_relay(source, config)
        if config["camera"]["enabled"]:
            camera = RealSenseCamera(config["camera"])
            camera.start()
            recorder.metadata["camera"] = dict(camera.metadata)
        with Keyboard() as keyboard:
            def check_keys():
                key = keyboard.read()
                if key == "q":
                    raise _Quit()
                if key is not None:
                    pending_keys.append(key)

            workers.start()
            startup_timeout = max(float(settings["arm"]["start_timeout_s"])
                                  for settings in configs.values()) + 10.0
            workers.collect("ready", 0, startup_timeout, check_keys)
            print("Both sides ready. R=re-anchor both, B=record, S=save, Q=quit.")
            while True:
                workers.check_health()
                key = pending_keys.popleft() if pending_keys else keyboard.read()
                if key == "q":
                    break
                if key == "b":
                    recorder.start()
                    print("Recording active.")
                elif key == "s":
                    saved = recorder.stop_and_save()
                    print(f"Recording saved: {saved}" if saved else "No active recording.")
                frames = _frames(source, configs)
                cycle += 1
                if any(frames.get(side) is None for side in SIDES):
                    if not workers.paused.is_set():
                        workers.paused.set()
                        print("PAUSED: Pico4 frame timeout; both sides require R after recovery.")
                    time.sleep(min(period, .02))
                    continue
                if key == "r":
                    workers.paused.set()
                    workers.request("anchor", cycle, frames)
                    try:
                        workers.collect("anchored", cycle, timeout, check_keys)
                    except _Pause as exc:
                        print(f"PAUSED: {exc}; press R with both hands tracked.")
                    else:
                        workers.paused.clear()
                        print("Both sides re-anchored; streaming resumed.")
                    continue
                if workers.paused.is_set():
                    time.sleep(min(period, .02))
                    continue
                started = time.perf_counter()
                observed_at = time.monotonic()
                image = camera.get_latest() if camera is not None and recorder.active else None
                stage = time.perf_counter()
                workers.request("prepare", cycle, frames)
                try:
                    workers.collect("prepared", cycle, timeout, check_keys)
                    prepare_wait = time.perf_counter() - stage
                    if not all(_fresh(frames[side], configs[side]) for side in SIDES):
                        raise _Pause("Prepared input expired; press R to re-anchor both sides")
                    stage = time.perf_counter()
                    workers.request("commit", cycle)
                    results = workers.collect("result", cycle, timeout, check_keys)
                    dispatch_wait = time.perf_counter() - stage
                except _Pause as exc:
                    workers.paused.set()
                    workers.request("cancel", cycle)
                    print(f"PAUSED: {exc}; press R to re-anchor both sides.")
                    continue
                row = dict(timestamp_monotonic=time.monotonic(),
                           observation_timestamp_monotonic=observed_at, cycle_id=cycle,
                           timing_prepare_wait_s=prepare_wait, timing_dispatch_wait_s=dispatch_wait)
                for side in SIDES:
                    row.update({f"{side}_{key}": value for key, value in results[side]["row"].items()})
                stage = time.perf_counter()
                recorder.append(camera_frame=image, **row)
                record_s = time.perf_counter() - stage
                work_s = time.perf_counter() - started
                remaining = period - work_s
                if remaining > 0:
                    time.sleep(remaining)
                cycle_s = time.perf_counter() - started
                timing.append(work_s, cycle_s, prepare_wait, dispatch_wait)
                for side in SIDES:
                    data = results[side]["row"]
                    timings[side].append(data, record_s, data["timing_prepare_s"] + data["timing_dispatch_s"],
                                         cycle_s, data)
    except (_Quit, KeyboardInterrupt):
        pass
    finally:
        try:
            workers.close()
        except Exception as exc:
            print(f"Worker shutdown failed: {type(exc).__name__}: {exc}")
        try:
            saved = recorder.stop_and_save()
            if saved:
                print(f"Active recording saved during shutdown: {saved}")
        except Exception as exc:
            print(f"Recording save failed during shutdown: {type(exc).__name__}: {exc}")
        for name, resource in (("Camera", camera), ("Pico4", source)):
            if resource is not None:
                try:
                    resource.stop()
                except Exception as exc:
                    print(f"{name} stop failed: {type(exc).__name__}: {exc}")
        timing.report()
        for side in SIDES:
            timings[side].report()
        print("Both control workers stopped; no automatic disable/e-stop was sent.")
