"""Isolate IK and hand optimization from input/camera/CAN Python threads.

One synchronous pipe per side, with at most one request in flight. A timeout is
fatal for this client: a late answer can never become a subsequent command.
The child owns mathematical models only and never constructs hardware objects.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time

import numpy as np

from robot_control.nero_ik import NeroIK

from .config import input_config
from .hand_retarget import L20Retargeter


def _compute_main(connection, config, use_arm, use_hand):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    request_id = 0
    try:
        ik = (NeroIK(max_joint_delta_rad=config["arm"]["max_joint_delta_rad"],
                     joint_limit_margin_rad=np.deg2rad(config["arm"].get("ik_joint_limit_margin_deg", 2.0)),
                     feedback_limit_tolerance_rad=np.deg2rad(
                         config["arm"].get("ik_feedback_limit_tolerance_deg", 1.0)))
              if use_arm else None)
        hand = config["hand"]
        retargeter = (L20Retargeter(hand["retarget_config"], input_config(config)["side"],
                                   hand["max_raw_step"]) if use_hand else None)
        connection.send(dict(id=0, value=None))
        while True:
            request = connection.recv()
            request_id = request["id"]
            method = request["method"]
            if method == "stop":
                break
            started, cpu_started = time.perf_counter(), time.thread_time()
            if method in ("forward", "solve") and ik is not None:
                model = ik
            elif method in ("reset", "retarget") and retargeter is not None:
                model = retargeter
            else:
                raise ValueError(f"Unavailable compute method: {method}")
            value = getattr(model, method)(*request["args"])
            elapsed, cpu = time.perf_counter() - started, time.thread_time() - cpu_started
            connection.send(dict(id=request_id, value=value, elapsed_s=elapsed, cpu_s=cpu,
                                 diagnostics=getattr(model, "last_diagnostics", {})))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(dict(id=request_id, error=f"{type(exc).__name__}: {exc}"))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class _RemoteIK:
    def __init__(self, owner):
        self.owner = owner
        self.last_diagnostics = {}

    def forward(self, joints):
        return self.owner.call("forward", joints)["value"]

    def solve(self, target, seed):
        reply = self.owner.call("solve", target, seed)
        self.last_diagnostics = reply["diagnostics"]
        return reply["value"]


class _RemoteHand:
    def __init__(self, owner):
        self.owner = owner
        self.last_diagnostics = {}

    def reset(self, previous_raw=None):
        self.owner.call("reset", previous_raw)

    def retarget(self, landmarks):
        reply = self.owner.call("retarget", landmarks)
        self.last_diagnostics = reply["diagnostics"]
        return reply["value"]


class ComputeProcess:
    def __init__(self, config, control, hand_mode="tracking", *, worker_target=_compute_main):
        self.config = config
        self.use_arm = control in ("arm", "both")
        self.use_hand = control in ("hand", "both") and hand_mode == "tracking"
        self.ik = _RemoteIK(self) if self.use_arm else None
        self.retargeter = _RemoteHand(self) if self.use_hand else None
        self.timeout_s = float(input_config(config)["max_age_s"])
        self._target = worker_target
        self._connection = self._process = None
        self._request_id = 0
        self._failed = False
        self.begin_cycle()

    def begin_cycle(self):
        self.timings = dict(timing_compute_ipc_s=0.0, timing_ik_cpu_s=0.0,
                            timing_hand_retarget_cpu_s=0.0)

    def start(self):
        if self._process is not None:
            raise RuntimeError("Compute process already started")
        # Finger replay without an arm has no optimization to isolate.
        if not (self.use_arm or self.use_hand):
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            name=f"teleop-compute-{input_config(self.config)['side']}", daemon=True,
            target=self._target, args=(child, self.config, self.use_arm, self.use_hand),
        )
        try:
            self._process.start()
        except BaseException:
            parent.close()
            self._connection = self._process = None
            raise
        finally:
            child.close()
        try:
            self._receive(0, 10.0)
        except BaseException:
            self.close()
            raise
        print(f"[{input_config(self.config)['side']}] IK/retarget compute process ready "
              f"(pid={self._process.pid}).", flush=True)

    def check_health(self):
        if self._failed:
            raise RuntimeError("Compute process is unavailable after a failed request")
        if self._process is not None and not self._process.is_alive():
            raise RuntimeError(f"Compute process exited ({self._process.exitcode})")

    def _receive(self, request_id, timeout_s):
        try:
            if not self._connection.poll(timeout_s):
                raise TimeoutError(f"Compute request {request_id} timed out after {timeout_s:g} s")
            reply = self._connection.recv()
            if reply["id"] != request_id:
                raise RuntimeError("Compute response does not match the current request")
            if "error" in reply:
                raise RuntimeError(f"Compute process: {reply['error']}")
            return reply
        except TimeoutError:
            self._failed = True
            raise
        except (EOFError, OSError) as exc:
            self._failed = True
            raise RuntimeError("Compute process disconnected") from exc
        except BaseException:
            self._failed = True
            raise

    def call(self, method, *args):
        self.check_health()
        if self._connection is None:
            raise RuntimeError("Compute process is not started")
        self._request_id += 1
        started = time.perf_counter()
        try:
            self._connection.send(dict(id=self._request_id, method=method, args=args))
        except (EOFError, OSError) as exc:
            self._failed = True
            raise RuntimeError("Compute process disconnected") from exc
        reply = self._receive(self._request_id, self.timeout_s)
        # Includes serialization and the I/O owner's scheduling delay on return.
        self.timings["timing_compute_ipc_s"] += max(
            0.0, time.perf_counter() - started - reply["elapsed_s"])
        stage = {"solve": "ik", "retarget": "hand_retarget"}.get(method)
        if stage:
            self.timings[f"timing_{stage}_s"] = reply["elapsed_s"]
            self.timings[f"timing_{stage}_cpu_s"] = reply["cpu_s"]
        return reply

    def close(self):
        if self._process is None:
            return
        try:
            if self._process.is_alive():
                try:
                    self._connection.send(dict(id=self._request_id + 1, method="stop"))
                except (EOFError, OSError):
                    pass
                self._process.join(1.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(1.0)
        finally:
            self._connection.close()
            self._connection = self._process = None
            self._failed = True
