"""Persistent devices and a small local bridge for ordinary Python scripts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition, Event, Lock, RLock, Thread
import json
import os
import secrets
import subprocess
import sys
import time
import uuid

import numpy as np

from .backends import HardwareBackend, MockBackend
from .executor import Executor, MotionFailed, MotionInterrupted
from .geometry import calculate
from .observations import Observations
from .planning import Planner, PlanningError


@dataclass
class Subtask:
    id: str
    directory: Path
    process: subprocess.Popen | None = None
    status: str = "running"
    result: dict | None = None
    reason: str = ""
    watcher: Thread | None = None
    done: Event = field(default_factory=Event)

    def snapshot(self):
        return {"subtask_id": self.id, "status": self.status, "result": self.result,
                "reason": self.reason, "directory": str(self.directory),
                "pid": self.process.pid if self.process else None}


class Runtime:
    def __init__(self, config, output_dir=None, *, backend=None, keypoints=None,
                 allow_hardware=False):
        self.config = config
        if config["backend"] == "hardware" and not allow_hardware:
            raise ValueError("Hardware backend requires the explicit --execute startup gate.")
        root = Path(output_dir or config["output_dir"]).resolve()
        if output_dir is None:
            root = root / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root
        self._events = deque()
        self._events_condition = Condition()
        self._log_lock = Lock()
        self._task_lock = RLock()
        self._recovery_condition = Condition()
        self._pending_recovery = None
        self._replies = {}
        self._plans = {}
        self._subtask = None
        self._subtasks = []
        self._closed = False
        self.task_status = "running"
        self.task_result = None
        self._token = secrets.token_hex(24)
        self.backend = backend or (MockBackend(config) if config["backend"] == "mock"
                                   else HardwareBackend(config))
        self.observations = self.executor = None
        try:
            self.planner = Planner(config)
            self.observations = Observations(config, self.backend, root, keypoints)
            self.executor = Executor(self.backend, config, self.emit)
            self._start_bridge()
        except BaseException:
            try:
                if self.executor is not None:
                    self.executor.close()
                if self.observations is not None:
                    self.observations.close()
            finally:
                self.backend.close()
            raise
        self.emit({"type": "task_started", "backend": config["backend"]})

    def emit(self, event):
        event = {"timestamp_monotonic": time.monotonic(), **event}
        with self._log_lock:
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        # Motion events remain in the log, but only decisions wake the agent.
        if event["type"] in {"recovery_requested", "subtask_completed", "subtask_failed",
                             "subtask_aborted", "executor_failed", "task_aborted"}:
            with self._events_condition:
                self._events.append(event)
                self._events_condition.notify_all()

    def observe(self, include_images=True):
        value = self.observations.snapshot(include_images)
        value["motion"] = self.executor.latest()
        with self._task_lock:
            value["subtask"] = self._subtask.snapshot() if self._subtask else None
        return value

    def plan_motion(self, arm="left", waypoints=None, mode="cartesian", target_joints=None):
        if self.task_status != "running":
            raise RuntimeError("Task has ended.")
        try:
            plan = self.planner.plan(arm, self.backend.get_joints(arm), waypoints=waypoints,
                                     mode=mode, target_joints=target_joints)
        except PlanningError as exc:
            self.emit({"type": "planning_failed", "arm": arm, **exc.as_dict()})
            return exc.as_dict()
        plan_id = uuid.uuid4().hex
        path = self.directory / "plans" / f"{plan_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(plan.as_dict(), allow_nan=False))
        self._plans[plan_id] = plan
        result = {"ok": True, "plan_id": plan_id, "arm": arm, "mode": plan.mode,
                  "duration": plan.duration, "samples": len(plan.times), "path": str(path)}
        self.emit({"type": "planning_completed", **result})
        return result

    def execute(self, plan_id):
        if self.task_status != "running":
            raise RuntimeError("Task has ended.")
        with self._recovery_condition:
            if self._pending_recovery is not None:
                raise RuntimeError("A recovery reply is still pending.")
        return self.executor.start(self._plans[plan_id])

    def geometry(self, operation, **values):
        if operation in ("camera_to_task", "pixel_to_task"):
            values["task_from_camera"] = self.config["observations"].get("task_from_camera")
        return calculate(operation, **values)

    def set_hand_pose(self, arm, pose):
        if self.task_status != "running":
            raise RuntimeError("Task has ended.")
        if isinstance(pose, str):
            raw = self.config["arms"][arm]["hand_poses"].get(pose)
            if raw is None:
                raise ValueError(f"Hand pose {arm}.{pose} has not been determined.")
            pose = raw
        self.executor.set_hand(arm, pose)
        self.emit({"type": "hand_command", "arm": arm})
        return {"status": "sent", "arm": arm}

    def submit_subtask(self, source: str, params=None, name="subtask"):
        # Syntax checking happens before starting the script or moving a robot.
        compile(source, "<subtask>", "exec")
        with self._task_lock:
            if self.task_status != "running":
                raise RuntimeError("Task has ended.")
            if self._subtask and not self._subtask.done.is_set():
                raise RuntimeError("A subtask is still running; its program cannot be replaced.")
            subtask_id = uuid.uuid4().hex
            directory = self.directory / f"subtask-{len(self._subtasks) + 1:03d}"
            directory.mkdir()
            program = directory / "program.py"
            program.write_text(source)
            params_path = directory / "params.json"
            params_path.write_text(json.dumps(params or {}, ensure_ascii=False, allow_nan=False))
            # Read once by Python on startup; recovery never reloads this source.
            program.chmod(0o444)
            params_path.chmod(0o444)
            env = os.environ.copy()
            env.update(CLOTH_AGENT_URL=self.url, CLOTH_AGENT_TOKEN=self._token,
                       CLOTH_AGENT_SUBTASK=subtask_id, CLOTH_AGENT_PARAMS=str(params_path),
                       PYTHONUNBUFFERED="1")
            project = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project + os.pathsep + env.get("PYTHONPATH", "")
            subtask = Subtask(subtask_id, directory)
            self._subtask = subtask
            self._subtasks.append(subtask)
            with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                subtask.process = subprocess.Popen([sys.executable, str(program)], env=env,
                                                   cwd=directory, stdout=stdout, stderr=stderr)
            subtask.watcher = Thread(target=self._watch, args=(subtask,),
                                     name="cloth-script-watcher", daemon=True)
            subtask.watcher.start()
            self.emit({"type": "subtask_started", "name": name, **subtask.snapshot()})
            return subtask.snapshot()

    def _watch(self, subtask):
        code = subtask.process.wait()
        running_motion = self.executor.active()
        reason = ""
        if running_motion:
            reason = "Program exited before its motion finished."
        if code != 0:
            reason = f"Program exited with code {code}; see stderr.log."
        try:
            self.executor.pause_and_hold(reason=reason or "subtask ended")
        except Exception as exc:
            reason = (reason + " " + str(exc)).strip()
        with self._recovery_condition:
            if self._pending_recovery and self._pending_recovery["subtask_id"] == subtask.id:
                reason = reason or "Program exited while waiting for recovery."
                self._pending_recovery = None
        with self._task_lock:
            if self.task_status == "aborted" or subtask.status == "aborted":
                subtask.status = "aborted"
                self.task_status = "aborted"
                self.task_result = self.task_result or {"reason": "Program abandoned the task."}
            elif reason or subtask.status == "failed":
                subtask.status = "failed"
                # Failure ends this attempt. It cannot be relabeled as a new
                # stage to edit the code and retry within the same task.
                self.task_status = "failed"
                self.task_result = {"reason": reason or subtask.reason}
            else:
                subtask.status = "completed"
            subtask.reason = reason or subtask.reason
            subtask.done.set()
            result = subtask.snapshot()
        with self._recovery_condition:
            self._recovery_condition.notify_all()
        (subtask.directory / "result.json").write_text(json.dumps(result, ensure_ascii=False))
        self.emit({"type": "subtask_" + subtask.status, **result})

    def request_agent(self, reason, observation=None):
        self.executor.pause_and_hold(reason=reason)
        state = self.observe()  # Feedback and RGB-D after hold acknowledgement.
        with self._task_lock:
            subtask = self._subtask
            if not subtask or subtask.done.is_set():
                raise RuntimeError("No live subtask is waiting for recovery.")
        request_id = uuid.uuid4().hex
        event = {"type": "recovery_requested", "request_id": request_id,
                 "subtask_id": subtask.id, "reason": reason, "observation": state}
        if observation is not None:
            event["trigger_observation"] = observation
        with self._recovery_condition:
            if self._pending_recovery:
                raise RuntimeError("Another recovery request is already pending.")
            self._pending_recovery = event
            self.emit(event)
            self._recovery_condition.wait_for(lambda: request_id in self._replies
                                              or self.task_status != "running")
            reply = self._replies.pop(request_id, {"action": "abort"})
            self._pending_recovery = None
            return reply

    def reply_recovery(self, request_id, reply):
        if reply.get("action") not in ("continue", "abort"):
            raise ValueError("Recovery action must be continue or abort; other fields are parameters.")
        with self._recovery_condition:
            if not self._pending_recovery or self._pending_recovery["request_id"] != request_id:
                raise ValueError("Recovery request is not pending.")
            if request_id in self._replies:
                raise ValueError("Recovery request has already been answered.")
            self._replies[request_id] = reply
            if reply["action"] == "abort":
                self.task_status = "aborted"
            self.emit({"type": "recovery_replied", "request_id": request_id, "reply": reply})
            self._recovery_condition.notify_all()
        return {"status": "accepted", "request_id": request_id}

    def wait_event(self, timeout_s=30):
        timeout_s = float(timeout_s)
        if not np.isfinite(timeout_s) or timeout_s < 0 or timeout_s > 60:
            raise ValueError("Event timeout must be between 0 and 60 seconds.")
        with self._events_condition:
            self._events_condition.wait_for(lambda: bool(self._events), timeout_s)
            if self._events:
                return self._events.popleft()
        return {"type": "running", "motion": self.executor.active(),
                "subtask": self._subtask.snapshot() if self._subtask else None}

    def report_result(self, status="completed", result=None):
        if status not in ("completed", "aborted", "failed"):
            raise ValueError("Invalid subtask result status.")
        with self._task_lock:
            if not self._subtask:
                raise RuntimeError("No subtask.")
            # 'completed' is finalized only after process exit and motion checks.
            self._subtask.result = result or {}
            if status != "completed":
                self._subtask.status = status
        return {"status": "recorded"}

    def finish_task(self, result=None):
        with self._task_lock:
            if self._subtask and not self._subtask.done.is_set():
                raise RuntimeError("Wait for the current program before finishing the task.")
            if not self._subtasks or any(item.status != "completed" for item in self._subtasks):
                raise RuntimeError("A task with no completed programs or failed/aborted programs cannot be successful.")
            if self.task_status != "running":
                raise RuntimeError("Task has already ended.")
            self.task_status = "completed"
            self.task_result = result or {}
            summary = self.summary()
        (self.directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False))
        self.emit({"type": "task_completed", "result": self.task_result})
        return summary

    def abort_task(self, reason="agent abandoned task"):
        hold_error = None
        try:
            self.executor.pause_and_hold(reason=reason)
        except Exception as exc:
            # Abandonment must still terminate the program when the device
            # connection itself has failed. Report that hold was not confirmed.
            hold_error = str(exc)
        self.task_status = "aborted"
        self.task_result = {"reason": reason}
        if hold_error:
            self.task_result["hold_error"] = hold_error
        with self._recovery_condition:
            self._recovery_condition.notify_all()
        with self._task_lock:
            subtask = self._subtask
            if subtask and not subtask.done.is_set():
                subtask.status = "aborted"
                process = subtask.process
            else:
                process = None
        if process and process.poll() is None:
            # Termination is task abandonment, never an edit-and-retry path.
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self.emit({"type": "task_aborted", "reason": reason, "hold_error": hold_error})
        return self.summary()

    def summary(self):
        return {"status": self.task_status, "result": self.task_result,
                "directory": str(self.directory), "backend": self.config["backend"],
                "subtasks": [item.snapshot() for item in self._subtasks]}

    def _start_bridge(self):
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    if self.headers.get("Authorization") != "Bearer " + runtime._token:
                        raise PermissionError("Invalid script session.")
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    with runtime._task_lock:
                        current = runtime._subtask
                        if (not current or current.id != request["subtask_id"]
                                or current.done.is_set()):
                            raise PermissionError("Script is not the active subtask.")
                    result = runtime._script_call(request["method"], request.get("params", {}))
                    body = {"ok": True, "result": result}
                except Exception as exc:
                    body = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._server_thread = Thread(target=self._server.serve_forever,
                                     name="cloth-script-bridge", daemon=True)
        self._server_thread.start()

    def _script_call(self, method, params):
        methods = {
            "observe": self.observe,
            "plan_motion": self.plan_motion,
            "execute": self.execute,
            "motion_status": self.executor.status,
            "wait_motion": self.executor.wait,
            "pause_and_hold": self.executor.pause_and_hold,
            "set_hand_pose": self.set_hand_pose,
            "request_agent": self.request_agent,
            "report_result": self.report_result,
            "calculate": self.geometry,
        }
        if method not in methods:
            raise ValueError(f"Unknown script tool: {method}")
        return methods[method](**params)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._subtask and not self._subtask.done.is_set():
                self.abort_task("runtime closing")
            if self._subtask and self._subtask.watcher:
                self._subtask.watcher.join(timeout=3)
        finally:
            try:
                self.executor.close()
            finally:
                self._server.shutdown()
                self._server.server_close()
                self._server_thread.join(timeout=2)
                try:
                    self.observations.close()
                finally:
                    self.backend.close()
        (self.directory / "summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
