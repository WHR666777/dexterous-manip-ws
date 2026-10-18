"""Local Codex app-server client and application-owned dynamic tools.

Protocol reference: https://learn.chatgpt.com/docs/app-server
Experimental dynamic-tool wire shapes were checked against codex-cli 0.162.1.
"""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import base64
import json
import subprocess
import time

from .geometry import calculate


INSTRUCTIONS = """You are the single manipulation agent for this application.
Complete one task within this thread. First inspect cloth_observe and make a
short stage list. Before each stage, compute geometry and nominal joint plans,
then submit an ordinary Python script through cloth_submit_subtask. The script
owns its main flow, supervision loops, and recovery branches. Do not define a
required policy/monitor callback template. Use only cloth tools to submit or
control motion; never open devices, edit the repository, restart programs, or
delegate to another agent. Read-only shell inspection is allowed if useful.

Generated scripts import cloth_agent.tools. Available interfaces:
parameters() -> initial params dict; observe(include_images=True) -> state;
execute_trajectory(plan_id, wait=True) -> MotionHandle;
move_J(arm, target_joints, wait=True), move_L(arm, target_tcp, wait=True);
plan_from_current(waypoints=None, arm='left', mode='cartesian',
                  target_joints=None) -> plan_id;
set_hand_pose(pose_or_preset, arm='left'); wait_next_observation(period_s=.05);
request_agent(reason, observation=None) -> reply dict;
report_result(status='completed', **result); geometry(operation, **values).
Handles provide status(), done(), wait(timeout=None), pause_and_hold(reason).
done() also covers paused/failed motions: call wait() to distinguish success.
MotionInterrupted and MotionFailed can be imported from cloth_agent.tools.
All data passed to tools must be JSON serializable. Positions are metres and
angles radians. TCP targets are 4x4 matrices or xyz/rpy lists in the task frame.

Plan complex nominal trajectories before script submission and pass plan IDs
through initial params. Motion is asynchronous with wait=False; while it runs,
the script reads observations, checks its own conditions, and requests pause.
Always pause before request_agent. The same script waits for new parameters,
replans from actual feedback in an existing branch, and continues. You may use
planning/calculation tools during recovery but cannot replace the live source.
On recovery, use cloth_reply_recovery with action continue or abort and data
the waiting program expects. Only one arm may move at a time. The other arm
retains its pose and grasp. Wait for subtask completion before the next script.
Use cloth_wait_event to receive completion, failure, and recovery requests.

Keypoint extraction, pinch hand poses and physical calibration are pending.
Never invent them, or interpret mock images/joints as proof of physical cloth
grasping or folding. If the requested physical task lacks required inputs,
use cloth_abort_task with the specific missing inputs. Planning failure is
feedback for waypoint changes, not proof of global unreachability.
Finish a successful task using cloth_finish_task only after all programs have
completed, with an evidence-based result. Failures must be reported truthfully.
"""


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def tool_specs():
    string = {"type": "string"}
    obj = {"type": "object", "additionalProperties": True}
    tools = [
        ("cloth_observe", "Get current device, RGB-D and optional keypoint observations.",
         _schema({"include_images": {"type": "boolean"}})),
        ("cloth_geometry", "Compute a transform, TCP-to-flange pose, distance, offset or pinhole projection.",
         _schema({"operation": string, "values": obj}, ["operation", "values"])),
        ("cloth_plan_motion", "Plan from current feedback. Supply TCP waypoints OR target_joints. Returns a plan ID or failure diagnostics.",
         _schema({"arm": {"type": "string", "enum": ["left", "right"]},
                  "waypoints": {"type": "array", "items": {}},
                  "mode": {"type": "string", "enum": ["joint", "cartesian"]},
                  "target_joints": {"type": "array", "items": {"type": "number"}}}, ["arm"])),
        ("cloth_submit_subtask", "Launch an ordinary Python script. The active program cannot be replaced, including while waiting for recovery.",
         _schema({"source": string, "params": obj, "name": string}, ["source"])),
        ("cloth_wait_event", "Wait for script completion or a recovery request. Motion holds while recovery is pending.",
         _schema({"timeout_s": {"type": "number", "minimum": 0, "maximum": 60}})),
        ("cloth_reply_recovery", "Reply with data to the existing program. reply.action is continue or abort.",
         _schema({"request_id": string, "reply": obj}, ["request_id", "reply"])),
        ("cloth_task_status", "Get task and subtask results.", _schema()),
        ("cloth_abort_task", "Abandon the task, pause devices and end the current program.",
         _schema({"reason": string}, ["reason"])),
        ("cloth_finish_task", "Finish after successful program execution; summarize evidence and its limits.",
         _schema({"result": obj})),
    ]
    return [{"type": "function", "name": name, "description": description,
             "inputSchema": schema} for name, description, schema in tools]


class CodexAppServer:
    def __init__(self, runtime, *, progress=None):
        self.runtime = runtime
        self.config = runtime.config["codex"]
        self.progress = progress or (lambda text: print(text, end="", flush=True))
        self._messages = Queue()
        self._responses = {}
        self._next_id = 0
        self.thread_id = None
        self.process = None

    def start(self):
        stderr = (self.runtime.directory / "codex-stderr.log").open("w")
        try:
            self.process = subprocess.Popen(
                [self.config["executable"], *self.config.get("extra_args", []),
                 "app-server", "--listen", "stdio://"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1)
        finally:
            stderr.close()
        self._reader = Thread(target=self._read_messages, name="codex-json-rpc", daemon=True)
        self._reader.start()
        self.request("initialize", {"clientInfo": {"name": "cloth_agent", "title": "Cloth Agent",
                                                    "version": "0.1.0"},
                                    "capabilities": {"experimentalApi": True}})
        self._send({"method": "initialized", "params": {}})
        params = {"cwd": str(Path(__file__).resolve().parents[1]), "approvalPolicy": "never",
                  "sandbox": "read-only", "dynamicTools": tool_specs(),
                  "developerInstructions": INSTRUCTIONS}
        if self.config.get("model"):
            params["model"] = self.config["model"]
        self.thread_id = self.request("thread/start", params)["thread"]["id"]
        (self.runtime.directory / "thread.json").write_text(json.dumps({"thread_id": self.thread_id}))

    def _read_messages(self):
        try:
            for line in self.process.stdout:
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self._messages.put({"client_error": "Non-JSON app-server stdout."})
        finally:
            self._messages.put({"client_error": "Codex app-server closed its stdout."})

    def _send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _receive(self, timeout=None):
        try:
            message = self._messages.get(timeout=timeout or self.config["request_timeout_s"])
        except Empty as exc:
            raise TimeoutError("Codex app-server response timed out.") from exc
        if "client_error" in message:
            raise RuntimeError(message["client_error"] + " See codex-stderr.log.")
        return message

    def request(self, method, params):
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.config["request_timeout_s"]
        while request_id not in self._responses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Codex RPC {method} timed out.")
            self._route(self._receive(remaining))
        reply = self._responses.pop(request_id)
        if "error" in reply:
            raise RuntimeError(f"Codex RPC {method}: {reply['error']}")
        return reply["result"]

    def _route(self, message):
        if "method" not in message:
            if "id" in message:
                self._responses[message["id"]] = message
            return
        method, params = message["method"], message.get("params", {})
        if "id" in message:
            if method == "item/tool/call":
                self._handle_tool(message["id"], params)
            else:
                self._send({"id": message["id"], "error": {
                    "code": -32601, "message": f"Interactive request unsupported in automatic mode: {method}"}})
        elif method == "item/agentMessage/delta":
            self.progress(params.get("delta", ""))

    def _dispatch(self, name, args):
        calls = {
            "cloth_observe": self.runtime.observe,
            "cloth_plan_motion": self.runtime.plan_motion,
            "cloth_submit_subtask": self.runtime.submit_subtask,
            "cloth_wait_event": self.runtime.wait_event,
            "cloth_reply_recovery": self.runtime.reply_recovery,
            "cloth_task_status": self.runtime.summary,
            "cloth_abort_task": self.runtime.abort_task,
            "cloth_finish_task": self.runtime.finish_task,
        }
        if name == "cloth_geometry":
            return calculate(args["operation"], **args["values"])
        if name not in calls:
            raise ValueError(f"Unknown application tool: {name}")
        self.runtime.emit({"type": "agent_tool_called", "tool": name})
        return calls[name](**args)

    @staticmethod
    def image_content(result):
        if not isinstance(result, dict):
            return []
        state = result.get("observation", result)
        image = state.get("images") or {}
        if image.get("available") and image.get("rgb_path"):
            data = base64.b64encode(Path(image["rgb_path"]).read_bytes()).decode()
            return [{"type": "inputImage", "imageUrl": "data:image/png;base64," + data}]
        return []

    def _handle_tool(self, request_id, params):
        try:
            name = params["tool"]
            args = params["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            self.progress(f"\n[tool] {name}\n")
            result = self._dispatch(name, args)
            content = [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}]
            content.extend(self.image_content(result))
            reply = {"contentItems": content, "success": True}
        except Exception as exc:
            reply = {"contentItems": [{"type": "inputText", "text":
                                       f"{type(exc).__name__}: {exc}"}], "success": False}
        self._send({"id": request_id, "result": reply})

    def _start_turn(self, text, image_path=None):
        inputs = [{"type": "text", "text": text}]
        if image_path:
            inputs.append({"type": "localImage", "path": image_path})
        self.request("turn/start", {"threadId": self.thread_id, "input": inputs})

    def run(self, task):
        try:
            self.start()
            self._start_turn(task)
            while True:
                message = self._receive()
                if message.get("method") != "turn/completed":
                    self._route(message)
                    continue
                turn = message["params"]["turn"]
                if turn["status"] != "completed":
                    raise RuntimeError(f"Codex turn ended with {turn['status']}: {turn.get('error')}")
                if self.runtime.task_status != "running":
                    return self.runtime.summary()
                subtask = self.runtime._subtask
                if subtask is None or subtask.done.is_set():
                    raise RuntimeError("Agent ended its turn without finishing or abandoning the task.")
                event = self.runtime.wait_event(60)
                while event["type"] == "running":
                    event = self.runtime.wait_event(60)
                image = (event.get("observation", {}).get("images") or {}).get("rgb_path")
                self._start_turn("Program event; continue this same task and thread:\n"
                                 + json.dumps(event, ensure_ascii=False), image)
        except BaseException:
            if self.runtime.task_status == "running":
                self.runtime.abort_task("Codex connection or task orchestration failed")
            raise
        finally:
            self.close()

    def close(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            self._reader.join(timeout=2)

