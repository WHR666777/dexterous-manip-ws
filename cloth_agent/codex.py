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

INSTRUCTIONS = """你是本应用唯一的操作 agent，在同一个 conversation thread 中完成一个任务。

语言 Language：所有对用户可见的计划、回复、代码注释、docstring、print/log、
异常原因 reason、子任务名称和 result 中的说明性文字，均使用中文为主、夹杂 English
技术术语的形式，便于用户阅读。函数名、变量名、工具名、JSON 字段及枚举值保留原样，
例如 status='completed'、action='continue'、mode='cartesian' 不翻译。
无需逐句重复中英文；优先写“暂停并保持 pause-and-hold”这样的自然表达。

工作流 Workflow：先调用 cloth_observe，列出简短的子任务 stage list。
每个 stage 开始前先完成几何计算和 nominal joint plan，再通过 cloth_submit_subtask
提交一份普通 Python 程序。程序自行组织主流程、监督循环和已有恢复分支，不要求固定的
policy/monitor 回调模板。运动提交和控制只能通过 cloth tools；不得直接打开设备、
修改仓库、重启程序或委托其他 agent。需要时允许 read-only shell 检查。

生成的程序 import cloth_agent.tools，可用接口 Interfaces：
parameters() -> 初始 params dict；observe(include_images=True) -> 当前 state；
execute_trajectory(plan_id, wait=True) -> MotionHandle;
move_J(arm, target_joints, wait=True), move_L(arm, target_tcp, wait=True);
plan_from_current(waypoints=None, arm='left', mode='cartesian',
                  target_joints=None) -> plan_id;
set_hand_pose(pose_or_preset, arm='left'); wait_next_observation(period_s=.05);
request_agent(reason, observation=None) -> reply dict;
report_result(status='completed', **result); geometry(operation, **values).
句柄 handle 提供 status()、done()、wait(timeout=None)、pause_and_hold(reason)。
done() 也包括 paused/failed，必须调用 wait() 确认是否成功；可从 cloth_agent.tools
导入 MotionInterrupted 和 MotionFailed。工具参数必须 JSON serializable；
位置单位 m，角度单位 rad；TCP 目标是 task frame 中的 4x4 矩阵或 xyz/rpy 列表。

执行与恢复 Execution/recovery：复杂轨迹在提交源码前规划好，通过 initial params
传入 plan_id。wait=False 异步执行时，程序持续读取 observation、自行检查条件并请求暂停。
request_agent 前必须先暂停。原 Python 程序保持存活，在既有分支中等待新参数、根据实际
feedback 重新规划并继续；恢复期间可调用 planning/geometry 工具，不得替换正在运行的源码。
通过 cloth_reply_recovery 回复 action='continue' 或 'abort'，其余字段匹配程序约定。
一次只能有一个臂运动，另一臂保持姿态和抓持。子任务完成后才能提交下一份程序。
用 cloth_wait_event 等待完成、失败或 recovery request。

坐标与输入 Frames/inputs：只能使用 cloth_observe 列出的机械臂，调用时显式指定 arm。
已配置的 reference_tcp_rotation 是默认拈布朝向；yaw 调整绕 task +Z 旋转。
camera_to_task(points=...) 和 pixel_to_task(pixel=..., depth_m=..., intrinsics=...)
使用已配置的 camera calibration，返回 task frame 坐标。深度和内参必须取自所点选的
同一份 RGB-D 图像，不得换用更新帧；传入的 depth 单位为 m。
pixel_to_point 返回 camera frame 坐标。关键点提取是外部模块；使用已有 calibration
和 hand presets，不编造缺失输入，也不把 mock 图像/关节结果当作真实抓布或折布成功。
实机任务缺少必要输入时调用 cloth_abort_task，明确说明缺少什么。
规划失败用于修改 waypoint 后重试，不代表目标在全局不可达。

任务收尾 Finish：所有程序正常完成后，才用 cloth_finish_task 提交有证据的结果。
如实报告失败以及尚未验证的物理效果。
"""


def _schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def tool_specs():
    string = {"type": "string"}
    obj = {"type": "object", "additionalProperties": True}
    tools = [
        ("cloth_observe", "读取当前设备 state、RGB-D 和可选 keypoints 观测。",
         _schema({"include_images": {"type": "boolean"}})),
        ("cloth_geometry", "计算 transform、TCP→flange、距离或反投影；camera_to_task(points) / pixel_to_task(pixel,depth_m,intrinsics) 自动使用相机外参。",
         _schema({"operation": string, "values": obj}, ["operation", "values"])),
        ("cloth_plan_motion", "从实际 feedback 规划运动；传 TCP waypoints 或 target_joints 二者之一，返回 plan_id 或失败 diagnostics。",
         _schema({"arm": {"type": "string", "enum": ["left", "right"]},
                  "waypoints": {"type": "array", "items": {}},
                  "mode": {"type": "string", "enum": ["joint", "cartesian"]},
                  "target_joints": {"type": "array", "items": {"type": "number"}}}, ["arm"])),
        ("cloth_submit_subtask", "启动普通 Python 子任务程序；运行中或等待 recovery 时都不能替换源码。",
         _schema({"source": string, "params": obj, "name": string}, ["source"])),
        ("cloth_wait_event", "等待程序完成或 recovery request；等待 agent 恢复回复期间保持当前姿态。",
         _schema({"timeout_s": {"type": "number", "minimum": 0, "maximum": 60}})),
        ("cloth_reply_recovery", "向存活的原程序回复恢复参数；reply.action 为 continue 或 abort。",
         _schema({"request_id": string, "reply": obj}, ["request_id", "reply"])),
        ("cloth_task_status", "读取 task 和 subtask 的状态与结果。", _schema()),
        ("cloth_abort_task", "放弃整个 task，暂停设备并结束当前程序。",
         _schema({"reason": string}, ["reason"])),
        ("cloth_finish_task", "所有程序成功后结束 task；用中英说明总结证据和验证范围。",
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
        self.record_path = self.runtime.directory / "codex.md"
        self._agent_message_open = False

    def _append_record(self, text):
        with self.record_path.open("a", encoding="utf-8") as stream:
            stream.write(text)

    def _end_agent_message(self):
        if self._agent_message_open:
            self._append_record("\n\n")
            self._agent_message_open = False

    def _record(self, title, body, language=None):
        self._end_agent_message()
        if not isinstance(body, str):
            body = json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False)
        if language:
            body = f"````{language}\n{body}\n````"
        self._append_record(f"## {title}\n\n{body}\n\n")

    def start(self):
        self.record_path.write_text("# Codex 运行记录 Run record\n\n", encoding="utf-8")
        self._record("开发者提示词 Developer instructions", INSTRUCTIONS)
        self._record("工具定义 Tool definitions", tool_specs(), "json")
        stderr = (self.runtime.directory / "codex-stderr.log").open("w")
        try:
            self.process = subprocess.Popen(
                [self.config["executable"], *self.config.get("extra_args", []),
                 "app-server", "--listen", "stdio://"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", bufsize=1)
        finally:
            stderr.close()
        self._reader = Thread(target=self._read_messages, name="codex-json-rpc", daemon=True)
        self._reader.start()
        self.request("initialize", {"clientInfo": {"name": "cloth_agent", "title": "折布助手 Cloth Agent",
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
        self._record("对话标识 Thread", {"thread_id": self.thread_id}, "json")

    def _read_messages(self):
        try:
            for line in self.process.stdout:
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self._messages.put({"client_error": "app-server 标准输出不是 JSON Non-JSON app-server stdout."})
        finally:
            self._messages.put({"client_error": "Codex app-server 已关闭输出 stdout closed."})

    def _send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _receive(self, timeout=None):
        try:
            message = self._messages.get(timeout=timeout or self.config["request_timeout_s"])
        except Empty as exc:
            raise TimeoutError("等待 Codex app-server 回复超时 response timed out.") from exc
        if "client_error" in message:
            raise RuntimeError(message["client_error"] + " 请查看 see codex-stderr.log.")
        return message

    def request(self, method, params):
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.config["request_timeout_s"]
        while request_id not in self._responses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Codex RPC 请求超时 timed out: {method}")
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
                    "code": -32601, "message": f"自动模式不支持交互请求 Interactive request unsupported: {method}"}})
        elif method == "item/agentMessage/delta":
            delta = params.get("delta", "")
            if not self._agent_message_open:
                self._append_record("## Codex 回复 Agent message\n\n")
                self._agent_message_open = True
            self._append_record(delta)
            self.progress(delta)

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
            return self.runtime.geometry(args["operation"], **args["values"])
        if name not in calls:
            raise ValueError(f"未知应用工具 Unknown application tool: {name}")
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
            self.progress(f"\n[工具 Tool] {name}\n")
            self._record(f"工具调用 Tool call：{name}", args, "json")
            if name == "cloth_submit_subtask" and args.get("source"):
                self._record("提交的程序 Subtask source", args["source"], "python")
            result = self._dispatch(name, args)
            self._record(f"工具结果 Tool result：{name}", result, "json")
            content = [{"type": "inputText", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}]
            content.extend(self.image_content(result))
            reply = {"contentItems": content, "success": True}
        except Exception as exc:
            error = f"工具执行失败 Tool failed: {type(exc).__name__}: {exc}"
            self._record("工具错误 Tool error", error)
            reply = {"contentItems": [{"type": "inputText", "text":
                                       error}], "success": False}
        self._send({"id": request_id, "result": reply})

    def _start_turn(self, text, image_path=None):
        self._record("发送给 Codex 的消息 Turn input", text)
        if image_path:
            self._record("图像输入 Image input", str(image_path))
        inputs = [{"type": "text", "text": text}]
        if image_path:
            inputs.append({"type": "localImage", "path": image_path})
        self.request("turn/start", {"threadId": self.thread_id, "input": inputs})

    def run(self, task):
        try:
            self.start()
            self._start_turn("用户任务 Task：\n" + task)
            while True:
                message = self._receive()
                if message.get("method") != "turn/completed":
                    self._route(message)
                    continue
                turn = message["params"]["turn"]
                self._record("回合结束 Turn completed", turn, "json")
                if turn["status"] != "completed":
                    raise RuntimeError(f"Codex 回合异常结束 turn ended: {turn['status']}: {turn.get('error')}")
                if self.runtime.task_status != "running":
                    return self.runtime.summary()
                subtask = self.runtime._subtask
                if subtask is None or subtask.done.is_set():
                    raise RuntimeError("Agent 未完成或放弃任务便结束回合 turn ended without task completion/abandonment.")
                event = self.runtime.wait_event(60)
                while event["type"] == "running":
                    event = self.runtime.wait_event(60)
                image = (event.get("observation", {}).get("images") or {}).get("rgb_path")
                self._start_turn("程序事件 Program event；请继续同一个 task 和 thread：\n"
                                 + json.dumps(event, ensure_ascii=False), image)
        except BaseException as exc:
            self._record("运行错误 Run error", f"{type(exc).__name__}: {exc}")
            if self.runtime.task_status == "running":
                self.runtime.abort_task("Codex 连接或任务调度失败 connection/task orchestration failed")
            raise
        finally:
            self._record("任务状态 Task summary", self.runtime.summary(), "json")
            self.close()

    def close(self):
        self._end_agent_message()
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

