"""任务协调：把 MCP、worker、控制进程的生命周期连接起来。

本模块只管理执行边界与记录，不规划折布，也不提供 robo。
控制进程先锁定人工急停，再由本模块结束 worker/Codex；两者没有反向等待。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
from threading import Event, Lock, Thread
import time
import uuid

from .control import ControlProcess
from .worker import PythonWorker, WorkerUnavailable


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("时间必须为有限正数 Timeout must be finite and positive")
    return value


class TaskSession:
    """单任务协调器。execute 串行；abort/人工急停监测不等待 cell 的执行锁。"""

    def __init__(self, output_dir=None, *, cell_timeout_s=60., task_timeout_s=900., max_cells=40):
        self.cell_timeout_s = positive_seconds(cell_timeout_s)
        self.task_timeout_s = positive_seconds(task_timeout_s)
        if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
            raise ValueError("cell 次数必须为正整数 max_cells must be a positive integer")
        self.max_cells = max_cells
        self.directory = Path(output_dir or (
            Path("data/agent_runs") / ("session-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                                      + "-" + uuid.uuid4().hex[:8]))).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        # 不覆盖旧记录；新任务使用新目录，方便逐次 review。
        self._events_file = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        (self.directory / "cells").mkdir()
        self._state_lock, self._cell_lock, self._log_lock = Lock(), Lock(), Lock()
        self._shutdown = Event()
        self.ended = Event()
        self.status = "running"
        self.reason = ""
        self.report = None
        self.cell_count = 0
        self.agent = None
        self.control = self.worker = None
        self._started_at = time.monotonic()
        try:
            self.control = ControlProcess()
            self.worker = PythonWorker(self.directory / "workspace")
            metadata = {"coordinator_pid": os.getpid(), "worker_pid": self.worker.process.pid,
                        "control_pid": self.control.process.pid,
                        "estop_socket": self.control.ready["estop_socket"],
                        "hardware_connected": False, "robo_available": False}
            self._write("session.json", metadata)
            self.log("session_started", "任务框架已启动 Session started; robo unavailable", **metadata)
        except BaseException:
            if self.worker:
                self.worker.stop()
            if self.control:
                self.control.close()
            self._events_file.close()
            raise
        self._monitor = Thread(target=self._watch, name="cloth-session-monitor", daemon=True)
        self._monitor.start()

    def _write(self, relative_path, value):
        (self.directory / relative_path).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(self, kind, description, **values):
        # 标识符保持英文便于解析；description 使用中英夹杂，原始源码和输出另存。
        event = {"type": kind, "description": description,
                 "timestamp_monotonic": time.monotonic(), **values}
        with self._log_lock:
            self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_file.flush()

    def summary(self):
        with self._state_lock:
            return {"status": self.status, "reason": self.reason, "report": self.report,
                    "cells": self.cell_count, "directory": str(self.directory),
                    "hardware_connected": False, "stop_confirmed": False,
                    "description": "仅实现进程框架 Process framework only; robo not implemented"}

    def execute(self, source):
        if not isinstance(source, str):
            raise ValueError("cell 源码必须是文本 Cell source must be text")
        with self._cell_lock:
            with self._state_lock:
                if self.status != "running":
                    return {"status": "rejected", "reason": self.reason or "任务已结束 Task ended"}
                exhausted = self.cell_count >= self.max_cells
                if not exhausted:
                    self.cell_count += 1
                cell_id = self.cell_count
            if exhausted:
                self.abort("cell 调用次数用尽 Cell budget exhausted")
                return {"status": "rejected", "reason": self.reason}
            prefix = f"cells/{cell_id:03d}"
            (self.directory / (prefix + ".py")).write_text(source, encoding="utf-8")
            self.log("cell_started", "开始执行 cell Cell started", cell_id=cell_id)
            try:
                result = self.worker.execute(source, cell_id, self.cell_timeout_s)
                # 本轮控制进程没有运动。保留 cell 收尾的位置，后续在设备层实现后，
                # 必须先确认没有残留推进动作，再允许结果返回 Codex。
                control = self.control.request("cell_end")
                result["control"] = control
                if control["estop_latched"]:
                    self.abort("人工急停 Manual e-stop", status="estopped")
            except (WorkerUnavailable, OSError, EOFError, TimeoutError) as exc:
                self.abort(str(exc))
                result = {"cell_id": cell_id, "status": "interrupted", "reason": str(exc)}
            # 竞争窗口内即使 worker 已算出结果，只要任务已终止就不能再报告 cell 正常完成。
            if self.ended.is_set():
                result.update(status="interrupted", reason=self.reason)
            self._write(prefix + ".json", result)
            self.log("cell_returned", "cell 返回 Cell returned", cell_id=cell_id,
                     status=result["status"])
            return result

    def finish(self, status, summary):
        if status not in {"success", "failure", "stuck"}:
            raise ValueError("非法结束状态 Invalid finish status")
        # 同一 MCP session 中不允许 finish 越过尚在执行的 cell；人工急停不受这个限制。
        if not self._cell_lock.acquire(blocking=False):
            raise RuntimeError("请先等待 cell 收尾 Wait for the active cell")
        try:
            with self._state_lock:
                if self.status != "running":
                    raise RuntimeError("任务已结束 Task already ended")
                self.report = {"status": status, "summary": summary}
            self._end("completed" if status == "success" else status,
                      "agent 声明结束 Agent requested finish", stop_agent=False)
            return self.summary()
        finally:
            self._cell_lock.release()

    def _stop_agent(self):
        # codex 使用独立进程组；取消任务也停止它可能创建的子进程。
        # 这里的等待只影响模型清理，不会延迟控制进程的人工急停锁定。
        agent = self.agent
        if agent is not None and agent.poll() is None:
            try:
                os.killpg(agent.pid, signal.SIGTERM)
                try:
                    agent.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    os.killpg(agent.pid, signal.SIGKILL)
                    agent.wait(timeout=.5)
            except ProcessLookupError:
                pass

    def attach_agent(self, process):
        with self._state_lock:
            self.agent = process
            running = self.status == "running"
        if not running:
            self._stop_agent()

    def _end(self, status, reason, *, stop_agent=True):
        with self._state_lock:
            # 人工急停优先于正常完成，任何普通结束请求都不能解除急停锁定。
            if self.status != "running" and not (status == "estopped" and self.status != "estopped"):
                return
            self.status, self.reason = status, reason
            self.ended.set()
        try:
            # 急停已经由控制进程自身处理；不要再把它塞回普通请求通道等待一次。
            if status != "estopped" and self.control.process.is_alive():
                self.control.request("end_task", reason=reason)
        except (OSError, EOFError, TimeoutError) as exc:
            with self._state_lock:
                self.status = "failed"
                self.reason += f"；控制收尾未确认 Control acknowledgement missing: {exc}"
        finally:
            self.worker.stop()
            if stop_agent:
                self._stop_agent()
        self.log("session_ended", "任务结束 Session ended", result=self.summary())
        self._write("result.json", self.summary())

    def abort(self, reason, *, status="aborted"):
        self._end(status, reason)

    def _watch(self):
        # 监听独立 Event，不获取 cell 执行锁、MCP 锁或控制普通请求锁。
        # 电子急停由控制进程先行锁定；此线程仅负责后续任务/模型清理。
        while not self._shutdown.wait(.02):
            if self.control.estopped.is_set():
                self.abort("操作者触发人工急停 Operator triggered e-stop", status="estopped")
            elif self.status == "running":
                if not self.control.process.is_alive():
                    self.abort("控制进程退出 Control process exited", status="failed")
                elif not self.worker.process.is_alive():
                    self.abort("worker 退出 Python worker exited", status="failed")
                elif time.monotonic() - self._started_at >= self.task_timeout_s:
                    self.abort("任务执行超时 Task timed out")

    def close(self):
        self.abort("协调进程关闭 Session closing")
        self._shutdown.set()
        self._monitor.join(3)
        self._stop_agent()
        self.worker.stop()
        self.control.close()
        self._events_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
