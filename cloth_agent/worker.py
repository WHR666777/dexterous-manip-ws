"""独立持久 Python worker：执行 cell 与几何/规划，通过 robo 代理请求设备操作。

一个任务只创建一个 spawn 子进程；每个 cell 都使用同一份 globals。
生成代码退出或发生普通异常时保留 globals；进程被终止后不尝试恢复内存。
这里的导入/语法限制用于约束生成代码，不是完整的操作系统安全沙箱。
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
import multiprocessing as mp
from pathlib import Path
from threading import Lock
import time
import traceback

from .robo import Robo, CellPaused


class WorkerUnavailable(RuntimeError):
    """worker 已退出或被取消，调用者应结束本次任务。"""


class CellTimeout(WorkerUnavailable):
    """当前 cell 超时；不能把仍在执行的源码留给下一次调用。"""


class _Output(io.StringIO):
    """限制回传文本与缓冲区大小，避免生成代码反复 print 耗尽宿主内存。"""

    def write(self, text):
        remaining = max(0, 16000 - self.tell())
        super().write(text[:remaining])
        return len(text)

# 子进程的相关代码，以_开头，表示私有。
def _namespace(robo):
    # 本轮仅允许计算所需模块。尤其不允许源码直接导入设备 SDK、进程、网络或文件库。
    # import numpy 等模块内部仍会使用其自身运行环境，因此不能据此宣称强安全隔离。
    allowed = {"math", "time", "numpy", "scipy", "json", "statistics", "collections",
               "itertools", "functools", "dataclasses", "typing", "random", "enum"}
    original_import = builtins.__import__

    def computation_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed:
            raise ImportError(f"仅允许计算模块 Computation imports only: {name}")
        return original_import(name, globals, locals, fromlist, level)

    safe = vars(builtins).copy()
    for name in ("open", "input", "eval", "exec", "compile", "breakpoint", "help",
                 "exit", "quit", "getattr", "setattr", "delattr", "globals", "locals", "vars"):
        safe.pop(name, None)
    safe["__import__"] = computation_import
    import numpy as np

    return {"__name__": "__cell__", "__builtins__": safe,
            "np": np, "math": math, "time": time, "robo": robo}


def _execute(source, namespace, cell_id):
    started = time.monotonic()
    stdout, stderr = _Output(), _Output()
    result = {"cell_id": cell_id, "status": "completed", "value": None}
    try:
        tree = ast.parse(source, filename=f"<cell {cell_id}>")
        # 拒绝直接访问 __globals__ 等内部对象；不允许覆盖宿主注入的 builtins。
        for node in ast.walk(tree):
            if ((isinstance(node, ast.Attribute) and node.attr.startswith("_"))
                    or (isinstance(node, ast.Name) and node.id.startswith("__"))):
                raise ValueError("禁止访问内部对象 Private Python internals are unavailable")
        tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(tree, f"<cell {cell_id}>", "exec"), namespace)
            if tail is not None:
                value = eval(compile(ast.Expression(tail.value), f"<cell {cell_id}>", "eval"), namespace)
                # 最后一行用于方便模型查看结果；变量本身仍留在 worker，回传值不是反序列化协议。
                result["value"] = repr(value)[:8000] if value is not None else None
    except CellPaused as exc:
        result.update(status="paused", reason=str(exc))
    except BaseException:
        # KeyboardInterrupt / SystemExit 也只结束当前 cell。进程级强制终止由父进程处理。
        result.update(status="error", exception=traceback.format_exc()[-8000:])
    result.update(stdout=stdout.getvalue(), stderr=stderr.getvalue(),
                  elapsed_s=time.monotonic() - started)
    return result


def _worker_main(connection, workspace, robot_connection, config):
    # 此函数在新解释器执行。先关闭对外能力，再进入串行收发循环；不创建任何设备对象。
    import os

    os.chdir(workspace)
    try:
        robo = Robo(robot_connection, config)
        namespace = _namespace(robo)
        connection.send({"status": "ready", "pid": os.getpid()})
        while True:
            message = connection.recv()
            if message["operation"] == "close":
                return
            robo._begin_cell(message["cell_id"])
            # 只有宿主生成的启动 cell 000 能看到此内部辅助函数。执行结束即
            # 删除，后续模型 cell 仍只使用原有 robo API；无需放开私有属性访问。
            if message["cell_id"] == 0:
                namespace["startup_move_ptp"] = robo._startup_move_ptp
            try:
                result = _execute(message["source"], namespace, message["cell_id"])
            finally:
                namespace.pop("startup_move_ptp", None)
            # 文件路径和 crop 元数据只交给宿主附图；不把整幅 RGB-D 转成文本日志。
            result["images"] = robo._shows
            result["robot_calls"] = robo._calls
            connection.send(result)
    except (EOFError, BrokenPipeError):
        # 协调进程退出时，worker 不能自行重连或继续接受旧任务。
        return
    finally:
        connection.close()
        robot_connection.close()


class PythonWorker:
    """父进程持有的 worker 句柄；execute 串行，stop 不等待 execute 的锁。"""

    def __init__(self, workspace, robot_connection, config):
        workspace = Path(workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self._execution_lock = Lock()
        self._stop_lock = Lock()
        self._stopped = False
        self.process = context.Process(target=_worker_main,
                                       args=(child, str(workspace), robot_connection, config),
                                       name="cloth-python-worker")
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(10):
                raise WorkerUnavailable("worker 启动超时 Startup timed out")
            self.ready = self.connection.recv()
        except (EOFError, OSError, WorkerUnavailable) as exc:
            self.stop()
            raise WorkerUnavailable(f"worker 启动失败 Startup failed: {exc}") from exc

    def execute(self, source, cell_id, timeout_s):
        with self._execution_lock:
            if self._stopped or not self.process.is_alive():
                raise WorkerUnavailable("worker 已停止 Worker stopped")
            deadline = time.monotonic() + timeout_s
            try:
                self.connection.send({"operation": "execute", "source": source, "cell_id": cell_id})
                while True:
                    if self._stopped:
                        raise WorkerUnavailable("cell 被取消 Cell cancelled")
                    if self.connection.poll(.02):
                        return self.connection.recv()
                    if not self.process.is_alive():
                        raise WorkerUnavailable("worker 意外退出 Worker exited")
                    if time.monotonic() >= deadline:
                        raise CellTimeout("cell 执行超时 Cell timed out")
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise WorkerUnavailable("worker 通信已结束 Worker connection closed") from exc

    def stop(self):
        # 独立于 execute 的锁：死循环中的 cell 不能阻止宿主终止进程。
        # 强制退出不会执行 cell 的 finally，所以设备连接必须放在控制进程中。
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(.5)
            self.connection.close()
