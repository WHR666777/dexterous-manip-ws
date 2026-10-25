"""独立控制进程与人工急停通道；本轮只有控制状态，不连接真实设备。

普通请求使用 multiprocessing Pipe；人工急停使用单独的 Unix datagram socket。
急停路径不调用 MCP、不经过 worker、不获取普通请求锁，也不依赖轨迹周期。
后续设备适配只应加入此进程；当前 hardware_connected/stop_confirmed 始终为 False。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import selectors
import socket
import tempfile
from threading import Lock
import time


def _control_main(connection, estopped):
    # 目录由控制进程创建，0700 权限由 TemporaryDirectory 提供；只允许本机同用户入口。
    # socket 路径放 /tmp，避免实验记录目录过长而超出 Unix socket 路径限制。
    with tempfile.TemporaryDirectory(prefix="cloth-control-") as directory:
        endpoint = str(Path(directory) / "estop.sock")
        state = {"status": "running", "pid": os.getpid(), "estop_latched": False,
                 "hardware_connected": False, "stop_confirmed": False,
                 "description": "控制进程已启动 Control ready; hardware not connected"}
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as emergency:
            emergency.bind(endpoint)
            emergency.setblocking(False)
            with selectors.DefaultSelector() as selector:
                selector.register(emergency, selectors.EVENT_READ, "estop")
                selector.register(connection.fileno(), selectors.EVENT_READ, "ordinary")
                connection.send({**state, "estop_socket": endpoint})
                try:
                    while True:
                        ready = selector.select()
                        # 两个通道同时可读时先处理急停。此处不执行规划、图像或磁盘 I/O。
                        ready.sort(key=lambda item: item[0].data != "estop")
                        for key, _ in ready:
                            if key.data == "estop":
                                data, sender = emergency.recvfrom(4096)
                                try:
                                    reason = str(json.loads(data)["reason"])
                                except (ValueError, KeyError, TypeError, UnicodeError):
                                    reason = "人工急停请求 Manual e-stop request"
                                state.update(status="estopped", estop_latched=True, reason=reason,
                                             requested_at=time.monotonic(),
                                             description="人工急停已锁定 E-stop latched; hardware not connected")
                                # 先锁定再通知父进程。未来电子急停发送应放在此处，不能等待
                                # 父进程清理 worker/Codex，也不能在持有普通动作锁时发送。
                                # 本轮没有设备对象，明确报告 not_sent，不能伪造已发送或已停机。
                                state["nero_estop"] = {"right": "not_sent: hardware not connected"}
                                estopped.set()
                                if sender:
                                    try:
                                        emergency.sendto(json.dumps(state, ensure_ascii=False).encode(), sender)
                                    except OSError:
                                        # 调用方退出不撤销急停；锁定状态仍然生效。
                                        pass
                                continue

                            request = connection.recv()
                            operation = request["operation"]
                            if operation == "close":
                                return
                            if operation == "end_task" and not state["estop_latched"]:
                                state.update(status="stopped", reason=request["reason"],
                                             description="任务准入已关闭 Task admission closed")
                            # cell_end 目前只确认框架边界，不宣称执行过暂停保持。
                            # 后续加入运动后，应在这里检查 cell 所属动作并完成实际收尾。
                            if operation not in {"status", "cell_end", "end_task"}:
                                connection.send({"status": "error", "reason":
                                                 "当前没有机器人操作 Robot operations not implemented"})
                            else:
                                connection.send(state.copy())
                except (EOFError, BrokenPipeError):
                    # 宿主退出后不继续接受普通任务。本轮无设备；未来需在设备层补齐暂停收尾。
                    return
                finally:
                    connection.close()


def request_estop(session_file, reason="操作者请求急停 Operator requested e-stop", timeout_s=2.):
    """人工命令直接访问控制进程；不导入/调用 MCP 或协调进程。

    返回的是控制进程的锁定回执。当前未接设备，回执不会声称已发送 Nero 急停。
    超时直接报错，操作者不能把超时解释为设备已经停止。
    """
    endpoint = json.loads(Path(session_file).read_text())["estop_socket"]
    payload = json.dumps({"reason": reason}, ensure_ascii=False).encode()
    if len(payload) > 4096:
        raise ValueError("急停原因过长 E-stop reason too long")
    with tempfile.TemporaryDirectory(prefix="cloth-stop-") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.bind(str(Path(directory) / "reply.sock"))
            client.settimeout(timeout_s)
            client.sendto(payload, endpoint)
            return json.loads(client.recv(8192))


class ControlProcess:
    """协调进程的控制句柄；人工通道在子进程内，完全不依赖 _request_lock。"""

    def __init__(self):
        context = mp.get_context("spawn")
        self.estopped = context.Event()
        self.connection, child = context.Pipe()
        self._request_lock = Lock()
        self.process = context.Process(target=_control_main, args=(child, self.estopped),
                                       name="cloth-robot-control")
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(10):
                raise TimeoutError("控制进程启动超时 Control startup timed out")
            self.ready = self.connection.recv()
        except BaseException:
            self.close()
            raise

    def request(self, operation, **values):
        with self._request_lock:
            self.connection.send({"operation": operation, **values})
            if not self.connection.poll(2):
                raise TimeoutError("控制进程未确认 Control acknowledgement missing")
            return self.connection.recv()

    def close(self):
        try:
            if self.process.is_alive():
                self.connection.send({"operation": "close"})
                self.process.join(.5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(.5)
        except (EOFError, OSError):
            if self.process.is_alive():
                self.process.kill()
                self.process.join(.5)
        finally:
            self.connection.close()
