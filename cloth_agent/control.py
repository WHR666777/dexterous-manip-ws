"""独立控制进程：普通服务与人工位置保持分别使用通道和处理路径。

普通 Pipe 由一个服务线程处理；普通轨迹由独立线程按固定周期执行。
主线程接人工命令并独立周期下发固定保持目标。
生成源码不在这里执行。worker 只能发有限 robo 数据请求，不能发送任意函数。
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty, SimpleQueue
import selectors
import signal
import socket
import tempfile
from threading import Event, Lock, Thread
import time
import os

from .hardware import RobotController


def _control_main(owner_connection, worker_connection, manual_hold, released, failed,
                  config, directory, allow_hardware):
    # 终端 Ctrl+C 不应连保持进程一起杀掉。正常退出由管理通道执行；人工保持
    # 必须在机械臂已获外部支撑后，由独立 release-hold 命令明确结束。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with tempfile.TemporaryDirectory(prefix="cloth-control-") as temporary:
        endpoint = str(Path(temporary) / "manual.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as emergency:
            emergency.bind(endpoint)
            emergency.settimeout(.05)
            try:
                robot = RobotController(config, directory, manual_hold, failed, allow_hardware)
            except Exception as exc:
                # 设备初始化可能因反馈超时或标定预检失败而退出。把具体原因通过
                # 启动握手交还父进程，避免只暴露 Pipe EOFError、丢失真正的故障信息。
                # RobotController 已负责清理初始化阶段创建的设备连接。
                failed.set()
                try:
                    owner_connection.send({"ok": False,
                                           "reason": f"{type(exc).__name__}: {exc}"})
                finally:
                    owner_connection.close()
                    worker_connection.close()
                return
            finished = Event()
            notifications = SimpleQueue()
            control_samples = SimpleQueue()
            motion_shutdown = Event()
            robot.on_control_sample = control_samples.put

            def run_motion():
                # 轨迹时钟只由 monotonic 定时器驱动，不能由 Pipe 请求唤醒次数
                # 决定。此线程不查询 L20、不压缩图像、不验证整条轨迹、不写文件。
                previous = next_tick = time.monotonic()
                while not motion_shutdown.is_set() and not released.is_set():
                    now = time.monotonic()
                    try:
                        robot.tick(now - previous)
                    except Exception as exc:
                        robot.fail(f"设备执行故障 Device execution failed: {exc}")
                    previous = now
                    next_tick += robot.period
                    now = time.monotonic()
                    if next_tick < now:
                        # 超时跳过已错过的时隙，不集中补发多个关节目标。
                        next_tick += (int((now - next_tick) / robot.period) + 1) * robot.period
                    motion_shutdown.wait(max(0., next_tick - time.monotonic()))

            def record(description, **values):
                # 仅普通线程落盘，人工停止保持接收路径只入内存队列，不等待日志写入。
                with (Path(directory) / "control-events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"description": description,
                                             "timestamp_monotonic": time.monotonic(), **values},
                                            ensure_ascii=False) + "\n")

            def serve_ordinary():
                try:
                    with selectors.DefaultSelector() as selector, (
                        Path(directory) / "control-samples.jsonl"
                    ).open("a", encoding="utf-8", buffering=1) as samples_file:
                        selector.register(owner_connection.fileno(), selectors.EVENT_READ, True)
                        selector.register(worker_connection.fileno(), selectors.EVENT_READ, False)
                        while not released.is_set():
                            # 轨迹线程只提交内存快照；磁盘慢也不能拖长 JS 发送间隔。
                            while True:
                                try:
                                    sample = control_samples.get_nowait()
                                except Empty:
                                    break
                                samples_file.write(json.dumps({
                                    "description": "关节控制采样 Joint control sample", **sample
                                }, ensure_ascii=False) + "\n")
                            while True:
                                try:
                                    record("人工停止保持反馈 Manual hold report", report=notifications.get_nowait())
                                except Empty:
                                    break
                            ready = selector.select(.05)
                            # 协调器的 cell_end/end_task 优先于旧 worker 迟到的普通请求。
                            ready.sort(key=lambda item: not item[0].data)
                            for key, _ in ready:
                                owner = key.data
                                connection = owner_connection if owner else worker_connection
                                try:
                                    request = connection.recv()
                                except EOFError:
                                    if manual_hold.is_set():
                                        # 人工停止会结束 worker；这个预期 EOF 不能带走
                                        # 位置保持进程或设备连接。保留其余仍有效的通道。
                                        selector.unregister(connection.fileno())
                                        continue
                                    raise
                                if owner and request["operation"] == "close":
                                    return
                                try:
                                    value = robot.dispatch(request, owner=owner)
                                    reply = {"ok": True, "result": value}
                                except Exception as exc:
                                    # 输入/规划起点等错误只拒绝当前请求；SDK RuntimeError 则撤销
                                    # 后续动作。未连接模式的明确拒绝不伪造设备故障。
                                    if isinstance(exc, RuntimeError) and robot.connected and not manual_hold.is_set():
                                        robot.fail(str(exc))
                                    reply = {"ok": False, "reason": str(exc)}
                                record("处理控制请求 Control request handled", operation=request["operation"],
                                       cell_id=request.get("cell_id"), ok=reply["ok"],
                                       reason=reply.get("reason", ""))
                                connection.send(reply)
                except (EOFError, BrokenPipeError, OSError) as exc:
                    failed.set()
                    record("控制通信已结束 Control connection closed", reason=str(exc))
                finally:
                    # 关闭普通服务不能关闭人工保持路径；设备由主线程在明确释放后清理。
                    finished.set()

            # 启动信息只包含固定 PID/入口和当前设备连接状态，不复制 SDK 对象到其他进程。
            owner_connection.send({"ok": True, "result": {
                "pid": os.getpid(), "manual_socket": endpoint,
                "manual_protocol": "position-hold-v1", **robot.snapshot()}})
            ordinary = Thread(target=serve_ordinary, name="cloth-control-io", daemon=True)
            motion_thread = Thread(target=run_motion, name="cloth-motion", daemon=True)
            motion_thread.start()
            ordinary.start()
            try:
                next_hold_at = time.monotonic()
                previous_report = None
                while not released.is_set() and (not finished.is_set() or manual_hold.is_set()):
                    now = time.monotonic()
                    if manual_hold.is_set() and now >= next_hold_at:
                        robot.manual_hold_tick()
                        next_hold_at = time.monotonic() + robot.period
                        state = robot.snapshot()
                        report = state["manual_hold_report"] or {}
                        marker = (report.get("right"), report.get("error"), state["hold_confirmed"])
                        if marker != previous_report:
                            notifications.put(state)
                            previous_report = marker
                    # 等待下一个独立保持周期或人工请求；不等待普通 I/O/cell 的锁。
                    emergency.settimeout(max(.001, min(.05, next_hold_at - time.monotonic()))
                                         if manual_hold.is_set() else .05)
                    try:
                        data, sender = emergency.recvfrom(4096)
                    except socket.timeout:
                        continue
                    try:
                        request = json.loads(data)
                        if request["protocol"] != "position-hold-v1":
                            raise ValueError("人工接口版本不匹配 Manual protocol mismatch")
                        operation = request["operation"]
                        if operation == "hold":
                            state = robot.manual_hold(str(request["reason"]))
                            next_hold_at = time.monotonic() + robot.period
                            notifications.put(state)
                        elif operation == "hold_status":
                            state = robot.snapshot()
                        elif operation == "release_hold":
                            if not manual_hold.is_set() or request.get("supported") is not True:
                                raise ValueError("需先保持并人工确认外部支撑 Confirm external support first")
                            state = robot.snapshot()
                            released.set()  # 不清动作锁、不恢复轨迹；只允许关闭整个控制进程。
                        else:
                            raise ValueError("未知人工请求 Unknown manual operation")
                        reply = {"status": "released" if released.is_set() else
                                 ("manual_hold" if manual_hold.is_set() else "running"),
                                 **state, "description": "人工位置保持 Manual position hold"}
                    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                        reply = {"status": "rejected", "reason": str(exc)}
                    if sender:
                        try:
                            emergency.sendto(json.dumps(reply, ensure_ascii=False).encode(), sender)
                        except OSError:
                            pass  # 操作者端退出不撤销人工保持。
            finally:
                motion_shutdown.set()
                motion_thread.join(2)
                ordinary.join(2)
                try:
                    if robot.motion and robot.motion["status"] == "running" and not manual_hold.is_set():
                        robot.pause("控制服务关闭 Control service closing")
                    robot.close()
                finally:
                    owner_connection.close()
                    worker_connection.close()


def request_manual(session_file, operation, reason="操作者请求位置保持 Operator requested hold",
                   *, supported=False, timeout_s=2.):
    """人工入口直接发往新控制进程；拒绝旧会话，防止误触阻尼急停。"""
    metadata = json.loads(Path(session_file).read_text())
    if metadata.get("manual_protocol") != "position-hold-v1":
        raise ValueError("旧会话不支持位置保持 Restart the server with the updated code")
    endpoint = metadata["manual_socket"]
    payload = json.dumps({"protocol": "position-hold-v1", "operation": operation,
                          "reason": reason, "supported": supported}, ensure_ascii=False).encode()
    if len(payload) > 4096:
        raise ValueError("人工请求过长 Manual request too long")
    with tempfile.TemporaryDirectory(prefix="cloth-stop-") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.bind(str(Path(directory) / "reply.sock"))
            client.settimeout(timeout_s)
            client.sendto(payload, endpoint)
            return json.loads(client.recv(8192))


class ControlProcess:
    """协调器拥有管理 Pipe；worker 拿到另一个普通 Pipe，不获得管理操作权限。"""

    def __init__(self, config=None, directory=None, *, allow_hardware=False):
        context = mp.get_context("spawn")
        self.manual_hold, self.released, self.failed = context.Event(), context.Event(), context.Event()
        self.connection, owner_child = context.Pipe()
        self.worker_connection, worker_child = context.Pipe()
        self._request_lock = Lock()
        self.process = context.Process(target=_control_main,
                                       args=(owner_child, worker_child, self.manual_hold, self.released, self.failed,
                                             config, str(directory), allow_hardware),
                                       name="cloth-robot-control")
        self.process.start()
        owner_child.close()
        worker_child.close()
        try:
            if not self.connection.poll(30):
                raise TimeoutError("控制进程启动超时 Control startup timed out")
            try:
                startup = self.connection.recv()
            except EOFError as exc:
                raise RuntimeError("控制进程在就绪前退出 Control exited before startup acknowledgement") from exc
            if not startup["ok"]:
                raise RuntimeError("控制进程启动失败 Control startup failed: " + startup["reason"])
            self.ready = startup["result"]
            self.last_state = self.ready
        except BaseException:
            self.close()
            raise

    def request(self, operation, **values):
        with self._request_lock:
            self.connection.send({"operation": operation, **values})
            if not self.connection.poll(5):
                raise TimeoutError("控制进程未确认 Control acknowledgement missing")
            reply = self.connection.recv()
            if not reply["ok"]:
                raise RuntimeError(reply["reason"])
            self.last_state = reply["result"]
            return self.last_state

    def _wait_manual_release(self):
        # 人工保持期间不能沿用旧的 terminate/kill 收尾，否则固定目标流会突然消失。
        # 正常无保持会话仍有界退出；人工模式由明确 release-hold 才结束。
        announced = False
        while self.process.is_alive() and self.manual_hold.is_set() and not self.released.is_set():
            if not announced:
                print("人工保持仍在运行 Manual hold remains active. "
                      "确认机械臂已有外部支撑后，另一个终端执行 release-hold。", flush=True)
                announced = True
            try:
                self.process.join(.2)
            except KeyboardInterrupt:
                print("Ctrl+C 不解除位置保持 Use release-hold after external support is confirmed", flush=True)

    def close(self):
        self._wait_manual_release()
        try:
            if self.process.is_alive():
                self.connection.send({"operation": "close"})
                self.process.join(2)
                # 管理关闭与人工停止可能并发；进入强制终止前再次确认，避免在
                # 上面 join 的窗口内刚锁定保持，就被父进程的清理代码杀死。
                self._wait_manual_release()
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(.5)
        except (EOFError, OSError):
            self._wait_manual_release()
            if self.process.is_alive():
                self.process.kill()
                self.process.join(.5)
        finally:
            self.connection.close()
            self.worker_connection.close()
