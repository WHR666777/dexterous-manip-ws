"""只读任务监控：增量读取现有记录，通过本机 HTTP/SSE 展示给操作者。

不导入 TaskSession、机器人 SDK 或 MCP，不创建任务、不发送机器人请求。
一个 monitor 固定读取一个任务目录；既可等待新任务生成记录，也可回看旧任务。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from threading import Event
import time
from urllib.parse import urlsplit


ASSETS = Path(__file__).with_name("monitor_assets")
STREAMS = {"session": "events.jsonl", "codex": "codex_events.jsonl",
           "control": "control-events.jsonl", "stderr": "codex_stderr.log"}


def read_json(path):
    """写入中的 JSON 可能尚未完整；保留旧画面，下次刷新继续读取。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return None


def without_image_data(value):
    """Codex 事件可能包含整幅图片的 base64；图片改走本次观测文件的只读路由。"""
    if isinstance(value, dict):
        return {key: without_image_data(item) for key, item in value.items()
                if not (value.get("type") == "image" and key == "data")}
    if isinstance(value, list):
        return [without_image_data(item) for item in value]
    return value


class TaskRecords:
    """每个浏览器连接独立持有读取游标；页面重连时从记录重建，不写辅助数据库。"""

    def __init__(self, directory):
        self.directory = directory
        self.offsets = {}
        self.versions = {}

    def inside(self, path):
        # HTTP 文件访问只允许固定任务目录中的记录，不能借观测路径读取其他文件。
        path = path.resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("记录路径超出任务目录 Record path is outside the task directory")
        return path

    def updates(self):
        for source, name in STREAMS.items():
            path = self.inside(self.directory / name)
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                old_identity, offset = self.offsets.get(source, (identity, 0))
                if identity != old_identity or stat.st_size < offset:
                    offset = 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while True:
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # 不消耗尚未写完的一行，包括大体积的 MCP 图片事件。
                            break
                        offset = stream.tell()
                        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                        if source == "stderr":
                            yield {"source": source, "data": text}
                        else:
                            try:
                                value = json.loads(text)
                            except json.JSONDecodeError:
                                yield {"source": "notice", "data": f"无法解析记录 Invalid JSON: {name}"}
                                continue
                            yield {"source": source, "data": without_image_data(value)}
                self.offsets[source] = (identity, offset)
            except FileNotFoundError:
                continue  # 支持在任务尚未启动时先打开监控页。

        # cell 源码在执行前写入，结果在执行结束后写入；两者分开推送。
        files = [self.directory / name for name in ("session.json", "result.json", "prompt.md")]
        files += sorted((self.directory / "cells").glob("*.py"))
        files += sorted((self.directory / "cells").glob("*.json"))
        for candidate in files:
            path = self.inside(candidate)
            try:
                stat = path.stat()
                version = (stat.st_mtime_ns, stat.st_size)
                if self.versions.get(str(path)) == version:
                    continue
                if path.suffix in {".py", ".md"}:
                    value = path.read_text(encoding="utf-8")
                else:
                    value = read_json(path)
                    if value is None:
                        continue
                if path.parent.name == "cells" and path.stem.isdigit():
                    # 只返回 cell 文件中实际记录的值，不解析 stdout 成伪造机器人状态。
                    payload = {"cell_id": int(path.stem), "kind": path.suffix[1:], "value": value}
                    if path.suffix == ".json" and isinstance(value, dict):
                        metadata = {}
                        for shown in value.get("images", []):
                            identifier = shown.get("observation_id", "")
                            if re.fullmatch(r"[0-9a-f]{32}", identifier):
                                meta_path = self.inside(self.directory / "observations" / f"{identifier}.json")
                                metadata[identifier] = read_json(meta_path)
                        payload["observations"] = metadata
                    yield {"source": "cell", "data": payload}
                elif path.name == "prompt.md":
                    # prompt 含固定操作说明和配置；顶部只显示本次任务描述。
                    yield {"source": "task", "data": value.rpartition("本次任务 Task：\n")[2]}
                else:
                    yield {"source": "metadata" if path.name == "session.json" else path.stem,
                           "data": value}
                self.versions[str(path)] = version
            except FileNotFoundError:
                continue


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, directory, port):
        self.directory = Path(directory).expanduser().resolve()
        self.stopping = Event()
        super().__init__(("127.0.0.1", port), MonitorHandler)


class MonitorHandler(BaseHTTPRequestHandler):
    """仅提供页面、事件流和观测 PNG；没有执行代码、停止机器人或修改文件的路由。"""

    def log_message(self, format, *args):
        pass  # 页面频繁刷新不刷终端；读取错误会通过 SSE 明确展示。

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/events":
            self.stream_events()
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/monitor.js": ("monitor.js", "text/javascript; charset=utf-8"),
                  "/monitor.css": ("monitor.css", "text/css; charset=utf-8")}
        if route in assets:
            name, content_type = assets[route]
            path = ASSETS / name
        elif re.fullmatch(r"/observations/[0-9a-f]{32}\.png", route):
            path = (self.server.directory / route.lstrip("/")).resolve()
            if not path.is_relative_to(self.server.directory):
                self.send_error(403)
                return
            content_type = "image/png"
        else:
            self.send_error(404)
            return
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(value):
            self.wfile.write(("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()

        try:
            # EventSource 自动重连时明确重建视图，避免把历史再次追加成重复卡片。
            send({"source": "reset", "data": {"directory": str(self.server.directory),
                                               "monotonic": time.monotonic()}})
            records = TaskRecords(self.server.directory)
            while not self.server.stopping.is_set():
                try:
                    for value in records.updates():
                        send(value)
                except (OSError, ValueError) as exc:
                    send({"source": "notice", "data": f"读取记录失败 Record read failed: {exc}"})
                # 注释作为心跳，不伪装成 agent 或机器人活动。
                self.wfile.write(b": monitor heartbeat\n\n")
                self.wfile.flush()
                self.server.stopping.wait(.4)
        except (BrokenPipeError, ConnectionResetError):
            return


def serve_monitor(directory, port=8765):
    """监控独立于 run/serve 生命周期；Ctrl+C 只关闭监控服务。"""
    with MonitorServer(directory, port) as server:
        print(f"只读监控已启动 Read-only monitor: http://127.0.0.1:{server.server_port}/\n"
              f"任务目录 Task directory: {server.directory}\n"
              "Ctrl+C 仅关闭监控，不停止机器人 Monitor exit does not stop the robot", flush=True)
        try:
            server.serve_forever(poll_interval=.4)
        except KeyboardInterrupt:
            pass
        finally:
            server.stopping.set()
    return 0
