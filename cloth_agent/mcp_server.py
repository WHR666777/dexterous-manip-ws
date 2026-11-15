"""临时 MCP 桥接：只接通 cell 与任务结束，不冻结后续机器人工具协议。

采用 MCP Python SDK 提供的 FastMCP/Streamable HTTP，不自己实现 JSON-RPC。
服务位于协调进程的线程中；生成代码始终在独立 worker 进程执行。
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import socket
from threading import Thread
import time
from typing import Literal


class CellMcpServer:
    def __init__(self, session):
        self.session = session
        self._server = None
        self._thread = None
        self._socket = None

    def start(self):
        try:
            from mcp.server.fastmcp import FastMCP
            from mcp.types import ToolAnnotations, TextContent, ImageContent
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("缺少 MCP 依赖 Install the project agent dependencies") from exc

        # 预先绑定系统分配的端口，避免“查询空闲端口后再绑定”的竞争窗口。
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("127.0.0.1", 0))
        port = self._socket.getsockname()[1]
        mcp = FastMCP("cloth_agent", host="127.0.0.1", port=port,
                      stateless_http=True, json_response=True)
        annotations = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)

        @mcp.tool(name="python", annotations=annotations)
        async def python(code: str):
            """执行持久 cell Execute Python with robo; show() attaches observation images."""
            try:
                result = await asyncio.to_thread(self.session.execute, code)
                attachments = []
                for shown in result.get("images", []):
                    try:
                        # 图片只能来自本次控制进程保存的观测目录。源码不获得任意文件读取入口。
                        path = Path(shown["rgb_path"]).resolve()
                        expected = self.session.directory / "observations" / (shown["observation_id"] + ".png")
                        if path != expected.resolve() or path.parent != expected.parent.resolve():
                            raise ValueError("图像不是本次观测 Image is outside the observation store")
                        data = await asyncio.to_thread(self._image_bytes, path, shown["crop"])
                        attachments.append(TextContent(type="text", text=json.dumps({
                            "description": "观测图像 Observation image; crop uses original pixel coordinates",
                            "observation_id": shown["observation_id"], "crop": shown["crop"]}, ensure_ascii=False)))
                        attachments.append(ImageContent(type="image", mimeType="image/png",
                                                        data=base64.b64encode(data).decode("ascii")))
                    except (OSError, ValueError, RuntimeError) as exc:
                        # 不把磁盘路径文本当作“模型已经看图”。附图失败必须明确回传。
                        result.setdefault("image_errors", []).append(str(exc))
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False)), *attachments]
            except asyncio.CancelledError:
                # MCP 客户端取消不会自动停止 to_thread 内的代码，需要显式撤销任务。
                # 这是普通任务取消；人工电子急停由控制进程的 Unix socket 独立接收。
                await asyncio.to_thread(self.session.abort, "MCP 调用取消 MCP call cancelled")
                raise

        @mcp.tool(name="finish", annotations=annotations)
        async def finish(status: Literal["success", "failure", "stuck"], summary: str) -> dict:
            """结束本次框架任务 Finish the session; no physical cloth success is measured yet."""
            return await asyncio.to_thread(self.session.finish, status, summary)

        app = mcp.streamable_http_app()
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                                    log_level="warning", access_log=False))
        self._thread = Thread(target=self._server.run, kwargs={"sockets": [self._socket]},
                              name="cloth-mcp-http", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("MCP 服务启动失败 MCP startup failed")
            time.sleep(.02)
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.session.log("mcp_started", "MCP 服务已启动 MCP server ready", url=self.url)
        return self.url

    @staticmethod
    def _image_bytes(path, crop):
        if crop is None:
            return path.read_bytes()
        import cv2

        # 裁剪只用于模型看局部；pixel_to_task 始终使用原始快照的完整图像坐标。
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError("图像无法读取 Observation image could not be read")
        u0, v0, u1, v1 = crop
        ok, encoded = cv2.imencode(".png", image[v0:v1, u0:u1])
        if not ok:
            raise RuntimeError("裁剪图像编码失败 Crop encoding failed")
        return encoded.tobytes()

    def close(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(2)
        if self._socket:
            self._socket.close()
