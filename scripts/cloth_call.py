"""人工发送一个已审阅的 Python cell；用于 robo/MCP 实机联调，不启动 Codex。

此脚本只是普通 MCP client，没有人工停止权限或设备连接。人工停止使用另一个
终端中的 ``python -m cloth_agent hold .../session.json``，不等待本脚本返回。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def call(url, code):
    # 给人工触发急停留出时间。实际 cell/任务预算仍由 serve 的参数控制；
    # HTTP 等待时间不会延长机器人运动权限，也不参与人工急停通道。
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("提交 cell Cell submitted; use a separate terminal for manual hold", flush=True)
            result = await session.call_tool("python", {"code": code},
                                             read_timeout_seconds=timedelta(seconds=650))
            for block in result.content:
                if block.type == "text":
                    print(block.text)
                else:
                    # 本脚本只显示文字；不把图像 base64 打到终端。需要看图时使用
                    # 支持 MCP ImageContent 的客户端，本次人工急停步骤不依赖图像。
                    print(f"非文字反馈 Non-text content: {block.type}")
            return 1 if result.isError else 0


def main():
    parser = argparse.ArgumentParser(description="人工发送 cell Manual MCP cell client")
    parser.add_argument("url", help="serve 终端打印的 mcp_url")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--code", help="直接提供已审阅的 Python 源码")
    source.add_argument("--code-file", type=Path, help="从本地文件读取已审阅的 cell")
    args = parser.parse_args()
    code = args.code if args.code is not None else args.code_file.read_text(encoding="utf-8")
    return asyncio.run(call(args.url, code))


if __name__ == "__main__":
    raise SystemExit(main())
