"""Cloth agent 命令行；仅显式启动 run/serve 才创建进程与 MCP 服务。"""

from __future__ import annotations

import argparse
import json

from .control import request_estop
from .session import TaskSession


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cloth agent 进程框架 Process framework; no robo")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="启动一次 Codex exec task")
    run.add_argument("--task", required=True, help="纯计算任务；当前不支持机器人操作")
    run.add_argument("--codex", default="codex", help="本机 Codex executable")
    run.add_argument("--model", default=None, help="省略时沿用 Codex 模型设置")
    serve = commands.add_parser("serve", help="仅启动 MCP 与两个子进程，等待外部 MCP client")
    for command in (run, serve):
        command.add_argument("--output-dir", help="新任务记录目录，不覆盖已有 events.jsonl")
        command.add_argument("--cell-timeout", type=float, default=60.)
        command.add_argument("--task-timeout", type=float, default=900.)
        command.add_argument("--max-cells", type=int, default=40)
    estop = commands.add_parser("estop", help="直接向控制进程请求人工急停锁定，不经过 MCP")
    estop.add_argument("session_file", help="本次任务的 session.json")
    estop.add_argument("--reason", default="操作者请求急停 Operator requested e-stop")
    args = parser.parse_args(argv)

    if args.command == "estop":
        # 此分支不创建 TaskSession，不启动 Codex，也不需要 MCP 依赖。
        # 当前回执仅证明控制状态锁定，hardware_connected/stop_confirmed 均为 False。
        print(json.dumps(request_estop(args.session_file, args.reason), ensure_ascii=False, indent=2))
        return 0

    from .mcp_server import CellMcpServer

    with TaskSession(args.output_dir, cell_timeout_s=args.cell_timeout,
                     task_timeout_s=args.task_timeout, max_cells=args.max_cells) as session:
        server = CellMcpServer(session)
        try:
            url = server.start()
            print(json.dumps({"description": "任务服务已启动 Task service ready; no hardware",
                              "mcp_url": url, "session_file": str(session.directory / "session.json")},
                             ensure_ascii=False), flush=True)
            if args.command == "run":
                from .codex_exec import run_codex

                result = run_codex(session, url, args.task, executable=args.codex, model=args.model)
            else:
                session.ended.wait()
                result = session.summary()
        except KeyboardInterrupt:
            # Ctrl+C 是普通任务取消；真正的人工电子急停必须使用独立入口。
            session.abort("终端取消任务 Terminal cancelled task")
            result = session.summary()
        finally:
            # 先结束生成代码，再等待 HTTP 服务退出，避免服务关闭被活动 cell 卡住。
            session.abort("MCP 服务关闭 MCP service closing")
            server.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    # spawn 子进程重新导入模块时不能再次进入 CLI，否则会递归创建进程。
    raise SystemExit(main())
