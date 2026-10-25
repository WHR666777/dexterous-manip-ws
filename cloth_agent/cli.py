"""Cloth agent 命令行；仅显式启动 run/serve 才创建进程与 MCP 服务。"""

from __future__ import annotations

import argparse
import json

from .control import request_manual
from .session import TaskSession


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cloth agent：Codex / MCP / persistent Python / robo")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="启动一次 Codex exec task")
    run.add_argument("--task", required=True, help="交给 agent 的本次任务 Task description")
    run.add_argument("--codex", default="codex", help="本机 Codex executable")
    run.add_argument("--model", default=None, help="省略时沿用 Codex 模型设置")
    serve = commands.add_parser("serve", help="仅启动 MCP 与两个子进程，等待外部 MCP client")
    for command in (run, serve):
        command.add_argument("--config", help="机器人标定/运动配置；--execute 时默认 configs/cloth_agent.yaml")
        command.add_argument("--execute", action="store_true", help="连接右臂、右手及配置相机并允许执行动作")
        command.add_argument("--output-dir", help="新任务记录目录，不覆盖已有 events.jsonl")
        command.add_argument("--cell-timeout", type=float, default=60.)
        command.add_argument("--task-timeout", type=float, default=900.)
        command.add_argument("--max-cells", type=int, default=40)
    for name, help_text in (("hold", "人工停止并保持固定关节位置，不经过 MCP"),
                            ("hold-status", "查询人工保持状态，不等待普通请求"),
                            ("release-hold", "已获得外部支撑后，结束保持并关闭控制进程"),
                            ("estop", "已禁用：旧阻尼急停不能用于空中保持")):
        manual = commands.add_parser(name, help=help_text)
        manual.add_argument("session_file", help="本次任务的 session.json")
        if name == "hold":
            manual.add_argument("--reason", default="操作者请求位置保持 Operator requested hold")
    args = parser.parse_args(argv)

    if args.command == "estop":
        # 明确禁用旧命令，不悄悄将 emergency stop 改成另一种物理语义。
        # 只提示、不向旧/新控制进程发包，防止继续触发会下落的阻尼急停。
        print("estop 已禁用：阻尼急停不能用于悬停。请重启新版服务并使用 hold。 "
              "Damped e-stop disabled; restart and use position hold.")
        return 1
    if args.command in {"hold", "hold-status", "release-hold"}:
        supported = False
        if args.command == "release-hold":
            # 这是停止持续位置下发，之后是否继续保持取决于硬件状态，不能假定。
            # 让现场操作者在机械臂已安全落座/外部支撑后明确确认；不自动 reset/失能。
            supported = input("结束保持可能失去支撑。确认机械臂已安全落座或获得外部支撑，"
                              "输入 SUPPORTED 确认 Release hold: ").strip() == "SUPPORTED"
            if not supported:
                print("取消释放 Hold remains active")
                return 1
        operation = {"hold": "hold", "hold-status": "hold_status",
                     "release-hold": "release_hold"}[args.command]
        result = request_manual(args.session_file, operation, supported=supported,
                                reason=getattr(args, "reason", "操作者请求 Operator request"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "rejected" else 0

    from .mcp_server import CellMcpServer

    # --execute 直接决定是否允许连接真实设备，不再读取交互确认词。
    # 未传此参数仍只启动无硬件会话；机器人使能仍由操作者通过已有流程处理。
    with TaskSession(args.output_dir, cell_timeout_s=args.cell_timeout,
                     task_timeout_s=args.task_timeout, max_cells=args.max_cells,
                     config_path=args.config, allow_hardware=args.execute) as session:
        server = CellMcpServer(session)
        try:
            url = server.start()
            print(json.dumps({"description": "任务服务已启动 Task service ready",
                              "hardware_connected": session.control.ready["hardware_connected"],
                              "mcp_url": url, "session_file": str(session.directory / "session.json")},
                             ensure_ascii=False), flush=True)
            if args.command == "run":
                from .codex_exec import run_codex

                result = run_codex(session, url, args.task, executable=args.codex, model=args.model)
            else:
                session.ended.wait()
                result = session.summary()
        except KeyboardInterrupt:
            # Ctrl+C 是普通任务取消；人工固定位置保持使用独立 hold 入口。
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
