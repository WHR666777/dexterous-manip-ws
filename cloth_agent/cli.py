"""Cloth agent 命令行；仅显式启动 run/serve 才创建进程与 MCP 服务。"""

from __future__ import annotations

import argparse
import json

from .paths import DEFAULT_CONFIG, configured_output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cloth agent：Codex / MCP / persistent Python / robo")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="启动一次 Codex exec task")
    run.add_argument("--task", required=True, help="交给 agent 的本次任务 Task description")
    run.add_argument("--codex", default="codex", help="本机 Codex executable")
    run.add_argument("--model", default=None, help="省略时沿用 Codex 模型设置")
    serve = commands.add_parser("serve", help="仅启动 MCP 与两个子进程，等待外部 MCP client")
    monitor = commands.add_parser("monitor", help="只读浏览器监控，不连接机器人、不启动 Codex")
    monitor.add_argument("directory", nargs="?", help="覆盖配置中的本轮目录；也可指定历史任务")
    monitor.add_argument("--port", type=int, default=8765, help="本地 HTTP 端口；0 为系统分配")
    command_parsers = [run, serve, monitor]
    for command in (run, serve):
        command.add_argument("--execute", action="store_true", help="连接右臂、右手及配置相机并允许执行动作")
        command.add_argument("--output-dir", help="覆盖 session.output_dir；不覆盖已有记录")
        command.add_argument("--cell-timeout", type=float, default=60.)
        command.add_argument("--task-timeout", type=float, default=900.)
        command.add_argument("--max-cells", type=int, default=40)
    for name, help_text in (("hold", "人工停止并保持固定关节位置，不经过 MCP"),
                            ("hold-status", "查询人工保持状态，不等待普通请求"),
                            ("release-hold", "已获得外部支撑后，结束保持并关闭控制进程"),
                            ("estop", "已禁用：旧阻尼急停不能用于空中保持")):
        manual = commands.add_parser(name, help=help_text)
        manual.add_argument("session_file", nargs="?", help="覆盖配置目录下的 session.json")
        command_parsers.append(manual)
        if name == "hold":
            manual.add_argument("--reason", default="操作者请求位置保持 Operator requested hold")
    for command in command_parsers:
        command.add_argument("--config", help="配置文件；默认 configs/cloth_agent.yaml，本轮目录读取 session.output_dir")
    args = parser.parse_args(argv)

    if args.command == "monitor":
        # 只读监控先行分支，不导入 TaskSession/控制模块，不创建硬件子进程。
        from .monitor import serve_monitor

        if not 0 <= args.port <= 65535:
            parser.error("--port 必须在 0 到 65535 之间 Port must be in 0..65535")
        directory = args.directory if args.directory is not None else configured_output_dir(args.config or DEFAULT_CONFIG)
        return serve_monitor(directory, args.port)
    if args.command == "estop":
        # 明确禁用旧命令，不悄悄将 emergency stop 改成另一种物理语义。
        # 只提示、不向旧/新控制进程发包，防止继续触发会下落的阻尼急停。
        print("estop 已禁用：阻尼急停不能用于悬停。请重启新版服务并使用 hold。 "
              "Damped e-stop disabled; restart and use position hold.")
        return 1
    if args.command in {"hold", "hold-status", "release-hold"}:
        from .control import request_manual

        # 显式路径优先；省略时只读取 YAML 中的本轮目录，不启动 TaskSession 或
        # 加载相机/标定。人工命令仍直接进入独立 socket，不经过 MCP/worker。
        session_file = (args.session_file if args.session_file is not None else
                        configured_output_dir(args.config or DEFAULT_CONFIG) / "session.json")
        supported = False
        if args.command == "release-hold":
            # 这是结束位置保持监督与设备连接，之后是否继续保持取决于硬件，不能假定。
            # 让现场操作者在机械臂已安全落座/外部支撑后明确确认；不自动 reset/失能。
            supported = input("结束保持可能失去支撑。确认机械臂已安全落座或获得外部支撑，"
                              "输入 SUPPORTED 确认 Release hold: ").strip() == "SUPPORTED"
            if not supported:
                print("取消释放 Hold remains active")
                return 1
        operation = {"hold": "hold", "hold-status": "hold_status",
                     "release-hold": "release_hold"}[args.command]
        result = request_manual(session_file, operation, supported=supported,
                                reason=getattr(args, "reason", "操作者请求 Operator request"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "rejected" else 0

    from .mcp_server import CellMcpServer
    from .session import TaskSession

    # --execute 直接决定是否允许连接真实设备，不再读取交互确认词。
    # 未传此参数仍只启动无硬件会话；机器人使能仍由操作者通过已有流程处理。
    output_dir = args.output_dir if args.output_dir is not None else configured_output_dir(args.config or DEFAULT_CONFIG)
    with TaskSession(output_dir, cell_timeout_s=args.cell_timeout,
                     task_timeout_s=args.task_timeout, max_cells=args.max_cells,
                     config_path=args.config, allow_hardware=args.execute) as session:
        server = CellMcpServer(session)
        try:
            # TaskSession 在启动 MCP/Codex 前完成固定准备 cell；失败或人工 hold 时
            # 直接进入本函数的 finally 和上下文清理，不把未准备的设备交给 agent。
            if session.status != "running":
                print(json.dumps(session.summary(), ensure_ascii=False, indent=2))
                return 1
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
