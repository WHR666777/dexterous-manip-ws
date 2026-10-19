"""Standalone entry point. No device connection until Runtime is constructed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, MOCK_CONFIG, load_config, validate_hardware_config
from .runtime import Runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description="折布 agent：Code-as-policy 与普通 Python 程序工具")
    parser.add_argument("--config", type=Path, help="任务默认 cloth_agent.yaml；demo 使用 cloth_agent_mock.yaml")
    parser.add_argument("--output-dir", type=Path, help="本次任务记录 artifacts 的完整目录")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="两个 mock 阶段与一次注入异常恢复")
    demo.add_argument("--with-codex", action="store_true", help="通过真实 Codex thread 生成程序并监督执行")
    run = subcommands.add_parser("run", help="让 Codex 生成并监督 task 程序")
    run.add_argument("--task", default="将平铺的矩形抹布折叠两次。")
    run.add_argument("--folds", type=int, default=None)
    run.add_argument("--execute", action="store_true")
    script = subcommands.add_parser("script", help="运行已有普通 Python subtask 程序")
    script.add_argument("source", type=Path)
    script.add_argument("--params", type=Path)
    script.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config or (MOCK_CONFIG if args.command == "demo" else DEFAULT_CONFIG))
    if args.command == "demo" and config["backend"] != "mock":
        parser.error("demo 要求 backend: mock")
    allow_hardware = False
    if config["backend"] == "hardware":
        validate_hardware_config(config)  # Check blanks before opening any device.
        if not args.execute:
            parser.error("实机 hardware 执行需要 --execute 和 EXECUTE 确认。")
        if input("确认工作区、机械臂使能和急停 e-stop 可用，输入 EXECUTE：").strip() != "EXECUTE":
            parser.error("未确认实机执行 Hardware execution was not confirmed.")
        allow_hardware = True
    with Runtime(config, args.output_dir, allow_hardware=allow_hardware) as runtime:
        if args.command == "demo":
            from .demo import SMOKE_TASK, run_demo
            if args.with_codex:
                from .codex import CodexAppServer
                result = CodexAppServer(runtime).run(SMOKE_TASK)
            else:
                result = run_demo(runtime)
        elif args.command == "run":
            from .codex import CodexAppServer
            folds = args.folds if args.folds is not None else config["folds"]
            if folds < 1:
                parser.error("折叠次数 folds 必须为正数。")
            result = CodexAppServer(runtime).run(f"{args.task}\n折叠次数 folds：{folds}。")
        else:
            params = json.loads(args.params.read_text()) if args.params else {}
            runtime.submit_subtask(args.source.read_text(), params, args.source.stem)
            while True:
                event = runtime.wait_event(30)
                print(json.dumps(event, ensure_ascii=False))
                if event["type"] == "recovery_requested":
                    reply = json.loads(input("恢复参数 Recovery JSON（action: continue/abort）："))
                    runtime.reply_recovery(event["request_id"], reply)
                elif event["type"] in ("subtask_completed", "subtask_failed", "subtask_aborted"):
                    if event["type"] == "subtask_completed":
                        result = runtime.finish_task({"kind": "script_execution", "description": "已有 Python 程序执行完成 script completed"})
                    else:
                        result = runtime.abort_task(event.get("reason", "程序失败 script failed"))
                    break
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
