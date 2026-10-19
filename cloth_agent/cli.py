"""Standalone entry point. No device connection until Runtime is constructed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, MOCK_CONFIG, load_config, validate_hardware_config
from .runtime import Runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description="Code-as-policy cloth agent and script tools")
    parser.add_argument("--config", type=Path, help="Defaults to cloth_agent.yaml; demo uses cloth_agent_mock.yaml")
    parser.add_argument("--output-dir", type=Path, help="Exact directory for this task's artifacts")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="Two mock stages and one injected recovery")
    demo.add_argument("--with-codex", action="store_true", help="Use the real Codex thread instead of the deterministic driver")
    run = subcommands.add_parser("run", help="Let Codex generate and supervise task programs")
    run.add_argument("--task", default="将平铺的矩形抹布折叠两次。")
    run.add_argument("--folds", type=int, default=None)
    run.add_argument("--execute", action="store_true")
    script = subcommands.add_parser("script", help="Launch an existing ordinary Python subtask")
    script.add_argument("source", type=Path)
    script.add_argument("--params", type=Path)
    script.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config or (MOCK_CONFIG if args.command == "demo" else DEFAULT_CONFIG))
    if args.command == "demo" and config["backend"] != "mock":
        parser.error("demo requires backend: mock")
    allow_hardware = False
    if config["backend"] == "hardware":
        validate_hardware_config(config)  # Check blanks before opening any device.
        if not args.execute:
            parser.error("Hardware execution requires --execute and the existing EXECUTE confirmation.")
        if input("Confirm workspace, enabled arms and available e-stop; type EXECUTE: ").strip() != "EXECUTE":
            parser.error("Hardware execution was not confirmed.")
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
                parser.error("folds must be positive")
            result = CodexAppServer(runtime).run(f"{args.task}\n折叠次数参数：{folds}。")
        else:
            params = json.loads(args.params.read_text()) if args.params else {}
            runtime.submit_subtask(args.source.read_text(), params, args.source.stem)
            while True:
                event = runtime.wait_event(30)
                print(json.dumps(event, ensure_ascii=False))
                if event["type"] == "recovery_requested":
                    reply = json.loads(input("Recovery JSON (action: continue/abort): "))
                    runtime.reply_recovery(event["request_id"], reply)
                elif event["type"] in ("subtask_completed", "subtask_failed", "subtask_aborted"):
                    if event["type"] == "subtask_completed":
                        result = runtime.finish_task({"kind": "script_execution"})
                    else:
                        result = runtime.abort_task(event.get("reason", "script failed"))
                    break
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
