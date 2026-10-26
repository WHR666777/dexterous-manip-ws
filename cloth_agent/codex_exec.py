"""一次 codex exec 对应一个任务，cell 的后续调用由 Codex 自己组织。

本模块只组装 prompt、配置 MCP 并保存 CLI 事件，不调用 app-server 或模型 HTTP API。
不硬编码模型名称；模型与推理强度可在任务配置中指定，null 沿用本机 Codex 设置。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


INSTRUCTIONS = """你是本次任务唯一的 agent，使用同一 Codex session 完成计算与执行反馈闭环。

语言 Language：计划、回复、代码注释、docstring、print/log、错误说明和总结均采用
中文为主、夹杂 English 技术术语的形式。工具名、JSON 字段、变量名和枚举值保留英文。

当前范围 Scope：使用 worker 中注入的 robo 操作已配置的右侧 Nero/L20。
只通过下方机器人操作文档列出的接口访问硬件；不导入设备 SDK，不修改项目或控制文件，
不使用 shell、网络或其他服务绕过 robo。设备未连接或能力缺失时如实报告，以 failure
或 stuck 结束，不编造物理效果。没有 info、公开 plan/execute 或抓布判定接口。

初始姿态 Startup：本次配置启用 startup 且 startup_prepared=true 时，宿主已经先回记录
关节位姿，再到约定的初始抓取位姿。不要重复执行准备 cell 000 或自行回零。
规划中的 yaw 是相对该初始抓取姿态、绕 task +Z 的增量，单位 rad；yaw=0 保持其朝向，
不是把法兰绝对 RPY yaw 设成 0，也不随每个 cell 的当前姿态重新锚定。

临时调用接口 Temporary tools：通过 cloth_agent MCP 服务的 python(code) 执行 cell，
通过 finish(status, summary) 结束任务；status 为 success / failure / stuck。
Python 环境预置 robo、np、math、time。变量、函数与允许的 import 跨 cell 保留。
最后一个表达式以 repr 文本回传；stdout/stderr/异常另行返回；robo.show 附加真实图片。
代码只能导入已允许的计算模块，不能访问文件、进程、网络或 Python 私有内部对象。

执行 Execution：一次只执行一个 cell；普通异常后环境保留，可以下一 cell 修正。
之前的变量修改和物理动作不会回滚。异步运动只能在当前 cell 内监督；返回前必须收尾。
pause_and_hold 确认保持后结束当前 cell，新 cell 重新观察再决策。
worker 超时/退出、任务预算用尽或人工停止保持都会结束任务，
此时不得重新执行旧代码。人工停止保持不由 agent 处理，不尝试清除锁定或复位设备。
完成后必须调用 finish，随后给出简短总结；运动到位不代表抓布或折布成功。
"""


def build_prompt(session, task):
    # 固定操作知识在启动时一次注入，不再让模型为了单位/TCP/手型调用 info 工具。
    operations = Path(__file__).with_name("robot_operations.md").read_text(encoding="utf-8")
    actual = {"hardware_connected": session.control.ready["hardware_connected"],
              "startup_prepared": session.startup_prepared,
              "robot_config": session.robot_config}
    return (INSTRUCTIONS + "\n" + operations
            + "\n\n本次冻结配置 Frozen configuration (null means not loaded)：\n```json\n"
            + json.dumps(actual, ensure_ascii=False, indent=2) + "\n```\n\n本次任务 Task：\n" + task)


def build_command(session, mcp_url, *, executable="codex", model=None):
    # 只把本次任务的 MCP 服务加入 CLI 参数；不改写用户全局 config.toml。
    # tool timeout 略长于 cell timeout，让宿主有机会先报告 worker 超时并执行清理。
    command = [executable, "exec", "--json", "--sandbox", "read-only",
               "--skip-git-repo-check", "--color", "never",
               "-C", str(session.directory / "workspace"),
               "-o", str(session.directory / "last_message.md"),
               "-c", "approval_policy=\"never\"",
               "-c", "project_doc_max_bytes=0",
               "-c", f"mcp_servers.cloth_agent.url={json.dumps(mcp_url)}",
               "-c", "mcp_servers.cloth_agent.required=true",
               "-c", "mcp_servers.cloth_agent.default_tools_approval_mode=\"approve\"",
               "-c", f"mcp_servers.cloth_agent.tool_timeout_sec={int(session.cell_timeout_s) + 10}"]
    # 优先级：显式 --model > 本轮 YAML > 本机 Codex 配置。推理强度从
    # 冻结后的 YAML 读取，用本次进程的 -c 覆盖，不修改 ~/.codex/config.toml。
    settings = session.robot_config or {}
    model = model if model is not None else settings.get("model")
    effort = settings.get("model_reasoning_effort")
    if model:
        command.extend(["--model", model])
    if effort is not None:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    command.append("-")
    return command


def run_codex(session, mcp_url, task, *, executable="codex", model=None):
    prompt = build_prompt(session, task)
    (session.directory / "prompt.md").write_text(prompt, encoding="utf-8")
    command = build_command(session, mcp_url, executable=executable, model=model)
    session.log("codex_starting", "启动 Codex exec Starting task agent", command=command)
    # 原始 JSONL 留作 review；stderr 单独保存，不能与 JSON 事件混流。
    with (session.directory / "codex_events.jsonl").open("wb") as events, (
            session.directory / "codex_stderr.log").open("wb") as errors:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                       start_new_session=True)
            session.attach_agent(process)
            try:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.close()
            except BrokenPipeError:
                pass  # CLI 提前退出时继续收集退出码，不能把它误判为任务完成。
            finished_at = None
            while process.poll() is None:
                if session.ended.wait(.05):
                    # finish 的 MCP 回执和最后一句模型总结需要时间写完，因此正常结束
                    # 不立即杀掉 Codex；人工停止保持/超时已由协调器另行终止进程组。
                    finished_at = finished_at or time.monotonic()
                    if time.monotonic() - finished_at > 5:
                        session._stop_agent()
                        break
                    time.sleep(.02)
            returncode = process.wait()
            session.log("codex_exited", "Codex 已退出 Task agent exited", returncode=returncode)
            if session.status == "running":
                session.abort(f"Codex 未提交结束结果 Agent exited without finish: {returncode}")
        except BaseException:
            session.abort("Codex 启动或运行失败 Agent launch/run failed")
            raise
    return session.summary()
