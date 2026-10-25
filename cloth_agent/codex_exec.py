"""一次 codex exec 对应一个任务，cell 的后续调用由 Codex 自己组织。

本模块只组装 prompt、配置 MCP 并保存 CLI 事件，不调用 app-server 或模型 HTTP API。
不硬编码模型名称；默认沿用操作者本机 Codex 的登录和模型配置。
"""

from __future__ import annotations

import json
import subprocess
import time


INSTRUCTIONS = """你是本次任务唯一的 agent，使用同一 Codex session 完成计算与执行反馈闭环。

语言 Language：计划、回复、代码注释、docstring、print/log、错误说明和总结均采用
中文为主、夹杂 English 技术术语的形式。工具名、JSON 字段、变量名和枚举值保留英文。

当前范围 Scope：本轮只有独立持久 Python worker 和独立控制进程的框架。
没有 robo、相机、关键点、运动规划、设备连接或真实抓布反馈。不得导入设备 SDK，
不得修改项目或控制文件，不得使用 shell、网络或其他服务绕过此限制。
遇到需要机器人能力的任务，报告当前缺项，以 failure 或 stuck 结束，不编造物理效果。

临时调用接口 Temporary tools：通过 cloth_agent MCP 服务的 python(code) 执行 cell，
通过 finish(status, summary) 结束任务；status 为 success / failure / stuck。
Python 环境预置 np、math、time。变量、函数与允许的 import 跨 cell 保留。
最后一个表达式以 repr 文本回传；stdout/stderr/异常另行返回。
代码只能导入已允许的计算模块，不能访问文件、进程、网络或 Python 私有内部对象。

执行 Execution：一次只执行一个 cell；普通异常后环境保留，可以下一 cell 修正。
之前的变量修改不会回滚。worker 超时/退出、任务预算用尽或人工急停都会结束任务，
此时不得重新执行旧代码。人工急停不由 agent 处理，不尝试清除锁定或复位设备。
完成后必须调用 finish，随后给出简短总结；框架计算成功不代表抓布或折布成功。
"""


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
    if model:
        command.extend(["--model", model])
    command.append("-")
    return command


def run_codex(session, mcp_url, task, *, executable="codex", model=None):
    prompt = INSTRUCTIONS + "\n本次任务 Task：\n" + task
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
                    # 不立即杀掉 Codex；人工急停/超时已由协调器另行终止进程组。
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
