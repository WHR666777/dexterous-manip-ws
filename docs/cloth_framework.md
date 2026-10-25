# 独立进程框架第一步：代码阅读与运行入口

本轮按 `stop-that-shit` 收住范围：只新增任务协调、`codex exec`、临时 MCP 入口、独立持久 Python worker 和独立控制进程。不实现 `robo`、运动、观测、Nero/L20 设备接入，也不实现 mock 运动。

按用户要求，本轮只检查 Python 语法，没有运行 mock、单元测试、MCP/Codex 联调或实机验证。以下是代码提供的入口，运行行为尚待后续验证。

## 阅读顺序

| 文件 | 这一轮实现的内容 |
| --- | --- |
| `cloth_agent/__main__.py`、`cloth_agent/cli.py` | 统一入口与 `run`、`serve`、`estop` 命令 |
| `cloth_agent/session.py` | 启动两个独立子进程；任务/次数/超时管理；源码与结果记录；进程监测 |
| `cloth_agent/worker.py` | spawn worker；跨 cell globals；stdout/stderr/末表达式/异常；进程终止 |
| `cloth_agent/control.py` | 控制状态进程；普通 Pipe 与专用 Unix datagram 人工急停入口；急停不可被普通任务结束解除 |
| `cloth_agent/mcp_server.py` | 使用 MCP SDK 暂时提供 `python`、`finish`；HTTP 在协调进程内，源码在 worker 执行 |
| `cloth_agent/codex_exec.py` | 中英混合 prompt；一次 `codex exec`；保存原始 JSONL 和 stderr |

主流程代码直接位于 `cloth_agent/`，不再保留 framework 子包或旧 Runtime/app-server、脚本参数回信、demo 兼容入口。旧流程专用的执行器、设备后端、观测封装和对应测试已删除。`geometry.py`、`planning.py`、`config.py` 及纯规划测试保留；其中几何代码仍供标定程序使用。后续机器人工具 schema 尚未冻结，当前 `python`、`finish` 仅用于打通基础调用链。

## 进程与退出路径

```text
协调进程（MCP 服务 + TaskSession）
    ├─ codex exec 进程
    ├─ 持久 Python worker 进程：只有 np/math/time 等计算环境，没有 robo
    └─ 控制进程：只有准入/结束/急停状态，没有设备连接

人工 estop 命令 ── Unix datagram ──> 控制进程先锁定
                                      └─ Event 通知协调进程，再清理 worker/Codex
```

普通 cell 异常返回 traceback，globals 保留。cell 超时或 worker 退出结束整个任务；不自动重启 worker。人工急停直接由控制进程接收，不排进 MCP、worker 或普通 Pipe 请求；协调器监测到锁定后终止本次任务。

**当前并没有发送 Nero 电子急停。** `estop` 只锁定控制状态，回执明确包含 `hardware_connected: false`、`stop_confirmed: false` 和右臂 `not_sent`。连接设备、实际暂停保持、电子急停发送及确认均留待设备适配阶段；此入口不能当成已验证的实机急停。

控制进程不支持清除急停、reset 或 enable。当前没有任何运动命令，也不自动操作 L20。

Python 的导入与私有属性限制只是生成代码约束，不是完整安全沙箱。worker 不持有设备或控制通道；持久进程被强制终止后，内存变量不会恢复。

## 依赖与命令

在选定的 Python 环境安装新框架依赖；这一轮没有完成依赖安装，也没有执行以下启动命令：

```bash
python -m pip install -e '.[agent]'

# 仅启动任务服务，随后可接外部 MCP client。
python -m cloth_agent serve --output-dir data/agent_runs/session-01

# 一次 Codex task；当前只能请求计算任务，不能请求折布。
python -m cloth_agent run \
  --output-dir data/agent_runs/session-02 \
  --task '在两个 cell 中定义并复用一个函数，检查结果后结束。'

# 在另一个终端直接请求控制状态锁定，无需 MCP/Codex。
python -m cloth_agent estop data/agent_runs/session-02/session.json
```

`run` 默认使用本机已登录的 `codex`，不指定模型；可显式传 `--codex` 或 `--model`。cell 默认 60 秒、任务 900 秒、最多 40 个 cell，均可通过 CLI 调整。Ctrl+C 是普通任务取消，与专用人工急停入口分开。

每个任务记录：`session.json`（PID 和人工入口）、`events.jsonl`（中英事件说明）、`cells/*.py` 与 `cells/*.json`、`result.json`；使用 Codex 时另有 `prompt.md`、`codex_events.jsonl`、`codex_stderr.log` 和 `last_message.md`。框架的 completed 仅表示本次计算会话结束，不能解释为物理折布成功。

后续 review 优先确认进程边界和退出语义，再讨论 `robo` 与机器人设备适配；架构基线见 [closth_agent.md](closth_agent.md)。
