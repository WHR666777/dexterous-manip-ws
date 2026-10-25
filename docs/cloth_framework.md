# Cloth agent：代码阅读与运行入口

主流程为 `codex exec + MCP + 独立持久 Python worker + 独立机器人控制进程`。
本阶段接入右侧 Nero/L20 的基础 robo 接口、L515 RGB-D、内部运动规划与执行。
按用户要求只做语法检查，没有运行测试、mock、MCP/Codex 联调或设备连接；以下命令未在本轮执行。

## 阅读顺序

| 文件 | 职责 |
| --- | --- |
| `cloth_agent/cli.py` | `run`、`serve`、专用 `hold` / `hold-status` / `release-hold`；显式硬件连接开关 |
| `cloth_agent/session.py` | 冻结配置、管理进程与预算、cell/任务收尾、记录 |
| `cloth_agent/worker.py` | 持久 globals、注入 robo、执行 cell、普通异常与暂停回传 |
| `cloth_agent/robo.py` | 模型使用的代理；几何和 IK/TOPPRA 在 worker 内计算 |
| `cloth_agent/hardware.py` | 控制进程内设备连接、轨迹复核/采样、反馈、普通保持、人工固定位置保持 |
| `cloth_agent/control.py` | 协调器管理 Pipe、worker 普通 Pipe、人工专用 Unix datagram；人工保持和普通 I/O 分开处理 |
| `cloth_agent/mcp_server.py` | 临时 `python` / `finish` 工具；文字反馈与真正的 MCP ImageContent |
| `cloth_agent/codex_exec.py` | 一次 Codex task；中英混合 prompt、固定操作说明与本次配置快照 |
| `cloth_agent/robot_operations.md` | 启动时提供给 agent 的静态操作知识和 API 说明；代替 info 工具 |

`geometry.py`、`planning.py`、`config.py` 复用已有几何、IK、TOPPRA 与标定文件。
`PyRUA-Lean/` 仅作源码参考，不是运行依赖。没有旧 Runtime/app-server、framework 子包或 demo 兼容入口。

## robo 第一版边界

完整签名、返回值和示例语义见 [机器人操作说明](../cloth_agent/robot_operations.md)。
接口仅包括：`state`、`observe`、`show`、`pixel_to_task`、`top_down_pose`、`move_J`、`move_L`、
`open_gripper`、`close_gripper`、`pause_and_hold`。不增加 info、公开 plan/execute、抓布判定或双臂接口。

运动使用拇指/食指闭合接触点作为固定虚拟 TCP。参考姿态来自标定；top-down yaw 是绕 task +Z
相对参考姿态的增量。内部逐点 IK / TOPPRA 与样条执行是近似路径，没有场景避碰或接触力控。
控制进程复核关节/工作区/速度/加速度和实际起点，按配置周期下发目标，用实测关节判断到位。

默认运动等待结果；`wait=False` 的句柄只在当前 cell 有效。cell 返回前会暂停遗留轨迹并等待
固定目标附近的反馈稳定；不能确认时结束任务并报告失败。`pause_and_hold` 确认后展开调用栈，
新 cell 可以重新观察和规划。普通代码异常保留 globals，实际动作不回滚。

开合手只向拇指/食指发送配置预设，其他手指不发位置命令。返回包含 SDK 调用结果与手反馈，
不宣称命令送达或抓布成功。agent 再次 `observe` / `show` 查看图像判断；没有单独抓布检测器。

观察保存完整 RGB-D、内参、相机到 task 标定、相机时间戳/帧号和单独采样的机器人状态。
`show(obs, crop)` 将指定快照图像附在本 cell 的 MCP 返回中；裁剪坐标来自原图。
像素反投影始终使用同一快照深度，深度无效直接报错。相机与机器人不保证硬件同步。

## 人工停止保持与普通执行

`nero` 环境与人工静止保持步骤，见 [人工位置保持联调](cloth_estop_test.md)。

```text
Codex ↔ MCP / TaskSession ↔ 持久 Python worker（robo + 规划）
               │                         │ 普通数据 Pipe
               │ 管理 Pipe               ↓
               └──────────────────> 机器人控制进程
                                     ├─ 普通 I/O：轨迹、相机、手部与落盘
人工 hold ── 独立 Unix datagram ──> └─ 主线程：锁定 → 捕获实测关节 → 持续位置保持
```

人工请求不经过 Codex/MCP/cell/普通请求队列，不等待相机、手部或落盘。
只在 SDK 关节发送处串行化，防止旧轨迹帧与保持帧交错；已经进入 SDK 的发送仍可能延迟切换。
固定目标只捕获一次，重复 hold 不重抓；保持失败不自动改用阻尼急停、reset 或使能。
Nero SDK 的 `electronic_emergency_stop()` 是阻尼急停，现场已观察到下落，旧 `estop` 已禁用。

人工锁定后结束 worker/Codex，控制进程独立持续发送固定目标。父进程不自动杀掉保持进程，
控制进程忽略终端 Ctrl+C。操作者确认机械臂已安全落座/外部支撑后，另一个终端执行
`release-hold` 并输入 `SUPPORTED` 才结束保持与设备连接；不清动作锁，不恢复旧轨迹。
`hold-status` 经人工通道读取状态。`hold_target_sent` 只代表 SDK 调用返回，
`hold_confirmed` 仅为反馈接近固定目标且稳定的证据，不等于独立硬件停机确认。
L20 不自动松手/reset。旧会话协议不兼容，必须重启新版服务并使用新任务目录。

普通 Ctrl+C、超时、模型/worker 退出仍走普通任务取消。进程退出不证明硬件已停止。
软件位置保持依赖供电、CAN、反馈、使能和控制进程，不提供故障时或断电时的悬停保证。
生成代码的 import/私有属性约束不是完整安全沙箱。

## 依赖与命令

```bash
# 纯计算/MCP 入口；不加载设备配置，不连接任何硬件。
python -m pip install -e '.[agent]'
python -m cloth_agent serve --output-dir data/agent_runs/session-01
python -m cloth_agent run \
  --output-dir data/agent_runs/session-02 \
  --task '在两个 cell 中定义并复用一个函数，检查结果后结束。'

# 机器人入口还需项目现有 Nero/L20 SDK；--execute 直接连接真实硬件，无交互确认。
python -m pip install -e '.[agent,planning,vision]'
python -m cloth_agent run --config configs/cloth_agent.yaml --execute \
  --output-dir data/agent_runs/robot-01 \
  --task '先观察并展示抹布现场，再判断是否具备执行折叠的条件。'

# 另一个本地终端；直接到控制进程，绕过 MCP/Codex。
python -m cloth_agent hold data/agent_runs/robot-01/session.json
```

`--execute` 才允许连接配置中的右臂、右手与相机；硬件需由操作者预先使能，程序不会自动使能。
只传 `--config` 会加载标定用于几何计算，但不会连接设备；`state` 明确报告未连接，动作与观测拒绝。
不传配置时仍注入 robo，纯计算可用，设备能力拒绝。`serve` 支持相同参数，供外部 MCP client 使用；
外部 client 需自行提供机器人操作说明和本次 `robot_config.json`，不会自动启动 Codex 注入 prompt。

`run` 默认使用本机已登录的 Codex，不指定模型；可传 `--codex` / `--model`。
默认 cell 60 秒、任务 900 秒、40 个 cell；长运动需在启动时明确调整 `--cell-timeout`。
本轮没有安装/更新依赖，也没有执行上述命令。

## 记录与验证边界

任务目录包含 `session.json`、`robot_config.json`、`events.jsonl`、`control-events.jsonl`、
`cells/*.py` / `*.json`、`result.json`。真实观察保存在 `observations/`（PNG、米制深度 NPY、元数据）；
控制进程接受的内部轨迹保存在 `motions/`。Codex 模式还保存 `prompt.md`、原始事件、stderr 与最后回复。
日志使用英文标识符和中英说明；不会把整幅 RGB-D 数组序列化成文字日志。

新版人工保持完成代码和语法检查，尚未验证运行时控制时序或真实悬停效果。
任务 `completed` 是 agent 声明并经宿主收尾后的会话状态，不是自动测得的折布成功。
