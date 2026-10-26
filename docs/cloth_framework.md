# Cloth agent：代码阅读与运行入口

主流程为 `codex exec + MCP + 独立持久 Python worker + 独立机器人控制进程`。
本阶段接入右侧 Nero/L20 的基础 robo 接口、L515 RGB-D、内部运动规划与执行。
初版仅做语法检查。2026-10-25 已在现场监护下执行真实控制进程的固定目标检查，
在 JS 目标不变时复现下移，未进入后续 J7 小幅运动；该检查失败，不能作为抓取可用的证明。
记录位于 `data/agent_runs/control-check-20261025-01/`。后续的超时保持修正已检查语法，
随后已改为普通 J 分段执行，并完成真实 worker/控制进程链路的小幅实机检查。

同日使用独立脚本 `scripts/nero/check_move_joints.py` 完成两轮普通 J 单目标检查：
每轮只发送原位、J7 +0.01 rad、返回三个目标，固件速度百分比为 10。
两轮最后一秒的关节反馈均与目标一致（仅指编码器反馈精度内），没有持续跟踪偏差。
第一轮从此前 JS 状态接管时，FK 换算的虚拟 TCP 短暂下移约 4.7 mm 后恢复；
第二轮已处于普通 J 模式，原位保持未记录到高度变化。小幅 J7 运动本身使虚拟 TCP
高度变化约 2.0 mm，属于该目标的几何结果。记录为
`data/agent_runs/joint-position-check-01/` 和 `joint-position-check-02/`。
上述单点检查之后，`agent-joint-check-01` 通过真实 TaskSession/持久 worker/控制进程验证：
J7 0.01 rad 往返、虚拟 TCP +Z 5 mm 分段运动（FK 抬升约 4.70 mm）及返回、普通暂停、
下一 cell 恢复，以及小幅运动中的独立人工 J 保持。人工捕获目标后末段关节误差为 0；
确认外部支撑后已释放并结束连接。未测试完整自动准备、Codex 抓取或更大路径。

`robot-grasp-04` 在自动准备的路径复核阶段因 J1 反馈年龄 0.515 s 超过 0.5 s 而失败，
没有发送运动目标。已删除控制进程内不执行的密集 TOPPRA 样条复核，并将固定变换校验
从逐点重复改为每次路径检查只做一次。保留实际 J 几何目标、关节分段插值、限位、工作区、
跨度与执行前真实起点复核，不放宽反馈新鲜度门限。`robot-grasp-04-validation-check`
只读实机重复复核同一计划 100 次：平均约 18 ms、最大约 40 ms，最大反馈年龄约 39 ms，
没有运动命令或关节变化。这不代替完整启动/抓取复测。

`robot-grasp-09` 已完成启动两次 PTP、原地抬高与开手；随后横移在第 18/34 个目标超时，
未执行抓取。最后 J4 实测 1.177 rad，旧目标 1.177379 rad，新目标 1.159258 rad；
后续约 30 秒停在旧目标附近。现有记录缺少固件状态，能确认新目标未有效执行，
尚不能确认是提前覆盖导致固件未接受，还是其他下发问题。
针对当前仅凭关节容差推进的缺口，新增新鲜状态帧门限：`mode_feedback=1`（J）、
`motion_status=0`（到位），且状态帧接收时间晚于本次下发；仍同时检查实际关节误差。
逐周期采样保存固件状态、目标下发 Unix 时间和点号；超时反馈包含逐轴误差与状态诊断。
不重发、不放宽容差、不增加固定中间等待，人工保持路径保持独立。此修改仅完成语法检查，
没有连接设备复测，不能宣称已解决固件未执行目标的问题。

随后 `robot-grasp-10` 在尚未发启动运动时读到 `arm_status=2`（固件 NO_SOLUTION），
因此拒绝启动。这是控制器反馈，不是本轮 Python IK 失败；可能是之前动作遗留的状态，
没有上一轮固件状态记录，无法确认起因。`motion_status=0` 不能抵消这个异常。
连接预检现在先检查固件异常，再连接 L20/相机；错误同时给出状态名称与原始字段。
通过现场操作流程排查恢复后再启动，agent 不自动 reset、使能或忽略该状态。

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
| `cloth_agent/monitor.py` / `monitor_assets/` | 独立只读 HTTP/SSE 观察页；时间线、cell 代码/结果和观测图片，不进入控制链路 |
| `cloth_agent/robot_operations.md` | 启动时提供给 agent 的静态操作知识和 API 说明；代替 info 工具 |

`geometry.py`、`planning.py`、`config.py` 复用已有几何、IK、TOPPRA 与标定文件。
`PyRUA-Lean/` 仅作源码参考，不是运行依赖。没有旧 Runtime/app-server、framework 子包或 demo 兼容入口。

## robo 第一版边界

完整签名、返回值和示例语义见 [机器人操作说明](../cloth_agent/robot_operations.md)。
接口仅包括：`state`、`observe`、`show`、`pixel_to_task`、`top_down_pose`、`move_J`、`move_L`、
`open_gripper`、`close_gripper`、`pause_and_hold`。不增加 info、公开 plan/execute、抓布判定或双臂接口。

运动使用拇指/食指闭合接触点作为固定虚拟 TCP。初始抓取姿态保留记录起始位姿的法兰位置/yaw，
roll/pitch 取标定的示教参考；规划 yaw 是相对该姿态绕 task +Z 的增量，单位 rad。
`startup.enabled: true` 时，人工入口与记录就绪后，宿主用 worker 中的固定 cell 000 先以
PTP 回记录关节，再对初始抓取位姿求一次最终 IK 并执行 PTP。每段仅通过普通 J 下发一次
最终关节目标，由固件平滑；不下发中间点，不进行 TCP 直线插值或 TOPPRA 时间规划。
启动专用辅助函数 `startup_move_ptp` 仅在 cell 000 中注入，之后删除，不新增 agent API。
整体关节变化上限与 teleop 回记录位姿一致，为 2π rad；限位、起点、反馈和工作区检查保留。
插值检查点仅用于工作区复核，不代表固件实际路径或场景避碰。等待到位、保持收尾并检查实际 TCP
满足 IK 的位置/角度容差后才启动 MCP/Codex；失败/超时/人工停止则结束准备，不启动 agent。
准备不占模型 cell 预算，源码/完整反馈保存在 `cells/000.py` / `000.json`，事件记录
`startup_started` / `startup_prepared`。准备总超时由 `startup.timeout_s` 控制，默认 120 秒。
已在记录关节容差内时跳过第一段，不为零位移下发多余运动。每条运动在最终终点反馈满足关节容差、
并按普通保持标准稳定至少 0.2 秒后才返回 completed，之后才允许规划下一段。
若 cell 收尾失败，仍保留原始 stdout、异常、robo 调用记录和最后控制快照；
`execution_status` 记录 worker 原始状态，`control_error` 单独记录收尾错误。
人工保持联调配置显式禁用 startup；不传 `--execute` 也不会准备或运动。
公开 `move_J(target_tcp, wait=True)` 与 `move_L(target_tcp, wait=True)` 都接收 task 系虚拟
TCP 位姿（4×4 SE(3) 或 xyz-rpy，m/rad）。`move_J` 以实测关节为参考，对目标法兰位姿
求一次 IK，生成 `mode=ptp` 的两个端点；普通 J 仅发送一次终点，由固件完成 PTP 平滑，
不做 TOPPRA，不保证中途 TCP 直线。PTP 复用启动回位的 2π rad 整体变化上限，
保留关节限位、工作区、起点与到位检查。agent 无需自行求 IK，IK 失败不发动作。
内部逐点 IK 与普通 J 几何分段仅用于 `move_L` 等路径规划，没有场景避碰或接触力控。
`MotionPlan.joint_waypoints` 保存原始几何分段点；控制进程复核这些点、分段关节插值的
限位/工作区/最大跨度和实际起点。TOPPRA 样条数值检查留在 worker，
控制进程不重复检查并未执行的密集样条，避免计算占用拖慢 CAN 接收。
独立轨迹线程按配置周期读反馈，仅切换分段时发送一次 `NeroArm.move_joints`，
中间点达到关节容差，并收到本次下发后的新鲜 J 模式到位反馈，再于下一监督周期发送下一段，
不再额外等待 0.2 秒稳定；
仅整条运动的最终终点和保持确认保留稳定检查，不把密集计时样本灌入普通 J。
`joint_speed_percent` 默认 10，到位容差默认 0.003 rad。固件执行自己的位置运动处理，
TOPPRA 时间不再驱动执行；规划 rad/s、rad/s² 约束不等于实际速度/加速度保证。
`move_L` 仍按虚拟 TCP 直线和姿态插值做 IK，由普通 J 分段近似，不保证连续严格直线。
到位超时保留最后目标并报告失败，不重抓下坠后的反馈作为新目标。
`control-samples.jsonl` 记录命令发送事件（`command_sent=true`）及每个监督周期的目标/实测关节与 TCP，
由普通线程消费内存队列后落盘；记录不是总线上逐帧 ACK。
实机连接另存 `arm-can.jsonl`：SDK 的每帧 Bus.send 返回/异常、独立 SocketCAN
接收到的模式/七轴目标 local echo、原始状态及关节反馈，以及每秒接口累计统计。
用时间戳关联目标与原始帧，用本轮前后 `tx_dropped` 差值判断新增丢包；
local echo 不代表机械臂已经接收或执行整条目标。抓包线程不发送、不重试、不改队列。
SDK 发送 socket 为此开启本机回环，自己的 SDK 接收仍不接收自身发送帧。
仅检查上升时可直接运行 `python scripts/nero/check_agent_lift.py`：使用本轮配置目录，
不启动 Codex/MCP、不执行启动 PTP，保持当前 TCP 方向上升 4 cm，不操作手指或自动下降。
复现已知失败时优先使用 `python scripts/nero/replay_agent_motion.py <历史 motions/id.json>
--output-dir <新目录>`。它读取当轮冻结配置，先以单次 PTP 回历史起点，再将原始
MotionPlan 送入正式控制进程，不重新求 IK 或改变关节分段；人工 hold 使用该目录
下的 `session.json`。它不恢复手部动作、抓布负载及当时的线程调度时序，因此相同
轨迹仍可能需要重复对照，单次成功不表示原问题已经修复。

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
                                     ├─ 普通 I/O：请求、轨迹校验、图像与落盘
                                     ├─ 轨迹线程：周期反馈监督与分段 J 单次下发
                                     ├─ L20 查询：复用遥操作后台反馈线程
人工 hold ── 独立 Unix datagram ──> └─ 主线程：锁定 → 捕获实测关节 → 持续位置保持
```

人工请求不经过 Codex/MCP/cell/普通请求队列，不等待相机、手部或落盘。
只在 SDK 关节发送处串行化，防止旧轨迹帧与保持帧交错；已经进入 SDK 的发送仍可能延迟切换。
固定目标只捕获一次，重复 hold 不重抓；保持失败不自动改用阻尼急停、reset 或使能。
Nero SDK 的 `electronic_emergency_stop()` 是阻尼急停，现场已观察到下落，旧 `estop` 已禁用。

人工锁定后结束 worker/Codex，控制进程发送一次普通 J 固定目标并独立监督反馈。父进程不自动杀掉保持进程，
控制进程忽略终端 Ctrl+C。操作者确认机械臂已安全落座/外部支撑后，另一个终端执行
`release-hold` 并输入 `SUPPORTED` 才结束保持与设备连接；不清动作锁，不恢复旧轨迹。
`hold-status` 经人工通道读取状态。`hold_target_sent` 只代表 SDK 调用返回，
`hold_confirmed` 仅为反馈接近固定目标且稳定的证据，不等于独立硬件停机确认。
L20 不自动松手/reset。普通 J 使用 `position-hold-J-v2` 协议；旧 JS 会话不兼容，必须重启新版服务并使用新任务目录。

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
python -m cloth_agent run --execute \
  --task '先观察并展示抹布现场，再判断是否具备执行折叠的条件。'

# 另一个本地终端；直接到控制进程，绕过 MCP/Codex。
python -m cloth_agent hold
```

`--execute` 才允许连接配置中的右臂、右手与相机；硬件需由操作者预先使能，程序不会自动使能。
CLI 默认使用 `configs/cloth_agent.yaml` 中的 `session.output_dir` 作为本轮记录目录，
相对路径以配置文件所在目录为基准。每轮测试前修改一次，本轮运行期间不改；旧记录不覆盖。
`run` / `serve` / `monitor` / `hold` / `hold-status` / `release-hold` 共用此目录；
人工命令自动补上 `session.json`。使用其他配置时，各命令加同一个 `--config`。
`--output-dir`、monitor 的目录位置参数、人工命令的 session 文件位置参数仍可显式覆盖。
目录解析位于轻量的 `paths.py`，仅读取 YAML，不计算标定/FK、不创建进程或连接硬件。
只传 `--config` 会加载标定用于几何计算，但不会连接设备；`state` 明确报告未连接，动作与观测拒绝。
不传配置时仍注入 robo，纯计算可用，设备能力拒绝。`serve` 支持相同参数，供外部 MCP client 使用；
外部 client 需自行提供机器人操作说明和本次 `robot_config.json`，不会自动启动 Codex 注入 prompt。

`run` 使用本机已登录的 Codex，可在 YAML 顶层配置 `model` 和 `model_reasoning_effort`。
两项默认 `null`，沿用本机 Codex 设置；例如填入所需模型名称及 `model_reasoning_effort: high`。
模型和推理强度的具体支持情况由 Codex 校验。`--model` 优先于 YAML 的 `model`，
`--codex` 可指定可执行文件；配置只覆盖本轮启动参数，不修改全局 Codex 配置。
默认 cell 60 秒、任务 900 秒、40 个 cell；长运动需在启动时明确调整 `--cell-timeout`。
本轮没有安装/更新依赖，也没有执行上述命令。

## 记录与验证边界

运行中可在另一个终端执行 `python -m cloth_agent monitor`（或显式传入历史任务目录），
打开终端打印的本地地址查看实时记录；也可回看旧任务。详细用法见 [观察页说明](cloth_monitor.md)。

任务目录包含 `session.json`、`robot_config.json`、`events.jsonl`、`control-events.jsonl`、
`cells/*.py` / `*.json`、`result.json`。真实观察保存在 `observations/`（PNG、米制深度 NPY、元数据）；
控制进程接受的内部轨迹保存在 `motions/`。Codex 模式还保存 `prompt.md`、原始事件、stderr 与最后回复。
日志使用英文标识符和中英说明；不会把整幅 RGB-D 数组序列化成文字日志。

新版人工保持完成代码和语法检查，尚未验证运行时控制时序或真实悬停效果。
任务 `completed` 是 agent 声明并经宿主收尾后的会话状态，不是自动测得的折布成功。
