# Cloth agent：代码阅读与运行入口

主流程为 `codex exec + MCP + 独立持久 Python worker + 独立机器人控制进程`。
当前支持左右 Nero/L20 独立 J/L 执行、软件同时启动和一份共享 L515 RGB-D 观测。
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
没有上一轮固件状态记录，无法确认起因。上述是该轮历史行为。
当前按操作者确认的含义，将 NO_SOLUTION 作为状态反馈保留，不视为需要清错的故障，
不锁存、不取消目标、不额外发送原位保持，也不等待 NORMAL 才允许新命令。
已移除此前引入的恢复接口和启动恢复流程；agent 可根据实测状态调整目标直接提交新运动，
仍遵守每侧一次一条运动与原有到位监督。人工 hold、通信及其他固件故障的处理保持不变。

可恢复操作现与设备故障分开：目标/限位拒绝和 J/L 超时在确认普通保持后返回动作失败；
相机采集/观测保存异常、手部反馈数据不可用或位置未到位作为 cell 错误返回，保留 worker
与 Codex 会话，由 agent 在后续 cell 调整。机械臂失能、反馈失效、通信故障、未知控制异常
及无法确认保持仍结束会话；启动失败、预算/超时、人工取消等生命周期退出规则不变。
fold-30 的保持目标不再直接回发轻微越限的实测值：仅对保持允许最多 1 mrad 边界差异，
将目标收至官方区间内的整 mrad 边界，保留原始反馈和边界调整日志；不放宽 agent 目标。
以上修改仅完成语法检查，没有连接实机验证。

## 阅读顺序

| 文件 | 职责 |
| --- | --- |
| `cloth_agent/cli.py` | `run`、`serve`、专用 `hold` / `hold-status` / `release-hold`；显式硬件连接开关 |
| `cloth_agent/session.py` | 冻结配置、管理进程与预算、cell/任务收尾、记录 |
| `cloth_agent/worker.py` | 持久 globals、注入 robo、执行 cell、普通异常与暂停回传 |
| `cloth_agent/robo.py` | 模型使用的代理；几何和 IK/TOPPRA 在 worker 内计算 |
| `cloth_agent/hardware.py` | 控制进程内设备连接、轨迹复核/采样、反馈、普通保持、人工固定位置保持 |
| `cloth_agent/control.py` | 协调器管理 Pipe、worker 普通 Pipe、人工专用 Unix datagram；人工保持和普通 I/O 分开处理 |
| `robot_control/errors.py` | 明确标记可返回 agent 的目标拒绝、观测/手部数据异常、手部到位超时；未标记的控制异常仍终止 |
| `cloth_agent/mcp_server.py` | 临时 `python` / `finish` 工具；文字反馈与真正的 MCP ImageContent |
| `cloth_agent/codex_exec.py` | 一次 Codex task；中英混合 prompt、固定操作说明与本次配置快照 |
| `cloth_agent/monitor.py` / `monitor_assets/` | 独立只读 HTTP/SSE 观察页；时间线、cell 代码/结果和观测图片，不进入控制链路 |
| `cloth_agent/robot_operations.md` | 启动时提供给 agent 的静态操作知识和 API 说明；代替 info 工具 |
| `cloth_agent/grasp_experience.md` | 正文随启动 prompt 注入，其中链接的 PNG 直接通过 `codex exec --image` 附加 |

`geometry.py`、`planning.py`、`config.py` 复用已有几何、IK、TOPPRA 与标定文件。
`PyRUA-Lean/` 仅作源码参考，不是运行依赖。没有旧 Runtime/app-server、framework 子包或 demo 兼容入口。


## 双臂执行与合并标定

`configs/cloth_agent.yaml` 通过 `calibration.file` 读取自包含的
`configs/cloth_agent_calibration.yaml`。公共 task 是右臂基座，左基座通过两份相机外参
计算；两侧 `flange_from_tcp` 完整保留原始标定，不镜像或重定义左侧工具轴。
左侧恢复原始参考角 `[-87.781805°, -20.890762°]`，拟合位置保持不变。双侧共享 open/closed/replay_start
手型；各侧 `calibration_hand_pose_raw` 记录各自实际标定手型。

`cloth_agent/bimanual.py` 共享相机、cell 和人工停止 Event，分别实例化
`hardware.RobotController(arm_name=...)`。单臂执行器各自拥有设备连接、反馈、运动、
目标发送锁和保持目标。控制周期逐侧推进已有 J/L 状态机，不进行联合轨迹规划。
规划和路径复核在周期锁外；提交时复核所有起点，再在同一调度边界发布两侧目标。
`move_both` 要求两侧无活动运动且固件空闲；准备或起点复核失败时两侧都不启动。
软件同时启动不保证硬件起动时刻或到达时刻相同。

`robo.state()` 增加 `arms.left/right`，顶层 joints/tcp 保留默认右侧视图；全局
quiescent 和 hold_confirmed 要求两侧满足。`state(arm=...)` 返回单侧状态。
运动、top_down_pose 和手型接口使用 `arm="left"` / `"right"`，省略时默认右侧。
`move_both(left=..., right=..., mode="J"或"L", wait=True)` 返回按侧索引的结果；
`wait=False` 返回两个独立句柄。句柄仍局限当前 cell，不能跨 cell 遗留运动。
启动准备先张开两手，再逐侧执行原有两段 PTP 并检查两侧实际 TCP。
PTP 大幅腕部转动的一次 IK 允许 240 次评估；遥操作默认预算、FK 精度、关节和工作区限制不变。

任一侧运动执行失败、普通暂停、cell 收尾或任务终止覆盖两侧；人工 hold 先撤销两侧
动作权限，再各自捕获一次固定保持目标。重复 hold 不重抓任一侧目标。
CAN 抓包和单侧计划位于 `arms/left/`、`arms/right/`；公共控制采样带 arm 字段，
监控页面分别呈现两侧运动结果。该变更已做离线执行器验证，尚未执行双臂实机运动。

## robo 第一版边界

完整签名、返回值和示例语义见 [机器人操作说明](../cloth_agent/robot_operations.md)。
接口仅包括：`state`、`observe`、`show`、`pixel_to_task`、`top_down_pose`、`move_J`、`move_L`、
`open_gripper`、`close_gripper`、`gripper_replay_start`、`pause_and_hold`。
`gripper_replay_start` 使用配置中的示范首帧手型，仅操作拇指/食指、不启动回放；
默认预设来自 `data/grasp_debug/success-4/robot_config.json`。
不增加 info、公开 plan/execute、抓布判定或双臂接口。

运动使用各自标定的固定虚拟 TCP：右侧沿用已有拈布点，左侧为第 0 帧手型的食指指尖参考点。
标定文件中的 `reference_roll_pitch_deg` 只描述工具轴定义，不决定初始运动目标。
启动 TCP 位姿在 `startup.tcp_poses` 中独立配置：直接值为 task 系 SE(3) 或
`[x,y,z,roll,pitch,yaw]`（m/rad），位置镜像项使用 `mirror_position_of` 和 `plane_y_m`。
当前右侧显式保留 `bi_fold-01` 的目标；左侧关于 task 的竖直平面
`Y=-0.26764081143104956 m` 仅镜像右侧位置。令 `S=diag(1,-1,1)`，位置为
`p_left=S @ p_right+[0,2*plane_y_m,0]`，旋转为 `R_left=R_right`，
因此两侧保持 X、Z 相同且在公共 task 中朝向一致。两侧 task Z 均为 `0.00772207748671605 m`。
`initial_grasp_tcp` 是解析后的启动目标，`reference_tcp_rotation` 取该目标旋转；
规划 yaw 是相对该朝向绕 task +Z 的增量，单位 rad。
启用启动时每侧必须配置 TCP 目标和 `workspace.start_pose_file`。禁用启动且未配置目标时，
不合成启动目标；计算接口只保留记录位姿的工具朝向。
`startup.enabled: true` 时，人工入口与记录就绪并确认 quiescent 后，宿主用固定 cell 000 调用
与 wrist-only 相同的 `open_hand()` 发送全手张开预设，再以 PTP 回记录关节，
随后对初始抓取位姿求一次最终 IK 并执行 PTP。全手张开只确认 SDK 返回，不宣称五指到位；
不执行 hand reset，不改变普通 `open_gripper` 只操作拇指/食指的约定。
每段仅通过普通 J 下发一次
最终关节目标，由固件平滑；不下发中间点，不进行 TCP 直线插值或 TOPPRA 时间规划。
启动专用辅助函数 `startup_open_hand` / `startup_move_ptp` 仅在 cell 000 中注入，之后删除，
不新增 agent API。禁用 startup 或不连接硬件时，不发送全手张开命令。
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
`move_L` 使用 `LinearMotionPlan`，只保存实测起始关节和 task 系虚拟 TCP 终点，
不生成主机逐点 IK、TOPPRA 时间表或普通 J 分段。控制进程复核起点与目标工作区，
按标定转换为 base 法兰 xyz-rpy，再调用 `NeroArm.move_linear`，由固件完成内部规划。
固定朝向时虚拟 TCP 也随法兰沿直线平移；同时旋转时，带偏移的 TCP 不保证严格直线。
这里只检查起点/终点工作区，不预测固件内部轨迹，也没有场景避碰或接触力控。

厂家 L 的监督移植自 `scripts/nero/check_move_linear_zbx.py`，拆成固定周期状态机：
先等 `motion_status=0`（`linear_pre_idle_timeout_s`，默认 10 s），发送后等待新鲜
L 模式 `motion_status=1`（`linear_start_timeout_s`，默认 1 s）。未确认启动时重发
同一目标，`linear_max_attempts` 默认 3 次、包含首次；耗尽后仍进入到位等待，兼容未采样到启动沿的短动作。
随后最多等待 `completion_timeout_s`（默认 30 s）：TCP 位置/角度误差达标，
相邻关节变化不超过 0.002 rad，持续稳定 0.2 s，且新鲜 L 模式状态为已到位。
默认位置/角度阈值沿用脚本的 **0.02 m / 2°**，分别可由 `linear_position_tolerance_m`
和 `linear_angle_tolerance_rad` 配置。完成后保持固件 L 目标，不额外切回 J；
超时则返回 failed，并通过普通 J 固定当前关节，保持确认后才能结束 cell。
各阶段不阻塞 RPC 或人工 hold；`wait=False` 仍返回原有 cell 内运动句柄。

`move_J` 与启动 PTP 继续使用 `MotionPlan` 的两个关节端点；完成必须同时满足本次
发送后的新鲜 J 模式 `motion_status=0` 和关节误差不超过 `joint_arrival_tolerance_rad`
（默认 0.015 rad）。`0` 且误差大时继续等待，避免将启动前的空闲反馈误判为到位失败；
`1` 时继续监督，短动作也不强制采样到 `1`。超过 `completion_timeout_s` 仍未达标则
报告 failed，并请求普通 J 固定当前关节，保留原目标/误差诊断，不让旧目标继续运动。
不增加固定等待或运动停稳窗口；普通保持确认仍需 0.2 s 稳定。J/L 的 `joint_speed_percent` 默认 10，
主机规划 rad/s、rad/s² 参数不构成固件实际速度/加速度保证。
`joint_frame_interval_s` 默认 0.001：七轴目标的 `0x155/156/157/170` 四帧
在主机相邻发送调用之间保留至少 1 ms，避免集中下发时实机出现停在上一目标的现象。
该节拍仅针对 J 的四帧关节目标，每个 J 目标仍只发送一次；不作用于厂家 L 的三帧法兰目标。
节拍只在 agent 控制连接的单组发送锁内运行，不通过 Codex/MCP 调度；遥操作不启用此发送配置。
`control-samples.jsonl` 记录命令发送事件（`command_sent=true`）及每个监督周期的目标/实测关节与 TCP，
L 另外记录阶段、发送次数、启动确认及 TCP 误差；由普通线程消费内存队列后落盘，
记录不是总线上逐帧 ACK。`MotionResult.diagnostics` 保留 L 的 TCP 误差、法兰目标及最新固件状态。
实机连接另存 `arm-can.jsonl`：SDK 的每帧 Bus.send 返回/异常、独立 SocketCAN
接收到的模式/七轴目标 local echo、原始状态及关节反馈，以及每秒接口累计统计。
用时间戳关联目标与原始帧，用本轮前后 `tx_dropped` 差值判断新增丢包；
local echo 不代表机械臂已经接收或执行整条目标。抓包线程不发送、不重试、不改队列。
SDK 发送 socket 为此开启本机回环，自己的 SDK 接收仍不接收自身发送帧。
仅检查上升时可直接运行 `python scripts/nero/check_agent_lift.py`：使用本轮配置目录，
不启动 Codex/MCP、不执行启动 PTP，保持当前 TCP 方向上升 4 cm，不操作手指或自动下降。
加 `--repeat 10 --output-dir <新目录>` 可在同一 worker/控制进程内执行 10 轮
上升和返回；所有轮次使用第一轮捕获的固定起点/终点，任何一段失败立即停止，
不会继续返回或重试失败动作。测试不修改配置中的日常任务目录。
复现已知失败时优先使用 `python scripts/nero/replay_agent_motion.py <历史 motions/id.json>
--output-dir <新目录>`。它读取当轮冻结配置，先以单次 PTP 回历史起点，再将原始
MotionPlan 送入正式控制进程，不重新求 IK 或改变关节分段；人工 hold 使用该目录
下的 `session.json`。它不恢复手部动作、抓布负载及当时的线程调度时序，因此相同
轨迹仍可能需要重复对照，单次成功不表示原问题已经修复。

夹爪位置到位监督复用 L20 底层 `set_thumb_index_positions_raw_and_wait`，在普通请求线程
阻塞轮询，不影响机械臂监督线程和独立人工 hold。默认每次等 2 秒、最多重试 2 次
（共 3 次发送），只比较拇指/食指七个有效槽位，容差 5，轮询间隔 0.1 秒；配置见顶层
`gripper`。超时抛出异常，重试只发送原目标；遥操作保留原非阻塞接口。
反馈仍受 SDK 缓存新鲜度限制，不能把到位检查当成逐帧 ACK 或抓布成功判定。

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

`--execute` 才允许连接配置中的两侧机械臂、手与共享相机；硬件需由操作者预先使能，程序不会自动使能。
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
