# Nero / L20 操作说明 Robot operations

本文在任务启动时进入 prompt；本次实际标定、限速和工作区见后附配置快照。
只有右臂 Nero + 右手 L20。硬件未连接时只允许计算与读取 disconnected state，不能假定动作已发生。

## 坐标与抓取 Coordinates and grasp

- 位置使用 m，角度及七轴关节使用 rad。`task` 是配置中 `task_from_base` 定义的固定坐标系。
- 运动位置始终指向 L20 **拇指、食指闭合接触点**这个虚拟 TCP；它固定在工具上，开手也不重定义。
- 主要进行近似 gripper 的 top-down 抓取。约定的初始抓取姿态：保留记录起始关节位姿的
  法兰位置和法兰 yaw，将 roll/pitch 调整为标定中的示教参考倾角，再乘固定工具变换得到 TCP。
- `top_down_pose(xyz, yaw)` 的旋转为 `Rz_task(yaw) @ R_initial_grasp`。
  **yaw 是相对该初始抓取姿态、绕 task +Z 的增量，单位 rad；yaw=0 保持初始抓取朝向。**
  不是法兰的绝对 RPY yaw，不是相对当前 cell 的姿态，也不要求 TCP X 对齐 task X。
  标定内部为计算工具固定旋转补入的 yaw=0 不作为实际起始航向。
- 工作区仅约束虚拟 TCP 的位置。J 在主机检查 IK、关节限位和工作区；L 在主机检查
  起点/终点工作区，内部 IK 与路径由固件处理。没有场景碰撞、桌面接触或力控能力。
  动作需根据图像和任务几何明确选择。

## 注入对象 API

Python globals 中预置 `robo`、`np`、`math`、`time`。
本次冻结配置含 `start_joint_positions`、`initial_grasp_tcp` 和 `reference_tcp_rotation`。
当 `startup_prepared=true` 时，宿主在启动 Codex 前已通过固定准备 cell 000 先调用
与 wrist-only 相同的 `open_hand()` 发送官方全手张开预设（包含中指、无名指、小指），
再以 PTP 回记录关节、以 PTP 到初始抓取姿态，并检查机械臂反馈。全手张开仅记录
SDK 返回，不保证五指已到位；正常任务的 `open_gripper()` 仍只操作拇指/食指。
每段 PTP 只发送一次最终关节目标，
启动第二段不要求 TCP 直线运动。不要重放该 cell。agent 仍应先读取真实状态并观察现场。
准备失败、超时或人工 hold 时不启动 agent，不自动切换抓取参考或继续旧轨迹。

| 调用 | 含义与结果 |
| --- | --- |
| `robo.state()` | dict：实测 joints、task 系 TCP 4×4、arm_status 固件反馈、手反馈、运动、fault、manual_hold_latched、hold_confirmed 等 |
| `obs = robo.observe()` | 固定 RGB-D 快照，含 id、rgb、depth_m、intrinsics、task_from_camera、timestamps、关联 state |
| `robo.show(obs, crop=None)` | 将 RGB 作为本 cell 返回的 MCP 图片；可选 `[u0,v0,u1,v1]` 原图坐标，右/下边界不包含 |
| `robo.pixel_to_task(obs, [u,v])` | 使用该快照同一像素深度反投影为 task XYZ；原图整数坐标，零/无效深度直接报错 |
| `robo.top_down_pose(xyz, yaw=0.0)` | 构造 task 系虚拟 TCP 的 4×4 位姿，不发动作 |
| `robo.move_J(target_tcp, wait=True)` | task 系虚拟 TCP 目标；内部一次 IK，再发送一次关节 PTP 目标 Single-target joint PTP |
| `robo.move_L(target_tcp, wait=True)` | task 系虚拟 TCP 目标转 base 法兰目标；原厂 `move_l` 执行，监督启动与 TCP 到位 Native linear motion |
| `robo.open_gripper()` / `robo.close_gripper()` | 拇指/食指到配置预设；阻塞检查位置，超时有限重发；其余三指不发位置命令 |
| `robo.gripper_replay_start()` | 到配置中的示范首帧预抓手型并等待位置反馈，用于食指贴合布边的视觉比对；不启动回放、不 reset |
| `robo.pause_and_hold(reason)` | 暂停轨迹、使用反馈作固定保持目标；确认后结束当前 cell，返回 status=paused |

`move_J` 和 `move_L` 都接收 task 系虚拟 TCP 位姿：4×4 SE(3)，或
`[x,y,z,roll,pitch,yaw]`（m/rad，scipy 外旋 xyz），不接收七轴关节角。
top-down 任务优先使用 `top_down_pose` 构造目标，避免把法兰姿态当成 TCP 姿态。
例如 `robo.move_J(robo.top_down_pose([x, y, z], yaw=0.0))`。
`move_J` 以当前实测关节为参考，把 TCP 目标转换成法兰目标并求一次 IK，
只下发一次最终关节目标，由固件平滑；不要求 agent 求 IK 或提供关节角。
PTP 整体关节变化上限为 2π rad，并检查限位、工作区与实际起点；中途 TCP 路径不保证直线或避碰。
IK 失败返回 `planning_failed`，不下发运动。关节目标只保留在宿主启动回位的内部入口。
`move_L` 不再逐点求主机 IK、不做 TOPPRA、不发送 J 分段；原厂 `move_l` 接收一个
base 系法兰 xyz-rpy 目标。固定朝向平移时，虚拟 TCP 也随法兰沿直线平移。
同时改变朝向时，工具偏移会影响虚拟 TCP 的中途路径，不保证该虚拟点严格沿直线；
top-down 抓取/搬运需要直线时保持 yaw 不变，调整朝向另选合适的撤高位置。
主机不会预先证明固件内部路径的工作区、关节轨迹或避碰。

**fold-31 搬运经验：同类 top-down 折布规划中，固定 `yaw=0`、TCP 高度约 `0.064 m`
向 task +X 水平搬运时，目标虚拟 TCP 的 task X 不应大于 `-0.47 m`（即 `x <= -0.47`，
越接近 0 越容易触及本次观察到的限位区域）。** fold-31 在 X≈`-0.552 m` 完成第一段 L，
第二段向 X≈`-0.404 m` 搬运时，在实测 X≈`-0.458 m`、J4≈`2.145 rad` 处停住，
固件报告 NO_SOLUTION，最终到位超时。因此先取 `-0.47 m` 为留有余量的经验规划边界；
它不是实测的安全极限，也不保证边界内的固件路径不会超限，仅适用于相近标定、朝向与高度。
若放置点需要 X>`-0.47 m`，应在接近边界前调整放置点或搬运姿态并重新评估；
不能仅靠拆短同一条直线、延长超时或直接改用 J 来认定限位风险已消除。

J/L 共用 `motion.joint_speed_percent`（默认 10%），百分比不是 rad/s，
配置中的主机规划 rad/s、rad/s² 上限不构成固件实际速度/加速度保证。

L 的执行逻辑来自实测脚本 `check_move_linear_zbx.py`：先等待 `motion_status=0`，
默认最多 10 s；发送后在 1 s 内检查新鲜 L 模式反馈是否出现 `motion_status=1`。
未确认启动时最多发送 3 次**同一目标**（含首次），不是换点抓取或重新规划。
次数耗尽后仍进入到位等待，因为很短的运动可能没有被采样到启动沿。
到位等待默认 30 s，由 `completion_timeout_s` 控制；人工 hold 可在任意阶段撤销动作权限。

## 反馈与 cell 闭环 Feedback

夹爪到位监督在 `robot_control/l20.py` 的 `set_thumb_index_positions_raw_and_wait` 中实现，
比较 `[0,5,10,15,1,6,16]` 七个有效槽位的原始位置误差，不比较保留位或其他三指。
参数读取 YAML 顶层 `gripper`：`timeout_s: 2.0` 为每次发送后的轮询等待期限，
`max_retries: 2` 不含首次发送（最多 3 次），`tolerance_raw: 5`，`poll_interval_s: 0.1`。
只重发相同目标，不加力、不修改目标或清故障；耗尽次数抛出异常，当前 cell 不再继续后续动作。
返回 `command_status=feedback_reached`、`reached`、`attempts`、`retries`、`max_error_raw`
及 `position_raw`；SDK 查询可能返回旧缓存，因此 `delivery_confirmed` 仍为 false，
到位仅表示本次查询返回值满足容差，不是逐帧 ACK 或抓布成功证明。
这是空载到位检查；接触物体可能阻止达到空载预设，不应把超时直接解释成发送丢包。
人工 hold/动作撤权会取消后续轮询和重试，已进入 SDK 的阻塞调用不能硬中断。

可恢复的操作失败不会锁存设备 `fault`：IK/目标/限位拒绝、手部位置未到位或反馈数据
不可用、相机缺帧/过期/采集失败、单次观测保存失败，均交回 agent 决定下一步。
运动执行中的目标拒绝或到位超时返回 `MotionResult(status="failed")`；观测和手部
操作失败以 cell 的 `status="error"`、`exception` 返回，持久变量保留。
失败前的物理动作不回滚。运动执行失败后本 cell 不再接收新动作；在下一 cell 读取真实状态、
必要时重新观察并调整目标，不需要清错或恢复接口。主机 `planning_failed` 未发送运动，
仍可在同一 cell 检查结果后调整目标。
人工 hold、失能、机械臂反馈失效、明确通信故障、未知控制程序异常和保持失败仍终止会话。
相机失效可以再次请求观测或报告需要人工处理；agent 没有重启相机/设备的接口。

运动默认返回 `MotionResult`，查看 `.status`、`.reached`、`.reason`、`.actual_joints`。
J 的 `.diagnostics` 含点号、目标/实测关节、逐轴误差和固件状态；L 包含阶段、发送次数、
启动是否确认、目标/实测 TCP、位置/角度误差、法兰命令及固件状态。失败后先停止并检查原因，
不直接重复失败动作；L 内部只在启动未确认时按上述次数重发。

固件 `arm_status=2 / NO_SOLUTION` 仅作为状态保留在 `state.arm_status` 和运动诊断中，
不据此锁存错误、取消运动、创建保持目标、清错或等待 NORMAL，也不阻止启动与发送新命令。
需要重新规划时读取当前真实状态、调整目标，直接调用 move_J/move_L；没有恢复接口。
主机 IK 返回 `planning_failed` 时没有下发运动，也可以直接调整目标重试。
是否到位仍按下面的新鲜模式/运动反馈与实测位置判断，不能仅凭 NO_SOLUTION 判成功或失败。
仍遵守一次只运行一条运动；需中断正在运行的动作时使用已有 pause_and_hold，在新 cell 重新规划。
人工 hold、通信和其他固件故障的终止流程不变。

`planning_failed` 不下发运动。J 的 `completed/reached=True` 要求发送后的新鲜 J 模式
`motion_status=0` 与关节偏差在 `joint_arrival_tolerance_rad`（默认 0.015 rad）以内同时满足。
`0` 且误差大时继续等待，兼容命令已发送、运动尚未启动的阶段；`1` 时继续监督。
短动作不强制采样到 `1`。到位误差超过 `joint_tolerance_rad`（默认 0.005 rad）时
记录 warning；超过 `completion_timeout_s` 仍未满足完成条件则 failed，并请求普通 J 原位保持。
J 不增加固定等待或终点停稳窗口；保持确认仍要求 0.2 s 稳定。
L 的完成要求：收到本次发送之后的 L 模式 `motion_status=0`，实测 TCP 位置/角度误差在
`linear_position_tolerance_m` / `linear_angle_tolerance_rad` 内，且相邻周期关节变化不超过
0.002 rad、连续稳定 0.2 s。默认阈值沿用实测脚本：**0.02 m（2 cm）/ 2°**，
不是毫米级抓取精度保证。完成后由固件保持，不额外发送 J 覆盖 L 目标。
L 空闲等待或到位超时返回 failed，并转为当前关节的普通 J 固定保持；保持未确认则任务结束。
暂停保持只捕获一次真实关节。反馈若仅在官方边界外不超过 0.001 rad，保持目标收至
边界内的整 mrad 值；例如 fold-30 的 J4：实测 2.147、上限 2.146755，保持目标为 2.146。
原始 state 反馈不改写，agent 的运动目标限位不放宽；明显反馈越限仍终止控制。
该 1 mrad 容差也用于实测规划起点和未下发的 PTP 插值检查点，使下一次合法回撤
不会仅因原始反馈轻微越限而被拒绝；实际下发的 PTP 终点仍单独严格检查限位。
普通保持需看到下发后的新鲜 J/idle 和稳定关节反馈；已完成的 L 继续使用 L/idle 监督。
不能把到位、SDK 返回、手指关节反馈解释为抓布成功。闭合并抬升后通过一次 `observe()` + `show()`
确认布角随手悬起；没有专门抓布判定接口。**确认夹布后，无异常的搬运、调整朝向、下降放置、
松手及撤离可在同一 cell 内连续执行；不因运动分段而例行观察，也不在放下前额外等待视觉反馈。**
每段运动仍检查到位结果，运动失败则停止后续动作；仅在异常或需要重新判断现场时补充观察。
最终松手撤离后再观察一次，用于判断对折结果。

`wait=False` 返回当前 cell 的 `MotionHandle`：`.status()` 返回 dict，`.done()` 返回 bool，
`.wait()` 返回 `MotionResult`。规划失败即使异步请求也直接返回 `MotionResult`；可用
`hasattr(result, "wait")` 区分。异步运动必须在当前 cell 内监督并等待，或者调用暂停。
旧句柄不可跨 cell 使用；cell 返回时遗留运动会被宿主暂停，不能在两次模型决策之间继续推进。
一次只允许一条运动，不能在它执行中提交下一条。

`pause_and_hold()` 成功后展开当前 cell 的调用栈，之后的语句不再执行；不要用
`except BaseException` 吞掉暂停。保持确认后新 cell 可恢复，但必须重读状态/观测并重新规划。
普通异常不会回滚 globals 或物理动作；worker 超时/退出、控制故障和人工停止结束本次任务。

`observe()` 返回最新完整帧，可能与上次是同一帧，查看 frame_number/时间戳。手部 SDK 没有
可靠 generation：`hand.freshness="unknown"`，`read_at` 是本机读取时间，不能证明新反馈。
相机和机器人时间分别记录，不声称硬件同步。显示裁剪图后，反投影仍需换回原图坐标。

人工 `hold` 直接到控制进程：锁定动作、捕获一次实测关节，发送一次普通 J 固定位置目标，之后持续监督反馈。
不调用 Nero 的阻尼急停；阻尼急停已观察到下落，不能用于空中保持。agent 不负责恢复，
不清锁、不 reset、不使能、不续播旧动作，不自动松开 L20。
任务结束后控制进程仍保持运行，只有操作者确认外部支撑后才能 release-hold；
agent 不得关闭保持进程。`hold_target_sent` 只表示调用返回，`hold_confirmed` 只是
关节反馈稳定证据；保持监督进程不反复重抓目标或连续重发 J。`stop_confirmed` 仍为 false，不保证断电/断流/设备故障时悬停。

任务开始时先用一个 cell 获取并 show 现场图像；收到初始图像反馈后再选点执行。
需要视觉判断时，`show()` 所在 cell 返回后才能看到图片，不要求每个动作或分段都拍照。
任务结束调用 MCP `finish`，summary 说明实际可见证据。
