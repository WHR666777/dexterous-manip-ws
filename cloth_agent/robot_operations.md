# Nero / L20 操作说明 Robot operations

本文在任务启动时进入 prompt；本次实际标定、限速和工作区见后附配置快照。
只有右臂 Nero + 右手 L20。硬件未连接时只允许计算与读取 disconnected state，不能假定动作已发生。

## 坐标与抓取 Coordinates and grasp

- 位置使用 m，角度及七轴关节使用 rad。`task` 是配置中 `task_from_base` 定义的固定坐标系。
- 运动位置始终指向 L20 **拇指、食指闭合接触点**这个虚拟 TCP；它固定在工具上，开手也不重定义。
- 主要进行近似 gripper 的 top-down 抓取。参考姿态来自标定，`yaw=0` 表示该参考姿态；
  `top_down_pose(xyz, yaw)` 的旋转为 `Rz_task(yaw) @ R_reference`。yaw 是绕 task +Z 的增量，
  不是法兰的绝对 RPY yaw，也不假设标定参考的 TCP X 恰好对齐 task X。
- 工作区仅约束虚拟 TCP 的位置。当前规划含关节限位、IK、路径工作区与速度/加速度检查，
  没有场景碰撞、桌面接触或力控能力。动作需根据图像和任务几何明确选择。

## 注入对象 API

Python globals 中预置 `robo`、`np`、`math`、`time`。

| 调用 | 含义与结果 |
| --- | --- |
| `robo.state()` | dict：实测 joints、task 系 TCP 4×4、手反馈、运动、fault、manual_hold_latched、hold_confirmed 等 |
| `obs = robo.observe()` | 固定 RGB-D 快照，含 id、rgb、depth_m、intrinsics、task_from_camera、timestamps、关联 state |
| `robo.show(obs, crop=None)` | 将 RGB 作为本 cell 返回的 MCP 图片；可选 `[u0,v0,u1,v1]` 原图坐标，右/下边界不包含 |
| `robo.pixel_to_task(obs, [u,v])` | 使用该快照同一像素深度反投影为 task XYZ；原图整数坐标，零/无效深度直接报错 |
| `robo.top_down_pose(xyz, yaw=0.0)` | 构造 task 系虚拟 TCP 的 4×4 位姿，不发动作 |
| `robo.move_J(target_joints, wait=True)` | 七轴关节目标；内部插值和 TOPPRA；默认等待到位结果 |
| `robo.move_L(target_tcp, wait=True)` | 虚拟 TCP 直线位置插值及姿态插值，内部逐点 IK/TOPPRA；不自动降级关节运动 |
| `robo.open_gripper()` / `robo.close_gripper()` | 只发送拇指/食指的配置预设；其余三指不发位置命令 |
| `robo.pause_and_hold(reason)` | 暂停轨迹、使用反馈作固定保持目标；确认后结束当前 cell，返回 status=paused |

`move_L` 的目标可用 4×4 SE(3)，或 `[x,y,z,roll,pitch,yaw]`（scipy 外旋 xyz）。
top-down 任务优先使用 `top_down_pose` 构造目标，避免把法兰姿态当成 TCP 姿态。
它是采样 IK 和样条轨迹实现的近似直线路径，不承诺连续空间的精确直线或避碰。

## 反馈与 cell 闭环 Feedback

运动默认返回 `MotionResult`，查看 `.status`、`.reached`、`.reason`、`.actual_joints`。
`planning_failed` 不下发运动；`completed/reached=True` 只表示关节反馈满足到位容差。
不能把到位、SDK 返回、手指关节反馈解释为抓布成功。第一版通过再次 `observe()` + `show()`，
让模型查看实际图像判断抓布效果；没有专门抓布判定接口。

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

人工 `hold` 直接到控制进程：锁定动作、捕获一次实测关节，持续下发固定 JS 位置目标。
不调用 Nero 的阻尼急停；阻尼急停已观察到下落，不能用于空中保持。agent 不负责恢复，
不清锁、不 reset、不使能、不续播旧动作，不自动松开 L20。
任务结束后控制进程仍保持运行，只有操作者确认外部支撑后才能 release-hold；
agent 不得关闭保持进程。`hold_target_sent` 只表示调用返回，`hold_confirmed` 只是
关节反馈稳定证据；`stop_confirmed` 仍为 false，不保证断电/断流/设备故障时悬停。

先用一个 cell 获取并 show 图像；收到图像反馈后，再写后续动作 cell。`show()` 并不会在 cell
执行中把图片实时送给模型。任务结束调用 MCP `finish`，summary 说明实际可见证据。
