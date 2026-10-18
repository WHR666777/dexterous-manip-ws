# 架构与控制状态

## 模块边界

- `AnyDexRetarget/example/input/quest3.py`：唯一 Quest TCP listener、CSV parser 和 Unity 左手系到右手系转换；保留已验证的 AnyDex 输入路径，并额外公开线程安全的完整手帧快照。
- `teleop.coordinate_frames`：不重复转换 Quest 坐标，只把现场 `R_base_quest` 应用于 AnyDex 已转换的右手系腕姿。
- `teleop.wrist_tracker`：从参考 `vr_teleop` 迁移的相对锚定。第一次帧锁定 Quest 原点，平移残差使用 `alpha*current + (1-alpha)*previous`，旋转使用当前相对初始腕姿。
- `robot_control.nero_ik`：基于固定 Nero URDF 的七轴局部 IK。Pinocchio 计算 FK 和解析 Jacobian，SciPy `least_squares` 保留原有目标函数、关节限位、连续性检查、误差验收与超时回退。模型只加载一次，不加载 mesh 或连接硬件；[离线轨迹对比](ik_benchmark.md)记录验证方法与结果。
- `teleop.hand_retarget`：只包装根目录 `AnyDexRetarget`；输出 qpos 后调用其公开 `l20_qpos_to_can_slots` 和 `slew_l20_command`。
- `teleop.controller`：拥有唯一控制循环；`both` 按 Nero 后 L20 顺序发送，记录各自发送时间，因此不宣称跨设备原子同步。
- `teleop.realsense_camera`：后台读取 L515 的 RGB8/Z16，将深度对齐到彩色像素网格，发布带时间戳的完整帧快照；不参与机械臂/手控制。
- `teleop.recording`：保持原录制接口和有界后台队列；联合采集暂存 RGB PNG、无损 uint16 HDF5 和轨迹，保存时调用 act 环境的 `teleop.lerobot_export`，由官方 API 生成 LeRobot v3.0。单设备调试保留原诊断格式。

## 状态转换

```text
start → validate recorded joints → move_joints to recorded pose → wait in tolerance → paused
paused --R + fresh frame--> streaming
streaming --Quest timeout / target step violation / IK failure--> paused
streaming --B--> streaming + recording
recording --S/Q/exception--> save episode
any --Q/Ctrl+C/exception--> stop sends → disconnect (no auto disable/e-stop)
```

## RGB-D 采样与录制

L515 以硬件支持的 30 FPS 采集；控制周期仍由现有 `control_hz` 决定（默认 20 Hz）。每个有效控制周期在 IK/动作发送前取最近 RGB-D；联合录制额外读取已有手部反馈缓存，并复用 IK seed 作为发送前机械臂状态，不新增 CAN 查询。NERO→L20 的发送顺序、IK、限位、原发送后诊断反馈和休眠规则保持不变。

相机原生配置、深度到彩色的空间对齐、中心裁剪和缩放均沿用原实现，输出 640×480。控制线程只向固定 16 帧队列入队；联合采集后台写 PNG 和 HDF5，编码及 LeRobot 写入在按 S 后进行。按 Q 或异常收尾也保存已收集的有效周期。

独立保存进程使用现有 act 环境；不把硬件依赖装入 act，也不把 Torch 装入遥操作环境。每次成功保存都 finalize 并用官方加载器重新打开、解码验证。深度、完整 20 槽诊断和原始时钟在附件中保留，训练只读取 RGB＋23 维状态/动作。缺帧、写入错误或既有数据集接口不兼容时拒绝导出，保留暂存文件。详见 [数据格式](dataset_schema.md)。


锁定是有意的：数据恢复后不会自动沿旧锚点继续运动，必须由操作者观察现场并按 `R`。

启动点到点运动显式覆盖遥操作的 10° 单周期关节增量限制，但仍受官方关节限位、低速百分比、到位容差和超时约束。进入 `move_js` 后立即恢复原单周期限制。运行中的 `R` 使用当前七轴反馈的 URDF/FK flange 重置 Quest 相对锚点，不执行返回起始姿态的运动；SDK flange 只作为 preflight 的独立一致性观测，避免两套近似模型混入同一次 IK。

机械臂到达记录姿态后，控制器用同一时刻的 SDK flange/TCP 反馈计算固定的 flange→TCP 变换，再将该变换应用到关节反馈的 URDF/FK flange，得到与 IK 模型一致的起始 TCP，并以它为中心建立 Nero base 轴对齐立方体。每个候选 flange 目标都先换算为 TCP；任一轴超出半边长时目标门锁定暂停，当前目标不发送。重新按 `R` 只更新遥操作锚点，不改变工作区中心。

## 为什么不使用 RobotSystem.step

`RobotSystem.step` 面向归一化 policy action，并通过一般关节命令路径发送 Nero；当前遥操作需要 Nero 的连续 `move_js()` 和 L20 的完整 20 槽 raw action。因此控制器直接调用两个已验证 wrapper，并明确记录非原子发送顺序。

## 双臂扩展边界

当前低层模块没有把 `right` 固化在算法里，side 由配置传入。双臂阶段应创建两套 tracker、gate、IK、hand retargeter 与硬件对象，再由上层同步调度；不要在当前单臂控制器中添加大量左右分支。
