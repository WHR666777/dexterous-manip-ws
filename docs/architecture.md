# 架构与控制状态

## 模块边界

- `AnyDexRetarget/example/input/quest3.py`：唯一 Quest TCP listener、CSV parser 和 Unity 左手系到右手系转换；保留已验证的 AnyDex 输入路径，并额外公开线程安全的完整手帧快照。
- `teleop.coordinate_frames`：不重复转换 Quest 坐标，只把现场 `R_base_quest` 应用于 AnyDex 已转换的右手系腕姿。
- `teleop.wrist_tracker`：从参考 `vr_teleop` 迁移的相对锚定。第一次帧锁定 Quest 原点，平移残差使用 `alpha*current + (1-alpha)*previous`，旋转使用当前相对初始腕姿。
- `robot_control.nero_ik`：基于固定 Nero URDF 的七轴局部 IK。它已存在且可复用，本次没有重新实现一套 IK。
- `teleop.hand_retarget`：只包装根目录 `AnyDexRetarget`；输出 qpos 后调用其公开 `l20_qpos_to_can_slots` 和 `slew_l20_command`。
- `teleop.controller`：拥有唯一控制循环；`both` 按 Nero 后 L20 顺序发送，记录各自发送时间，因此不宣称跨设备原子同步。
- `teleop.recording`：保存一个 `.npz` 数据文件和同名 `.json` 元数据。

## 状态转换

```text
start → validate recorded joints → move_joints to recorded pose → wait in tolerance → paused
paused --R + fresh frame--> streaming
streaming --Quest timeout / target step violation / IK failure--> paused
streaming --B--> streaming + recording
recording --S/Q/exception--> save episode
any --Q/Ctrl+C/exception--> stop sends → disconnect (no auto disable/e-stop)
```

锁定是有意的：数据恢复后不会自动沿旧锚点继续运动，必须由操作者观察现场并按 `R`。

启动点到点运动显式覆盖遥操作的 10° 单周期关节增量限制，但仍受官方关节限位、低速百分比、到位容差和超时约束。进入 `move_js` 后立即恢复原单周期限制。运行中的 `R` 只重置 Quest/当前 flange 的相对锚点，不执行返回起始姿态的运动。

机械臂到达记录姿态后，控制器用同一时刻的 SDK flange/TCP 反馈计算固定的 flange→TCP 变换，并以该 TCP 为中心建立 Nero base 轴对齐立方体。每个候选 flange 目标都先换算为 TCP；任一轴超出半边长时目标门锁定暂停，当前目标不发送。重新按 `R` 只更新遥操作锚点，不改变工作区中心。

## 为什么不使用 RobotSystem.step

`RobotSystem.step` 面向归一化 policy action，并通过一般关节命令路径发送 Nero；当前遥操作需要 Nero 的连续 `move_js()` 和 L20 的完整 20 槽 raw action。因此控制器直接调用两个已验证 wrapper，并明确记录非原子发送顺序。

## 双臂扩展边界

当前低层模块没有把 `right` 固化在算法里，side 由配置传入。双臂阶段应创建两套 tracker、gate、IK、hand retargeter 与硬件对象，再由上层同步调度；不要在当前单臂控制器中添加大量左右分支。
