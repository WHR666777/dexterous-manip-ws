# 代码 review 导航

建议按以下顺序审阅：

1. `configs/quest3_nero_l20.yaml`：所有会改变现场行为的参数集中在这里；CAN 和标定默认不可执行。
2. `AnyDexRetarget/example/input/quest3.py`：确认它仍是唯一 TCP/CSV/Unity→RH 实现；集成改动只增加 `get_hand_frame()` 快照。
3. `teleop/coordinate_frames.py`：确认这里只应用现场 `R_base_quest`，没有第二次 Unity→RH 转换。
4. `teleop/wrist_tracker.py`：对照参考项目，检查相对锚定、平移 EMA、旋转残差乘法顺序。
5. `robot_control/nero_ik.py` 与 `assets/nero/nero_description.urdf`：确认 link7 是 SDK flange，并现场比较 FK。
6. `teleop/hand_retarget.py`：确认只从根目录 AnyDex 导入、qpos joint names 用于名称映射、保留槽为 255。
7. `teleop/target_gate.py` 与 `teleop/controller.py`：检查锁定条件、`R` 恢复、Nero→L20 非原子发送顺序及退出不自动 disable/e-stop。
8. `teleop/recording.py` 与 `docs/dataset_schema.md`：确认 OP 训练需要的 action/observation/timestamp 是否完整。

## 本次保留的既有逻辑

- `robot_control/nero.py`、`l20.py` 的 SDK 边界、输入验证与反馈接口。
- 既有 `NeroIK` 的 FK、限位、多初值局部求解、2 mm / 2° 解验收与 10° 关节跳变限制；只修正 URDF 来源。
- 根目录 AnyDex 的 Quest3 L20 adaptive 配置、名称感知 20 槽映射与 raw slew。
- 参考 `vr_teleop` 的腕部初始相对锚定、平移残差 EMA 和相对四元数目标构造。

## 有意未实现

- 双臂/双手、ROS、相机、策略训练与回放。
- 自动探测 CAN、自动计算现场外参、自动使能/失能 Nero、自动电子急停。
- 跨 Nero/L20 的原子发送或硬件时间同步。
- 为无法在本副本中验证的 Quest App/SDK 猜测新协议。

## 尚需现场确认

- Quest Streamer 继续使用已经实测通过的 AnyDex CSV/TCP 路径与 `adb reverse tcp:8000 tcp:8000`。
- Nero 与 L20 的真实 CAN 通道。
- `R_base_quest`、Nero flange/link7 定义与欧拉顺序的一致性。
- 20 Hz 下 `move_js`、L20 CAN 和反馈读取的实际延迟与稳定性。
- 数据字段对后续具体 OP 训练代码的命名/时间对齐要求。
