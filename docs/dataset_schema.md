# OP episode 数据格式

每个 episode 由同名文件组成：

- `episode_<UTC>.npz`：按控制循环堆叠的数值数组。
- `episode_<UTC>.json`：配置路径、控制模式、手侧、样本数和每个字段的最终 shape。

## NPZ 字段

| 字段 | 单样本 shape | 含义 |
| --- | --- | --- |
| `timestamp_monotonic` | scalar | PC 单调时钟，秒 |
| `quest_wrist` | `(7,)` | AnyDex Quest plugin 转换后的 RH `xyz + xyzw` |
| `quest_landmarks` | `(21,3)` | AnyDex Quest plugin 输出的 world landmarks |
| `quest_received_at` | scalar | 完整帧形成时的单调时钟 |
| `target_base_flange` | `(4,4)` | Nero base 到目标 flange；hand-only 时为 NaN |
| `arm_action` | `(7,)` | 发送的 Nero joint target，rad；hand-only 为 NaN |
| `hand_action_raw` | `(20,)` | 发送的 L20 raw target；arm-only 为 -1 |
| `arm_joint_position` | `(7,)` | Nero 反馈，rad |
| `arm_joint_torque` | `(7,)` | Nero 反馈，N·m |
| `arm_tcp_pose` | `(6,)` | `[m,m,m,rad,rad,rad]` |
| `hand_position_raw` | `(20,)` | L20 缓存反馈；含 11–14 保留槽 |
| `arm_sent_at` | scalar | Nero 调用返回时的单调时钟 |
| `hand_sent_at` | scalar | L20 调用返回时的单调时钟 |

发送时间不代表设备真正执行时间；L20 SDK 也不提供可靠的反馈 generation。训练前应按字段掩码约定处理 NaN/-1，不要把 arm-only 或 hand-only 的占位值当观测。
