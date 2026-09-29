# OP episode 数据格式

新录制的每个 episode 使用独立目录（`schema_version: 2`）；已有的 `.npz + .json` 文件不迁移：

```text
data/episodes/
└── episode_<开始录制的UTC>/
    ├── trajectory.npz
    ├── metadata.json
    └── camera/              # camera.enabled=true 时存在
        ├── color.mp4
        └── depth.h5
```

`trajectory.npz` 按有效控制周期堆叠数值数组。`metadata.json` 保存配置路径、控制模式、手侧、名义采样频率、样本数、字段 shape，以及相机实际型号、序列号、内参、深度单位和图像文件路径。`B` 开始，`S` 保存；`Q`/正常异常收尾保存仍在进行的 episode。重复 `B` 不丢弃当前记录；空 episode 的 `samples` 为 0。

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
| `camera_frame_index` | scalar | 零起始的存储帧索引；RGB 视频与 Depth 共用；缺失为 -1 |
| `camera_valid` | scalar | 当前行是否存在可引用的 RGB-D |
| `camera_timestamp_monotonic` | scalar | 主机收到相机帧组时的单调时钟，秒；对齐处理之前记录 |
| `camera_color_timestamp_ms` | scalar | SDK 彩色帧时间戳，毫秒 |
| `camera_depth_timestamp_ms` | scalar | SDK 深度帧时间戳，毫秒 |
| `camera_color_frame_number` | scalar | 相机原生彩色帧号 |
| `camera_depth_frame_number` | scalar | 相机原生深度帧号 |
| `camera_color_timestamp_domain` | scalar string | SDK 彩色帧时钟域 |
| `camera_depth_timestamp_domain` | scalar string | SDK 深度帧时钟域 |

发送时间不代表设备真正执行时间；L20 SDK 也不提供可靠的反馈 generation。训练前应按字段掩码约定处理 NaN/-1，不要把 arm-only 或 hand-only 的占位值当观测。

`camera_*` 字段仅在启用相机时存在。相机帧在计算/发送动作之前选择，机器人反馈仍在发送之后读取，原有字段语义不变。相机缺失或超过 `camera.max_age_s` 时，该行不写图像，索引/原生帧号为 -1、时间戳为 NaN、时钟域为空字符串。若设备重复提供同一原生帧，不伪造新设备帧号；存储仍按每条有效视觉样本追加一组图像。

## 图像格式和时间对应

- `camera/color.mp4`：mp4v 有损彩色视频，默认 640×480，保存名义帧率等于 `control_hz`。相机封装提供 RGB8，送入 OpenCV 编码前转换为 BGR；使用 OpenCV 解码时返回 BGR，训练需要 RGB 时应转换。
- `camera/depth.h5`：数据集 `depth`，shape 为 `(K, H, W)`，dtype 为 `uint16`，逐帧 chunk、LZF 无损压缩。深度已对齐到彩色相机网格，米数为 `depth_raw * metadata["camera"]["depth_scale_m"]`，原始 0 表示无有效深度。
- `K` 为实际保存的 RGB-D 帧组数，可能小于轨迹行数 `N`。第 `i` 行先检查 `camera_valid[i]`，再用 `k = camera_frame_index[i]` 读取视频第 `k` 帧和 `depth[k]`。禁止将 -1 当作最后一帧读取。

MP4 的播放时间不作为真实采集时间：暂停、循环超时、缺失帧会造成不等时间间隔，训练应使用 NPZ 时间戳。L515 原生 30 FPS，录制取帧跟随既有控制循环，默认目标 20 Hz；不保证控制循环始终达到名义频率。SDK 时间戳保留其时钟域，不直接与 PC 单调时钟相减。RGB-D 对齐是空间对齐，不等于与机器人曝光/执行时刻硬同步。

相机原生 RGB 为 960×540，深度为 640×480；深度先对齐到彩色，随后共同中心裁剪、缩放成配置的保存尺寸（默认 640×480）。彩色使用 area，深度使用 nearest-exact，不对距离值做线性插值。相机元数据包含适用于输出图像的 `color_intrinsics`、`aligned_depth_intrinsics`，以及 `native_color_intrinsics`、`native_depth_intrinsics`、`color_crop_xywh`、`resize`、`depth_scale_m` 和 `depth_aligned_to: color`。各组内参包含 fx/fy/ppx/ppy、畸变模型和系数；裁剪缩放使用像素中心映射 `(p - crop_origin + 0.5) * scale - 0.5` 换算主点，焦距乘以 scale。仓库没有相机相对机械臂的外参，不能据此直接把深度点云转换到 Nero base；需要三维机器人坐标时再做现场标定。
