# NERO＋L20 LeRobot 数据格式

正式联合示教使用固定的 LeRobot 0.4.4 官方 API，生成 **LeRobotDataset v3.0**。旧 `data/episodes` 不迁移、不修改。遥操作仍在原环境运行，仅保存阶段调用 `act`；无需安装或调整两套环境的依赖。

## 双侧布局

`configs/pico4_bimanual_nero_l20.yaml` 使用独立目录 `data/lerobot/nero_l20_bimanual_act` 和 robot_type `nero_l20_bimanual`。`observation.state`、`action` 均为 float32 `(46,)`，固定顺序为 **左 Nero 7＋左 L20 16＋右 Nero 7＋右 L20 16**。名称分别加 `left.`、`right.` 前缀，例如 `left.nero.joint1`、`right.l20.thumb_base`；各侧 L20 仍使用索引 `0–10、15–19`，单位与单侧相同。

主进程只将同一个 `cycle_id`、双侧均完成发送的结果作为一行追加。状态是各侧准备阶段、所有本轮发送之前读取的真实反馈；RGB-D 在本轮准备之前选择，每行只保存一份共享图像和深度。并行调度不等于硬件同步，左右观测和各设备发送时刻分别保留。

双侧诊断使用 `schema_version: 3`、`input_sides: [left, right]`；原单侧字段加 `left_` 或 `right_` 前缀。公共字段为 `cycle_id`、`observation_timestamp_monotonic`（主进程开始选择观测）、`timestamp_monotonic`（双方结果收齐）以及 `camera_*`。左右的 `*_observation_timestamp_monotonic` 为各自开始读取状态的时间。`timing_prepare_wait_s`、`timing_dispatch_wait_s` 记录协调进程等待准备和发送结果的耗时；各侧另存 `*_timing_prepare_s`、`*_timing_dispatch_s` 和原分阶段耗时。

暂停、准备失败或部分发送的周期不追加；不拿旧侧数据与新侧结果拼接。任一侧计算失败后两侧都需要 R 重锚定。采样间隔和 `actual_hz` 仍使用真实主机时间计算，不把名义 20 Hz 当作实测频率。

相机、深度附件、导出失败保留暂存和官方重载验证均沿用下文规则。23/46 维的形状、通道名称、手侧顺序和 robot_type 必须与既有数据集一致，混合追加会被拒绝。`scripts/check_lerobot_act.py` 根据数据集名称列表检查 23 或 46 维，双侧动作块输出为 `(1,60,46)`。

## 配置与采集

Pico、Quest 配置均已开启 L515，输出空间对齐的 640×480 RGB-D。相机仍为 30 FPS，录制跟随原 `control_hz: 20` 的有效周期。

```yaml
recording:
  output_dir: data/lerobot/nero_l20_act
  lerobot:
    python: /home/lab404/anaconda3/envs/act/bin/python
    repo_id: local/nero_l20_act
    task: "NERO and L20 manipulation demonstration"
```

采集前把 `task` 改为实际任务描述。`repo_id` 是本地标识，不会上传。同一目录支持连续记录和跨进程追加；版本、频率、机器人类型、通道顺序或手侧不一致时拒绝追加。

在项目根目录、原遥操作环境运行：

```bash
python3 -m teleop.cli --config configs/pico4_nero_l20.yaml run --control both --execute
```

R 重锚定、B 开始、S 保存、Q 保存并退出保持原操作。重复 B 不清空数据；暂停时不增加样本；空录制不生成正式 episode。S 等待后台暂存完成，再调用 act 的官方写入器，完成并重新加载验证后返回。

单设备 `--control arm/hand` 的诊断录制继续使用旧格式，位置为输出目录旁的 `nero_l20_act_diagnostics/`，不混入联合训练集。没有 `recording.lerobot` 的旧配置也保留原诊断录制接口。

## 官方训练接口

| 字段 | dtype／单样本 shape | 含义 |
| --- | --- | --- |
| `observation.state` | float32 `(23,)` | 发送前实际反馈：NERO 7＋L20 16 |
| `action` | float32 `(23,)` | 本轮传入发送函数的位置目标：NERO 7＋L20 16 |
| `observation.images.front` | video `(480,640,3)` | RGB，H.264；加载后为 `(3,480,640)` float32 |

前七维为 `nero.joint1` 至 `nero.joint7`，单位 rad。后十六维按现有 `L20_ACTIVE_POSITION_INDICES`，即零起始 SDK 索引 `0–10、15–19`；排除保留槽 `11–14`。L20 保持 `0–255`，保存时不归一化；ACT 使用官方数据集统计归一化。状态和动作在 `meta/info.json` 中使用相同的 names 顺序。

NERO 状态复用本周期 IK 前已读取的 seed；L20 状态取发送前已有后台反馈缓存，不新增硬件查询。原发送后 `arm_joint_position`、`hand_position_raw` 继续作为诊断，不能替代发送前状态。动作来自原重定向、限制和 slew 之后的目标。发送调用时间不代表实际执行时刻或执行成功证明。

索引、任务、边界和统计由官方生成。`timestamp = frame_index / fps` 是名义采样索引时间，不是真实曝光或执行时间；真实时钟另存于诊断，没有重采样或改变控制频率。未来动作块和 `action_is_pad` 由加载器按 episode 边界生成。

## 文件结构与深度对应

```text
data/lerobot/nero_l20_act/
├── data/chunk-000/file-000.parquet
├── videos/observation.images.front/chunk-000/file-000.mp4
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── episodes/chunk-000/file-000.parquet
│   └── recording.json
├── depth/episode-000000.h5
└── diagnostics/
    ├── episode-000000.npz
    └── episode-000000.json
```

官方负责分片；不要假定一个 MP4/Parquet 文件只有一个 episode。

`depth/episode-E.h5` 的数据集 `depth` 为 `(N,480,640)` uint16、逐帧 LZF 无损压缩。episode E 的 `frame_index=i` 对应 `depth[i]`，与该训练样本的 RGB 对齐。深度原值乘诊断 JSON 的 `camera.depth_scale_m` 得到米；0 表示无效深度，65535 等原值完整保留。深度不属于本次 ACT 输入，不走 RGB 有损编码。

诊断 JSON 保留型号、序列号、输出及原生内参、裁剪、缩放和深度单位。相机实现沿用原方案：原生 RGB 960×540、深度 640×480，先对齐至彩色，再共同中心裁剪/缩放；RGB 用 area，深度用 nearest-exact。空间对齐不等于机器人与相机硬件同步。

诊断 NPZ 保留原有腕姿、landmarks、目标变换、力矩、TCP、完整 20 槽动作/反馈、`timing_*_s` 和 `camera_*` 字段；新增：

| 字段 | 含义 |
| --- | --- |
| `observation_timestamp_monotonic` | 开始选择本轮观测的主机单调时钟 |
| `observation_arm_joint_position` | IK seed 独立副本，7 维 rad |
| `observation_hand_position_raw` | 发送前手部缓存快照，20 槽 |
| `observation_hand_feedback_query_started_at/finished_at/sequence` | 上述快照的后台查询时间及序号 |

JSON 的 `actual_hz` 和 `sample_interval_s` 按真实观测时间计算，能识别控制变慢和停顿。查询序号不是硬件反馈 generation。原生相机帧号可以重复，不伪造帧号。

## 保存失败

联合录制暂存在输出目录旁的 `.nero_l20_act-staging/episode_<UTC>/`，包含 RGB PNG、深度 HDF5、数值轨迹及元数据。RGB 只在官方保存阶段编码一次。

缺图、非法关节值、队列满或写盘失败时不填零、不删行冒充完整 episode。保留暂存源和 `export.log`；成功并重新加载验证后清理暂存。若官方写入中断，根目录保留 `.recording-in-progress.json`，后续追加会拒绝继续；应先按标记的源目录检查或恢复，不能直接删除标记后盲目重试。

## 离线 ACT 检查与训练

在 act 环境、项目根目录执行，沿用交接文档的系统 libffi 修复：

```bash
conda activate act
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
python scripts/check_lerobot_act.py data/lerobot/nero_l20_act
```

该检查只读取本地数据，不连接机器人、不下载权重；用实际 640×480 RGB、23 维状态和 60 步动作块，检查前向、反向、梯度和 `(1,60,23)` 输出。可加 `--device cpu/cuda`。

最小训练命令（按资源调整训练参数）：

```bash
lerobot-train \
  --dataset.repo_id=local/nero_l20_act \
  --dataset.root="$PWD/data/lerobot/nero_l20_act" \
  --policy.type=act \
  --policy.chunk_size=60 \
  --policy.n_action_steps=20 \
  --policy.pretrained_backbone_weights=null \
  --policy.push_to_hub=false \
  --output_dir=outputs/nero_l20_act
```

旧 `scripts/replay_episode.py` 仍只支持旧诊断格式；本次未增加新格式实机回放或 ACT 实机推理。
