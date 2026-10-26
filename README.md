# Pico 4 / Quest 3 → Nero + LinkerHand L20 遥操作

本仓库的遥操作部分用 Pico 4 或 Quest 3 控制 AgileX Nero 七轴机械臂和 LinkerHand L20，并记录后续 OP 训练所需的数据。默认仍是**单臂 + 单手**；Pico 4 另支持显式配置的**双臂 + 双手并行控制及联合录制**。折布 agent 使用 `codex exec + MCP + 独立持久 Python worker/控制进程`；已接入右侧 robo、设备控制和观测代码，目前仅检查语法，见 [框架阅读说明](docs/cloth_framework.md)。

`AnyDexRetarget/` 是唯一有效的 VR 输入和手部重定向实现。原 `vr_teleop` 只作为腕部相对锚定和平滑方案的参考，所需跟踪逻辑已经迁入 `teleop/`，不再作为运行依赖。

## Pico 4 输入（当前默认）

完整启动步骤见 [notes-zyd.md](notes-zyd.md)。头显打开 robo、勾选手部追踪并 reconnect；USB 连接时执行 `adb reverse tcp:63901 tcp:63901`。先在一个终端保持运行原 AnyDex 转发：

```bash
python3 AnyDexRetarget/example/input/pico4_daemon.py
```

在另一个终端检查输入，再运行联合遥操：

```bash
python3 -m teleop.cli --config configs/pico4_nero_l20.yaml input-check
python3 -m teleop.cli --config configs/pico4_nero_l20.yaml run --control both --hand-mode tracking --execute --record-dir data/lerobot/nero_l20_act-3
```

Pico 使用 TCP 63901 → 常驻 daemon → localhost 63902，复用 AnyDex 的 26→21 关节映射和 Pico/L20 adaptive 配置。腕部和手指来自同一条 JSON；腕部位置/旋转转换为 RH x前、y左、z上，手指保留原 Pico 重定向坐标。PXR 原生腕姿格式和坐标依据 [XRoboToolkit 发送实现](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/main/Assets/Scripts/TrackingData.cs) 与 [PXR SDK 关节读取实现](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/main/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/Features/PXR_HandTracking.cs)。

新配置保留原机械臂/手的 CAN、控制频率、相对锚定、平滑、IK、运动限制、相机与 R/B/S/Q 流程。`calibration.R_base_pico4` 以原 Quest 矩阵为初值，现场确认方向后将 `calibration.calibrated` 改为 `true` 才能运行 `arm/both`。`input-check` 和 `hand` 模式不要求腕部到机械臂的标定。无需运行 `quest-listener`；需要直连调试时，将 `pico4.input_mode` 改为 `direct` 并停止 daemon。

以下保留原 Quest 配置和操作说明；使用 Pico 执行同一硬件命令时，把配置路径换为 `configs/pico4_nero_l20.yaml`，手侧字段使用 `pico4.side`，标定矩阵使用 `R_base_pico4`。

## Pico 4 双侧遥操作

使用独立配置 `configs/pico4_bimanual_nero_l20.yaml`；原配置和默认命令继续控制单侧。`sides.left/right` 各自定义 `arm`、`hand`、`calibration`，其余参数共用。左右 Nero base 已确认同向，配置使用相同旋转矩阵，但起始关节姿态分别记录。

| 侧别 | Nero CAN | L20 CAN | 起始姿态 |
| --- | --- | --- | --- |
| 左 | `can_arm_b` | `can_hand_b` | `configs/nero_left_start_pose.yaml` |
| 右 | `can_arm_a` | `can_hand_a` | `configs/nero_start_pose.yaml` |

先配置并启用四条 CAN（`scripts/nero/can_up_all.sh` 已包含四条接口），保持原 Pico daemon 运行。将左臂放到所需起始位置后，只读记录它的姿态，再检查双侧输入及设备：

```bash
python3 -m teleop.cli --config configs/pico4_bimanual_nero_l20.yaml --side left record-start-pose
python3 -m teleop.cli --config configs/pico4_bimanual_nero_l20.yaml input-check
python3 -m teleop.cli --config configs/pico4_bimanual_nero_l20.yaml preflight --control both
python3 -m teleop.cli --config configs/pico4_bimanual_nero_l20.yaml run --control both --execute
```

两侧都到达各自起始姿态后，R 同时重锚定并恢复，B 开始联合录制，S 保存，Q 保存并退出。普通 IK 求解失败（`IK_FAILED`、`IK_FK_ERROR`、`IK_JUMP`）显示 `IK HOLD`：双方跳过这一轮发送和录制，继续尝试新鲜目标；把手移回最近可达姿态附近，通过 IK 和目标门后自动继续，也可按 R 在机械臂当前位置重新锚定，无需退出或回起始位。IK 目标默认距 URDF 上下限各保留 2° 缓冲；反馈越过 URDF 边界不超过 1° 时，仍允许求向内的有效解，避免连回退目标都被 `IK_JOINT_LIMIT` 拒绝。任一侧输入失效、工作区或目标步长越界、反馈越限超过恢复容差、数值异常仍会锁定暂停。使能、重置及 preflight 也接受全局 `--side left|right|both`；`record-start-pose` 必须选择单侧。

主进程独占 Pico 输入、键盘和相机，两个 `spawn` 控制子进程分别持有左/右侧的反馈线程和 CAN 对象；每侧再使用一个只做 IK 和 retarget 的计算进程，将求解与硬件 I/O 线程隔离。每轮左右并行计算，双方通过后才通知并行下发；失败或过期的一轮不会继续下发。每个计算进程最多处理一个请求，超时、通信异常或进程退出时停止整套控制并保存已有完整样本。CAN 调用不具备跨设备原子性，发送阶段异常可能留下部分已发出的命令，此时整套控制退出。

`control_hz: 20` 是调度目标，实际取决于较慢一侧的计算与发送耗时。输出 `[left control timing]`、`[right control timing]` 和 `[bimanual timing]`，报告实测频率、各阶段耗时及超周期次数；并行本身不保证达到 20 Hz。

联合录制保存在 `data/lerobot/nero_l20_bimanual_act/`，状态/动作顺序为左臂 7＋左手 16＋右臂 7＋右手 16，共 46 维，每个完整双侧周期对应一帧共享 RGB-D；格式见 [数据格式](docs/dataset_schema.md)。`--control arm/hand` 仅保存诊断。单独选择 `--side left/right` 时复用原单侧控制及 23 维录制，默认输出目录和 repo_id 加对应侧后缀。双侧目前支持 `tracking`，原单侧 `wrist-only` 入口保留。

## 数据流

```text
Quest 3 Hand Tracking Streamer (ADB reverse + TCP 8000)
  ├─ wrist xyz + quaternion
  │    → AnyDex Unity→RH → 现场 R_base_quest
  │    → 相对锚定 + 平移 EMA → Nero flange target
  │    → NeroIK → move_js
  └─ 21×3 hand landmarks
       → root AnyDexRetarget → L20 20-slot raw command
       → raw slew limit → set_joint_positions_raw
```

两个设备共享同一个完整 Quest 帧。腕部与 landmarks 的接收时间差超过配置值，或完整帧过期时，不发送旧目标；控制进入锁定状态，必须按 `R` 重新锚定。

## 安装

实验机需要 Linux、Python 3.10+、已安装的 Nero `pyAgxArm`、LinkerHand Python SDK，以及配置好的 SocketCAN：

```bash
python3 -m pip install -e .
```

若 LinkerHand SDK 未安装为 Python 包，将固定副本放在根目录 `linkerhand-python-sdk/`。本仓库不保存 sudo 密码，也不替操作者创建或激活 CAN 接口。

默认配置启用 RealSense L515 的 RGB-D 录制，需要额外安装视觉依赖：

```bash
python3 -m pip install -e '.[vision]'
```

L515 的官方兼容表列出 SDK 2.50.0 已验证、2.54.2 支持但未验证，因此视觉依赖限制为 `pyrealsense2>=2.50,<2.55`；不要直接升级为不支持 L515 的新版本。[RealSense 兼容表](https://github.com/realsenseai/librealsense/releases)

## 现场必须填写

编辑 `configs/quest3_nero_l20.yaml`：

1. 将 `arm.can_channel` 和 `hand.can_channel` 从 `CHANGE_ME` 改为现场确认的通道。
2. 标定 `calibration.R_base_quest`，再把 `calibration.calibrated` 改为 `true`。
3. 确认 `quest.side` 与 `hand.type` 对应同一只手。
4. 用下面的 `record-start-pose` 命令记录 Nero 的七轴起始姿态。
5. 相机默认 `camera.model: L515`、640×480，RGB 和深度均采集，深度对齐到彩色。单相机 `camera.serial` 可留空，多相机应填序列号；只做无视觉控制时设 `camera.enabled: false`。

640×480 是保存尺寸：L515 原生彩色使用 960×540、深度使用 640×480，先对齐，再共同中心裁剪/缩放至 640×480，内参也相应换算。这样保留中央 4:3 视野。L515 的原生彩色格式见 [官方规格表 3-7](https://dev.realsenseai.com/download/7691/)。

在这些值未完成前，`run --control arm/both` 会拒绝继续，不会猜测 CAN 通道、安装方向或起始姿态。

## 无硬件输入检查

先确认 Quest，并复用你已经验证过的 ADB reverse：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

先启动本机常驻监听。它使用根目录 AnyDexRetarget 的 Quest3 plugin 监听 HTS 的 TCP 8000，并只在 `127.0.0.1:8001` 转发最新完整帧。保持这个终端运行；之后重启 `run`、`input-check` 不会断开 HTS，也无需重启头显端 streamer：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml quest-listener
```

再运行只检查 Quest 与 AnyDex、不打开 CAN 的命令：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml input-check
```

如需恢复每个命令自行监听 TCP 8000 的旧方式，将 `quest.input_mode` 改为 `direct`；此时不要运行 `quest-listener`，且重启控制进程可能导致 HTS 断线。

该命令不包含另一套 TCP 或 CSV 解析器；它与 `AnyDexRetarget/example/teleop_sim.py`、`teleop_real.py` 使用同一个 `example/input/quest3.py`。AnyDex 已完成 Unity 左手系到右手系、腕姿和 landmarks 转换，集成控制只额外应用现场 `R_base_quest`。

原有 AnyDex 单独测试命令保持有效，可作为集成前的回归基线：

```bash
cd AnyDexRetarget/example

python teleop_sim.py \
  --input quest3 \
  --robot linker_l20 \
  --port 8000 \
  --protocol tcp \
  --hand right

python teleop_real.py \
  --input quest3 \
  --robot linker_l20 \
  --hand right \
  --quest3-protocol tcp \
  --quest3-port 8000 \
  --l20-transport can \
  --l20-can-channel can1 \
  --l20-can-speed 30 \
  --l20-can-command-hz 30 \
  --l20-can-max-step 4
```

## 只读实机检查

填写 CAN 通道后，先分别检查，最后联合检查。该命令只连接和读取，不使能、不发送目标：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml preflight --control arm
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml preflight --control hand
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml preflight --control both
```

Nero preflight 会比较当前七轴反馈经本仓库 URDF/FK 算出的 flange 与 SDK flange 反馈；默认允许 `10 mm / 5°`，超出就停止。它是启动一致性检查，不是标定替代品。

确认 preflight 后，可用项目内的“只使能”命令使能 Nero。它不发送关节或笛卡尔目标，指定 `--execute` 后直接执行，仍需现场监护、急停可达：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml arm-enable --execute
```

该进程使能后断开连接，不自动失能；如果现场已有批准的使能流程，也可以使用现场流程。

## 记录 Nero 起始姿态

先通过现场允许的方式将 Nero 放到期望起始姿态，然后读取并保存当前七轴关节角：

```bash
python3 -m teleop.cli \
  --config configs/quest3_nero_l20.yaml \
  record-start-pose \
  --output configs/nero_start_pose.yaml
```

该命令只连接、读取和保存，不使能、不发送运动目标；连接后最多等待 3 秒以收齐七轴反馈，终端会同时打印 rad 和 degree。目标文件已存在时默认拒绝覆盖，确认需要替换后添加 `--overwrite`。

起始姿态文件只包含固定顺序的 `joint1`–`joint7` 和七个 rad 值。`arm.start_pose_file` 指向该文件；启动运动参数为：

```yaml
arm:
  start_pose_file: configs/nero_start_pose.yaml
  start_speed_percent: 10
  start_tolerance_rad: 0.02
  start_timeout_s: 30.0
```

需要恢复 Nero 控制状态时，可使用以下命令。它先请求失能并确认已失能，才发送
`reset`；失能或急停后的复位可能造成机械臂下落，因此必须先提供机械支撑、清空
工作区并确认急停可达。它不是物理急停的替代品。

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml arm-reset --execute
```

## 正式控制与录制

`--control` 必须明确指定，`run` 指定 `--execute` 后直接执行，无交互确认。Nero 必须已通过上面的命令或现场批准流程使能；`run` 不会隐式使能或自动失能。

对于 `--control arm/both`，程序首先校验记录的七轴目标和官方关节限位，然后使用低速 `move_joints` 到达起始姿态并等待反馈误差收敛。到位后仍保持暂停：操作者摆好 Quest，再按 `R` 才建立腕部零点并开始叠加。运行中再次按 `R` 只在机械臂当前位置重新锚定，不会自动返回起始姿态。

单侧普通 IK 求解失败也使用 `IK HOLD`，不发送当前帧，也不追加录制；继续检查新目标，可达且通过原有步长、工作区和关节连续性检查后自动恢复。输入丢失、越界及数值异常仍锁定暂停，需 R。R/B/S/Q 使用无文本预读的终端按键读取，连续按键不会因文本缓冲而滞留。

关节限位预防与恢复由 `arm.ik_joint_limit_margin_deg`（默认 2°）和 `arm.ik_feedback_limit_tolerance_deg`（默认 1°）控制，双侧配置在 `sides.left.arm` 定义并由右侧继承。旧逻辑要求真实反馈严格落在 URDF 范围内，否则直接拒绝求解；本机 SDK 的部分限位稍宽于 URDF，例如 J6 上限为约 55°，URDF 为 54.431°，因此会出现 SDK 仍合法而 IK 无法向内恢复的状态。现在仅将优化器初值投影到缓冲区内，真实反馈保持原值；求解结果必须位于内侧命令区间，相对于真实反馈通过步长检查，并通过原有 FK 精度验收。不会裁剪求解后的关节角或重设目标姿态来伪造成功。超过恢复容差的反馈仍报 `IK_JOINT_LIMIT` 并打印具体关节；该处理不清除控制器故障或重新使能硬件。

到达起始姿态后，程序用同一时刻的 SDK flange/TCP 反馈求固定的 flange→TCP 偏置，再把它应用到当前关节反馈的 URDF/FK flange，以此建立与 IK 模型一致的起始 TCP 和 Nero base 轴对齐立方体工作区。默认边长为 `safety.tcp_workspace_cube_side_m: 0.30`，即每个轴相对起始 TCP 最多移动 ±0.15 m。任何一轴越界都会在发送机械臂目标前暂停，不会把目标截断到边界。检查现场空间后再调整此值；按 `R` 不会移动或重置立方体中心。

```bash
# 先手，再臂，最后联合验证
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control hand --execute
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control arm --execute
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control both --execute --record-dir data/lerobot/nero_l20_act
```

运行键：

- `R`：以当前 Quest 腕部和当前七轴反馈经 URDF/FK 得到的 Nero flange 重新锚定；L20 滤波/速率状态同步重置。
- `B`：开始一个 episode。
- `S`：停止并保存当前 episode。
- `Q`：停止发送；若正在录制则保存，然后断开。

录制按原控制循环采样，`control_hz` 默认仍为 20。L515 的硬件深度流为 30 FPS，后台读取后由每个有效控制周期选取最近帧；图像采集和写盘不在控制线程等待。`B` 开始一个 episode，重复按 `B` 不会清空正在录制的数据；`S` 等待已排队图像写完并保存，随后仍可遥操作或再次按 `B`。暂停期间不新增轨迹/视觉样本。

运行时每约 2 秒有效控制时间输出一次 `[control timing]`（退出时也输出剩余统计），显示 IK、手部 retarget、两设备发送、seed/发送后反馈读取、轨迹追加和周期耗时的平均值/P95/最大值（ms），以及有效周期频率和超过名义周期的次数。无需开始录制即可观察；开始录制后，各阶段耗时同时存入 `diagnostics/episode-XXXXXX.npz` 的 `timing_*_s` 字段，单位为秒。暂停和重锚定不混入有效周期统计，计时不改变控制调用顺序或限位逻辑；字段说明见 [数据格式](docs/dataset_schema.md)。

单侧 `run`（包括 `--side left/right`）也默认使用一个独立计算进程，命令和 R/B/S/Q 操作不变。启动时打印计算进程 PID；IK、retarget 的阶段时间由计算进程测量，`compute_ipc` 记录进程通信及控制进程恢复执行的等待，`ik_cpu`、`hand_retarget_cpu` 分别记录实际计算线程 CPU 时间，CPU 时间是对应阶段的组成部分，不能重复相加。发送前若输入已过期则暂停，需 R 重新锚定；超时或计算进程退出则保存已有样本并停止。`wrist-only`/标定模式仅把 IK 移入计算进程，手指回放保持原流程。

联合录制现在直接保存为 **LeRobot v3.0（LeRobot 0.4.4）**，供原版 ACT 使用：

```text
data/lerobot/nero_l20_act/
├── data/                       # 官方 Parquet：23 维状态、23 维目标
├── videos/observation.images.front/  # 640×480 RGB，H.264
├── meta/                       # 官方索引、任务、统计信息
├── depth/episode-000000.h5      # 对齐 Z16 深度，uint16 无损
└── diagnostics/episode-000000.{npz,json}  # 原始诊断、时钟、内参
```

采集继续使用原来的遥操作环境；按 `S` 时自动调用配置中的 `act` Python，使用官方 API 写入并重新加载检查。同一目录可以追加多个 episode，也可以退出后继续追加。RGB 在控制期间后台暂存为 PNG，只在保存时编码一次。保存后不需要手动转换。

`observation.state` 是发送前反馈（NERO 7 维 rad＋L20 16 维 SDK raw）；`action` 是本轮最终下发目标。L20 仅取零起始索引 `0–10、15–19`，其余四个保留槽不参与训练。深度通过 episode 和 `frame_index` 对应 RGB，乘元数据中的 `depth_scale_m` 得到米；本次 ACT 只使用 RGB。

两个配置已开启 L515。正式联合采集需要 `--control both`、完整 RGB-D 和有效关节反馈；缺帧或保存失败时保留临时源文件并报错，不生成填零的训练样本。暂存目录为输出目录旁的 `.nero_l20_act-staging/`。单独调试 `arm` 或 `hand` 的录制仍保存原诊断格式，写到输出目录旁的 `nero_l20_act_diagnostics/`，不混入联合训练集。旧 `data/episodes` 不转换、不修改；原 `replay_episode.py` 仍用于旧格式，不支持新 LeRobot 数据集。

采集前把 `recording.lerobot.task` 改为本次任务描述；`repo_id` 是本地标识，不会触发上传。详细格式、离线 ACT 检查和训练命令见 [数据格式](docs/dataset_schema.md)。

正常退出、Quest 超时、步长越界和程序异常都不会自动触发电子急停或 Nero disable。程序的职责是停止继续发送并断开；现场急停与故障处置仍由操作者执行。

单个起始关节姿态只定义终点，不定义无碰撞路径。启动运动使用 Nero 官方点到点关节运动；现场必须保证从当前姿态到记录姿态的路径可用。如果需要绕开障碍物，应另行定义多个 waypoint，而不是提高速度或放宽到位检查。

## 只跟踪手腕＋按键重播手指

```bash
python3 -m teleop.cli --config configs/pico4_nero_l20.yaml \
  run --control both --hand-mode wrist-only --execute
```

该模式不使用 VR 手指重定向。启动时，中指、无名指、小指设为官方“张开”姿态；拇指、食指设为 `videos/observation.images.front/chunk-000/file-001.mp4` **9.000 秒**处的目标姿态。当前张开→闭合片段取自 episode 1 的视频 9～10 秒（第 180～199 帧），保存在 `hand.replay_clip_file` 指向的 [CSV](configs/l20_thumb_index_9_10s.csv)，共 20 帧、名义 20 FPS，运行时无需打开原视频或数据集。

按 `R` 锚定后，机械臂跟踪腕部位置和 **yaw**；**roll、pitch 固定为示教参考值**。参考值由该视频 9.000 秒处的关节反馈经同一套 URDF/FK 得到，保存在 `arm.wrist_only_roll_pitch_deg`，约为 `[-98.184°, +15.964°]`，采用 Nero base 固定坐标系下的 XYZ 外旋欧拉角。按 `R` 后，roll/pitch 目标以每周期最多 1° 的角度向量步长平滑收敛到参考值；yaw 使用锚定时机械臂 yaw 加上 Pico 的偏航增量，Pico 的 roll/pitch 不参与目标。再次按 `R` 会重新建立平移和 yaw 锚点，固定的 roll/pitch 参考值保持不变。

`P` 从头重播一次两指动作，结束后保持末帧；再次按 `P` 从头开始。`O` 中断重播并回到片段第一帧。其余三指始终保持张开。重播按片段时间选择当前目标，随控制周期发送，不额外阻塞手腕控制；控制周期较慢时会跳过已经过去的片段帧。

输入超时、IK 失败或工作区越界时，两设备跳过当前帧，手指重播同时取消。普通 IK 求解失败可在目标恢复可达后自动继续，其他暂停需按 R。恢复时保持最后一次发送的手指目标，需要再次按 `P` 才重播，或按 `O` 回到初始姿态。`B/S/Q` 仍用于录制、保存和退出；录制的手部 action 为本模式实际发送的目标。省略 `--hand-mode wrist-only` 时仍使用原来的完整手部跟踪。

## RGB-D 与拈布 TCP 联合标定

保持 replay 闭合手型，通过 Pico 改变位置和完整腕部姿态，按 `C` 保存静止后的 RGB-D 与 Nero FK；退出采集后人工点选接触点，再联合求解相机外参与 TCP 偏移：

```bash
python3 scripts/calibrate_tcp_camera.py collect data/calibration/pinch-01 \
  --config configs/pico4_nero_l20.yaml --execute
python3 scripts/calibrate_tcp_camera.py annotate data/calibration/pinch-01 --scale 2
python3 scripts/calibrate_tcp_camera.py solve data/calibration/pinch-01
```

默认固定 replay 最后一帧，可用 `--hand-frame` 选择。采集用 `R` 锚定/恢复、`C` 暂停并等待反馈稳定后采样、`Q` 退出；必须包含不同 roll/pitch 倾斜，不能只改变位置和 yaw。结果写入会话目录的 `calibration.yaml`，包含 `base_from_camera` 和完整的 `flange_from_tcp`。坐标约定、点选要求及结果使用见 [标定说明](docs/calibration.md#5-rgb-d-相机外参与拈布虚拟-tcp-联合标定)。

## 轨迹重放与视频查看

让 Nero 重现录制的实际关节轨迹，并同步显示当时的 RGB-D（替换为实际 episode 目录）：

```bash
python3 scripts/replay_episode.py data/episodes/episode_20260929T120000.000000Z \
  --config configs/quest3_nero_l20.yaml --execute
```

同时重放机械臂和灵巧手，加 `--control both`；仅灵巧手用 `--control hand`。默认 `--control arm`，原命令仍只控制 Nero：

```bash
python3 scripts/replay_episode.py data/episodes/episode_20260929T120000.000000Z \
  --config configs/quest3_nero_l20.yaml --control both --execute
```

指定 `--execute` 后，先用 `arm.start_speed_percent` 点到点移动到该 episode 的第一条 `arm_joint_position`，按 `start_tolerance_rad` / `start_timeout_s` 等待到位；不使用 `nero_start_pose.yaml`。必须保证起始移动及整段录制轨迹的现场路径无碰撞。机械臂需已使能；脚本不会自动使能，可通过已有的 `teleop.cli --config ... arm-enable --execute` 命令使能。

选中灵巧手时，使用 `hand.can_channel` / `hand.type` 连接，按 `hand.speed` 设置五指速度，并发送第一条 `hand_position_raw`；`both` 模式在 Nero 初始到位后才发送手部首姿态。请确认手部周围无夹持风险、手指已到首姿态，再按空格开始。手部不自动判断到位；仅手模式不连接 Nero。

到位后，在视频窗口按空格开始/暂停发送。Nero 把 `arm_joint_position` 作为 `move_js` 目标；L20 把同一行 `hand_position_raw` 原样作为 raw 位置目标，保留 20 槽位顺序。不使用 `arm_action` / `hand_action_raw`，不调用 Quest/IK。手部数据必须为 `(N,20)` 的 0–255 整数，且记录的 `quest_side` 必须与 `hand.type` 一致；缺失、-1 占位或侧别不匹配会在机械臂开始移动前拒绝，不回退到动作目标。L20 原始记录来自 SDK 反馈缓存，不能保证每行都是新反馈。

发送间隔取自轨迹时间戳；可加 `--speed 0.5` 延长为两倍间隔，实机不允许大于 1 倍速。`--speed` 不改变 L20 的 `hand.speed`。慢帧只延后后续发送，不突发补发；同一行按顺序发送给两台设备，不是硬件同步或精确到位保证。Nero 检查全段关节限位、相邻关节步长，并在每次发送前按 `arm.max_joint_delta_rad` 检查目标与当前反馈的差值。

实机模式禁用 `A/D` 跳帧及末尾循环重播；末尾停止发送，`Q/Esc` 或 Ctrl+C 退出。**暂停/退出只停止后续目标，不是急停，机械臂和灵巧手可能继续到达最后目标；不会自动失能、松手或 reset。** 需紧急停止时使用现场急停。RGB-D 是录制视频，与同一条轨迹样本关联，不是实时相机画面。

离线回放已录制的 episode（不连接设备），将示例目录替换为实际录制目录：

```bash
python3 scripts/replay_episode.py data/episodes/episode_20260929T120000.000000Z
# 可选：半速播放，深度伪彩色固定显示 0～3 米
python3 scripts/replay_episode.py data/episodes/episode_20260929T120000.000000Z --speed 0.5 --depth-max-m 3
```

窗口显示 RGB、深度伪彩色和轨迹字段；空格暂停/继续，`A/D` 暂停并前后逐条查看，`J/L` 翻页查看全部 NPZ 字段（原始单位），`Q/Esc` 退出。离线模式末尾自动暂停，空格从头重播。完整元数据打印到终端。播放按轨迹时间戳调度，RGB-D 按 `camera_frame_index` 关联，缺失图像明确显示为空。支持无相机 episode 和原有同名 `.npz + .json`（传入 `.npz` 路径）。需要有图形桌面的 OpenCV、NumPy、h5py，沿用视觉依赖，无需 RealSense SDK；仅实机重放额外需要所选设备的 Nero / LinkerHand SDK 和 CAN 连接。

## 低层只读示例

单设备排障命令必须显式给通道：

```bash
python3 examples/nero_example.py --can-channel can0
python3 examples/l20_example.py --can-channel can1 --hand-type right
```

它们默认只读。带 `--execute` 后直接执行单步动作，只用于现场分级验证。

## 折布 agent

架构见 [docs/closth_agent.md](docs/closth_agent.md)，代码阅读与运行说明见 [docs/cloth_framework.md](docs/cloth_framework.md)。唯一主流程使用 `codex exec + MCP + 独立持久 Python worker / 控制进程`，入口为 `python -m cloth_agent`。

使用已配置好的 `nero` 环境进行首轮 agent 实机测试，连接右侧 Nero、右侧 L20 和 L515。
启动前确认右臂已使能、配置中的 CAN 通道/相机序列号与当前设备一致，并使用当前现场标定。
`run` 先启动 worker/控制进程并公布人工 `hold` 入口，再自动完成起始位姿准备，之后才启动 MCP/Codex，无需预先启动 `serve`。
默认 `startup.enabled: true`：先回到 `nero_start_pose.yaml` 的记录关节位姿，再保留其法兰位置/yaw、
将 roll/pitch 调整到标定的示教倾角。此时的抓取朝向定义为 `yaw=0`；后续规划 yaw 是相对它绕 task +Z 的增量，单位 rad。
准备过程中也会实际运动；未确认到位、规划失败、超时或人工停止时不会启动 agent。
首轮任务完成一次“观察 → 抓取 → 小幅抬升 → 再观察 → 放回”的闭环：

每轮测试前，在 `configs/cloth_agent.yaml` 中修改一次本轮记录目录。相对路径以配置文件所在的 `configs/` 目录为基准：

```yaml
session:
  output_dir: ../data/agent_runs/robot-grasp-02
```

```bash
conda activate nero
cd /home/lab404/WHR/dexterous-manip-ws

python -m cloth_agent run \
  --execute \
  --task '进行第一次抹布实机抓取测试。先读取真实机器人状态，获取并展示现场图像；收到图像反馈后，选择清晰可见且深度有效的布料抓取点。以 L20 食指、拇指闭合接触点为虚拟 TCP，进行一次 top-down 抓取，沿 task +Z 抬升最多 0.05 m，再获取并展示图像，判断是否抓住。验证后将布料放回原处、松手并撤离布面，再结束任务。目标或深度不明确、规划或执行失败时停止并报告，不盲试、不重复抓取，本轮不执行折叠。'
```

另一个终端同样进入 `nero` 环境和项目目录；需要人工停止并保持时，直接通知本次控制进程：

```bash
python -m cloth_agent hold
python -m cloth_agent hold-status
```

上面的 `--execute` 会直接连接设备并按配置准备起始抓取位姿，无需输入确认词；程序不自动使能、执行硬件回零或复位。
固定准备流程及完整反馈保存在 `cells/000.py` / `000.json`，不占 agent 的 cell 预算；准备超时由 `startup.timeout_s` 控制。
`configs/cloth_agent.yaml` 启用相机与自动准备；人工保持测试用的 `configs/cloth_agent_estop.yaml` 关闭了相机和自动准备，不用于这次视觉抓取任务。
每次重试只需把配置中的 `session.output_dir` 换成新目录，`run` / `serve` / `monitor` / `hold` / `hold-status` / `release-hold` 共用它，旧记录不覆盖。本轮运行期间保持该字段不变。
默认读取 `configs/cloth_agent.yaml`；使用其他配置时，各命令加同一个 `--config 配置文件`。
临时覆盖仍可使用 `run/serve --output-dir 目录`、`monitor 目录` 或 `hold/hold-status/release-hold 路径/session.json`，显式参数优先于配置。
需要改测试目标时修改 `--task`；仅手动使用外部 MCP client 时才使用 `serve`。

`robo` 提供状态、RGB-D、附图、像素反投影、top-down 位姿、`move_J` / `move_L`、拇指/食指开合及普通暂停；固定知识随启动 prompt 注入，没有 info、公开 plan/execute 或抓布判定接口。操作约定见 [robot_operations.md](cloth_agent/robot_operations.md)。人工 `hold` 绕过 MCP/Codex，锁定动作后发送一次普通 J 固定关节目标并持续监督反馈，agent 结束后保持进程仍运行。只有机械臂已安全落座或获得外部支撑后，操作者才能 `release-hold`。旧 `estop` 已禁用：Nero 阻尼急停现场出现下落，不用于悬停。人工静止保持已由操作者现场验证，记录见 `data/agent_runs/estop-idle-02/`，其中已连接真实设备、发送固定关节目标且记录的关节误差为 0；该记录没有执行运动 cell，不代替本次视觉抓取与运动中停止验证。保持操作见 [人工保持说明](docs/cloth_estop_test.md)。

在另一个终端启动只读观察页，默认读取同一配置中的本轮目录：

```bash
conda activate nero
python -m cloth_agent monitor
```

浏览器打开 `http://127.0.0.1:8765/`，查看 agent 时间线、当前 cell 代码与反馈、观测图片和任务状态。
可在 agent 启动前打开，也可回看已经结束的任务；端口冲突时加 `--port 8766`。
监控不连接机器人、不发送控制命令；关闭页面或监控不会停止机器人。
stdout 和运动反馈在 cell 返回后更新，图片为观测快照；详细说明见 [实时观察页](docs/cloth_monitor.md)。

旧 app-server、普通子任务脚本提交、参数回信、demo 与对应执行后端已删除，不再提供原来的 `demo/script` 命令或 `cloth_agent.tools` 接口。

`geometry.py`、`planning.py`、`config.py` 与标定配置已接入新任务主流程；IK/TOPPRA 在 worker 计算，设备进程复核与执行。几何函数仍供现有 TCP/相机标定使用。`configs/cloth_agent_mock.yaml` 仅作为规划测试的数值输入，不对应假设备运行或 demo。

## 目录

```text
AnyDexRetarget/              权威手部重定向实现与 Quest3/L20 配置
assets/nero/                 本项目固定使用的 Nero URDF
configs/                     唯一遥操作现场配置
robot_control/               Nero、L20 封装与 Nero IK
cloth_agent/                 Codex exec、MCP、独立 worker/控制进程及几何规划底层
teleop/                      AnyDex 输入适配、现场坐标、锚定、控制、录制、CLI
examples/                    单设备低层只读/小步诊断
tests/                       不接硬件的纯逻辑与 wrapper 测试
docs/architecture.md         模块边界与控制状态
docs/calibration.md          现场标定与首次验证顺序
docs/dataset_schema.md       episode 字段与单位
docs/review.md               本次改造的代码 review 导航
```

## 当前验证边界

当前 agent 按用户要求仅检查语法，没有运行 mock、测试、Codex 联调或实机运动；运行行为尚待验证，范围见 [docs/cloth_framework.md](docs/cloth_framework.md)。遥操作的现场验证顺序见 `docs/calibration.md`。
