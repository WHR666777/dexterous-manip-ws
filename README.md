# Quest 3 → Nero + LinkerHand L20 遥操作

本仓库当前只解决一个明确目标：用一只 Quest 3 手同时控制一台 AgileX Nero 七轴机械臂和一只 LinkerHand L20，并记录后续 OP 训练所需的数据。当前实现是**单臂 + 单手**；双臂 + 双手留作两套单臂控制器的后续组合，不在当前范围内。

`AnyDexRetarget/` 是唯一有效的 Quest 输入和手部重定向实现。原 `vr_teleop` 只作为腕部相对锚定和平滑方案的参考，所需跟踪逻辑已经迁入 `teleop/`，不再作为运行依赖。

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

## 现场必须填写

编辑 `configs/quest3_nero_l20.yaml`：

1. 将 `arm.can_channel` 和 `hand.can_channel` 从 `CHANGE_ME` 改为现场确认的通道。
2. 标定 `calibration.R_base_quest`，再把 `calibration.calibrated` 改为 `true`。
3. 确认 `quest.side` 与 `hand.type` 对应同一只手。

在这些值未完成前，`preflight`/`run` 会拒绝继续，不会猜测 CAN 通道或安装方向。

## 无硬件输入检查

先确认 Quest，并复用你已经验证过的 ADB reverse：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

PC 侧直接使用根目录 AnyDexRetarget 的 Quest3 plugin 监听 TCP 8000。再运行只检查 Quest 与 AnyDex、不打开 CAN 的命令：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml input-check
```

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

确认 preflight 后，可用项目内的“只使能”命令使能 Nero。它不发送关节或笛卡尔目标，但仍需现场监护、急停可达，并要求双重确认：

```bash
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml arm-enable --execute
```

该进程使能后断开连接，不自动失能；如果现场已有批准的使能流程，也可以使用现场流程。

## 正式控制与录制

`--control` 必须明确指定，`run` 还要求 `--execute` 并在终端输入 `EXECUTE`。Nero 必须已通过上面的命令或现场批准流程使能；`run` 不会隐式使能或自动失能。

```bash
# 先手，再臂，最后联合验证
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control hand --execute
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control arm --execute
python3 -m teleop.cli --config configs/quest3_nero_l20.yaml run --control both --execute --record-dir data/episodes
```

运行键：

- `R`：以当前 Quest 腕部和当前 Nero flange 重新锚定；L20 滤波/速率状态同步重置。
- `B`：开始一个 episode。
- `S`：停止并保存当前 episode。
- `Q`：停止发送；若正在录制则保存，然后断开。

正常退出、Quest 超时、步长越界和程序异常都不会自动触发电子急停或 Nero disable。程序的职责是停止继续发送并断开；现场急停与故障处置仍由操作者执行。

## 低层只读示例

单设备排障命令必须显式给通道：

```bash
python3 examples/nero_example.py --can-channel can0
python3 examples/l20_example.py --can-channel can1 --hand-type right
```

它们默认只读。带 `--execute` 的单步动作仍需输入 `EXECUTE`，只用于现场分级验证。

## 目录

```text
AnyDexRetarget/              权威手部重定向实现与 Quest3/L20 配置
assets/nero/                 本项目固定使用的 Nero URDF
configs/                     唯一遥操作现场配置
robot_control/               Nero、L20 封装与 Nero IK
teleop/                      AnyDex 输入适配、现场坐标、锚定、控制、录制、CLI
examples/                    单设备低层只读/小步诊断
tests/                       不接硬件的纯逻辑与 wrapper 测试
docs/architecture.md         模块边界与控制状态
docs/calibration.md          现场标定与首次验证顺序
docs/dataset_schema.md       episode 字段与单位
docs/review.md               本次改造的代码 review 导航
```

## 当前验证边界

此工作副本无法运行项目或连接硬件。因此本次只做静态代码与引用检查，没有执行 Python、测试、Quest、CAN 或实机运动。现场验证顺序见 `docs/calibration.md`；在这些步骤通过之前，不应把本实现视为已验证的机械安全系统。
