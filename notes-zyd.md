## Pico 4 → Nero + L20

以下命令均在项目根目录运行，使用原来的 Python/SDK 环境。

头显：左下角资源库 → robo → 勾选手部追踪、reconnect。

USB 连接时先开启 ADB 转发：

```bash
adb reverse tcp:63901 tcp:63901
adb reverse --list
```

终端 1：启动原 AnyDex 的中转转发，并保持运行。

```bash
python3 AnyDexRetarget/example/input/pico4_daemon.py
```

终端 2：检查 Pico 腕部和手指输入，不打开 CAN。

```bash
python3 -m teleop.cli --config configs/pico4_nero_l20.yaml input-check
```

`configs/pico4_nero_l20.yaml` 沿用原来的 CAN、控制频率、IK、平滑、运动限制和相机参数，手部改用 AnyDex 的 Pico/L20 adaptive 配置。`calibration.R_base_pico4` 暂用原 Quest 矩阵作初值；现场确认 Pico 腕部方向与 Nero base 的映射后，将 `calibration.calibrated` 改为 `true`，再运行机械臂。仅检查输入或控制手部不要求这项标定。

终端 2：沿用原联合遥操与录制流程。

```bash
python3 -m teleop.cli \
  --config configs/pico4_nero_l20.yaml \
  run --control both --execute \
  --record-dir data/episodes
```

输入 `EXECUTE` 后，按 R 开始跟随，B 开始录制，S 保存，Q 保存并退出。重复按 B 不会清空正在录制的数据。输入失效或超时后仍需按 R 重新锚定。

CLI 默认配置也已改为 Pico；原 Quest 链路可继续显式使用 `--config configs/quest3_nero_l20.yaml`。

## 保存结构

```text
data/episodes/
└── episode_<UTC>/
    ├── trajectory.npz
    ├── metadata.json
    └── camera/
        ├── color.mp4
        └── depth.h5
```

为兼容现有回放，NPZ 继续使用 `quest_wrist`、`quest_landmarks`、`quest_received_at` 字段名；Pico 录制的 metadata 使用 `input_device: pico4` 标明来源。
