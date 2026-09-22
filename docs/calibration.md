# 现场标定与首次验证

## 1. 通道确认

在没有其他控制程序运行时，用系统工具确认 Nero 与 L20 各自对应的 SocketCAN 通道和 bitrate。将结果写入 `configs/quest3_nero_l20.yaml`，不要依赖设备插入顺序猜测。先分别运行 `preflight --control arm` 和 `preflight --control hand`，再运行 `both`。

## 2. 坐标方向标定

Quest 原始 Unity 坐标由 AnyDex 的 `example/input/quest3.py` 转换到右手系。集成控制不会再次执行该转换。`R_base_quest` 只描述 AnyDex 右手系到 Nero base 方向的旋转：

```text
p_base = R_base_quest · p_anydex_rh
R_base_wrist = R_base_quest · R_anydex_rh_wrist
```

现场至少用三个不共线的小方向动作确认正负方向。将正交、行列式为 `+1` 的 3×3 矩阵写入 YAML 后，才把 `calibrated` 改为 `true`。

绝对平移不需要测量：按 `R` 时，当前 Quest 腕位置与当前 Nero flange 被相对锚定。旋转也以按 `R` 时的腕姿作为零残差。

## 3. URDF/FK 一致性

`preflight --control arm` 用当前关节反馈做 FK，并与 SDK flange 反馈比较。默认阈值为 10 mm / 5°。超出时优先检查 URDF 版本、关节顺序/符号、flange 定义和 SDK 姿态欧拉约定；不要放宽阈值掩盖模型不一致。

## 4. 分级运动验证

1. 运行 `input-check`，只动手和腕，确认 TCP 持续、side 正确、L20 raw 数值随手指合理变化。
2. 分别执行两项 preflight，确认反馈稳定且无其他控制者。
3. 将 Nero 放到确认过的起点，用 `record-start-pose` 保存七轴 rad；检查 YAML 顺序是 `joint1`–`joint7`。
4. 仅运行 `--control hand`，使用最慢、最小范围动作检查 20 槽映射。
5. 在 Nero 已通过 `arm-enable --execute` 或现场流程使能，并有人监护、急停可达时，仅运行 `--control arm`；确认它先低速到达记录姿态，随后保持暂停，按 `R` 后再做毫米级位移和很小旋转。
6. 确认 1 cm / 5° 目标步门、10° 遥操作关节变化限制、TCP 立方体边界和 Quest 超时锁定均符合预期；首次检查立方体时逐轴缓慢靠近边界，并确认越界目标未发送。
7. 最后才运行 `--control both`，先不录制，再按 `B/S` 录短 episode 并离线检查字段。

初次实机验证期间不要提高 20 Hz、步长或 L20 raw slew；这些值只有在记录到现场证据后才能调整。
