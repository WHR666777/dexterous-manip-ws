# 示教与 ACT：报告起草上下文

这份文档给负责把「示教」和「ACT」写成报告章节的 Agent 用。示教和 cloth 侧执行以 `dexterous-manip-ws` 为准；训练和推理脚本以 `C:\Work\dex-mani\lerobot`（LeRobot 0.4.4）为准。样式参照 `docs/report/agent_harness.tex`：按模块写清数据从哪来、公式是什么、参数取多少、和论文差在哪里、工程上为什么这样切。不要把下面的不确定项写成已实现的事实。

`agent_harness.tex` 开头已经把这两块定位成前两个模块：「怎样采集人类示教数据」和「怎样让策略网络把动作重放出来」。Agent Harness 是第三块，负责折布前半段的摆放，再把现场交回 ACT。起草时保持这个分工，不要把 cloth agent 的规划、Codex、TOPPRA 写进这两章，只在衔接处点到为止。

## 0. 起草约束

写中文技术叙述，和 `agent_harness.tex` 一样：先讲这一段解决什么，再讲模块和一次完整数据流，然后给公式、配置和取舍。代码标识用 `\texttt{}`。单位写清楚。不要把论文里的 ALOHA 硬件、50 Hz、四相机、14 维夹爪动作写成我们的系统。

下面几处文档和代码不一致。正文以代码和当前 YAML 为准，不要折中成一个新数字。

| 主题 | 代码 / 当前配置 | 另一处写法 | 起草时用哪边 |
| --- | --- | --- | --- |
| 遥操作与 ACT 的单周期关节增量 | `max_joint_delta_rad` / `motion.max_ik_step_rad` = `0.3345329252` rad，约 19.167°。`docs/ik_benchmark.md` 用的就是这个值 | `NeroIK` 类默认 10°；`docs/architecture.md` 有一处仍写「10° 单周期」 | 写 19.167°，并说明类默认是 10°、现场配置覆盖了它 |
| `run_act` 默认步数 | `cloth_agent/robo.py` 的 `run_act(policy="fold", steps=500)` | `robot_operations.md` 有一句「第一版默认 100 步」；`dex-mani/lerobot/README.md` 也写成 `steps=100`。独立推理脚本 `infer_act_bimanual.py` 的 `--steps` 默认确实是 100 | 任务 prompt 不传 `steps` 时，cloth 路径实际是 500。100 只是独立脚本和旧说明的默认 |
| ACT 重建损失 | LeRobot 0.4.4 用 L1，pad 帧不进损失 | 论文 Algorithm 1 写成 MSE；正文 IV.C 又说改用 L1 | 写我们跟随 LeRobot / 论文正文的 L1，不要写 MSE |
| Temporal ensemble | 两份写下来的训练命令都是 `chunk_size=60`、`n_action_steps=20`，没有设置 `temporal_ensemble_coeff`。LeRobot 0.4.4 里这两项互斥，默认 coeff 是 `None`。推理调用的是 `select_action` 的动作队列，不是 Algorithm 2 | 论文每步查询并做 \(w_i=\exp(-0.01 i)\) | 不要写成已启用 temporal ensemble。队列行为见第 11 节 |
| `run` 是否使能 | `teleop/controller.py` 和双侧 worker 都会调用 `enable_normal_mode()`，内部重试 `enable()` + `set_normal_mode()`，超时 5 s | README 写「run 不会隐式使能」 | 写代码实际会走正常模式使能；退出时不自动失能、不发急停 |
| 跟随速度百分比 | `move_js` 不设置速度百分比，注释写明由调用方控制更新频率。`follow_speed_percent: 30` 只留在配置里，对应已被注释掉的 `move_joints` | 容易写成「30% 速度跟随」 | 遥操作和 ACT 的连续目标都走 `move_js`，不要写 30% |
| 手速 | 双侧 Pico 配置 `hand.speed: [240, 240, 240, 240, 240]`；Quest 单臂配置是 160 | 不要混用 | ACT 数据采集以 Pico 双侧为准 |

`C:\Work\dex-mani\lerobot\outputs\` 是空的。`act_bimanual_default_run01` 的 `config.json`、归一化统计和 `model.safetensors` 没有跟源码一起拷下来。交接文档只说明 `checkpoints/last` 指向 step `050000`。因此不要从磁盘上的权重文件复述学习率或 ImageNet 开关；下面凡是写到该 checkpoint 的训练超参，都标明出处是哪份命令，而不是「打开权重核对过」。

不要编造成功率、演示分钟数、episode 条数，也不要和 ALOHA 论文的表格做实验对比。两边仓库都没有这些数字。

## 1. 系统在整条链路里的位置

硬件是两条 AgileX Nero 七轴臂和两只 LinkerHand L20，共用一台 Intel RealSense L515。示教把 Pico 4 的双手追踪变成两侧的关节目标，同时记下 RGB、对齐深度，以及发送前的关节反馈和实际下发的目标。正式训练集是 LeRobot v3.0，由钉死的 LeRobot 0.4.4 写成。ACT 只吃 RGB 和关节，不吃深度。

部署时的策略是双侧 checkpoint `act_bimanual_default_run01`，名字在 `configs/cloth_agent.yaml` 的 `act.policies.fold`。Cloth agent 先用自己的运动接口把毛巾对折并摆到垫子中央，再把两条臂送回记录的起始关节、把两只手设成训练集某一帧的手型，然后调用 `robo.run_act`。ACT 在固定步数内独占双臂和双手。返回值只表示步数跑完并确认位置保持，不表示毛巾折好了。

单臂 23 维数据路径还在：`configs/pico4_nero_l20.yaml`、`configs/quest3_nero_l20.yaml`，输出 `data/lerobot/nero_l20_act`，`robot_type=nero_l20`。当前折布任务用的是双侧 46 维，`configs/pico4_bimanual_nero_l20.yaml`，输出 `data/lerobot/nero_l20_bimanual_act`，`robot_type=nero_l20_bimanual`。两套不能追加进同一个目录。

进程和 Python 环境是分开的：

- 遥操作控制进程不导入 Torch。按 S 保存时，另起 `recording.lerobot.python`（配置为 `.../envs/act/bin/python`）跑 `python -m teleop.lerobot_export`。
- ACT 推理时，cloth 控制进程持有 CAN 和相机。模型在 act 环境的子进程里，双方只通过 stdin/stdout 传数值。
- 保存和推理都会在 `libffi.so.7` 存在时把它放进 `LD_PRELOAD`，给 TorchCodec / FFmpeg 4 用。保存另设 `HF_HUB_OFFLINE=1`，不上传。

## 2. 硬件、坐标和动作维度

### 2.1 臂

Nero 关节 1–7，单位 rad。IK 和示教目标使用仓库内固定 URDF `assets/nero/nero_description.urdf`，链是 `base_link` 到 `link7`。SDK 自己的法兰只在 preflight 和启动时做一致性对照，不拿来当 IK 模型。

Preflight 比较「当前关节反馈的 URDF/FK 法兰」和「SDK 法兰」：默认 10 mm、5°，超出就停。这是启动检查，不是标定。

双侧 CAN，写在 `configs/pico4_bimanual_nero_l20.yaml`：

| 侧 | Nero | L20 | 起始关节文件 |
| --- | --- | --- | --- |
| 左 | `can_arm_b` | `can_hand_b` | `configs/nero_left_start_pose.yaml` |
| 右 | `can_arm_a` | `can_hand_a` | `configs/nero_start_pose.yaml` |

左右 base 轴向已经确认相同，所以两侧用同一张 `R_base_pico4`。起始关节不复制。

`move_js` 把七轴目标交给 SDK 的 JS（follower）模式，发完即返回，不等到位，不做轨迹规划，不改速度百分比。发送前 `validate_joint_command` 检查官方关节限位，以及相对当前反馈的 `max_joint_delta`。超限直接拒绝，不裁剪。SDK 若把 `ENOBUFS` / `ENETDOWN` 记在通信错误里，封装会把它抛成 `RuntimeError`。

启动回起始姿态走另一条路：低速 `move_joints`，`start_speed_percent: 10`，到位容差 `0.02` rad，超时 30 s。这一步显式使用自己的速度百分比，不受遥操作单周期增量的同一套调用路径约束。到位后才允许按 R 锚定。运行中的 R 只在当前法兰上重锚定，不把臂送回起始姿态。

### 2.2 手

L20 官方协议是 20 个 0–255 的 raw 槽。索引 11–14 是保留槽，命令里固定写 255，不进训练，推理时也不改，沿用当前反馈。

有效 16 槽和 URDF 关节的对应关系在 `AnyDexRetarget/anydexretarget/linker_l20.py` 的 `L20_SLOT_MAP`。SDK 名字里的 “tip” 不是 URDF 里的指尖 mimic 关节：

| 槽 | SDK 名 | URDF 关节 | 方向 |
| --- | --- | --- | --- |
| 0 | thumb_base | THUMB_CMC_PITCH | decreasing |
| 1–4 | index/middle/ring/little_base | 各指 MCP_PITCH | decreasing |
| 5 | thumb_abduction | THUMB_CMC_ROLL | decreasing |
| 6–9 | 四指 abduction | 各指 MCP_ROLL | increasing |
| 10 | thumb_roll | THUMB_CMC_YAW | decreasing |
| 11–14 | reserved | 无 | 固定 255 |
| 15 | thumb_tip | THUMB_MCP | decreasing |
| 16–19 | 四指 tip | 各指 PIP | decreasing |

角度到 raw：把关节角在标定上下限里归一化到 \[0,1\]，decreasing 的槽用 `1-ratio`，再 `round` 到 0–255。标定限位（度）也在同一张表：拇指 CMC pitch 0–47.56，四指 MCP pitch 0–69.90，拇指 roll 0–79.64，四指侧摆 ±13.18，拇指 yaw 0–89.95，拇指 MCP 0–71.62，四指 PIP 0–100.27。保存进数据集时不把 raw 归一化；归一化留给 ACT 的数据集统计。

URDF 还有 5 个 mimic，优化器不把它们当独立变量。右手 `linker_l20_right_vis.urdf`：拇指 IP mimic `RIGHT_T_MCP`，倍率 1.03；食指、中指、无名指 DIP mimic 各自 PIP，倍率 0.89；小指 DIP 的 mimic 父关节写成了 `RIGHT_LF_MCP_PITCH`，倍率 0.89，和另外三指不同。左手小指 DIP mimic 的是 `LEFT_LF_PIP`。指尖 link 是 DIP/IP 上的 fixed 偏移，给重定向 FK 用。CAN 的 16 个有效槽对应的是独立驱动关节，不是这 5 个 mimic。

`slew_l20_command` 对 16 个有效槽做 `|Δ| ≤ max_raw_step`，保留槽保持目标值。示教和 ACT 都用 `max_raw_step = 10`。这是每个控制周期相对上一份已接受命令的限幅，不是相对很早的反馈。

手指速度是 5 个数，按指下发。双侧示教是 240。ACT 执行本身不重新设置手速。

### 2.3 训练向量

单臂 23 维：Nero 7 + L20 有效 16，顺序 `nero.joint1..7`，然后 `l20.thumb_base, index_base, middle_base, ring_base, little_base, thumb_abduction, index_abduction, middle_abduction, ring_abduction, little_abduction, thumb_roll, thumb_tip, index_tip, middle_tip, ring_tip, little_tip`。

双侧 46 维：左 23 + 右 23。名字加 `left.` / `right.`。`observation.state` 和 `action` 同序、同名。

状态是本周期发送之前读到的反馈。动作是本周期真正传给 `move_js` / `set_joint_positions_raw` 的目标。发送时刻不是执行完成时刻，也不是成功证明。发送之后再读的关节和手反馈只进诊断，不能拿来替换 `observation.state`。

### 2.4 相机

L515。原生彩色 960×540 RGB8，深度 640×480 Z16，硬件 30 FPS。`rs.align` 把深度对齐到彩色，再对两者做同一个中心裁剪，缩放到 640×480。彩色用 `INTER_AREA`，深度用 `INTER_NEAREST_EXACT`。内参按裁剪和 OpenCV 像素中心约定重算：`fx' = fx * sx`，主点 `(ppx - left + 0.5) * sx - 0.5`。深度 0 表示无效。米 = raw × `depth_scale_m`。空间对齐不等于和臂、手的硬件同步。

控制周期不跟着相机走。`control_hz: 20`，每个有效周期非阻塞地拿最近一帧。帧龄超过 `camera.max_age_s`（0.25 s）就当没有帧。

ACT 只使用这张 640×480 RGB。深度以 episode 为单位无损放在旁边。

## 3. 一次示教怎么发生

当前正式入口是 Pico 双侧：

```text
python3 AnyDexRetarget/example/input/pico4_daemon.py
python3 -m teleop.cli --config configs/pico4_bimanual_nero_l20.yaml run --control both --execute
```

USB 时头显侧需要 `adb reverse tcp:63901 tcp:63901`。头显开 robo，勾选手部追踪。数据流：

```text
Pico 4 双手 JSON
  TCP 63901 → pico4_daemon → 127.0.0.1:63902
  主进程独占这份输入、键盘、L515
  ├─ 左/右 spawn 控制进程：各自的反馈线程、CAN、目标门
  │    └─ 每侧再 spawn 一个计算进程：只做 IK 和 retarget，不碰硬件
  └─ 双方 PREPARE 都成功且输入没过期 → 并行 COMMIT → 一行联合样本
```

按键：

- R：两侧都用当前 URDF/FK 法兰和当前手腕重新锚定。手的滤波和 slew 基准一起重置。两侧都完成才清除暂停。
- B：开始一个 episode。再按 B 不会清空正在录的数据。
- S：等后台图像写完，调用 act 环境导出，成功并重新加载验证后删暂存。
- Q 或异常：停止发送；正在录的有效样本会保存。不自动失能。

暂停期间不加样本。空录制不产生正式 episode。

单臂 `teleop.controller.run` 仍保留，计算也放在独立进程里。Quest 3 走 HTS TCP 8000 → `quest-listener` → `127.0.0.1:8001`，解析器只有 `AnyDexRetarget/example/input/quest3.py`。集成层不再做 Unity 左手系到右手系，只乘现场 `R_base_quest`。Quest 配置和 Pico 用的是同一组旋转矩阵数值。

`wrist-only` 和 `calibration` 不是 ACT 示教路径。双侧配置只走 `tracking`。`wrist-only` 把 roll/pitch 收到一个固定的 base 系欧拉角，yaw 仍跟 VR，手指改为回放 CSV，不跑 AnyDex。标定模式把手指锁在回放片段的某一帧，用来采 TCP–相机点。报告若写示教主路径，不要把这两种模式写成数据采集方式；最多一句带过。

## 4. Pico 输入

实现：`AnyDexRetarget/example/input/pico4.py`。一条 JSON 同时带左右手。PXR 原生轴是 x 右、y 上、z 后，和 Quest HTS 的 Unity 轴不同。只把腕部变到右手系 x 前、y 左、z 上；手指点保留 Pico 原来的腕部相对坐标，因为 adaptive YAML 是按这套坐标调的。

26 个 SDK 关节去掉 Palm 和四指的 metacarpal，剩 21 点，排成 MediaPipe 顺序。腕是索引 1 的位置。点减去腕，得到腕部相对 landmarks。腕的位置和旋转再乘

\[
R_{\text{Pico}\to\text{RH}}
=
\begin{bmatrix}
0 & 0 & -1 \\
-1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}.
\]

旋转是 \(R_{\text{Pico}\to\text{RH}} R_{\text{wrist}} R_{\text{Pico}\to\text{RH}}^{\mathsf T}\)，四元数 xyzw。

不完整的帧直接丢掉，不用 0 填充。`isActive != 1`、关节数不是 26、位置非有限、腕四元数范数过小，都算无效。若消息里带 `locationStatus`，位置必须 PositionValid；腕还要 OrientationValid。

新鲜度：帧龄 ≤ `pico4.max_age_s`（0.25 s），且左右 `pair_skew` ≤ `max_pair_skew_s`（0.05 s）。任一侧不新鲜，双侧暂停，必须 R。过期目标不发送。

现场标定矩阵，两侧相同，`calibrated: true`：

\[
R_{\text{base}\leftarrow\text{Pico}}
=
\begin{bmatrix}
0 & 1 & 0 \\
-1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}.
\]

`teleop/coordinate_frames.py` 只做这一步：\(p_{\text{base}} = R p_{\text{RH}}\)，\(R_{\text{base}} = R R_{\text{RH}}\)。`arm` / `both` 在 `calibrated` 仍为 false 时拒绝运行。`input-check` 和纯手模式不要求这张矩阵。

## 5. 腕部相对锚定、TCP 和工作区

`teleop/wrist_tracker.py`。按 R 时：

1. 用当前七轴反馈做 URDF FK，得到法兰位姿。不用 SDK 法兰当锚。
2. 把这一帧手腕记为零点。锚点法兰就是当前法兰，所以按住 R 的瞬间目标就是「停在这里」。
3. 手的 retarget 滤波器清空，slew 基准设成当前 20 槽反馈。

之后每个周期：

\[
\begin{aligned}
r_t &= p^{\text{wrist}}_t - p^{\text{wrist}}_0, \\
\bar r_t &= \alpha r_t + (1-\alpha)\bar r_{t-1}, \\
p^{\text{flange}}_t &= p^{\text{flange}}_0 + s\,\bar r_t.
\end{aligned}
\]

配置 `position_scale s = 1`，`ema_alpha α = 0.4`。类构造函数的默认 α 是 0.8，YAML 覆盖成 0.4。平移死区 0。

旋转不用 EMA。相对旋转左乘到锚点法兰上：

\[
R^{\text{flange}}_t
=
\bigl(R^{\text{wrist}}_t (R^{\text{wrist}}_0)^{-1}\bigr)
R^{\text{flange}}_0.
\]

`negate_rot_xy: false`，旋转死区 0。这是 1:1 的增量遥操作：人的手在 Pico 右手系里移多少，法兰在 Nero base 里就移多少；人的手相对锚定瞬间转多少，法兰就相对锚定瞬间转多少。没有绝对位姿跟踪，所以人可以站在舒服的位置上按 R。

`wrist-only` 才把 roll/pitch 从这条相对旋转里拆出去。双侧 tracking 不走那条分支。

工作区在到达起始关节之后建立，之后 R 不移动中心：

1. 同一时刻读 SDK 法兰和 SDK TCP，\(T_{\text{flange}}^{\text{tcp}} = (T_{\text{SDK flange}})^{-1} T_{\text{SDK TCP}}\)。
2. 用关节反馈的 URDF 法兰乘这个固定偏置，得到和 IK 一致的起始 TCP。
3. 以该点为中心，在该臂 base 里放一个轴对齐立方体。双侧边长 `tcp_workspace_cube_side_m: 0.9`，即每轴 ±0.45 m。单臂 Quest 配置同样是 0.9。

每个候选法兰先换成 TCP。任一轴超出半边长，或者相对上一份已接受目标的平移超过 `max_position_step_m: 0.1`、转角超过 `max_rotation_step_deg: 16`，目标门锁存暂停。不把目标夹到边界上。IK 失败的周期不更新这扇门的基准，所以恢复后步长仍相对最后一次真正发出去的目标。

Cloth agent 里还有另一套工作区：以起始关节 FK + 标定 TCP 为中心的立方体，检查的是 task 系 TCP。右臂 task 就是右臂 base。ACT 下发前会走 cloth 的 `validator._check_path`，那是 task 系工作区加 URDF 限位，不是示教进程里的 `TargetGate`。两套立方体边长都来自 0.9 m 这一档配置，但坐标系和代码路径不同，不要写成同一个对象。

## 6. Nero 局部 IK

`robot_control/nero_ik.py`。只做数学，不连接 SDK。模型用 Pinocchio 从同一份 URDF 加载一次，不加载 mesh。若 URDF 在 `base_link` 前还有固定 world，接口仍返回 \(T_{\text{base}}^{\text{flange}}\)。

每个周期一个 seed：发送前刚读到的七轴反馈。不采样多初始值，失败不解释成全局不可达。

优化变量限制在命令区间内。命令区间是 URDF 限位各向内收 `ik_joint_limit_margin_deg: 2`。反馈允许比 URDF 限位再向外 `ik_feedback_limit_tolerance_deg: 1`，这样 SDK 限位略宽于 URDF 时，IK 还能求一个向内的解。超出这 1° 返回 `IK_JOINT_LIMIT`，并列出关节。只把优化初值投影进命令区间；正则、步长和连续性都相对真实反馈，不相对投影后的点。解不会在验收前被裁剪。

残差交给 SciPy `least_squares`，带解析 Jacobian，`max_nfev=80`。位置、旋转都按验收阈值归一化，所以 2 mm 和 2° 在残差里权重大致相当：

\[
\begin{aligned}
e_p &= (p_{\text{actual}} - p_{\text{target}}) / 0.002, \\
e_R &= \log(R_{\text{target}} R_{\text{actual}}^{\mathsf T})^{\vee} / \mathrm{rad}(2^\circ), \\
e_q &= 0.05\,\mathrm{wrap}(q - q_{\text{seed}}).
\end{aligned}
\]

`wrap` 只用于正则里的角差，不能绕过有限关节的行程。旋转导数是 \(-\mathrm{Jlog}_3(R_{\text{target}} R_{\text{actual}}^{\mathsf T})\,J_\omega\)，不是 SE(3) 对数的导数。线速度和角速度 Jacobian 都旋到 base 轴。

验收，必须同时满足才返回解：

- 有限，且落在向内 2° 的命令区间；
- FK 位置误差 < 2 mm，旋转误差 < 2°；
- 相对真实 seed 的最大关节变化 ≤ `max_joint_delta_rad`（配置 0.3345329252 rad ≈ 19.167°），并且这个直接差的 wrap 也不超过同一限制。Nero 七轴都不是 continuous 关节。

75 ms 是软预算。超时则拿目前最好的中间解做同一套验收，过了才用，并打上 `deadline_fallback`。不过就返回 `IK_FAILED`。离线对比见 `docs/ik_benchmark.md`：在 2298 个历史目标上，Pinocchio + 解析 Jacobian 平均 9.6 ms，旧实现 46.8 ms，验收都是 100%。那次对比用的就是 19.167° 限制。它是模型内误差，不是实机标定精度。

失败码：

| 码 | 含义 | 示教里怎么处理 |
| --- | --- | --- |
| `IK_FAILED`、`IK_FK_ERROR`、`IK_JUMP` | 数值没收敛、精度不够、步长过大 | `IK HOLD`：本周期两侧都不发送、不录制，下个周期用新目标重试。手回到可达区域或按 R 都能恢复 |
| `IK_JOINT_LIMIT`、`IK_NUMERICAL_ERROR` | 反馈越限超过恢复容差，或目标/seed 非法 | 锁存暂停，必须 R |
| 输入过期、目标门越界 | 不是 IK 失败 | 锁存暂停，必须 R |

计算进程在 `teleop/compute.py`。`spawn`，每侧一根同步 pipe，同时最多一个请求。超时等于 `pico4.max_age_s`（0.25 s）。超时是致命的：晚到的解不能变成下一周期的命令。子进程里没有硬件对象。`timing_ik_s` 是求解墙钟，`timing_ik_cpu_s` 是线程 CPU，`timing_compute_ipc_s` 是往返减去求解时间。CPU 时间已经包含在阶段时间里，不能再加一次。

## 7. 手部重定向

权威实现是仓库根目录的 `AnyDexRetarget`（Shiquan Qiu 的 AnyDexRetarget，2025）。`teleop/hand_retarget.py` 只包一层：读 YAML，调用 `Retargeter`，再 `l20_qpos_to_can_slots` 和 `slew_l20_command`。双侧配置：

- 右：`AnyDexRetarget/example/config/adaptive/pico4/pico4_linker_l20.yaml`
- 左：`.../pico4_linker_l20_left.yaml`

优化器类型 `AdaptiveOptimizerAnalytical`。它是 dex-retargeting 这条线上的向量重定向，加上对指时的自适应权重。直接祖先是 DexPilot 的「远时匹配姿态向量、近时改保接触」；AnyTeleop / dex-retargeting 把这件事做成了可换手的优化器。本仓库用的是 AnyDex 的解析梯度版本，不是每帧自动求导。

一次 `retarget(landmarks)`：

1. 再减一次腕，用腕、食指 MCP、中指 MCP 估手掌坐标系，乘 `OPERATOR2MANO_RIGHT` 或 `LEFT`。这一步把输入点变到优化器使用的手掌帧。
2. 额外的固定外旋 XYZ 欧拉角。右手 \((-10^\circ, 0, -10^\circ)\)，左手 \((+10^\circ, 0, +10^\circ)\)。注释说明左右 MediaPipe 的 Y 是镜像的，所以修正也镜像。旋转作用在点上：\(P \leftarrow P R^{\mathsf T}\)。
3. NLopt `LD_SLSQP`，独立关节维，`maxeval=50`，`ftol_abs=1e-4`，边界是 URDF 限位。初值是上一帧解，否则 neutral，否则限位中点。失败就留下初值，不抛给控制环。
4. 关节低通。`lp_alpha: 0.4`，\(y \leftarrow y + 0.4(x-y)\)。R 时 `reset()` 清掉滤波器和 `last_qpos`。
5. 独立关节展开成含 mimic 的全 `qpos`，再按第 2.2 节映射到 20 槽，然后 slew。

损失在厘米下计算。Pico L20 配置的权重：`huber_delta=2.0`（cm），`huber_delta_dir=0.5`，`w_pos=5`，`w_dir=1`，`w_full_hand=1`，`w_contact=6`，`norm_delta=0.04`，`lateral_scaling=0.7`，`pinch_scaling=1.5101`。Huber 是标准 smooth-L1：\(|x|\le\delta\) 时 \(x^2/2\)，否则 \(\delta(|x|-\delta/2)\)。

对指权重。四指指尖到拇指指尖的距离 \(d\)（cm），阈值 `d1=2`、`d2=4`：

\[
\alpha_f = \mathrm{clip}\!\left(\frac{d_2-d_f}{d_2-d_1},\,0,\,\alpha_{\max}\right),\qquad
\alpha_{\text{thumb}}=\max_f \alpha_f.
\]

YAML 写了 `alpha: 1.0`，构造函数用它覆盖类常量 `PINCH_ALPHA_MAX=0.7`。所以现行配置的 \(\alpha_{\max}\) 是 1，不是注释里的 0.7。α 高时，全手姿态项的系数 \((1-\alpha)\) 可以降到 0。不要把 0.7 写成运行值。

每个手指的损失是两项的混合：

\[
L_f = \alpha_f L^{\text{tip}}_f + (1-\alpha_f) L^{\text{full}}_f.
\]

\(L^{\text{tip}}\) 是腕到指尖向量的 Huber（权重 `w_pos`）加上 DIP→tip 方向的 Huber（权重 `w_dir`）。\(L^{\text{full}}\) 是腕到 PIP、DIP、指尖三段向量 Huber 的平均，权重 `w_full_hand`。全手目标按每节骨头单独缩放，不把机器人指根硬锚到人的指根：

\[
\begin{aligned}
p_{\text{MCP}} &= s_{\text{MCP}}\,(p^{\text{human}}_{\text{MCP}}-p_{\text{wrist}}), \\
p_{\text{PIP}} &= p_{\text{MCP}} + s_{\text{PIP}}\,(p^{\text{human}}_{\text{PIP}}-p^{\text{human}}_{\text{MCP}}),
\end{aligned}
\]

DIP、TIP 同样递推。缩放之前，掌的横向轴（默认 axis 1）先乘 `lateral_scaling=0.7`，用来减小侧摆，不改指节长度。四元组是 `[MCP, PIP, DIP, TIP]`，例如右手拇指 `[1.269, 2.361, 1.232, 1.033]`，食指 `[1.753, 1.178, 1.512, 1.092]`。左右 YAML 各自一份。

对指进行时，α 最大的那根手指和拇指的指尖位置目标，会从「分节缩放」混向「整段腕→指尖统一乘 `pinch_scaling`」：

\[
\beta=\min(\alpha/\alpha_{\max},1),\qquad
v \leftarrow (1-\beta) v_{\text{scaled}} + \beta v_{\text{pinch}}.
\]

另有接触项。只对 α 最大的那根非拇指手指：

\[
L_{\text{contact}} = w_{\text{contact}}\,\alpha\,\|p_{\text{thumb tip}}-p_{\text{partner tip}}\|^2.
\]

正则 \(L_{\text{reg}} = 0.04\|q-q_{\text{prev}}\|^2\)，作用在优化变量上。

梯度是解析的：FK 和点 Jacobian 用 Pinocchio 批量算，Huber 对距离的导数乘单位方向，再乘 Jacobian 差。mimic 关节在 FK 前展开，梯度再映射回独立关节。

这和 DexPilot 的关系要写准确：思想是距离自适应的向量重定向和对指接触，实现、解析梯度、L20 的缩放和 20 槽映射是 AnyDex + 本仓库的。我们没有用 DexPilot 的多 RealSense 手套-free 视觉管线，也没有用 AnyTeleop 的统一多手遥操作栈去驱动 Nero。臂是另一套相对腕锚定加七轴 IK，不是手部优化器的一部分。

`input-check` 会在不打开 CAN 的情况下对双侧各跑一遍 retarget，用来看 21 点能不能变成 20 槽。

## 8. 双臂调度和安全状态

主进程在 `teleop/bimanual_controller.py`。两侧 worker 是 `spawn` 进程。每轮一个 `cycle_id`：

1. 主进程在发准备之前取一次共享 RGB-D，并记下 `observation_timestamp_monotonic`。
2. 两侧并行 PREPARE：检查帧新鲜、更新腕目标、目标门只检查不提交、读关节 seed 和手反馈、IK、retarget。
3. 收齐。若任一侧是可重试 IK 失败，向两侧发 cancel。已准备好的手 slew 基准退回本轮发送前的反馈。不 COMMIT。
4. 若任一侧暂停或输入在准备期间过期，共享暂停事件置位，cancel，必须 R。
5. 两侧都 prepared，且准备完输入仍新鲜，才并行 COMMIT。COMMIT 时再查一次帧新鲜和暂停标志，然后该侧 `move_js`，再 `set_joint_positions_raw`，然后读发送后诊断。
6. 主进程收齐同一轮的 RESULT，才追加一行。不把旧的一侧和新的一侧拼起来。

名义 20 Hz。实际周期是较慢一侧加 IPC 和记录。代码不承诺稳定 20 Hz。`[bimanual timing]` 打实测频率、prepare/dispatch 等待和超周期次数。CAN 没有跨设备原子性。COMMIT 途中 SDK 异常会导致整套控制退出，已经发出去的那一侧命令留在硬件上。退出时先协作停 worker，5 s 后还活着再 terminate。不自动失能。

单臂状态机见 `docs/architecture.md`。和双侧的差别只是没有「等另一侧」：可重试 IK 失败同样是 hold 而不是锁存；输入丢失、目标门、非法 IK 输入仍然锁存。

录制队列最多 16 帧 RGB-D。队列满就失败并保留已写部分，不填零、不删行。联合录制的 RGB 先写成 PNG（压缩等级 1），深度是可增长的 uint16 HDF5，LZF，chunk 为一帧。正式 MP4 只在导出时由 LeRobot 编一次 H.264。

诊断里保留腕姿、21 点、目标法兰、力矩、TCP、完整 20 槽、各阶段耗时、相机时间戳和帧号。帧号允许重复，不伪造。`actual_hz` 用真实观测时间算，不用名义 20 Hz 冒充。

## 9. 数据集

导出在 `teleop/lerobot_export.py`，必须是 LeRobot `0.4.4`，否则拒绝。导出前先检查：`control==both`、没有 `recording_error`、状态和动作维一致且有限、手值是 0–255 整数、时间戳严格递增、双侧 `cycle_id` 严格递增、每行都有有效 RGB-D、640×480、深度对齐、`depth_scale_m` 为正。

LeRobot 特征只有三个：

| 键 | 形状 | dtype | 含义 |
| --- | --- | --- | --- |
| `observation.state` | (46,) 或 (23,) | float32 | 发送前反馈 |
| `action` | 同左 | float32 | 本轮下发目标 |
| `observation.images.front` | 存 (480,640,3)，加载后 (3,480,640) | 视频 H.264 / float32 | 该周期选中的 RGB |

`timestamp = frame_index / fps`，fps 是名义 `control_hz`（20）。这不是曝光时间，也不是执行时间。真实时钟在 `diagnostics/episode-XXXXXX.json` 和 `.npz`。

深度 `depth/episode-XXXXXX.h5` 的 `depth` 数据集是 `(N,480,640)` uint16。episode 内 `frame_index=i` 对 `depth[i]`，也对应该样本的 RGB。ACT 训练不读它。

同一目录可以跨进程追加。`codebase_version`、fps、`robot_type`、特征形状和名字、`meta/recording.json` 里的手侧和单位不一致，就拒绝。写盘用目录旁的 `.{name}.lock`。中断会留下 `.recording-in-progress.json`，后续追加会拒绝，直到人检查暂存。暂存目录是数据集旁边的 `.nero_l20_bimanual_act-staging/episode_<UTC>/`。成功路径：`save_episode` → `finalize` → 拷贝深度和诊断 → 用官方 `LeRobotDataset` 重新打开，解码该 episode 首尾帧，确认图像是 `(3,480,640)` → 删标记和暂存。

`--control arm` 或 `hand` 即使配置里有 `lerobot`，也改写到输出目录旁边的 `*_diagnostics/`，不进训练集。

离线形状检查是 `scripts/check_lerobot_act.py`，在 act 环境跑，不连机器人。它用数据集特征构造 `ACTConfig(chunk_size=60, n_action_steps=20, pretrained_backbone_weights=None)`，断言动作维是 23 或 46，取一个 batch 做前向、反向和 `predict_action_chunk`，期望预测形状 `(1, 60, 23)` 或 `(1, 60, 46)`。这只证明数据能喂进这组形状，不读取已训练权重。

写下来的训练命令有两份，不要合成一条。

采集工作区 `docs/dataset_schema.md` 的最小单臂命令。骨干设为 `null`，也就是不加载 ImageNet；没有写 batch、step、学习率，这些就落回 LeRobot 0.4.4 的默认（AdamW `1e-5`，`use_imagenet_stats` 默认 true）。数据集是 `local/nero_l20_act`，23 维。

```text
lerobot-train \
  --dataset.repo_id=local/nero_l20_act \
  --dataset.root=.../data/lerobot/nero_l20_act \
  --policy.type=act \
  --policy.chunk_size=60 \
  --policy.n_action_steps=20 \
  --policy.pretrained_backbone_weights=null \
  --policy.push_to_hub=false
```

`C:\Work\dex-mani\lerobot\ACT_train.md` 的双侧命令。这是后来的一次训练，输出目录是 `outputs/train/act_bimanual_action_1/`，数据集是 `nero_l20_bimanual_act_action_1`，不是采集配置里的默认目录 `nero_l20_bimanual_act`。

```text
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/nero_l20_bimanual_act \
  --dataset.root=.../data/lerobot/nero_l20_bimanual_act_action_1 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=60 \
  --policy.n_action_steps=20 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.optimizer_lr=1e-5 \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --num_workers=2 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=100 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=outputs/train/act_bimanual_action_1/
```

这次显式加载 ImageNet ResNet18，并且 `use_imagenet_stats=false`：图像的 mean/std 用这份数据集自己的统计，不用 ImageNet 的 0.485/0.456/0.406。状态和动作仍按数据集 mean/std。`eval_freq=0` 表示训练循环里不跑环境评估。没有设 `temporal_ensemble_coeff`、`kl_weight`、`chunk` 以外的结构超参，所以那些保持 0.4.4 默认：CVAE 开、`latent_dim=32`、`kl_weight=10`、decoder 1 层、ensemble 关。

Cloth 部署的 `act.policies.fold` 仍指向 `act_bimanual_default_run01/checkpoints/last/pretrained_model`。交接文档（`C:\Work\dex-mani\交接文档.md`，2026-11-02）说该实验的 `last` 指向 `050000`，和「50000 step、每 5000 步存一次」的编号方式一致，但没有把那次的 argv 存进源码树。不要把 `action_1` 这次的 ImageNet 开关和数据目录安到 `default_run01` 上。两份命令共同写死的只有：46 或 23 维由数据集决定、`chunk_size=60`、`n_action_steps=20`、类型 `act`、不上传。

动作块和 `action_is_pad` 由 LeRobot 加载器按 episode 边界生成。检查脚本用 `delta_timestamps = i/fps, i=0..59`。跨 episode 的未来动作会被 pad，不参与 L1。

## 10. ACT 论文里我们实际沿用的部分

论文：Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. arXiv:2304.13705, 2023。本地 PDF：`docs/report/papers/2304.13705.pdf`。算法名叫 Action Chunking with Transformers。硬件平台 ALOHA 不是我们的硬件。

论文要解决的两件事，也是我们用它的原因：

- 高频模仿学习的误差会滚大。一次预测未来 \(k\) 步关节目标，有效决策视野大约缩短到原来的 \(1/k\)。
- 同一观测下人的示范不唯一。用 CVAE 的风格变量 \(z\) 表示这种多模态；测试时 \(z=0\)，退化为确定解码。

论文结构，LeRobot 0.4.4 基本照这个实现（`modeling_act.py` 文件头写明论文和 `tonyzhaozh/act`）：

- 图像：ResNet18，BatchNorm 换成 `FrozenBatchNorm2d`。`replace_final_stride_with_dilation` 默认 false，所以 480×640 的 layer4 是 \(15\times20\times512\)，展平后 300 个视觉 token，加 2D 正弦位置编码。论文用 4 个相机，我们用 1 个。推理端先把 uint8 HWC 除以 255 变成 CHW 的 0–1 浮点，再交给 checkpoint 里保存的 mean/std 归一化。训练若开 `use_imagenet_stats`（库默认 true），视觉统计被换成 ImageNet 的 mean \((0.485,0.456,0.406)\)、std \((0.229,0.224,0.225)\)；`ACT_train.md` 那次双侧训练显式关掉了它。
- CVAE 编码器：BERT 式 transformer。输入是本体、未来动作块、一个 `[CLS]`。`[CLS]` 预测 \(z\) 的对角高斯。训练时从中采样；图像不进这个编码器。测试丢掉它，\(z=0\)。
- 策略本体：transformer encoder 融合相机 token、当前关节和 \(z\)；transformer decoder 用 \(k\) 个可学习查询，交叉注意力读 encoder，一次输出 \(k\) 个动作。这是 DETR 的查询式解码，不是自回归。DETR 是 arXiv:2005.12872，本地已有。LeRobot 把 decoder 层数设为 1，注释说明原 ACT 仓库虽然写了 7 层，但代码 bug 导致只用了第一层，LeRobot 选择匹配实际行为。
- 动作是绝对关节目标，不是增量。论文写过增量动作变差。我们也存绝对目标。
- 重建损失在实现里是 L1。KL 把 \(q(z\mid a_{t:t+k},\bar o_t)\) 拉向 \(\mathcal N(0,I)\)，权重 \(\beta\)。LeRobot 默认 `kl_weight=10`，和论文常用的 \(\beta=10\) 一致。公式：

\[
\begin{aligned}
\mathcal L_{\text{L1}}
&=
\mathrm{mean}\bigl(\| \hat a - a \|_1 \odot (1-\text{pad})\bigr), \\
\mathcal L_{\text{KL}}
&=
\mathrm{mean}\sum_j \Bigl(-\tfrac12\bigl(1+\log\sigma_j^2-\mu_j^2-e^{\log\sigma_j^2}\bigr)\Bigr), \\
\mathcal L
&=
\mathcal L_{\text{L1}} + \beta\,\mathcal L_{\text{KL}}.
\end{aligned}
\]

KL 的写法 LeRobot 指向 Kingma & Welling, Auto-Encoding Variational Bayes, arXiv:1312.6114 附录 B。本地没有这份 PDF。不必再下载，除非起草想单独引用 VAE。

论文推理有两条用法：

- 朴素 chunk：每 \(k\) 步看一次，执行完整个块。
- Temporal ensemble（Algorithm 2）：每步都查询，对「预测同一时刻」的多次输出做 \(w_i=\exp(-m i)\)，\(i=0\) 是最旧的预测，再归一化。论文常用 \(m=0.01\)，并且更信任旧预测。这不是相邻时刻动作的滑动平均。

LeRobot 0.4.4 的默认是朴素队列，不是 ensemble：

- `chunk_size` 默认 100，`n_action_steps` 默认 100。队列空时预测一整块，只把前 `n_action_steps` 步放进队列，其余丢掉。每次 `select_action` 弹出一步。
- `temporal_ensemble_coeff` 默认 `None`。设成非空时，配置校验强制 `n_action_steps==1`，并且每个环境步都跑网络。
- `use_vae=True`，`latent_dim=32`，VAE encoder 4 层，encoder 4 层，decoder 1 层，`dim_model=512`，8 头，FFN 3200，dropout 0.1。
- 归一化默认对视觉、状态、动作都是 mean/std。统计来自数据集，不在采集时手写。
- `pretrained_backbone_weights` 默认 ImageNet `ResNet18_Weights.IMAGENET1K_V1`。我们写进文档的训练命令把它设成 `null`，也就是骨干从随机初始化训练。论文说每个任务 from scratch。不要写成「加载了 ImageNet」。
- 优化器预设 AdamW，学习率 `1e-5`，weight decay `1e-4`，骨干学习率同样 `1e-5`。没有 scheduler。这些是 0.4.4 的默认，仓库命令没有覆盖它们。
- `n_obs_steps` 必须是 1。没有历史图像堆叠。

论文和我们的系统差在这些地方，建议在章里单独成段，避免审稿人或读者把 ALOHA 的数字安到 Nero 上：

| | ALOHA / 论文 | 本系统 |
| --- | --- | --- |
| 遥操作 | WidowX leader 关节直接映射到 ViperX follower，50 Hz | Pico 4 手部追踪；腕增量 + IK；手指优化重定向；名义 20 Hz |
| 臂 | 两侧 6 自由度 + 平行夹爪，动作 14 维 | 两侧 7 自由度 + L20 16 槽，动作 46 维 |
| 动作语义 | leader 关节当 action，follower 关节当观测。力隐含在两者之差 | action 是下发的关节目标，state 是发送前反馈。没有 leader 臂，没有电流/力矩进策略 |
| 相机 | 4 路 480×640 网络摄像头，含两个腕部 | 1 路 L515，对齐后 640×480。深度只存档 |
| 频率和块长 | 论文实验是 50 Hz。正文把块长记成固定的 \(k\) | 名义 20 Hz。写下来的命令都是预测 60 步（3 s）、队列里执行 20 步（1 s）再重新推理 |
| Ensemble | 论文主推每步查询 + 指数加权 | 推理代码走 `select_action` 队列。写下来的训练命令没有打开 ensemble，且 `n_action_steps=20` 与它互斥 |
| 执行器内环 | Dynamixel PID 跟踪目标关节 | Nero `move_js`；L20 raw 位置，另有每周期 ≤10 raw 的主机限幅 |

LeRobot 库论文是 arXiv:2602.22818（ICLR 2026），本地 `docs/report/papers/2602.22818.pdf`。引用库的时候以我们钉死的 0.4.4 源码为准。2602.22818 写的是整库，不要用它替换 0.4.4 的默认超参。

## 11. 实机推理

脚本在 `C:\Work\dex-mani\lerobot\scripts\infer_act_bimanual.py`。LeRobot 版本在该仓库 `pyproject.toml` 里是 `0.4.4`。有两条入口，模型前向是同一段 `policy_worker`。

部署配置 `configs/cloth_agent.yaml`：

```yaml
act:
  python: /home/lab404/anaconda3/envs/act/bin/python
  script: /home/lab404/dex-mani/lerobot/scripts/infer_act_bimanual.py
  device: cuda
  control_hz: 20          # 本仓库执行节拍用的是顶层 control_hz，也是 20
  startup_timeout_s: 120
  inference_timeout_s: 5
  hand_max_raw_step: 10
  hand_feedback_poll_hz: 20
  hand_feedback_max_age_s: 0.40
  policies:
    fold: .../act_bimanual_default_run01/checkpoints/last/pretrained_model
```

`act.control_hz` 没有传给子进程。`ActExecution` 用控制器的 `period = 1/control_hz`。

谁调用：

- 任务正文 `cloth_agent/task/task.md`：摆好毛巾 → 两侧 `move_J(start_joint_positions)` 并确认 `.reached` 和 quiescent → `prepare_act_hands()` → `run_act(policy="fold")` → `observe/show`。
- `robo.run_act` 默认 `steps=500`。20 Hz 下纯节拍是 25 s，还要加上推理时间和结束时的保持确认。首次加载模型算在同一次调用里。cell 默认超时 60 s，更长要操作者改 `--cell-timeout`，agent 不能跨 cell 续播。
- 直接检查脚本 `scripts/act_start_cell.py` 默认 20 步，不走 Codex。

模型进程协议，`cloth_agent/act.py` 的 `ActPolicyClient`。子进程命令：

```text
<act python> infer_act_bimanual.py --rpc-worker --checkpoint <dir> --device cuda
```

字节，和 worker 里的 `--rpc-worker` 分支一致：

1. 加载前把 `config.pretrained_backbone_weights` 改成 `None`。这只是避免 torchvision 再下载 ImageNet；注释写明完整骨干已经在 `model.safetensors` 里。不能据此说训练时没有用 ImageNet 初始化。
2. 检查输入 state `(46,)`、图像 `(3,480,640)`、输出 action `(46,)`。不是这个形状就拒绝，所以这份脚本不能加载 23 维单臂权重。
3. 用全 0 的状态和图像做一次 `select_action` 预热，然后 `policy.reset()`，再写 `READY\n`。预热产生的队列被丢掉，不会变成第一步真实动作。
4. `R` → `policy.reset()` → `OK\n`。`reset` 在 ensemble 关闭时把队列长度上限设成 `n_action_steps` 并清空。
5. `P` 后接 46 个 little-endian float32 状态，再接 `480*640*3` 的 RGB8。Worker 把图像转成 CHW 并除以 255，走 checkpoint 保存的 preprocessor（加 batch、搬到 device、mean/std），调用 `policy.select_action`，再把动作反归一化成 rad 和 0–255 量纲的 float32。
6. 非 `R`/`P` 的命令字节直接报错。Cloth 侧单次读写超时 5 s，启动超时 120 s。

`select_action` 的队列是实装，不是猜测。`temporal_ensemble_coeff is None` 时，队列空才跑网络，取出预测块的前 `n_action_steps` 步放进队列，然后每调用一次弹出一步。队列非空时，这次随 `P` 送来的新 RGB 和新状态被函数读到，但不会进入网络。按训练命令的 20，就是每 20 个控制周期才看一次新图像，中间 19 步执行上次预测的绝对关节目标。每次 `run_act` 开头的 `R` 把队列清掉，所以一段 ACT 的第一步一定会重新推理。不要写成 temporal ensemble，也不要写成每个 20 Hz 周期都前向一次。

同一函数还有 `--worker` 分支，给脚本自己的硬件主循环用：没有 `R`/`P` 前缀，stdin 直接是 state 字节加图像字节。Cloth 不用这条。主循环在 `nero` 环境，按配置里的 `recording.lerobot.python` 拉起 act 子进程。`--offline` 只读数据集做若干步 `select_action`，不打开 CAN。`--execute` 才 `enable_normal_mode` 并 `set_speed`。默认 `--steps 100`。独立脚本的发送前检查只有 `validate_joint_command`（官方限位和 `max_joint_delta_rad`）以及手的 10 raw 限幅；没有 cloth 那条 task 系 TCP 工作区检查。观测变旧则整段抛错，不再发本周期目标。Ctrl+C 停止后续目标，不关使能。

失败、暂停、人工 hold 或 cell 结束会关掉 worker。原因是管道可能停在半个请求上，下次不能接着读。正常跑完则保留进程，同一 policy 下次只 reset，不重新加载。换 policy 名字会关掉重开。

### 11.1 观测和限幅

状态按左到右拼：左 7 关节 rad、左 16 raw、右 7、右 16。关节反馈龄度超过 `camera.max_age_s`（0.25 s），或手反馈超过 `hand_feedback_max_age_s`（0.40 s），这一步失败，不发目标。图像必须是 `(480,640,3)` uint8。推理期间若这帧相机已经老于 0.25 s，也不发。

关节目标先过 Nero 官方限位和 `max_ik_step_rad`（0.3345329252）。再过 cloth `validator._check_path`：URDF 硬限位（这里不用反馈容差）和 task 系 TCP 工作区。两侧都通过以后才发送。顺序是左侧 `move_js` 然后左手，再右侧 `move_js` 然后右手。

手：把动作四舍五入并夹到 0–255，只改 16 个有效槽。限幅基准是上一份 ACT 发出的命令；第一步用当前反馈。每步最多 10 raw。保留槽保持反馈值。

发送后 `steps_executed += 1`。这个计数表示两侧的 SDK 调用都返回了，不是设备 ACK，也不是到位。

### 11.2 起始手型

`TRAINING_START_HANDS` 来自 `act_bimanual_default_run01` 训练集的 episode 36、frame 0，左右取自同一条示范。该帧的臂关节和当前 `start_joint_positions` 一致。不要用数据集统计的中位数拼手型。任务图 7–9 是另一条示范的 episode 0，只说明动作长什么样，和这组手型不是同一帧。

16 个数，槽位顺序就是第 2.3 节的有效槽：

```text
left:  [3, 122, 158, 150, 130, 244, 253, 171, 126, 12, 253, 176, 224, 208, 214, 231]
right: [58, 112, 146, 139, 100, 226, 254, 160, 95, 1, 254, 173, 230, 222, 232, 246]
```

`prepare_act_hands()` 要求双臂已经停稳。它不移动臂，不加载模型。按 `hand_max_raw_step` 渐近这个目标，周期是控制周期。发完再等最多 5 s，要求新的一次手部查询（查询开始时间晚于发送）在 16 槽上的最大误差 ≤ 10 raw。超时抛 `GripperTimeout`。返回值里 `delivery_confirmed` 仍是 false：查询到的数值接近目标，不是逐帧硬件 ACK。`run_act` 不会自动做这件事。`open_gripper` 只动拇指和食指，不能用来准备 ACT。

### 11.3 结束和失败

固定步数走完后，两侧用当前关节反馈做位置保持，而不是把模型最后一个目标当成已经到位。最多再等 3 s 确认 quiescent。`completed` 只表示步数完成且保持已确认。

`paused` / `manual_hold`：不再发新目标，不松手，不失能。`failed`：已经发出的动作不回滚。相机丢帧、状态过期、关节或工作区拒绝、管道超时，都是失败。控制进程退出时若线程还在，会杀掉 GPU 子进程，避免留下脱离控制的模型。

日志：`act-worker.log`。每步下发记入 `control-samples.jsonl`，带目标关节、20 槽手目标和完整 action。

### 11.4 没有接到 cloth 上的 DINOv3 变体

`C:\Work\dex-mani\lerobot\scripts\train_act_dinov3.md` 和 `infer_act_dinov3_bimanual.py` 是后做的对照实验。Cloth 的 `act.policies.fold` 不指向它，`infer_act_bimanual.py` 也不能加载它的权重。报告正文若只写现行系统，用一句话带过即可，不要把它的主干写成 ResNet 那次部署。

它保留 CVAE、L1+KL、60/20 的块长，换掉视觉主干：冻结的 DINOv3 ViT-S/16（本地权重 `pretrained/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`，SHA256 `08c60483…b941d`）。480×640 得到 30×40 个 patch，2×2 平均池化成 15×20，不用 CLS。图像用 ImageNet 归一化，和状态/动作的数据集统计分开。训练增强只动亮度、对比度、饱和度（0.9–1.1），无翻转、旋转、平移。标准化后的状态加 \(\mathcal N(0, 0.01^2)\)，只在训练时加，动作标签不加。AdamW，学习率 `1e-5`，batch 8，50000 step，每 5000 步保存。默认数据仍是 `nero_l20_bimanual_act_action_1`。README 写明这个入口还没有做实机运动验证。

## 12. 和前后章节怎么接

示教章的结尾应落到：人的手没有进训练向量；进训练的是机器人自己的发送前状态和将要执行的关节目标，加上一张前向 RGB。深度和 20 槽诊断是附件。

ACT 章的结尾应落到：策略在关节空间里开环一小段、再看新图像；它不规划 TCP，也不知道毛巾的关键点。起始条件（布在垫子中央、臂在记录关节、手在 episode 36 frame 0）由前一个模块准备。跑完之后由前一个模块看图决定是否结束。`agent_harness.tex` 的第一段就是这个衔接，不要另起一套任务定义。

任务本身，若需要一句背景：先把毛巾对折成双层长方形并放到黑色垫子中央，朝向与训练起始帧一致；ACT 负责后续折叠。任务步骤的细节在 `cloth_agent/task/task.md`，不属于这两章的实现正文。

## 13. 论文和本地 PDF

下列 PDF 已经在 `docs/report/papers/`。示教和 ACT 会用到的是前五份。后一份是手部模仿学习的相邻工作，AnyDex 的 README 没有直接引用它；dex-retargeting 的谱系里有这篇。不要把它写成我们实现的遥操作方法。

| 文件 | 文献 | 和这两章的关系 |
| --- | --- | --- |
| `2304.13705.pdf` | Zhao et al., 2023, arXiv:2304.13705 | ACT 算法原文 |
| `2602.22818.pdf` | Cadene et al., LeRobot, arXiv:2602.22818, ICLR 2026 | 我们使用的库。实现细节以 0.4.4 为准 |
| `2005.12872.pdf` | Carion et al., DETR, arXiv:2005.12872 | ACT decoder 的查询式 transformer 来源 |
| `1910.03135.pdf` | Handa et al., DexPilot, arXiv:1910.03135 | 视觉遥操作和对指感知重定向的前身。AnyDex README 致谢了它 |
| `2307.04577.pdf` | Qin et al., AnyTeleop, arXiv:2307.04577 | 通用视觉手部遥操作。AnyDex 写明建在 dex-retargeting 上，而 dex-retargeting 来自这条线 |
| `2204.12490.pdf` | Qin et al., From One Hand to Multiple Hands, arXiv:2204.12490 | 单相机人手示范再换到机器人手。不是本仓库的采集方法 |

没有单独论文、但必须引用的软件：

- AnyDexRetarget，Shiquan Qiu，2025，https://github.com/qqsq12321/AnyDexRetarget 。优化器、Pico 26→21、L20 YAML 都在这里。
- dex-retargeting，https://github.com/dexsuite/dex-retargeting 。AnyDex 的算法前身，本仓库不直接导入它。
- 原始 ACT 代码 https://github.com/tonyzhaozh/act 。我们跑的是 LeRobot 里的移植，不是这个仓库。
- Pico 坐标依据 XRoboToolkit 的 `TrackingData.cs` 和 `PXR_HandTracking.cs`。README 里有链接。

建议 BibTeX 至少包括 Zhao 2023、Cadene 2026、Handa 2019、Qin AnyTeleop 2023、Carion 2020，以及 AnyDexRetarget 软件引用（`AnyDexRetarget/README.zh.md` 末尾已有）。

DexPilot 不要写成我们的系统：它用多台 RealSense 和 GPU 检测裸手，驱动的是 Allegro 那一类平台。我们取的是「靠近时加强对指、远离时匹配手形」这一损失结构，检测和臂的控制是 Pico 追踪加 Nero IK。

## 14. 建议的章节骨架

和 `agent_harness.tex` 对齐，示教、ACT 各一节即可。下面是可以装进内容的小节，不是要逐条都画一张 TikZ。

示教：

1. 这一节采集什么。一句说明 VR 手不进训练集，关节目标和发送前反馈才进。
2. 硬件和 46 维排列。一张小表，槽位和单位。
3. 从 Pico 到法兰的坐标和相对锚定。两张矩阵和 EMA 公式。
4. 七轴 IK。残差、2 mm / 2°、2° 内缩、1° 恢复、19.167° 步长、IK HOLD 和锁存暂停的区别。
5. 手指重定向。α 混合、接触项、解析 SLSQP、raw 映射和 slew。说明 α 上限按 YAML 是 1。
6. 双侧调度。PREPARE/COMMIT、非原子 CAN、20 Hz 只是调度目标。
7. 相机和 LeRobot 包。PNG 暂存、H.264 一次编码、深度旁路、失败不填零。

ACT：

1. 为什么用动作块，而不是单步行为克隆。用论文的 compounding error 和示范多模态，不要复述 ALOHA 的六个任务成绩，除非明确标成论文结果。
2. 网络：ResNet18、CVAE、DETR 式 decoder、L1+βKL、\(z=0\)。给出我们图像分辨率和 46 维，对照论文的 4 相机和 14 维。
3. 训练接口：60 / 20、L1+βKL、pad。把单臂 `pretrained=null` 的最小命令、双侧 `action_1` 的 ImageNet 初始化命令、以及 cloth 实际加载的 `default_run01` 分成三件事。
4. 实机闭环：`select_action` 每 20 步才看一次新图像、字节协议、每步限幅、起始手型、cloth 默认 500 步、保持和 `completed` 的含义。
5. 交给 Agent Harness 的边界。

图如果要画，示教一张数据流（Pico、两个计算进程、两个 CAN、L515、LeRobot），ACT 一张「队列空时才把当前 RGB 和状态送进网络，弹出一步 46 维动作，经限幅后 `move_js`」。不要画成每个控制周期都前向一次，也不要把 cloth agent 的 Codex 图重画进来。

## 15. 源文件索引

示教：

- `teleop/bimanual_controller.py`：双侧主循环和 worker 协议。
- `teleop/controller.py`：单臂循环、R/B/S/Q、起始姿态、TCP 立方体。`RETRYABLE_IK_CODES`。
- `teleop/wrist_tracker.py`、`teleop/coordinate_frames.py`、`teleop/target_gate.py`。
- `teleop/compute.py`、`teleop/hand_retarget.py`、`teleop/hand_feedback.py`。
- `teleop/realsense_camera.py`、`teleop/recording.py`、`teleop/lerobot_export.py`。
- `teleop/config.py`：配置校验和单侧选择。
- `configs/pico4_bimanual_nero_l20.yaml`：现行双侧参数。
- `robot_control/nero_ik.py`、`robot_control/nero.py` 的 `move_js` 与 `enable_normal_mode`。
- `robot_control/l20.py`：20 槽名字、有效索引、开合预设。
- `AnyDexRetarget/anydexretarget/optimizer/analytical_optimizer.py`：损失和 α。
- `AnyDexRetarget/anydexretarget/optimizer/base_optimizer.py`：SLSQP、全手向量缩放。
- `AnyDexRetarget/anydexretarget/linker_l20.py`：角度到 raw。
- `AnyDexRetarget/example/input/pico4.py`：26→21 和腕坐标。
- `docs/architecture.md`、`docs/dataset_schema.md`、`docs/ik_benchmark.md`、`README.md`。

ACT：

- `cloth_agent/act.py`：协议、限幅、起始手型、执行线程。
- `cloth_agent/robo.py`：`prepare_act_hands`、`run_act` 默认 500。
- `cloth_agent/bimanual.py`：ACT 独占设备、取消。
- `cloth_agent/planning.py` 的 `_check_path`：ACT 关节目标的工作区检查。
- `cloth_agent/hardware.py`：`NeroArm(..., max_joint_delta=max_ik_step_rad)`。
- `configs/cloth_agent.yaml` 的 `act:` 段。
- `cloth_agent/robot_operations.md`、`cloth_agent/task/task.md`。
- `scripts/check_lerobot_act.py`、`scripts/act_start_cell.py`。

LeRobot 0.4.4，本地路径 `C:\Work\dex-mani\lerobot`：

- `scripts/infer_act_bimanual.py`：`READY` / `R` / `P`、`/255`、`select_action`、独立硬件环。
- `ACT_train.md`：双侧 `action_1` 的 50000 step 训练命令。
- `交接文档.md`：`default_run01` 的 `last` → `050000`；库版本和环境约束。
- `src/lerobot/policies/act/configuration_act.py`、`modeling_act.py`。
- `src/lerobot/datasets/factory.py` 的 `IMAGENET_STATS` 和 `use_imagenet_stats`。
- `scripts/train_act_dinov3.md`：未接入 cloth 的对照实验。

标签页：https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/modeling_act.py
