# NERO Tracker → 七轴 IK → move_js 调试版

## 交付范围

新增 `robot_control/nero_ik.py`、`teleop_nero_tracker_move_js_debug.py`、`tests/test_nero_ik_debug.py`、本说明和 `docs/superpowers/plans/2026-09-09-nero-js-ik.md`。

没有修改任何旧程序、`config.py`、SDK 或 URDF；没有新增依赖、连接 CAN、启动 SteamVR 或运行机械臂。旧 JS、pose_debug、键盘和测试脚本保持原状。

注意：项目现有 `teleop_nero_tracker_move_js.py` 仍引用不存在的 `teleop_nero_tracker`，其旧诊断测试也因此无法导入。这是已有问题，本次按“不覆盖旧文件”要求保留。新入口直接引用存在的 `teleop_nero_tracker_move_pose_debug.py`。

## 实际 API 与模型

- `NeroArm.move_js(self, joints: Any) -> None`，七轴数组 `[q1,...,q7]`，shape `(7,)`，单位 rad；没有 `speed_percent`，JS 没有轨迹规划。
- `get_joint_positions()` 返回七轴 rad；内部依次读 `get_motor_states(1..7).msg.position`。新入口通过公共 `get_raw_motor_states()` 读取同一组数据，同时检查每个消息的时间戳。
- `get_flange_pose()` 返回 `[x,y,z,roll,pitch,yaw]`，单位 m/rad。新控制链完全不读 TCP、不加工具长度。
- 状态使用 `get_arm_status()`、`get_raw_arm_status()`，七轴使能沿用已存在的 `_driver.get_joint_enable_status(i)`；低速帧新鲜度读取真实存在的 `_driver.get_driver_states(i)`。
- 包装支持 `enable()`、`disable()`、`reset()`、`emergency_stop()`、`is_enabled()`、`is_ok()`。本入口仅人工确认后的启动阶段调用 `enable()` 和 `set_normal_mode()`；异常不自动调用恢复接口。
- 固定 V111 状态：`arm_status=0` 正常、1 电子急停、2 无解。控制模式要求 `ctrl_mode=1`（CAN），3 是以太网；其余原始字段全部记录，不凭返回值假设运动已接受或到位。

模型使用项目相对路径：

```text
agilex_teleop/pyAgxArm/asserts/agx_arm_urdf/nero/urdf/nero_description.urdf
```

`base_link → link1 → … → link7`，七个旋转关节依次 `joint1` … `joint7`，末端采用 `link7`。该 URDF 没有单独命名的 flange link。每个旋转轴在局部坐标系均为 Z。运行时再次核对 SDK `joint_names` 顺序，并将 URDF 和 SDK 限位取交集。

| 关节（URDF=SDK 顺序） | URDF 下限 rad | URDF 上限 rad | SDK 下限 rad | SDK 上限 rad |
|---|---:|---:|---:|---:|
| joint1 | -2.70526 | 2.70526 | -2.705261 | 2.705261 |
| joint2 | -1.74 | 1.74 | -1.74533 | 1.74533 |
| joint3 | -2.75 | 2.75 | -2.757621 | 2.757621 |
| joint4 | -1.01 | 2.14 | -1.012291 | 2.146755 |
| joint5 | -2.75 | 2.75 | -2.757621 | 2.757621 |
| joint6 | -0.73 | 0.95 | -0.733039 | 0.959932 |
| joint7 | -1.5707963 | 1.5707963 | -1.570797 | 1.570797 |

零关节角时的位置约为 `[0,-0.0235,0.71801] m`，旋转矩阵约为 `[[0,-1,0],[0,0,-1],[1,0,0]]`，**零位姿不是单位矩阵**。历史关节/法兰日志的离线 FK 对比支持这一末端定义，但不能替代实机标定；启动和 R 对齐仍检查实际法兰与模型 FK。

本机 nero 环境实际导入 `/home/lab404/anaconda3/envs/nero/lib/python3.10/site-packages/pyAgxArm/__init__.py`，配置创建与七轴限位交集已离线验证；未发现当前入口的 SDK 同名遮蔽。没有删除两个同名仓库目录，也没有把它们加入 `sys.path`。运行时记录真实 SDK 路径便于换电脑核对。

## 旧 IK 的问题与新实现

旧版采用 XML 手动 FK + SciPy `least_squares`，七变量、单初值，最多40次评估。**它已经使用实际七轴反馈作 seed，也已有弱正则、限位、有限性、FK 和5°连续性检查**，不能说旧版每帧固定零位或完全没有安全检查。

实际限制是：搜索域提前缩到实测关节±5°，单初值容易局部失败；每帧重新求解、没有多候选筛选；失败/目标超限后参考点停止推进；频繁诊断输出可能占用循环时间；旧模块现已无法导入。历史 move_pose 控制器 `arm_status=2` 不能直接证明本地 IK 发生肘部翻转，也没有证据证明是 SDK/URDF 顺序错误。速度、奇异性、模型误差和控制器行为仍须看新日志，不保证更换 IK 就解决硬件问题。

新 IK 仍只依赖现有 NumPy/SciPy：

1. 目标为 `T_base_flange`，严格检查4×4齐次矩阵、七轴有限 seed。
2. 优先实际反馈 seed；真实反馈缺失时只允许上一合法 IK 作为诊断 seed，禁止本周期发送；没有历史解则不求解。
3. 有界优化搜索完整合法关节范围，残差包含位置、相对旋转向量和等权 seed 正则项。三个确定性初值：实际 seed、七轴交替±2°扰动及反向扰动；正则参考始终是同一个 seed。
4. 只接受优化器成功收敛的候选，再做 FK、限位、10°分支检查；在取得的合法候选中按 `norm(wrap(q-seed))` 最小选解。预算内未找到解不意味着数学上全局不可达。
5. 七轴全部包含正则和检查，q7不例外。精确相同的目标和 seed 直接保持 seed，避免静止目标的数值零空间漂移；这不是2mm/2°死区。
6. 数学角差处理 ±π，但 Nero 七轴均为有限转角关节。实际命令仍检查直接路径，禁止通过“短角”穿越关节限位。例如179°→-179°本来就不是这些关节的合法区间。
7. 相对实际反馈及上一合法 IK 检查分支；近似固定目标下累计关节变化超过1°时报告疑似冗余漂移并暂停等R。这是保守诊断，不是已证实的物理原因。
8. `q_ik` 相对上一成功发送值逐轴限幅到5°，得到 `q_proposed`；再次检查与实际反馈的距离和限位。发送前还要刷新反馈、核验官方 SDK 限位，最后唯一运动接口是 `arm.move_js(q_cmd)`。

限幅后的 `q_cmd` 可能尚未到达目标：日志分别记录候选 FK 与命令 FK，不能用候选解的精度冒充实际到位精度。

## 完整映射与数据流

```text
SteamVR 有效 Tracker 矩阵
  → 世界系相对初始锚点增量
  → 现有 R_base_tracker 与 POSITION_SCALE 映射
  → T_base_flange_target
  → Tracker/目标跳变门控
  → 实测七轴 seed → 有界多初值 IK
  → 有限性/限位/FK/分支/冗余漂移检查
  → 相对上一发送值七轴限幅
  → 刷新实际反馈、状态、官方限位
  → move_js（rad）
```

完全复用现有 `tracker_target()`，没有更换左右乘约定：

```text
p_target = p_flange_start + scale * Rmap * (p_tracker - p_tracker_start)
R_target = Rmap * (R_tracker * R_tracker_start.T) * Rmap.T * R_flange_start
```

当前继承 scale=0.6、`Rmap=[[1,0,0],[0,0,-1],[0,1,0]]`。保留原 `BASE_MAPPING_CALIBRATED` 检查；该布尔值不是自动标定证明。修改标定仍在旧 pose_debug 文件，不在新文件重复维护。原文件没有经过验证的完整笛卡尔工作空间边界，本次不虚构；局部步长、限位和IK可解也不能排除连杆碰撞。

## 参数与安全边界

| 检查 | 默认值/行为 |
|---|---|
| IK FK 位置/姿态 | 严格小于2mm / 2° |
| IK 分支变化 | 每轴≤10°，相对实际反馈及上一合法 IK |
| 命令单周期变化 | 每轴≤5°，相对上一成功发送值；与最新实测值也必须≤5° |
| Tracker 单次跳变 | ≤50mm / 10°（由原30mm目标步长除scale=0.6得出） |
| 法兰目标步长 | ≤30mm / 10°，继承原版；超限锁定等R |
| 控制频率 | 10Hz；超时不补发追赶 |
| IK 每候选评估/总软预算 | 80次 / 75ms，不能保证实时调度 |
| IK 正则权重/初值扰动 | 0.05 / 七轴交替±2° |
| 近似固定目标 | 相对诊断锚点≤0.1mm / 0.1° |
| 固定目标下七轴累计漂移 | >1°拒绝，疑似冗余变化，等R |
| 机器人消息年龄/组成帧跨度 | ≤250ms / ≤100ms，采用真实CAN接收时间戳 |
| Tracker 本地读取耗时 | ≤100ms，否则无效帧 |
| 本地取得 Tracker 至发送前 | ≤250ms，否则拒绝 |
| 初始对齐等待 | 最多5秒 |
| 启动/重新对齐模型一致性 | ≤10mm / 5°；仅防明显坐标系错误，不是到位精度 |
| 正常终端日志 | 每0.5秒；异常立即输出；CSV每周期flush |

特别注意：Tracker API 不提供源样本时间戳，因此 `tracker_age` 留空；`tracker_local_age_s` 只是本地取得矩阵后的耗时，**无法证明SteamVR源数据没有内部缓存或停更**。不能通过“位置不变”认定过期，否则会错误拒绝静止设备。

V111 的 `get_flange_pose().timestamp` 仅代表最后一个组成帧，本入口按已核对的 `_parser.end_pose_xy/end_pose_zrx/end_pose_ryrz` 检查三帧时间戳。这个窄适配依赖V111形状，未知SDK形状会拒绝对齐而不是悄悄忽略检查。

Tracker 丢失、跳变、目标跳变、关节反馈缺失或疑似静止目标冗余漂移会暂停，恢复后保持静止按R重新对齐。R要求有效Tracker、新鲜关节/法兰、正常控制器、`motion_status=0` 和FK一致性，本周期不发送。**motion_status=0不是物理停稳证明，需要操作者观察。**

机器人状态异常、SDK发送异常锁定后，R也不能恢复；退出后人工排查。单次数值IK失败/FK失败/命令超限仅跳过本周期，不发送零位、不重放旧解。

Ctrl+C及退出仅断开，不调用disable/reset/急停/回零，不主动改变使能。**这不保证机械臂保持使能、不保证停稳，也不保证防坠落；停止发送不取消在途目标。** 必须准备机械支撑和硬件安全措施，危险时使用硬件急停，不能把Python脚本当成安全控制器。

## 运行方法

在项目目录执行：

```bash
conda activate nero
python -m robot_control.nero_ik
python teleop_nero_tracker_move_js_debug.py --ik-test
python -m unittest discover -s tests -p test_nero_ik_debug.py -v
```

以上全部离线，不初始化CAN/SteamVR。自检采用非零七轴 `[.2,-.3,.15,1,-.1,.2,-.4] rad`，FK→同seed IK→FK，打印J1至J7误差。它证明数学自洽，不能单独证明URDF与硬件一致。

在已核实硬件安全、CAN就绪、控制器无故障、Tracker与外参正确后，交互终端运行：

```bash
python teleop_nero_tracker_move_js_debug.py
```

输入 `MOVE` 才继续启动使能和对齐。保持Tracker和机械臂静止完成对齐，再缓慢移动Tracker。需要重新对齐时保持静止按R。可用 `--urdf /实际路径/nero_description.urdf` 覆盖模型路径，但不得使用未经核验的模型驱动真机。

## 日志与“微动+急停”排查

日志位于项目 `teleop_debug_logs/nero_tracker_YYYYMMDD_HHMMSS_微秒.csv`，同时有同名 `.log` 保存启动及异常信息。CSV所有 `q_*`、`dq_*` 都为 **deg**；xyz为m，roll/pitch/yaw为rad；带 `_mm`、`_deg`、`_s` 的字段单位如名。缺失值留空，不伪装为零。

重点依次查看：

1. `code`、`sent`、`sent_count`：问题发生前究竟有没有本次调用？`q_proposed`只是待发送值，`q_cmd`仅调用成功返回后填写；异常可能发生在部分CAN帧已发出之后，因此返回失败也不能证明硬件没收到任何内容。
2. `tracker_valid`、`delta_tracker_position_mm/rotation_deg`、`tracker_read_ms`：Tracker丢失或突变。
3. `target_*`、`delta_target_position_mm/rotation_deg`：目标映射/累计目标门控。
4. `seed_source`、`joint_age_s`、`joint_skew_s`、`q_actual_1..7`：是否新鲜实际反馈，还是只做降级诊断。
5. `ik_detail`：各候选是否收敛、次数、FK误差与七轴变化；`IK_FAILED`/预算耗尽不等于全局不可达。
6. `q_ik_1..7`、`dq_ik_1..7`、`dq_ik_max_deg`、`possible_redundancy_flip`、`dq_anchor_1..7`：特别关注q3/q4/q5/q7。`IK_REDUNDANCY_FLIP`表示近似固定目标下累计姿态漂移疑似异常，不宣称已证明翻肘。
7. `q_proposed_*`、`q_cmd_*`、`dq_cmd_*`、`command_clipped`：真正请求的单步幅度；`command_fk_*`与`fk_*`必须分开看。
8. `robot_state`、`error_code`、`enabled`、`emergency_stop`、`pre_send_feedback`、`post_send_state`：区分先发生控制器故障还是先发生发送异常。返回和反馈是异步的，不能把下一帧故障必然归给最后一条目标。

本脚本不会自动请求电子急停，所以若本次日志出现`arm_status=1`，应检查控制器、外部进程及硬件；不能再把它归因于本脚本的退出急停调用。

## 验证记录与局限

nero环境30项新增离线测试通过；两个IK自检入口、`--help`、Python语法编译通过。另做20组非零姿态附近的数学目标求解，20/20找到合格解，本机当次耗时约57–72ms；不是所有姿态的成功率或实时性保证。

旧 `tests/test_teleop_diagnostics.py` 仍因缺少 `teleop_nero_tracker` 导入失败，未宣称全项目测试通过。没有安装pytest，也没有跑可能涉及硬件的完整测试集。

本实现没有碰撞检测、经过认证的速度/加速度约束、制动保持功能或硬件安全回路。五度/周期是沿用旧脚本的上限，不是对当前负载与环境的安全承诺。真机效果和“微动+急停”的最终根因仍需要你在受控安全条件下取得新日志。
