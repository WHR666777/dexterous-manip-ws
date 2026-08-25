# Nero + LinkerHand L20 研究控制封装设计

## 1. 目标

在 AgileX Nero 官方 `pyAgxArm` SDK 与 LinkerHand L20 官方 Python SDK 之上增加轻量研究封装，使硬件通信与 DP、ACT、VLA、teleoperation 和 demonstration 数据采集解耦。

分层固定为：

```text
Official SDK -> Research Wrapper -> Policy / Teleoperation / Data Collection
```

本阶段不实现 CAN 编解码、IK、MIT/CPV 控制、ROS、相机、DP、ACT 或 VLA。

## 2. 已核查的官方版本

- `agilexrobotics/pyAgxArm`：提交 `8cd90f9106219a156c3c0d7e58ee36d838a89baf`
- `linker-bot/linkerhand-python-sdk`：提交 `0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8`
- Nero 实机固件：`v1.11`
- Nero Driver 固定选择：`NeroFW.V111`
- L20 通信：CAN，bitrate `1000000`

两个官方仓库作为项目根目录下的独立源码 checkout 保持原样，研究封装不得修改其核心源码。

## 3. 项目结构

```text
project_root/
├── pyAgxArm/
├── linkerhand-python-sdk/
├── robot_control/
│   ├── __init__.py
│   ├── nero.py
│   ├── l20.py
│   └── robot_system.py
├── examples/
│   ├── nero_example.py
│   ├── l20_example.py
│   └── nero_l20_example.py
├── tests/
│   ├── test_nero.py
│   ├── test_l20.py
│   └── test_robot_system.py
├── config.py
├── requirements.txt
└── README.md
```

除测试和三个示例外，不增加额外框架层。运行前自检作为 `RobotSystem.self_check()` 提供，不新增独立 preflight 程序。

## 4. SDK 加载策略

### 4.1 Nero

Nero SDK 支持本地安装。README 要求使用：

```bash
python3 -m pip install -e ./pyAgxArm
```

Wrapper 通过公开根包导入：

```python
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
```

### 4.2 L20

LinkerHand 仓库没有 `setup.py` 或 `pyproject.toml`，不强制改造成 editable package。`robot_control.l20` 在导入官方 API 时按以下顺序处理：

1. 尝试正常导入 `LinkerHand.linker_hand_api.LinkerHandApi`；
2. 若失败，检查项目根目录下的 `linkerhand-python-sdk`；
3. 仅把该目录加入 `sys.path` 后重试；
4. 仍失败时抛出包含安装路径建议的 `ImportError`。

SDK 导入在 `connect()` 时执行，导入 Wrapper 本身不应连接硬件。

## 5. 配置

`config.py` 只保存运行所需配置：

```python
NERO_CAN_INTERFACE = "socketcan"
NERO_CAN_CHANNEL = "can0"
NERO_FIRMWARE = "1.11"
NERO_MAX_JOINT_DELTA = None
NERO_SPEED_PERCENT = 10

L20_HAND_TYPE = "right"
L20_HAND_MODEL = "L20"
L20_CAN_CHANNEL = "can1"

CONTROL_HZ = 20
```

`NERO_MAX_JOINT_DELTA` 单位为 rad。`None` 表示未启用额外的单步变化限制；Wrapper 仍始终执行官方关节限位检查。项目配置不得保存 sudo 密码。

## 6. NeroArm

### 6.1 初始化

底层 Driver 使用：

```python
create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.V111,
    interface=can_interface,
    channel=can_channel,
)
```

官方参数名中的 `firmeware_version` 拼写保持不变。Wrapper 不允许把固件选择静默降级为 `NeroFW.DEFAULT`。

Driver 创建后立即调用 `set_joint_limits_enabled(True)`。Wrapper 从 `driver.get_config()["joint_limits"]` 读取 7 轴限位，不自行另建未经来源验证的机械限位表。

### 6.2 生命周期

公开接口：

```python
connect() -> None
disconnect() -> None
enable(timeout: float = 5.0, poll_interval: float = 0.05) -> None
disable(timeout: float = 5.0, poll_interval: float = 0.05) -> None
reset() -> None
emergency_stop() -> None
is_connected() -> bool
is_enabled() -> bool
is_ok() -> bool
```

v1.11 官方 `enable()`/`disable()` 不接受 timeout。Wrapper 使用 `time.monotonic()` 有限重试，超时抛出 `TimeoutError`。负 timeout 或非正 poll interval 抛出 `ValueError`。

`emergency_stop()` 映射 `electronic_emergency_stop()`。`reset()` 不自动调用急停。README 明确说明 `disable()` 和急停后的 `reset()` 都可能导致机械臂下落。

`disconnect()` 保持幂等，不隐式 disable，因为自动失能本身可能引发机械臂下落。示例在明确安全的 `finally` 流程中由用户选择 disable 后再 disconnect。

### 6.3 状态

公开状态接口：

```python
get_joint_positions() -> np.ndarray       # (7,), rad
get_joint_torques() -> np.ndarray         # (7,), N*m
get_flange_pose() -> np.ndarray           # (6,), [m, m, m, rad, rad, rad]
get_tcp_pose() -> np.ndarray              # (6,), [m, m, m, rad, rad, rad]
get_firmware() -> dict[str, object]
get_arm_status() -> dict[str, object]
get_observation() -> dict[str, object]
get_state_vector() -> np.ndarray          # (7,), q only
```

Nero v1.11 的 `get_motor_states().msg.velocity` 被官方 V111 Driver 强制置零，因此不提供 `get_joint_velocities()`，也不通过差分估计生成替代值。

底层反馈未就绪时抛出 `RuntimeError`，不返回全零数组。数组统一复制为 `np.float64`，避免上层意外修改 SDK 缓存。

`get_firmware()` 映射官方同名方法并返回普通 dict，供 `self_check()` 对照固定的 `NeroFW.V111` 配置；它不根据返回值自动切换 Wrapper 固件实现。`get_arm_status()` 返回普通 dict，至少包含 SDK 实际状态字段：`ctrl_mode`、`arm_status`、`mode_feedback`、`teach_status`、`motion_status`、`trajectory_num`。错误状态按 SDK 对象可用的公开属性转换为嵌套 dict；不把 `MessageAbstract` 暴露给上层。

为调试保留：

```python
get_raw_joint_positions()
get_raw_motor_states()
get_raw_flange_pose()
get_raw_arm_status()
```

### 6.4 Observation

```python
{
    "joint_position": np.ndarray(shape=(7,)),
    "joint_torque": np.ndarray(shape=(7,)),
    "tcp_pose": np.ndarray(shape=(6,)),
    "timestamp": float,
}
```

`timestamp` 是 Wrapper 完成该次读取时的 Unix wall-clock seconds，由 `time.time()` 产生。原始 SDK timestamp 仅可通过 raw 接口检查。

`get_state_vector()` 固定只拼接 `joint_position`，shape `(7,)`，单位 rad。v1.11 不加入伪造的 `dq`。

### 6.5 运动和安全验证

公开接口：

```python
validate_joint_command(joints, *, max_joint_delta=None) -> np.ndarray
move_joints(joints, *, speed_percent=None) -> None
command_joint_positions(joints) -> None
move_pose(pose, *, speed_percent=None) -> None
move_linear(pose, *, speed_percent=None) -> None
```

所有命令要求已 connect 且 7 轴均 enabled。输入必须是一维、长度正确、数值 finite。

`validate_joint_command()`：

1. 转换为 `np.float64`；
2. 检查 shape `(7,)`；
3. 检查 finite；
4. 使用 SDK config 中的 rad 限位逐轴检查，越界直接抛出 `ValueError`，不截断；
5. 若启用 `max_joint_delta`，读取当前 q 并检查逐轴绝对差值。

`speed_percent` 是官方 `set_speed_percent()` 的整数百分比，范围 `[0, 100]`。它不是 rad/s。

`command_joint_positions()` 调用官方 `move_j()` 一次，不等待运动完成。官方文档明确允许连续 `move_j()` 覆盖上一目标；文档未承诺安全控制频率，10/20/30/50 Hz 必须真机逐级验证。

`move_pose()` 映射 `move_p()`；`move_linear()` 映射 `move_l()`。README 明确说明官方禁止把 `move_l()` 用作连续目标流。

## 7. LinkerHandL20

### 7.1 固定型号与位置顺序

Wrapper 只支持 L20，不引入其他型号抽象。`L20_JOINT_NAMES` 固定为：

```text
0  thumb_base
1  index_base
2  middle_base
3  ring_base
4  little_base
5  thumb_abduction
6  index_abduction
7  middle_abduction
8  ring_abduction
9  little_abduction
10 thumb_roll
11 reserved_11
12 reserved_12
13 reserved_13
14 reserved_14
15 thumb_tip
16 index_tip
17 middle_tip
18 ring_tip
19 little_tip
```

`L20_ACTIVE_POSITION_INDICES` 为 `(0, 1, ..., 10, 15, 16, 17, 18, 19)`。reserved 维度仍保留在所有 20 维 action 和 state 中。

### 7.2 生命周期

公开接口：

```python
connect() -> None
disconnect() -> None
is_connected() -> bool
```

`connect()` 延迟构造：

```python
LinkerHandApi(
    hand_type=hand_type,
    hand_joint="L20",
    modbus="None",
    can=can_channel,
)
```

官方构造函数会打开 CAN 并启动接收线程。Wrapper 不伪造 L20 enable/disable，因为 L20 官方 API 没有相应能力。

官方 `LinkerHandApi.close_can()` 在当前提交中引用未定义的 `modbus`。`disconnect()` 使用当前 L20 Driver 的窄兼容流程：将 `hand.running` 设为 `False`，调用 `hand.close_can_interface()`，并以有限 timeout 等待 `receive_thread`。该兼容代码集中在一个私有方法中，并在 README 标记对应官方提交和原因。

断开时只 shutdown 当前 `python-can` Bus，不执行 `ip link set canX down`。

### 7.3 Position

公开接口：

```python
get_joint_positions_raw(*, fresh: bool = True) -> np.ndarray
get_cached_joint_positions_raw() -> np.ndarray
set_joint_positions_raw(positions) -> None
set_joint_positions_normalized(action) -> None
get_active_joint_indices() -> tuple[int, ...]
```

raw position：

- shape `(20,)`
- 每项必须为整数值
- 范围 `[0, 255]`
- 非整数浮点数不允许静默截断

normalized position：

- shape `(20,)`
- dtype 可为任意实数数值类型
- 范围 `[-1, 1]`
- 映射公式：`floor((action + 1) * 127.5 + 0.5)`
- `-1 -> 0`、`0 -> 128`、`1 -> 255`

`set_joint_positions_raw()` 映射官方 `finger_move(pose=...)`。`get_joint_positions_raw(fresh=True)` 映射 `get_state()`；`fresh=False` 使用 `get_state_for_pub()`。无论哪种方式，返回值必须严格验证为 shape `(20,)` 和 finite，否则抛出 `RuntimeError`。

官方 fresh 读取依次查询四组反馈，当前源码包含约 40 ms 的发送等待，因此不能承诺超过 20 Hz 的 fresh observation。cached 状态必须在名称和文档中明确，不冒充最新测量。

### 7.4 其他实际支持状态

公开接口：

```python
set_speed(speed) -> None             # (5,), integer raw [0, 255]
get_speed() -> np.ndarray            # (5,), raw [0, 255]
set_current(current) -> None         # (5,), integer raw [0, 255]
get_current() -> np.ndarray          # (5,), raw [0, 255]
get_temperature() -> np.ndarray      # runtime expected (20,), SDK unit undocumented
get_fault() -> np.ndarray            # (5,), documented codes
clear_faults() -> None
get_sdk_version() -> str
```

fault code 按官方 API 文档解释：`0` 正常、`1` 电流过载、`2` 温度过高、`3` 编码错误、`4` 过压/欠压。

L20 当前 Driver 明确标记 torque 和 embedded version 不支持，并分别返回伪造全零数组，因此 Wrapper 不提供 `set_torque()`、`get_torque()` 或 `get_version()`。`get_sdk_version()` 只返回官方 `setting.yaml` 中的 Python SDK 版本。

官方没有明确 temperature 的物理单位。Wrapper 保留方法名 `get_temperature()`，docstring 和 README 将单位标记为“SDK 未说明，需真机/厂商确认”，不写成摄氏度。

### 7.5 Presets

提供：

```python
open_hand() -> None
close_hand() -> None
```

两组 20 维值仅来自官方 GUI `HAND_CONFIGS["L20"].preset_actions` 的“张开”和“握拳”。代码注释记录来源提交。示例默认不调用 preset；必须通过 `--execute` 和交互确认后才允许发送小范围命令。

## 8. RobotSystem

### 8.1 生命周期

```python
connect() -> None
enable(timeout: float = 5.0) -> None
disable(timeout: float = 5.0) -> None
disconnect() -> None
self_check() -> dict[str, object]
```

连接顺序为 arm 后 hand。若 hand 连接失败，立即断开已经连接的 arm，再重新抛出包含设备名称的异常。

`enable()` / `disable()` 只作用于 Nero。L20 不存在伪造的 enable 状态。

`disconnect()` 尝试分别释放 hand 和 arm；即使第一个释放失败也继续释放另一个，最后汇总异常。它不隐式 disable Nero。

`self_check()` 只读取接口、连接状态、Nero 固件配置、Nero 状态、L20 position 和 fault，不发送运动命令。返回结构化结果并打印 `[OK]` / `[FAIL]` 摘要。

### 8.2 Observation

```python
{
    "arm": {
        "joint_position": np.ndarray(shape=(7,)),
        "joint_torque": np.ndarray(shape=(7,)),
        "tcp_pose": np.ndarray(shape=(6,)),
    },
    "hand": {
        "joint_position_raw": np.ndarray(shape=(20,)),
    },
    "timestamp": float,
}
```

联合 observation 任一必需字段未就绪时抛出明确异常，不返回不完整结构。`timestamp` 是联合读取完成时的 Unix seconds。

### 8.3 Action 与 step

统一 action：

```python
{
    "arm_joint_position": np.ndarray(shape=(7,)),      # rad
    "hand_joint_position": np.ndarray(shape=(20,)),    # normalized [-1, 1]
}
```

两个 key 均为必需，未知 key 抛出 `ValueError`，防止拼写错误被忽略。

`step()` 执行：

1. 在发送任何命令前完整验证 arm 和 hand action；
2. 调用 `arm.command_joint_positions()`；
3. 调用 `hand.set_joint_positions_normalized()`；
4. 不等待动作完成。

CAN 命令无法跨两个设备原子提交。如果 arm 发送成功而 hand 发送失败，Wrapper 抛出包含“arm command may already have been sent”的 `RuntimeError`，不谎称回滚成功。

`RobotSystem.get_observation()` 默认请求 L20 fresh position。未来需要更高控制频率时，policy rollout 可以降低 observation 频率、在一个 action chunk 内连续调用 `step()`；不把 cached position 标成 fresh feedback。

## 9. 类型、注释和公开 API 文档

类名、函数名和变量名使用英文；README 与 docstring 以中文为主。对外数组统一使用 `np.ndarray`，输入接受 `Sequence[float]` 或 `Sequence[int]` 后立即转换和验证。

每个公开类、公开方法和公开函数都必须提供完整 docstring，至少包含：

- 参数类型、shape、顺序、单位、范围和默认值；
- 返回类型、shape、单位与字段定义；
- `ValueError`、`RuntimeError`、`TimeoutError` 等可能异常；
- 对应的官方 SDK 方法名；
- 固件或型号限制；
- 是否发送一次命令、是否等待动作完成；
- 对机械运动、失能下落或 stale cache 有影响时的 Notes/Warning。

私有适配函数只记录其必要职责和所兼容的官方提交，不为 SDK 内部 CAN 协议补写第二套抽象。

## 10. 错误处理

不建立复杂异常继承体系，使用：

- `ImportError`：SDK 路径或依赖缺失
- `ValueError`：shape、范围、finite、配置或 key 错误
- `RuntimeError`：未连接、未使能、反馈未就绪或底层命令失败
- `TimeoutError`：enable、disable 或线程退出超时

错误信息包含设备、预期 shape/range 和实际值，例如：

```text
Expected 7 Nero joint values in radians, got shape (6,).
```

## 11. 示例安全策略

三个示例默认只读。只有显式 `--execute` 后才进入动作分支，并再次要求终端确认。

- Nero 示例：读取当前 q，以当前 q 为基准只对一个轴增加很小且通过限位验证的 delta；delta 是命令行参数且有保守上限。
- L20 示例：读取当前 20 维位置，以当前值为基准只改变一个主动 index 的少量 raw value；不默认握拳。
- 联合示例：默认只打印 observation；执行模式从当前 arm/hand 状态构造一次小动作，然后展示有限次数的 policy 接入循环位置。

所有示例使用 `try/finally` 和 `KeyboardInterrupt` 处理。由于 Nero disable 可能导致下落，finally 中在执行 disable 前打印警告；是否自动 disable 由显式执行模式和安全配置决定，disconnect 始终尝试执行。

## 12. 测试策略

无真机测试使用依赖注入的 fake Driver/API，覆盖：

- 固件固定为 `NeroFW.V111`
- Nero enable/disable 成功与超时
- Nero 7 维、finite、官方 joint limit、max delta 验证
- `command_joint_positions()` 只发送一次且不等待
- v1.11 observation 不含 joint velocity
- L20 raw shape、整数性和 `[0, 255]` 验证
- normalized `[-1, 1] -> [0, 255]` 映射
- reserved index 保留
- L20 speed/current/fault/temperature 返回维度验证
- L20 teardown 兼容路径
- RobotSystem 连接失败回滚
- `step()` 先完整验证再发送
- observation/action key、shape、dtype 与单位约定

集成静态验证包括：

```bash
python3 -m compileall robot_control examples tests config.py
python3 -m pytest -q
```

在两个 SDK 已放入根目录且依赖安装后，额外验证真实 imports 和官方函数存在性。该验证只能报告“静态验证通过”，不能报告“真机测试成功”。

## 13. Runtime requirements

根目录 `requirements.txt` 仅包含 Wrapper 和三个示例直接需要的依赖：

```text
numpy
python-can>=3.3.4
PyYAML
typing-extensions>=3.7.4.3
```

`pytest` 作为开发/验证工具在 README 单独安装，不进入 runtime requirements。LinkerHand 官方 requirements 中的 GUI、ROS、wandb、mediapipe、dm_control、PyQt 和 Web 依赖不进入本项目。

## 14. README 范围

README 使用中文，覆盖：项目目的、架构、文件结构、SDK 来源与提交、安装、CAN 配置、Nero v1.11 兼容性、L20 配置、运行前检查、控制冲突、单机与联合示例、API、Action/Observation 表、单位、DP/VLA 接入、常见错误和机械安全。

README 明确区分：

- 官方事实；
- Wrapper 约定；
- 静态验证结果；
- 必须真机确认的行为。

## 15. 真机待确认项

以下项目不能通过静态源码验证替代：

- 实际 can0/can1 与左右手配置；
- Nero 实机固件读取确为 1.11；
- Nero enable/disable 时的负载和下落行为；
- Nero `move_j()` 在 10/20/30/50 Hz 下的控制稳定性；
- L20 fresh state 实际刷新率与延迟；
- L20 temperature 返回长度及物理单位；
- L20 current/speed 五维的具体电机对应顺序；
- 官方 L20 open/close preset 是否适合当前手型、安装方向和机械环境；
- 同进程同时打开两个 SocketCAN Bus 的硬件稳定性。

第一次真机验证必须按以下顺序推进：

```text
环境/import
-> CAN
-> 只读状态
-> Nero enable
-> L20 单独小动作
-> Nero 单独小动作
-> 联合动作
-> 连续 step
-> 数据采集
-> DP/VLA
```
