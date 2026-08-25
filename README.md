# Nero + LinkerHand L20 研究控制封装

## 项目目的

这是面向研究控制的薄封装：在官方 Nero `pyAgxArm` 和 LinkerHand Python SDK 之上提供稳定的 NumPy 接口，以便接入策略、遥操作和数据采集。它不是安全认证控制器，也不实现 CAN 编解码、IK、ROS、相机或训练算法。默认示例只读；只有同时给出 `--execute` 并在终端精确输入 `EXECUTE`，才会发送一次保守动作。

本文以四类信息标注边界：**官方事实**来自固定 SDK；**封装约定**是本项目公开接口；**静态验证**仅指无硬件测试；**待真机验证**绝不等同于已经在机器上安全可用。

## 系统架构

```text
官方 SDK（固定源码） -> robot_control 研究封装 -> DP / ACT / VLA / Teleoperation / 数据采集
```

Nero 与 L20 使用不同 CAN 通道：默认 `can0` 为 Nero、`can1` 为 L20。`RobotSystem` 组合两者，构造时不连接；显式 `connect()` 后仍须显式 `enable()` Nero。L20 官方 API 没有 enable/disable，封装不会伪造该能力。

## 文件结构

```text
config.py                         默认 CAN、型号与频率
robot_control/                    NeroArm、LinkerHandL20、RobotSystem
examples/                         默认只读、二次确认执行的示例
tests/                            无硬件单元/契约测试
pyAgxArm/                         官方 Nero SDK checkout
linkerhand-python-sdk/            官方 LinkerHand SDK checkout
```

## 官方 SDK 来源与固定提交

官方事实：

- AgileX `pyAgxArm`：`agilexrobotics/pyAgxArm`，`8cd90f9106219a156c3c0d7e58ee36d838a89baf`。
- LinkerHand SDK：`linker-bot/linkerhand-python-sdk`，`0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8`。

两个目录是独立、固定的源码 checkout，不能为了本封装改动其核心代码。L20 的断开 shim 也以 LinkerHand 提交 `0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8` 为依据：该版本 `close_can()` 引用未定义的 `modbus`；封装仅在私有清理流程中停止 `hand.running`、调用 `hand.close_can_interface()` 并有界等待接收线程。它只关闭当前 SDK bus，绝不执行 `ip link set canX down`。

## 环境安装

在项目根目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./pyAgxArm
export PYTHONPATH="$PWD/linkerhand-python-sdk:$PYTHONPATH"
```

`requirements.txt` 只列运行时直接依赖：NumPy、python-can、PyYAML 和 typing-extensions。`pytest` 是开发/静态验证工具，刻意不属于运行时依赖；需要静态验证时单独安装：

```bash
python3 -m pip install pytest
```

## CAN 配置

先做不改变状态的检查，再由有权限的操作者激活接口：

```bash
lsusb
ip link show
ip -details link show can0
ip -details link show can1
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
candump can0
candump can1
```

前四条为只读检查；最后两条 `candump` 仅监听 CAN 帧。官方 LinkerHand `LinkerHand/config/setting.yaml` 有 `PASSWORD` 设置，供 SDK 在 Linux 上激活 CAN 时把管理员密码交给 `sudo`；本项目不存放、复制或建议填写真实 sudo 密码。请预先由操作者手动激活 CAN（如上命令），并在需要时按组织安全流程处理权限。两个接口均要求 `bitrate 1000000`；确认设备枚举和通道归属后，才运行 Python。

## Nero v1.11 compatibility

官方事实：Nero 固件 `1.11` 必须创建为 `NeroFW.V111`（官方配置参数名保留拼写 `firmeware_version`），Linux 默认是 `socketcan`/`can0`。封装约定：只支持七轴 V111，并在构造后开启官方软件关节限位；不静默回退到 `NeroFW.DEFAULT`、V112 或 V120。

V111 的官方 motor-state `velocity` 不可信（Driver 强制为零），所以本封装**不提供** `joint_velocity`，也不通过差分伪造 `dq`。同样不暴露未经核查的 IK、MIT、CPV 或 JS 控制。Nero 位置命令是 `(7,)`、单位 rad、有限数并受官方限位检查；`speed_percent` 是 `[0,100]` 的整数百分比，不是 rad/s。不要将 `move_l()` 用作连续目标流。

## L20 配置与控制冲突

封装约定：仅支持 `L20`、默认右手、CAN `can1`，位置永远使用官方 20 槽位顺序。官方也提示不要同时运行 linker_hand_sdk_ros、动捕手套或其他控制灵巧手的 topic。`candump` 虽不发命令，正常控制也不得与**任何会发送 L20 命令的进程**并行运行；同一时刻只保留一个控制者。

L20 原始位置是 20 个整数、每项 `[0,255]`。索引 `11..14` 为 reserved，仍必须保留在所有 20 维 action/state 中，不能删掉、压缩或猜测其含义；主动位置索引仅为 `0..10,15..19`。官方当前实现的 torque 和 embedded version 返回伪造占位值，因此封装不公开 `get_torque()`、`set_torque()` 或 `get_version()`；`get_sdk_version()` 仅是 Python SDK 版本。封装在运行时校验 temperature 为 `(20,)`，但实际硬件返回长度和物理单位均待真机/厂商确认，不能标为摄氏度。

新鲜的 `get_joint_positions_raw(fresh=True)` 会查询四组反馈，官方源码典型等待约 40 ms，因此不能承诺新鲜观测超过 20 Hz。缓存读取必须明确使用 `get_cached_joint_positions_raw()`，不能把缓存称作最新实测。

## 运行前检查

1. 固定机械臂和灵巧手，清出运动范围，准备急停，并确认电源、USB-CAN 和线缆。
2. 确认没有其他 ROS、GUI、脚本、遥操作或 policy 进程占用/发送两个设备的命令。
3. 用 `ip -details link show can0`、`can1` 核实接口已 up 且为 1 Mbps；用 `candump` 只做监听排查后退出。
4. 核对 `config.py`：`can0`/`can1` 不得相同，固件为 `1.11`、型号为 `L20`。
5. 从最小只读示例开始，观察 `self_check()` 的连接、固件、状态、故障和位置结果；失败时不要使能或重试运动。

## 单独运行 Nero

```bash
python3 examples/nero_example.py
python3 examples/nero_example.py --execute
```

第一条只连接、读取状态并断开；第二条还要求键入 `EXECUTE`，才会使能并以当前 `(7,)` rad 反馈为基准只改变一个关节。示例 finally 会尝试断开，但不会自动 `disable()`，因为失能可能下落。

## 单独运行 L20

```bash
python3 examples/l20_example.py
python3 examples/l20_example.py --execute
```

默认路径读取 SDK 版本、20 槽位位置、五电机速度/电流/故障与 20 槽温度。执行路径同样需要 `EXECUTE`，只相对当前反馈改变一个主动 raw 槽位；不操作 reserved 槽位。

## 联合运行

```bash
python3 examples/nero_l20_example.py
python3 examples/nero_l20_example.py --execute
```

默认联合示例执行 `connect()`、`self_check()` 和 observation 读取，不使能 Nero、不调用 `step()`。确认执行后最多调用一次 `RobotSystem.step()`：Nero 保持当前目标，L20 仅改一个主动槽位。`step()` 会在发送前验证两个 action，但**不是跨设备原子事务**：发送顺序是 Nero 后 L20；若 L20 发送失败，Nero 命令可能已经发出。调用者必须设计故障处置和安全边界。

## API 说明

`NeroArm` 提供连接、使能、状态、七轴 rad 命令与只读观测；`LinkerHandL20` 提供惰性连接、20 槽 raw/normalized 位置和经核查的五电机诊断；`RobotSystem` 负责连接回滚、组合诊断、规范 observation 与 action 分发。所有连接/读取/发送都可能因 CAN、SDK 或反馈无效而抛出 `RuntimeError`；输入形状、范围或有限性错误为 `ValueError`。

`RobotSystem.connect()` 只连接，不使能。`disconnect()` 按 L20 再 Nero 尝试清理并聚合失败；重复调用安全。不要依赖断开替你失能 Nero。

## Action / Observation

| 边界 | 名称/键 | 形状与 dtype | 范围、单位和含义 |
| --- | --- | --- | --- |
| Nero action | `NeroArm.command_joint_positions(joints)`；组合键 `arm_joint_position` | `(7,)`，有限数，转为 `float64` | rad；逐轴官方限位，且必须已连接、七轴 enabled |
| L20 action | `LinkerHandL20.set_joint_positions_raw(positions)` | `(20,)`，整数 | raw `[0,255]`；`11..14` 仍在数组内 |
| 组合 L20 action | `hand_joint_position` | `(20,)`，有限 `float64` | normalized `[-1,1]`；映射为 `floor((x+1)*127.5+0.5)`，`-1→0`、`0→128`、`1→255` |
| Nero observation | `arm.joint_position`、`arm.joint_torque`、`arm.tcp_pose` | `(7,) float64`、`(7,) float64`、`(6,) float64` | 分别为 rad、N*m、`[m,m,m,rad,rad,rad]`；没有 joint velocity |
| L20 observation | `hand.joint_position_raw` | `(20,)`，`int64` | 新鲜 raw `[0,255]`；含四个 reserved 槽位 |
| 组合 observation | `{"arm": ..., "hand": ..., "timestamp": ...}` | 嵌套 dict；`timestamp` 为 `float` | 两设备读取完成后的 Unix wall-clock seconds |

单机 `NeroArm.get_observation()` 也含自己的读取完成时间戳；组合 API 丢弃它，只保留一个顶层时间戳，避免混用不同完成时刻。读取与动作不表示设备运动完成。

## 单位与维度

Nero joint position 为 7 维 `float64` rad，joint torque 为 7 维 `float64` N*m，flange/TCP pose 为 6 维 `float64` `[m,m,m,rad,rad,rad]`。L20 position 是 20 维 raw `[0,255]` 或 normalized `[-1,1]`；速度和电流是 5 维 raw `[0,255]`，fault 是 5 维官方错误码（0 正常、1 电流过载、2 温度过高、3 编码错误、4 过压/欠压）。L20 temperature 是封装期望并校验的 20 维运行时反馈，实际硬件长度和单位尚未确认，不能标为摄氏度。

## Diffusion Policy / ACT 接入

DP/ACT 应把本封装当作硬件边界，而不是依赖官方消息对象。建议在 policy 侧保存上述明确的 key、shape、单位和 slot 顺序；将模型输出裁剪/拒绝在 policy 安全层后，再调用 `RobotSystem.step()`。观测循环必须容忍 L20 fresh 约 40 ms、无超过 20 Hz 保证和设备时间不同步；初期以只读日志/回放验证，再在受监督下逐级提高命令频率。此仓库不包含 DP 或 ACT 实现、训练权重或安全证明。

## VLA / Teleoperation 接入

VLA 或 Teleoperation 同样只能作为单一命令源，先经过人工监督、工作空间/速率限制和本封装的 action 校验。遥操作采集应记录 canonical observation、动作、时间戳、`config.py`、SDK 固定提交和是否 fresh/cached；不要把缺失 V111 velocity、L20 torque/version 占位数据或 reserved 槽位伪造成训练特征。VLA/遥操作实现不在本仓库，且其闭环安全性待真机验证。

## 常见错误

- `ModuleNotFoundError: LinkerHand`：重新执行 `export PYTHONPATH="$PWD/linkerhand-python-sdk:$PYTHONPATH"`，并确认 SDK checkout 与依赖存在。
- 找不到 `can0`/`can1` 或 bitrate 不对：检查 USB-CAN 枚举、线缆和前述 `ip` 输出；由操作者预先激活 CAN。
- `RuntimeError` 反馈不可用：停止命令，检查总线、设备电源与控制冲突；不要把缺失反馈替换为零。
- `ValueError` action：修正 `(7,)` rad / `(20,)` normalized 或 raw 的 shape、范围和 reserved 槽位约束，勿靠截断绕过。
- L20 断开超时：接收线程仍在清理；不要立即重连，待线程退出后再调用 `disconnect()` 重试。

## 安全注意事项

仅在评估过的机械安全条件下使用。先固定设备、清空人和障碍物、准备急停，并确保只存在一个控制进程。默认只读不代表接线或 CAN 绝对无风险。

`disable()` 以及电子急停后的 `reset()` 可能使 Nero 下落；不要让 finally、异常处理或脚本自动 disable。断开也不会隐式 disable。所有动作从当前反馈出发、小幅度、单步、低速开始；若观察异常、故障、碰撞风险或控制权不明，立即停止发送命令并按现场安全流程处理。

## 验证状态

**静态验证：**项目测试使用注入假驱动、官方 API/源码契约和命令行 `--help`，不连接 CAN、不创建真实设备、不发送硬件命令。根目录 `python3 -m pytest -q` 仅发现 `tests/`，避免嵌套官方 SDK 的同名测试包干扰。

**真机验证：**尚未执行。以下均为 true-hardware pending，不得由静态测试推断为已通过：实际 `can0`（Nero）/`can1`（L20）与右手映射、Nero 固件 report、Nero enable 是否实际成功及 disable/急停后 reset 的下落行为、Nero `move_j()` 控制在 10/20/30/50 Hz 的稳定性、L20 fresh 延迟、L20 temperature 的长度和单位、五电机 speed/current 与实体电机的对应关系、官方 open/close presets 对当前安装姿态的适用性，以及同一 Python 进程中双 SocketCAN 的稳定性。CAN bitrate、断开 shim、Nero/L20 实际运动、DP/ACT/VLA/遥操作闭环和机械安全也同样待真机验证。

## 建议的第一次真机测试顺序

1. **环境/import：**断电或安全固定状态下确认机械安装、急停、供电、线缆和唯一控制进程；完成“环境安装”的三条运行命令及 `python3 -m pip install pytest`，然后执行下列只导入检查，确认 Python 能导入两个 SDK 和本封装（不构造设备）。

   ```bash
   python3 -c 'import pyAgxArm; import LinkerHand.linker_hand_api; from robot_control import LinkerHandL20, NeroArm, RobotSystem; print("imports OK")'
   ```
2. **CAN：**上电后只运行 `lsusb`、`ip link show`、`ip -details link show can0`、`can1`，由操作者激活两个 1 Mbps 接口；短时 `candump can0`/`candump can1` 观察帧后退出监听。
3. **只读状态：**分别运行三个不带 `--execute` 的示例，检查连接、firmware、位置与 fault；任何失败即停止。
4. **Nero enable：**在人员监护、低速、清空工作空间条件下，仅使能 Nero 并观察状态；确认没有非预期运动，再继续。
5. **L20 单独小动作：**运行 `l20_example.py --execute`，输入 `EXECUTE`，只发送一次相对当前反馈的最小主动 raw 槽位变化；重新读取反馈。
6. **Nero 单独小动作：**运行 `nero_example.py --execute`，输入 `EXECUTE`，只发送一次相对当前 `(7,)` rad 反馈的最小关节变化；重新读取反馈。
7. **联合动作：**运行 `nero_l20_example.py --execute`，输入 `EXECUTE`，只执行一次最小 `RobotSystem.step()`，并记录时间戳、输出和异常处理。
8. **连续 step：**只在单次联合动作和频率边界已逐级验证后，才在现场监督下小步扩大连续 `step()` 的频率；它不是跨设备原子事务。
9. **数据采集：**在已验证的低风险频率下记录 canonical observation、action、timestamp、SDK 提交和 fresh/cached 标记，且保持单一控制者。
10. **DP/VLA：**最后才接入 Diffusion Policy/ACT、VLA 或 Teleoperation；先只读/回放，再在上述机械与频率边界内受监督运行。
