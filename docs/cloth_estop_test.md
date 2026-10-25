# Nero 人工停止与位置保持

## 先纠正物理语义

现场已观察到 `electronic_emergency_stop()` 让机械臂下落。当前安装的 Nero SDK 源码
将其描述为 damped emergency stop（阻尼急停），没有承诺固定位置保持。
旧的 `cloth_agent estop` 命令已禁用，不向设备发包；不要继续按旧测试步骤操作。

新入口是 `hold`：锁定任务动作，捕获一次当前实测七轴位置，持续发送同一个 JS 位置目标。
不调用电子急停、disable、reset 或自动 enable，不默认松开 L20。
它依赖供电、使能、反馈、CAN 和控制进程，不能保证设备/软件故障或断电时悬停，
也不是安全等级的硬件急停。当前新实现仅做语法检查，未验证实机效果。

旧进程不会因源码变更自动更新。先按现场流程支撑、稳定机械臂，结束旧控制程序，
再启动新服务。新的人工接口拒绝旧 `session.json`，防止误触旧阻尼急停。
不要为了恢复直接 reset；恢复使能必须由现场操作者按已有流程处理。

## 环境与配置

每个终端先执行：

```bash
conda activate nero
cd /home/lab404/WHR/dexterous-manip-ws
which python
```

应为 `/home/lab404/anaconda3/envs/nero/bin/python`。已安装 MCP 1.30.0、TOPPRA 0.6.10，
并以 editable 方式注册项目；保留既有 NumPy、OpenCV、Nero/L20 SDK。
环境原有 cmeel-boost/nlopt 对 NumPy、wandb 对 protobuf 的依赖声明冲突未改；
导入成功不能代替数值兼容和实机验证。

`configs/cloth_agent_estop.yaml` 保留历史文件名，现用于位置保持联调：
右侧 `can_arm_a` / `can_hand_a`，相机关闭，不写手速，不主动发送开合手目标。
启动仍需右臂与 L20 反馈可用。程序不自动使能、不回零、不复位。
连接后在 3 秒内等待使能、关节、法兰反馈；缺失反馈不等于机械臂未使能。

## 先验证静止、受支撑条件下的保持

机械臂已有可靠外部支撑，停止其他遥操作/回放/控制程序，由操作者确认接线、姿态、
通道与独立物理急停可用。不要先尝试无支撑悬空或运动中停止。

终端 A 使用新目录，保留所有旧记录：

```bash
python -m cloth_agent serve --execute \
  --config configs/cloth_agent_estop.yaml \
  --output-dir data/agent_runs/hold-idle-01 \
  --cell-timeout 600 --task-timeout 900
```

带 `--execute` 启动后直接连接设备；出现 `hardware_connected: true` 和 `session_file` 才表示初始化完成。
在初始化完成前，人工 socket 尚未公布，不能依赖此入口停止初始化过程。

终端 B：

```bash
python -m cloth_agent hold data/agent_runs/hold-idle-01/session.json
```

新实现的处理顺序：

1. 独立主线程锁定任务权限，拒绝新动作、旧轨迹和普通保持发送。
2. 捕获一次带新鲜度检查的实测关节，作为不变的保持目标；重复 hold 不重新捕获。
3. 主线程以配置周期持续下发同一目标，不等待 worker/Codex 退出、相机或日志。
4. worker/Codex 随后结束，任务状态为 `manual_hold`，控制进程持续运行。

普通和人工 JS 下发只在单次 SDK 发送处互斥，避免两条七轴目标的 CAN 帧交错。
已在执行中的 SDK 调用仍可能延迟切换，没有硬实时停止时限承诺。

终端 B 查询：

```bash
python -m cloth_agent hold-status data/agent_runs/hold-idle-01/session.json
```

查看 `manual_hold_latched`、`manual_hold_target`、`manual_hold_report`、`hold_confirmed`。
`hold_target_sent` 只表示 SDK 调用返回。`hold_confirmed: true` 表示一段时间内关节反馈
接近固定目标且相邻采样变化较小，不是独立测得的物理停机或安全保证。
`stop_confirmed` 保持 false；现场必须观察是否仍有漂移、跳动或下落趋势。

`hold_failed`、使能/反馈丢失、请求超时都不能视为已保持。保持失败不会自动改用阻尼急停、
reset 或重新使能；使用已有外部支撑和现场处置流程，不撤掉支撑继续尝试。

## 明确释放保持后再退出

人工保持会有意保留控制进程。不要 kill、SIGSTOP 或关闭终端来替代释放流程，
否则主动位置指令流可能消失；Ctrl+C 不会解除人工保持。

确认机械臂已安全落座或外部支撑能够独立承重后，终端 B 执行：

```bash
python -m cloth_agent release-hold data/agent_runs/hold-idle-01/session.json
```

按提示输入 `SUPPORTED`。程序随后停止保持并关闭设备连接，不恢复轨迹、清除故障、
自动失能、reset 或松手。释放回执不保证硬件会继续承重，因此必须先完成现场支撑确认。
控制进程退出前终端 A 可能仍需短暂等待清理；任务记录的 `manual_hold` 表示人工撤权原因，
不因释放而改写为任务成功。

## 后续测试顺序与记录

先完成以上受支撑静止验证，再由现场操作者判断能否在可接住/限制下落的条件下逐步减少支撑，
确认实际保持效果；此前不要执行运动中停止。忙碌 cell 可在新的会话中用人工 MCP 客户端提交：

```bash
python scripts/cloth_call.py http://127.0.0.1:本次端口/mcp --code 'time.sleep(600)'
```

确认 `events.jsonl` 出现 `cell_started` 后，从另一终端执行该新会话的 `hold`，
检查 cell 被中断且保持进程仍运行；测试结束也必须先确认支撑再 release-hold。
客户端的退出码不是实际悬停的判据。

记录目录、初始支撑和使能状态、固定目标、反馈误差、是否漂移/跳动/下落及所有错误。
`control-events.jsonl` 保存人工保持状态变化，`cells/*.json` 记录 cell 中断，
`result.json` 记录任务终态。`hold-status` 可在普通 I/O 阻塞时通过独立人工通道查询；
SDK 发送或控制进程本身阻塞时仍不能保证及时回执。

旧 `data/agent_runs/estop-env-check-20261025/` 只验证旧方案的无硬件软件链路。
其中约 1.24 ms 的回执时间不能作为新版保持、CAN 发送、硬件停机或悬停的验证证据。
