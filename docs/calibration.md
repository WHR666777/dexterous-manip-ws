# 现场标定与首次验证

## 1. 通道确认

在没有其他控制程序运行时，用系统工具确认 Nero 与 L20 各自对应的 SocketCAN 通道和 bitrate。将结果写入 `configs/quest3_nero_l20.yaml`，不要依赖设备插入顺序猜测。先分别运行 `preflight --control arm` 和 `preflight --control hand`，再运行 `both`。


## 5. RGB-D 相机外参与拈布虚拟 TCP 联合标定

入口为 `scripts/calibrate_tcp_camera.py`，分为采集、人工点选、离线求解三步。使用现有 teleop 环境及 `vision` 依赖（OpenCV、RealSense）。采集沿用原来的使能、起始姿态、工作区与 IK 检查；先启动 Pico daemon，确保没有其他设备控制程序占用机械臂或相机。

### 坐标定义

- `base` 是 Nero URDF 的 `base_link`，`flange` 是 `link7`，`camera` 是对齐 RGB-D 使用的彩色相机光学坐标系。
- TCP 原点是固定手型下选定的指尖参考点：现有右侧使用拈布点，左侧使用第 0 帧的食指指尖。采集默认固定 replay CSV 的第 0 帧；使用 `--hand-frame -1` 可选择最后一帧闭合手型，其余三指张开。整个采集期间 P/O 不会改变手型。
- 参考 roll、pitch 读取 `arm.wrist_only_roll_pitch_deg`。假设 Nero base 的 +Z 实际竖直向上，在参考姿态下定义 TCP +Z 向下、+X 沿法兰 +Z 的水平投影、+Y = +Z × +X。参考 yaw 取 0 即可，算出的法兰到 TCP 固定旋转不受参考 yaw 影响。
- **TCP 固定在手上。** 采集时允许腕部倾斜，TCP 的轴随手转动；不会在每张图片中重新把 TCP Z 轴设成向下。倾斜采样不改变之后使用的 replay 抓姿。

### 采集

```bash
python3 scripts/calibrate_tcp_camera.py collect data/calibration/pinch-01 \
  --config configs/pico4_nero_l20.yaml --execute
```

指定 `--execute` 后直接开始：机械臂会先到配置中的起始姿态，手指设为所选 replay 帧。与 `wrist-only` 不同，标定模式允许通过 Pico 控制完整腕部旋转，包括 roll、pitch。

1. 按 `R` 锚定，通过 Pico 改变位置和倾斜角度。
2. 保持姿态不动，按 `C` 停止发送后续运动目标。程序等待七轴实际反馈在 0.4 s 内保持稳定（范围 0.002 rad），然后保存一组 RGB-D、实际关节反馈、URDF/FK 和时间信息。停止发送不是急停；最后目标可能仍在执行，所以必须等待反馈稳定。
3. 采样成功或失败后都保持暂停。按 `R` 重新锚定，再移动到下一姿态；失败也可保持不动再按 `C`。
4. 建议采集 15～25 组，覆盖不同位置、高度、yaw，并加入绕不同方向的 roll/pitch 倾斜。只改变位置和 yaw 无法唯一分离相机平移与 TCP 偏移。至少需要 6 组，求解器还会检查可观测性。
5. 按 `Q` 退出。相机、基座和手的安装关系在整个采集中保持不变；数据目录必须是新目录。

这里只关联静止姿态下的图像和关节反馈，记录的是主机接收/查询时间，不声称硬件同步。

### 人工点选

```bash
python3 scripts/calibrate_tcp_camera.py annotate data/calibration/pinch-01 --scale 2
```

每张 RGB 图上左键选择**同一个物理接触点**。Enter/N 保存并下一张，S 跳过，A 返回上一张，D 删除当前标注，Q/Esc 退出；已经确认的标注保存在 `annotations.json`，可再次打开修改。

程序使用所点像素的对齐深度和保存的相机内参反投影，不用邻近像素填补无效深度。点选处深度为零时不能确认。接触点必须可见；若像素深度来自前侧手指表面或背景，就不能当作被遮挡接触点的位置。

### 求解与结果

```bash
python3 scripts/calibrate_tcp_camera.py solve data/calibration/pinch-01
```

联合最小化三维位置误差：

```text
R_base_camera * point_camera_i + t_base_camera
    = R_base_flange_i * tcp_offset_flange + t_base_flange_i
```

相机外参 6 个自由度与 TCP 原点在法兰中的 3 个坐标由数据求解；法兰到 TCP 的旋转由参考姿态和坐标定义计算。求解不会自动删除样本，会输出各样本残差、RMS、最大误差和几何条件数。姿态退化或拟合 RMS 超过默认 5 mm 时拒绝生成新结果；`--max-rms-mm` 可显式设置拟合阈值。已有结果不会被失败的求解覆盖。拟合残差不代表独立的实机定位精度。

结果默认写入会话目录的 `calibration.yaml`，也可用 `--output` 指定路径，包含：

- `base_from_camera` / `camera_from_base`：相机与 Nero base 的变换及逆变换。
- `flange_from_tcp` / `tcp_from_flange`：TCP 与法兰的完整固定变换及逆变换。
- 闭合手型 raw 值、参考 roll/pitch、样本名称和拟合误差。

`a_from_b` 表示将 B 坐标转换到 A 坐标，平移单位是米。标定程序不改写运行配置；agent 配置通过 `calibration.file` 引用结果 YAML、通过 `calibration.arm` 指定对应机械臂。路径相对于配置文件所在目录，也支持绝对路径。

`configs/cloth_agent.yaml` 通过合并标定文件读取相机外参、固定工具变换、手型和工具参考角；
记录起始关节位姿的 FK 仍用于任务工作区和第一段回位。实际启动 TCP 位姿从
`startup.tcp_poses` 独立读取，不再由标定参考 roll/pitch 与记录法兰位置拼接。
agent 的 yaw 是相对配置的启动 TCP 朝向绕 task +Z 的增量；`startup.enabled` 时
宿主先准备到该姿态，再启动模型。重新标定后修改
`calibration.file` 即可，不必复制矩阵。TaskSession 冻结配置，只有 `--execute` 才连接设备。
当前 task 坐标系取右臂 Nero base；左臂标定与坐标转换见下节。

几何底层 `cloth_agent.geometry.calculate` 保留 `camera_to_task` 和 `pixel_to_task` 操作，需要显式传入 `task_from_camera`，深度与内参必须来自同一 RGB-D 快照。旧观测封装已删除，当前没有关键点自动坐标转换或 agent 几何工具入口。

### 左手标定：复用右手第 0 帧手型

使用 `configs/pico4_nero_l20_left.yaml`。它选择左侧 Pico 输入、`can_arm_b`、
`can_hand_b`、左手 SDK 类型和 `nero_left_start_pose.yaml`；相机固定为右侧使用的
L515（序列号 `f1420931`）。左右手共享 `configs/l20_thumb_index_9_10s.csv`，
不取反或交换 raw 槽位。左侧明确使用 `--hand-frame 0` 的预抓手型，
7 个有效拇指/食指槽位与右侧 `replay_start` 相同。
其他三指由启动时的官方全手张开预设设置，与右侧相同。

左侧 TCP 使用原始标定参考角 `[-87.781805°, -20.890762°]` 定义工具轴。
左右侧均按第 0 帧标定处理；合并时不镜像、重定义左侧工具朝向。采集允许完整腕部旋转。

在仓库根目录、已安装 teleop/vision 依赖的 Python 环境中执行。本机现有 `nero`
环境可用：`conda activate nero`；项目 `.venv` 当前缺少 `pyrealsense2`。
先在另一个终端启动 Pico daemon：

```bash
python3 AnyDexRetarget/example/input/pico4_daemon.py
```

左手标定仍使用同样的三步流程：

```bash
python3 scripts/calibrate_tcp_camera.py collect data/calibration/pinch-left-01 \
  --config configs/pico4_nero_l20_left.yaml --hand-frame 0 --execute
python3 scripts/calibrate_tcp_camera.py annotate data/calibration/pinch-left-01 --scale 2
python3 scripts/calibrate_tcp_camera.py solve data/calibration/pinch-left-01
```

`collect` 会使能左臂、回左侧记录起始位姿，设置共享的第 0 帧手型；它不连接右臂或右手。
按 `R` 锚定/恢复，保持静止后按 `C` 采样，按 `Q` 退出。采集不同位置和 roll/pitch
倾斜的样本，然后点选同一个可见食指指尖参考点。采集程序自行创建目录，
不要预先创建 `pinch-left-01`；重采使用新的目录名。

结果中的 `hand_side` 为 `left`，`flange_from_tcp` 保留在左法兰坐标系。
在相机和两臂基座保持固定、左右两份标定对应同一个相机坐标系时，以相机为桥梁：

```python
task_from_left_base = right_base_from_camera @ np.linalg.inv(left_base_from_camera)
```

其中 `right_base_from_camera` 来自右侧 `pinch-04/calibration.yaml`，
`left_base_from_camera` 来自本次左侧标定。若相机或基座在两次采集之间移动过，
应重新采集对应标定，再计算基座关系。左手采集和求解不需要预先知道右侧 task 变换。

### 合成为双臂 agent 标定文件

```bash
python3 scripts/calibrate_tcp_camera.py merge \
  data/calibration/pinch-04/calibration.yaml \
  data/calibration/pinch-left-01/calibration.yaml \
  --output configs/cloth_agent_calibration.yaml
```

`configs/cloth_agent.yaml` 的 `calibration.file` 引用这份自包含文件。它保存公共
`task_from_camera`、两侧 `task_from_base` / `flange_from_tcp`、参考角、原标定手型和拟合残差。
合并时完整保留两侧原始 `flange_from_tcp` 和参考角，以及已拟合的相机外参。
重复 merge 不会镜像或重定义左侧 TCP 朝向。启动目标单独配置为位置左右对称、朝向一致。
两侧 `open` / `closed` / `replay_start` 使用共用手型；各侧 `hand_pose_raw` 只记录标定手型，
不会把左侧第 0 帧改作闭合命令。相机、基座关系和原拟合残差仍决定实机精度。
