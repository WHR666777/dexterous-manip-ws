# fold-33：捻布与粗略对折经验（双臂接口版）

fold-33 是右臂成功夹起、搬运、松手并撤离的历史示范，结果仍有卷边、褶皱和错位。
以下 4 个参考 cell 已适配双臂接口，尚未实机验证；历史照片与坐标均来自右臂。
公共 task 系为右臂基座，位置用 m；yaw 是相对**各侧自身初始朝向**绕 task +Z 的增量，单位 rad。
**双臂一起移动用 `move_both()`；保持左负、右正 yaw，分 3 次 J 转到左 -0.70、右 +0.70。**
各段 yaw 都相对初始参考姿态，不逐段累加；三次转姿锁定同一对 TCP 目标位置，避免每段重新取点造成累计漂移。
各侧 `top_down_pose(..., arm=...)` 使用本侧参考姿态，不能把右侧位姿矩阵直接交给左臂。
分段减小单次关节变化及布料扰动，仍须让两侧点位和路径保持空间分离；接口不做碰撞检测，也不保证 J 的途中 TCP 固定。

- 对边看指尖是否接触布边，而非只看 TCP 投影。fold-33 的有效修正主要是下降，目标 Z 从 0.018 降至 0.012；指尖悬空时小幅调低，不能只反复调整 X/Y。
- 每侧闭合后立即先抬 10 mm，再到相对该侧闭合前累计 48 mm；图像确认两侧夹布后才搬运。
- 原单臂示范在 X=-0.560 处将右 yaw 一次调到 -0.70；双臂候选将右 yaw 反转为正，分三段到 +0.70，左侧取反，随后保持最终朝向搬运。
  抓放点、高度及换朝向位置须按各侧标定和现场选择，不能直接镜像历史 X/Y。
- 确认夹布后连续搬运、放置、松手、撤离，中间不例行观察，最终撤离后再看结果。

![fold-33 右臂成功捻布：左为闭合前，右为累计抬升约48 mm后](grasp_examples/fold33_grasp.png)

原始 RGB 程序裁剪放大：cell 006 / 007；闭合前观察实测右 TCP≈`[-0.71963, -0.09012, 0.01162]`。

### Cell 1：双侧预抓手型、状态与初始观察

```python
arms = ("left", "right")

def reached(result):
    print(result)
    assert result.reached, result

def move_pair(xyz, mode="L", yaw_right=0.0):
    yaw = {"right": yaw_right, "left": -yaw_right}
    pose = {a: robo.top_down_pose(xyz[a], yaw=yaw[a], arm=a) for a in arms}
    result = robo.move_both(left=pose["left"], right=pose["right"], mode=mode, wait=True)
    for a in arms:
        reached(result[a])

print(robo.state())
for a in arms:
    print(a, robo.gripper_replay_start(arm=a))
obs = robo.observe()
robo.show(obs)
```

### Cell 2：分别选点，到布角上方再下降，观察对边

收到 Cell 1 图像后，在本 cell 开头定义下列字典，**每项都要包含 `left`、`right`**。
左侧 TCP 为单独标定的预抓手型食指指尖参考点，与历史右侧拈布点不同，须分别确定高度。

| 变量 | 每侧内容 | 原右臂示范值，仅作参考 |
| --- | --- | --- |
| `grasp` | 抓取 XYZ | `[-0.720, -0.090, 0.012]` |
| `via_xy` | 换朝向点 XY、后续中间点 XY | `[[-0.560, -0.080], [-0.480, -0.074]]` |
| `place_xy` | 放置 XY | `[-0.423, -0.067]` |
| `hover_z` / `place_z` / `retract_z` | 接近、放置、撤离高度 | `0.066` / `0.028` / `0.078` |

```python
# 先按当前观察填写上述左右字典；不要照搬历史坐标。
s = robo.state()
p = {a: np.array(s["arms"][a]["tcp"])[:3, 3].copy() for a in arms}
hover = {a: max(float(p[a][2]), hover_z[a]) for a in arms}
move_pair({a: [p[a][0], p[a][1], hover[a]] for a in arms}, mode="J")
move_pair({a: [grasp[a][0], grasp[a][1], hover[a]] for a in arms})
move_pair(grasp, mode="J")
obs = robo.observe()
robo.show(obs)
```

### Cell 3：逐侧闭合后立即小抬，再双侧抬升并确认夹布

```python
s = robo.state()
preclose = {a: np.array(s["arms"][a]["tcp"])[:3, 3].copy() for a in arms}
for a in arms:
    print(a, robo.close_gripper(arm=a))
    reached(robo.move_J(robo.top_down_pose(preclose[a] + [0, 0, 0.010],
                                         yaw=0.0, arm=a), arm=a))
move_pair({a: preclose[a] + [0, 0, 0.048] for a in arms}, mode="J")
obs = robo.observe()
robo.show(obs)
```

### Cell 4：固定转姿目标点，3 次 J 调整 yaw，再连续搬运放置

先以 yaw=0 抬到约 0.120 m，再到各侧换朝向点。bi_fold-04 离线估算使用右
`[-0.560, -0.063, 0.120]`、左 `[-0.605, -0.347, 0.120]`，当前点位须重新选取。

| J 段 | 左 yaw | 右 yaw |
| --- | --- | --- |
| 1 | -0.2333 | +0.2333 |
| 2 | -0.4667 | +0.4667 |
| 3 | -0.7000 | +0.7000 |

bi_fold-04 起点的离线关节直线插值估算中，一次转到该最终角度时最大关节变化约 56.5°、
TCP 最大偏移约 31 mm；三段后分别约 26.3°、7.3 mm。该插值不是固件实际轨迹，尚未实机验证。
每段 `move_both(..., mode="J", wait=True)` 等两侧到位后再继续，不增加段间观察；
转完后保持 yaw，后续平移、下降与撤离使用 L，避免再做大幅 J 转姿。
原 bi_fold-04 右放置点 `[-0.418, -0.055, 0.025]` 在最终 yaw 下的离线 IK 未通过，
须重新选择放置点或高度，不能因转姿检查通过就照搬整条路线。

```python
s = robo.state()
carry_z = {a: float(np.array(s["arms"][a]["tcp"])[2, 3]) for a in arms}
p = {a: np.array(s["arms"][a]["tcp"])[:3, 3].copy() for a in arms}
turn_z = {a: max(carry_z[a], 0.120) for a in arms}
move_pair({a: [p[a][0], p[a][1], turn_z[a]] for a in arms})
move_pair({a: [*via_xy[a][0], turn_z[a]] for a in arms})
s = robo.state()
turn_xyz = {a: np.array(s["arms"][a]["tcp"])[:3, 3].copy() for a in arms}
for alpha in [0.70 / 3, 0.70 * 2 / 3, 0.70]:
    move_pair(turn_xyz, mode="J", yaw_right=alpha)  # 左 -alpha，右 +alpha
carry_yaw = +0.70
move_pair({a: [*via_xy[a][1], max(carry_z[a], 0.085)] for a in arms}, yaw_right=carry_yaw)
move_pair({a: [*place_xy[a], max(carry_z[a], 0.060)] for a in arms}, yaw_right=carry_yaw)
move_pair({a: [*place_xy[a], place_z[a]] for a in arms}, yaw_right=carry_yaw)
for a in arms:
    print(a, robo.open_gripper(arm=a))
move_pair({a: [*place_xy[a], retract_z[a]] for a in arms}, yaw_right=carry_yaw)
obs = robo.observe()
robo.show(obs)
s = robo.state()
for a in arms:
    print(a, s["arms"][a]["arm_status"])
print(s["hold_confirmed"], s["quiescent"], s["fault"])
```

![fold-33 右臂松手并撤离后的粗略对折结果](grasp_examples/fold33_result.png)

原示范最终右 TCP≈`[-0.42329, -0.06775, 0.07795]`，NORMAL、idle、保持已确认、fault为空。
依据本轮最终图像提交 MCP `finish`；不把到位或手指反馈单独当成对折成功。
原始证据：`data/agent_runs/fold-33/cells/006–013`、`result.json`。
