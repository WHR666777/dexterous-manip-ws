# fold-33：捻布与粗略对折成功经验

本例把最下方布角折回图像左侧角附近，成功夹起、搬运、松手并撤离；仍有卷边、褶皱和错位。
沿用 pinch-04 标定与初始朝向，坐标为 task 系 m，yaw 为相对初始姿态的 rad。
下列 4 个 cell 整理自成功片段，去掉首次空抓和探索取点；合并版未实机重放。
**先根据当前图像重选抓取点和放置点；下面的绝对坐标只描述本次示范。抓起并确认夹布后，正常搬运至放置、松手、撤离连续执行，中间不例行拍照或等待视觉反馈。**

- 对边应看指尖是否实际接触布边，而非只看 TCP 投影落点。按操作者对本轮图像的判断，首次空抓可先沿布边向图像右下（粗红箭头正向）小幅推进，使指尖接触布边；后续成功调整中主要有效的是下降，目标 Z 从 0.018 降至 0.012 m。若指尖仍悬空，应小幅调低，不要仅在 X/Y 上反复探索；TCP 高度不等于指尖与布面的间隙。
- 捻布目标 `[-0.720, -0.090, 0.012]`、yaw=0；闭合前观察的实测 TCP 见下图说明。
- 闭合后立即先抬 10 mm，再抬到相对闭合前累计 48 mm；图像确认一次布角悬起后才搬运。之后仅在运动失败或需要重新判断现场时补充观察，最终松手撤离后再看结果。
- 固定 yaw=0 只搬到 X=-0.560；在此调整 yaw=-0.70，再保持新朝向分两段 L 搬到 X=-0.423。
  本次调整后 J4 从约 1.754 降至 1.289 rad；最终搬运到位时约 1.856 rad。
  这是 fold-31 的 X≤-0.50 经验边界之外的一次成功姿态方案，不保证其他构型同样可行。

![fold-33 成功捻布：左为闭合前，右为累计抬升约48 mm后](grasp_examples/fold33_grasp.png)

原始 RGB 裁剪放大，无生成内容；左 cell 006、右 cell 007。
左图观察时实测 TCP≈`[-0.71963, -0.09012, 0.01162]`，右图布角随手悬起，作为成功夹布证据。

### Cell 1：读取状态、预抓手型、观察（原 001–002）

```python
def reached(result):
    print(result)
    assert result.reached, result

print(robo.state())
print(robo.gripper_replay_start())
obs = robo.observe()
robo.show(obs)
```

### Cell 2：到布角上方再下降，观察对边（整理原 003、006）

```python
grasp = np.array([-0.720, -0.090, 0.012])
place_xy = [-0.423, -0.067]
p = np.array(robo.state()['tcp'])[:3, 3]
reached(robo.move_J(robo.top_down_pose([p[0], p[1], 0.066], yaw=0.0)))
reached(robo.move_L(robo.top_down_pose([grasp[0], grasp[1], 0.066], yaw=0.0)))
reached(robo.move_J(robo.top_down_pose(grasp, yaw=0.0)))
obs = robo.observe()
robo.show(obs)
```

### Cell 3：闭合后立即抬升，再确认夹布（原 007）

```python
preclose = np.array(robo.state()['tcp'])[:3, 3].copy()
print(robo.close_gripper())
reached(robo.move_J(robo.top_down_pose(preclose + [0, 0, 0.010], yaw=0.0)))
reached(robo.move_J(robo.top_down_pose(preclose + [0, 0, 0.048], yaw=0.0)))
obs = robo.observe()
robo.show(obs)
```

### Cell 4：调整朝向、分段搬运、放置撤离，最后观察（原 008–013）

```python
carry_z = float(np.array(robo.state()['tcp'])[2, 3])
reached(robo.move_L(robo.top_down_pose([-0.560, -0.080, carry_z], yaw=0.0)))
p = np.array(robo.state()['tcp'])[:3, 3]
carry_yaw = -0.70
reached(robo.move_J(robo.top_down_pose(p, yaw=carry_yaw)))
reached(robo.move_L(robo.top_down_pose([-0.480, -0.074, carry_z], yaw=carry_yaw)))
reached(robo.move_L(robo.top_down_pose([*place_xy, carry_z], yaw=carry_yaw)))
reached(robo.move_J(robo.top_down_pose([*place_xy, 0.028], yaw=carry_yaw)))
print(robo.open_gripper())
reached(robo.move_J(robo.top_down_pose([*place_xy, 0.078], yaw=carry_yaw)))
obs = robo.observe()
robo.show(obs)
s = robo.state()
print(s['arm_status'], s['hold_confirmed'], s['quiescent'], s['fault'])
```

![fold-33 松手并撤离后的粗略对折结果](grasp_examples/fold33_result.png)

最终实测 TCP≈`[-0.42329, -0.06775, 0.07795]`，NORMAL、idle、保持已确认、fault为空。
依据当前图像提交 MCP `finish`；不把到位或手指反馈单独当成对折成功。
原始证据：`data/agent_runs/fold-33/cells/006–013`、`result.json`。
