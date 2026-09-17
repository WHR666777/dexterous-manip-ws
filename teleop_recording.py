"""三个遥操入口共用的低维 DP 录制：主机时间对齐，10 Hz，Zarr episode。

观测为真实七轴关节角(rad)和手部位置进度；action 为已发送的七轴目标+开合目标。
位置缓存没有硬件同步保证；保存源时间供检查，不把读取时间伪称为 CAN 接收时间。
"""
import importlib.util
from pathlib import Path
import time

import numpy as np

from robot_control.l20 import L20_GRIPPER_OPEN, L20_GRIPPER_CLOSE

RECORD_HZ = 10
MAX_SAMPLE_AGE = .15  # 时间网格前最近观测的最大间隔；断流截断 episode，不补造数据。
GRIPPER_TOLERANCE = 8  # 原始位置到位容差；所有变化槽位到位才记为 0/1。


def gripper_state(raw):
    """将实际位置投影到张开→闭合路径；受阻未到位时不会仅因收到 C 就记 1。"""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (20,) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 255)):
        raise ValueError('灵巧手位置反馈非法')
    opened, closed = np.asarray(L20_GRIPPER_OPEN), np.asarray(L20_GRIPPER_CLOSE)
    moving = opened != closed  # 排除保留槽位和姿态中不变的关节。
    if np.max(np.abs(raw[moving] - opened[moving])) <= GRIPPER_TOLERANCE:
        return 0.
    if np.max(np.abs(raw[moving] - closed[moving])) <= GRIPPER_TOLERANCE:
        return 1.
    delta = (closed - opened)[moving]
    return float(np.clip(np.dot((raw - opened)[moving], delta) / np.dot(delta, delta), .001, .999))


def load_replay_class():
    """只加载 demo 的 ReplayBuffer 文件，避免引入相机、UR5 等依赖。"""
    path = Path(__file__).resolve().parent / 'dp_real_0309/diffusion_policy/common/replay_buffer.py'
    spec = importlib.util.spec_from_file_location('_nero_replay_buffer', path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError('录制依赖缺失，请运行 pip install -r requirements.txt') from exc
    return module.ReplayBuffer


def build_episode(observations, actions, start, wall_start, stop):
    """观测/动作分别按过去最近样本对齐到同一时间网格，绝不使用未来样本。"""
    if not observations or not actions:
        return {}
    obs_times = np.array([row[0] for row in observations])
    action_times = np.array([row[0] for row in actions])
    # 网格从 B 后 0.1 秒开始；最后一帧不晚于最后实际观测和停止时刻。
    count = max(0, int(np.floor((min(stop, obs_times[-1]) - start) * RECORD_HZ + 1e-6)))
    grid = start + np.arange(1, count + 1) / RECORD_HZ
    oi = np.searchsorted(obs_times, grid, side='right') - 1
    ai = np.searchsorted(action_times, grid, side='right') - 1
    valid = (oi >= 0) & (ai >= 0) & (grid - obs_times[np.maximum(oi, 0)] <= MAX_SAMPLE_AGE)
    bad = np.flatnonzero(~valid)
    if len(bad):
        print('录制存在采样空缺，仅保存空缺之前的连续片段。')
        grid, oi, ai = grid[:bad[0]], oi[:bad[0]], ai[:bad[0]]
    if not len(grid):
        return {}
    offset = wall_start - start  # 单次固定时钟映射；系统校时不改变 10 Hz 网格。
    return dict(
        timestamp=grid + offset,
        robot_joint=np.array([observations[i][1] for i in oi], dtype=np.float64),
        robot_gripper=np.array([[observations[i][2]] for i in oi], dtype=np.float64),
        action=np.array([actions[i][1] for i in ai], dtype=np.float64),
        stage=np.zeros(len(grid), dtype=np.int64),
        observation_timestamp=obs_times[oi] + offset,
        action_timestamp=action_times[ai] + offset,
        robot_joint_timestamp=np.array([observations[i][3] for i in oi]),
        gripper_read_timestamp=np.array([observations[i][4] for i in oi]),
    )


class EpisodeRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.replay = None
        self.recording = False
        self.last_command = None
        self.gripper_target = None  # 启动时不猜测手部状态，首次 O/C 后才允许录制。
        self.observations, self.actions = [], []

    def command(self, joints):
        """只在 move_js 成功返回后调用，保留最后一次有效目标。"""
        self.last_command = np.asarray(joints, dtype=float).copy()
        if self.recording:
            self.actions.append((time.monotonic(), np.r_[self.last_command, self.gripper_target]))

    def gripper(self, opened):
        """记录已经发送的夹爪目标；目标 1 与实际闭合到位状态分开保存。"""
        self.gripper_target = float(not opened)
        if self.recording and self.last_command is not None:
            self.command(self.last_command)

    def key(self, key):
        if key == 'b':
            self.start()
        elif key == 's':
            self.stop()

    def start(self):
        if self.recording:
            return
        if self.last_command is None or self.gripper_target is None:
            print('尚未开始录制：请先完成对齐/跟随，并按一次 O 或 C，再按 B。')
            return
        if self.replay is None:
            replay_class = load_replay_class()
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / 'videos').mkdir(exist_ok=True)  # 与 demo 目录结构兼容，本任务不录视频。
            self.replay = replay_class.create_from_path(str(self.output / 'replay_buffer.zarr'), mode='a')
        self.started, self.wall_start = time.monotonic(), time.time()
        self.observations = []
        self.actions = [(self.started, np.r_[self.last_command, self.gripper_target])]
        self.recording = True
        print(f'开始录制 episode {self.replay.n_episodes}，10 Hz；S 保存，Q 退出并保存。')

    def capture(self, arm, hand):
        """控制循环采集，保存时重采样为 10 Hz；失败时结束片段，遥操仍可继续。"""
        if not self.recording:
            return
        try:
            messages = arm.get_raw_motor_states()
            joints = np.array([message.msg.position for message in messages], dtype=float)
            stamps = np.array([message.timestamp for message in messages], dtype=float)
            now = time.time()
            if (joints.shape != (7,) or not np.isfinite(joints).all()
                    or stamps.shape != (7,) or not np.isfinite(stamps).all()
                    or np.any(now - stamps > .2) or np.any(stamps > now + .01)
                    or np.ptp(stamps) > .05):
                raise RuntimeError('七轴反馈缺失、过期或帧间偏差超过 50 ms')
            grip = gripper_state(hand.get_joint_positions_raw())
            # G20 无位置接收时间戳：此处仅标记主机完成读取的时间。
            sampled = time.monotonic()
            self.observations.append((sampled, joints, grip, stamps,
                                      self.wall_start + sampled - self.started))
        except Exception as exc:
            print(f'录制采集失败，结束当前片段：{exc}')
            self.stop()

    def stop(self):
        """S、Q、Ctrl+C 和异常退出共用；空 episode 不写入。"""
        if not self.recording:
            return
        self.recording = False
        episode = build_episode(self.observations, self.actions, self.started, self.wall_start, time.monotonic())
        if episode:
            # 首次写入后只向同一 schema 追加，避免与其他任务的数据混写。
            if self.replay.n_episodes and (
                    set(self.replay.data.keys()) != set(episode)
                    or any(self.replay[key].shape[1:] != value.shape[1:]
                           or self.replay[key].dtype != value.dtype for key, value in episode.items())):
                raise ValueError('输出目录已有其他数据格式，请使用新的 --output 目录')
            self.replay.add_episode(episode, compressors='disk')
            print(f'已保存 episode {self.replay.n_episodes - 1}，{len(episode["timestamp"])} 帧：{self.output}')
        else:
            print('片段太短或没有有效对齐数据，未保存。')
