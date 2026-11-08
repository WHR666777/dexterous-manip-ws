"""worker 中的双臂机器人代理：计算留在本进程，设备操作通过 Pipe 请求控制进程。

只公开收敛后的基础能力。固定机器人知识随启动 prompt 提供，没有 info/plan/execute
或抓布判定接口；运动计划只在 move_J/move_L 内部使用，不交给生成代码管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import calculate, transform


class CellPaused(BaseException):
    """普通暂停已完成收尾，展开当前 cell 调用栈，将下一步决策交还 Codex。"""


@dataclass(frozen=True)
class Observation:
    """一份固定 RGB-D 快照；时间戳分别记录，不宣称相机与机器人硬件同步。"""

    id: str
    rgb: np.ndarray = field(repr=False)
    depth_m: np.ndarray = field(repr=False)
    intrinsics: dict
    task_from_camera: list | None
    timestamps: dict
    state: dict
    _rgb_path: str = field(repr=False)


@dataclass(frozen=True)
class MotionResult:
    """运动反馈；completed/reached 仅描述到位，不能推断抓布或折叠成功。"""

    status: str
    reached: bool
    reason: str = ""
    motion_id: str | None = None
    joint_error_rad: float | None = None
    actual_joints: list | None = None
    diagnostics: dict = field(default_factory=dict)


class MotionHandle:
    """异步运动句柄只属于创建它的 cell，不能跨 cell 继续监督旧动作。"""

    def __init__(self, robo, motion_id, cell_id, arm="right"):
        self._robo, self._motion_id, self._cell_id = robo, motion_id, cell_id
        self._arm = arm

    def status(self):
        if self._robo._cell_id != self._cell_id:
            raise RuntimeError("旧 cell 的运动句柄已失效 Motion handle belongs to an earlier cell")
        return self._robo._call("motion_status", motion_id=self._motion_id, arm=self._arm)

    def done(self):
        return self.status()["status"] != "running"

    def wait(self):
        # 每次只发短查询，控制进程仍可推进轨迹；没有把 wait 塞进普通 RPC 队列中阻塞。
        while True:
            result = self.status()
            if result["status"] != "running":
                return MotionResult(**result)
            time.sleep(.05)


class Robo:
    """注入 worker globals 的代理对象；省略 arm 时沿用右侧。"""

    def __init__(self, connection, config):
        self._connection, self._config = connection, config
        self._planner = None
        self._cell_id = None
        self._shows, self._calls = [], []

    def _begin_cell(self, cell_id):
        self._cell_id = cell_id
        self._shows, self._calls = [], []

    def _call(self, operation, **values):
        started = time.monotonic()
        # 不传 Python 回调或任意源码。控制进程只接受明确操作、数据和活动 cell 标识。
        self._connection.send({"operation": operation, "cell_id": self._cell_id, **values})
        reply = self._connection.recv()
        self._calls.append({"operation": operation, "elapsed_s": time.monotonic() - started,
                            "ok": reply["ok"], "reason": reply.get("reason", ""),
                            **({"arm": values["arm"]} if "arm" in values else {}),
                            **({"arms": list(values["plans"])} if "plans" in values else {})})
        if not reply["ok"]:
            raise RuntimeError(reply["reason"])
        return reply["result"]

    def _check_arm(self, arm):
        if self._config is None or arm not in self._config["arms"]:
            raise ValueError(f"未配置机械臂 Arm is not configured: {arm}")

    def state(self, arm=None):
        """省略 arm 返回全局状态及 arms；指定侧返回该臂关节/TCP/手/运动。"""
        if arm is not None:
            self._check_arm(arm)
            return self._call("state", arm=arm)
        return self._call("state")

    def observe(self):
        """读取一份固定 RGB-D 与关联状态 Capture an identified observation."""
        values = self._call("observe")
        # 限制代码意外原地改写原始快照；需要图像加工时可显式复制 rgb 数组。
        values["rgb"].setflags(write=False)
        values["depth_m"].setflags(write=False)
        return Observation(**values)

    def show(self, observation, crop=None):
        """将指定快照附给下一次模型反馈 Attach image; crop=[u0,v0,u1,v1]."""
        if not isinstance(observation, Observation):
            raise TypeError("需要 observe 返回的快照 Expected an Observation")
        if crop is not None:
            box = np.asarray(crop)
            height, width = observation.rgb.shape[:2]
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or np.any(box != np.floor(box))
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise ValueError("裁剪范围无效 Invalid crop [u0,v0,u1,v1]")
            crop = box.astype(int).tolist()
        self._shows.append({"observation_id": observation.id,
                            "rgb_path": observation._rgb_path, "crop": crop})

    def pixel_to_task(self, observation, pixel):
        """用同一快照的深度和标定反投影 Back-project [u,v] into task frame."""
        if not isinstance(observation, Observation):
            raise TypeError("需要 observe 返回的快照 Expected an Observation")
        uv = np.asarray(pixel, dtype=float)
        height, width = observation.depth_m.shape
        if (uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv != np.floor(uv))
                or not (0 <= uv[0] < width and 0 <= uv[1] < height)):
            raise ValueError("像素坐标无效 Invalid pixel [u,v]")
        u, v = uv.astype(int)
        # 无效深度由已有几何函数明确拒绝；不改读最新帧，也不静默取邻近桌面深度。
        return np.asarray(calculate("pixel_to_task", pixel=[int(u), int(v)],
                                    depth_m=float(observation.depth_m[v, u]),
                                    intrinsics=observation.intrinsics,
                                    task_from_camera=observation.task_from_camera))

    def top_down_pose(self, xyz, yaw=0., *, arm="right"):
        """构造虚拟 TCP 位姿；yaw 为相对初始抓取朝向绕 task +Z 的增量，rad。"""
        if self._config is None:
            raise RuntimeError("尚未加载 TCP 标定 Robot calibration not loaded")
        self._check_arm(arm)
        point = np.asarray(xyz, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or np.ndim(yaw) != 0 or not np.isfinite(yaw)):
            raise ValueError("位置或 yaw 无效 Invalid xyz/yaw")
        pose = np.eye(4)
        pose[:3, 3] = point
        pose[:3, :3] = (Rotation.from_euler("z", yaw).as_matrix()
                           @ np.asarray(self._config["arms"][arm]["reference_tcp_rotation"]))
        return pose

    def _startup_open_hand(self, *, arm="right"):
        """启动专用全手张开；普通 agent 的 open_gripper 仍只操作拇指/食指。"""
        if self._cell_id != 0:
            raise RuntimeError("全手初始化仅用于启动准备 Startup open_hand requires cell 000")
        self._check_arm(arm)
        return self._call("startup_open_hand", arm=arm)

    def _startup_move_ptp(self, *, target_joints=None, target_tcp=None, arm="right"):
        """仅供宿主固定 cell 000 调用，不增加 agent 公开运动 API。"""
        if self._cell_id != 0:
            raise RuntimeError("PTP 仅用于启动准备 Startup PTP requires cell 000")
        return self._move(target_joints=target_joints, target_tcp=target_tcp, startup_ptp=True, arm=arm)

    def _plan_motion(self, state, *, target_joints=None, target_tcp=None, startup_ptp=False,
                     ptp=False, arm="right"):
        self._check_arm(arm)
        if not state["hardware_connected"]:
            raise RuntimeError(f"{arm}: 设备未连接 Hardware not connected")
        if self._planner is None:
            from .planning import Planner

            self._planner = Planner(self._config)
        if startup_ptp:
            return self._planner.plan_startup_ptp(state["joints"], target_joints=target_joints,
                                                 target_tcp=target_tcp, arm=arm)
        if ptp:
            return self._planner.plan_ptp(state["joints"], target_tcp=target_tcp, arm=arm)
        return self._planner.plan_linear(state["joints"], target_tcp, arm=arm)

    def _move(self, *, target_joints=None, target_tcp=None, wait=True, startup_ptp=False, ptp=False,
              arm="right"):
        if not isinstance(wait, bool):
            raise ValueError("wait 必须为布尔值 wait must be bool")
        state = self.state(arm=arm)
        from .planning import PlanningError

        try:
            # J 的 IK 在 worker 内计算，L 只校验目标，内部规划交给固件。
            # 控制进程仍复核实际起点与目标，不把持久 globals 当成现场真值。
            plan = self._plan_motion(state, target_joints=target_joints, target_tcp=target_tcp,
                                     startup_ptp=startup_ptp, ptp=ptp, arm=arm)
        except PlanningError as exc:
            return MotionResult("planning_failed", False, str(exc), diagnostics=exc.as_dict())
        started = self._call("start_motion", plan=plan.as_dict(), arm=arm)
        handle = MotionHandle(self, started["motion_id"], self._cell_id, arm=arm)
        return handle.wait() if wait else handle

    def move_J(self, target_tcp, wait=True, *, arm="right"):
        """task 系虚拟 TCP 位姿；一次 IK + 关节 PTP，SE(3)/xyz-rpy，m/rad。"""
        return self._move(target_tcp=transform(target_tcp), wait=wait, ptp=True, arm=arm)

    def move_L(self, target_tcp, wait=True, *, arm="right"):
        """task 系 TCP 目标转法兰后使用厂家 move_l；SE(3)/xyz-rpy，m/rad。

        固定朝向时虚拟 TCP 随法兰沿直线平移；同时旋转时，带工具偏移的
        TCP 中途路径由固件的法兰运动决定，不保证该虚拟点仍严格沿直线。
        """
        return self._move(target_tcp=transform(target_tcp), wait=wait, arm=arm)

    def move_both(self, *, left, right, mode="L", wait=True):
        """两侧分别规划，一次提交并在同轮软件调度启动；返回按侧索引的结果/句柄。"""
        if mode not in {"J", "L"} or not isinstance(wait, bool):
            raise ValueError("mode 必须是 J/L，wait 必须为 bool")
        for arm in ("left", "right"):
            self._check_arm(arm)
        state = self.state()
        from .planning import PlanningError

        plans = {}
        try:
            for arm, target in (("left", left), ("right", right)):
                plans[arm] = self._plan_motion(state["arms"][arm], target_tcp=transform(target),
                                               ptp=mode == "J", arm=arm).as_dict()
        except PlanningError as exc:
            return {side: MotionResult("planning_failed", False, f"{arm}: {exc}",
                                       diagnostics=exc.as_dict()) for side in ("left", "right")}
        started = self._call("start_motions", plans=plans)
        handles = {arm: MotionHandle(self, result["motion_id"], self._cell_id, arm=arm)
                   for arm, result in started.items()}
        return {arm: handle.wait() for arm, handle in handles.items()} if wait else handles

    def open_gripper(self, *, arm="right"):
        """拇指/食指到配置张开姿态，阻塞检查到位并有限重试 Open and wait."""
        self._check_arm(arm)
        return self._call("set_gripper", preset="open", arm=arm)

    def close_gripper(self, *, arm="right"):
        """到标定闭合姿态，阻塞检查到位并有限重试；不判定抓布成功。"""
        self._check_arm(arm)
        return self._call("set_gripper", preset="closed", arm=arm)

    def gripper_replay_start(self, *, arm="right"):
        """到 CSV 首帧拇指/食指姿态并等待到位，不启动回放或设备 reset。"""
        self._check_arm(arm)
        return self._call("set_gripper", preset="replay_start", arm=arm)

    def pause_and_hold(self, reason="任务监督请求暂停 Task monitor requested pause"):
        """确认普通保持后结束本 cell；不是人工位置保持 Pause and return to agent."""
        state = self._call("pause", reason=str(reason))
        deadline = time.monotonic() + 3
        while not state["quiescent"]:
            if time.monotonic() >= deadline or state["fault"]:
                raise RuntimeError("暂停保持未确认 Hold not confirmed")
            time.sleep(.05)
            state = self.state()
        raise CellPaused(str(reason))
