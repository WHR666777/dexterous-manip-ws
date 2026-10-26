"""worker 中的右臂机器人代理：计算留在本进程，设备操作通过 Pipe 请求控制进程。

只公开收敛后的基础能力。固定机器人知识随启动 prompt 提供，没有 info/plan/execute
或抓布判定接口；MotionPlan 只在 move_J/move_L 内部使用，不交给生成代码管理。
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

    def __init__(self, robo, motion_id, cell_id):
        self._robo, self._motion_id, self._cell_id = robo, motion_id, cell_id

    def status(self):
        if self._robo._cell_id != self._cell_id:
            raise RuntimeError("旧 cell 的运动句柄已失效 Motion handle belongs to an earlier cell")
        return self._robo._call("motion_status", motion_id=self._motion_id)

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
    """注入 worker globals 的代理对象。第一版固定使用已配置的右臂和右手。"""

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
                            "ok": reply["ok"], "reason": reply.get("reason", "")})
        if not reply["ok"]:
            raise RuntimeError(reply["reason"])
        return reply["result"]

    def state(self):
        """读取关节、虚拟 TCP、手反馈、运动及控制状态 Read current robot state."""
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

    def top_down_pose(self, xyz, yaw=0.):
        """构造虚拟 TCP 位姿；yaw 为相对初始抓取朝向绕 task +Z 的增量，rad。"""
        if self._config is None:
            raise RuntimeError("尚未加载 TCP 标定 Robot calibration not loaded")
        point = np.asarray(xyz, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or np.ndim(yaw) != 0 or not np.isfinite(yaw)):
            raise ValueError("位置或 yaw 无效 Invalid xyz/yaw")
        pose = np.eye(4)
        pose[:3, 3] = point
        pose[:3, :3] = (Rotation.from_euler("z", yaw).as_matrix()
                           @ np.asarray(self._config["arms"]["right"]["reference_tcp_rotation"]))
        return pose

    def _startup_move_ptp(self, *, target_joints=None, target_tcp=None):
        """仅供宿主固定 cell 000 调用，不增加 agent 公开运动 API。"""
        if self._cell_id != 0:
            raise RuntimeError("PTP 仅用于启动准备 Startup PTP requires cell 000")
        return self._move(target_joints=target_joints, target_tcp=target_tcp, startup_ptp=True)

    def _move(self, *, target_joints=None, target_tcp=None, wait=True, startup_ptp=False):
        if not isinstance(wait, bool):
            raise ValueError("wait 必须为布尔值 wait must be bool")
        state = self.state()
        if not state["hardware_connected"]:
            raise RuntimeError("设备未连接 Hardware not connected")
        if self._planner is None:
            from .planning import Planner

            self._planner = Planner(self._config)
        from .planning import PlanningError

        try:
            # IK/TOPPRA 在 worker 内计算；计算期间不占住设备进程的人工停止保持路径。
            # 执行时控制进程仍会复核起点、限位和轨迹，不把持久 globals 当成现场真值。
            if startup_ptp:
                plan = self._planner.plan_startup_ptp(state["joints"], target_joints=target_joints,
                                                     target_tcp=target_tcp)
            else:
                plan = self._planner.plan("right", state["joints"], target_joints=target_joints,
                                          waypoints=[transform(target_tcp)] if target_tcp is not None else None,
                                          mode="cartesian" if target_tcp is not None else "joint")
        except PlanningError as exc:
            return MotionResult("planning_failed", False, str(exc), diagnostics=exc.as_dict())
        started = self._call("start_motion", plan=plan.as_dict())
        handle = MotionHandle(self, started["motion_id"], self._cell_id)
        return handle.wait() if wait else handle

    def move_J(self, target_joints, wait=True):
        """七轴关节目标，rad；几何分段以普通 J 逐段到位 Joint point-to-point."""
        return self._move(target_joints=target_joints, wait=wait)

    def move_L(self, target_tcp, wait=True):
        """task 系虚拟 TCP 直线插值，普通 J 分段近似；SE(3)/xyz-rpy，m/rad；规划失败停止。"""
        return self._move(target_tcp=target_tcp, wait=wait)

    def open_gripper(self):
        """使用配置中的张开姿态，只发送拇指和食指 Open thumb/index."""
        return self._call("set_gripper", preset="open")

    def close_gripper(self):
        """使用 replay 标定闭合姿态 Close thumb/index; no grasp-success judgement."""
        return self._call("set_gripper", preset="closed")

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
