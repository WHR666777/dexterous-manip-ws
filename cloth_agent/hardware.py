"""控制进程内的右臂设备与运动执行。

普通 I/O、轨迹检查、图像保存运行在普通服务线程；独立轨迹线程保持固定发送周期。
人工停止由控制主线程独立锁定，
读取一次实测关节作为固定目标，并周期性发送 JS 位置保持；不调用电子阻尼急停。
SDK 已经执行中的调用不能撤回；软件保持不构成硬件停止时限或失电保持保证。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import Lock, RLock
import time
import uuid

import numpy as np


class RobotController:
    def __init__(self, config, directory, manual_stop, failed, allow_hardware):
        self.config, self.directory = config, Path(directory)
        self.manual_stop, self.failed = manual_stop, failed
        self.arm = self.hand = self.camera = self.validator = None
        self.hand_feedback = None
        self._feedback_ik = None
        self._feedback_fk_lock = Lock()
        self.connected = False
        self.active_cell = None
        self.ended = False
        self.cell_paused = False
        self.fault = ""
        self.motion = None
        self._lock = Lock()  # 只保护少量内存状态；绝不在持锁时做 SDK/磁盘 I/O。
        # 只串行化一次 SDK 关节发送，避免普通轨迹七轴帧与人工保持帧交错。
        # 不保护规划/相机/手部/落盘；人工先置位 Event，等待中的普通发送重查后放弃。
        self._arm_send_lock = Lock()
        # 轨迹线程与普通请求只在“换目标/暂停”时共享此锁。相机、手部查询、
        # 轨迹校验及文件保存均在锁外；人工保持只使用上面的单次发送锁。
        self._trajectory_lock = RLock()
        self.on_control_sample = None  # 控制进程注入内存队列入口；周期内不写磁盘。
        self._last_send_at = None
        self._joints = None
        self._tcp = None
        self._hand_feedback = None
        self._hand_error = "手部未连接 Hand not connected"
        self._hand_target = None
        self._sampled_at = None
        self._hand_read_at = None
        self._hold = None
        self._stable_since = None
        self._hold_feedback = None
        self._hold_confirmed = False
        self.manual_hold_report = None
        # 人工保持目标只捕获一次，重复按停止不能让目标跟着漂移的机械臂不断下移。
        # 此目标和反馈稳定计时仅由独立人工处理主线程写入。
        self._manual_target = None
        self._manual_feedback = None
        self._manual_stable_since = None
        self._manual_confirmed = False
        self.settings = (config or {}).get("arms", {}).get("right", {})
        self.motion_settings = (config or {}).get("motion", {})
        self.period = 1. / (config or {}).get("control_hz", 20.)
        if allow_hardware:
            try:
                self._connect()
            except BaseException:
                self.close()
                raise

    def _connect(self):
        # 只在 --execute 控制子进程内连接，不使能、不 reset。回起始位姿由宿主在
        # 人工 socket/worker/记录就绪后组织，不能在设备初始化里直接移动。
        from robot_control.nero import NeroArm
        from robot_control.nero_ik import NeroIK, pose_error
        from robot_control.l20 import LinkerHandL20
        from .geometry import transform
        from .planning import Planner
        from teleop.hand_feedback import HandFeedbackPoller

        self.validator = Planner(self.config)
        # Pinocchio 的 Data 会被 forward 原地更新。周期反馈不能与普通线程的
        # 整条路径校验共用同一份 Data；人工/普通反馈之间只串行一次很短的 FK。
        self._feedback_ik = NeroIK()
        self.arm = NeroArm(can_channel=self.settings["can_channel"],
                           max_joint_delta=self.motion_settings["max_ik_step_rad"])
        self.arm.connect()
        # connect() 只启动 SDK 后台 CAN 接收线程，不保证低速驱动状态已经到齐。
        # 官方 get_joint_enable_status() 把“尚无反馈”也表示为 False，所以使能
        # 确认必须与关节/位姿反馈一起等待，不能刚连接就认定物理机械臂未使能。
        # 此循环只读取反馈；绝不调用 enable/set_normal_mode/reset 去改变现场状态。
        deadline = time.monotonic() + 3
        while True:
            try:
                if not self.arm.is_enabled():
                    raise RuntimeError("尚未确认七轴使能 All-joint enable feedback not confirmed; "
                                       "SDK also returns False when feedback is missing")
                q = self.arm.get_joint_positions()
                sdk_pose = transform(self.arm.get_flange_pose())
                break
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    # 超时只能说明本连接没有确认所需反馈；不能据此断言机械臂未使能。
                    raise RuntimeError(
                        f"右臂启动反馈等待超时 Nero startup feedback timed out after 3 s; "
                        f"CAN={self.settings['can_channel']}; last feedback: {exc}"
                    ) from exc
                time.sleep(.05)
        position_error, angle_error = pose_error(self.validator.ik["right"].forward(q), sdk_pose)
        if position_error > .01 or angle_error > np.deg2rad(5):
            raise RuntimeError(f"FK/SDK 预检不一致 Preflight mismatch: {position_error} m, {angle_error} rad")
        self.hand = LinkerHandL20(hand_type="right", can_channel=self.settings["hand_can_channel"])
        self.hand.connect()
        if self.settings.get("hand_speed") is not None:
            self.hand.set_speed(self.settings["hand_speed"])
        # 与已正常工作的遥操作复用相同的反馈轮询器。L20 的 fresh=True 会
        # 发请求并等待，不能放到机械臂 JS 目标发送的周期里。
        self.hand_feedback = HandFeedbackPoller(self.hand, poll_hz=4., max_age_s=1.)
        self.hand_feedback.start()
        # 相机封装独立后台采集；普通请求只取最新完整快照，不阻塞人工请求接收。
        if self.config.get("camera", {}).get("enabled"):
            from teleop.realsense_camera import RealSenseCamera

            self.camera = RealSenseCamera(self.config["camera"])
            self.camera.start()
        self.connected = True
        self._measure()

    def _measure(self):
        if not self.connected:
            return
        q = self.arm.get_joint_positions()
        tcp = self._feedback_tcp(q)
        now = time.monotonic()
        hand_feedback, hand_error = self._hand_feedback, self._hand_error
        # 这里只拷贝后台查询结果，不做手部 CAN 请求，也不等待五指查询。
        # query_finished_at 是主机查询结束时间，不代表设备反馈 generation。
        try:
            feedback = self.hand_feedback.get_latest()
            hand_feedback, hand_error = feedback.positions_raw.tolist(), None
            self._hand_read_at = feedback.query_finished_at
        except RuntimeError as exc:
            hand_feedback, hand_error = None, str(exc)
        with self._lock:
            self._joints, self._tcp, self._sampled_at = q.copy(), tcp.tolist(), now
            self._hand_feedback, self._hand_error = hand_feedback, hand_error

    def _feedback_tcp(self, q):
        from .geometry import transform

        with self._feedback_fk_lock:
            flange = self._feedback_ik.forward(q)
        return (transform(self.settings["task_from_base"]) @ flange
                @ transform(self.settings["flange_from_tcp"]))

    def _motion_result(self):
        if self.motion is None:
            return None
        return {name: deepcopy(self.motion[name]) for name in
                ("status", "reached", "reason", "motion_id", "joint_error_rad", "actual_joints")}

    def snapshot(self):
        with self._lock:
            return {"hardware_connected": self.connected, "joints":
                    self._joints.tolist() if self._joints is not None else None,
                    "tcp": deepcopy(self._tcp), "sampled_at": self._sampled_at,
                    "hand": {"target_raw": deepcopy(self._hand_target),
                             "position_raw": deepcopy(self._hand_feedback),
                             "read_at": self._hand_read_at, "freshness": "unknown",
                             "error": self._hand_error},
                    "motion": self._motion_result(), "fault": self.fault,
                    "manual_hold_latched": self.manual_stop.is_set(),
                    "manual_hold_report": deepcopy(self.manual_hold_report),
                    "manual_hold_target": (self._manual_target.tolist()
                                           if self._manual_target is not None else None),
                    "hold_target": self._hold.tolist() if self._hold is not None else None,
                    "hold_error_rad": (float(np.max(np.abs(self._joints - self._hold)))
                                       if self._joints is not None and self._hold is not None else None),
                    "hold_confirmed": (self._manual_confirmed if self.manual_stop.is_set()
                                       else self._hold_confirmed), "stop_confirmed": False,
                    "quiescent": not self.manual_stop.is_set() and not self.fault
                    and (self.motion is None or self.motion["status"] != "running")
                    and (self._hold is None or self._hold_confirmed)}

    def _admit(self, cell_id):
        if (self.manual_stop.is_set() or self.ended or self.fault
                or self.cell_paused or cell_id != self.active_cell):
            raise ValueError("动作权限已失效 Robot action admission revoked")

    def _require_hardware(self):
        if not self.connected:
            raise RuntimeError("设备未连接 Hardware not connected; use operator --execute gate")

    def _send(self, q):
        # 人工请求不进入普通请求队列。物理发送必须串行化，锁内再次检查，
        # 防止等锁的旧目标在人工目标之后才发送；已进入 SDK 的调用仍无法撤回。
        with self._arm_send_lock:
            if not self.manual_stop.is_set():
                started = time.monotonic()
                self.arm.move_js(q)
                sent_at = time.monotonic()
                if self.on_control_sample is not None:
                    self.on_control_sample({
                        "sent_at": sent_at,
                        "send_interval_s": (None if self._last_send_at is None
                                            else sent_at - self._last_send_at),
                        "send_duration_s": sent_at - started,
                        "target_joints": np.asarray(q).tolist(),
                        "actual_joints": self._joints.tolist(),
                        "tcp": deepcopy(self._tcp), "sampled_at": self._sampled_at,
                        "motion_id": self.motion["motion_id"] if self.motion else None,
                    })
                self._last_send_at = sent_at

    def pause(self, reason):
        with self._trajectory_lock:
            return self._pause(reason)

    def _pause(self, reason):
        self.cell_paused = True
        with self._lock:
            if self.motion and self.motion["status"] == "running":
                self.motion.update(status="paused", reached=False, reason=reason,
                                   actual_joints=self._joints.tolist() if self._joints is not None else None)
        if self.connected and not self.manual_stop.is_set():
            # 读真实反馈作为新的固定目标；不继续原轨迹，也不松开 L20。
            q = self.arm.get_joint_positions()
            self._hold = q.copy()
            self._hold_confirmed = False
            self._stable_since = None
            self._hold_feedback = None
            self._send(q)
        return self.snapshot()

    def manual_hold(self, reason):
        """人工停止先撤权，再捕获固定关节目标；保持循环不依赖普通请求线程。"""
        first_request = not self.manual_stop.is_set()
        self.manual_stop.set()
        with self._lock:
            self.cell_paused = True
            if self.motion:
                self.motion.update(status="manual_hold", reached=False, reason=reason)
            self._hold_confirmed = False
        if not first_request:
            return self.snapshot()  # 幂等：保留第一次的目标及诊断，不重新捕获当前位置。
        report = {"requested_at": time.monotonic(), "reason": reason,
                  "right": "not_connected", "mode": "fixed_joint_position",
                  "stop_confirmed": False, "l20": "no_release_or_reset_command"}
        if self.connected:
            try:
                # 使用带新鲜度检查的七轴实测值，不使用旧轨迹目标或部分填零缓存。
                # 捕获失败时不在后续周期重抓目标，也不自动切换成阻尼急停。
                self._manual_target = self.arm.get_joint_positions().copy()
                report["right"] = "target_captured"
            except Exception as exc:
                report["right"] = "hold_failed"
                report["error"] = str(exc)
        with self._lock:
            self.manual_hold_report = report
        self.manual_hold_tick()
        return self.snapshot()

    def manual_hold_tick(self):
        """独立人工路径持续重发同一目标，并记录有限的关节反馈稳定证据。"""
        if not self.manual_stop.is_set() or self._manual_target is None:
            return
        try:
            # 普通 I/O 故障或 worker/Codex 退出也不停止此路径。move_js 仍校验
            # 使能/限位/反馈及 CAN 错误；不会自动使能、清错或放开电机。
            with self._arm_send_lock:
                self.arm.move_js(self._manual_target)
            q = self.arm.get_joint_positions()
            tcp = self._feedback_tcp(q)
            error = float(np.max(np.abs(q - self._manual_target)))
            stable = (self._manual_feedback is not None
                      and np.max(np.abs(q - self._manual_feedback)) <= .002)
            self._manual_feedback = q.copy()
            now = time.monotonic()
            if error <= self.motion_settings["joint_tolerance_rad"] and stable:
                self._manual_stable_since = self._manual_stable_since or now
                self._manual_confirmed = now - self._manual_stable_since >= .2
            else:
                self._manual_stable_since, self._manual_confirmed = None, False
            with self._lock:
                self._joints, self._tcp, self._sampled_at = q.copy(), tcp.tolist(), now
                self.manual_hold_report.update(right="hold_target_sent", error=None,
                                               joint_error_rad=error, last_send_at=now)
        except Exception as exc:
            # 反馈失效/使能丢失/CAN 故障时不能声称悬停。保留固定目标供下周期
            # 重试；绝不退化调用 electronic_emergency_stop，也不追随下落重抓目标。
            self._manual_stable_since, self._manual_confirmed = None, False
            with self._lock:
                self.manual_hold_report.update(right="hold_failed", error=str(exc))

    def tick(self, delta):
        """独立轨迹线程推进一次；RPC、图像落盘和手部查询不能改变下发周期。"""
        with self._trajectory_lock:
            self._tick(delta)

    def _tick(self, delta):
        if not self.connected or self.manual_stop.is_set():
            return
        self._measure()
        motion = self.motion
        if motion and motion["status"] == "running" and not self.fault:
            motion["elapsed"] += min(delta, self.period)  # 慢周期延长轨迹，不突发补发。
            self._hold = motion["plan"].sample(motion["elapsed"])
            self._send(self._hold)
            if motion["elapsed"] >= motion["plan"].duration:
                error = float(np.max(np.abs(self._joints - motion["plan"].positions[-1])))
                # 轨迹时间结束不等于机械臂已稳定。终点保持连续确认后才完成句柄，
                # 避免下一段使用仍在变化的反馈作为规划起点。
                self._confirm_hold(error)
                if self._hold_confirmed:
                    with self._lock:
                        if not self.manual_stop.is_set() and motion["status"] == "running":
                            motion.update(status="completed", reached=True, joint_error_rad=error,
                                          actual_joints=self._joints.tolist())
                elif motion["elapsed"] > motion["plan"].duration + self.motion_settings["completion_timeout_s"]:
                    # 超时只报告失败并停止推进，持续保持原终点；不把下坠后的
                    # 实测位置重新捕获为目标，避免再次向下迁移保持点。
                    self.cell_paused = True
                    with self._lock:
                        if not self.manual_stop.is_set():
                            motion.update(status="failed", reason="机械臂到位超时 Motion arrival timed out",
                                          joint_error_rad=error)
        elif self._hold is not None:
            self._send(self._hold)
            error = float(np.max(np.abs(self._joints - self._hold)))
            self._confirm_hold(error)

    def _confirm_hold(self, error):
        """统一确认终点到位与普通保持；只用反馈，不放宽既有关节/稳定容差。"""
        stable = (self._hold_feedback is not None
                  and np.max(np.abs(self._joints - self._hold_feedback)) <= .002)
        self._hold_feedback = self._joints.copy()
        with self._lock:
            if self.manual_stop.is_set():
                return
            if error <= self.motion_settings["joint_tolerance_rad"] and stable:
                self._stable_since = self._stable_since or time.monotonic()
                self._hold_confirmed = time.monotonic() - self._stable_since >= .2
            else:
                self._stable_since, self._hold_confirmed = None, False

    def fail(self, reason):
        with self._trajectory_lock:
            try:
                # 已有发送目标时保留该固定目标，避免每次故障重抓实测值。
                # 尚未发运动目标才捕获一次；后续周期继续尝试原位保持。
                self.cell_paused = True
                if self.connected and self._hold is None:
                    self._hold = self.arm.get_joint_positions().copy()
            except Exception as exc:
                reason += f"；保持请求失败 Hold request failed: {exc}"
            with self._lock:
                self.fault = reason
                self._hold_confirmed = False
                if self.motion:
                    self.motion.update(status="failed", reached=False, reason=reason)
        self.failed.set()

    def _start_motion(self, cell_id, payload):
        from .planning import MotionPlan

        self._require_hardware()
        plan = MotionPlan.from_dict(payload)
        if plan.arm != "right":
            raise ValueError("第一版只允许右臂 Only right arm is enabled")
        self.validator._check_path("right", plan.positions)
        sampled = plan.sample(np.linspace(0, plan.duration, max(100, len(plan.times) * 2)))
        self.validator._check_path("right", sampled)
        velocity, acceleration = plan.peak_rates()
        if (np.any(velocity > np.asarray(self.settings["velocity_limits"]) * 1.00001)
                or np.any(acceleration > np.asarray(self.settings["acceleration_limits"]) * 1.00001)):
            raise ValueError("轨迹超过配置上限 Trajectory exceeds configured limits")
        motion_id = uuid.uuid4().hex
        plans = self.directory / "motions"
        plans.mkdir(exist_ok=True)
        (plans / f"{motion_id}.json").write_text(json.dumps(plan.as_dict()), encoding="utf-8")
        with self._trajectory_lock:
            # 耗时的路径校验和落盘结束后，再复核实测起点并原子发布新轨迹。
            # 轨迹线程不能在复核与发布之间继续推进上一段运动。
            q = self.arm.get_joint_positions()
            seed_error = float(np.max(np.abs(q - plan.positions[0])))
            if seed_error > self.motion_settings["joint_tolerance_rad"]:
                raise ValueError(
                    "轨迹起点已变化 Recompute motion from current feedback; "
                    f"max_error_rad={seed_error}, tolerance_rad={self.motion_settings['joint_tolerance_rad']}, "
                    f"planned_start={plan.positions[0].tolist()}, actual_joints={q.tolist()}"
                )
            with self._lock:
                self._admit(cell_id)
                if self.motion and self.motion["status"] == "running":
                    raise ValueError("已有运动正在执行 A motion is already running")
                self._hold_confirmed, self._stable_since = False, None
                self._hold_feedback = None
                self.motion = {"plan": plan, "elapsed": 0., "status": "running", "reached": False,
                               "motion_id": motion_id, "reason": "", "joint_error_rad": None,
                               "actual_joints": q.tolist()}
                return self._motion_result()

    def _observe(self):
        if self.camera is None:
            raise ValueError("相机未连接 Camera not connected")
        frame = self.camera.get_latest()
        if frame is None:
            raise RuntimeError("RGB-D 缺失或过期 RGB-D unavailable/stale")
        import cv2

        observation_id = uuid.uuid4().hex
        directory = self.directory / "observations"
        directory.mkdir(exist_ok=True)
        rgb_path = directory / f"{observation_id}.png"
        if not cv2.imwrite(str(rgb_path), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("图像保存失败 RGB snapshot could not be saved")
        depth = frame.depth.astype(np.float32) * self.camera.metadata["depth_scale_m"]
        np.save(directory / f"{observation_id}-depth.npy", depth, allow_pickle=False)
        state = self.snapshot()
        metadata = {"id": observation_id, "intrinsics": deepcopy(self.camera.metadata["color_intrinsics"]),
                    "task_from_camera": self.config["observations"].get("task_from_camera"),
                    "timestamps": {"received_at": frame.received_at,
                                   "color_timestamp_ms": frame.color_timestamp_ms,
                                   "depth_timestamp_ms": frame.depth_timestamp_ms,
                                   "color_frame_number": frame.color_frame_number,
                                   "depth_frame_number": frame.depth_frame_number,
                                   "color_timestamp_domain": frame.color_timestamp_domain,
                                   "depth_timestamp_domain": frame.depth_timestamp_domain},
                    "state": state, "_rgb_path": str(rgb_path)}
        (directory / f"{observation_id}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**metadata, "rgb": frame.color.copy(), "depth_m": depth}

    def dispatch(self, request, *, owner):
        operation = request["operation"]
        if owner:
            with self._trajectory_lock:
                return self._dispatch_owner(request)
        return self._dispatch_worker(request)

    def _dispatch_owner(self, request):
        operation = request["operation"]
        interrupted = False
        if operation == "begin_cell":
            if self.ended or self.manual_stop.is_set() or self.fault or not self.snapshot()["quiescent"]:
                raise ValueError("控制状态不允许新 cell Robot is not ready for a new cell")
            self.active_cell, self.cell_paused = request["cell_id"], False
        elif operation in {"cell_end", "end_task"}:
            self.active_cell = None
            if operation == "end_task":
                self.ended = True
            if self.motion and self.motion["status"] == "running":
                interrupted = True
                self.pause(request.get("reason", "cell 已结束 Cell ended"))
            # 已结束的轨迹保留最后一个固定目标；收尾不重新捕获偏离后的关节，
            # 由协调器检查保持确认并报告失败，不能用换目标掩盖跟踪偏差。
        elif operation != "status":
            raise ValueError("未知协调请求 Unknown coordinator request")
        return {**self.snapshot(), "cell_interrupted": interrupted}

    def _dispatch_worker(self, request):
        operation = request["operation"]
        cell_id = request.get("cell_id")
        if cell_id != self.active_cell or self.ended or self.manual_stop.is_set():
            raise ValueError("cell 已失效 Cell admission revoked")
        if operation == "state":
            # 反馈由轨迹线程固定频率刷新；状态请求不再额外推进/阻塞控制周期。
            return self.snapshot()
        if operation == "observe":
            return self._observe()
        if operation == "motion_status":
            if not self.motion or self.motion["motion_id"] != request["motion_id"]:
                raise ValueError("未知运动句柄 Unknown motion handle")
            return self.snapshot()["motion"]
        if operation == "pause":
            return self.pause(request["reason"])
        self._admit(cell_id)
        if operation == "start_motion":
            return self._start_motion(cell_id, request["plan"])
        if operation == "set_gripper":
            self._require_hardware()
            preset = request["preset"]
            if preset not in {"open", "closed"}:
                raise ValueError("未知手型 Unknown gripper preset")
            raw = self.settings["hand_poses"][preset]
            self._admit(cell_id)
            self.hand.set_thumb_index_positions_raw(raw)
            self._hand_target = list(raw)
            self._hand_read_at = None
            return {"command_status": "sdk_returned", "preset": preset,
                    "commanded_fingers": ["thumb", "index"],
                    "delivery_confirmed": False, "hand": self.snapshot()["hand"]}
        raise ValueError("未知 robo 操作 Unknown robo operation")

    def close(self):
        # 清理不 reset、不失能、不松手；先由协调器请求普通暂停，再结束设备连接。
        errors = []
        for device, operation in ((self.camera, "stop"), (self.hand_feedback, "stop"),
                                  (self.hand, "disconnect"), (self.arm, "disconnect")):
            if device is not None:
                try:
                    getattr(device, operation)()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("设备清理失败 Device cleanup failed: " + "; ".join(errors))
