"""控制进程内的单侧设备与运动执行，由双臂控制器分别实例化。

普通 I/O、轨迹检查、图像保存运行在普通服务线程；独立轨迹线程定期读取反馈。
人工停止由控制主线程独立锁定，
读取一次实测关节作为固定目标，发送普通 J 位置保持并监督；不调用电子阻尼急停。
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

from robot_control.errors import CommandRejected, OperationError


class RobotController:
    def __init__(self, config, directory, manual_stop, failed, allow_hardware, *, arm_name="right"):
        self.config, self.directory = config, Path(directory)
        self.arm_name = arm_name
        self.manual_stop, self.failed = manual_stop, failed
        self.arm = self.hand = self.camera = self.validator = None
        self.can_trace = None
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
        # 只串行化一次 SDK 目标发送，避免 J/L 目标帧与人工保持帧交错。
        # 不保护规划/相机/手部/落盘；人工先置位 Event，等待中的普通发送重查后放弃。
        self._arm_send_lock = Lock()
        # 轨迹线程与普通请求只在“换目标/暂停”时共享此锁。相机、手部查询、
        # 轨迹校验及文件保存均在锁外；人工保持只使用上面的单次发送锁。
        self._trajectory_lock = RLock()
        self.on_control_sample = None  # 控制进程注入内存队列入口；周期内不写磁盘。
        self._last_send_at = None
        self._last_send_wall_at = None
        self._joints = None
        self._tcp = None
        self._arm_status = None
        self._hand_feedback = None
        self._hand_error = "手部未连接 Hand not connected"
        self._hand_target = None
        self._sampled_at = None
        self._hand_read_at = None
        self._hold = None
        self._hold_sent = False
        self._hold_mode = 1  # 记录实际保持模式；显式暂停完成后的 L 也会改为 J。
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
        self._manual_sent = False
        self.settings = (config or {}).get("arms", {}).get(arm_name, {})
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
        from .can_trace import ArmCanTrace

        # 在 SDK 连接之前开始被动抓包，覆盖启动准备运动；不发送任何指令。
        self.can_trace = ArmCanTrace(self.settings["can_channel"], self.directory)
        self.arm.connect()
        self.arm.configure_can_sender(self.can_trace.sent,
                                      joint_frame_interval_s=self.motion_settings.get("joint_frame_interval_s", .001))
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
                arm_status = self._read_arm_status()  # 后续 J 到位监督也依赖状态帧，启动时一并等待。
                break
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    # 超时只能说明本连接没有确认所需反馈；不能据此断言机械臂未使能。
                    raise RuntimeError(
                        f"{self.arm_name} 启动反馈等待超时 Nero startup feedback timed out after 3 s; "
                        f"CAN={self.settings['can_channel']}; last feedback: {exc}"
                    ) from exc
                time.sleep(.05)
        # 已收到新鲜状态帧后立即解释固件故障，不把真实故障混成“反馈等待
        # 超时”，也不先连接 L20/相机再报错。此时尚未发任何机械臂运动目标。
        self._check_arm_status(arm_status, phase="连接预检 Startup preflight")
        position_error, angle_error = pose_error(self.validator.ik[self.arm_name].forward(q), sdk_pose)
        if position_error > .01 or angle_error > np.deg2rad(5):
            raise RuntimeError(f"FK/SDK 预检不一致 Preflight mismatch: {position_error} m, {angle_error} rad")
        self.hand = LinkerHandL20(hand_type=self.arm_name, can_channel=self.settings["hand_can_channel"])
        self.hand.connect()
        if self.settings.get("hand_speed") is not None:
            self.hand.set_speed(self.settings["hand_speed"])
        # 与已正常工作的遥操作复用相同的反馈轮询器。L20 的 fresh=True 会
        # 发请求并等待，不能放到机械臂的反馈监督周期里。
        act = self.config.get("act") or {}
        self.hand_feedback = HandFeedbackPoller(
            self.hand, poll_hz=act.get("hand_feedback_poll_hz", 4.),
            max_age_s=act.get("hand_feedback_max_age_s", 1.))
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
        arm_status = self._read_arm_status()
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
            self._arm_status = arm_status
            self._hand_feedback, self._hand_error = hand_feedback, hand_error
        self._check_arm_status(arm_status, phase="反馈监督 Motion supervision")

    @staticmethod
    def _check_arm_status(status, *, phase):
        """NO_SOLUTION 仅记录为状态，不阻止新命令；其他固件故障仍终止任务。"""
        if status["arm_status"] in {0, 2}:
            return
        # NO_SOLUTION 不需要清错或等待 NORMAL；到位仍由新鲜模式/运动反馈
        # 和实测位置判断。其余故障保留原有处理，不与主机 IK 无解混为一谈。
        detail = status["arm_status_name"]
        raise RuntimeError(
            f"Nero 固件报告异常 Firmware arm fault during {phase}: {detail}; "
            f"status={status}；请通过现场操作流程排查并恢复正常状态后重新启动 "
            "Operator recovery required; no automatic reset/enable"
        )

    def _read_arm_status(self):
        """读取 CAN 状态帧；到位判定必须使用新鲜反馈，不能沿用上一段的缓存。"""
        from robot_control.nero import NERO_MOTOR_FEEDBACK_MAX_AGE_S

        message = self.arm.get_raw_arm_status()
        # 与电机反馈相同，SDK 的接收时间戳是 Unix 秒。此时间要与命令
        # 下发的 time.time() 比较，不能与日志使用的 monotonic 混用。
        received_at = float(message.timestamp)
        if (not np.isfinite(received_at) or received_at <= 0
                or time.time() - received_at > NERO_MOTOR_FEEDBACK_MAX_AGE_S):
            raise RuntimeError(f"Nero 状态帧失效 Firmware status unavailable/stale: {received_at}")
        return {name: int(getattr(message.msg, name)) for name in
                ("ctrl_mode", "arm_status", "mode_feedback", "motion_status")} | {
                    "received_at_unix": received_at,
                    "arm_status_name": getattr(message.msg.arm_status, "name", "UNKNOWN")}

    def _feedback_tcp(self, q):
        from .geometry import transform

        with self._feedback_fk_lock:
            flange = self._feedback_ik.forward(q)
        return (transform(self.settings["task_from_base"]) @ flange
                @ transform(self.settings["flange_from_tcp"]))

    def _motion_result(self):
        if self.motion is None:
            return None
        result = {name: deepcopy(self.motion[name]) for name in
                  ("status", "reached", "reason", "motion_id", "joint_error_rad", "actual_joints", "diagnostics")}
        if "software_start_at" in self.motion:
            result["diagnostics"]["software_start_at"] = self.motion["software_start_at"]
        return result

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
                    "arm_status": deepcopy(self._arm_status),
                    "executor": (self.motion.get("executor", "joint_position_J")
                                 if self.motion and self.motion["status"] in {"running", "completed"}
                                 and (self.motion.get("executor") != "act_JS" or self.motion["status"] == "running")
                                 and not self.manual_stop.is_set() else "joint_position_J"),
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
            raise CommandRejected("动作权限已失效 Robot action admission revoked")

    def _require_hardware(self):
        if not self.connected:
            raise RuntimeError("设备未连接 Hardware not connected; use operator --execute gate")

    def _send(self, q):
        # 人工请求不进入普通请求队列。物理发送必须串行化，锁内再次检查，
        # 防止等锁的旧目标在人工目标之后才发送；已进入 SDK 的调用仍无法撤回。
        with self._arm_send_lock:
            if not self.manual_stop.is_set():
                started = time.monotonic()
                # move_J 与固定启动 PTP 使用整体回位上限；普通暂停和人工
                # 保持仍使用默认的单步限制，不把 PTP 上限全局套到保持请求。
                from .planning import PTP_MAX_JOINT_DELTA_RAD

                ptp = (self.motion is not None and self.motion["status"] == "running"
                       and self.motion.get("executor") != "act_JS"
                       and self.motion["plan"].mode in {"ptp", "startup_ptp"})
                self.arm.move_joints(q, speed_percent=self.motion_settings["joint_speed_percent"],
                                     max_joint_delta=PTP_MAX_JOINT_DELTA_RAD if ptp else None)
                sent_at = time.monotonic()
                self._last_send_wall_at = time.time()
                self._hold_mode = 1
                if self.on_control_sample is not None:
                    self.on_control_sample({
                        "sent_at": sent_at, "command_sent": True,
                        "executor": "joint_position_J",
                        "sent_at_unix": self._last_send_wall_at,
                        "send_started_at_unix": self._last_send_wall_at - (sent_at - started),
                        "arm_status_before_send": deepcopy(self._arm_status),
                        "waypoint_index": self.motion["waypoint_index"] if self.motion else None,
                        "send_interval_s": (None if self._last_send_at is None
                                            else sent_at - self._last_send_at),
                        "send_duration_s": sent_at - started,
                        "target_joints": np.asarray(q).tolist(),
                        "actual_joints": self._joints.tolist(),
                        "tcp": deepcopy(self._tcp), "sampled_at": self._sampled_at,
                        "motion_id": self.motion["motion_id"] if self.motion else None,
                    })
                self._last_send_at = sent_at
                self._hold_sent = True
                return True
        return False

    def _linear_event(self, motion, description, **values):
        """运动线程只入队；打印与落盘仍由普通 I/O 线程处理。"""
        if self.on_control_sample is not None:
            self.on_control_sample({"command_sent": False, "executor": "native_linear_L",
                                    "description": description, "motion_id": motion["motion_id"],
                                    "phase": motion["phase"], "attempt": motion["attempt"],
                                    "sampled_at": self._sampled_at,
                                    "arm_status": deepcopy(self._arm_status), **values})

    def _send_linear(self, motion):
        """只发送同一个法兰终点一次；启动确认与有界重发由后续 tick 决定。"""
        with self._arm_send_lock:
            # 人工 hold 可随时置位；即使本次重发已经在等锁，也不能覆盖人工目标。
            if self.manual_stop.is_set():
                return
            started, sent_wall = time.monotonic(), time.time()
            self.arm.move_linear(motion["flange_pose"],
                                 speed_percent=self.motion_settings["joint_speed_percent"])
            self._last_send_wall_at = sent_wall
            now = time.monotonic()
            motion.update(attempt=motion["attempt"] + 1, target_sent=True,
                          target_sent_at=now, target_sent_wall_at=sent_wall,
                          phase="confirm_start", phase_started_at=now)
            self._hold_feedback, self._stable_since = None, None
            self._linear_event(
                motion, "厂家 L 目标已调用 SDK Native move_l target submitted",
                command_sent=True, sent_at=now, sent_at_unix=sent_wall,
                send_started_at_unix=sent_wall, send_duration_s=now - started,
                waypoint_index=None, target_joints=None,
                target_tcp=motion["plan"].target_tcp.tolist(), flange_pose=motion["flange_pose"],
                actual_joints=self._joints.tolist(), tcp=deepcopy(self._tcp))

    def _fail_motion(self, motion, reason):
        """动作失败后固定当前关节；不锁存设备故障，允许下一 cell 重新规划。"""
        with self._lock:
            if self.manual_stop.is_set():
                return
            motion.update(status="failed", reached=False, reason=reason,
                          actual_joints=self._joints.tolist())
        if motion.get("executor") == "native_linear_L":
            self._linear_event(motion, reason)
        # 先保留 failed 及完整诊断，再复用普通暂停：只捕获一次反馈，发送一个
        # J 保持目标；不重新规划、不松手、不调用会导致下坠的阻尼 emergency_stop。
        self._pause(reason)

    def _tick_linear(self, motion):
        """将实测脚本的三个等待循环拆成状态机，每次只处理一个反馈周期。

        wait_idle 最多 10 s；confirm_start 每次最多 1 s、总共最多 3 次发送；
        wait_arrival 沿用 completion_timeout_s。这里不 sleep，也不把等待放进
        Pipe/MCP 请求中，因此暂停和独立人工 hold 不必排在整个运动后面。
        """
        from robot_control.nero_ik import pose_error

        now, status = time.monotonic(), self._arm_status
        position_error, angle_error = pose_error(np.asarray(self._tcp), motion["plan"].target_tcp)
        with self._lock:
            motion["actual_joints"] = self._joints.tolist()
            motion["diagnostics"] = {
                "executor": "native_linear_L", "phase": motion["phase"],
                "attempt": motion["attempt"], "max_attempts": self.motion_settings["linear_max_attempts"],
                "start_confirmed": motion["start_confirmed"],
                "position_error_m": position_error, "angle_error_rad": angle_error,
                "position_tolerance_m": self.motion_settings["linear_position_tolerance_m"],
                "angle_tolerance_rad": self.motion_settings["linear_angle_tolerance_rad"],
                "target_tcp": motion["plan"].target_tcp.tolist(), "actual_tcp": deepcopy(self._tcp),
                "flange_pose": motion["flange_pose"], "arm_status": deepcopy(status),
                "target_sent_at_unix": motion["target_sent_wall_at"],
                "phase_elapsed_s": now - motion["phase_started_at"]}
        if self.on_control_sample is not None:
            # 与 J 一样保存每个监督周期；不带 description，普通 I/O 线程只
            # 落盘而不逐行刷终端。启动/重发/完成等事件另由 _linear_event 输出。
            self.on_control_sample({"command_sent": False, "sent_at": None,
                                    "motion_id": motion["motion_id"], "waypoint_index": None,
                                    "sampled_at": self._sampled_at, "actual_joints": self._joints.tolist(),
                                    "tcp": deepcopy(self._tcp), "target_joints": None,
                                    **deepcopy(motion["diagnostics"])})

        if motion["phase"] == "wait_idle":
            if status["motion_status"] == 0:
                self._send_linear(motion)
            elif now - motion["phase_started_at"] >= self.motion_settings["linear_pre_idle_timeout_s"]:
                self._fail_motion(motion, "厂家 L 发送前等待空闲超时 Native L pre-idle timed out; "
                                  f"firmware={status}")
            return

        fresh_l = (status["received_at_unix"] > motion["target_sent_wall_at"]
                   and status["mode_feedback"] == 2)
        if motion["phase"] == "confirm_start":
            if fresh_l and status["motion_status"] == 1:
                motion.update(start_confirmed=True, phase="wait_arrival", phase_started_at=now)
                self._linear_event(motion, "厂家 L 已确认启动 Native L motion_status 0->1 confirmed")
            elif now - motion["target_sent_at"] >= self.motion_settings["linear_start_timeout_s"]:
                if motion["attempt"] < self.motion_settings["linear_max_attempts"]:
                    self._linear_event(motion, "厂家 L 启动未确认，重发同一目标 Native L start unconfirmed; retry same target")
                    self._send_linear(motion)
                    return
                # 脚本在最后一次未看到 0->1 时仍等待到位：极短的运动可能已在
                # 两次采样间完成。不能仅凭未捕获到启动沿就直接认定执行失败。
                motion.update(phase="wait_arrival", phase_started_at=now)
                self._linear_event(motion, "厂家 L 启动确认次数已用尽，继续检查到位 Native L attempts exhausted; await arrival")
            else:
                return

        stable = (self._hold_feedback is not None
                  and np.max(np.abs(self._joints - self._hold_feedback)) <= .002)
        self._hold_feedback = self._joints.copy()
        arrived = (fresh_l and status["motion_status"] == 0 and stable
                   and position_error <= self.motion_settings["linear_position_tolerance_m"]
                   and angle_error <= self.motion_settings["linear_angle_tolerance_rad"])
        self._stable_since = (self._stable_since or now) if arrived else None
        if self._stable_since is not None and now - self._stable_since >= .2:
            with self._lock:
                if self.manual_stop.is_set() or motion["status"] != "running":
                    return
                # 固件 L 已完成并自行保持。不额外发送 J，否则可能覆盖刚完成
                # 的 L 目标；只保存当前关节，供原有静止监督/下一 cell 收尾使用。
                self._hold, self._hold_sent, self._hold_confirmed = self._joints.copy(), True, True
                self._hold_mode = 2
                motion.update(status="completed", reached=True)
            self._linear_event(motion, "厂家 L 到位已确认 Native L arrival confirmed",
                               position_error_m=position_error, angle_error_rad=angle_error)
        elif now - motion["phase_started_at"] >= self.motion_settings["completion_timeout_s"]:
            self._fail_motion(motion, "厂家 L 到位超时 Native L arrival timed out; "
                              f"position_error_m={position_error:.6f}, angle_error_rad={angle_error:.6f}; "
                              f"attempts={motion['attempt']}, firmware={status}")

    def pause(self, reason):
        with self._trajectory_lock:
            return self._pause(reason)

    def _capture_hold_target(self):
        """只捕获一次真实关节；仅保持目标进行小幅边界处理，状态反馈保持原值。"""
        feedback = self.arm.get_joint_positions()
        target = self.arm.hold_target_from_feedback(feedback)
        if not np.array_equal(feedback, target) and self.on_control_sample is not None:
            self.on_control_sample({
                "description": "保持边界调整 Hold target moved inside official limits",
                "command_sent": False, "actual_joints": feedback.tolist(),
                "target_joints": target.tolist(), "sampled_at": time.monotonic()})
        return target

    def _pause(self, reason, *, pause_cell=True):
        if pause_cell:
            self.cell_paused = True
        with self._lock:
            if self.motion and self.motion["status"] == "running":
                self.motion.update(status="paused", reached=False, reason=reason,
                                   actual_joints=self._joints.tolist() if self._joints is not None else None)
        if self.connected and not self.manual_stop.is_set():
            # 读真实反馈作为新的固定目标；不继续原轨迹，也不松开 L20。
            q = self._capture_hold_target()
            self._hold = q.copy()
            self._hold_sent = False
            self._hold_confirmed = False
            self._stable_since = None
            self._hold_feedback = None
            try:
                self._send(q)
            except CommandRejected as exc:
                # 同样的目标拒绝在普通动作中可以返回；但保持目标也无法发送
                # 意味着旧运动未能可靠收尾，必须升级为控制故障，而不是继续抓取。
                raise RuntimeError(f"保持目标被拒绝 Hold command rejected: {exc}") from exc
        return self.snapshot()

    def manual_hold(self, reason):
        """人工停止先撤权，再捕获固定关节目标；保持循环不依赖普通请求线程。"""
        # 双侧共用撤权 Event；另一侧已置位不能跳过本侧首次捕获。
        first_request = self.manual_hold_report is None
        self.manual_stop.set()
        with self._lock:
            self.cell_paused = True
            if self.motion:
                self.motion.update(status="manual_hold", reached=False, reason=reason)
            self._hold_confirmed = False
        if not first_request:
            return self.snapshot()  # 幂等：保留第一次的目标及诊断，不重新捕获当前位置。
        report = {"requested_at": time.monotonic(), "reason": reason,
                  self.arm_name: "not_connected", "mode": "fixed_joint_position_J",
                  "stop_confirmed": False, "l20": "no_release_or_reset_command"}
        if self.connected:
            try:
                # 使用带新鲜度检查的七轴实测值，不使用旧轨迹目标或部分填零缓存。
                # 捕获失败时不在后续周期重抓目标，也不自动切换成阻尼急停。
                self._manual_target = self._capture_hold_target()
                report[self.arm_name] = "target_captured"
            except Exception as exc:
                report[self.arm_name] = "hold_failed"
                report["error"] = str(exc)
        with self._lock:
            self.manual_hold_report = report
        self.manual_hold_tick()
        return self.snapshot()

    def manual_hold_tick(self):
        """独立人工路径发送一次普通 J 保持目标，再持续监督关节反馈。"""
        if not self.manual_stop.is_set() or self._manual_target is None:
            return
        try:
            # 普通 I/O 故障或 worker/Codex 退出也不停止此路径。位置发送仍校验
            # 使能/限位/反馈及 CAN 错误；不会自动使能、清错或放开电机。
            if not self._manual_sent:
                with self._arm_send_lock:
                    self.arm.move_joints(self._manual_target,
                                         speed_percent=self.motion_settings["joint_speed_percent"])
                self._manual_sent = True
                self.manual_hold_report["last_send_at"] = time.monotonic()
            q = self.arm.get_joint_positions()
            tcp = self._feedback_tcp(q)
            error = float(np.max(np.abs(q - self._manual_target)))
            stable = (self._manual_feedback is not None
                      and np.max(np.abs(q - self._manual_feedback)) <= .005)
            self._manual_feedback = q.copy()
            now = time.monotonic()
            if error <= self.motion_settings["joint_arrival_tolerance_rad"] and stable:
                self._manual_stable_since = self._manual_stable_since or now
                self._manual_confirmed = now - self._manual_stable_since >= .2
            else:
                self._manual_stable_since, self._manual_confirmed = None, False
            with self._lock:
                self._joints, self._tcp, self._sampled_at = q.copy(), tcp.tolist(), now
                self.manual_hold_report.update({self.arm_name: "hold_target_sent", "error": None,
                                                "joint_error_rad": error, "last_feedback_at": now})
        except Exception as exc:
            # 反馈失效/使能丢失/CAN 故障时不能声称悬停。保留固定目标供下周期
            # 重试未成功的发送；已成功发送后不因反馈偏差反复覆盖固件目标。
            # 绝不退化调用 electronic_emergency_stop，也不追随下落重抓目标。
            self._manual_stable_since, self._manual_confirmed = None, False
            with self._lock:
                self.manual_hold_report.update({self.arm_name: "hold_failed", "error": str(exc)})

    def tick(self, delta):
        """独立轨迹线程推进一次；RPC、图像落盘和手部查询不能改变下发周期。"""
        with self._trajectory_lock:
            try:
                self._tick(delta)
            except CommandRejected as exc:
                # 只有 SDK 发送前明确拒绝的动作目标可恢复；不能把反馈失效、
                # CAN 错误或未知 ValueError 一并吞掉。保持失败也不能退回普通动作。
                if not self.motion or self.motion["status"] != "running":
                    raise RuntimeError(f"保持指令失败 Hold command failed: {exc}") from exc
                self.motion["diagnostics"]["command_rejected"] = True
                self._fail_motion(self.motion, f"动作目标被拒绝 Motion command rejected: {exc}")

    def _tick(self, delta):
        if not self.connected or self.manual_stop.is_set():
            return
        self._measure()
        motion = self.motion
        native_running = (motion and motion["status"] == "running" and not self.fault
                          and motion.get("executor") == "native_linear_L")
        if self.on_control_sample is not None and not native_running:
            # J/L 目标之间不会连续发命令，仍保存每个监督周期的真实反馈，便于
            # 区分固件运动过程、目标下发时刻及停稳后的误差。
            self.on_control_sample({"command_sent": False, "sent_at": None,
                                    "sampled_at": self._sampled_at,
                                    "arm_status": deepcopy(self._arm_status),
                                    "motion_id": self.motion["motion_id"] if self.motion else None,
                                    "waypoint_index": self.motion["waypoint_index"] if self.motion else None,
                                    "executor": self.motion.get("executor", "joint_position_J") if self.motion else "joint_position_J",
                                    "phase": self.motion.get("phase") if self.motion else None,
                                    "attempt": self.motion.get("attempt") if self.motion else None,
                                    "actual_joints": self._joints.tolist(),
                                    "tcp": deepcopy(self._tcp),
                                    "target_joints": (self._hold.tolist()
                                                      if self._hold is not None else None)})
        if motion and motion["status"] == "running" and not self.fault:
            if motion.get("executor") == "act_JS":
                # ACT I/O 线程发送连续 JS；本线程只监督反馈，不能重发旧 J/L 目标。
                with self._lock:
                    motion["actual_joints"] = self._joints.tolist()
                return
            if native_running:
                self._tick_linear(motion)
                return  # 厂家 L 期间绝不能执行旧的 J 分段或重发旧保持目标。
            if not motion["target_sent"]:
                self._hold = motion["plan"].joint_waypoints[motion["waypoint_index"]].copy()
                self._hold_sent = False
                self._hold_confirmed, self._stable_since, self._hold_feedback = False, None, None
                if not self._send(self._hold):
                    return  # 人工线程已锁定；不再推进普通目标。
                motion["target_sent"], motion["target_sent_at"] = True, time.monotonic()
                motion["target_sent_wall_at"] = self._last_send_wall_at
            error = float(np.max(np.abs(self._joints - self._hold)))
            # 新鲜状态帧只说明刚收到反馈，不能确认本次命令已启动。fold-14
            # 中发送后约 39 ms 仍反馈 J/0，约 55 ms 才变为运动中 J/1；
            # 因而 J/0 且误差大时应继续等待，不能立即判“到位但偏差过大”。
            # 完成必须同时满足发送后的新鲜 J/0 和实测关节误差达标。
            # 不强制采样到 1：很短的动作可能在相邻监督周期之间完成。
            firmware_idle = (self._arm_status["received_at_unix"] > motion["target_sent_wall_at"]
                             and self._arm_status["mode_feedback"] == 1
                             and self._arm_status["motion_status"] == 0)
            arrival_tol = self.motion_settings["joint_arrival_tolerance_rad"]
            warn_tol = self.motion_settings["joint_tolerance_rad"]
            final_waypoint = motion["waypoint_index"] == len(motion["plan"].joint_waypoints) - 1
            if firmware_idle and error <= arrival_tol:
                if error > warn_tol and self.on_control_sample is not None:
                    self.on_control_sample({
                        "command_sent": False, "executor": "joint_position_J",
                        "description": "到位警告 Arrival warning: error > tolerance",
                        "waypoint_index": motion["waypoint_index"],
                        "joint_error_rad": error, "warn_tolerance_rad": warn_tol})
                if final_waypoint:
                    with self._lock:
                        if not self.manual_stop.is_set() and motion["status"] == "running":
                            self._hold_confirmed = True
                            motion.update(status="completed", reached=True, joint_error_rad=error,
                                          actual_joints=self._joints.tolist())
                else:
                    motion["waypoint_index"] += 1
                    motion["target_sent"] = False
            elif time.monotonic() - motion["target_sent_at"] > self.motion_settings["completion_timeout_s"]:
                self.cell_paused = True
                index = motion["waypoint_index"]
                with self._lock:
                    if self.manual_stop.is_set():
                        return
                    motion.update(status="failed", reason=(
                        "关节运动到位超时 Joint arrival timed out; "
                        f"waypoint={index}/{len(motion['plan'].joint_waypoints) - 1}, "
                        f"joint_error_rad={error:.6f}, arrival_tolerance_rad={arrival_tol}, "
                        f"firmware_motion_status={self._arm_status['motion_status']}, "
                        f"firmware_mode={self._arm_status['mode_feedback']}"),
                        joint_error_rad=error, actual_joints=self._joints.tolist(),
                        diagnostics={
                            "waypoint_index": index,
                            "waypoint_count": len(motion["plan"].joint_waypoints) - 1,
                            "target_joints": self._hold.tolist(),
                            "actual_joints": self._joints.tolist(),
                            "joint_errors_rad": (self._joints - self._hold).tolist(),
                            "arrival_tolerance_rad": arrival_tol,
                            "arm_status": deepcopy(self._arm_status),
                            "target_sent_at_unix": motion["target_sent_wall_at"],
                            "firmware_idle": firmware_idle,
                            "joint_reached": error <= arrival_tol})
                # 保留失败瞬间的原目标与误差，再请求一次当前关节的固定保持。
                # 仅标 failed 并保留旧运动目标，会让已报失败的机械臂继续运动，
                # 还会使 cell_end 跳过 running 分支、过早耗尽 3 s 收尾时间。
                # _pause 不覆盖 failed 结果；人工 hold 置位后也不会再发普通目标。
                self._pause(motion["reason"])
        elif self._hold is not None:
            # 普通 J 成功接收目标后由固件保持；仅未成功发送的固定目标会再次尝试。
            if not self._hold_sent:
                self._send(self._hold)
            error = float(np.max(np.abs(self._joints - self._hold)))
            self._confirm_hold(error)

    def _confirm_hold(self, error):
        """确认普通保持到位（用于 quiescent 判定）。使用放宽的到位容差。"""
        arrival_tol = self.motion_settings["joint_arrival_tolerance_rad"]
        stable = (self._hold_feedback is not None
                  and np.max(np.abs(self._joints - self._hold_feedback)) <= .005)
        self._hold_feedback = self._joints.copy()
        # 关节暂时不动不能证明旧 L 已被 J 保持接管。暂停后必须看到发送后的
        # 新鲜 J/idle；已成功完成的 L 则继续监督其 L/idle，不额外切换控制模式。
        sent_at = self._last_send_wall_at
        idle = (self._hold_sent and sent_at is not None
                and self._arm_status["received_at_unix"] > sent_at
                and self._arm_status["mode_feedback"] == self._hold_mode
                and self._arm_status["motion_status"] == 0)
        with self._lock:
            if self.manual_stop.is_set():
                return
            if idle and error <= arrival_tol and stable:
                self._stable_since = self._stable_since or time.monotonic()
                self._hold_confirmed = time.monotonic() - self._stable_since >= .2
            else:
                self._stable_since, self._hold_confirmed = None, False

    def fail(self, reason):
        with self._trajectory_lock:
            try:
                if (self.connected and self.motion and self.motion["status"] == "running"
                        and self.motion.get("executor") in {"native_linear_L", "act_JS"}):
                    # 设备/通信故障也必须撤销正在执行的 L/ACT；普通保持仍需通过
                    # SDK 的使能与反馈检查，失败时保留原因，人工通道仍独立可用。
                    self._pause(reason)
                # 已有发送目标时保留该固定目标，避免每次故障重抓实测值。
                # 尚未发运动目标才捕获一次；后续周期继续尝试原位保持。
                self.cell_paused = True
                if self.connected and self._hold is None:
                    self._hold = self._capture_hold_target()
                    self._hold_sent = False
            except Exception as exc:
                reason += f"；保持请求失败 Hold request failed: {exc}"
            with self._lock:
                self.fault = reason
                self._hold_confirmed = False
                if self.motion:
                    self.motion.update(status="failed", reached=False, reason=reason)
        self.failed.set()

    def _start_motion(self, cell_id, payload, *, prepare_only=False):
        from .planning import MotionPlan, PTP_MAX_JOINT_DELTA_RAD

        self._require_hardware()
        if payload.get("mode") == "native_linear":
            return self._start_linear_motion(cell_id, payload, prepare_only=prepare_only)
        try:
            plan = MotionPlan.from_dict(payload)
        except ValueError as exc:
            raise CommandRejected(f"运动数据无效 Invalid motion data: {exc}") from exc
        if plan.arm != self.arm_name:
            raise CommandRejected("运动计划与机械臂不匹配 Motion plan arm mismatch")
        if plan.mode not in {"joint", "cartesian", "ptp", "startup_ptp"} or plan.joint_waypoints is None:
            raise CommandRejected("普通 J 执行需要几何分段点 Joint executor requires geometric waypoints")
        startup_ptp = plan.mode == "startup_ptp"
        ptp = plan.mode in {"ptp", "startup_ptp"}
        if ptp and len(plan.joint_waypoints) != 2:
            raise CommandRejected("PTP 仅接受起点和单个终点 PTP requires two endpoints")
        if startup_ptp and (cell_id != 0 or not self.config.get("startup", {}).get("enabled", False)):
            raise CommandRejected("启动 PTP 只允许固定准备 cell Startup PTP requires cell 000")
        # 发送的是真正的几何分段点，不是计时样本。这里仅复核
        # 每段关节插值的限位/工作区和最大跨度；不把固件内部路径当作已知曲线。
        if ptp:
            # 实测起点可能有 1 mrad 边界差异，但最终下发的单个 PTP 目标
            # 必须严格位于官方区间；不能将反馈容差用于放宽 agent 的目标。
            self.validator._check_path(self.arm_name, plan.joint_waypoints[1:])
        self.validator._check_path(self.arm_name, plan.joint_waypoints, feedback_path=ptp)
        starts, ends = plan.joint_waypoints[:-1], plan.joint_waypoints[1:]
        delta_limit = (PTP_MAX_JOINT_DELTA_RAD if ptp
                       else self.motion_settings["max_ik_step_rad"])
        if np.max(np.abs(ends - starts)) > delta_limit:
            raise CommandRejected("J 分段跨度过大 Joint waypoint step exceeded")
        # PTP 整体跨度可能大于任务分段；按 joint_step_rad 生成只读检查点，
        # 保留路径工作区检查的密度。执行数据仍只有起点/终点，不发送检查点。
        count = (max(10, int(np.ceil(np.max(np.abs(ends - starts))
                                    / self.motion_settings["joint_step_rad"])) + 1)
                 if ptp else 10)
        segments = np.linspace(starts, ends, count, axis=1).reshape(-1, 7)
        self.validator._check_path(self.arm_name, segments, feedback_path=ptp)
        # 普通 J 不执行 TOPPRA 的计时样本和样条，设备进程只复核实际下发的
        # 几何目标及分段。密集样条仍由 worker 在规划时检查，不能在这里用
        # 无关的重复计算拖慢 CAN 接收，也不能将样条限速误称为固件限速。
        motion_id = uuid.uuid4().hex
        plans = self.directory / "motions"
        plans.mkdir(exist_ok=True)
        (plans / f"{motion_id}.json").write_text(json.dumps(plan.as_dict()), encoding="utf-8")
        motion = {"plan": plan, "waypoint_index": 1, "target_sent": False,
                  "target_sent_at": None, "target_sent_wall_at": None,
                  "status": "running", "reached": False, "diagnostics": {},
                  "motion_id": motion_id, "reason": "", "joint_error_rad": None,
                  "actual_joints": None}
        if prepare_only:
            return motion
        with self._trajectory_lock:
            return self._install_motion(cell_id, motion, self._check_motion_start(cell_id, motion))

    def _start_linear_motion(self, cell_id, payload, *, prepare_only=False):
        """发布单目标厂家 L；只接收数据，不在控制进程执行 worker 源码。"""
        from scipy.spatial.transform import Rotation
        from .geometry import tcp_to_flange
        from .planning import LinearMotionPlan

        try:
            plan = LinearMotionPlan.from_dict(payload)
        except ValueError as exc:
            raise CommandRejected(f"L 目标数据无效 Invalid native L data: {exc}") from exc
        if plan.arm != self.arm_name:
            raise CommandRejected("运动计划与机械臂不匹配 Motion plan arm mismatch")
        # 控制进程重新检查 seed 限位、起点/终点工作区；不信任 worker 内的
        # 持久变量，也不恢复已删除的逐点 IK/TOPPRA 主机执行路径。
        plan = self.validator.plan_linear(plan.start_joints, plan.target_tcp, arm=self.arm_name)
        flange = tcp_to_flange(plan.target_tcp, self.settings["task_from_base"],
                               self.settings["flange_from_tcp"])
        # 完全沿用实测脚本：task 虚拟 TCP -> base 法兰 -> xyz + 外旋 xyz RPY。
        # 此处单位始终为 m/rad；不直接把虚拟 TCP 坐标传给 SDK。
        pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler("xyz")]
        motion_id = uuid.uuid4().hex
        plans = self.directory / "motions"
        plans.mkdir(exist_ok=True)
        (plans / f"{motion_id}.json").write_text(json.dumps(plan.as_dict()), encoding="utf-8")
        motion = {"plan": plan, "executor": "native_linear_L", "waypoint_index": None,
                  "flange_pose": pose.tolist(), "phase": "wait_idle",
                  "phase_started_at": time.monotonic(), "attempt": 0, "start_confirmed": False,
                  "target_sent": False, "target_sent_at": None, "target_sent_wall_at": None,
                  "status": "running", "reached": False, "diagnostics": {},
                  "motion_id": motion_id, "reason": "", "joint_error_rad": None,
                  "actual_joints": None}
        if prepare_only:
            return motion
        with self._trajectory_lock:
            return self._install_motion(cell_id, motion, self._check_motion_start(cell_id, motion))

    def _check_motion_start(self, cell_id, motion):
        self._admit(cell_id)
        if self.motion and self.motion["status"] == "running":
            raise CommandRejected(f"{self.arm_name}: 已有运动正在执行 A motion is already running")
        q = self.arm.get_joint_positions()
        plan = motion["plan"]
        seed = plan.start_joints if plan.mode == "native_linear" else plan.positions[0]
        error = float(np.max(np.abs(q - seed)))
        if error > self.motion_settings["joint_arrival_tolerance_rad"]:
            raise CommandRejected(
                f"{self.arm_name}: 运动起点已变化 Recompute motion from current feedback; "
                f"max_error_rad={error}, tolerance_rad={self.motion_settings['joint_arrival_tolerance_rad']}, "
                f"planned_start={seed.tolist()}, actual_joints={q.tolist()}")
        return q

    def _install_motion(self, cell_id, motion, q):
        # 调用方持有轨迹锁；双臂先复核所有起点，再在同一调度边界发布。
        with self._lock:
            self._admit(cell_id)
            self._hold_confirmed, self._stable_since, self._hold_feedback = False, None, None
            if motion.get("executor") == "native_linear_L":
                self._hold, self._hold_sent = None, False
                motion["phase_started_at"] = time.monotonic()
            motion["actual_joints"] = q.tolist()
            self.motion = motion
            return self._motion_result()

    def _observe(self):
        # 只有观测链路的已知采集/保存异常回到 agent。不要把 KeyError 等未知
        # 程序错误，或机械臂读取/发送错误笼统包装成“可重试的相机问题”。
        if self.camera is None:
            raise CommandRejected("相机未连接 Camera not connected")
        import cv2

        try:
            return self._capture_observation()
        except (RuntimeError, OSError, cv2.error) as exc:
            raise OperationError(f"观测失败 Observation failed: {exc}") from exc

    def _capture_observation(self):
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
        if owner:
            with self._trajectory_lock:
                return self._dispatch_owner(request)
        return self._dispatch_worker(request)

    def _dispatch_owner(self, request):
        operation = request["operation"]
        interrupted = False
        if operation == "begin_cell":
            if self.ended or self.manual_stop.is_set() or self.fault or not self.snapshot()["quiescent"]:
                nquencent_reason = f"Motion: {('None' if self.motion is None else self.motion['status'])}, " + f"Hold: {('None' if self._hold is None else self._hold_confirmed)}"
                raise CommandRejected("控制状态不允许新 cell Robot is not ready for a new cell: " +
                                 f"Ended: {self.ended}, Fault: {self.fault}, !Quiescent: ({('Y' if self.snapshot()['quiescent'] else nquencent_reason)})")
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
            raise CommandRejected("未知协调请求 Unknown coordinator request")
        return {**self.snapshot(), "cell_interrupted": interrupted}

    def _dispatch_worker(self, request):
        operation = request["operation"]
        cell_id = request.get("cell_id")
        if cell_id != self.active_cell or self.ended or self.manual_stop.is_set():
            raise CommandRejected("cell 已失效 Cell admission revoked")
        if operation == "state":
            # 反馈由轨迹线程固定频率刷新；状态请求不再额外推进/阻塞控制周期。
            return self.snapshot()
        if operation == "observe":
            return self._observe()
        if operation == "motion_status":
            if not self.motion or self.motion["motion_id"] != request["motion_id"]:
                raise CommandRejected("未知运动句柄 Unknown motion handle")
            return self.snapshot()["motion"]
        if operation == "pause":
            return self.pause(request["reason"])
        self._admit(cell_id)
        if operation == "startup_open_hand":
            self._require_hardware()
            if cell_id != 0 or not self.config.get("startup", {}).get("enabled", False):
                raise CommandRejected("全手张开仅允许固定启动流程 Full-hand open requires startup cell 000")
            # 复用 wrist-only 的同一个 SDK 包装方法，一次发送官方全手张开
            # 预设，包含其余三指。这里不是 thumb/index gripper 操作，也不是
            # reset；与 teleop 一样不把 SDK 返回宣称为五指已确认到位。
            self.hand.open_hand()
            self._hand_target, self._hand_read_at = None, None
            return {"command_status": "sdk_returned", "preset": "official_full_hand_open",
                    "commanded_fingers": ["thumb", "index", "middle", "ring", "little"],
                    "delivery_confirmed": False, "hand": self.snapshot()["hand"]}
        if operation == "start_motion":
            return self._start_motion(cell_id, request["plan"])
        if operation == "set_gripper":
            self._require_hardware()
            preset = request["preset"]
            if preset not in {"open", "closed", "replay_start"}:
                raise CommandRejected("未知手型 Unknown gripper preset")
            # replay_start 可由普通配置保存示范首帧，或由调试脚本从指定 CSV
            # 第一帧生成；均经同一权限检查与 thumb/index 发送路径，不接受
            # cell 自定义关节数组，也不在这里启动整段回放。
            raw = self.settings["hand_poses"][preset]
            self._admit(cell_id)
            # 在普通请求线程阻塞等待，机械臂监督线程和独立人工 hold 入口
            # 继续运行。每次重发/读反馈前复核权限，人工 hold 后不继续重试。
            self._hand_target = list(raw)
            self._hand_read_at = None
            result = self.hand.set_thumb_index_positions_raw_and_wait(
                raw, **self.config["gripper"], check_cancelled=lambda: self._admit(cell_id))
            return {"command_status": "feedback_reached", "preset": preset, **result,
                    "commanded_fingers": ["thumb", "index"],
                    "delivery_confirmed": False, "hand": self.snapshot()["hand"]}
        raise CommandRejected("未知 robo 操作 Unknown robo operation")

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
        # 设备断开后再关闭抓包，保留收尾阶段的帧与最终接口统计。
        if self.can_trace is not None:
            self.can_trace.close()
        if errors:
            raise RuntimeError("设备清理失败 Device cleanup failed: " + "; ".join(errors))
