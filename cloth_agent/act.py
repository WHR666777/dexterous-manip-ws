"""ACT 模型管道与固定步数执行；模型依赖留在独立 Python 环境。

后台线程等待推理，设备仍由 cloth 控制进程持有；普通反馈监督和人工 hold
不等待模型。每次运行重置策略，推理帧直接取相机缓存，不逐帧保存 RGB-D。
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import selectors
import subprocess
from threading import Event, Thread
import time
import uuid

import numpy as np

from robot_control.errors import CommandRejected, GripperTimeout, OperationError


SIDES = ("left", "right")
ACTIVE = list(range(11)) + list(range(15, 20))
STATE_BYTES = 46 * 4


# act_bimanual_default_run01 训练集 episode 36 / frame 0，左右取自同一示范。
# 该帧双臂关节与 start_joint_positions 一致；16 项对应 ACTIVE 槽位，单位 raw。
TRAINING_START_HANDS = {
    "left": [3, 122, 158, 150, 130, 244, 253, 171, 126, 12, 253, 176, 224, 208, 214, 231],
    "right": [58, 112, 146, 139, 100, 226, 254, 160, 95, 1, 254, 173, 230, 222, 232, 246],
}


def prepare_act_hands(robot, cell_id):
    """设置固定 ACT 五指起始手型；渐进发送、检查反馈，不移动机械臂或加载模型。"""
    config = (robot.config or {}).get("act")
    if not config or set(robot.arms) != set(SIDES) or not robot.connected:
        raise CommandRejected("ACT 手型准备需要 ACT 配置与已连接的双臂双手")

    def check_cancelled():
        robot._admit(cell_id)
        for arm in robot.arms.values():
            arm._admit(cell_id)

    def feedback(side):
        check_cancelled()
        try:
            return robot.arms[side].hand_feedback.get_latest()
        except RuntimeError as exc:
            raise OperationError(f"{side}: ACT 起始手型反馈不可用: {exc}") from exc

    with robot._trajectory_lock:
        check_cancelled()
        if not robot.snapshot()["quiescent"]:
            raise CommandRejected("请先等待双臂运动结束并确认保持，再准备 ACT 手型")
    previous = {side: feedback(side).positions_raw.copy() for side in SIDES}
    limit = config["hand_max_raw_step"]
    while True:
        reached = True
        for side in SIDES:
            arm = robot.arms[side]
            target = feedback(side).positions_raw.copy()
            delta = np.asarray(TRAINING_START_HANDS[side]) - previous[side][ACTIVE]
            target[ACTIVE] = previous[side][ACTIVE] + np.clip(delta, -limit, limit)
            with arm._arm_send_lock:
                check_cancelled()
                arm.hand.set_joint_positions_raw(target)
            with arm._lock:
                arm._hand_target = target.tolist()
            if robot.on_control_sample is not None:
                robot.on_control_sample({"description": "ACT 起始手型目标已调用 SDK",
                                         "executor": "act_start_hand", "arm": side,
                                         "command_sent": True, "sent_at": time.monotonic(),
                                         "target_hand_raw": target.tolist()})
            previous[side] = target
            reached = reached and np.max(np.abs(delta)) <= limit
        time.sleep(robot.period)
        if reached:
            break
    sent_at = time.monotonic()
    deadline = sent_at + 5.
    while True:
        reports = {}
        for side in SIDES:
            sample = feedback(side)
            actual = sample.positions_raw[ACTIVE]
            error = int(np.max(np.abs(actual - TRAINING_START_HANDS[side])))
            reports[side] = {"target_raw16": list(TRAINING_START_HANDS[side]),
                             "actual_raw16": actual.tolist(), "max_error_raw": error,
                             "fresh_query": sample.query_started_at > sent_at,
                             "delivery_confirmed": False}
        if all(r["fresh_query"] and r["max_error_raw"] <= 10 for r in reports.values()):
            return reports
        if time.monotonic() >= deadline:
            raise GripperTimeout(f"ACT 起始手型未到位 ACT start hand timeout: {reports}")
        time.sleep(.05)


class ActCancelled(OperationError):
    """当前 ACT 运行已被暂停、cell 收尾或人工 hold 撤权。"""


class ActPolicyClient:
    """按配置启动模型解释器；仅传输数值，不导入 torch/lerobot 或设备 SDK。"""

    def __init__(self, config, directory):
        self.config = config
        self.log_path = Path(directory) / "act-worker.log"
        self.process = None
        self.policy = None
        self._log = None

    def _transfer(self, *, data=None, size=0, timeout, check_cancelled):
        deadline = time.monotonic() + timeout
        stream = self.process.stdin if data is not None else self.process.stdout
        payload = memoryview(data) if data is not None else bytearray()
        offset = 0
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_WRITE if data is not None else selectors.EVENT_READ)
            while offset < (len(payload) if data is not None else size):
                check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OperationError(f"ACT 管道等待超时 ACT pipe timed out; log={self.log_path}")
                if not selector.select(min(.05, remaining)):
                    continue
                try:
                    if data is not None:
                        count = os.write(stream.fileno(), payload[offset:offset + 65536])
                    else:
                        chunk = os.read(stream.fileno(), size - offset)
                        count = len(chunk)
                        payload.extend(chunk)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise OperationError(f"ACT 管道通信失败 ACT pipe failed; log={self.log_path}: {exc}") from exc
                if count == 0:
                    raise OperationError(f"ACT 模型进程已退出 ACT worker closed its pipe; log={self.log_path}")
                offset += count
        return bytes(payload) if data is None else None

    def ready(self, policy, check_cancelled):
        if self.policy == policy and self.process is not None and self.process.poll() is None:
            return
        self.close()
        check_cancelled()
        settings = self.config
        command = [settings["python"], settings["script"], "--rpc-worker",
                   "--checkpoint", settings["policies"][policy], "--device", settings["device"]]
        env = os.environ.copy()
        libffi = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
        if Path(libffi).exists():
            env["LD_PRELOAD"] = libffi + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        try:
            self._log = self.log_path.open("ab", buffering=0)
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._log, env=env, bufsize=0, start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
        except OSError as exc:
            self.close()
            raise OperationError(f"无法启动 ACT 模型进程 ACT startup failed: {exc}") from exc
        if self._transfer(size=6, timeout=settings["startup_timeout_s"],
                          check_cancelled=check_cancelled) != b"READY\n":
            raise OperationError(f"ACT 就绪协议不匹配 Invalid ACT handshake; log={self.log_path}")
        self.policy = policy

    def reset(self, check_cancelled):
        options = {"timeout": self.config["inference_timeout_s"], "check_cancelled": check_cancelled}
        self._transfer(data=b"R", **options)
        if self._transfer(size=3, **options) != b"OK\n":
            raise OperationError("ACT 策略重置未确认 ACT reset not acknowledged")

    def predict(self, state, rgb, check_cancelled):
        options = {"timeout": self.config["inference_timeout_s"], "check_cancelled": check_cancelled}
        self._transfer(data=b"P" + state.astype("<f4").tobytes() + rgb.tobytes(), **options)
        action = np.frombuffer(self._transfer(size=STATE_BYTES, **options), dtype="<f4").copy()
        if not np.isfinite(action).all():
            raise OperationError("ACT 输出包含非有限数值 ACT returned a non-finite action")
        return action

    def close(self):
        process, self.process = self.process, None
        self.policy = None
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=.5)
                process.stdin.close()
                process.stdout.close()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


class ActExecution:
    """双臂控制器内部的 ACT 执行模式；一次调用独占两侧动作，完成后确认 J 保持。"""

    def __init__(self, controller):
        self.controller = controller
        self.config = (controller.config or {}).get("act")
        self.client = ActPolicyClient(self.config, controller.directory) if self.config else None
        self.run = None
        self._cancel = Event()
        self._cancel_reason = ""
        self._thread = None

    @property
    def active(self):
        return self.run is not None and self.run["status"] == "running"

    def snapshot(self):
        return deepcopy(self.run)

    def status(self, run_id):
        if self.run is None or self.run["run_id"] != run_id:
            raise CommandRejected("未知 ACT 运行 Unknown ACT run")
        return self.snapshot()

    def cancel(self, reason):
        if self.active and not self._cancel.is_set():
            self._cancel_reason = str(reason)
            self._cancel.set()

    def _check(self, run):
        if self._cancel.is_set() or self.controller.manual_stop.is_set():
            raise ActCancelled(self._cancel_reason or "ACT 动作权限已撤销 ACT cancelled")
        self.controller._admit(run["cell_id"])
        for arm in self.controller.arms.values():
            arm._admit(run["cell_id"])

    def start(self, cell_id, policy, steps):
        robot = self.controller
        if not self.config:
            raise CommandRejected("尚未配置 ACT 请填写 act 配置 ACT is not configured")
        if not isinstance(policy, str) or policy not in self.config["policies"]:
            raise CommandRejected("未知 ACT policy；请选择 act.policies 中的名称 Unknown ACT policy")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise CommandRejected("ACT steps 必须为正整数 ACT steps must be a positive integer")
        if set(robot.arms) != set(SIDES) or not robot.connected or robot.camera is None:
            raise CommandRejected("ACT 需要已连接的双臂、双手和相机 ACT requires bimanual hardware and camera")
        with robot._trajectory_lock:
            robot._admit(cell_id)
            if self.active or not robot.snapshot()["quiescent"]:
                raise CommandRejected("ACT 启动前需要两侧运动结束并保持 Both arms must be quiescent")
            for arm in robot.arms.values():
                arm._admit(cell_id)
                if arm._read_arm_status()["motion_status"] != 0:
                    raise CommandRejected("ACT 启动前需要两侧固件空闲 Both arms must be idle")
            # 同时发布两侧 ACT 状态；人工路径先置位 Event，再取得这些短状态锁。
            with ExitStack() as stack:
                for arm in robot.arms.values():
                    stack.enter_context(arm._lock)
                for arm in robot.arms.values():
                    arm._admit(cell_id)
                self._cancel = Event()
                self._cancel_reason = ""
                run = {"run_id": uuid.uuid4().hex, "cell_id": cell_id, "policy": policy,
                       "status": "running", "phase": "loading", "steps_requested": steps,
                       "steps_executed": 0, "reason": "", "elapsed_s": 0., "hold_confirmed": False}
                self.run = run
                for arm in robot.arms.values():
                    arm.motion = {"executor": "act_JS", "waypoint_index": 0, "status": "running",
                                  "reached": False, "reason": "", "motion_id": run["run_id"],
                                  "joint_error_rad": None, "actual_joints": None,
                                  "diagnostics": {"policy": policy, "steps_requested": steps}}
                    arm._hold, arm._hold_sent, arm._hold_confirmed = None, False, False
            self._thread = Thread(target=self._execute, args=(run,), name="cloth-act-inference", daemon=True)
            self._thread.start()
            return self.snapshot()

    def _observation(self, run):
        self._check(run)
        robot = self.controller
        try:
            frame = robot.camera.get_latest()
        except RuntimeError as exc:
            raise OperationError(f"ACT 相机读取失败 ACT camera unavailable: {exc}") from exc
        if frame is None:
            raise OperationError("ACT 相机帧缺失或过期 ACT camera frame unavailable/stale")
        if frame.color.shape != (480, 640, 3) or frame.color.dtype != np.uint8:
            raise OperationError("ACT 需要 640x480 RGB8 ACT requires 640x480 RGB8")
        parts, hands = [], {}
        now = time.monotonic()
        for side in SIDES:
            state = robot.arms[side].snapshot()
            if state["sampled_at"] is None or now - state["sampled_at"] > robot.config["camera"]["max_age_s"]:
                raise OperationError(f"{side}: ACT 机械臂状态过期 ACT arm observation stale")
            hand = state["hand"]
            if (hand["position_raw"] is None or hand["read_at"] is None
                    or now - hand["read_at"] > self.config["hand_feedback_max_age_s"]):
                raise OperationError(f"{side}: ACT 手部反馈不可用 ACT hand observation stale: {hand['error']}")
            hands[side] = np.asarray(hand["position_raw"], dtype=int)
            parts.extend((state["joints"], hands[side][ACTIVE]))
        values = np.concatenate(parts).astype("<f4")
        if values.shape != (46,) or not np.isfinite(values).all():
            raise OperationError("ACT 观测状态无效 Invalid ACT state")
        return frame, values, hands

    def _send(self, run, frame, action, hands, previous):
        robot = self.controller
        targets, hand_targets = {}, {}
        with robot._trajectory_lock:
            self._check(run)
            # 两侧都先验证；保持原脚本的关节限位、单步限制和手部 raw 限幅。
            for side, offset in (("left", 0), ("right", 23)):
                arm = robot.arms[side]
                try:
                    targets[side] = arm.arm.validate_joint_command(action[offset:offset + 7])
                except ValueError as exc:
                    raise CommandRejected(f"{side}: ACT 关节目标被拒绝 ACT target rejected: {exc}") from exc
                with arm.validator._lock:
                    arm.validator._check_path(side, targets[side][None])
                target = hands[side].copy()
                raw = np.rint(np.clip(action[offset + 7:offset + 23], 0, 255)).astype(int)
                baseline = (previous or hands)[side][ACTIVE]
                limit = self.config["hand_max_raw_step"]
                target[ACTIVE] = baseline + np.clip(raw - baseline, -limit, limit)
                hand_targets[side] = target
            if time.monotonic() - frame.received_at > robot.config["camera"]["max_age_s"]:
                raise OperationError("ACT 推理期间观测过期，没有下发本轮目标 ACT observation became stale")
            for side in SIDES:
                arm = robot.arms[side]
                with arm._arm_send_lock:
                    self._check(run)
                    try:
                        arm.arm.move_js(targets[side])
                    except ValueError as exc:
                        raise CommandRejected(f"{side}: ACT JS 目标被拒绝 ACT JS rejected: {exc}") from exc
                    self._check(run)
                    arm.hand.set_joint_positions_raw(hand_targets[side])
                with arm._lock:
                    arm._hand_target = hand_targets[side].tolist()
                    arm.motion["waypoint_index"] = run["steps_executed"] + 1
                    arm.motion["diagnostics"]["steps_executed"] = run["steps_executed"] + 1
                if robot.on_control_sample is not None:
                    robot.on_control_sample({"description": "ACT JS 目标已调用 SDK ACT target submitted",
                                             "executor": "act_JS", "arm": side, "command_sent": True,
                                             "motion_id": run["run_id"], "sent_at": time.monotonic(),
                                             "waypoint_index": run["steps_executed"] + 1,
                                             "target_joints": targets[side].tolist(),
                                             "target_hand_raw": hand_targets[side].tolist(),
                                             "action": action.tolist()})
            run["steps_executed"] += 1
        return hand_targets

    def _hold(self, run, status, reason):
        robot = self.controller
        with robot._trajectory_lock:
            if robot._ended:
                return "paused", self._cancel_reason or reason
            if robot.manual_stop.is_set():
                return "manual_hold", self._cancel_reason or reason
            if self._cancel.is_set() and status == "completed":
                status, reason = "paused", self._cancel_reason
            if any(arm.fault for arm in robot.arms.values()):
                return "failed", "; ".join(arm.fault for arm in robot.arms.values() if arm.fault)
            run["phase"] = "holding"
            for arm in robot.arms.values():
                with arm._lock:
                    if robot.manual_stop.is_set():
                        return "manual_hold", self._cancel_reason or reason
                    arm.motion.update(status=status, reached=False, reason=reason,
                                      actual_joints=arm._joints.tolist())
                # 固定步数结束只捕获当前关节保持，不将模型最后一个目标判为到位。
                if not arm.cell_paused:
                    arm._pause(reason, pause_cell=status != "completed")
        deadline = time.monotonic() + 3
        while True:
            if robot._ended:
                return "paused", self._cancel_reason or reason
            if robot.manual_stop.is_set():
                return "manual_hold", self._cancel_reason or reason
            states = [arm.snapshot() for arm in robot.arms.values()]
            if any(state["fault"] for state in states):
                return "failed", "; ".join(state["fault"] for state in states if state["fault"])
            if all(state["quiescent"] for state in states):
                run["hold_confirmed"] = all(state["hold_confirmed"] for state in states)
                return status, reason
            if time.monotonic() >= deadline:
                raise RuntimeError("ACT 结束后位置保持未确认 ACT final hold not confirmed")
            time.sleep(.05)

    def _execute(self, run):
        started = time.monotonic()
        status, reason = "completed", "ACT 固定步数执行结束，任务效果需观察 ACT steps completed"
        try:
            check = lambda: self._check(run)
            self.client.ready(run["policy"], check)
            self.client.reset(check)
            run["phase"] = "executing"
            previous = None
            for _ in range(run["steps_requested"]):
                cycle = time.monotonic()
                frame, state, hands = self._observation(run)
                action = self.client.predict(state, frame.color, check)
                # 校验失败也保留模型输入/输出，便于定位具体越限轴。
                run["last_observation_state"] = state.tolist()
                run["last_action"] = action.tolist()
                previous = self._send(run, frame, action, hands, previous)
                self._cancel.wait(max(0., self.controller.period - (time.monotonic() - cycle)))
            check()
        except ActCancelled as exc:
            status, reason = "paused", str(exc)
        except (CommandRejected, OperationError) as exc:
            status, reason = "failed", str(exc)
        except Exception as exc:
            status, reason = "failed", f"ACT 设备执行故障 ACT execution fault: {exc}"
            self.controller.fail(reason)
        try:
            status, reason = self._hold(run, status, reason)
        except Exception as exc:
            status, reason = "failed", f"ACT 收尾失败 ACT cleanup failed: {exc}"
            self.controller.fail(reason)
        try:
            if status != "completed":
                # 中断可能留下半个请求/未读 action；下次调用重新启动模型，不复用错位的管道。
                self.client.close()
        except Exception as exc:
            status, reason = "failed", f"{reason}; ACT 模型进程清理失败 Worker cleanup failed: {exc}"
        finally:
            with self.controller._trajectory_lock:
                if self.controller.manual_stop.is_set():
                    status, reason = "manual_hold", self._cancel_reason or reason
                    run["hold_confirmed"] = all(arm.snapshot()["hold_confirmed"]
                                                for arm in self.controller.arms.values())
                elif self._cancel.is_set() and status == "completed":
                    status, reason = "paused", self._cancel_reason
                run.update(status=status, phase="finished", reason=reason,
                           elapsed_s=time.monotonic() - started)

    def close(self):
        self.cancel("控制进程关闭 ACT controller closing")
        if self._thread is not None:
            self._thread.join(1)
        if self.client is not None and (self._thread is None or not self._thread.is_alive()):
            self.client.close()
        elif self.client is not None:
            # 即使设备 SDK 尚未返回，也不能留下脱离控制进程的 GPU 子进程。
            process = self.client.process
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
