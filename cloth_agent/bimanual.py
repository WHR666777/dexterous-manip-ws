"""同一控制进程内复用两套单臂执行器，相机和 cell 生命周期共用。"""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from threading import RLock
import time

from robot_control.errors import CommandRejected
from .act import ActExecution, prepare_act_hands
from .hardware import RobotController as ArmController


class RobotController:
    def __init__(self, config, directory, manual_stop, failed, allow_hardware):
        self.config, self.directory = config, Path(directory)
        self.manual_stop, self.failed = manual_stop, failed
        self.period = 1. / (config or {}).get("control_hz", 20.)
        self.camera = None
        self.arms = {}
        self._trajectory_lock = RLock()
        self._active_cell = None
        self._ended = False
        self._on_control_sample = None
        self.act = ActExecution(self)
        try:
            for name in (config or {}).get("arms", {"right": {}}):
                settings = deepcopy(config) if config else None
                if settings:
                    settings["arms"] = {name: settings["arms"][name]}
                    settings["camera"] = {"enabled": False}
                arm_dir = self.directory / "arms" / name
                arm_dir.mkdir(parents=True, exist_ok=True)
                arm = ArmController(settings, arm_dir, manual_stop, failed, False, arm_name=name)
                self.arms[name] = arm
                if allow_hardware:
                    arm._connect()
            if allow_hardware and config.get("camera", {}).get("enabled"):
                from teleop.realsense_camera import RealSenseCamera

                self.camera = RealSenseCamera(config["camera"])
                self.camera.start()
        except BaseException:
            self.close()
            raise

    @property
    def connected(self):
        return bool(self.arms) and all(arm.connected for arm in self.arms.values())

    @property
    def motion(self):
        # 控制进程的关闭检查兼容单臂专用检查器。
        motions = [arm.motion for arm in self.arms.values() if arm.motion]
        return next((motion for motion in motions if motion["status"] == "running"),
                    motions[0] if motions else None)

    @property
    def on_control_sample(self):
        return self._on_control_sample

    @on_control_sample.setter
    def on_control_sample(self, callback):
        self._on_control_sample = callback
        for name, arm in self.arms.items():
            arm.on_control_sample = (lambda sample, side=name: callback({"arm": side, **sample})) if callback else None

    def snapshot(self):
        states = {name: arm.snapshot() for name, arm in self.arms.items()}
        default = states.get("right", next(iter(states.values())))
        reports = {name: state["manual_hold_report"] for name, state in states.items()}
        report = None
        if any(value is not None for value in reports.values()):
            report = {"arms": reports, "error": "; ".join(
                f"{name}: {value['error']}" for name, value in reports.items()
                if value and value.get("error"))}
            report.update({name: (value or {}).get(name, "pending") for name, value in reports.items()})
        # 顶层 joints/tcp 等保留默认右侧视图；arms 始终包含所有侧。
        return {**default, "arms": states, "hardware_connected": self.connected,
                "quiescent": not self.act.active and all(state["quiescent"] for state in states.values()),
                "hold_confirmed": all(state["hold_confirmed"] for state in states.values()),
                "fault": "; ".join(f"{name}: {state['fault']}" for name, state in states.items() if state["fault"]),
                "manual_hold_report": report, "act": self.act.snapshot()}

    _observe = ArmController._observe
    _capture_observation = ArmController._capture_observation

    def _arm(self, name):
        if name not in self.arms:
            raise CommandRejected(f"未配置机械臂 Arm is not configured: {name}")
        return self.arms[name]

    def _admit(self, cell_id):
        if self._ended or self.manual_stop.is_set() or cell_id != self._active_cell:
            raise CommandRejected("cell 已失效 Cell admission revoked")

    def _start_motions(self, cell_id, plans):
        self._admit(cell_id)
        for arm in self.arms.values():
            arm._admit(cell_id)
        if not isinstance(plans, dict) or not plans:
            raise CommandRejected("运动请求必须包含机械臂目标 Expected arm plans")
        prepared = {}
        for name, payload in plans.items():
            arm = self._arm(name)
            arm._admit(cell_id)
            # IK/路径检查与文件写入不占用周期执行锁。
            prepared[name] = arm._start_motion(cell_id, payload, prepare_only=True)
        with self._trajectory_lock, ExitStack() as stack:
            for arm in self.arms.values():
                arm._admit(cell_id)
            for name in prepared:
                stack.enter_context(self.arms[name]._trajectory_lock)
            feedback = {name: self.arms[name]._check_motion_start(cell_id, motion)
                        for name, motion in prepared.items()}
            if len(prepared) > 1:
                # 两侧都空闲后才一起发布；L 不允许一侧先等旧运动结束。
                for name in prepared:
                    status = self.arms[name]._read_arm_status()
                    if status["motion_status"] != 0:
                        raise CommandRejected(f"{name}: 双臂启动需要两侧固件空闲 Both arms must be idle")
            results = {}
            previous = {name: self.arms[name].motion for name in prepared}
            started_at = time.monotonic()
            try:
                for name, motion in prepared.items():
                    motion["software_start_at"] = started_at
                    results[name] = self.arms[name]._install_motion(cell_id, motion, feedback[name])
            except BaseException:
                # 周期线程尚未获得锁，不存在已发送的半组目标。
                for name, motion in prepared.items():
                    arm = self.arms[name]
                    with arm._lock:
                        if arm.motion is motion and not self.manual_stop.is_set():
                            arm.motion = previous[name]
                raise
            return results

    def dispatch(self, request, *, owner):
        operation = request["operation"]
        if owner:
            with self._trajectory_lock:
                if operation == "status":
                    return self.snapshot()
                if operation == "begin_cell":
                    if not self.snapshot()["quiescent"] or self._ended:
                        raise CommandRejected("控制状态不允许新 cell Arms are not ready for a new cell")
                elif operation in {"cell_end", "end_task"}:
                    self.act.cancel(request.get("reason", "cell 已结束 Cell ended"))
                    self._active_cell = None
                    self._ended = self._ended or operation == "end_task"
                else:
                    raise CommandRejected("未知协调请求 Unknown coordinator request")
                results, errors = [], []
                for arm in self.arms.values():
                    try:
                        results.append(arm.dispatch(request, owner=True))
                    except Exception as exc:
                        errors.append(str(exc))
                if errors:
                    raise RuntimeError("; ".join(errors))
                if operation == "begin_cell":
                    self._active_cell = request["cell_id"]
                return {**self.snapshot(), "cell_interrupted": any(
                    result.get("cell_interrupted", False) for result in results)}
        cell_id = request.get("cell_id")
        self._admit(cell_id)
        if operation == "state":
            if request.get("arm") is not None:
                return self._arm(request["arm"]).snapshot()
            return self.snapshot()
        if operation == "observe":
            return self._observe()
        if operation == "pause":
            return self.pause(request["reason"])
        if operation == "act_status":
            return self.act.status(request["run_id"])
        if operation == "start_act":
            return self.act.start(cell_id, request["policy"], request["steps"])
        if self.act.active:
            raise CommandRejected("ACT 正在独占双臂与双手，请先等待或暂停 ACT is running")
        if operation == "prepare_act_hands":
            return prepare_act_hands(self, cell_id)
        if operation == "start_motions":
            if set(request["plans"]) != {"left", "right"}:
                raise CommandRejected("双臂请求需要左右两个目标 Both arm targets are required")
            return self._start_motions(cell_id, request["plans"])
        if operation == "start_motion":
            name = request.get("arm", "right")
            return self._start_motions(cell_id, {name: request["plan"]})[name]
        if operation != "motion_status":
            for arm in self.arms.values():
                arm._admit(cell_id)
        return self._arm(request.get("arm", "right")).dispatch(request, owner=False)

    def tick(self, delta):
        with self._trajectory_lock:
            for arm in self.arms.values():
                arm.tick(delta)
                if arm.cell_paused and not self.manual_stop.is_set():
                    reason = (arm.motion or {}).get("reason") or "另一侧已暂停 Other arm paused"
                    self.pause(reason)

    def pause(self, reason):
        with self._trajectory_lock:
            self.act.cancel(reason)
            errors = []
            for arm in self.arms.values():
                if arm.cell_paused:
                    continue
                try:
                    arm.pause(reason)
                except Exception as exc:
                    errors.append(f"{arm.arm_name}: {exc}")
            if errors:
                raise RuntimeError("; ".join(errors))
            return self.snapshot()

    def fail(self, reason):
        self.act.cancel(reason)
        errors = []
        with self._trajectory_lock:
            for arm in self.arms.values():
                try:
                    arm.fail(reason)
                except Exception as exc:
                    errors.append(f"{arm.arm_name}: {exc}")
        self.failed.set()
        if errors:
            # 保留原因并继续人工入口，不使运动线程因二次清理错误退出。
            for arm in self.arms.values():
                arm.fault = reason + "; " + "; ".join(errors)

    def manual_hold(self, reason):
        # 不取得普通轨迹锁；先撤销两侧权限，再分别捕获各自固定目标。
        self.manual_stop.set()
        self.act.cancel(reason)
        for arm in self.arms.values():
            arm.manual_hold(reason)
        return self.snapshot()

    def manual_hold_tick(self):
        for arm in self.arms.values():
            arm.manual_hold_tick()

    def close(self):
        errors = []
        self._ended = True
        try:
            self.act.close()
        except Exception as exc:
            errors.append(f"ACT: {exc}")
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as exc:
                errors.append(str(exc))
        for name, arm in self.arms.items():
            try:
                arm.close()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("设备清理失败 Device cleanup failed: " + "; ".join(errors))
