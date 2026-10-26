"""任务协调：把 MCP、worker、控制进程的生命周期连接起来。

本模块管理执行边界与记录，将 robo 代理和冻结配置注入 worker，不规划折布。
控制进程先锁定人工停止保持，再由本模块结束 worker/Codex；两者没有反向等待。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
from threading import Event, Lock, Thread
import time
import uuid

from .control import ControlProcess
from .worker import PythonWorker, WorkerUnavailable
from .config import load_config, validate_hardware_config, DEFAULT_CONFIG


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("时间必须为有限正数 Timeout must be finite and positive")
    return value


class TaskSession:
    """单任务协调器。execute 串行；abort/人工停止保持监测不等待 cell 的执行锁。"""

    def __init__(self, output_dir=None, *, cell_timeout_s=60., task_timeout_s=900., max_cells=40,
                 config_path=None, allow_hardware=False):
        self.cell_timeout_s = positive_seconds(cell_timeout_s)
        self.task_timeout_s = positive_seconds(task_timeout_s)
        if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
            raise ValueError("cell 次数必须为正整数 max_cells must be a positive integer")
        self.max_cells = max_cells
        if allow_hardware and config_path is None:
            config_path = DEFAULT_CONFIG
        self.robot_config = load_config(config_path) if config_path is not None else None
        if self.robot_config is not None:
            if self.robot_config["backend"] != "hardware" or set(self.robot_config["arms"]) != {"right"}:
                raise ValueError("robo 只接受右臂硬件配置 Only right-arm hardware configuration is supported")
            validate_hardware_config(self.robot_config)
        self.directory = Path(output_dir or (
            Path("data/agent_runs") / ("session-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                                      + "-" + uuid.uuid4().hex[:8]))).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        # 不覆盖旧记录；新任务使用新目录，方便逐次 review。
        self._events_file = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        (self.directory / "cells").mkdir()
        self._state_lock, self._cell_lock, self._log_lock = Lock(), Lock(), Lock()
        self._shutdown = Event()
        self.ended = Event()
        self.status = "running"
        self.reason = ""
        self.report = None
        self.cell_count = 0
        self.startup_prepared = False
        self.agent = None
        self.control = self.worker = None
        self._started_at = time.monotonic()
        try:
            self.control = ControlProcess(self.robot_config, self.directory, allow_hardware=allow_hardware)
            self.worker = PythonWorker(self.directory / "workspace", self.control.worker_connection,
                                       self.robot_config)
            self.control.worker_connection.close()  # 父进程不持有 worker 的普通设备通道。
            metadata = {"coordinator_pid": os.getpid(), "worker_pid": self.worker.process.pid,
                        "control_pid": self.control.process.pid,
                        "manual_socket": self.control.ready["manual_socket"],
                        "manual_protocol": self.control.ready["manual_protocol"],
                        "hardware_connected": self.control.ready["hardware_connected"],
                        "robo_available": True}
            self._write("session.json", metadata)
            self._write("robot_config.json", self.robot_config)
            self.log("session_started", "任务已启动 Session started; robo injected", **metadata)
        except BaseException:
            if self.worker:
                self.worker.stop()
            if self.control:
                self.control.close()
            self._events_file.close()
            raise
        self._monitor = Thread(target=self._watch, name="cloth-session-monitor", daemon=True)
        self._monitor.start()
        # 控制进程、人工 socket、worker 和 session.json 都准备好之后才允许自动运动。
        # 不能把准备运动放到设备构造函数里，否则操作者在运动时还拿不到 hold 入口。
        try:
            self._prepare_robot()
        except KeyboardInterrupt:
            self.abort("初始抓取姿态准备被终端取消 Startup preparation cancelled")
        except Exception as exc:
            self.abort(f"初始抓取姿态准备失败 Startup preparation failed: {exc}", status="failed")

    def _write(self, relative_path, value):
        (self.directory / relative_path).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(self, kind, description, **values):
        # 标识符保持英文便于解析；description 使用中英夹杂，原始源码和输出另存。
        event = {"type": kind, "description": description,
                 "timestamp_monotonic": time.monotonic(), **values}
        with self._log_lock:
            self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._events_file.flush()

    def summary(self):
        with self._state_lock:
            return {"status": self.status, "reason": self.reason, "report": self.report,
                    "cells": self.cell_count, "directory": str(self.directory),
                    "hardware_connected": self.control.ready["hardware_connected"],
                    "startup_prepared": self.startup_prepared,
                    "stop_confirmed": False,
                    "description": "机器人任务会话 Robot task session; task outcome needs visual evidence"}

    def _settle_control(self, state):
        # 控制进程发送普通 J 固定保持目标并监督反馈；宿主只做短查询等待确认。
        # 不等到位时不把 cell 结果交给模型，人工停止/故障则立即终止本次任务。
        deadline = time.monotonic() + 3
        while not state["quiescent"]:
            if state["fault"] or state["manual_hold_latched"] or time.monotonic() >= deadline:
                raise RuntimeError("控制收尾未确认 Cell/task hold not confirmed")
            time.sleep(.05)
            state = self.control.request("status")
        return state

    def _prepare_robot(self):
        """固定宿主流程，两次 PTP 回记录关节/初始抓取姿态；完成前不会启动 Codex。"""
        startup = (self.robot_config or {}).get("startup", {})
        if not self.control.ready["hardware_connected"] or not startup.get("enabled", False):
            return
        settings = self.robot_config["arms"]["right"]
        self.log("startup_started", "PTP 准备初始抓取姿态 Preparing initial grasp pose with PTP",
                 start_joints=settings["start_joint_positions"],
                 initial_grasp_tcp=settings["initial_grasp_tcp"],
                 reference_flange_yaw_rad=settings["reference_flange_yaw_rad"])
        print(json.dumps({"description": "即将自动准备初始抓取姿态 Preparing robot before agent; hold is available",
                          "session_file": str(self.directory / "session.json")}, ensure_ascii=False), flush=True)
        # 固定 cell 000 使用内部 PTP 辅助函数，复用轨迹复核、到位与暂停收尾。
        # 每段只下发一个最终关节目标；第二段只求最终姿态 IK，不做直线插值或 TOPPRA。
        # 源码由宿主构造，不由模型决定；不新增公开回零工具，也不占 agent 的 cell 预算。
        # 第一段只回记录关节，第二段到保持记录法兰位置/yaw 的示教抓取倾角。
        source = (
            # 与 teleop 回记录位姿一致：已在容差内就跳过第一段，避免为原地
            # 等待构造一条零位移轨迹、无必要地下发目标并立刻开始下一段。
            "startup_joints = np.array(" + repr(settings["start_joint_positions"]) + ", dtype=float)\n"
            "startup_state = robo.state()\n"
            "if np.max(np.abs(np.array(startup_state['joints']) - startup_joints)) > "
            + repr(self.robot_config["motion"]["joint_tolerance_rad"]) + ":\n"
            "    startup_joint_result = startup_move_ptp(target_joints=startup_joints.tolist())\n"
            "    print('PTP 回记录关节反馈 Recorded joint PTP:', startup_joint_result)\n"
            "    if startup_joint_result.status != 'completed' or not startup_joint_result.reached:\n"
            "        raise RuntimeError('回记录关节位姿失败 Recorded joint move failed: ' + repr(startup_joint_result))\n"
            "else:\n"
            "    print('已在记录关节位姿，跳过原地运动 Already at recorded joint pose; skip PTP')\n"
            "startup_grasp_result = startup_move_ptp(target_tcp=" + repr(settings["initial_grasp_tcp"]) + ")\n"
            "print('PTP 初始抓取位姿反馈 Initial grasp PTP:', startup_grasp_result)\n"
            "if startup_grasp_result.status != 'completed' or not startup_grasp_result.reached:\n"
            "    raise RuntimeError('初始抓取姿态准备失败 Initial grasp move failed: ' + repr(startup_grasp_result))\n"
        )
        result = self.execute(source, _startup=True)
        if self.status != "running":
            return  # 人工停止/超时已收尾，不将其改写成普通准备失败或继续发动作。
        if result["status"] != "completed":
            raise RuntimeError(result.get("exception") or result.get("reason") or str(result))
        from robot_control.nero_ik import pose_error, MAX_FK_POSITION_ERROR_M, MAX_FK_ROTATION_ERROR_DEG
        from .geometry import transform

        # 关节到位后还检查实际 TCP 与初始抓取位姿一致，不能只凭 Python cell 正常
        # 返回就向模型宣称“已准备”。沿用 IK 的位置/角度容差，不放宽规划门限。
        actual = transform(result["control"]["tcp"])
        target = transform(settings["initial_grasp_tcp"])
        position_error, angle_error = pose_error(actual, target)
        if position_error >= MAX_FK_POSITION_ERROR_M or math.degrees(angle_error) >= MAX_FK_ROTATION_ERROR_DEG:
            raise RuntimeError(f"初始抓取位姿反馈未达标 Startup pose feedback mismatch: "
                               f"{position_error} m, {math.degrees(angle_error)} deg")
        with self._state_lock:
            if self.status != "running" or self.control.manual_hold.is_set():
                return
            self.startup_prepared = True
        self.log("startup_prepared", "已到达初始抓取姿态 Initial grasp pose ready; yaw=0",
                 actual_tcp=actual.tolist(), position_error_m=position_error,
                 rotation_error_deg=math.degrees(angle_error))

    def execute(self, source, *, _startup=False):
        if not isinstance(source, str):
            raise ValueError("cell 源码必须是文本 Cell source must be text")
        with self._cell_lock:
            with self._state_lock:
                if self.status != "running":
                    return {"status": "rejected", "reason": self.reason or "任务已结束 Task ended"}
                exhausted = not _startup and self.cell_count >= self.max_cells
                if not exhausted and not _startup:
                    self.cell_count += 1
                cell_id = 0 if _startup else self.cell_count
            if exhausted:
                self.abort("cell 调用次数用尽 Cell budget exhausted")
                return {"status": "rejected", "reason": self.reason}
            prefix = f"cells/{cell_id:03d}"
            (self.directory / (prefix + ".py")).write_text(source, encoding="utf-8")
            self.log("cell_started", "开始执行 cell Cell started", cell_id=cell_id)
            result = None
            try:
                self.control.request("begin_cell", cell_id=cell_id)
                timeout = self.robot_config["startup"]["timeout_s"] if _startup else self.cell_timeout_s
                result = self.worker.execute(source, cell_id, timeout)
                control = self.control.request("cell_end")
                interrupted = control.get("cell_interrupted", False)
                result["control"] = self._settle_control(control)
                if interrupted:
                    result.update(status="paused", reason="cell 遗留异步运动，已暂停 Cell returned with an active motion")
                if control["manual_hold_latched"]:
                    self.abort("人工位置保持 Manual position hold", status="manual_hold")
            except (WorkerUnavailable, OSError, EOFError, TimeoutError, RuntimeError) as exc:
                # 收尾可能在 worker 已返回之后失败。保留原来的 stdout、异常、调用
                # 记录和最后控制快照，不能再用“保持未确认”覆盖起点变化等首发错误。
                state_at_failure = self.control.last_state
                result = result if result is not None else {"cell_id": cell_id}
                result["execution_status"] = result.get("status", "interrupted")
                result["control"] = state_at_failure
                result["control_error"] = str(exc)
                original = result.get("exception", "").strip().splitlines()
                reason = ((original[-1] + "；收尾失败 Cleanup failed: ") if original else "") + str(exc)
                self.abort(reason)
                result.update(status="interrupted", reason=self.reason)
            # 竞争窗口内即使 worker 已算出结果，只要任务已终止就不能再报告 cell 正常完成。
            if self.status != "running":
                result.update(status="interrupted", reason=self.reason)
            self._write(prefix + ".json", result)
            self.log("cell_returned", "cell 返回 Cell returned", cell_id=cell_id,
                     status=result["status"], robot_calls=result.get("robot_calls", []))
            return result

    def finish(self, status, summary):
        if status not in {"success", "failure", "stuck"}:
            raise ValueError("非法结束状态 Invalid finish status")
        # 同一 MCP session 中不允许 finish 越过尚在执行的 cell；人工停止保持不受这个限制。
        if not self._cell_lock.acquire(blocking=False):
            raise RuntimeError("请先等待 cell 收尾 Wait for the active cell")
        try:
            with self._state_lock:
                if self.status != "running":
                    raise RuntimeError("任务已结束 Task already ended")
                self.report = {"status": status, "summary": summary}
            self._end("completed" if status == "success" else status,
                      "agent 声明结束 Agent requested finish", stop_agent=False)
            return self.summary()
        finally:
            self._cell_lock.release()

    def _stop_agent(self):
        # codex 使用独立进程组；取消任务也停止它可能创建的子进程。
        # 这里的等待只影响模型清理，不会延迟控制进程的人工停止保持锁定。
        agent = self.agent
        if agent is not None and agent.poll() is None:
            try:
                os.killpg(agent.pid, signal.SIGTERM)
                try:
                    agent.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    os.killpg(agent.pid, signal.SIGKILL)
                    agent.wait(timeout=.5)
            except ProcessLookupError:
                pass

    def attach_agent(self, process):
        with self._state_lock:
            self.agent = process
            running = self.status == "running"
        if not running:
            self._stop_agent()

    def _end(self, status, reason, *, stop_agent=True):
        with self._state_lock:
            # 人工停止保持优先于正常完成，任何普通结束请求都不能解除人工保持锁定。
            if self.status != "running" and not (status == "manual_hold" and self.status != "manual_hold"):
                return
            self.status, self.reason = status, reason
        try:
            # 人工保持已经由控制进程自身处理；不要再把它塞回普通请求通道等待一次。
            if status != "manual_hold":
                if not self.control.process.is_alive():
                    raise RuntimeError("控制进程已退出，收尾未确认 Control exited before acknowledgement")
                self._settle_control(self.control.request("end_task", reason=reason))
        except (OSError, EOFError, TimeoutError, RuntimeError) as exc:
            with self._state_lock:
                # 人工停止保持与普通收尾可能并发；后者失败不能覆盖已经锁定的人工保持状态。
                self.status = "manual_hold" if self.control.manual_hold.is_set() else "failed"
                self.reason += f"；控制收尾未确认 Control acknowledgement missing: {exc}"
        finally:
            self.worker.stop()
            if stop_agent:
                self._stop_agent()
        self.log("session_ended", "任务结束 Session ended", result=self.summary())
        self._write("result.json", self.summary())
        # 撤销动作准入在上方立即生效；结束通知在实际收尾后才发给 serve/Codex 退出路径。
        self.ended.set()

    def abort(self, reason, *, status="aborted"):
        self._end(status, reason)

    def _watch(self):
        # 监听独立 Event，不获取 cell 执行锁、MCP 锁或控制普通请求锁。
        # 人工保持由控制进程先行锁定；此线程仅负责后续任务/模型清理。
        while not self._shutdown.wait(.02):
            if self.control.manual_hold.is_set():
                self.abort("操作者锁定位置保持 Operator latched position hold", status="manual_hold")
            elif self.status == "running":
                if not self.control.process.is_alive():
                    self.abort("控制进程退出 Control process exited", status="failed")
                elif self.control.failed.is_set():
                    # Event 只表示故障存在，具体首发错误由控制快照携带。
                    # 不能让监视线程抢先用泛化描述结束任务，掩盖 worker 已收到的
                    # stale feedback/CAN 故障；保留原因供终端和 result.json review。
                    try:
                        state = self.control.request("status")
                        detail = state.get("fault") or "故障细节缺失 Device fault detail unavailable"
                    except (OSError, EOFError, TimeoutError, RuntimeError) as exc:
                        detail = str(exc)
                    self.abort("控制设备故障 Device/control fault: " + detail, status="failed")
                elif not self.worker.process.is_alive():
                    self.abort("worker 退出 Python worker exited", status="failed")
                elif time.monotonic() - self._started_at >= self.task_timeout_s:
                    self.abort("任务执行超时 Task timed out")

    def close(self):
        self.abort("协调进程关闭 Session closing")
        self._shutdown.set()
        self._monitor.join(3)
        self._stop_agent()
        self.worker.stop()
        self.control.close()
        self._events_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
