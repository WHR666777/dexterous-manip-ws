"""厂家 move_l 实机检查；不接入 robo，不启动 Codex/MCP。

先用普通 J PTP 回 configs/nero_start_pose.yaml 的记录关节位姿；不调整
初始抓取朝向。随后保持实测朝向，沿 task +Z 上升 4 cm，只发送一次
NeroArm.move_linear。--suite reliability 先检查历史水平线，再扩大范围并重复。
复用独立控制进程和人工 hold，不操作手指，任何失败均停止后续目标。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_agent.config import load_config
from cloth_agent.control import ControlProcess, request_manual
from cloth_agent.geometry import tcp_to_flange, transform
from cloth_agent.hardware import RobotController
from cloth_agent.paths import DEFAULT_CONFIG
from cloth_agent.planning import Planner
from robot_control.nero_ik import pose_error


class LinearCheckController(RobotController):
    """仅测试脚本使用的真实设备控制器，新增操作只接受可信宿主管理 Pipe。"""

    def __init__(self, *args):
        self.native_linear = None
        self.native_sequence = 0
        super().__init__(*args)

    def _dispatch_owner(self, request):
        if request["operation"] == "begin_cell" and self.native_linear is not None:
            if not self.snapshot()["quiescent"]:
                raise ValueError("上一段厂家 L 尚未稳定 Previous native L is not quiescent")
            # 上一段确认完成后允许可信宿主开始下一段；正式 robo 不增加接口。
            self.native_linear = None
        if request["operation"] != "check_native_linear":
            return super()._dispatch_owner(request)
        self._admit(request["cell_id"])
        self._require_hardware()
        if not self.snapshot()["quiescent"]:
            raise ValueError("厂家 L 检查需要静止起点 Native L check requires a quiescent start")
        target_tcp = transform(request["target_tcp"])
        flange = tcp_to_flange(target_tcp, self.settings["task_from_base"], self.settings["flange_from_tcp"])
        pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')]
        with self._arm_send_lock:
            self._admit(request["cell_id"])  # 等发送锁期间人工 hold 可能已经撤销准入。
            # 撤掉旧 PTP 的软件保持目标，避免轨迹线程在厂家 L 运动期间
            # 用旧关节目标覆盖它。这里没有伪造 J MotionPlan 或额外下发关节点。
            self.motion, self._hold = None, None
            self._hold_confirmed = False
            self._hold_feedback = None
            self.native_sequence += 1
            self.native_linear = {"status": "running", "target_tcp": target_tcp.tolist(),
                                  "start_tcp": self._tcp, "sequence": self.native_sequence,
                                  "label": request.get("label", "native_L"),
                                  "max_cross_track_error_m": 0., "max_angle_error_rad": 0.,
                                  "flange_pose": pose.tolist(), "stable_since": None}
            self.arm.move_linear(pose, speed_percent=self.motion_settings["joint_speed_percent"])
            self.native_linear["sent_at_unix"] = time.time()
            self.native_linear["sent_at"] = time.monotonic()
        return self.snapshot()

    def snapshot(self):
        state = super().snapshot()
        if self.native_linear is not None:
            state["executor"] = "native_linear_L"
            state["native_linear"] = dict(self.native_linear)
            state["quiescent"] = (self.native_linear["status"] == "completed"
                                  and state["hold_confirmed"]
                                  and not state["fault"] and not state["manual_hold_latched"])
        return state

    def _tick(self, delta):
        if self.native_linear is None:
            return super()._tick(delta)  # 前置 PTP 完整复用正式执行器。
        if not self.connected or self.manual_stop.is_set():
            return
        self._measure()
        native = self.native_linear
        actual, target, start = (transform(value) for value in
                                 (self._tcp, native["target_tcp"], native["start_tcp"]))
        position_error, angle_error = pose_error(actual, target)
        direction = target[:3, 3] - start[:3, 3]
        length_squared = float(direction @ direction)
        fraction = (float(np.clip((actual[:3, 3] - start[:3, 3]) @ direction / length_squared, 0, 1))
                    if length_squared > 1e-12 else 0.)
        cross_error = float(np.linalg.norm(actual[:3, 3] - (start[:3, 3] + fraction * direction)))
        native["max_cross_track_error_m"] = max(native["max_cross_track_error_m"], cross_error)
        native["max_angle_error_rad"] = max(native["max_angle_error_rad"], angle_error)
        stable = (self._hold_feedback is not None
                  and np.max(np.abs(self._joints - self._hold_feedback)) <= .002)
        self._hold_feedback = self._joints.copy()
        # 不单独信任固件 motion_status=0：必须实测 TCP 到位、关节反馈稳定，
        # 且收到目标发送后的新鲜 L 模式反馈。容差与现有 IK/末态检查一致。
        arrived = (position_error <= .002 and angle_error <= np.deg2rad(2) and stable
                   and self._arm_status["mode_feedback"] == 2
                   and self._arm_status["received_at_unix"] > native["sent_at_unix"])
        native["stable_since"] = (native["stable_since"] or time.monotonic()) if arrived else None
        self._hold_confirmed = native["stable_since"] is not None and time.monotonic() - native["stable_since"] >= .2
        if native["status"] == "running":
            # 本套测试全程固定朝向。监测的不只是终点，明显离开直线走廊或
            # 改变朝向也判失败，由宿主进入独立人工 hold，取消剩余测试。
            if cross_error > .005 or angle_error > np.deg2rad(2):
                native.update(status="failed", reason="厂家 L 偏离测试直线 Native L path deviation")
            elif self._hold_confirmed:
                native["status"] = "completed"
            elif time.monotonic() - native["sent_at"] > 30:
                native.update(status="failed", reason="厂家 L 到位超时 Native L arrival timed out")
        native.update(position_error_m=position_error, angle_error_rad=angle_error)
        if self.on_control_sample is not None:
            self.on_control_sample({"command_sent": False, "executor": "native_linear_L",
                                    "sampled_at": self._sampled_at, "actual_joints": self._joints.tolist(),
                                    "tcp": self._tcp, "arm_status": dict(self._arm_status),
                                    "native_sequence": native["sequence"], "label": native["label"],
                                    "cross_track_error_m": cross_error,
                                    "position_error_m": position_error, "angle_error_rad": angle_error})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True, help="新的实机记录目录，不覆盖旧记录")
    parser.add_argument("--suite", choices=("lift", "reliability"), default="lift",
                        help="lift 单次上升；reliability 历史水平线及更大范围的重复运动")
    parser.add_argument("--history-motion", default=str(Path(__file__).resolve().parents[2] /
                        "data/agent_runs/fold-8/motions/07cf6d1d27554ca09874a234847240ab.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    config["startup"]["enabled"] = True  # 只开放宿主 cell 000 的记录关节 PTP。
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "robot_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    report = {"description": "原生 L 实机检查 Native move_l hardware check", "passed": False,
              "suite": args.suite, "trials": [], "ptp_moves": []}
    control = None
    try:
        control = ControlProcess(config, directory, allow_hardware=True, _controller_type=LinearCheckController)
        session_file = directory / "session.json"
        session_file.write_text(json.dumps({"coordinator_pid": os.getpid(), "control_pid": control.process.pid,
                                           "manual_socket": control.ready["manual_socket"],
                                           "manual_protocol": control.ready["manual_protocol"]}, indent=2))
        print(json.dumps({"description": "测试就绪 Native L check ready; manual hold available",
                          "session_file": str(session_file)}, ensure_ascii=False), flush=True)

        def check_state():
            state = control.request("status")
            if state["fault"] or state["manual_hold_latched"]:
                raise RuntimeError("设备故障或人工停止 Device fault/manual hold: " + str(state["fault"]))
            return state

        planner = Planner(config)

        def save_progress():
            (directory / "progress.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

        def ptp(joints, label):
            state = check_state()
            plan = planner.plan_startup_ptp(state["joints"], target_joints=joints)
            control.request("begin_cell", cell_id=0)
            control.worker_connection.send({"operation": "start_motion", "cell_id": 0, "plan": plan.as_dict()})
            if not control.worker_connection.poll(5):
                raise TimeoutError("PTP 请求未确认 PTP start not acknowledged")
            reply = control.worker_connection.recv()
            if not reply["ok"]:
                raise RuntimeError(reply["reason"])
            deadline = time.monotonic() + 120
            while True:
                state = check_state()
                if state["motion"]["status"] != "running":
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("PTP 到位超时 PTP arrival timed out")
                time.sleep(.05)
            report["ptp_moves"].append({"label": label, "state": state})
            save_progress()
            if state["motion"]["status"] != "completed" or not state["motion"]["reached"]:
                raise RuntimeError("PTP 失败 PTP failed: " + state["motion"]["reason"])
            control.request("cell_end")
            print(json.dumps({"description": label, "joint_error_rad": state["motion"]["joint_error_rad"]},
                             ensure_ascii=False), flush=True)
            return state

        def linear(target, label):
            before = check_state()
            # 主机采样 IK 只做可达性/限位/工作区预检，不向设备发送这些点。
            # 实际轨迹仍由一次厂家 move_l 生成，冗余关节选择由固件决定。
            preflight = planner.plan("right", before["joints"], waypoints=[target], mode="cartesian")
            trial = {"label": label, "before": before, "target_tcp": target.tolist(),
                     "planned_distance_m": float(np.linalg.norm(target[:3, 3] - transform(before["tcp"])[:3, 3])),
                     "preflight_points": len(preflight.joint_waypoints), "passed": False}
            report["trials"].append(trial)
            save_progress()
            cell_id = len(report["trials"])
            control.request("begin_cell", cell_id=cell_id)
            control.request("check_native_linear", cell_id=cell_id, target_tcp=target.tolist(), label=label)
            print(json.dumps({"description": "已发单次 L Native L submitted", "label": label,
                              "distance_m": trial["planned_distance_m"]}, ensure_ascii=False), flush=True)
            while True:
                state = check_state()
                if state["native_linear"]["status"] != "running":
                    break
                time.sleep(.05)
            trial.update(after=state, passed=state["native_linear"]["status"] == "completed")
            save_progress()
            print(json.dumps({"description": "L 阶段结果 Native L result", "label": label,
                              **state["native_linear"]}, ensure_ascii=False), flush=True)
            if not trial["passed"]:
                raise RuntimeError(state["native_linear"]["reason"])
            control.request("cell_end")
            return state

        # 原位置严格指 nero_start_pose.yaml 的记录七轴关节；明确执行 PTP，
        # 不把当前位置或约定的初始抓取姿态替代记录位姿。
        recorded = np.asarray(config["arms"]["right"]["start_joint_positions"])
        state = ptp(recorded, "PTP 回记录关节 Return to recorded joints")
        report["preparation"], report["before"] = state, state
        if args.suite == "lift":
            start = transform(state["tcp"])
            target = start.copy()
            target[2, 3] += .04
            state = linear(target, "固定朝向上升 4 cm Fixed-orientation lift")
            report["target_tcp"] = target.tolist()
            report["displacement_m"] = (transform(state["tcp"])[:3, 3] - start[:3, 3]).tolist()
        else:
            source = Path(args.history_motion).resolve()
            historical = json.loads(source.read_text())
            q_start, q_end = np.asarray(historical["joint_waypoints"])[[0, -1]]
            settings = config["arms"]["right"]
            task_base, flange_tcp = transform(settings["task_from_base"]), transform(settings["flange_from_tcp"])
            historical_end = task_base @ planner.ik["right"].forward(q_end) @ flange_tcp
            state = ptp(q_start, "PTP 到历史水平线起点 PTP to historical start")
            anchor = transform(state["tcp"])
            rotation = anchor[:3, :3].copy()

            def pose(xyz):
                result = np.eye(4)
                result[:3, :3], result[:3, 3] = rotation, xyz
                return result

            report["history_source"] = str(source)
            # 同一水平线往返 5 次；固定起点实测朝向，不把历史终点的微小
            # FK 姿态误差引入旋转测试。每个方向只发送一次完整 L 终点。
            for index in range(1, 6):
                linear(pose(historical_end[:3, 3]), f"历史水平线 {index}/5 去程 Historical outbound")
                linear(anchor, f"历史水平线 {index}/5 回程 Historical inbound")

            stages = [
                ("水平 30 cm Horizontal", [-.50, -.15, .22], [-.50, .15, .22]),
                ("垂直 12 cm Vertical", [-.50, 0., .16], [-.50, 0., .28]),
                ("水平对角 36 cm Horizontal diagonal", [-.58, -.15, .22], [-.38, .15, .22]),
                ("空间对角 38 cm Spatial diagonal", [-.58, -.15, .16], [-.38, .15, .28]),
            ]
            for label, a, b in stages:
                linear(pose(a), label + " 进入起点 Approach")
                for index in range(1, 4):
                    linear(pose(b), f"{label} {index}/3 去程 Outbound")
                    linear(pose(a), f"{label} {index}/3 回程 Inbound")
            corners = [[-.58, -.15, .22], [-.38, -.15, .22],
                       [-.38, .15, .22], [-.58, .15, .22]]
            linear(pose(corners[0]), "矩形起点 Rectangle start")
            for index in range(1, 3):
                for corner_index, xyz in enumerate(corners[1:] + corners[:1], 1):
                    linear(pose(xyz), f"矩形 {index}/2 边 {corner_index}/4 Rectangle edge")
            linear(anchor, "返回历史起点 Return to historical start")
            state = ptp(recorded, "测试完成 PTP 回记录关节 Final PTP to recorded joints")
        report["after"], report["passed"] = state, True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        # 故障不自动重试/返回。人工 hold 入口仍独立有效；若已锁定，close
        # 沿用原有 release-hold 流程，不用 terminate 带走保持连接。
    finally:
        if control is not None:
            if report["passed"]:
                control.request("end_task", reason="厂家 L 检查结束 Native L check finished")
            else:
                report["hold"] = request_manual(directory / "session.json", "hold",
                                                reason="厂家 L 检查未完成 Native L check failed")
            (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            control.close()
        else:
            (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
