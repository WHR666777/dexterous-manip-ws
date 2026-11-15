"""重放一个历史 move_L 的实际关节分段；不启动 Codex/MCP，不重新求 IK。

先用一个低速 J PTP 回到历史轨迹起点，再把原始 MotionPlan 原样送入
agent 使用的 ControlProcess。限位、起点检查、到位判定、超时和人工 hold
全部复用正式执行器。只重放运动，不恢复历史抓布/开合手动作。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_agent.control import ControlProcess
from cloth_agent.planning import MotionPlan, Planner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", help="历史任务 motions/<id>.json")
    parser.add_argument("--output-dir", required=True, help="新的记录目录，不覆盖旧记录")
    parser.add_argument("--joint-frame-gap-ms", type=float,
                        help="发送时序对照：关节目标帧最小间隔 ms；省略沿用冻结配置或当前默认 1 ms")
    args = parser.parse_args()
    source = Path(args.motion).resolve()
    # 读取当轮冻结配置，避免后来改过的 TCP 标定、速度、步长悄悄进入对照。
    config = json.loads((source.parent.parent / "robot_config.json").read_text())
    # 旧记录可能没有这个字段；补到本轮冻结配置，明确实际生效的发送节拍。
    config["motion"].setdefault("joint_frame_interval_s", .001)
    if args.joint_frame_gap_ms is not None:
        if not np.isfinite(args.joint_frame_gap_ms) or args.joint_frame_gap_ms < 0:
            parser.error("帧间隔必须有限且非负 Frame gap must be finite and nonnegative")
        config["motion"]["joint_frame_interval_s"] = args.joint_frame_gap_ms / 1000.
    # 只打开可信宿主 cell 000 的 PTP 准入，不自动运行日常启动准备。
    # 必须在启动控制子进程前设置，spawn 之后修改父进程配置不会传入子进程。
    config["startup"]["enabled"] = True
    original = json.loads(source.read_text())
    plan = MotionPlan.from_dict(original)
    if plan.mode != "cartesian" or plan.arm != "right":
        raise ValueError("仅重放右臂 move_L 历史计划 Expected right-arm Cartesian motion")
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "robot_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (directory / "source-motion.json").write_text(json.dumps(original))
    report = {"description": "历史 move_L 原始分段重放 Recorded move_L replay",
              "source_motion": str(source), "passed": False,
              "original_waypoint_count": len(plan.joint_waypoints) - 1}
    control = None
    try:
        control = ControlProcess(config, directory, allow_hardware=True)
        session_file = directory / "session.json"
        session_file.write_text(json.dumps({
            "coordinator_pid": os.getpid(), "control_pid": control.process.pid,
            "manual_socket": control.ready["manual_socket"],
            "manual_protocol": control.ready["manual_protocol"],
            "hardware_connected": control.ready["hardware_connected"],
        }, ensure_ascii=False, indent=2))
        print(json.dumps({"description": "原始轨迹重放就绪 Recorded motion replay ready",
                          "session_file": str(session_file), **report}, ensure_ascii=False), flush=True)

        def worker_request(operation, **values):
            # 历史计划已经计算完成；可信宿主直接使用控制进程的 worker Pipe。
            # 不执行生成代码，也不新增 agent 可调用的公开 plan/execute 接口。
            control.worker_connection.send({"operation": operation, "cell_id": cell_id, **values})
            if not control.worker_connection.poll(5):
                raise TimeoutError("重放请求未确认 Replay request not acknowledged")
            reply = control.worker_connection.recv()
            if not reply["ok"]:
                raise RuntimeError(reply["reason"])
            return reply["result"]

        def execute(recorded, label):
            control.request("begin_cell", cell_id=cell_id)
            started = worker_request("start_motion", plan=recorded)
            deadline = time.monotonic() + 90
            while True:
                state = control.request("status")
                if state["manual_hold_latched"] or state["fault"]:
                    raise RuntimeError("重放已被停止 Replay stopped: " + str(state))
                result = worker_request("motion_status", motion_id=started["motion_id"])
                if result["status"] != "running":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("重放监督超时 Replay supervision timed out")
                time.sleep(.05)
            report[label] = {"result": result, "state": state}
            print(json.dumps({"description": label, **report[label]}, ensure_ascii=False), flush=True)
            control.request("cell_end")
            if result["status"] != "completed" or not result["reached"]:
                raise RuntimeError("历史运动未完成 Recorded motion failed: " + result["reason"])

        # 只将准备段构造成两个端点的 PTP；真正的重放段不插值、不改点、不重算。
        # PTP 仍检查路径/限位，并由控制进程确认到位后才接收下一条运动。
        cell_id = 0
        current = control.request("status")["joints"]
        seed = plan.joint_waypoints[0]
        if np.max(np.abs(np.asarray(current) - seed)) > config["motion"]["joint_tolerance_rad"]:
            preparation = Planner(config).plan_startup_ptp(current, target_joints=seed)
            execute(preparation.as_dict(), "回历史起点 PTP to recorded start")
        cell_id = 1
        # 使用 original 字典，保留历史 times、positions、velocities 和
        # joint_waypoints 原值；正式 J 执行器只按 joint_waypoints 推进。
        execute(original, "原始 move_L 分段 Recorded move_L waypoints")
        report["passed"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"description": "重放未通过 Replay failed", "reason": report["error"]},
                         ensure_ascii=False), flush=True)
    finally:
        if control is not None:
            try:
                # 与任务结束一样撤销准入，保留最后目标；不自动重试、松手、
                # 使能或复位。人工 hold 若已锁定，close 会等待明确 release。
                report["final_state"] = control.request("end_task", reason="历史运动重放结束 Recorded replay ended")
            except (RuntimeError, OSError, EOFError, TimeoutError) as exc:
                report["cleanup_error"] = str(exc)
            control.close()
        (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
