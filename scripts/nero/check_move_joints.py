"""右臂普通 J 单目标实机检查：原位保持、J7 小幅运动、返回。

本脚本用于比较普通 J 与 JS 的真实跟踪表现，不经过 agent 的轨迹发送循环。
每段只调用一次 move_joints，之后只读反馈；不重新使能、不复位、不切入 JS，
不连接或操作 L20。需要现场监护及可限制意外下坠的支撑。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cloth_agent.config import DEFAULT_CONFIG, load_config
from cloth_agent.geometry import transform
from robot_control.nero import NeroArm
from robot_control.nero_ik import NeroIK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="连接右臂并发送三个有限 J 目标")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True, help="新的记录目录，不覆盖旧文件")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("实机检查需要显式 --execute Hardware execution gate required")

    config = load_config(args.config)
    settings = config["arms"]["right"]
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    ik = NeroIK()
    task_base = transform(settings["task_from_base"])
    flange_tcp = transform(settings["flange_from_tcp"])
    arm = NeroArm(can_channel=settings["can_channel"], max_joint_delta=.025)
    report = {"description": "普通 J 单目标实机检查 Live single-target move_joints check",
              "passed": False, "commands": [], "phases": {}, "pid": os.getpid(),
              "speed_percent": 10, "joint_step_rad": .01,
              "arrival_tolerance_rad": .003, "directory": str(directory)}
    command_target = None
    baseline = None
    baseline_tcp = None
    started = time.monotonic()

    with (directory / "feedback.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        def read(phase):
            # 七轴反馈沿用封装中的完整性及 0.5 秒新鲜度校验。
            # TCP 为实测关节经 FK/标定换算的估计值，不称为独立测量。
            q = arm.get_joint_positions()
            tcp = task_base @ ik.forward(q) @ flange_tcp
            status = arm.get_arm_status()
            row = {"phase": phase, "elapsed_s": time.monotonic() - started,
                   "joints": q.tolist(), "tcp": tcp.tolist(), "arm_status": status,
                   "target": None if command_target is None else command_target.tolist()}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (not arm.is_enabled() or status["arm_status"] != 0
                    or any(status.get("err_status", {}).values())):
                raise RuntimeError(f"设备反馈异常 Robot feedback fault: {status}")
            return row

        def sample(phase, target, duration):
            rows = []
            deadline = time.monotonic() + duration
            next_read = time.monotonic()
            while time.monotonic() < deadline:
                row = read(phase)
                rows.append(row)
                q = np.asarray(row["joints"])
                tcp = np.asarray(row["tcp"])
                # 检查只决定是否取消后续目标，不因偏差重抓关节作为新目标。
                # 此限制是主机反馈监督，不能保证物理停止距离。
                if np.max(np.abs(q - target)) > .015:
                    raise RuntimeError(f"关节偏差过大 Joint deviation exceeded: {phase}, q={q.tolist()}")
                if baseline_tcp is not None and tcp[2, 3] < baseline_tcp[2, 3] - .005:
                    raise RuntimeError(f"TCP 下移超过 5 mm TCP drop exceeded: {phase}")
                next_read += .05
                time.sleep(max(0., next_read - time.monotonic()))

            values = np.asarray([r["joints"] for r in rows])
            xyz = np.asarray([np.asarray(r["tcp"])[:3, 3] for r in rows])
            # 到位不能只用一次反馈判断；最后约 1 秒均需接近目标且变化很小。
            tail = values[-20:]
            settled_error = float(np.max(np.abs(tail - target)))
            settled_range = float(np.max(np.ptp(tail, axis=0)))
            report["phases"][phase] = {
                "samples": len(rows), "target": target.tolist(),
                "final_joints": values[-1].tolist(), "settled_max_error_rad": settled_error,
                "settled_joint_range_rad": settled_range,
                "tcp_z_min_m": float(xyz[:, 2].min()),
                "tcp_z_max_m": float(xyz[:, 2].max()),
                "final_tcp_xyz_m": xyz[-1].tolist()}
            print("阶段结果 Phase result:", phase, json.dumps(report["phases"][phase]), flush=True)
            if settled_error > .003 or settled_range > .002:
                raise RuntimeError(f"未确认稳定到位 Arrival not confirmed: {phase}")
            return rows[-1]

        def send(phase, target):
            nonlocal command_target
            # 这是低速单目标对照，不是把 move_joints 放进 20 Hz 发送循环。
            # 固件速度百分比不是 rad/s；本次安全范围来自有限的 0.01 rad 位移。
            arm.move_joints(target, speed_percent=10)
            error = arm._driver.get_comm_error()
            if error is not None:
                raise RuntimeError(f"CAN 发送失败 CAN send failed: {error}")
            command_target = target.copy()
            report["commands"].append({"phase": phase, "elapsed_s": time.monotonic() - started,
                                       "target": target.tolist()})
            print("单次目标 Target sent once:", phase, target.tolist(), flush=True)

        try:
            arm.connect()
            # connect 只启动接收线程，先等完整反馈，绝不自动补发 enable/reset。
            deadline = time.monotonic() + 3.
            while True:
                try:
                    initial = read("startup_feedback")
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
            baseline = np.asarray(initial["joints"])
            baseline_tcp = np.asarray(initial["tcp"])
            report["baseline_joints"] = baseline.tolist()
            report["baseline_tcp"] = baseline_tcp.tolist()
            report["initial_arm_status"] = initial["arm_status"]
            print("当前反馈 Baseline:", json.dumps(initial, ensure_ascii=False), flush=True)
            sample("read_only", baseline, 1.2)

            target = baseline.copy()
            target[6] += .01
            arm.validate_joint_command(target)
            # 在发命令前核对这次单关节变化的几何幅度；不执行回初始位姿或 IK。
            target_tcp = task_base @ ik.forward(target) @ flange_tcp
            if np.linalg.norm(target_tcp[:3, 3] - baseline_tcp[:3, 3]) > .005:
                raise ValueError("预期 TCP 位移超过 5 mm Planned TCP displacement exceeded")

            send("fixed_target", baseline)
            sample("fixed_target", baseline, 3.)
            send("J7_plus_0.01", target)
            sample("J7_plus_0.01", target, 4.)
            send("J7_return", baseline)
            sample("J7_return", baseline, 4.)
            report["passed"] = True
        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            print("检查未通过 Check failed:", report["error"], flush=True)
            # 不自动切到 JS 或重新使能，不因异常反复追随下移重抓目标。
            # 保留最后一个有限目标，等现场确认外部支撑后再结束连接。
            if command_target is not None:
                print("后续目标已取消 Further targets cancelled. "
                      "请在现场支撑机械臂，确认后输入 SUPPORTED 结束连接。", flush=True)
                while True:
                    try:
                        if input("External support confirmed: ").strip() == "SUPPORTED":
                            break
                    except KeyboardInterrupt:
                        print("等待外部支撑确认 Awaiting external support", flush=True)
        finally:
            report["duration_s"] = time.monotonic() - started
            (directory / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            arm.disconnect()  # 只断 CAN，不发送失能、急停或复位。
    print("检查结束 Check completed:", json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
