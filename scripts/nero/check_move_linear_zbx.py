"""厂家 move_l 与 IK+move_j 笛卡尔移动可靠性对比检查。

直接连接 NeroArm，不使用 ControlProcess 子进程。
先 PTP 回 nero_start_pose，沿 task +Z 上升 4 cm 作为测试原点，
然后在 task X 或 Y 方向来回移动并验证到位精度。

信号约定：
  Ctrl+C   优雅停止，完成当前移动后回原点退出；二次 Ctrl+C 电急停。
  Ctrl+\   立即电急停。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_agent.config import load_config
from cloth_agent.geometry import tcp_to_flange, transform
from cloth_agent.paths import DEFAULT_CONFIG
from cloth_agent.planning import Planner
from robot_control.nero import NeroArm
from robot_control.nero_ik import NeroIK, pose_error

POSITION_TOL_M = 0.02
ANGLE_TOL_RAD = np.deg2rad(2)
JOINT_TOL_RAD = 0.02
STABLE_WINDOW_S = 0.2
ARRIVAL_TIMEOUT_S = 30.0

# ---------------------------------------------------------------------------
# 全局状态（信号处理器需要访问）
# ---------------------------------------------------------------------------
_arm: NeroArm | None = None
_graceful_stop = False


def _handle_sigint(sig, frame):
    global _graceful_stop
    if _graceful_stop:
        if _arm is not None:
            _arm.emergency_stop()
        print("\n二次 Ctrl+C — 电急停已发送！", flush=True)
        sys.exit(1)
    _graceful_stop = True
    print("\n收到 Ctrl+C，将在本轮结束后回原点退出...", flush=True)


def _handle_sigquit(sig, frame):
    if _arm is not None:
        _arm.emergency_stop()
    print("\nCtrl+\\ — 电急停已发送！", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _read_arm_status(arm: NeroArm) -> dict:
    msg = arm.get_raw_arm_status()
    received_at = float(msg.timestamp)
    return {name: int(getattr(msg.msg, name)) for name in
            ("ctrl_mode", "arm_status", "mode_feedback", "motion_status")} | {
                "received_at_unix": received_at}


def _tcp_from_joints(q, ik, task_base, flange_tcp):
    return task_base @ ik.forward(q) @ flange_tcp


def _flange_pose_from_tcp(target_tcp, task_from_base, flange_from_tcp):
    """将 task-frame TCP 4x4 变换矩阵转换为厂家 move_l 使用的 6D flange pose。"""
    flange = tcp_to_flange(target_tcp, task_from_base, flange_from_tcp)
    return np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global _arm, _graceful_stop

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("native_l", "ik_j"), required=True,
                        help="native_l: 厂家 move_l；ik_j: IK + move_j 逐点执行")
    parser.add_argument("--axis", choices=("x", "y"), default="x",
                        help="task 空间来回移动的轴方向 (default: x)")
    parser.add_argument("--distance", type=float, default=0.20,
                        help="单方向移动距离 m (default: 0.20)")
    parser.add_argument("--speed-percent", type=int, default=10,
                        help="SDK 速度百分比 [1,100] (default: 10)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", 
                        help="记录输出目录，不可已存在")
    args = parser.parse_args()

    if args.distance <= 0 or args.distance > 0.5:
        parser.error("--distance 必须在 (0, 0.5] m 范围内")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent 必须在 [1, 100] 范围内")

    config = load_config(args.config)
    settings = config["arms"]["right"]
    task_from_base = settings["task_from_base"]
    flange_from_tcp = settings["flange_from_tcp"]
    task_base = transform(task_from_base)
    flange_tcp = transform(flange_from_tcp)
    speed = args.speed_percent
    axis_index = 0 if args.axis == "x" else 1

    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    ik = NeroIK()
    planner = Planner(config) if args.mode == "ik_j" else None
    arm = NeroArm(can_channel=settings["can_channel"])
    _arm = arm

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGQUIT, _handle_sigquit)

    recorded = np.asarray(settings["start_joint_positions"])
    report = {
        "description": f"笛卡尔移动可靠性检查 mode={args.mode} axis={args.axis}",
        "mode": args.mode, "axis": args.axis, "distance_m": args.distance,
        "speed_percent": speed, "passed": False, "trials": [], "pid": os.getpid(),
    }
    started = time.monotonic()
    command_sent = False

    with (directory / "feedback.jsonl").open("w", encoding="utf-8", buffering=1) as log:

        # -- 反馈读取 -------------------------------------------------------
        def read(phase: str) -> dict:
            q = arm.get_joint_positions()
            tcp = _tcp_from_joints(q, ik, task_base, flange_tcp)
            status = _read_arm_status(arm)
            row = {"phase": phase, "t": time.monotonic() - started,
                   "joints": q.tolist(), "tcp": tcp.tolist(), "arm_status": status}
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            return row

        # -- 到位等待（TCP 精度） -------------------------------------------
        def wait_tcp_arrived(target_tcp, phase, timeout=ARRIVAL_TIMEOUT_S):
            """轮询关节反馈 → FK → pose_error，直到到位+稳定或超时。"""
            deadline = time.monotonic() + timeout
            stable_since = None
            prev_q = None
            while True:
                row = read(phase)
                q = np.asarray(row["joints"])
                actual_tcp = np.asarray(row["tcp"])
                p_err, a_err = pose_error(actual_tcp, target_tcp)
                stable = (prev_q is not None
                          and np.max(np.abs(q - prev_q)) <= 0.002)
                prev_q = q.copy()
                if p_err <= POSITION_TOL_M and a_err <= ANGLE_TOL_RAD and stable:
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= STABLE_WINDOW_S:
                        return p_err, a_err
                else:
                    stable_since = None
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"到位超时 {phase}: pos={p_err*1000:.1f}mm "
                        f"ang={np.rad2deg(a_err):.2f}deg")
                time.sleep(0.05)

        # -- 到位等待（关节精度，用于 ik_j 中间点） -------------------------
        def wait_joint_arrived(target_q, phase, timeout=ARRIVAL_TIMEOUT_S):
            """轮询关节反馈，直到关节误差+固件状态均确认到位。"""
            deadline = time.monotonic() + timeout
            sent_wall = time.time()
            while True:
                row = read(phase)
                q = np.asarray(row["joints"])
                status = row["arm_status"]
                error = float(np.max(np.abs(q - target_q)))
                firmware_ok = (status["received_at_unix"] > sent_wall
                               and status["mode_feedback"] == 1
                               and status["motion_status"] == 0)
                if error <= JOINT_TOL_RAD and firmware_ok:
                    return error
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"关节到位超时 {phase}: error={error:.4f}rad "
                        f"firmware={status}")
                time.sleep(0.05)

        # -- 移动：native_l -------------------------------------------------
        def move_native_l(target_tcp, phase):
            nonlocal command_sent
            pose = _flange_pose_from_tcp(target_tcp, task_from_base, flange_from_tcp)
            arm.move_linear(pose, speed_percent=speed)
            command_sent = True
            p_err, a_err = wait_tcp_arrived(target_tcp, phase)
            return p_err, a_err

        # -- 移动：ik_j -----------------------------------------------------
        def move_ik_j(target_tcp, phase, seed_q):
            nonlocal command_sent
            plan = planner.plan("right", seed_q, waypoints=[target_tcp],
                                mode="cartesian")
            waypoints = plan.joint_waypoints
            q = seed_q
            for i, wq in enumerate(waypoints[1:], 1):
                label = f"{phase} wp{i}/{len(waypoints)-1}"
                arm.move_joints(wq, speed_percent=speed,
                                max_joint_delta=config["motion"]["max_ik_step_rad"])
                command_sent = True
                wait_joint_arrived(wq, label)
                q = wq
            p_err, a_err = wait_tcp_arrived(target_tcp, phase)
            return p_err, a_err, q

        # -- PTP（关节空间，回 nero_start_pose / 上升原点前先用 J 到位） ----
        def ptp(target_q, phase):
            nonlocal command_sent
            arm.move_joints(target_q, speed_percent=speed,
                            max_joint_delta=2 * np.pi)
            command_sent = True
            error = wait_joint_arrived(target_q, phase)
            print(f"PTP {phase}: joint_error={error:.4f}rad", flush=True)

        # -- 通用 move_to ---------------------------------------------------
        def move_to(target_tcp, phase, seed_q):
            """根据 mode 选择 native_l 或 ik_j。返回 (p_err, a_err, current_q)。"""
            if args.mode == "native_l":
                p_err, a_err = move_native_l(target_tcp, phase)
                q = arm.get_joint_positions()
                return p_err, a_err, q
            else:
                return move_ik_j(target_tcp, phase, seed_q)

        # == 主执行 =========================================================
        try:
            arm.connect()
            deadline = time.monotonic() + 3.0
            while True:
                try:
                    read("startup")
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            print("反馈就绪", flush=True)

            # 1) PTP 回 nero_start_pose
            ptp(recorded, "PTP→nero_start_pose")

            # 2) 当前 TCP 上升 4cm → 测试原点
            q = arm.get_joint_positions()
            current_tcp = _tcp_from_joints(q, ik, task_base, flange_tcp)
            origin_tcp = current_tcp.copy()
            origin_tcp[2, 3] += 0.04
            p, a, q = move_to(origin_tcp, "上升4cm→测试原点", q)
            print(f"测试原点就位: pos_err={p*1000:.2f}mm ang_err={np.rad2deg(a):.2f}deg",
                  flush=True)

            # 保存原点的旋转（整个测试固定朝向）
            rotation = origin_tcp[:3, :3].copy()
            origin_xyz = origin_tcp[:3, 3].copy()

            def make_tcp(offset_m):
                """在原点基础上沿指定轴偏移，保持朝向。"""
                tcp = np.eye(4)
                tcp[:3, :3] = rotation
                tcp[:3, 3] = origin_xyz.copy()
                tcp[axis_index, 3] += offset_m
                return tcp

            # 3) 来回移动循环
            trial_num = 0
            while not _graceful_stop:
                trial_num += 1
                dist = args.distance

                # +distance
                target_plus = make_tcp(+dist)
                p, a, q = move_to(target_plus, f"T{trial_num}→+{dist}m", q)
                trial = {"trial": trial_num, "leg": f"+{args.axis}",
                         "pos_err_mm": p * 1000, "ang_err_deg": np.rad2deg(a),
                         "passed": p <= POSITION_TOL_M and a <= ANGLE_TOL_RAD}
                report["trials"].append(trial)
                print(f"Trial {trial_num} +{args.axis}: "
                      f"pos={p*1000:.2f}mm ang={np.rad2deg(a):.2f}deg "
                      f"{'PASS' if trial['passed'] else 'FAIL'}", flush=True)
                if not trial["passed"]:
                    raise RuntimeError(f"Trial {trial_num} +{args.axis} 精度不合格")

                if _graceful_stop:
                    break

                # -distance
                target_minus = make_tcp(-dist)
                p, a, q = move_to(target_minus, f"T{trial_num}→-{dist}m", q)
                trial = {"trial": trial_num, "leg": f"-{args.axis}",
                         "pos_err_mm": p * 1000, "ang_err_deg": np.rad2deg(a),
                         "passed": p <= POSITION_TOL_M and a <= ANGLE_TOL_RAD}
                report["trials"].append(trial)
                print(f"Trial {trial_num} -{args.axis}: "
                      f"pos={p*1000:.2f}mm ang={np.rad2deg(a):.2f}deg "
                      f"{'PASS' if trial['passed'] else 'FAIL'}", flush=True)
                if not trial["passed"]:
                    raise RuntimeError(f"Trial {trial_num} -{args.axis} 精度不合格")

                if _graceful_stop:
                    break

                # 回原点
                p, a, q = move_to(origin_tcp, f"T{trial_num}→origin", q)
                print(f"Trial {trial_num} 回原点: "
                      f"pos={p*1000:.2f}mm ang={np.rad2deg(a):.2f}deg", flush=True)

            # 优雅停止：回原点
            if _graceful_stop:
                print("优雅停止：回原点...", flush=True)
                move_to(origin_tcp, "graceful→origin", q)

            report["passed"] = True
            report["total_trials"] = trial_num

        except (Exception, KeyboardInterrupt) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(f"检查失败: {report['error']}", flush=True)
            if command_sent:
                print("已发命令，请确认外部支撑后输入 SUPPORTED 结束连接。",
                      flush=True)
                while True:
                    try:
                        if input("External support confirmed: ").strip() == "SUPPORTED":
                            break
                    except (KeyboardInterrupt, EOFError):
                        print("等待外部支撑确认", flush=True)
        finally:
            report["duration_s"] = time.monotonic() - started
            (directory / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            arm.disconnect()

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
