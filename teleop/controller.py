"""Single-arm Quest 3 teleoperation orchestration."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from robot_control.l20 import LinkerHandL20
from robot_control.nero import NeroArm
from robot_control.nero_ik import NeroIK, pose_error

from .coordinate_frames import (
    anydex_wrist_to_base,
    pose6_to_matrix,
    position_quaternion_to_matrix,
)
from .hand_retarget import L20Retargeter
from .keyboard import Keyboard
from .recording import EpisodeRecorder
from .quest_listener import QuestRelayClient
from .target_gate import TargetGate
from .wrist_tracker import WristTracker


NERO_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
START_MOVE_MAX_JOINT_DELTA_RAD = float(2.0 * np.pi)
START_POSE_FEEDBACK_TIMEOUT_S = 3.0
START_POSE_FEEDBACK_POLL_S = 0.05


def _quest_input(config: dict[str, Any]):
    """Construct the configured Quest input, direct or persistent relay."""
    quest = config["quest"]
    if quest.get("input_mode") == "relay":
        return QuestRelayClient(quest["relay_host"], quest["relay_port"])

    # Direct mode remains available for isolated debugging.  The persistent
    # listener owns this same authoritative AnyDex input in normal operation.
    project_root = Path(__file__).resolve().parents[1]
    anydex_root = project_root / "AnyDexRetarget"
    example_root = anydex_root / "example"
    for path in (anydex_root, example_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from input.quest3 import Quest3

    return Quest3(
        host=quest["listen_host"],
        port=quest["port"],
        protocol=quest["transport"],
    )


def _latest_frame(quest_input, config: dict[str, Any]):
    quest = config["quest"]
    frame = quest_input.get_hand_frame(quest["side"])
    if frame is None:
        return None
    if frame.pair_skew > float(quest["max_pair_skew_s"]):
        return None
    if time.monotonic() - frame.received_at > float(quest["max_age_s"]):
        return None
    return frame


def _retargeter(config: dict[str, Any]) -> L20Retargeter:
    hand = config["hand"]
    return L20Retargeter(hand["retarget_config"], config["quest"]["side"], hand["max_raw_step"])


def input_check(config: dict[str, Any]) -> None:
    """Validate the Quest stream and AnyDex result without opening CAN."""
    retargeter = _retargeter(config)
    quest_input = _quest_input(config)
    if config["quest"].get("input_mode") == "relay":
        print(
            "Reading Quest frames from persistent local relay "
            f"{config['quest']['relay_host']}:{config['quest']['relay_port']} (Ctrl+C to stop)"
        )
    else:
        print(f"Listening for Quest TCP on {config['quest']['listen_host']}:{config['quest']['port']} (Ctrl+C to stop)")
    seen_at = 0.0
    try:
        while True:
            frame = _latest_frame(quest_input, config)
            if frame is not None and frame.received_at != seen_at:
                seen_at = frame.received_at
                qpos, raw = retargeter.retarget(frame.landmarks)
                age_ms = (time.monotonic() - frame.received_at) * 1000.0
                print(
                    f"frame age={age_ms:.1f}ms skew={frame.pair_skew*1000:.1f}ms "
                    f"wrist={np.round(frame.wrist_position, 3)} qpos={qpos.shape} raw={raw.tolist()}"
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        quest_input.stop()


def _make_arm(config: dict[str, Any]) -> NeroArm:
    arm = config["arm"]
    return NeroArm(
        can_interface=arm["can_interface"],
        can_channel=arm["can_channel"],
        max_joint_delta=float(arm["max_joint_delta_rad"]),
    )


def _make_hand(config: dict[str, Any]) -> LinkerHandL20:
    hand = config["hand"]
    return LinkerHandL20(hand_type=hand["type"], can_channel=hand["can_channel"])


def _disconnect_devices(hand: LinkerHandL20 | None, arm: NeroArm | None) -> list[str]:
    """Attempt both disconnects; a hand cleanup failure must not skip the arm."""
    errors: list[str] = []
    for name, device in (("L20", hand), ("Nero", arm)):
        if device is None:
            continue
        try:
            device.disconnect()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return errors


def _read_complete_arm_joint_positions(arm: NeroArm) -> np.ndarray:
    """Wait briefly for all seven read-only Nero motor-state feedback frames."""
    deadline = time.monotonic() + START_POSE_FEEDBACK_TIMEOUT_S
    last_error: RuntimeError | None = None
    while True:
        try:
            return arm.get_joint_positions()
        except RuntimeError as exc:
            if not str(exc).startswith("Nero SDK feedback is unavailable"):
                raise
            last_error = exc
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Nero 未在 3.0 s 内收到完整的七轴关节反馈。"
            ) from last_error
        time.sleep(START_POSE_FEEDBACK_POLL_S)


def _print_ik_failure_diagnostics(
    ik: NeroIK,
    seed: np.ndarray,
    target: np.ndarray,
    previous_target: np.ndarray,
) -> None:
    """Print compact, actionable diagnostics after a failed IK solve."""
    diag = ik.last_diagnostics
    target_step_position, target_step_rotation = pose_error(previous_target, target)
    feedback_position, feedback_rotation = pose_error(ik.forward(seed), target)
    print(
        "IK diagnostics: "
        f"code={diag.get('code', 'IK_FAILED')} "
        f"solve_ms={float(diag.get('solve_ms', 0.0)):.1f} "
        f"budget_exhausted={bool(diag.get('budget_exhausted', False))} "
        f"target_step={target_step_position * 1000.0:.2f}mm/"
        f"{np.degrees(target_step_rotation):.2f}deg "
        f"target_from_feedback={feedback_position * 1000.0:.2f}mm/"
        f"{np.degrees(feedback_rotation):.2f}deg"
    )
    print(
        "IK seed deg J1..J7: "
        + np.array2string(np.degrees(seed), precision=2, suppress_small=True)
    )
    best = diag.get("best_intermediate")
    if best is not None:
        details = [
            f"residual_evals={diag.get('residual_evaluations', 'n/a')}",
            f"fk_position={best['fk_position_error_mm']:.3f}mm",
            f"fk_rotation={best['fk_rotation_error_deg']:.3f}deg",
        ]
        dq = np.asarray(best.get("dq_ik_deg", []), dtype=np.float64)
        if dq.shape == (7,) and np.all(np.isfinite(dq)):
            joint_index = int(np.argmax(np.abs(dq)))
            details.append(f"max_dq=J{joint_index + 1}:{dq[joint_index]:+.2f}deg")
            details.append(
                "dq=" + np.array2string(dq, precision=2, suppress_small=True)
            )
        print("IK best intermediate: " + " ".join(details))
    candidates = diag.get("candidates", [])
    if not candidates:
        print("IK candidates: none completed within the solve budget.")
        return
    for index, candidate in enumerate(candidates):
        details = [
            f"code={candidate.get('code', 'unknown')}",
            f"nfev={candidate.get('nfev', 'n/a')}",
        ]
        if "fk_position_error_mm" in candidate:
            details.append(f"fk_position={candidate['fk_position_error_mm']:.3f}mm")
        if "fk_rotation_error_deg" in candidate:
            details.append(f"fk_rotation={candidate['fk_rotation_error_deg']:.3f}deg")
        dq = np.asarray(candidate.get("dq_ik_deg", []), dtype=np.float64)
        if dq.shape == (7,) and np.all(np.isfinite(dq)):
            joint_index = int(np.argmax(np.abs(dq)))
            details.append(f"max_dq=J{joint_index + 1}:{dq[joint_index]:+.2f}deg")
            details.append(
                "dq=" + np.array2string(dq, precision=2, suppress_small=True)
            )
        if "error" in candidate:
            details.append(f"error={candidate['error']}")
        print(f"IK candidate {index}: " + " ".join(details))


def record_start_pose(
    config: dict[str, Any], output: str | None, *, overwrite: bool = False
) -> Path:
    """Read the current Nero joints and save them without enabling or moving."""
    output_path = Path(output).expanduser().resolve() if output else Path(
        config["arm"]["start_pose_file"]
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Start pose already exists: {output_path}. Use --overwrite to replace it."
        )

    arm = _make_arm(config)
    try:
        arm.connect()
        joints = _read_complete_arm_joint_positions(arm)
        print("Nero start pose (rad):", joints.tolist())
        print("Nero start pose (deg):", np.degrees(joints).tolist())
    finally:
        errors = _disconnect_devices(None, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "robot": "nero",
        "joint_names": list(NERO_JOINT_NAMES),
        "joint_positions_rad": [float(value) for value in joints],
    }
    output_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    print(f"Nero start pose saved: {output_path}")
    return output_path


def _load_start_pose(config: dict[str, Any]) -> np.ndarray:
    path = Path(config["arm"]["start_pose_file"])
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("robot") != "nero":
        raise ValueError(f"Invalid Nero start pose file: {path}")
    if tuple(document.get("joint_names", ())) != NERO_JOINT_NAMES:
        raise ValueError(
            "Nero start pose joint_names must be joint1 through joint7 in order."
        )
    try:
        joints = np.asarray(document["joint_positions_rad"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Nero start pose must contain seven numeric joint angles.") from exc
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("Nero start pose must contain seven finite joint angles.")
    return joints


def _move_arm_to_start_pose(arm: NeroArm, config: dict[str, Any]) -> None:
    """Move to the recorded joint pose and wait for feedback convergence."""
    arm_config = config["arm"]
    target = _load_start_pose(config)
    tolerance = float(arm_config["start_tolerance_rad"])
    timeout = float(arm_config["start_timeout_s"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("arm.start_tolerance_rad must be finite and positive.")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("arm.start_timeout_s must be finite and positive.")

    target = arm.validate_joint_command(
        target, max_joint_delta=START_MOVE_MAX_JOINT_DELTA_RAD
    )
    current = arm.get_joint_positions()
    error = float(np.max(np.abs(target - current)))
    print("Nero recorded start pose (rad):", target.tolist())
    if error <= tolerance:
        print(f"Nero is already at the recorded start pose; max error={error:.6f} rad.")
        return

    arm.move_joints(
        target,
        speed_percent=arm_config["start_speed_percent"],
        max_joint_delta=START_MOVE_MAX_JOINT_DELTA_RAD,
    )
    deadline = time.monotonic() + timeout
    while True:
        current = arm.get_joint_positions()
        error = float(np.max(np.abs(target - current)))
        if error <= tolerance:
            print(f"Nero reached the recorded start pose; max error={error:.6f} rad.")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Nero start-pose move timed out; max joint error={error:.6f} rad."
            )
        time.sleep(0.05)


def preflight(config: dict[str, Any], control: str) -> None:
    """Read feedback only. This function never enables or commands a device."""
    arm = _make_arm(config) if control in ("arm", "both") else None
    hand = _make_hand(config) if control in ("hand", "both") else None
    try:
        if arm is not None:
            arm.connect()
            arm.enable_normal_mode()
            joints = arm.get_joint_positions()
            flange = pose6_to_matrix(arm.get_flange_pose())
            predicted = NeroIK().forward(joints)
            position_error, rotation_error = pose_error(predicted, flange)
            print(
                f"Nero feedback OK; enabled={arm.is_enabled()}; "
                f"URDF/FK error={position_error*1000:.2f}mm/{np.degrees(rotation_error):.2f}deg"
            )
            safety = config["safety"]
            if position_error > safety["startup_fk_position_error_m"] or np.degrees(rotation_error) > safety["startup_fk_rotation_error_deg"]:
                raise RuntimeError("Nero URDF/FK check exceeded the configured startup tolerance.")
        if hand is not None:
            hand.connect()
            raw = hand.get_joint_positions_raw(fresh=True)
            print(f"L20 feedback OK; raw={raw.tolist()}")
    finally:
        errors = _disconnect_devices(hand, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))


def enable_arm(config: dict[str, Any]) -> None:
    """Enable Nero without sending any joint or Cartesian target."""
    arm = _make_arm(config)
    try:
        arm.connect()
        arm.enable_normal_mode()
        print("Nero joint position before enable (rad):", arm.get_joint_positions())
        arm.enable()
        print("Nero enabled:", arm.is_enabled())
    finally:
        errors = _disconnect_devices(None, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))
        print("Nero disconnect attempted; no motion target or automatic disable was sent.")


def reset_arm(config: dict[str, Any]) -> None:
    """Disable Nero and request a control-state reset without motion targets.

    Disabling can cause the arm to fall.  The caller must ensure the arm is
    mechanically supported and the workcell is clear before invoking this.
    """
    arm = _make_arm(config)
    try:
        arm.connect()
        enabled_before = arm.is_enabled()
        print("Nero enabled before reset:", enabled_before)
        if enabled_before:
            arm.disable()
        if arm.is_enabled():
            raise RuntimeError("Nero remains enabled after disable request; reset was not sent.")
        arm.reset()
        print("Nero reset requested; enabled:", arm.is_enabled())
    finally:
        errors = _disconnect_devices(None, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))
        print("Nero disconnect attempted after disable/reset; no motion target was sent.")


def run(config: dict[str, Any], control: str, record_dir: str | None) -> None:
    arm = _make_arm(config) if control in ("arm", "both") else None
    hand = _make_hand(config) if control in ("hand", "both") else None
    ik = NeroIK() if arm is not None else None
    hand_retargeter = _retargeter(config) if hand is not None else None
    quest_input = None
    calibration = np.asarray(config["calibration"]["R_base_quest"], dtype=np.float64)
    tracking = config["tracking"]
    safety = config["safety"]
    gate = TargetGate(
        safety["max_position_step_m"],
        safety["max_rotation_step_deg"],
        safety["tcp_workspace_cube_side_m"],
    )
    flange_to_tcp: np.ndarray | None = None
    tracker: WristTracker | None = None
    last_accepted_target: np.ndarray | None = None
    paused = True
    recorder = EpisodeRecorder(
        record_dir or config["recording"]["output_dir"],
        {"config": config["_path"], "control": control, "quest_side": config["quest"]["side"]},
    )

    def reanchor(frame) -> None:
        nonlocal tracker, last_accepted_target, paused
        if arm is not None:
            # Keep the teleoperation target in the same URDF model used by IK.
            # SDK flange feedback remains an independent preflight observation.
            flange = ik.forward(arm.get_joint_positions())
            wrist_position, wrist_quaternion = anydex_wrist_to_base(
                frame.wrist_position, frame.wrist_quat, calibration
            )
            flange_quaternion = Rotation.from_matrix(flange[:3, :3]).as_quat()
            tracker = WristTracker(
                flange[:3, 3], flange_quaternion,
                position_scale=tracking["position_scale"],
                ema_alpha=tracking["ema_alpha"],
                negate_rot_xy=tracking["negate_rot_xy"],
                position_deadband=tracking["position_deadband_m"],
                rotation_deadband_deg=tracking["rotation_deadband_deg"],
            )
            tracker.update(wrist_position, wrist_quaternion)
            gate.reanchor(flange)
            last_accepted_target = flange.copy()
        if hand is not None:
            hand_retargeter.reset(hand.get_joint_positions_raw(fresh=True))
        paused = False
        print("Re-anchored; command streaming resumed.")

    try:
        quest_input = _quest_input(config)
        if isinstance(quest_input, QuestRelayClient) and not quest_input.is_available():
            raise RuntimeError(
                "Persistent Quest listener is unavailable. Start `teleop.cli ... "
                "quest-listener` before running teleoperation."
            )
        if arm is not None:
            arm.connect()
            arm.enable_normal_mode()
            if not arm.is_enabled():
                raise RuntimeError("Nero is not enabled. Enable it using the approved site procedure before run.")
            _move_arm_to_start_pose(arm, config)
            start_joints = arm.get_joint_positions()
            model_start_flange = ik.forward(start_joints)
            sdk_start_flange = pose6_to_matrix(arm.get_flange_pose())
            sdk_start_tcp = pose6_to_matrix(arm.get_tcp_pose())
            flange_to_tcp = np.linalg.inv(sdk_start_flange) @ sdk_start_tcp
            model_start_tcp = model_start_flange @ flange_to_tcp
            gate.set_tcp_workspace_center(model_start_tcp[:3, 3])
            half_side = gate.tcp_workspace_cube_side_m / 2.0
            print(
                "TCP workspace cube centered at model-consistent startup TCP "
                "(base frame): "
                f"center={model_start_tcp[:3, 3].tolist()}, half-side={half_side:.4f} m."
            )
        if hand is not None:
            hand.connect()
            hand.set_speed(config["hand"]["speed"])
        period = 1.0 / float(config["control_hz"])
        if arm is not None:
            print("Start pose ready. Hold a comfortable Quest pose, then press R to anchor.")
        print("Keys: R=re-anchor, B=begin recording, S=save recording, Q=quit")
        with Keyboard() as keyboard:
            while True:
                loop_started = time.monotonic()
                key = keyboard.read()
                if key == "q":
                    break
                if key == "b":
                    recorder.start()
                    print("Recording started.")
                elif key == "s":
                    saved = recorder.stop_and_save()
                    print(f"Recording saved: {saved}" if saved else "No active/non-empty recording.")

                frame = _latest_frame(quest_input, config)
                if frame is None:
                    if not paused:
                        paused = True
                        gate.pause("Quest frame timeout")
                        print("PAUSED: Quest frame timeout; stale targets are not sent. Press R after recovery.")
                    time.sleep(min(period, 0.02))
                    continue
                if key == "r":
                    reanchor(frame)
                    continue
                if paused:
                    time.sleep(min(period, 0.02))
                    continue

                wrist_position, wrist_quaternion = anydex_wrist_to_base(
                    frame.wrist_position, frame.wrist_quat, calibration
                )
                landmarks = frame.landmarks
                arm_action = np.full(7, np.nan)
                hand_action = np.full(20, -1, dtype=np.int64)
                target_transform = np.full((4, 4), np.nan)
                if arm is not None:
                    if flange_to_tcp is None:
                        raise RuntimeError("TCP workspace transform is not initialized.")
                    target_position, target_quaternion = tracker.update(wrist_position, wrist_quaternion)
                    target_transform = position_quaternion_to_matrix(target_position, target_quaternion)
                    target_tcp_position = (target_transform @ flange_to_tcp)[:3, 3]
                    seed = arm.get_joint_positions()
                    solved = ik.solve(target_transform, seed)
                    if solved is None:
                        paused = True
                        reason = ik.last_diagnostics.get("code", "IK failed")
                        print(f"PAUSED: {reason}; press R to re-anchor.")
                        if last_accepted_target is None:
                            raise RuntimeError("IK diagnostic anchor is not initialized.")
                        _print_ik_failure_diagnostics(
                            ik, seed, target_transform, last_accepted_target
                        )
                        continue
                    if not gate.accept(target_transform, target_tcp_position):
                        paused = True
                        print(f"PAUSED: {gate.reason}; press R to re-anchor.")
                        continue
                    arm_action = solved
                    last_accepted_target = target_transform.copy()
                if hand is not None:
                    _, hand_action = hand_retargeter.retarget(landmarks)

                arm_sent_at = np.nan
                hand_sent_at = np.nan
                if arm is not None:
                    arm.move_js(arm_action)
                    arm_sent_at = time.monotonic()
                if hand is not None:
                    hand.set_joint_positions_raw(hand_action)
                    hand_sent_at = time.monotonic()

                arm_joints = arm.get_joint_positions() if arm is not None else np.full(7, np.nan)
                arm_torques = arm.get_joint_torques() if arm is not None else np.full(7, np.nan)
                arm_tcp = arm.get_tcp_pose() if arm is not None else np.full(6, np.nan)
                hand_raw = hand.get_joint_positions_raw(fresh=False) if hand is not None else np.full(20, -1)
                recorder.append(
                    timestamp_monotonic=time.monotonic(),
                    quest_wrist=np.concatenate((frame.wrist_position, frame.wrist_quat)),
                    quest_landmarks=frame.landmarks,
                    quest_received_at=frame.received_at,
                    target_base_flange=target_transform,
                    arm_action=arm_action,
                    hand_action_raw=hand_action,
                    arm_joint_position=arm_joints,
                    arm_joint_torque=arm_torques,
                    arm_tcp_pose=arm_tcp,
                    hand_position_raw=hand_raw,
                    arm_sent_at=arm_sent_at,
                    hand_sent_at=hand_sent_at,
                )
                remaining = period - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        try:
            saved = recorder.stop_and_save()
            if saved:
                print(f"Active recording saved during shutdown: {saved}")
        except Exception as exc:
            print(f"Recording save failed during shutdown: {type(exc).__name__}: {exc}")
        if quest_input is not None:
            quest_input.stop()
        errors = _disconnect_devices(hand, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))
        print("Command streaming stopped; disconnect was attempted without automatic disable/e-stop.")
