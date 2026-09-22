"""Single-arm Quest 3 teleoperation orchestration."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

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
from .target_gate import TargetGate
from .wrist_tracker import WristTracker


def _quest_input(config: dict[str, Any]):
    """Construct the authoritative AnyDex Quest3 plugin."""
    project_root = Path(__file__).resolve().parents[1]
    anydex_root = project_root / "AnyDexRetarget"
    example_root = anydex_root / "example"
    for path in (anydex_root, example_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from input.quest3 import Quest3

    quest = config["quest"]
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


def preflight(config: dict[str, Any], control: str) -> None:
    """Read feedback only. This function never enables or commands a device."""
    arm = _make_arm(config) if control in ("arm", "both") else None
    hand = _make_hand(config) if control in ("hand", "both") else None
    try:
        if arm is not None:
            arm.connect()
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
        print("Nero joint position before enable (rad):", arm.get_joint_positions())
        arm.enable()
        print("Nero enabled:", arm.is_enabled())
    finally:
        errors = _disconnect_devices(None, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))
        print("Nero disconnect attempted; no motion target or automatic disable was sent.")


def run(config: dict[str, Any], control: str, record_dir: str | None) -> None:
    arm = _make_arm(config) if control in ("arm", "both") else None
    hand = _make_hand(config) if control in ("hand", "both") else None
    ik = NeroIK() if arm is not None else None
    hand_retargeter = _retargeter(config) if hand is not None else None
    quest_input = None
    calibration = np.asarray(config["calibration"]["R_base_quest"], dtype=np.float64)
    tracking = config["tracking"]
    safety = config["safety"]
    gate = TargetGate(safety["max_position_step_m"], safety["max_rotation_step_deg"])
    tracker: WristTracker | None = None
    paused = True
    recorder = EpisodeRecorder(
        record_dir or config["recording"]["output_dir"],
        {"config": config["_path"], "control": control, "quest_side": config["quest"]["side"]},
    )

    def reanchor(frame) -> None:
        nonlocal tracker, paused
        if arm is not None:
            flange = pose6_to_matrix(arm.get_flange_pose())
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
        if hand is not None:
            hand_retargeter.reset(hand.get_joint_positions_raw(fresh=True))
        paused = False
        print("Re-anchored; command streaming resumed.")

    try:
        quest_input = _quest_input(config)
        if arm is not None:
            arm.connect()
            if not arm.is_enabled():
                raise RuntimeError("Nero is not enabled. Enable it using the approved site procedure before run.")
        if hand is not None:
            hand.connect()
            hand.set_speed(config["hand"]["speed"])
        period = 1.0 / float(config["control_hz"])
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
                    target_position, target_quaternion = tracker.update(wrist_position, wrist_quaternion)
                    target_transform = position_quaternion_to_matrix(target_position, target_quaternion)
                    seed = arm.get_joint_positions()
                    solved = ik.solve(target_transform, seed)
                    if solved is None or not gate.accept(target_transform):
                        paused = True
                        reason = gate.reason if gate.paused else ik.last_diagnostics.get("code", "IK failed")
                        print(f"PAUSED: {reason}; press R to re-anchor.")
                        continue
                    arm_action = solved
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
