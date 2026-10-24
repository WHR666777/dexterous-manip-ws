"""Single-arm VR teleoperation orchestration."""

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
from .config import calibration_rotation, input_config
from .compute import ComputeProcess
from .hand_feedback import HandFeedbackPoller
from .hand_replay import HandReplay
from .hand_retarget import HAND_TIMING_FIELDS, L20Retargeter
from .keyboard import Keyboard
from .recording import EpisodeRecorder
from .realsense_camera import RealSenseCamera
from .quest_listener import QuestRelayClient
from .target_gate import TargetGate
from .wrist_tracker import WristTracker


NERO_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
START_MOVE_MAX_JOINT_DELTA_RAD = float(2.0 * np.pi)
START_POSE_FEEDBACK_TIMEOUT_S = 3.0
START_POSE_FEEDBACK_POLL_S = 0.05
STARTUP_POSE_FEEDBACK_TIMEOUT_S = 3.0
STARTUP_POSE_FEEDBACK_POLL_S = 0.05


class _ControlTiming:
    """Summarize completed control cycles without including pauses/re-anchoring."""

    FIELDS = (
        "timing_ik_s", "timing_hand_retarget_s", "timing_arm_send_s",
        "timing_hand_send_s", "timing_seed_feedback_s", "timing_feedback_s",
        "timing_hand_send_wait_s",
        "timing_compute_ipc_s", "timing_ik_cpu_s", "timing_hand_retarget_cpu_s",
    )

    def __init__(self, control_hz: float, label: str = "") -> None:
        self.period = 1.0 / control_hz
        self.label = f"{label} " if label else ""
        self.rows: list[tuple[float, ...]] = []
        self.hand_rows: list[tuple[float, ...]] = []
        self.elapsed = 0.0

    def append(
        self, stages: dict[str, float], record_s: float, work_s: float, cycle_s: float,
        hand_diagnostics: dict[str, float] | None = None,
    ) -> bool:
        self.rows.append(tuple(stages[key] for key in self.FIELDS) + (record_s, work_s, cycle_s))
        if hand_diagnostics and hand_diagnostics.get('hand_retarget_numevals', -1) >= 0:
            self.hand_rows.append(tuple(hand_diagnostics.get(key, np.nan)
                                        for key in HAND_TIMING_FIELDS)
                                  + (hand_diagnostics['hand_retarget_numevals'],))
        self.elapsed += cycle_s
        if self.elapsed >= 2.0:
            self.report()
            return True
        return False

    def report(self) -> None:
        if not self.rows:
            return
        values = np.asarray(self.rows)
        names = tuple(key.removeprefix("timing_").removesuffix("_s") for key in self.FIELDS)
        names += ("record", "work", "cycle")
        mean_ms = values.mean(axis=0) * 1000.0
        p95_ms = np.percentile(values, 95, axis=0) * 1000.0
        max_ms = values.max(axis=0) * 1000.0
        stages = " ".join(
            f"{name}={mean:.1f}/{p95:.1f}/{maximum:.1f}"
            for name, mean, p95, maximum in zip(names, mean_ms, p95_ms, max_ms)
        )
        overruns = int(np.count_nonzero(values[:, -2] > self.period))
        print(
            f"[{self.label}control timing] cycles={len(values)} active_hz={len(values) / self.elapsed:.2f} "
            f"target_hz={1.0 / self.period:.1f} overruns={overruns}/{len(values)} "
            f"ms(mean/p95/max): {stages}"
        )
        if self.hand_rows:
            hand_values = np.asarray(self.hand_rows)
            evals = hand_values[:, -1]
            details = []
            for column, key in zip(hand_values[:, :-1].T, HAND_TIMING_FIELDS):
                ms = column[np.isfinite(column)] * 1000
                if len(ms):
                    name = key.removeprefix('timing_hand_').removesuffix('_s')
                    details.append(f'{name}={ms.mean():.1f}/{np.percentile(ms, 95):.1f}/{ms.max():.1f}')
            print(f'[{self.label}hand retarget] cycles={len(evals)} '
                  f'numevals(mean/p95/max)={evals.mean():.1f}/'
                  f'{np.percentile(evals, 95):.1f}/{evals.max():.0f} '
                  f'ms(mean/p95/max): {" ".join(details)}')
        self.rows.clear()
        self.hand_rows.clear()
        self.elapsed = 0.0


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


def _input_device(config: dict[str, Any]):
    """Select the AnyDex input; both expose the same complete-frame interface."""
    if config.get("input_device", "quest3") == "quest3":
        return _quest_input(config)
    project_root = Path(__file__).resolve().parents[1]
    example_root = project_root / "AnyDexRetarget" / "example"
    if str(example_root) not in sys.path:
        sys.path.insert(0, str(example_root))
    from input.pico4 import Pico4

    source = input_config(config)
    return Pico4(
        mode=source.get("input_mode", "direct"),
        relay_host=source.get("relay_host", "127.0.0.1"),
        relay_port=source.get("relay_port", 63902),
        port=source["port"],
        broadcast_port=source["broadcast_port"],
    )


def _latest_frame(quest_input, config: dict[str, Any]):
    quest = input_config(config)
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
    return L20Retargeter(hand["retarget_config"], input_config(config)["side"], hand["max_raw_step"])


def _make_compute(config, control, hand_mode="tracking"):
    return ComputeProcess(config, control, hand_mode)


def input_check(config: dict[str, Any]) -> None:
    """Validate the configured VR stream and AnyDex result without opening CAN."""
    retargeter = _retargeter(config)
    quest_input = _input_device(config)
    source = input_config(config)
    device_name = config.get("input_device", "quest3")
    if source.get("input_mode") == "relay":
        print(
            f"Reading {device_name} frames from persistent local relay "
            f"{source['relay_host']}:{source['relay_port']} (Ctrl+C to stop)"
        )
    else:
        print(f"Listening for {device_name} TCP on port {source['port']} (Ctrl+C to stop)")
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


def _read_complete_arm_startup_poses(arm: NeroArm) -> tuple[np.ndarray, np.ndarray]:
    """Wait for SDK flange/TCP feedback before establishing the TCP workspace."""
    deadline = time.monotonic() + STARTUP_POSE_FEEDBACK_TIMEOUT_S
    waiting = False
    while True:
        stage = "flange"
        try:
            flange = arm.get_flange_pose()
            stage = "TCP"
            tcp = arm.get_tcp_pose()
            return flange, tcp
        except RuntimeError as exc:
            if str(exc) not in (
                "Nero SDK feedback is unavailable: None",
                "Nero SDK pose feedback is incomplete.",
            ):
                raise
            if not waiting:
                print(
                    "Waiting for complete Nero flange/TCP feedback "
                    f"(up to {STARTUP_POSE_FEEDBACK_TIMEOUT_S:.1f} s)."
                )
                waiting = True
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Nero flange/TCP feedback did not become available within "
                    f"{STARTUP_POSE_FEEDBACK_TIMEOUT_S:.1f} s; "
                    f"waiting for {stage}: {exc}"
                ) from exc
        time.sleep(STARTUP_POSE_FEEDBACK_POLL_S)


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
            joints = _read_complete_arm_joint_positions(arm)
            flange_pose, _ = _read_complete_arm_startup_poses(arm)
            flange = pose6_to_matrix(flange_pose)
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


def run(
    config: dict[str, Any], control: str, record_dir: str | None,
    *, hand_mode: str = "tracking", calibration_dir: str | None = None,
    calibration_hand_frame: int = -1,
) -> None:
    if hand_mode not in ("tracking", "wrist-only", "calibration"):
        raise ValueError(f"Unknown hand mode: {hand_mode}")
    if hand_mode == "calibration" and (control != "both" or calibration_dir is None
                                       or not config.get("camera", {}).get("enabled")):
        raise ValueError("Calibration needs both devices, enabled RGB-D and a new calibration directory.")
    if hand_mode == "wrist-only" and control == "arm":
        raise ValueError("wrist-only mode needs --control both (or hand for finger testing).")
    fixed_roll_pitch = None
    if hand_mode == "wrist-only" and control == "both":
        roll_pitch_deg = np.asarray(config["arm"].get("wrist_only_roll_pitch_deg"), dtype=float)
        if roll_pitch_deg.shape != (2,) or not np.isfinite(roll_pitch_deg).all():
            raise ValueError("arm.wrist_only_roll_pitch_deg must contain finite roll/pitch degrees.")
        fixed_roll_pitch = np.deg2rad(roll_pitch_deg)
    hand_replay = (HandReplay(config["hand"]["replay_clip_file"])
                   if hand_mode in ("wrist-only", "calibration") else None)
    calibration_hand_pose = None
    calibration_session = None
    if hand_mode == "calibration":
        if not -len(hand_replay.actions) <= calibration_hand_frame < len(hand_replay.actions):
            raise ValueError("Calibration hand frame is outside the replay clip.")
        calibration_hand_frame %= len(hand_replay.actions)
        calibration_hand_pose = hand_replay.actions[calibration_hand_frame].copy()
        if Path(calibration_dir).expanduser().exists():
            raise ValueError("Calibration directory already exists; select a new session directory.")
    arm = _make_arm(config) if control in ("arm", "both") else None
    hand = _make_hand(config) if control in ("hand", "both") else None
    compute = _make_compute(config, control, hand_mode)
    ik, hand_retargeter = compute.ik, compute.retargeter
    hand_feedback = (HandFeedbackPoller(
        hand, poll_hz=config['hand'].get('feedback_poll_hz', 20.0),
        max_age_s=config['hand'].get('feedback_max_age_s', .25),
    ) if hand is not None else None)
    quest_input = None
    camera_config = config.get("camera", {})
    camera = RealSenseCamera(camera_config) if camera_config.get("enabled", False) else None
    calibration = calibration_rotation(config)
    device_name = config.get("input_device", "quest3")
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
    record_root = Path(record_dir or config["recording"]["output_dir"])
    lerobot_recording = config["recording"].get("lerobot")
    if control != "both" and lerobot_recording is not None:
        # Partial-device debug captures remain diagnostics, never joint ACT data.
        record_root = record_root.parent / f"{record_root.name}_diagnostics"
        lerobot_recording = None
    recorder = EpisodeRecorder(
        record_root,
        {"config": config["_path"], "control": control, "hand_mode": hand_mode,
         "input_device": device_name, "input_side": input_config(config)["side"],
         "quest_side": input_config(config)["side"],
         "hand_feedback": ({'poll_hz': hand_feedback.poll_hz,
                            'max_age_s': hand_feedback.max_age_s,
                            'timestamps': 'host query calls; no hardware generation'}
                           if hand_feedback is not None else None)},
        sample_hz=float(config["control_hz"]),
        lerobot=lerobot_recording,
    )
    timing = _ControlTiming(float(config["control_hz"]))
    if hand_replay is not None:
        recorder.metadata["hand_replay_clip_file"] = str(hand_replay.clip_file)
    if fixed_roll_pitch is not None:
        recorder.metadata["fixed_roll_pitch_deg"] = np.rad2deg(fixed_roll_pitch).tolist()

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
                fixed_roll_pitch=fixed_roll_pitch,
                roll_pitch_step_deg=min(1.0, float(safety["max_rotation_step_deg"]) / 2.0),
            )
            tracker.update(wrist_position, wrist_quaternion)
            gate.reanchor(flange)
            last_accepted_target = flange.copy()
        if hand_retargeter is not None:
            hand_retargeter.reset(hand_feedback.get_latest().positions_raw)
        if hand_replay is not None:
            hand_replay.stop()
        paused = False
        print("Re-anchored; command streaming resumed.")

    try:
        compute.start()
        arm_config = config["arm"]
        quest_input = _input_device(config)
        if isinstance(quest_input, QuestRelayClient) and not quest_input.is_available():
            raise RuntimeError(
                "Persistent Quest listener is unavailable. Start `teleop.cli ... "
                "quest-listener` before running teleoperation."
            )
        if (device_name == "pico4" and input_config(config).get("input_mode") == "relay"
                and not quest_input.is_available()):
            raise RuntimeError(
                "Pico 4 relay is unavailable. Start "
                "`python3 AnyDexRetarget/example/input/pico4_daemon.py` before teleoperation."
            )
        if camera is not None:
            camera.start()
            recorder.metadata["camera"] = dict(camera.metadata)
            print(
                f"RealSense {camera.metadata['model']}: aligned RGB-D "
                f"{camera_config['width']}x{camera_config['height']}; "
                f"recording follows control_hz={config['control_hz']}."
            )
        if hand_mode == "calibration":
            from .tcp_calibration import CalibrationSession

            calibration_session = CalibrationSession(
                calibration_dir, config, calibration_hand_pose, calibration_hand_frame, camera.metadata,
            )
        if arm is not None:
            print(f"Nero connecting: {arm_config['can_channel']}", flush=True)
            arm.connect()
            print("Nero enabling normal mode (timeout=5.0 s).", flush=True)
            arm.enable_normal_mode()
            if not arm.is_enabled():
                raise RuntimeError("Nero is not enabled. Enable it using the approved site procedure before run.")
            _move_arm_to_start_pose(arm, config)
            startup_flange_pose, startup_tcp_pose = _read_complete_arm_startup_poses(arm)
            start_joints = arm.get_joint_positions()
            model_start_flange = ik.forward(start_joints)
            sdk_start_flange = pose6_to_matrix(startup_flange_pose)
            sdk_start_tcp = pose6_to_matrix(startup_tcp_pose)
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
            if hand_replay is not None:
                hand.open_hand()
                hand.set_thumb_index_positions_raw(
                    calibration_hand_pose if calibration_hand_pose is not None
                    else hand_replay.target(time.monotonic())
                )
                print(f"Finger clip: {hand_replay.clip_file} ({hand_replay.duration:.2f} s).")
                if calibration_session is not None:
                    print(f"Calibration: fingers fixed at frame {calibration_hand_frame}; full wrist rotation enabled.")
                else:
                    print("Wrist-only mode: thumb/index at clip start; other fingers open.")
            hand_feedback.start()
        period = 1.0 / float(config["control_hz"])
        if arm is not None:
            print(f"Start pose ready. Hold a comfortable {device_name} pose, then press R to anchor.")
        if fixed_roll_pitch is not None:
            print(
                "Wrist-only orientation: base-frame roll/pitch locked to "
                f"{np.round(np.rad2deg(fixed_roll_pitch), 3).tolist()} deg; yaw follows VR. "
                "After R, tilt targets approach the reference by at most 1 deg per cycle."
            )
        if calibration_session is not None:
            print("Keys: R=re-anchor/resume, C=pause and save stationary RGB-D/FK, Q=quit.")
            print("Capture 15-25 positions with independent roll/pitch tilts; keep the contact point visible.")
        else:
            print("Keys: R=re-anchor, B=begin recording, S=save recording, Q=quit")
        if hand_mode == "wrist-only":
            print("Finger keys after R: P=replay from start, O=cancel replay and return to start.")
        with Keyboard() as keyboard:
            while True:
                loop_started = time.monotonic()
                timing_started = time.perf_counter()
                key = keyboard.read()
                if key == "q":
                    break
                compute.check_health()
                if key == "c" and calibration_session is not None:
                    paused = True
                    print("Capture paused target streaming; keep still while feedback settles.")
                    try:
                        saved = calibration_session.capture(arm, ik, camera, hand_feedback)
                        print(f"Saved {saved}. Press R to re-anchor before changing pose.")
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        print(f"Sample not saved: {exc}. Remain paused; C=retry, R=resume.")
                    continue
                if key == "b" and calibration_session is None:
                    if recorder.active:
                        print("Recording already active; press S to save it first.")
                    else:
                        recorder.start()
                        print("Recording started.")
                elif key == "s" and calibration_session is None:
                    saved = recorder.stop_and_save()
                    print(f"Recording saved: {saved}" if saved else "No active recording.")

                frame = _latest_frame(quest_input, config)
                if frame is None:
                    if hand_replay is not None:
                        hand_replay.stop()
                    if not paused:
                        paused = True
                        gate.pause(f"{device_name} frame timeout")
                        print(f"PAUSED: {device_name} frame timeout; stale targets are not sent. Press R after recovery.")
                    time.sleep(min(period, 0.02))
                    continue
                if key == "r":
                    reanchor(frame)
                    continue
                if paused:
                    if hand_mode == "wrist-only" and key in ("p", "o"):
                        print("PAUSED: press R with valid wrist tracking before using P/O.")
                    time.sleep(min(period, 0.02))
                    continue

                stage_times = dict.fromkeys(_ControlTiming.FIELDS, 0.0)
                compute.begin_cycle()
                hand_diagnostics = dict.fromkeys(HAND_TIMING_FIELDS, np.nan)
                hand_diagnostics['hand_retarget_numevals'] = -1
                # Snapshot the observation before IK/actions; never wait for a camera frame here.
                observation_timestamp = time.monotonic()
                camera_frame = camera.get_latest() if camera is not None and recorder.active else None
                observation_hand = (hand_feedback.get_latest()
                                    if hand_feedback is not None and recorder.active
                                    and lerobot_recording is not None else None)
                observation_arm_joints = np.full(7, np.nan)
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
                    stage_started = time.perf_counter()
                    seed = arm.get_joint_positions()
                    observation_arm_joints = seed.copy()
                    stage_times["timing_seed_feedback_s"] = time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    solved = ik.solve(target_transform, seed)
                    stage_times["timing_ik_s"] = time.perf_counter() - stage_started
                    if solved is None:
                        paused = True
                        if hand_replay is not None:
                            hand_replay.stop()
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
                        if hand_replay is not None:
                            hand_replay.stop()
                        print(f"PAUSED: {gate.reason}; press R to re-anchor.")
                        continue
                    arm_action = solved
                    last_accepted_target = target_transform.copy()
                if hand is not None:
                    stage_started = time.perf_counter()
                    if hand_replay is not None:
                        if calibration_hand_pose is not None:
                            hand_action = calibration_hand_pose.copy()
                        elif key == "p":
                            hand_replay.start(time.monotonic())
                            print("Finger replay started from clip start.")
                        elif key == "o":
                            hand_replay.reset()
                            print("Finger replay cancelled; returning to clip start.")
                        if calibration_hand_pose is None:
                            hand_action = hand_replay.target(time.monotonic())
                    else:
                        _, hand_action = hand_retargeter.retarget(landmarks)
                    stage_times["timing_hand_retarget_s"] = time.perf_counter() - stage_started
                    diagnostics = getattr(hand_retargeter, 'last_diagnostics', {})
                    hand_diagnostics.update({key: diagnostics[key] for key in hand_diagnostics
                                             if key in diagnostics})

                stage_times.update(compute.timings)
                if time.monotonic() - frame.received_at > float(input_config(config)["max_age_s"]):
                    paused = True
                    gate.pause("Input expired during computation")
                    if hand_replay is not None:
                        hand_replay.stop()
                    print("PAUSED: input expired during computation; press R to re-anchor.")
                    continue

                arm_sent_at = np.nan
                hand_sent_at = np.nan
                if arm is not None:
                    stage_started = time.perf_counter()
                    arm.move_js(arm_action)
                    # arm.move_joints(
                    #     arm_action,
                    #     speed_percent=arm_config["follow_speed_percent"],
                    # )
                    arm_sent_at = time.monotonic()
                    stage_times["timing_arm_send_s"] = time.perf_counter() - stage_started
                if hand is not None:
                    stage_started = time.perf_counter()
                    if hand_replay is not None:
                        hand.set_thumb_index_positions_raw(hand_action)
                    else:
                        hand.set_joint_positions_raw(hand_action)
                    hand_sent_at = time.monotonic()
                    stage_times["timing_hand_send_s"] = time.perf_counter() - stage_started
                    stage_times['timing_hand_send_wait_s'] = getattr(
                        hand, 'last_position_send_wait_s', 0.0
                    )

                stage_started = time.perf_counter()
                arm_joints = arm.get_joint_positions() if arm is not None else np.full(7, np.nan)
                arm_torques = arm.get_joint_torques() if arm is not None else np.full(7, np.nan)
                arm_tcp = arm.get_tcp_pose() if arm is not None else np.full(6, np.nan)
                feedback = hand_feedback.get_latest() if hand_feedback is not None else None
                hand_raw = feedback.positions_raw if feedback is not None else np.full(20, -1)
                hand_read_at = time.monotonic() if feedback is not None else np.nan
                stage_times["timing_feedback_s"] = time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                recorder.append(
                    camera_frame=camera_frame,
                    timestamp_monotonic=time.monotonic(),
                    observation_timestamp_monotonic=observation_timestamp,
                    observation_arm_joint_position=observation_arm_joints,
                    observation_hand_position_raw=(observation_hand.positions_raw
                                                   if observation_hand is not None else np.full(20, -1)),
                    observation_hand_feedback_query_started_at=(
                        observation_hand.query_started_at if observation_hand is not None else np.nan),
                    observation_hand_feedback_query_finished_at=(
                        observation_hand.query_finished_at if observation_hand is not None else np.nan),
                    observation_hand_feedback_query_sequence=(
                        observation_hand.query_sequence if observation_hand is not None else -1),
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
                    hand_feedback_query_started_at=(feedback.query_started_at if feedback else np.nan),
                    hand_feedback_query_finished_at=(feedback.query_finished_at if feedback else np.nan),
                    hand_feedback_query_sequence=(feedback.query_sequence if feedback else -1),
                    hand_feedback_snapshot_read_at=hand_read_at,
                    hand_feedback_query_age_s=(hand_read_at - feedback.query_finished_at
                                               if feedback else np.nan),
                    **hand_diagnostics,
                    **stage_times,
                )
                record_s = time.perf_counter() - stage_started
                work_s = time.perf_counter() - timing_started
                remaining = period - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
                if timing.append(stage_times, record_s, work_s,
                                 time.perf_counter() - timing_started, hand_diagnostics):
                    if hand_feedback is not None:
                        hand_feedback.report()
    finally:
        try:
            compute.close()
        except Exception as exc:
            print(f"Compute process stop failed: {type(exc).__name__}: {exc}")
        feedback_stopped = True
        if hand_feedback is not None:
            try:
                hand_feedback.stop()
            except Exception as exc:
                feedback_stopped = False
                print(f'Hand feedback stop failed: {type(exc).__name__}: {exc}; '
                      'hand disconnect deferred until its query exits.')
        try:
            saved = recorder.stop_and_save()
            if saved:
                print(f"Active recording saved during shutdown: {saved}")
        except Exception as exc:
            print(f"Recording save failed during shutdown: {type(exc).__name__}: {exc}")
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:
                print(f"Camera stop failed: {type(exc).__name__}: {exc}")
        if quest_input is not None:
            quest_input.stop()
        errors = _disconnect_devices(hand if feedback_stopped else None, arm)
        if errors:
            print("Disconnect errors: " + "; ".join(errors))
        timing.report()
        if hand_feedback is not None:
            hand_feedback.report()
        print("Command streaming stopped; disconnect was attempted without automatic disable/e-stop.")
