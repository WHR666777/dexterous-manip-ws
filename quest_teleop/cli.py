"""Command-line entrypoints for staged Quest teleoperation validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
import yaml

from .camera import RealSenseCapture
from .configuration import load_config, serializable_config, validate_hardware_config
from .control import (
    ArmControlSession,
    VirtualArm,
    apply_sdk_joint_limits,
    dispatch_arm_proposals,
    read_arm_feedback,
)
from .geometry import calibrate_base_rotation, matrix_to_pose7, pose_distance, valid_rotation
from .hand import AnyDexL20Mapper, VirtualHand
from .recording import EpisodeRecorder, NullRecorder
from .tracking import QuestTrackingReceiver, WorldResetDetector


class Keyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("interactive terminal required for R/Space/B/S/Q")
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return ""
        chars = os.read(self.fd, 128).decode("utf-8", errors="ignore").lower()
        if "q" in chars: return "q"
        if "r" in chars: return "r"
        if "b" in chars: return "b"
        if "s" in chars: return "s"
        if " " in chars: return "space"
        return ""

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)


def _common_parser(description, *, side=False, recording=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--calibration", action="append", default=[],
        help="side->rotation YAML override; repeat for separate left/right files",
    )
    parser.add_argument("--transport", choices=("tcp", "tcp_server", "tcp_client", "udp"), default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="connect and command real hardware")
    if side:
        parser.add_argument("--side", choices=("left", "right"), default="right")
    if recording:
        parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--output", default=None)
        parser.add_argument("--camera", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _apply_calibration(config, path):
    if not path:
        return
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    sides = values.get("sides", values)
    for side in ("left", "right"):
        if side in sides:
            matrix = sides[side].get("base_from_quest_rotation", sides[side])
            matrix = np.asarray(matrix, dtype=float)
            if not valid_rotation(matrix):
                raise ValueError(f"calibration for {side} is not a valid SO(3) rotation")
            config["sides"][side]["base_from_quest_rotation"] = matrix
            config["sides"][side]["calibrated"] = True


def _tracking_from_args(args, config):
    quest = config["quest"]
    return QuestTrackingReceiver(
        transport=args.transport or quest.get("transport", "tcp"),
        host=args.host or quest.get("host", "0.0.0.0"),
        port=args.port or quest.get("port", 8000),
    )


def tracking_check_main(argv=None):
    parser = _common_parser("Inspect Quest hands/head without hardware", recording=False)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--head-turn-check", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    receiver = _tracking_from_args(args, config)
    started = time.monotonic()
    baseline = None
    last_sequences = {"left": -1, "right": -1, "head": -1}
    counts = dict.fromkeys(last_sequences, 0)
    sequence_jumps = dict.fromkeys(last_sequences, 0)
    last_report = started
    try:
        while not args.duration or time.monotonic() - started < args.duration:
            snapshot = receiver.snapshot()
            if snapshot.error:
                print("HTS error:", snapshot.error)
                return 2
            for side in ("left", "right"):
                hand = snapshot.hand(side)
                if hand is not None and hand.sequence_id > last_sequences[side]:
                    if last_sequences[side] >= 0:
                        sequence_jumps[side] += max(0, hand.sequence_id - last_sequences[side] - 1)
                    counts[side] += 1
                    last_sequences[side] = hand.sequence_id
            if snapshot.head is not None and snapshot.head.sequence_id > last_sequences["head"]:
                if last_sequences["head"] >= 0:
                    sequence_jumps["head"] += max(0, snapshot.head.sequence_id - last_sequences["head"] - 1)
                counts["head"] += 1
                last_sequences["head"] = snapshot.head.sequence_id
            if snapshot.sequence_reset:
                print("HTS sequence rollback/restart detected")
                receiver.clear_sequence_reset()
            if args.head_turn_check and baseline is None and snapshot.valid():
                baseline = {side: snapshot.hand(side).wrist_world.copy() for side in ("left", "right")}
                print("Head-turn check baseline captured: keep both wrists fixed and rotate only the headset.")
            now = time.monotonic()
            if now - last_report >= 1.0:
                elapsed = now - started
                line = {name: round(count / elapsed, 1) for name, count in counts.items()}
                line["valid_pair"] = snapshot.valid()
                line["sequence"] = last_sequences.copy()
                line["sequence_jumps"] = sequence_jumps.copy()
                line["ages_ms"] = {
                    side: None if snapshot.hand(side) is None else round(snapshot.hand(side).age_s() * 1000, 1)
                    for side in ("left", "right")
                }
                line["wrist_pose_flu"] = {
                    side: matrix_to_pose7(snapshot.hand(side).wrist_world).round(5).tolist()
                    for side in ("left", "right") if snapshot.hand(side) is not None
                }
                line["head_pose_flu"] = (
                    None if snapshot.head is None
                    else matrix_to_pose7(snapshot.head.pose_world).round(5).tolist()
                )
                if baseline is not None:
                    line["wrist_drift"] = {
                        side: dict(zip(("mm", "deg"), (
                            pose_distance(snapshot.hand(side).wrist_world, baseline[side])[0] * 1000,
                            np.rad2deg(pose_distance(snapshot.hand(side).wrist_world, baseline[side])[1]),
                        ))) for side in ("left", "right") if snapshot.hand(side) is not None
                    }
                print(json.dumps(line, ensure_ascii=False))
                last_report = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
    return 0


def _capture_position(receiver, side, prompt):
    input(prompt)
    samples = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        hand = receiver.snapshot().hand(side)
        if hand is not None and hand.age_s() < 0.1:
            samples.append(hand.wrist_world[:3, 3].copy())
        time.sleep(0.02)
    if len(samples) < 10:
        raise RuntimeError("not enough fresh Quest wrist samples")
    return np.median(np.stack(samples), axis=0)


def calibrate_main(argv=None):
    parser = _common_parser("Guided Quest-world to robot-base rotation calibration", side=True, recording=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify", default=None, help="validate and print an existing calibration YAML")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.verify:
        _apply_calibration(config, args.verify)
        rotation = config["sides"][args.side]["base_from_quest_rotation"]
        print(f"{args.side} calibration is valid SO(3):")
        print(rotation)
        return 0
    receiver = _tracking_from_args(args, config)
    output = Path(args.output or f"quest_calibration_{args.side}.yaml")
    try:
        print("Keep the selected wrist still at each prompt; each capture uses a 2 s median.")
        origin_x = _capture_position(receiver, args.side, "Place wrist at origin, press Enter: ")
        plus_x = _capture_position(receiver, args.side, "Move along robot Base +X by >=5 cm, press Enter: ")
        origin_y = _capture_position(receiver, args.side, "Return to origin, press Enter: ")
        plus_y = _capture_position(receiver, args.side, "Move along robot Base +Y by >=5 cm, press Enter: ")
        rotation = calibrate_base_rotation(plus_x - origin_x, plus_y - origin_y)
        payload = {"sides": {args.side: {"base_from_quest_rotation": rotation.tolist(), "calibrated": True}}}
        with output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
        print("Calibration written:", output)
        print(rotation)
    finally:
        receiver.stop()
    return 0


def _build_session(side, config, ik):
    control = config["control"]
    return ArmControlSession(
        side,
        config["sides"][side]["base_from_quest_rotation"],
        position_scale=control.get("position_scale", 0.5),
        max_tracker_position_step_m=control.get("max_tracker_position_step_m", 0.04),
        max_tracker_rotation_step_deg=control.get("max_tracker_rotation_step_deg", 12),
        max_target_position_step_m=control.get("max_target_position_step_m", 0.01),
        max_target_rotation_step_deg=control.get("max_target_rotation_step_deg", 5),
        max_joint_step_deg=control.get("max_joint_step_deg", 5),
        ik=ik,
    )


def _create_real_arm(side, side_config, resources, ik):
    from robot_control import NeroArm
    arm = NeroArm(can_interface="socketcan", can_channel=side_config["arm_can"])
    resources.callback(arm.disconnect)
    # Configuration/URDF mismatches must fail before connecting or enabling.
    apply_sdk_joint_limits(ik, arm._driver.get_config())
    arm.connect()
    arm.enable(timeout=5.0)
    arm._driver.set_normal_mode()
    return arm


def _create_real_hand(side, side_config, resources, speed):
    from robot_control import LinkerHandL20
    hand = LinkerHandL20(hand_type=side, can_channel=side_config["hand_can"])
    resources.callback(hand.disconnect)
    hand.connect()
    hand.set_speed(np.full(5, int(speed), dtype=np.int64))
    return hand


def _feedback(arm, execute):
    return read_arm_feedback(arm) if execute else arm.feedback()


def _sample(snapshot, sides, feedbacks, proposals, hand_data, sent):
    now_ns = time.monotonic_ns()
    head = snapshot.head
    sample = {
        "monotonic_ns": np.asarray(now_ns, dtype=np.int64),
        "unix_ns": np.asarray(time.time_ns(), dtype=np.int64),
        "sent": np.asarray(bool(sent), dtype=np.bool_),
        "head/pose": np.full(7, np.nan) if head is None else matrix_to_pose7(head.pose_world),
        "head/valid": np.asarray(head is not None, dtype=np.bool_),
        "head/recv_monotonic_ns": np.asarray(-1 if head is None else head.recv_monotonic_ns, dtype=np.int64),
        "head/recv_unix_ns": np.asarray(-1 if head is None or head.recv_unix_ns is None else head.recv_unix_ns, dtype=np.int64),
        "head/source_ts_ns": np.asarray(-1 if head is None or head.source_ts_ns is None else head.source_ts_ns, dtype=np.int64),
        "head/sequence": np.asarray(-1 if head is None else head.sequence_id, dtype=np.int64),
    }
    for side in sides:
        hand = snapshot.hand(side)
        feedback = feedbacks[side]
        proposal = proposals[side]
        sample.update({
            f"quest/{side}/wrist": matrix_to_pose7(hand.wrist_world),
            f"quest/{side}/landmarks": hand.landmarks_local,
            f"quest/{side}/source_ts_ns": np.asarray(hand.source_ts_ns if hand.source_ts_ns is not None else -1, dtype=np.int64),
            f"quest/{side}/recv_monotonic_ns": np.asarray(hand.recv_monotonic_ns, dtype=np.int64),
            f"quest/{side}/recv_unix_ns": np.asarray(hand.recv_unix_ns if hand.recv_unix_ns is not None else -1, dtype=np.int64),
            f"quest/{side}/age_s": np.asarray(hand.age_s(now_ns), dtype=np.float64),
            f"quest/{side}/valid": np.asarray(True, dtype=np.bool_),
            f"quest/{side}/sequence": np.asarray(hand.sequence_id, dtype=np.int64),
            f"observation/{side}/arm_joint": feedback.joints,
            f"observation/{side}/arm_torque": feedback.torques,
            f"observation/{side}/joint_timestamps": feedback.joint_timestamps,
            f"observation/{side}/state_timestamps": feedback.state_timestamps,
            f"observation/{side}/flange_timestamps": feedback.flange_timestamps,
            f"observation/{side}/sampled_monotonic_s": np.asarray(feedback.sampled_monotonic, dtype=np.float64),
            f"observation/{side}/flange": matrix_to_pose7(feedback.flange),
            f"observation/{side}/tcp": np.full(7, np.nan) if not np.isfinite(feedback.tcp).all() else matrix_to_pose7(feedback.tcp),
            f"action/{side}/flange": matrix_to_pose7(proposal.target_flange),
            f"action/{side}/ik_joint": proposal.ik_joints,
            f"action/{side}/arm_joint": proposal.command_joints,
            f"action/{side}/sent": np.asarray(bool(sent), dtype=np.bool_),
            # 0 is the canonical READY-and-sent status. Failed cycles terminate
            # the episode and are represented by the episode stop reason.
            f"action/{side}/status_code": np.asarray(0, dtype=np.int16),
        })
        if side in hand_data:
            qpos, raw, actual, actual_time = hand_data[side]
            sample[f"action/{side}/hand_qpos"] = qpos
            sample[f"action/{side}/hand_raw"] = raw
            sample[f"observation/{side}/hand_raw"] = actual
            sample[f"observation/{side}/hand_recv_monotonic_ns"] = np.asarray(actual_time, dtype=np.int64)
    return sample


def run_teleop(mode: str, argv=None):
    single = mode in ("arm", "arm_hand")
    include_hands = mode in ("arm_hand", "bimanual")
    parser = _common_parser(f"Quest {mode} teleoperation", side=single, recording=True)
    args = parser.parse_args(argv)
    if args.record and not args.output:
        parser.error("--record requires --output")
    config = load_config(args.config)
    for calibration in args.calibration:
        _apply_calibration(config, calibration)
    sides = (args.side,) if single else ("left", "right")
    if args.execute:
        validate_hardware_config(config, sides, include_hands)
        uncalibrated = [side for side in sides if not config["sides"][side].get("calibrated")]
        if uncalibrated:
            parser.error(f"calibration required before --execute: {uncalibrated}")
        phrase = "EXECUTE " + "+".join(side.upper() for side in sides)
        if input(f"Type {phrase} to enable hardware: ").strip() != phrase:
            return 1
    receiver = _tracking_from_args(args, config)
    control_hz = float(config["control"].get("hz", 20))
    camera = None
    recorder = NullRecorder()
    with ExitStack() as resources:
        resources.callback(receiver.stop)
        if args.record:
            camera_config = config["recording"].get("realsense", {})
            if args.camera and camera_config.get("enabled", True):
                camera = RealSenseCapture(**{
                    key: camera_config.get(key) for key in ("serial", "width", "height", "fps")
                })
                resources.callback(camera.close)
            recorder = EpisodeRecorder(
                args.output,
                metadata={
                    "config": serializable_config(config),
                    "mode": mode,
                    "sides": sides,
                    "coordinate_convention": {
                        "quest_input": "Unity world: +X right, +Y up, +Z forward (left-handed)",
                        "quest_canonical": "FLU: +X forward, +Y left, +Z up (right-handed)",
                        "robot_target": "NERO base to flange",
                    },
                    "units": {
                        "position": "m", "rotation": "quaternion_xyzw",
                        "arm_joint": "rad", "arm_torque": "N*m",
                        "hand_raw": "integer [0,255]",
                        "time": "field suffix determines seconds or nanoseconds",
                    },
                    "training_vector_order": "left_arm_7,left_hand_20,right_arm_7,right_hand_20",
                    "status_codes": {"0": "READY_SENT"},
                },
                camera_metadata=None if camera is None else camera.metadata,
                video_fps=30 if camera is None else camera.fps,
            )
            resources.callback(recorder.close)
        arms, sessions, hands, mappers = {}, {}, {}, {}
        from robot_control.nero_ik import NeroIK
        for side in sides:
            ik = NeroIK()
            arms[side] = _create_real_arm(side, config["sides"][side], resources, ik) if args.execute else VirtualArm(ik)
            sessions[side] = _build_session(side, config, ik)
            if include_hands:
                hands[side] = _create_real_hand(side, config["sides"][side], resources, speed=30) if args.execute else VirtualHand()
                mappers[side] = AnyDexL20Mapper(side, config["control"].get("hand_max_raw_step", 10))
                mappers[side].seed(hands[side].get_joint_positions_raw())
        print("R=reanchor  Space=toggle  B=start record  S=save  Q=quit")
        previous_source = {side: -1 for side in sides}
        reset_detector = WorldResetDetector()
        camera_fault_reported = False
        with Keyboard() as keyboard, ThreadPoolExecutor(max_workers=len(sides)) as pool:
            while True:
                cycle = time.monotonic()
                key = keyboard.read()
                if key == "q": break
                snapshot = receiver.snapshot()
                if snapshot.error:
                    raise RuntimeError(snapshot.error)
                tracking_valid = snapshot.valid(
                    sides,
                    config["quest"].get("max_age_s", 0.15),
                    config["quest"].get("max_pair_skew_s", 0.04),
                )
                if snapshot.sequence_reset or reset_detector.update(snapshot):
                    for session in sessions.values(): session.pause("QUEST_WORLD_RESET")
                    if recorder.recording: recorder.stop("quest_world_reset")
                    receiver.clear_sequence_reset()
                    print("Quest tracking/world reset detected; press R to reanchor")
                    time.sleep(max(0, 1/control_hz - (time.monotonic()-cycle)))
                    continue
                if key == "b":
                    if isinstance(recorder, NullRecorder): print("Recording disabled: restart with --record --output PATH")
                    elif all(not session.paused for session in sessions.values()):
                        started = recorder.start()
                        # Frames captured before B belong to no episode. Do not
                        # discard frames if B is pressed during an active episode.
                        if started and camera is not None: camera.drain()
                    else: print("Reanchor and enable control before recording")
                if key == "s": recorder.stop("operator")
                if camera is not None and camera.error is not None:
                    if recorder.recording: recorder.stop("camera_stream_error")
                    if not camera_fault_reported:
                        print("RealSense stream stopped; episode truncated:", camera.error)
                        camera_fault_reported = True
                try:
                    feedbacks = {side: _feedback(arms[side], args.execute) for side in sides}
                except Exception:
                    for session in sessions.values(): session.pause("FEEDBACK_READ_ERROR", fault=True)
                    recorder.stop("feedback_read_error")
                    raise
                if key == "r":
                    if tracking_valid:
                        results = [sessions[side].reanchor(snapshot.hand(side).wrist_world, feedbacks[side]) for side in sides]
                        print("reanchor:", dict(zip(sides, results)))
                    else: print("reanchor rejected: tracking pair invalid/stale")
                    time.sleep(max(0, 1/control_hz - (time.monotonic()-cycle)))
                    continue
                if key == "space":
                    states = {side: sessions[side].toggle() for side in sides}
                    if not all(states.values()):
                        for session in sessions.values(): session.pause("GROUP_ENABLE_REJECTED")
                    print("control enabled:", states)
                if not tracking_valid:
                    for session in sessions.values(): session.pause("TRACKING_STALE")
                    if recorder.recording: recorder.stop("tracking_stale")
                    time.sleep(max(0, 1/control_hz - (time.monotonic()-cycle)))
                    continue
                # Do not send the same HTS source frame twice.
                if any(snapshot.hand(side).sequence_id <= previous_source[side] for side in sides):
                    time.sleep(max(0, 1/control_hz - (time.monotonic()-cycle)))
                    continue
                for side in sides: previous_source[side] = snapshot.hand(side).sequence_id
                futures = {
                    side: pool.submit(sessions[side].propose, snapshot.hand(side).wrist_world, feedbacks[side])
                    for side in sides
                }
                proposals = {side: future.result() for side, future in futures.items()}
                if not all(proposal.ready for proposal in proposals.values()):
                    codes = {side: proposal.code for side, proposal in proposals.items()}
                    if any(code != "PAUSED" for code in codes.values()):
                        for session in sessions.values(): session.pause("GROUP_PROPOSAL_FAILED")
                        if recorder.recording: recorder.stop("proposal_failed")
                        print("control paused; proposal rejected:", codes)
                    time.sleep(max(0, 1/control_hz - (time.monotonic()-cycle)))
                    continue
                hand_data = {}
                if include_hands:
                    try:
                        for side in sides:
                            qpos, raw = mappers[side].map(snapshot.hand(side).landmarks_local)
                            actual = np.asarray(hands[side].get_joint_positions_raw(), dtype=np.int64)
                            actual_time = time.monotonic_ns()
                            if raw.shape != (20,) or actual.shape != (20,): raise ValueError("invalid L20 shape")
                            hand_data[side] = (qpos, raw, actual, actual_time)
                    except Exception:
                        for session in sessions.values(): session.pause("HAND_MAPPING_ERROR", fault=True)
                        recorder.stop("hand_mapping_error")
                        raise
                try:
                    dispatch_arm_proposals((sessions[side], arms[side], proposals[side]) for side in sides)
                    for side in sides:
                        if include_hands:
                            hands[side].set_joint_positions_raw(hand_data[side][1])
                            mappers[side].commit(hand_data[side][1])
                except Exception:
                    for session in sessions.values(): session.pause("DISPATCH_ERROR", fault=True)
                    recorder.stop("dispatch_error")
                    raise
                if recorder.recording:
                    frames = () if camera is None else camera.drain()
                    try:
                        recorder.append(_sample(snapshot, sides, feedbacks, proposals, hand_data, True), frames)
                    except Exception:
                        for session in sessions.values(): session.pause("RECORDING_ERROR")
                        recorder.abort("recording_error")
                        raise
                elapsed = time.monotonic() - cycle
                if elapsed > 1/control_hz:
                    print(f"control deadline missed: {elapsed*1000:.1f} ms")
                time.sleep(max(0, 1/control_hz - elapsed))
        recorder.stop("quit")
    return 0


def single_arm_main(argv=None): return run_teleop("arm", argv)
def single_arm_hand_main(argv=None): return run_teleop("arm_hand", argv)
def bimanual_main(argv=None): return run_teleop("bimanual", argv)
