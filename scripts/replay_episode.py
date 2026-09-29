"""Replay recorded Nero/L20 joint feedback and RGB-D; hardware requires --execute."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


WINDOW = "Episode replay"
PAGE_LINES = 13
UNITS = {
    "arm_action": "rad", "arm_joint_position": "rad", "arm_joint_torque": "N*m",
    "arm_tcp_pose": "xyz m, rpy rad", "hand_action_raw": "raw 0..255",
    "hand_position_raw": "raw 0..255", "quest_wrist": "xyz m, quaternion xyzw",
    "quest_landmarks": "m", "target_base_flange": "4x4 transform, translation m",
}


class Episode:
    def __init__(self, path: Path, resources: ExitStack):
        trajectory = path / "trajectory.npz" if path.is_dir() else path
        meta_path = (trajectory.parent / "metadata.json" if trajectory.name == "trajectory.npz"
                     else trajectory.with_suffix(".json"))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(trajectory, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        timestamps = self.data.get("timestamp_monotonic", np.array([]))
        if timestamps.ndim != 1 or not len(timestamps):
            raise ValueError("Episode has no timestamped trajectory samples.")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            raise ValueError("Trajectory timestamps must be finite and nondecreasing.")
        if any(value.ndim == 0 or len(value) != len(timestamps) for value in self.data.values()):
            raise ValueError("Trajectory fields have inconsistent sample counts.")
        self.times = timestamps - timestamps[0]
        self.name = trajectory.parent.name if trajectory.name == "trajectory.npz" else trajectory.stem
        self.video = self.depth = None
        self.cached_index = -1
        self.cached_images = None
        camera = self.metadata.get("camera")
        if camera is not None:
            import h5py

            if not {"camera_frame_index", "camera_valid"} <= self.data.keys():
                raise ValueError("RGB-D episode is missing camera indices or validity flags.")
            self.depth_scale = float(camera["depth_scale_m"])
            if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
                raise ValueError("Camera depth_scale_m must be finite and positive.")
            depth_file = resources.enter_context(h5py.File(trajectory.parent / camera["depth_file"], "r"))
            self.depth = depth_file[camera.get("depth_dataset", "depth")]
            self.video = cv2.VideoCapture(str(trajectory.parent / camera["color_file"]))
            resources.callback(self.video.release)
            if not self.video.isOpened():
                raise ValueError("Cannot open the episode's RGB video.")

    def images(self, row: int):
        """Decode only the referenced frame; never let an invalid -1 index select depth[-1]."""
        if self.video is None:
            return None, "No camera recorded"
        index = int(self.data["camera_frame_index"][row])
        if not self.data["camera_valid"][row] or index < 0:
            return None, "No valid RGB-D at this sample"
        if index >= len(self.depth):
            raise ValueError(f"Sample {row}: depth frame {index} is out of range.")
        if index != self.cached_index:
            if int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.video.read()
            if not ok:
                raise ValueError(f"Sample {row}: cannot decode RGB frame {index}.")
            self.cached_images = bgr, self.depth[index].astype(np.float32) * self.depth_scale
            self.cached_index = index
        return self.cached_images, f"Stored RGB-D frame {index}"

    def field_lines(self, row: int) -> list[str]:
        lines = []
        for key, values in self.data.items():
            value = np.array2string(np.asarray(values[row]), precision=6, threshold=np.inf,
                                    max_line_width=110)
            units = f" ({UNITS[key]})" if key in UNITS else ""
            lines.append(f"{key}{units}:")
            lines.extend("  " + line for line in value.splitlines())
        return lines


def text(canvas, message, x, y, color=(220, 220, 220)):
    cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def tile(image):
    """Fit a frame into a 640x480 panel without changing its aspect ratio."""
    height, width = image.shape[:2]
    scale = min(640 / width, 480 / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = resized.shape[:2]
    panel[(480 - h) // 2:(480 - h) // 2 + h, (640 - w) // 2:(640 - w) // 2 + w] = resized
    return panel


def render(episode: Episode, row: int, page: int, paused: bool, depth_max: float, speed: float,
           execute: bool = False):
    canvas = np.full((860, 1280, 3), 24, dtype=np.uint8)
    images, image_status = episode.images(row)
    if images is not None:
        bgr, meters = images
        intensity = (np.clip(meters / depth_max, 0, 1) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
        colors[meters <= 0] = 0
        canvas[60:540, :640] = tile(bgr)
        canvas[60:540, 640:] = tile(colors)
    else:
        text(canvas, image_status, 150, 300)
        text(canvas, image_status, 790, 300)
    state = "PAUSED" if paused else "PLAYING"
    text(canvas, f"{state} | sample {row + 1}/{len(episode.times)} | "
         f"t={episode.times[row]:.3f}/{episode.times[-1]:.3f}s | {speed:g}x | {image_status}", 12, 22)
    text(canvas, "RGB", 12, 48)
    text(canvas, f"Depth: 0 .. {depth_max:g} m (fixed scale; invalid=black)", 650, 48)
    lines = episode.field_lines(row)
    pages = max(1, (len(lines) + PAGE_LINES - 1) // PAGE_LINES)
    page %= pages
    controls = ("HARDWARE | Space: start/pause stream | A/D: disabled" if execute else
                "Space: pause/play | A/D: previous/next sample (pauses)")
    text(canvas, f"Fields {page + 1}/{pages} | {controls} | J/L: field pages | Q/Esc: quit", 12, 562)
    for i, line in enumerate(lines[page * PAGE_LINES:(page + 1) * PAGE_LINES]):
        text(canvas, line, 12, 586 + 20 * i)
    return canvas, pages


def prepare_arm(episode: Episode, config: dict, resources: ExitStack):
    """Validate feedback samples, then approach this episode's first sample at low speed."""
    from robot_control.nero import NeroArm

    joints = np.asarray(episode.data.get("arm_joint_position"), dtype=float)
    if joints.shape != (len(episode.times), 7) or not np.all(np.isfinite(joints)):
        raise ValueError("Replay requires finite arm_joint_position (N, 7); hand-only data cannot drive Nero.")
    if np.any(np.diff(episode.times) <= 0):
        raise ValueError("Hardware replay requires strictly increasing trajectory timestamps.")
    max_delta = positive_float(config["max_joint_delta_rad"])
    tolerance = positive_float(config["start_tolerance_rad"])
    timeout = positive_float(config["start_timeout_s"])
    if np.any(np.abs(np.diff(joints, axis=0)) > max_delta):
        raise ValueError("Recorded joint step exceeds arm.max_joint_delta_rad; no motion sent.")
    # No feedback delta during offline limit validation; the streaming loop checks it explicitly.
    arm = NeroArm(can_interface=config["can_interface"], can_channel=config["can_channel"])
    resources.callback(arm.disconnect)
    for target in joints:
        arm.validate_joint_command(target)
    arm.connect()
    deadline = time.monotonic() + 3.0
    while True:
        try:
            current = arm.get_joint_positions()
            break
        except RuntimeError as exc:
            if not str(exc).startswith("Nero SDK feedback is unavailable"):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError("Nero did not provide complete joint feedback within 3 s.") from exc
            time.sleep(0.05)
    if not arm.is_enabled():
        raise RuntimeError("Nero is disabled; enable it with teleop.cli arm-enable --execute first.")
    target = joints[0]
    print("Episode first arm_joint_position (rad):", target.tolist())
    if np.max(np.abs(target - current)) > tolerance:
        arm.move_joints(target, speed_percent=config["start_speed_percent"])
        deadline = time.monotonic() + timeout
        while np.max(np.abs(target - current)) > tolerance:
            if time.monotonic() >= deadline:
                raise TimeoutError("Nero did not reach the episode's first sample in time.")
            time.sleep(0.05)
            current = arm.get_joint_positions()
    print("Nero first sample reached.")
    return arm


def prepare_hand(episode: Episode, config: dict, resources: ExitStack):
    """Validate the recorded raw feedback before opening the configured L20."""
    from robot_control.l20 import LinkerHandL20

    positions = np.asarray(episode.data.get("hand_position_raw"), dtype=float)
    if (positions.shape != (len(episode.times), 20) or not np.all(np.isfinite(positions))
            or np.any(positions != np.floor(positions)) or np.any((positions < 0) | (positions > 255))):
        raise ValueError("Replay requires hand_position_raw (N, 20) integers in [0, 255]; no action fallback.")
    if np.any(np.diff(episode.times) <= 0):
        raise ValueError("Hardware replay requires strictly increasing trajectory timestamps.")
    if episode.metadata.get("quest_side") != config["type"]:
        raise ValueError("Recorded quest_side must match hand.type for L20 replay.")
    speed = np.asarray(config["speed"], dtype=float)
    if (speed.shape != (5,) or not np.all(np.isfinite(speed)) or np.any(speed != np.floor(speed))
            or np.any((speed < 0) | (speed > 255))):
        raise ValueError("hand.speed must contain five integers in [0, 255].")
    hand = LinkerHandL20(hand_type=config["type"], can_channel=config["can_channel"])
    resources.callback(hand.disconnect)
    hand.connect()
    hand.set_speed(config["speed"])
    return hand


def replay(episode: Episode, depth_max: float, speed: float, *, arm_config: dict | None = None,
           hand_config: dict | None = None):
    row = page = 0
    hardware = arm_config is not None or hand_config is not None
    paused = hardware or len(episode.times) == 1
    data_anchor = 0.0
    drawn = None
    sent_row = None
    pages = 1
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 860)
    resources = ExitStack()
    try:
        # Reject missing/wrong-side hand data before the arm's initial approach can move it.
        hand = prepare_hand(episode, hand_config, resources) if hand_config is not None else None
        arm = prepare_arm(episode, arm_config, resources) if arm_config is not None else None
        if hand is not None:
            hand.set_joint_positions_raw(episode.data["hand_position_raw"][0])
            print("L20 first feedback pose sent; confirm the hand has settled before pressing Space.")
        if hardware:
            print("Space starts recorded feedback replay; no Quest input or IK.")
        wall_anchor = time.monotonic()
        while True:
            if not paused and drawn is not None:
                elapsed = (time.monotonic() - wall_anchor) * speed + data_anchor
                # Advance at most one row per draw so every recorded sample is inspectable.
                if row + 1 < len(episode.times) and episode.times[row + 1] <= elapsed:
                    row += 1
                if not hardware and row == len(episode.times) - 1:
                    paused = True
            if drawn != (row, page, paused):
                canvas, pages = render(episode, row, page, paused, depth_max, speed, hardware)
                if hardware and not paused and sent_row != row:
                    if arm is not None:
                        target = episode.data["arm_joint_position"][row]
                        arm.validate_joint_command(target, max_joint_delta=float(arm_config["max_joint_delta_rad"]))
                        arm.move_js(target)
                    if hand is not None:
                        hand.set_joint_positions_raw(episode.data["hand_position_raw"][row])
                    sent_row = row
                    # Preserve each interval even when decoding/CAN is slow; never burst to catch up.
                    wall_anchor, data_anchor = time.monotonic(), episode.times[row]
                cv2.imshow(WINDOW, canvas)
                drawn = (row, page, paused)
                if hardware and sent_row == len(episode.times) - 1:
                    paused = True
            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                if hardware and sent_row == len(episode.times) - 1:
                    continue  # Never jump from the last physical pose back to the first.
                paused = not paused
                if not paused:
                    if row == len(episode.times) - 1:
                        row = 0
                    wall_anchor, data_anchor = time.monotonic(), episode.times[row]
            elif not hardware and key in (ord("a"), ord("A"), ord("d"), ord("D")):
                row = int(np.clip(row + (1 if key in (ord("d"), ord("D")) else -1),
                                  0, len(episode.times) - 1))
                paused = True
            elif key in (ord("j"), ord("J"), ord("l"), ord("L")):
                page = (page + (1 if key in (ord("l"), ord("L")) else -1)) % pages
    finally:
        try:
            resources.close()
        finally:
            cv2.destroyAllWindows()
            if hardware:
                print("Joint streaming stopped; disconnect attempted. No automatic disable or emergency stop.")


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="episode directory or trajectory .npz")
    parser.add_argument("--depth-max-m", type=positive_float, default=2.0,
                        help="fixed depth color range in meters (default: 2)")
    parser.add_argument("--speed", type=positive_float, default=1.0, help="playback speed (default: 1)")
    parser.add_argument("--config", default="configs/quest3_nero_l20.yaml", help="robot connection/start settings")
    parser.add_argument("--control", choices=("arm", "hand", "both"), default="arm",
                        help="hardware to replay (default: arm)")
    parser.add_argument("--execute", action="store_true", help="physically replay recorded robot feedback")
    args = parser.parse_args()
    try:
        with ExitStack() as resources:
            episode = Episode(args.episode, resources)
            print(json.dumps(episode.metadata, ensure_ascii=False, indent=2))
            print(f"Loaded {episode.name}: {len(episode.times)} samples, {episode.times[-1]:.3f} s")
            arm_config = hand_config = None
            if args.execute:
                if args.speed > 1:
                    parser.error("Hardware replay supports --speed <= 1 only.")
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from teleop.config import load_config, require_site_configuration

                config = load_config(args.config)
                require_site_configuration(config, control=args.control, require_calibration=False)
                print(f"Selected hardware: {args.control}. It will move to episode sample 0; clear the path/hand.")
                if input("Type EXECUTE to allow robot motion: ").strip() != "EXECUTE":
                    parser.exit(1, "Confirmation did not match; no hardware commands sent.\n")
                if args.control in ("arm", "both"):
                    arm_config = config["arm"]
                if args.control in ("hand", "both"):
                    hand_config = config["hand"]
            replay(episode, args.depth_max_m, args.speed, arm_config=arm_config, hand_config=hand_config)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
