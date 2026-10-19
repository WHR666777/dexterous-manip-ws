"""RGB-D/FK sample capture and offline manual contact-point annotation."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from cloth_agent.geometry import calculate
from robot_control.tcp_calibration import tcp_rotation_from_reference


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class CalibrationSession:
    def __init__(self, directory, config, hand_pose_raw, hand_frame, camera_metadata):
        reference_rp = np.asarray(config["arm"].get("wrist_only_roll_pitch_deg"), dtype=float)
        if reference_rp.shape != (2,) or not np.isfinite(reference_rp).all():
            raise ValueError("arm.wrist_only_roll_pitch_deg must contain two finite reference angles.")
        reference = np.eye(4)
        reference[:3, :3] = Rotation.from_euler("xyz", [*reference_rp, 0], degrees=True).as_matrix()
        rotation = tcp_rotation_from_reference(reference)
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "samples").mkdir()
        self.count = 0
        self.metadata = {
            "version": 1,
            "config": config["_path"],
            "camera": camera_metadata,
            "hand_side": config["hand"]["type"],
            "hand_replay_clip": config["hand"]["replay_clip_file"],
            "hand_replay_frame": int(hand_frame),
            "hand_pose_raw": np.asarray(hand_pose_raw, dtype=int).tolist(),
            "reference_roll_pitch_deg": reference_rp.tolist(),
            "reference_base_from_flange": reference.tolist(),
            "flange_from_tcp_rotation": rotation.tolist(),
            "tcp_axes": "At reference: Z=base down, X=horizontal projection of flange +Z, Y=Z cross X",
            "base_vertical_assumption": "Nero base +Z is physically upward",
            "fk_model": "assets/nero/nero_description.urdf: base_link -> link7",
            "timing": "Host receipt/query times; samples accepted only while joint feedback is stationary",
        }
        write_json(self.directory / "session.json", self.metadata)

    def capture(self, arm, ik, camera, hand_feedback, *, stable_s=.4,
                tolerance_rad=.002, timeout_s=5.) -> Path:
        """Caller stops issuing motion targets first. This method never commands devices.

        Keep joint feedback within one small range for a stationary interval. Accept
        only an RGB-D frame received within it, bracketed by consistent feedback.
        This is a static-pose association, not hardware synchronization.
        """
        deadline = time.monotonic() + timeout_s
        anchor = None
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            before_at = time.monotonic()
            before = np.asarray(arm.get_joint_positions(), dtype=float)
            if before.shape != (7,) or not np.isfinite(before).all():
                raise RuntimeError("Incomplete seven-joint feedback; sample was not saved.")
            if anchor is None or np.max(np.abs(before - anchor)) > tolerance_rad:
                anchor, stable_since = before.copy(), before_at
            frame = camera.get_latest()
            if before_at - stable_since >= stable_s and frame is not None and frame.received_at >= stable_since:
                after = np.asarray(arm.get_joint_positions(), dtype=float)
                after_at = time.monotonic()
                if (after.shape != (7,) or not np.isfinite(after).all()
                        or np.max(np.ptp(np.stack((anchor, before, after)), axis=0)) > tolerance_rad):
                    anchor = None
                    continue
                feedback = hand_feedback.get_latest()
                joints = (before + after) / 2
                frame_metadata = asdict(frame)
                frame_metadata.pop("color")
                frame_metadata.pop("depth")
                path = self.directory / "samples" / f"sample-{self.count + 1:04d}.npz"
                np.savez_compressed(
                    path, color=frame.color, depth=frame.depth,
                    arm_joint_position_rad=joints,
                    base_from_flange=ik.forward(joints),
                    joint_feedback_before=before, joint_feedback_after=after,
                    joint_query_before_at=before_at, joint_query_after_at=after_at,
                    hand_position_raw=feedback.positions_raw,
                    hand_query_finished_at=feedback.query_finished_at,
                    frame_metadata_json=json.dumps(frame_metadata),
                )
                self.count += 1
                return path
            time.sleep(.02)
        raise TimeoutError("No stationary, fresh RGB-D/FK sample within 5 s; hold still and press C again.")


def point_from_pixel(depth, pixel, camera_metadata) -> np.ndarray:
    """Use the exact clicked depth pixel; never fill holes from a different surface."""
    u, v = pixel
    if int(u) != u or int(v) != v or not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("Click must be an integer pixel inside the saved image.")
    z = float(depth[int(v), int(u)]) * float(camera_metadata["depth_scale_m"])
    return np.asarray(calculate("pixel_to_point", pixel=[u, v], depth_m=z,
                                intrinsics=camera_metadata["color_intrinsics"]))


def read_session(directory):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    samples = sorted((directory / "samples").glob("sample-*.npz"))
    if not samples:
        raise ValueError("No captured samples in this session.")
    return directory, metadata, samples


def load_observations(directory):
    """Recompute 3D observations from saved pixels/depth, retaining sample names."""
    directory, metadata, samples = read_session(directory)
    annotations = json.loads((directory / "annotations.json").read_text(encoding="utf-8"))
    names, points, poses = [], [], []
    for path in samples:
        if path.name not in annotations:
            continue
        with np.load(path, allow_pickle=False) as sample:
            points.append(point_from_pixel(sample["depth"], annotations[path.name]["pixel_uv"], metadata["camera"]))
            poses.append(sample["base_from_flange"].copy())
        names.append(path.name)
    return metadata, names, np.asarray(points).reshape(-1, 3), np.asarray(poses)


def annotate(directory, scale=1.) -> None:
    """Offline OpenCV GUI. Enter/N commits; S skips; A goes back; D deletes; Q exits."""
    import cv2

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Display scale must be finite and positive.")
    directory, metadata, samples = read_session(directory)
    destination = directory / "annotations.json"
    annotations = json.loads(destination.read_text()) if destination.exists() else {}
    window = "TCP contact point (RGB-D)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    index = 0
    print("Left click the same closed-finger contact point. Enter/N=save+next, S=skip, A=back, D=delete, Q=quit.")
    try:
        while index < len(samples):
            path = samples[index]
            with np.load(path, allow_pickle=False) as sample:
                rgb, depth = sample["color"].copy(), sample["depth"].copy()
            selection = [annotations.get(path.name)]

            def clicked(event, x, y, _flags, _param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                pixel = [int(x / scale), int(y / scale)]
                try:
                    point = point_from_pixel(depth, pixel, metadata["camera"])
                except ValueError as exc:
                    print(f"{path.name}: {exc}")
                    selection[0] = None
                    return
                selection[0] = {"pixel_uv": pixel, "point_camera_m": point.tolist()}
                print(f"{path.name}: pixel={pixel}, camera XYZ (m)={point.round(6).tolist()}")

            cv2.setMouseCallback(window, clicked)
            while True:
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if selection[0] is not None:
                    cv2.drawMarker(display, tuple(selection[0]["pixel_uv"]), (0, 255, 0),
                                   cv2.MARKER_CROSS, 14, 1)
                display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                cv2.putText(display, f"{index + 1}/{len(samples)} {path.name} | Enter=save S=skip A=back Q=quit",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xff
                if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key in (10, 13, ord("n")) and selection[0] is not None:
                    annotations[path.name] = selection[0]
                    write_json(destination, annotations)
                    index += 1
                    break
                if key == ord("d"):
                    selection[0] = None
                    annotations.pop(path.name, None)
                    write_json(destination, annotations)
                if key in (ord("s"), ord("a")):
                    index = index + 1 if key == ord("s") else max(0, index - 1)
                    break
    finally:
        cv2.destroyWindow(window)
