"""Snapshot inputs. Keypoint extraction remains an external module."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Protocol
import json
import struct
import time
import zlib

import numpy as np

from robot_control.nero_ik import NeroIK
from .config import effective_arm
from .geometry import transform


class KeypointProvider(Protocol):
    def latest(self) -> dict | None: ...


class FileKeypointProvider:
    """Consume an upstream atomic JSON publication; does not detect or track."""

    def __init__(self, path, max_age_s=.25):
        self.path = Path(path) if path else None
        self.max_age_s = max_age_s

    def latest(self):
        if self.path is None or not self.path.exists():
            return None
        value = json.loads(self.path.read_text())
        age = time.monotonic() - float(value["timestamp_monotonic"])
        if not np.isfinite(age) or age < 0 or age > self.max_age_s:
            return None
        if value["frame"] != "task":
            raise ValueError("Keypoints must be expressed in the task frame.")
        for point in value["points"].values():
            xyz = np.asarray(point, dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError("Invalid external keypoint coordinates.")
        return value


def _mock_png():
    """Small synthetic RGB image, without a camera/OpenCV dependency."""
    width, height = 64, 48
    rows = []
    for y in range(height):
        pixels = b"".join(bytes((220, 220, 210) if 12 < x < 52 and 10 < y < 38
                                else (70, 80, 90)) for x in range(width))
        rows.append(b"\x00" + pixels)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


class Observations:
    def __init__(self, config, backend, output_dir, keypoints=None):
        self.config, self.backend = config, backend
        self.directory = Path(output_dir) / "observations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keypoints = keypoints or FileKeypointProvider(
            config["observations"]["keypoints_file"],
            config["observations"]["keypoints_max_age_s"])
        self._lock = Lock()
        self._fk = {name: NeroIK() for name in config["arms"]}
        self.camera = None
        self.recording = config["observations"]["recorded_episode"]
        self._video, self._depth = None, None
        try:
            if self.recording:
                self._open_recording(Path(self.recording))
            elif config["backend"] == "hardware" and config["camera"]["enabled"]:
                from teleop.realsense_camera import RealSenseCamera
                self.camera = RealSenseCamera(config["camera"])
                self.camera.start()
        except BaseException:
            self.close()
            raise
        self._frame_key = None
        self._image_info = None

    def _open_recording(self, directory):
        import cv2
        import h5py

        metadata = json.loads((directory / "metadata.json").read_text())
        camera = metadata["camera"]
        self._record_metadata = camera
        self._video = cv2.VideoCapture(str(directory / camera["color_file"]))
        if not self._video.isOpened():
            raise ValueError("Could not open recorded RGB video.")
        self._depth = h5py.File(directory / camera["depth_file"], "r")
        with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
            self._record_times = data["timestamp_monotonic"].copy()
            self._record_frames = data["camera_frame_index"].copy()
            self._record_valid = data["camera_valid"].copy()
        if (not len(self._record_times) or not np.isfinite(self._record_times).all()
                or np.any(np.diff(self._record_times) < 0)
                or self._record_frames.shape != self._record_times.shape
                or self._record_valid.shape != self._record_times.shape):
            raise ValueError("Invalid recorded observation timeline.")
        self._record_started = time.monotonic()

    def _images(self):
        if self.recording:
            import cv2
            elapsed = time.monotonic() - self._record_started
            row = min(len(self._record_times) - 1, int(np.searchsorted(
                self._record_times - self._record_times[0], elapsed)))
            index = int(self._record_frames[row])
            if not self._record_valid[row] or index < 0:
                return {"available": False, "source": "recording"}
            key = ("recording", index)
            if self._frame_key == key:
                return self._image_info.copy()
            self._video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self._video.read()
            if not ok:
                raise RuntimeError("Recorded RGB frame is unavailable.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dataset = self._depth[self._record_metadata.get("depth_dataset", "depth")]
            if index >= len(dataset):
                raise ValueError("Recorded depth frame is out of range.")
            depth = dataset[index]
            metadata = self._record_metadata
        elif self.camera:
            frame = self.camera.get_latest()
            if frame is None:
                return {"available": False, "source": "camera"}
            key = ("camera", frame.color_frame_number, frame.depth_frame_number)
            if self._frame_key == key:
                return self._image_info.copy()
            rgb, depth, metadata = frame.color, frame.depth, self.camera.metadata
        elif self.config["backend"] == "mock":
            key = ("synthetic", 0)
            if self._frame_key == key:
                return self._image_info.copy()
            rgb, depth = None, np.full((48, 64), 1000, dtype=np.uint16)
            metadata = {"depth_scale_m": .001, "color_intrinsics":
                        {"fx": 60., "fy": 60., "ppx": 31.5, "ppy": 23.5}}
        else:
            return {"available": False, "source": "camera_disabled"}
        suffix = "-".join(str(part) for part in key)
        color_path = self.directory / f"{suffix}.png"
        depth_path = self.directory / f"{suffix}-depth.npy"
        if rgb is None:
            color_path.write_bytes(_mock_png())
        else:
            import cv2
            if not cv2.imwrite(str(color_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("Could not save RGB snapshot.")
        np.save(depth_path, depth, allow_pickle=False)
        self._frame_key = key
        self._image_info = {"available": True, "source": key[0], "rgb_path": str(color_path),
                            "depth_path": str(depth_path), "metadata": metadata}
        return self._image_info.copy()

    def snapshot(self, include_images=True):
        with self._lock:
            arms = {}
            for arm in self.config["arms"]:
                q = self.backend.get_joints(arm)
                settings = effective_arm(self.config, arm)
                tcp = (transform(settings["task_from_base"]) @ self._fk[arm].forward(q)
                       @ transform(settings["flange_from_tcp"]))
                try:
                    hand = self.backend.get_hand(arm).tolist()
                    hand_error = None
                except RuntimeError as exc:
                    hand, hand_error = None, str(exc)
                arms[arm] = {"joints": q.tolist(), "tcp": tcp.tolist(),
                             "hand_raw": hand, "hand_error": hand_error}
            return {"timestamp_monotonic": time.monotonic(), "backend": self.config["backend"],
                    "arms": arms, "keypoints": self.keypoints.latest(),
                    "images": self._images() if include_images else None}

    def close(self):
        if self.camera:
            self.camera.stop()
        if self._video:
            self._video.release()
        if self._depth is not None:
            self._depth.close()
