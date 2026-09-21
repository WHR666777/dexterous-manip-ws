"""Optional RealSense RGB-D capture; imported only when recording is enabled."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    index: int
    color_bgr: np.ndarray
    depth_u16: np.ndarray
    host_monotonic_ns: int
    host_unix_ns: int
    device_timestamp_ms: float


class RealSenseCapture:
    def __init__(self, serial=None, width=640, height=480, fps=30, queue_size=256):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("pyrealsense2 is required only when --record is enabled") from exc
        self.rs = rs
        self.serial = serial
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self._queue = deque(maxlen=int(queue_size))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error = None
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.metadata = {
            "serial": profile.get_device().get_info(rs.camera_info.serial_number),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "color_intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
                "coeffs": list(intrinsics.coeffs), "model": str(intrinsics.model),
            },
            "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
        }
        self._next_index = 0
        self._thread = threading.Thread(target=self._run, name="realsense-capture", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                frames = self._align.process(self._pipeline.wait_for_frames(1000))
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                frame = RGBDFrame(
                    index=self._next_index,
                    color_bgr=np.asanyarray(color.get_data()).copy(),
                    depth_u16=np.asanyarray(depth.get_data()).copy(),
                    host_monotonic_ns=time.monotonic_ns(),
                    host_unix_ns=time.time_ns(),
                    device_timestamp_ms=float(color.get_timestamp()),
                )
                self._next_index += 1
                with self._lock:
                    self._queue.append(frame)
            except Exception as exc:
                if not self._stop.is_set():
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._stop.set()
                break

    def drain(self) -> list[RGBDFrame]:
        with self._lock:
            frames = list(self._queue)
            self._queue.clear()
        return frames

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._pipeline.stop()
