"""Background RGB-D capture; depth is always aligned to the color image."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    color: np.ndarray  # RGB uint8, H x W x 3; owned copy, never modified after publishing
    depth: np.ndarray  # aligned Z16 uint16, H x W
    received_at: float  # host monotonic seconds, before alignment
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_frame_number: int
    depth_frame_number: int
    color_timestamp_domain: str
    depth_timestamp_domain: str


def _intrinsics(profile) -> dict[str, Any]:
    value = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": value.width, "height": value.height,
        "fx": value.fx, "fy": value.fy, "ppx": value.ppx, "ppy": value.ppy,
        "model": str(value.model), "coeffs": list(value.coeffs),
    }


def _output_intrinsics(native: dict[str, Any], width: int, height: int):
    """Center-crop to the output aspect ratio, then use OpenCV's pixel-center mapping."""
    crop_width = min(native["width"], round(native["height"] * width / height))
    crop_height = min(native["height"], round(native["width"] * height / width))
    left = (native["width"] - crop_width) // 2
    top = (native["height"] - crop_height) // 2
    sx, sy = width / crop_width, height / crop_height
    output = {
        **native, "width": width, "height": height,
        "fx": native["fx"] * sx, "fy": native["fy"] * sy,
        "ppx": (native["ppx"] - left + 0.5) * sx - 0.5,
        "ppy": (native["ppy"] - top + 0.5) * sy - 0.5,
    }
    return output, (left, top, crop_width, crop_height)


class RealSenseCamera:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.metadata: dict[str, Any] = {}
        self._pipeline = None
        self._thread: Thread | None = None
        self._stop = Event()
        self._ready = Event()
        self._lock = Lock()
        self._latest: RGBDFrame | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        import cv2
        import pyrealsense2 as rs

        self._cv2 = cv2
        cfg = rs.config()
        if self.config["serial"]:
            cfg.enable_device(self.config["serial"])
        width, height, fps = (self.config[key] for key in ("width", "height", "fps"))
        # L515 has no native VGA color profile. Resize only after spatial alignment.
        cfg.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(cfg)
        self._pipeline = pipeline
        try:
            device = profile.get_device()
            model = device.get_info(rs.camera_info.name)
            if self.config["model"].lower() not in model.lower():
                raise RuntimeError(f"Expected {self.config['model']}, found {model}.")
            native_color = _intrinsics(profile.get_stream(rs.stream.color))
            output_color, self._crop = _output_intrinsics(native_color, width, height)
            self.metadata = {
                "model": model,
                "serial": device.get_info(rs.camera_info.serial_number),
                "width": width, "height": height, "capture_fps": fps,
                "color_format": "RGB8", "depth_format": "Z16",
                "depth_scale_m": device.first_depth_sensor().get_depth_scale(),
                "depth_aligned_to": "color",
                "color_intrinsics": output_color,
                "native_color_intrinsics": native_color,
                "native_depth_intrinsics": _intrinsics(profile.get_stream(rs.stream.depth)),
                "color_crop_xywh": list(self._crop),
                "resize": {"color": "area", "depth": "nearest_exact"},
            }
            align = rs.align(rs.stream.color)
            self._thread = Thread(target=self._capture, args=(align,), daemon=True,
                                  name="realsense-capture")
            self._thread.start()
            if not self._ready.wait(timeout=5.0):
                raise TimeoutError("RealSense did not deliver a complete RGB-D frame within 5 s.")
            if self.get_latest() is None:
                raise RuntimeError("RealSense startup frame is stale.")
        except BaseException:
            self.stop()
            raise

    def _capture(self, align) -> None:
        try:
            while not self._stop.is_set():
                available, frames = self._pipeline.try_wait_for_frames(timeout_ms=1000)
                if not available:
                    continue
                received_at = time.monotonic()
                aligned = align.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    continue
                x, y, width, height = self._crop
                output_size = (self.config["width"], self.config["height"])
                rgb = np.asanyarray(color.get_data())[y:y + height, x:x + width]
                z16 = np.asanyarray(depth.get_data())[y:y + height, x:x + width]
                frame = RGBDFrame(
                    color=self._cv2.resize(rgb, output_size, interpolation=self._cv2.INTER_AREA),
                    depth=self._cv2.resize(z16, output_size, interpolation=self._cv2.INTER_NEAREST_EXACT),
                    received_at=received_at,
                    color_timestamp_ms=color.get_timestamp(),
                    depth_timestamp_ms=depth.get_timestamp(),
                    color_frame_number=color.get_frame_number(),
                    depth_frame_number=depth.get_frame_number(),
                    color_timestamp_domain=str(color.get_frame_timestamp_domain()),
                    depth_timestamp_domain=str(depth.get_frame_timestamp_domain()),
                )
                with self._lock:
                    if self._latest is None:
                        self.metadata["aligned_depth_intrinsics"] = _output_intrinsics(
                            _intrinsics(depth.profile), *output_size,
                        )[0]
                    self._latest = frame
                self._ready.set()
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._ready.set()

    def get_latest(self) -> RGBDFrame | None:
        """Non-blocking snapshot; stale/missing images are not presented as new data."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError("RealSense capture failed.") from self._error
            frame = self._latest
        if frame is None or time.monotonic() - frame.received_at > self.config["max_age_s"]:
            return None
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()  # try_wait_for_frames has a bounded timeout
            self._thread = None
        if self._pipeline is not None:
            pipeline, self._pipeline = self._pipeline, None
            pipeline.stop()
