"""Episode trajectories and bounded, asynchronous RGB-D storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any

import numpy as np

from .realsense_camera import RGBDFrame


class _RGBDWriter:
    """One worker owns both files, so video and depth share the same frame index."""

    def __init__(self, directory: Path, width: int, height: int, fps: float) -> None:
        import cv2
        import h5py

        directory.mkdir()
        self._cv2 = cv2
        self._video = cv2.VideoWriter(
            str(directory / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        if not self._video.isOpened():
            self._video.release()
            raise RuntimeError("Cannot open RGB video writer (MP4/mp4v).")
        self._depth_file = None
        try:
            self._depth_file = h5py.File(directory / "depth.h5", "w")
            self._depth = self._depth_file.create_dataset(
                "depth", shape=(0, height, width), maxshape=(None, height, width),
                dtype="uint16", chunks=(1, height, width), compression="lzf",
            )
        except BaseException:
            self._video.release()
            if self._depth_file is not None:
                self._depth_file.close()
            raise
        self._shape = (height, width)
        self._queue: Queue[RGBDFrame | None] = Queue(maxsize=16)
        self.error: Exception | None = None
        self.submitted = 0
        self.written = 0
        self._thread = Thread(target=self._write, daemon=True, name="rgbd-writer")
        self._thread.start()

    def append(self, frame: RGBDFrame) -> int:
        if self.error is not None:
            raise RuntimeError("RGB-D writer failed.") from self.error
        if (frame.color.shape != (*self._shape, 3) or frame.color.dtype != np.uint8
                or frame.depth.shape != self._shape or frame.depth.dtype != np.uint16):
            raise ValueError("Expected configured-size RGB uint8 and aligned depth uint16.")
        try:
            self._queue.put_nowait(frame)
        except Full as exc:
            raise RuntimeError("RGB-D disk writer cannot keep up; recording queue is full.") from exc
        index = self.submitted
        self.submitted += 1
        return index

    def _write(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                # Drain pending work after an error so close() can always send its sentinel.
                if self.error is not None:
                    continue
                try:
                    self._depth.resize(self.written + 1, axis=0)
                    self._depth[self.written] = frame.depth
                    self._video.write(self._cv2.cvtColor(frame.color, self._cv2.COLOR_RGB2BGR))
                    self.written += 1
                except Exception as exc:
                    self.error = exc
        finally:
            for finish in (
                self._video.release,
                lambda: self._depth.resize(self.written, axis=0),
                self._depth_file.close,
            ):
                try:
                    finish()
                except Exception as exc:
                    if self.error is None:
                        self.error = exc

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


class EpisodeRecorder:
    def __init__(
        self, output_dir: str | Path, metadata: dict[str, Any], *, sample_hz: float = 20.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata)
        self.sample_hz = sample_hz
        self.active = False
        self._rows: list[dict[str, Any]] = []
        self._directory: Path | None = None
        self._writer: _RGBDWriter | None = None
        self._stamp = ""

    def start(self) -> None:
        if self.active:
            return  # A repeated B must not discard the current episode.
        self._stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self._directory = self.output_dir / f"episode_{self._stamp}"
        self._directory.mkdir(parents=True)
        camera = self.metadata.get("camera")
        if camera is not None:
            self._writer = _RGBDWriter(
                self._directory / "camera", camera["width"], camera["height"], self.sample_hz,
            )
        self._rows.clear()
        self.active = True

    def append(self, *, camera_frame: RGBDFrame | None = None, **row: Any) -> None:
        if not self.active:
            return
        if self._writer is not None:
            index = self._writer.append(camera_frame) if camera_frame is not None else -1
            row.update(
                camera_frame_index=index,
                camera_valid=camera_frame is not None,
                camera_timestamp_monotonic=camera_frame.received_at if camera_frame else np.nan,
                camera_color_timestamp_ms=camera_frame.color_timestamp_ms if camera_frame else np.nan,
                camera_depth_timestamp_ms=camera_frame.depth_timestamp_ms if camera_frame else np.nan,
                camera_color_frame_number=camera_frame.color_frame_number if camera_frame else -1,
                camera_depth_frame_number=camera_frame.depth_frame_number if camera_frame else -1,
                camera_color_timestamp_domain=camera_frame.color_timestamp_domain if camera_frame else "",
                camera_depth_timestamp_domain=camera_frame.depth_timestamp_domain if camera_frame else "",
            )
        self._rows.append(row)

    def stop_and_save(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
            for row in self._rows:
                if row["camera_frame_index"] >= writer.written:
                    row["camera_frame_index"] = -1
                    row["camera_valid"] = False
        keys = self._rows[0].keys() if self._rows else ()
        arrays = {key: np.asarray([row[key] for row in self._rows]) for key in keys}
        path = self._directory / "trajectory.npz"
        np.savez_compressed(path, **arrays)
        metadata = {
            **self.metadata,
            "schema_version": 2,
            "created_at_utc": self._stamp,
            "sample_hz": self.sample_hz,
            "samples": len(self._rows),
            "fields": {key: list(value.shape) for key, value in arrays.items()},
        }
        if writer is not None:
            metadata["camera"] = {
                **metadata["camera"],
                "color_file": "camera/color.mp4", "color_codec": "mp4v",
                "video_fps": self.sample_hz,
                "depth_file": "camera/depth.h5", "depth_dataset": "depth",
                "frames": writer.written,
                "write_error": str(writer.error) if writer.error else None,
            }
        (self._directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._rows.clear()
        if writer is not None and writer.error is not None:
            raise RuntimeError(f"RGB-D write failed; partial episode saved to {path}.") from writer.error
        return path
