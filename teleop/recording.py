"""Episode trajectories and bounded, asynchronous RGB-D storage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Full, Queue
import shutil
import subprocess
from threading import Thread
from typing import Any

import numpy as np

from .realsense_camera import RGBDFrame


class _RGBDWriter:
    """One worker owns RGB/depth storage, keeping their frame indices identical."""

    def __init__(
        self, directory: Path, width: int, height: int, fps: float, *, save_png: bool = False,
    ) -> None:
        import cv2
        import h5py

        directory.mkdir()
        self._cv2 = cv2
        self._png_dir = directory / "color" if save_png else None
        self._video = None
        if self._png_dir is not None:
            self._png_dir.mkdir()
        else:
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
            if self._video is not None:
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
                    bgr = self._cv2.cvtColor(frame.color, self._cv2.COLOR_RGB2BGR)
                    if self._png_dir is not None:
                        if not self._cv2.imwrite(
                            str(self._png_dir / f"frame-{self.written:06d}.png"), bgr,
                            [self._cv2.IMWRITE_PNG_COMPRESSION, 1],
                        ):
                            raise OSError("Cannot write staged RGB PNG.")
                    else:
                        self._video.write(bgr)
                    self.written += 1
                except Exception as exc:
                    self.error = exc
        finally:
            for finish in (
                lambda: self._video.release() if self._video is not None else None,
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
        lerobot: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata)
        self.sample_hz = sample_hz
        self.lerobot = dict(lerobot) if lerobot is not None else None
        self.active = False
        self._rows: list[dict[str, Any]] = []
        self._directory: Path | None = None
        self._writer: _RGBDWriter | None = None
        self._stamp = ""
        self._failure: Exception | None = None

    def start(self) -> None:
        if self.active:
            return  # A repeated B must not discard the current episode.
        camera = self.metadata.get("camera")
        if self.lerobot is not None:
            if self.metadata.get("control") != "both" or camera is None:
                raise ValueError("LeRobot joint recording requires --control both and an enabled camera.")
            if (camera["width"], camera["height"]) != (640, 480):
                raise ValueError("LeRobot RGB-D recording requires aligned 640x480 images.")
        self._stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        # Keep staging outside the dataset: LeRobot.create requires a new root.
        staging_root = (self.output_dir.parent / f".{self.output_dir.name}-staging"
                        if self.lerobot is not None else self.output_dir)
        self._directory = staging_root / f"episode_{self._stamp}"
        self._directory.mkdir(parents=True)
        if camera is not None:
            self._writer = _RGBDWriter(
                self._directory / "camera", camera["width"], camera["height"], self.sample_hz,
                save_png=self.lerobot is not None,
            )
        self._rows.clear()
        self._failure = None
        self.active = True

    def append(self, *, camera_frame: RGBDFrame | None = None, **row: Any) -> None:
        if not self.active:
            return
        if self._writer is not None:
            try:
                index = self._writer.append(camera_frame) if camera_frame is not None else -1
            except Exception as exc:
                self._failure = exc
                raise
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
            "recording_error": str(self._failure) if self._failure else None,
        }
        if writer is not None:
            metadata["camera"] = {
                **metadata["camera"],
                "color_file": ("camera/color/frame-{frame_index:06d}.png"
                               if self.lerobot is not None else "camera/color.mp4"),
                "color_codec": "png" if self.lerobot is not None else "mp4v",
                "video_fps": self.sample_hz,
                "depth_file": "camera/depth.h5", "depth_dataset": "depth",
                "frames": writer.written,
                "write_error": str(writer.error) if writer.error else None,
            }
        if self.lerobot is not None:
            metadata["lerobot"] = self.lerobot
        (self._directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._rows.clear()
        failure = self._failure or (writer.error if writer is not None else None)
        if failure is not None:
            raise RuntimeError(f"RGB-D write failed; partial episode saved to {path}.") from failure
        if self.lerobot is not None:
            if not metadata["samples"]:
                shutil.rmtree(self._directory)
                return None
            return self._export_lerobot(self._directory)
        return path

    def _export_lerobot(self, staging: Path) -> Path:
        """Run only the save step in act; never import Torch in the control process."""
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        env["HF_HOME"] = str(staging / ".hf-cache")
        env["HF_DATASETS_CACHE"] = str(staging / ".hf-cache/datasets")
        # The pinned act environment needs system libffi for TorchCodec/FFmpeg 4.
        libffi = self.lerobot.get("libffi_preload", "/usr/lib/x86_64-linux-gnu/libffi.so.7")
        if libffi:
            env["LD_PRELOAD"] = str(libffi) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        command = [
            self.lerobot["python"], "-B", "-m", "teleop.lerobot_export",
            "--staging", str(staging.resolve()), "--output", str(self.output_dir.resolve()),
        ]
        log_path = staging / "export.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"LeRobot save failed; staged data retained at {staging}. See {log_path}."
            ) from exc
        result = json.loads((staging / "export_result.json").read_text(encoding="utf-8"))
        print(f"LeRobot episode {result['episode_index']} saved: {self.output_dir}; "
              f"actual sample rate={result['actual_hz']:.2f} Hz, nominal={self.sample_hz:g} Hz.")
        shutil.rmtree(staging)
        return self.output_dir
