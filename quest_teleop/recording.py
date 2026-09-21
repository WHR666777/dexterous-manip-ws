"""Self-contained episode recorder with opt-in filesystem creation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import numpy as np


class NullRecorder:
    recording = False

    def start(self): return False
    def append(self, sample, camera_frames=()): return None
    def stop(self, reason="operator"): return False
    def close(self): return None


def _git_commit(path: Path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


class EpisodeRecorder:
    """Buffer one episode and write it atomically enough for interrupted experiments.

    Zarr and OpenCV are lazy dependencies.  Constructing this object has no
    filesystem side effect; directories are created by :meth:`start` only.
    """

    def __init__(self, output, metadata, camera_metadata=None, video_fps=30):
        self.output = Path(output)
        self.metadata = dict(metadata)
        self.camera_metadata = camera_metadata
        self.video_fps = float(video_fps)
        self.recording = False
        self.samples: list[dict] = []
        self._root = self._episode = None
        self._episode_number = None
        self._video_writer = None
        self._camera_count = 0
        self._camera_host_times: list[int] = []
        self.started_monotonic = None
        self.started_unix = None

    def start(self):
        if self.recording:
            return False
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("zarr is required only when --record is enabled") from exc
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "videos").mkdir(exist_ok=True)
        self._root = zarr.open_group(str(self.output / "dataset.zarr"), mode="a")
        self._episode = None
        self._episode_number = None
        self._video_writer = None
        self._camera_count = 0
        self._camera_host_times = []
        self.samples = []
        self.started_monotonic, self.started_unix = time.monotonic(), time.time()
        self.recording = True
        return True

    def _ensure_episode(self):
        if self._episode is not None:
            return
        if self._root is None:
            return  # Used only by dependency-free append unit tests.
        self._episode_number = self._episode_index(self._root)
        self._episode = self._root.require_group(f"episodes/{self._episode_number:06d}")
        self._episode.attrs.update({
            "complete": False,
            "started_monotonic": self.started_monotonic,
            "started_unix": self.started_unix,
        })

    def append(self, sample: dict, camera_frames=()):
        if not self.recording:
            return
        normalized = {}
        for key, value in sample.items():
            array = np.asarray(value)
            if array.dtype == object:
                raise ValueError(f"record sample {key} has object dtype")
            normalized[str(key)] = array.copy()
        self._ensure_episode()
        self._append_camera_frames(list(camera_frames))
        if self.camera_metadata is not None:
            sample_time = int(normalized["monotonic_ns"])
            # Store an episode-local MP4/depth index, never the capture thread's
            # process-global frame index.  Only past frames are causally valid.
            eligible = [
                index for index, timestamp in enumerate(self._camera_host_times)
                if timestamp <= sample_time
            ]
            normalized["camera/frame_index"] = np.asarray(eligible[-1] if eligible else -1, dtype=np.int64)
        self.samples.append(normalized)

    def _episode_index(self, root):
        group = root.require_group("episodes")
        used = [int(name) for name in group.group_keys() if str(name).isdigit()]
        return max(used, default=-1) + 1

    def stop(self, reason="operator"):
        if not self.recording:
            return False
        self.recording = False
        if not self.samples:
            self._reset_episode_state()
            return False
        root, episode = self._root, self._episode
        if root is None or episode is None:
            raise RuntimeError("recorder episode was not initialized")
        try:
            keys = set(self.samples[0])
            if any(set(sample) != keys for sample in self.samples):
                raise ValueError("episode samples do not share one schema")
            for key in sorted(keys):
                values = np.stack([sample[key] for sample in self.samples])
                episode.create_dataset(key, data=values, overwrite=False)
            self._close_video_writer()
            root.attrs["schema_version"] = 1
            root.attrs["metadata_json"] = json.dumps(self.metadata, ensure_ascii=False)
            root.attrs["git_commit"] = _git_commit(Path(__file__).resolve().parents[1])
            episode.attrs.update({
                "reason": str(reason),
                "complete": True,
                "started_monotonic": self.started_monotonic,
                "started_unix": self.started_unix,
                "stopped_monotonic": time.monotonic(),
                "stopped_unix": time.time(),
                "monotonic_to_unix_offset": self.started_unix - self.started_monotonic,
            })
        except Exception as exc:
            self._close_video_writer()
            episode.attrs.update({"complete": False, "reason": f"write_error: {exc}"})
            self._reset_episode_state()
            raise
        self._reset_episode_state()
        return True

    def _append_camera_frames(self, frames):
        if not frames:
            return
        start = self._camera_count
        stop = start + len(frames)
        self._camera_host_times.extend(int(frame.host_monotonic_ns) for frame in frames)
        self._camera_count = stop
        if self._episode is None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required to record color video") from exc
        if self._video_writer is None:
            height, width = frames[0].color_bgr.shape[:2]
            video_path = self.output / "videos" / f"episode_{self._episode_number:06d}_realsense.mp4"
            self._video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, (width, height)
            )
            if not self._video_writer.isOpened():
                self._video_writer = None
                raise RuntimeError(f"failed to open video writer: {video_path}")
            camera = self._episode.require_group("camera")
            camera.attrs["video"] = str(video_path.relative_to(self.output))
            camera.attrs["metadata_json"] = json.dumps(self.camera_metadata or {}, ensure_ascii=False)
        for frame in frames:
            self._video_writer.write(frame.color_bgr)
        camera = self._episode.require_group("camera")
        batches = {
            "depth_u16": np.stack([f.depth_u16 for f in frames]),
            "host_monotonic_ns": np.asarray([f.host_monotonic_ns for f in frames], dtype=np.int64),
            "host_unix_ns": np.asarray([f.host_unix_ns for f in frames], dtype=np.int64),
            "device_timestamp_ms": np.asarray([f.device_timestamp_ms for f in frames], dtype=np.float64),
        }
        for name, values in batches.items():
            if name not in camera:
                camera.create_dataset(
                    name, shape=(0, *values.shape[1:]),
                    chunks=(max(1, min(64, len(values))), *values.shape[1:]),
                    dtype=values.dtype,
                )
            dataset = camera[name]
            dataset.resize((stop, *values.shape[1:]))
            dataset[start:stop] = values

    def _close_video_writer(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _reset_episode_state(self):
        self._close_video_writer()
        self.samples = []
        self._root = self._episode = None
        self._episode_number = None
        self._camera_count = 0
        self._camera_host_times = []

    def abort(self, reason="recording_error"):
        """Close streaming resources and retain an explicitly incomplete episode."""
        was_active = self.recording or self._episode is not None
        self.recording = False
        self._close_video_writer()
        if self._episode is not None:
            self._episode.attrs.update({"complete": False, "reason": str(reason)})
        self._reset_episode_state()
        return was_active

    def close(self):
        self.stop("close")
