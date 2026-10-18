"""Save a newly staged joint demonstration using LeRobot 0.4.4 (act environment).

This module never opens hardware. RGB is encoded once; aligned Z16 depth remains
in an HDF5 sidecar. Existing legacy episodes are not searched or converted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

from robot_control.l20 import L20_ACTIVE_POSITION_INDICES, L20_JOINT_NAMES


JOINT_NAMES = [f"nero.joint{i}" for i in range(1, 8)] + [
    f"l20.{L20_JOINT_NAMES[i]}" for i in L20_ACTIVE_POSITION_INDICES
]
ROBOT_TYPE = "nero_l20"
IMAGE_KEY = "observation.images.front"


def dataset_features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (23,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (23,), "names": JOINT_NAMES},
        IMAGE_KEY: {"dtype": "video", "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"]},
    }


def joint_vectors(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Use pre-command feedback and the final targets, in identical channel order."""
    arrays = {}
    for key, width in (("observation_arm_joint_position", 7),
                       ("observation_hand_position_raw", 20),
                       ("arm_action", 7), ("hand_action_raw", 20)):
        if key not in data:
            raise ValueError(f"Missing pre-command observation/action field: {key}")
        value = np.asarray(data[key])
        if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or non-finite values in {key}.")
        if width == 20 and (np.any(value < 0) or np.any(value > 255)
                            or np.any(value != np.floor(value))):
            raise ValueError(f"{key} must contain integer SDK values in 0..255.")
        arrays[key] = value
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("Joint arrays must have the same nonzero sample count.")
    active = list(L20_ACTIVE_POSITION_INDICES)
    state = np.concatenate((arrays["observation_arm_joint_position"],
                            arrays["observation_hand_position_raw"][:, active]), axis=1)
    action = np.concatenate((arrays["arm_action"], arrays["hand_action_raw"][:, active]), axis=1)
    return state.astype(np.float32), action.astype(np.float32)


def validate_staging(staging: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    import h5py
    from PIL import Image

    metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
    with np.load(staging / "trajectory.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if metadata.get("control") != "both" or metadata.get("recording_error"):
        raise ValueError("A complete, error-free --control both recording is required.")
    state, action = joint_vectors(data)
    count = len(state)
    if metadata["samples"] != count or any(len(v) != count for v in data.values()):
        raise ValueError("Trajectory sample counts do not agree.")
    for key in ("timestamp_monotonic", "observation_timestamp_monotonic"):
        times = data[key]
        if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{key} must contain strictly increasing host timestamps.")
    if (data["camera_valid"].shape != (count,) or not data["camera_valid"].all()
            or not np.array_equal(data["camera_frame_index"], np.arange(count))):
        raise ValueError("Missing RGB-D samples; the episode cannot be exported without dropping/faking rows.")
    camera = metadata["camera"]
    if ((camera["width"], camera["height"]) != (640, 480)
            or camera.get("depth_aligned_to") != "color" or camera.get("write_error")
            or camera["frames"] != count):
        raise ValueError("Expected complete, aligned 640x480 RGB-D frames.")
    scale = camera["depth_scale_m"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_m must be finite and positive.")
    with h5py.File(staging / "camera/depth.h5", "r") as depth:
        if depth["depth"].shape != (count, 480, 640) or depth["depth"].dtype != np.uint16:
            raise ValueError("Depth must be lossless uint16 with one aligned frame per sample.")
    for i in range(count):
        with Image.open(staging / "camera/color" / f"frame-{i:06d}.png") as image:
            if image.size != (640, 480) or image.mode != "RGB":
                raise ValueError(f"Invalid RGB frame {i}.")
            image.verify()
    fps = metadata["sample_hz"]
    if not np.isfinite(fps) or fps <= 0 or fps != int(fps):
        raise ValueError("LeRobot recording fps must be a positive integer.")
    for key in ("repo_id", "task"):
        if not isinstance(metadata["lerobot"].get(key), str) or not metadata["lerobot"][key].strip():
            raise ValueError(f"A nonempty recording.lerobot.{key} is required.")
    return metadata, data, state, action


@contextmanager
def dataset_lock(root: Path):
    """Serialize save processes, including recordings started from another terminal."""
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another recorder is saving to {root}.") from exc
        yield


def recording_identity(metadata: dict) -> dict:
    return {"repo_id": metadata["lerobot"]["repo_id"],
            "hand_side": metadata.get("input_side", metadata.get("quest_side")),
            "joint_names": JOINT_NAMES,
            "l20_active_position_indices": list(L20_ACTIVE_POSITION_INDICES),
            "units": {"nero": "rad", "l20": "raw_0_255"}}


def check_existing(root: Path, metadata: dict) -> None:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if (info.get("codebase_version") != "v3.0" or info.get("fps") != metadata["sample_hz"]
            or info.get("robot_type") != ROBOT_TYPE):
        raise ValueError("Existing dataset version, fps or robot_type differs; use a new output directory.")
    default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if set(info["features"]) != set(dataset_features()) | default_keys:
        raise ValueError("Existing dataset feature keys differ.")
    for key, expected in dataset_features().items():
        actual = info["features"].get(key, {})
        if (actual.get("dtype") != expected["dtype"]
                or tuple(actual.get("shape", ())) != expected["shape"]
                or actual.get("names") != expected["names"]):
            raise ValueError(f"Existing dataset feature differs: {key}.")
    identity = json.loads((root / "meta/recording.json").read_text(encoding="utf-8"))
    if identity != recording_identity(metadata):
        raise ValueError("Existing dataset recording identity/hand side differs.")
    if info.get("total_episodes", 0) < 1:
        raise ValueError("Existing dataset has no complete episode; use a new output directory.")


def export_episode(staging: Path, root: Path) -> dict:
    # Validate source files before importing the training stack or touching the dataset.
    metadata, data, state, action = validate_staging(staging)
    if version("lerobot") != "0.4.4":
        raise RuntimeError("The save process requires the pinned LeRobot 0.4.4 environment.")
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = int(metadata["sample_hz"])
    repo_id = metadata["lerobot"]["repo_id"]
    marker = root / ".recording-in-progress.json"
    with dataset_lock(root):
        if marker.exists():
            raise RuntimeError(f"An earlier save was interrupted. Inspect {marker}; staged sources are retained.")
        if root.exists():
            check_existing(root, metadata)
            dataset = LeRobotDataset(repo_id=repo_id, root=root, vcodec="h264", encoder_threads=2)
        else:
            dataset = LeRobotDataset.create(
                repo_id=repo_id, root=root, fps=fps, features=dataset_features(),
                robot_type=ROBOT_TYPE, vcodec="h264", metadata_buffer_size=1, encoder_threads=2,
            )
            (root / "meta/recording.json").write_text(
                json.dumps(recording_identity(metadata), ensure_ascii=False, indent=2), encoding="utf-8",
            )
        episode_index = dataset.meta.total_episodes
        sidecar_name = f"episode-{episode_index:06d}"
        marker.write_text(json.dumps({"episode_index": episode_index, "staging": str(staging)}),
                          encoding="utf-8")
        try:
            for i in range(len(state)):
                with Image.open(staging / "camera/color" / f"frame-{i:06d}.png") as image:
                    rgb = np.array(image)
                dataset.add_frame({"observation.state": state[i], "action": action[i],
                                   IMAGE_KEY: rgb, "task": metadata["lerobot"]["task"]})
            dataset.save_episode(parallel_encoding=False)
        finally:
            dataset.finalize()

        # Keep full camera calibration and original host/device timestamps, outside
        # policy features so ACT does not mistake diagnostics for extra inputs.
        (root / "depth").mkdir(exist_ok=True)
        (root / "diagnostics").mkdir(exist_ok=True)
        shutil.copyfile(staging / "camera/depth.h5", root / "depth" / f"{sidecar_name}.h5")
        shutil.copyfile(staging / "trajectory.npz", root / "diagnostics" / f"{sidecar_name}.npz")
        times = data["observation_timestamp_monotonic"]
        intervals = np.diff(times)
        actual_hz = float((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 else 0.0
        metadata.update(
            episode_index=episode_index, joint_names=JOINT_NAMES,
            l20_active_position_indices=list(L20_ACTIVE_POSITION_INDICES),
            timestamp_semantics="LeRobot: frame_index/fps; diagnostics: original host/device clocks",
            actual_hz=actual_hz,
            sample_interval_s={"median": float(np.median(intervals)) if len(intervals) else None,
                               "max": float(intervals.max()) if len(intervals) else None},
        )
        metadata["camera"].update(color_file=IMAGE_KEY,
                                  color_codec="h264",
                                  depth_file=f"depth/{sidecar_name}.h5",
                                  depth_frame_index="LeRobot frame_index within episode")
        (root / "diagnostics" / f"{sidecar_name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        # Verify the official dataset can reopen and decode both ends before
        # acknowledging success or removing the caller's staged source files.
        loaded = LeRobotDataset(repo_id=repo_id, root=root)
        episode = loaded.meta.episodes[episode_index]
        for index in (episode["dataset_from_index"], episode["dataset_to_index"] - 1):
            sample = loaded[index]
            if tuple(sample[IMAGE_KEY].shape) != (3, 480, 640):
                raise RuntimeError("Saved RGB could not be decoded at the expected shape.")
        result = {"episode_index": episode_index, "samples": len(state), "actual_hz": actual_hz}
        (staging / "export_result.json").write_text(json.dumps(result), encoding="utf-8")
        marker.unlink()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_episode(args.staging.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
