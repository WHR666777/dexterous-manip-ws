"""Export canonical Quest episodes to Diffusion Policy Zarr or ACT HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _episode_names(root):
    if "episodes" not in root:
        return []
    episodes = root["episodes"]
    return [
        name for name in sorted(episodes.group_keys())
        if bool(episodes[name].attrs.get("complete", False))
    ]


def _state(episode, prefix):
    values = []
    for side in ("left", "right"):
        arm = np.asarray(episode[f"{prefix}/{side}/arm_joint"])
        hand = np.asarray(episode[f"{prefix}/{side}/hand_raw"], dtype=np.float64)
        values.extend((arm, hand / 127.5 - 1.0))
    return np.concatenate(values, axis=1).astype(np.float32)


def _aligned_rgb(dataset_dir: Path, episode, episode_name: str):
    if "camera/frame_index" not in episode:
        return None
    import cv2
    path = dataset_dir / "videos" / f"episode_{int(episode_name):06d}_realsense.mp4"
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        return None
    indices = np.asarray(episode["camera/frame_index"], dtype=np.int64)
    result = np.zeros((len(indices), *frames[0].shape), dtype=np.uint8)
    for i, frame_index in enumerate(indices):
        if 0 <= frame_index < len(frames): result[i] = frames[frame_index]
    return result


def _aligned_depth(episode):
    if "camera/frame_index" not in episode or "camera/depth_u16" not in episode:
        return None
    depth = np.asarray(episode["camera/depth_u16"], dtype=np.uint16)
    indices = np.asarray(episode["camera/frame_index"], dtype=np.int64)
    if not len(depth):
        return None
    result = np.zeros((len(indices), *depth.shape[1:]), dtype=np.uint16)
    valid = (indices >= 0) & (indices < len(depth))
    result[valid] = depth[indices[valid]]
    return result


def export_diffusion(source: Path, destination: Path, include_images=True):
    import zarr
    source_root = zarr.open_group(str(source / "dataset.zarr"), mode="r")
    names = _episode_names(source_root)
    episodes = [source_root["episodes"][name] for name in names]
    if not episodes:
        raise ValueError("source contains no complete episodes")
    states, actions, ends, images, depths = [], [], [], [], []
    total = 0
    for name, episode in zip(names, episodes):
        states.append(_state(episode, "observation"))
        actions.append(_state(episode, "action"))
        total += len(states[-1]); ends.append(total)
        if include_images:
            image = _aligned_rgb(source, episode, name)
            if image is not None: images.append(image)
            depth = _aligned_depth(episode)
            if depth is not None: depths.append(depth)
    destination.mkdir(parents=True, exist_ok=False)
    root = zarr.open_group(str(destination / "replay_buffer.zarr"), mode="w")
    data, meta = root.require_group("data"), root.require_group("meta")
    data.create_dataset("state", data=np.concatenate(states))
    data.create_dataset("action", data=np.concatenate(actions))
    if images and len(images) == len(episodes):
        data.create_dataset("realsense_rgb", data=np.concatenate(images))
    if depths and len(depths) == len(episodes):
        data.create_dataset("realsense_depth", data=np.concatenate(depths))
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))


def export_act(source: Path, destination: Path, include_images=True):
    try:
        import h5py
        import zarr
    except ImportError as exc:
        raise ImportError("ACT export requires h5py and zarr") from exc
    root = zarr.open_group(str(source / "dataset.zarr"), mode="r")
    names = _episode_names(root)
    if not names:
        raise ValueError("source contains no complete episodes")
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        episode = root["episodes"][name]
        with h5py.File(destination / f"episode_{int(name):06d}.hdf5", "w") as output:
            output.attrs["sim"] = False
            observations = output.require_group("observations")
            observations.create_dataset("qpos", data=_state(episode, "observation"), compression="gzip")
            output.create_dataset("action", data=_state(episode, "action"), compression="gzip")
            if include_images:
                image = _aligned_rgb(source, episode, name)
                if image is not None:
                    observations.require_group("images").create_dataset("realsense", data=image, compression="gzip")
                depth = _aligned_depth(episode)
                if depth is not None:
                    observations.require_group("images").create_dataset("realsense_depth", data=depth, compression="gzip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--format", choices=("diffusion-policy", "act"), required=True)
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    source, destination = Path(args.source), Path(args.destination)
    if args.format == "diffusion-policy":
        export_diffusion(source, destination, args.include_images)
    else:
        export_act(source, destination, args.include_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
