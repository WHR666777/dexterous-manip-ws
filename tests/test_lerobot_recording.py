"""Joint-channel, lossless RGB-D and real pinned-LeRobot save regression tests."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from teleop.lerobot_export import (
    BIMANUAL_JOINT_NAMES, JOINT_NAMES, check_existing, dataset_features,
    joint_vectors, validate_staging,
)
from teleop.recording import EpisodeRecorder
from teleop.realsense_camera import RGBDFrame


ACT_PYTHON = Path("/home/lab404/anaconda3/envs/act/bin/python")
SETTINGS = {"python": str(ACT_PYTHON), "repo_id": "local/nero_l20_test", "task": "Pick up a block"}
METADATA = {"control": "both", "input_side": "right", "camera": {
    "width": 640, "height": 480, "depth_scale_m": 0.00025, "depth_aligned_to": "color",
    "color_intrinsics": {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240},
}}
DUAL_METADATA = {key: value for key, value in METADATA.items() if key != "input_side"}
DUAL_METADATA["input_sides"] = ["left", "right"]


def row(i=0):
    return dict(
        timestamp_monotonic=10.04 + i * .05, observation_timestamp_monotonic=10.0 + i * .05,
        observation_arm_joint_position=np.arange(7) * .01 + i * .001,
        observation_hand_position_raw=np.arange(20) + i,
        arm_action=np.arange(7) * .01 + .1 + i * .001,
        hand_action_raw=np.arange(20) + 50 + i,
    )


def dual_row(i=0):
    base = row(i)
    result = {key: base[key] for key in ("timestamp_monotonic", "observation_timestamp_monotonic")}
    result["cycle_id"] = i + 1
    for side in ("left", "right"):
        for key, value in base.items():
            if key == "timestamp_monotonic":
                continue
            if side == "right" and isinstance(value, np.ndarray):
                value = value + (30 if "hand" in key else .05)
            result[f"{side}_{key}"] = value
    return result


def frame(i=0):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180 + i
    rgb[:, :, 2] = 30
    depth = np.full((480, 640), 1234 + i, dtype=np.uint16)
    depth[0, :2] = [0, 65535]
    return RGBDFrame(rgb, depth, 9.99 + i * .05, 100 + i, 99 + i,
                     i, i, "hardware_clock", "hardware_clock")


def staged(tmp_path, monkeypatch, count=3, bimanual=False):
    recorder = EpisodeRecorder(tmp_path / "dataset", DUAL_METADATA if bimanual else METADATA,
                               lerobot=SETTINGS)
    monkeypatch.setattr(recorder, "_export_lerobot", lambda directory: directory)
    recorder.start()
    for i in range(count):
        recorder.append(camera_frame=frame(i), **(dual_row(i) if bimanual else row(i)))
    recorder.start()  # Repeated begin cannot discard data.
    return recorder, recorder.stop_and_save()


def test_active_channels_and_pre_command_feedback():
    rows = [row(0), row(1)]
    state, action = joint_vectors({k: np.array([r[k] for r in rows]) for k in rows[0]})
    assert state.dtype == action.dtype == np.float32
    assert state.shape == action.shape == (2, 23)
    assert len(JOINT_NAMES) == 23
    expected = np.array(list(range(11)) + list(range(15, 20)))
    np.testing.assert_array_equal(state[0, 7:], expected)
    np.testing.assert_array_equal(action[0, 7:], expected + 50)
    np.testing.assert_allclose(action[:, :7] - state[:, :7], .1, atol=1e-7)


def test_bimanual_vectors_preserve_side_order_and_pre_command_feedback():
    rows = [dual_row(0), dual_row(1)]
    data = {key: np.array([sample[key] for sample in rows]) for key in rows[0]}
    state, action = joint_vectors(data, bimanual=True)
    active = np.array(list(range(11)) + list(range(15, 20)))
    assert state.shape == action.shape == (2, 46)
    assert state.dtype == action.dtype == np.float32
    np.testing.assert_array_equal(state[0, 7:23], active)
    np.testing.assert_array_equal(state[0, 30:], active + 30)
    np.testing.assert_array_equal(action[0, 7:23], active + 50)
    np.testing.assert_array_equal(action[0, 30:], active + 80)
    assert BIMANUAL_JOINT_NAMES == [f"{side}.{name}" for side in ("left", "right") for name in JOINT_NAMES]
    assert dataset_features(True)["action"]["names"] == BIMANUAL_JOINT_NAMES
    assert dataset_features()["action"]["shape"] == (23,)


def test_bimanual_rgbd_staging_and_schema(tmp_path, monkeypatch):
    import h5py

    recorder, directory = staged(tmp_path, monkeypatch, bimanual=True)
    metadata, data, state, action = validate_staging(directory)
    assert metadata["schema_version"] == 3
    assert metadata["input_sides"] == ["left", "right"]
    assert state.shape == action.shape == (3, 46)
    np.testing.assert_array_equal(data["cycle_id"], [1, 2, 3])
    assert not recorder.output_dir.exists()
    with h5py.File(directory / "camera/depth.h5", "r") as depth:
        np.testing.assert_array_equal(depth["depth"][:], [frame(i).depth for i in range(3)])


@pytest.mark.parametrize("fault", ["missing_right", "reused_cycle", "bad_raw"])
def test_bimanual_staging_rejects_partial_or_invalid_samples(tmp_path, monkeypatch, fault):
    _, directory = staged(tmp_path, monkeypatch, bimanual=True)
    with np.load(directory / "trajectory.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    if fault == "missing_right":
        del data["right_arm_action"]
    elif fault == "reused_cycle":
        data["cycle_id"][1] = data["cycle_id"][0]
    else:
        data["right_observation_hand_position_raw"][0, 0] = -1
    np.savez_compressed(directory / "trajectory.npz", **data)
    with pytest.raises(ValueError):
        validate_staging(directory)


def test_png_and_z16_staging_are_lossless(tmp_path, monkeypatch):
    import cv2
    import h5py

    recorder, directory = staged(tmp_path, monkeypatch)
    metadata, data, state, action = validate_staging(directory)
    assert not recorder.active
    assert metadata["samples"] == 3
    assert state.shape == action.shape == (3, 23)
    assert not recorder.output_dir.exists()
    assert recorder.stop_and_save() is None
    with h5py.File(directory / "camera/depth.h5", "r") as depth:
        for i in range(3):
            np.testing.assert_array_equal(depth["depth"][i], frame(i).depth)
            bgr = cv2.imread(str(directory / "camera/color" / f"frame-{i:06d}.png"))
            np.testing.assert_array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), frame(i).color)
    np.testing.assert_array_equal(data["camera_frame_index"], [0, 1, 2])


def test_empty_native_recording_does_not_create_dataset(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "dataset", METADATA, lerobot=SETTINGS)
    recorder.start()
    assert recorder.stop_and_save() is None
    assert not recorder.output_dir.exists()


@pytest.mark.parametrize("metadata", [{"control": "hand"}, {"control": "both"}])
def test_native_recording_requires_both_devices_and_camera(tmp_path, metadata):
    recorder = EpisodeRecorder(tmp_path / "dataset", metadata, lerobot=SETTINGS)
    with pytest.raises(ValueError, match="requires --control both"):
        recorder.start()
    assert not recorder.active


def test_missing_rgbd_rejects_whole_episode(tmp_path, monkeypatch):
    recorder = EpisodeRecorder(tmp_path / "dataset", METADATA, lerobot=SETTINGS)
    monkeypatch.setattr(recorder, "_export_lerobot", validate_staging)
    recorder.start()
    recorder.append(camera_frame=frame(), **row())
    recorder.append(camera_frame=None, **row(1))
    with pytest.raises(ValueError, match="Missing RGB-D"):
        recorder.stop_and_save()
    assert (recorder._directory / "trajectory.npz").is_file()
    assert not recorder.output_dir.exists()


def test_failed_export_keeps_staged_data_and_log(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "dataset", METADATA,
                               lerobot={**SETTINGS, "python": "/no/such/python"})
    recorder.start()
    recorder.append(camera_frame=frame(), **row())
    with pytest.raises(RuntimeError, match="staged data retained"):
        recorder.stop_and_save()
    assert (recorder._directory / "trajectory.npz").is_file()
    assert (recorder._directory / "export.log").is_file()
    assert not recorder.active


def test_queue_failure_never_exports_truncated_native_episode(tmp_path, monkeypatch):
    recorder = EpisodeRecorder(tmp_path / "dataset", METADATA, lerobot=SETTINGS)
    recorder.start()
    recorder.append(camera_frame=frame(), **row())

    def fail(_):
        raise RuntimeError("queue is full")

    monkeypatch.setattr(recorder._writer, "append", fail)
    with pytest.raises(RuntimeError, match="queue is full"):
        recorder.append(camera_frame=frame(1), **row(1))
    with pytest.raises(RuntimeError, match="partial episode saved"):
        recorder.stop_and_save()
    assert not recorder.output_dir.exists()


def test_png_write_failure_keeps_staging_and_never_exports(tmp_path, monkeypatch):
    recorder = EpisodeRecorder(tmp_path / "dataset", METADATA, lerobot=SETTINGS)
    recorder.start()
    monkeypatch.setattr(recorder._writer._cv2, "imwrite", lambda *args: False)
    recorder.append(camera_frame=frame(), **row())
    with pytest.raises(RuntimeError, match="partial episode saved"):
        recorder.stop_and_save()
    metadata = json.loads((recorder._directory / "metadata.json").read_text())
    assert "Cannot write staged RGB PNG" in metadata["camera"]["write_error"]
    assert not recorder.output_dir.exists()


@pytest.mark.parametrize("key", ["arm_action", "observation_arm_joint_position",
                                  "hand_action_raw", "observation_hand_position_raw"])
def test_invalid_joint_values_rejected_before_writing(tmp_path, monkeypatch, key):
    recorder, directory = staged(tmp_path, monkeypatch)
    with np.load(directory / "trajectory.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    data[key] = data[key].astype(float)
    data[key][0, 0] = np.nan if "arm" in key else -1
    np.savez_compressed(directory / "trajectory.npz", **data)
    with pytest.raises(ValueError):
        validate_staging(directory)
    assert not recorder.output_dir.exists()


@pytest.mark.skipif(not ACT_PYTHON.is_file(), reason="Pinned act environment is not installed")
def test_real_lerobot_multi_episode_restart_and_mismatch(tmp_path, monkeypatch):
    import h5py

    root = tmp_path / "dataset"
    for episode_index in range(3):
        # A new object/process each time also covers recording across restarts.
        recorder = EpisodeRecorder(root, METADATA, lerobot=SETTINGS)
        recorder.start()
        for i in range(3):
            recorder.append(camera_frame=frame(i), **row(i))
        assert recorder.stop_and_save() == root
        info = json.loads((root / "meta/info.json").read_text())
        assert info["total_episodes"] == episode_index + 1
        assert info["total_frames"] == (episode_index + 1) * 3
        with h5py.File(root / "depth" / f"episode-{episode_index:06d}.h5", "r") as depth:
            np.testing.assert_array_equal(depth["depth"][:], [frame(i).depth for i in range(3)])
        meta = json.loads((root / "diagnostics" / f"episode-{episode_index:06d}.json").read_text())
        assert meta["actual_hz"] == pytest.approx(20)
        assert meta["camera"]["depth_scale_m"] == .00025
        assert not recorder._directory.exists()
    check = r'''
import sys
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local/nero_l20_test", root=sys.argv[1],
                    delta_timestamps={"action": [i/20 for i in range(5)]})
assert ds.num_episodes == 3 and len(ds) == 9
active = np.array(list(range(11)) + list(range(15, 20)))
for episode in range(3):
    sample = ds[episode * 3]
    assert sample["frame_index"].item() == 0
    assert sample["episode_index"].item() == episode
    np.testing.assert_allclose(sample["observation.state"][:7], np.arange(7)*.01, atol=1e-7)
    np.testing.assert_array_equal(sample["observation.state"][7:], active)
    np.testing.assert_array_equal(sample["action"][0,7:], active+50)
    assert sample["action_is_pad"].tolist() == [False, False, False, True, True]
    np.testing.assert_array_equal(sample["action"][3], sample["action"][2])
    image = sample["observation.images.front"]
    assert image.shape == (3,480,640)
    np.testing.assert_allclose(image.mean((1,2))*255, [180,0,30], atol=15)
assert np.isfinite(ds.meta.stats["observation.state"]["std"]).all()
assert np.asarray(ds.meta.stats["observation.state"]["mean"]).shape == (23,)
print("PASS: official loader, RGB channels, 23-D vectors, stats and episode padding")
'''
    subprocess.run([str(ACT_PYTHON), "-B", "-c", check, str(root)], check=True,
                   env={**os.environ, "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libffi.so.7",
                        "HF_HOME": str(tmp_path / "hf-cache"), "HF_HUB_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1"})
    before = (root / "meta/info.json").read_bytes()
    mismatch = EpisodeRecorder(root, METADATA, lerobot=SETTINGS, sample_hz=10)
    mismatch.start()
    mismatch.append(camera_frame=frame(), **row())
    with pytest.raises(RuntimeError, match="LeRobot save failed"):
        mismatch.stop_and_save()
    assert (root / "meta/info.json").read_bytes() == before
    assert not (root / ".recording-in-progress.json").exists()
    marker = root / ".recording-in-progress.json"
    marker.write_text('{"staging": "interrupted test save"}')
    interrupted = EpisodeRecorder(root, METADATA, lerobot=SETTINGS)
    interrupted.start()
    interrupted.append(camera_frame=frame(), **row())
    with pytest.raises(RuntimeError, match="LeRobot save failed"):
        interrupted.stop_and_save()
    assert "earlier save was interrupted" in (interrupted._directory / "export.log").read_text()
    assert (root / "meta/info.json").read_bytes() == before
    marker.unlink()
    # Corrupt a feature in a copy of the metadata, without touching recordings.
    info = json.loads(before)
    info["features"]["action"]["shape"] = [27]
    from teleop.lerobot_export import check_existing
    (root / "meta/info.json").write_text(json.dumps(info))
    metadata = {**METADATA, "lerobot": SETTINGS, "sample_hz": 20}
    with pytest.raises(ValueError, match="feature differs"):
        check_existing(root, metadata)
    (root / "meta/info.json").write_bytes(before)


@pytest.mark.skipif(not ACT_PYTHON.is_file(), reason="Pinned act environment is not installed")
def test_real_bimanual_lerobot_restart_loader_and_act(tmp_path):
    import h5py

    root = tmp_path / "bimanual_dataset"
    for episode in range(2):
        recorder = EpisodeRecorder(root, DUAL_METADATA, lerobot=SETTINGS)
        recorder.start()
        for i in range(3):
            recorder.append(camera_frame=frame(i), **dual_row(i))
        assert recorder.stop_and_save() == root
        info = json.loads((root / "meta/info.json").read_text())
        assert info["robot_type"] == "nero_l20_bimanual"
        assert info["total_episodes"] == episode + 1
        assert info["features"]["action"]["shape"] == [46]
        assert info["features"]["action"]["names"] == BIMANUAL_JOINT_NAMES
        with h5py.File(root / "depth" / f"episode-{episode:06d}.h5", "r") as depth:
            np.testing.assert_array_equal(depth["depth"][:], [frame(i).depth for i in range(3)])
        meta = json.loads((root / "diagnostics" / f"episode-{episode:06d}.json").read_text())
        assert meta["actual_hz"] == pytest.approx(20.)
    # Refuse a single-side append before touching official data files.
    before = (root / "meta/info.json").read_bytes()
    with pytest.raises(ValueError, match="robot_type"):
        check_existing(root, {**METADATA, "lerobot": SETTINGS, "sample_hz": 20})
    assert (root / "meta/info.json").read_bytes() == before
    env = {**os.environ, "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libffi.so.7",
           "HF_HOME": str(tmp_path / "hf-cache"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    check = r'''
import sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local/nero_l20_test", root=sys.argv[1],
                    delta_timestamps={"action": [i/20 for i in range(5)]})
assert ds.num_episodes == 2 and len(ds) == 6
active = np.array(list(range(11)) + list(range(15,20)))
for episode in range(2):
    sample = ds[episode*3]
    assert sample["observation.state"].shape == (46,)
    np.testing.assert_array_equal(sample["observation.state"][7:23], active)
    np.testing.assert_array_equal(sample["observation.state"][30:], active+30)
    np.testing.assert_array_equal(sample["action"][0,7:23], active+50)
    np.testing.assert_array_equal(sample["action"][0,30:], active+80)
    assert sample["action_is_pad"].tolist() == [False,False,False,True,True]
    assert sample["observation.images.front"].shape == (3,480,640)
'''
    subprocess.run([str(ACT_PYTHON), "-B", "-c", check, str(root)], check=True, env=env)
    subprocess.run([str(ACT_PYTHON), "-B", "scripts/check_lerobot_act.py", str(root), "--device", "cpu"],
                   check=True, env=env, cwd=Path(__file__).resolve().parents[1])
