import json
import time

import numpy as np
import pytest

from cloth_agent.config import load_config
from cloth_agent.geometry import calculate
from cloth_agent.observations import FileKeypointProvider
from cloth_agent.runtime import Runtime


def test_external_keypoints_are_optional_and_expire(tmp_path):
    path = tmp_path / "points.json"
    provider = FileKeypointProvider(path, .25)
    assert provider.latest() is None
    frame = {"timestamp_monotonic": time.monotonic(), "frame": "task",
             "points": {"corner": [.1, .2, .3]}}
    path.write_text(json.dumps(frame))
    assert provider.latest()["points"]["corner"] == [.1, .2, .3]
    frame["timestamp_monotonic"] -= 1
    path.write_text(json.dumps(frame))
    assert provider.latest() is None
    frame["timestamp_monotonic"] = time.monotonic()
    frame["frame"] = "camera"
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match="task frame"):
        provider.latest()


def test_projection_uses_current_rgb_distortion_model():
    cv2 = pytest.importorskip("cv2")
    intr = {"fx": 300., "fy": 300., "ppx": 320., "ppy": 240.,
            "model": "distortion.brown_conrady", "coeffs": [.2, -.1, 0, 0, 0]}
    point = np.array([[[.3, .2, 1.]]])
    matrix = np.array([[300., 0, 320.], [0, 300., 240.], [0, 0, 1.]])
    pixel, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), matrix,
                                 np.array(intr["coeffs"]))
    recovered = calculate("pixel_to_point", pixel=pixel[0, 0], depth_m=1., intrinsics=intr)
    np.testing.assert_allclose(recovered, point[0, 0], atol=1e-7)


def make_recording(tmp_path, valid=True):
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    directory = tmp_path / "episode"
    directory.mkdir()
    writer = cv2.VideoWriter(str(directory / "rgb.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             20., (32, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), [40, 80, 160], dtype=np.uint8))
    writer.release()
    with h5py.File(directory / "depth.h5", "w") as file:
        file.create_dataset("custom_depth", data=np.full((1, 24, 32), 2345, dtype=np.uint16))
    np.savez(directory / "trajectory.npz", timestamp_monotonic=[0.],
             camera_frame_index=[0 if valid else -1], camera_valid=[valid])
    (directory / "metadata.json").write_text(json.dumps({"camera": {
        "color_file": "rgb.avi", "depth_file": "depth.h5", "depth_dataset": "custom_depth",
        "depth_scale_m": .001, "color_intrinsics": {"fx": 30., "fy": 30., "ppx": 16., "ppy": 12.}}}))
    return directory


@pytest.mark.parametrize("valid", [True, False])
def test_existing_rgbd_episode_format_and_invalid_frame_handling(tmp_path, valid):
    episode = make_recording(tmp_path, valid)
    config = load_config()
    config["observations"]["recorded_episode"] = str(episode)
    with Runtime(config, tmp_path / "run") as runtime:
        observation = runtime.observe()
        assert observation["keypoints"] is None
        assert observation["images"]["source"] == "recording"
        assert observation["images"]["available"] == valid
        if valid:
            np.testing.assert_array_equal(np.load(observation["images"]["depth_path"]),
                                          np.full((24, 32), 2345, dtype=np.uint16))


def test_mock_snapshot_is_explicitly_synthetic_and_has_no_invented_keypoints(tmp_path):
    with Runtime(load_config(), tmp_path / "mock") as runtime:
        first = runtime.observe()
        second = runtime.observe()
        assert first["backend"] == "mock"
        assert first["keypoints"] is None
        assert first["images"]["source"] == "synthetic"
        assert first["images"]["rgb_path"] == second["images"]["rgb_path"]
