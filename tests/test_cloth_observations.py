import json
import time

import numpy as np
import pytest

from cloth_agent.config import MOCK_CONFIG, load_config
from cloth_agent.geometry import calculate, transform, transform_points
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


def test_camera_keypoints_are_transformed_once_and_task_keypoints_are_preserved(tmp_path):
    matrix = transform([.3, -.2, .1, .1, -.2, 1.])
    path = tmp_path / "points.json"
    provider = FileKeypointProvider(path, task_from_camera=matrix)
    point = [.1, .2, .7]
    frame = {"timestamp_monotonic": time.monotonic(), "frame": "camera", "points": {"corner": point}}
    path.write_text(json.dumps(frame))
    result = provider.latest()
    assert result["frame"] == "task"
    np.testing.assert_allclose(result["points"]["corner"], transform_points(matrix, point))
    assert provider.latest() == result  # Repeated reads never transform twice.
    frame["frame"] = "task"
    path.write_text(json.dumps(frame))
    assert provider.latest()["points"]["corner"] == point


def test_configured_extrinsics_reach_codex_and_script_geometry_tools(tmp_path):
    from cloth_agent.codex import CodexAppServer

    config = load_config(MOCK_CONFIG)
    matrix = transform([.3, -.2, .1, .1, -.2, 1.])
    config["observations"]["task_from_camera"] = matrix.tolist()
    intrinsics = {"fx": 100., "fy": 100., "ppx": 0., "ppy": 0.}
    with Runtime(config, tmp_path / "run") as runtime:
        client = CodexAppServer(runtime)
        values = {"pixel": [10, 20], "depth_m": .8, "intrinsics": intrinsics}
        expected = transform_points(matrix, [.08, .16, .8])
        # An argument cannot silently replace the runtime's calibrated transform.
        values["task_from_camera"] = np.eye(4).tolist()
        np.testing.assert_allclose(client._dispatch("cloth_geometry", {
            "operation": "pixel_to_task", "values": values}), expected)
        np.testing.assert_allclose(runtime._script_call("calculate", {
            "operation": "pixel_to_task", **values}), expected)
        np.testing.assert_allclose(runtime.geometry("camera_to_task", points=[.08, .16, .8]), expected)
        assert runtime.observe()["images"]["task_from_camera"] == matrix.tolist()
        runtime.config["observations"]["task_from_camera"] = None
        with pytest.raises(ValueError, match="calibration"):
            runtime.geometry("pixel_to_task", **values)


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
    config = load_config(MOCK_CONFIG)
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
    with Runtime(load_config(MOCK_CONFIG), tmp_path / "mock") as runtime:
        first = runtime.observe()
        second = runtime.observe()
        assert first["backend"] == "mock"
        assert first["keypoints"] is None
        assert first["images"]["source"] == "synthetic"
        assert first["images"]["rgb_path"] == second["images"]["rgb_path"]
