import json
from threading import Event

import numpy as np
import pytest

from teleop.recording import EpisodeRecorder, _RGBDWriter
from teleop.realsense_camera import RGBDFrame


def frame(value=0):
    color = np.zeros((48, 64, 3), dtype=np.uint8)
    color[:, :, value % 3] = 240
    depth = np.full((48, 64), value + 1000, dtype=np.uint16)
    depth[0, :2] = [0, 65535]
    return RGBDFrame(color, depth, 10.0 + value, 100.0 + value, 99.0 + value,
                     value + 20, value + 30, "hardware_clock", "hardware_clock")


def test_rgbd_round_trip_and_missing_frame_indices(tmp_path):
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    recorder = EpisodeRecorder(tmp_path, {"camera": {"width": 64, "height": 48,
                                                   "depth_scale_m": 0.00025}}, sample_hz=20)
    first, second = frame(0), frame(1)
    recorder.start()
    recorder.append(timestamp_monotonic=11.0, camera_frame=first)
    recorder.start()  # Repeated B preserves existing samples.
    recorder.append(timestamp_monotonic=12.0, camera_frame=None)
    recorder.append(timestamp_monotonic=13.0, camera_frame=second)
    path = recorder.stop_and_save()
    assert path.name == "trajectory.npz"
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["camera_frame_index"], [0, -1, 1])
        np.testing.assert_array_equal(data["camera_valid"], [True, False, True])
        np.testing.assert_array_equal(data["timestamp_monotonic"], [11.0, 12.0, 13.0])
        assert np.isnan(data["camera_timestamp_monotonic"][1])
        assert data["camera_color_timestamp_domain"][0] == "hardware_clock"
    with h5py.File(path.parent / "camera/depth.h5", "r") as depths:
        assert depths["depth"].dtype == np.uint16
        np.testing.assert_array_equal(depths["depth"][:], [first.depth, second.depth])
    video = cv2.VideoCapture(str(path.parent / "camera/color.mp4"))
    try:
        assert video.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
        for expected in (first, second):
            ok, bgr = video.read()
            assert ok
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert np.max(np.abs(rgb.astype(int) - expected.color.astype(int))) < 15
        assert not video.read()[0]
    finally:
        video.release()
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["samples"] == 3
    assert metadata["camera"]["frames"] == 2
    assert metadata["camera"]["depth_scale_m"] == 0.00025
    assert recorder.stop_and_save() is None
    recorder.start()
    recorder.append(timestamp_monotonic=14.0, camera_frame=first)
    next_path = recorder.stop_and_save()
    assert next_path.parent != path.parent
    with np.load(next_path) as data:
        assert data["camera_frame_index"].tolist() == [0]


def test_camera_disabled_keeps_numeric_trajectory_only(tmp_path):
    recorder = EpisodeRecorder(tmp_path, {"control": "hand"})
    recorder.append(action=np.ones(3))
    assert recorder.stop_and_save() is None
    recorder.start()
    recorder.append(action=np.arange(3))
    path = recorder.stop_and_save()
    assert not (path.parent / "camera").exists()
    with np.load(path) as data:
        assert data.files == ["action"]
        np.testing.assert_array_equal(data["action"], [[0, 1, 2]])


def test_writer_backlog_is_bounded_and_fails_without_waiting(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("h5py")
    release = Event()
    original = _RGBDWriter._write

    def delayed_write(self):
        release.wait(timeout=5)
        original(self)

    monkeypatch.setattr(_RGBDWriter, "_write", delayed_write)
    writer = _RGBDWriter(tmp_path / "camera", 64, 48, 20)
    try:
        for index in range(16):
            assert writer.append(frame()) == index
        with pytest.raises(RuntimeError, match="queue is full"):
            writer.append(frame())
    finally:
        release.set()
        writer.close()
    assert writer.written == 16


def test_writer_error_saves_trajectory_with_invalid_image_references(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("h5py")

    def fail_conversion(*args):
        raise OSError("simulated encoding failure")

    recorder = EpisodeRecorder(tmp_path, {"camera": {"width": 64, "height": 48}})
    recorder.start()
    monkeypatch.setattr(recorder._writer._cv2, "cvtColor", fail_conversion)
    recorder.append(timestamp_monotonic=1.0, camera_frame=frame())
    with pytest.raises(RuntimeError, match="partial episode saved"):
        recorder.stop_and_save()
    path = next(tmp_path.glob("*/trajectory.npz"))
    with np.load(path) as data:
        assert data["camera_frame_index"].tolist() == [-1]
        assert data["camera_valid"].tolist() == [False]
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["camera"]["frames"] == 0
    assert "simulated encoding failure" in metadata["camera"]["write_error"]
