from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

from teleop.realsense_camera import RealSenseCamera


@pytest.fixture
def sdk(monkeypatch):
    pytest.importorskip("cv2")
    intrinsics = SimpleNamespace(width=960, height=540, fx=500, fy=510, ppx=480, ppy=270,
                                 model="none", coeffs=[0.0] * 5)
    profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(
        get_intrinsics=lambda: intrinsics))

    def make_frame(data, stamp, number):
        return SimpleNamespace(get_data=lambda: data, get_timestamp=lambda: stamp,
                               get_frame_number=lambda: number,
                               get_frame_timestamp_domain=lambda: "hardware_clock", profile=profile)

    color_array = np.full((540, 960, 3), 123, dtype=np.uint8)
    depth_array = np.full((540, 960), 4001, dtype=np.uint16)
    frames = SimpleNamespace(get_color_frame=lambda: make_frame(color_array, 1000, 10),
                             get_depth_frame=lambda: make_frame(depth_array, 999, 11))
    calls = []
    device = SimpleNamespace(get_info=lambda key: {"name": "Intel RealSense L515", "serial": "123"}[key],
                             first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.00025))

    class Pipeline:
        sent = False

        def start(self, cfg):
            calls.append("start")
            return SimpleNamespace(get_device=lambda: device, get_stream=lambda _: profile)

        def try_wait_for_frames(self, timeout_ms):
            if not self.sent:
                self.sent = True
                return True, object()
            time.sleep(0.005)
            return False, None

        def stop(self):
            calls.append("stop")

    pipeline = Pipeline()
    cfg = SimpleNamespace(enable_device=lambda serial: calls.append(("device", serial)),
                          enable_stream=lambda *args: calls.append(args))

    def align(target):
        calls.append(("align", target))
        return SimpleNamespace(process=lambda _: frames)

    rs = SimpleNamespace(config=lambda: cfg, pipeline=lambda: pipeline, align=align,
                         stream=SimpleNamespace(color="color", depth="depth"),
                         format=SimpleNamespace(rgb8="rgb8", z16="z16"),
                         camera_info=SimpleNamespace(name="name", serial_number="serial"))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    return SimpleNamespace(calls=calls, color=color_array, depth=depth_array,
                           pipeline=pipeline, device=device)


def config():
    return dict(model="L515", serial="123", width=64, height=48, fps=30, max_age_s=1.0)


def test_camera_aligns_and_publishes_owned_snapshot(sdk, monkeypatch):
    camera = RealSenseCamera(config())
    try:
        camera.start()
        frame = camera.get_latest()
        assert ("color", 960, 540, "rgb8", 30) in sdk.calls
        assert ("depth", 640, 480, "z16", 30) in sdk.calls
        assert ("align", "color") in sdk.calls
        assert frame.color_frame_number == 10
        assert frame.depth_frame_number == 11
        assert frame.depth_timestamp_ms == 999
        assert camera.metadata["depth_scale_m"] == 0.00025
        assert camera.metadata["aligned_depth_intrinsics"]["width"] == 64
        assert camera.metadata["color_crop_xywh"] == [120, 0, 720, 540]
        assert camera.metadata["color_intrinsics"]["fx"] == pytest.approx(500 * 64 / 720)
        assert camera.metadata["color_intrinsics"]["ppx"] == pytest.approx((480 - 120 + 0.5) * 64 / 720 - 0.5)
        assert frame.color.shape == (48, 64, 3)
        assert frame.depth.shape == (48, 64)
        sdk.color.fill(0)
        sdk.depth.fill(0)
        assert frame.color[0, 0, 0] == 123
        assert frame.depth[0, 0] == 4001
        monkeypatch.setattr("teleop.realsense_camera.time.monotonic", lambda: frame.received_at + 2.0)
        assert camera.get_latest() is None
    finally:
        camera.stop()
        camera.stop()
    assert sdk.calls.count("stop") == 1


def test_capture_error_stops_pipeline_on_startup(sdk, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("camera disconnected")

    monkeypatch.setattr(sdk.pipeline, "try_wait_for_frames", fail)
    camera = RealSenseCamera(config())
    with pytest.raises(RuntimeError, match="capture failed"):
        camera.start()
    assert sdk.calls.count("stop") == 1


def test_default_vga_output_crops_both_streams_without_mixing_depth(sdk):
    sdk.color[:] = [0, 0, 240]  # Borders should be removed, not stretched into the image.
    sdk.color[:, 120:480] = [240, 0, 0]
    sdk.color[:, 480:840] = [0, 240, 0]
    sdk.depth[:] = 0
    sdk.depth[:, 120:480] = 1000
    sdk.depth[:, 480:840] = 65535
    cfg = config()
    cfg.update(width=640, height=480)
    camera = RealSenseCamera(cfg)
    try:
        camera.start()
        frame = camera.get_latest()
        assert frame.color.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
        assert np.all(frame.color[:, :320] == [240, 0, 0])
        assert np.all(frame.color[:, 320:] == [0, 240, 0])
        assert np.all(frame.depth[:, :320] == 1000)
        assert np.all(frame.depth[:, 320:] == 65535)
        assert camera.metadata["color_intrinsics"]["fx"] == pytest.approx(500 * 8 / 9)
    finally:
        camera.stop()


def test_wrong_device_is_rejected_and_closed(sdk):
    sdk.device.get_info = lambda _: "Intel RealSense D435"
    camera = RealSenseCamera(config())
    with pytest.raises(RuntimeError, match="Expected L515"):
        camera.start()
    assert sdk.calls.count("stop") == 1
