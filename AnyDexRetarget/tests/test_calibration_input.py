from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
TEST_SCRIPT_ROOT = EXAMPLE_ROOT / "test"
if str(TEST_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_SCRIPT_ROOT))

from calibrate_scaling import create_input_device


class CalibrationInputTests(unittest.TestCase):
    def test_mediapipe_without_video_uses_selected_webcam(self):
        class FakeCamera:
            def __init__(self, camera_id, show_preview):
                self.camera_id = camera_id
                self.show_preview = show_preview

        fake_camera_module = types.SimpleNamespace(Camera=FakeCamera)
        args = Namespace(
            input="mediapipe",
            video=None,
            camera_id=1,
            realsense=False,
            hand="right",
            show_video=False,
        )

        with patch.dict(sys.modules, {"input.camera": fake_camera_module}):
            device = create_input_device(args)

        self.assertIsInstance(device, FakeCamera)
        self.assertEqual(device.camera_id, 1)
        self.assertTrue(device.show_preview)

    def test_realsense_is_used_only_when_explicitly_requested(self):
        class FakeRealsense:
            def __init__(self, hand_side, show_video):
                self.hand_side = hand_side
                self.show_video = show_video

        fake_realsense_module = types.SimpleNamespace(Realsense=FakeRealsense)
        args = Namespace(
            input="mediapipe",
            video=None,
            camera_id=0,
            realsense=True,
            hand="left",
            show_video=False,
        )

        with patch.dict(sys.modules, {"input.realsense": fake_realsense_module}):
            device = create_input_device(args)

        self.assertIsInstance(device, FakeRealsense)
        self.assertEqual(device.hand_side, "left")
        self.assertTrue(device.show_video)

    def test_video_takes_precedence_over_live_camera_options(self):
        class FakeVideo:
            def __init__(self, video_path, hand_side, show_video, loop):
                self.video_path = video_path
                self.hand_side = hand_side
                self.show_video = show_video
                self.loop = loop

        fake_video_module = types.SimpleNamespace(Video=FakeVideo)
        args = Namespace(
            input="mediapipe",
            video="capture.mp4",
            camera_id=1,
            realsense=True,
            hand="right",
            show_video=True,
        )

        with patch.dict(sys.modules, {"input.video": fake_video_module}):
            device = create_input_device(args)

        self.assertIsInstance(device, FakeVideo)
        self.assertEqual(device.video_path, "capture.mp4")
        self.assertTrue(device.loop)


if __name__ == "__main__":
    unittest.main()
