import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from robot_control.tcp_calibration import (
    CalibrationError, solve_camera_tcp, tcp_rotation_from_reference,
)
from scripts.calibrate_tcp_camera import main, solve_session
from teleop.config import load_config
from teleop.hand_replay import HandReplay
from teleop.realsense_camera import RGBDFrame
from teleop.tcp_calibration import (
    CalibrationSession, load_observations, point_from_pixel, write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_samples(noise_m=0.):
    rng = np.random.default_rng(7)
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("xyz", [.3, -.5, 1.2]).as_matrix()
    camera[:3, 3] = [.4, -.6, .7]
    offset = np.array([.035, -.17, .09])
    poses = np.tile(np.eye(4), (30, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler("xyz", rng.uniform(-.6, .6, (30, 3))).as_matrix()
    poses[:, :3, 3] = rng.uniform(-.3, .3, (30, 3))
    points_base = np.einsum("nij,j->ni", poses[:, :3, :3], offset) + poses[:, :3, 3]
    points_camera = (points_base - camera[:3, 3]) @ camera[:3, :3]
    return camera, offset, points_camera + rng.normal(0, noise_m, points_camera.shape), poses


@pytest.mark.parametrize("noise_m,tolerance", [(0., 1e-8), (.0005, .0015)])
def test_joint_fit_recovers_both_transforms(noise_m, tolerance):
    expected_camera, offset, points, poses = synthetic_samples(noise_m)
    fitted = solve_camera_tcp(points, poses)
    np.testing.assert_allclose(fitted.base_from_camera, expected_camera, atol=tolerance)
    np.testing.assert_allclose(fitted.tcp_offset_flange_m, offset, atol=tolerance)
    exported = fitted.as_dict(np.eye(3))
    np.testing.assert_allclose(np.asarray(exported["tcp_from_flange"]) @ exported["flange_from_tcp"],
                               np.eye(4), atol=1e-10)
    assert exported["fit"]["rms_m"] < max(1e-8, noise_m * 2)


@pytest.mark.parametrize("yaw_only", [False, True])
def test_fixed_tilt_samples_are_rejected_despite_zero_residual(yaw_only):
    camera, offset, _, poses = synthetic_samples()
    reference = Rotation.from_euler("xyz", [-97.5, -6.3, 0], degrees=True).as_matrix()
    yaw = np.linspace(-1, 1, len(poses)) if yaw_only else np.zeros(len(poses))
    poses[:, :3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ reference
    points = (np.einsum("nij,j->ni", poses[:, :3, :3], offset) + poses[:, :3, 3]
              - camera[:3, 3]) @ camera[:3, :3]
    with pytest.raises(CalibrationError, match="yaw-only"):
        solve_camera_tcp(points, poses)


def test_tool_rotation_has_reference_axes_and_is_independent_of_reference_yaw():
    results = []
    for yaw in (0, 45, -100):
        reference = np.eye(4)
        reference[:3, :3] = Rotation.from_euler("xyz", [-97.507, -6.326, yaw], degrees=True).as_matrix()
        rotation = tcp_rotation_from_reference(reference)
        actual = reference[:3, :3] @ rotation
        np.testing.assert_allclose(actual[:, 2], [0, 0, -1], atol=1e-10)
        expected_x = reference[:3, 2].copy()
        expected_x[2] = 0
        expected_x /= np.linalg.norm(expected_x)
        np.testing.assert_allclose(actual[:, 0], expected_x, atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-10)
        results.append(rotation)
    np.testing.assert_allclose(results[0], results[1], atol=1e-10)
    np.testing.assert_allclose(results[0], results[2], atol=1e-10)
    with pytest.raises(CalibrationError, match="undefined"):
        tcp_rotation_from_reference(np.eye(4))


def make_session(tmp_path, camera=None):
    config = load_config(ROOT / "configs/pico4_nero_l20.yaml")
    clip = HandReplay(config["hand"]["replay_clip_file"])
    camera = camera or {
        "depth_scale_m": .001,
        "color_intrinsics": {"fx": 100., "fy": 100., "ppx": 8., "ppy": 6., "coeffs": []},
    }
    return CalibrationSession(tmp_path / "session", config, clip.actions[-1], len(clip.actions) - 1, camera)


def test_capture_waits_for_stability_and_saves_actual_feedback(monkeypatch, tmp_path):
    from teleop import tcp_calibration as capture

    clock = [0.]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(capture.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    session = make_session(tmp_path)
    # Moving for .2 s, stationary thereafter. No motion-command API is available.
    arm = SimpleNamespace(get_joint_positions=lambda: np.full(7, min(clock[0], .2)))
    camera = SimpleNamespace(get_latest=lambda: RGBDFrame(
        np.full((12, 16, 3), 80, np.uint8), np.full((12, 16), 600, np.uint16),
        clock[0], 100., 100., 1, 1, "hardware_clock", "hardware_clock"))
    expected_fk = np.eye(4)
    expected_fk[0, 3] = .123
    ik = SimpleNamespace(forward=lambda q: expected_fk.copy())
    hand = SimpleNamespace(get_latest=lambda: SimpleNamespace(
        positions_raw=np.full(20, 12), query_finished_at=clock[0]))
    path = session.capture(arm, ik, camera, hand)
    assert clock[0] >= .58
    with np.load(path) as sample:
        np.testing.assert_allclose(sample["arm_joint_position_rad"], .2)
        np.testing.assert_array_equal(sample["base_from_flange"], expected_fk)
        np.testing.assert_array_equal(sample["depth"], 600)
        assert "received_at" in json.loads(str(sample["frame_metadata_json"]))
    with pytest.raises(FileExistsError):
        make_session(tmp_path)


@pytest.mark.parametrize("fault", ["moving", "stale_image"])
def test_capture_does_not_save_moving_or_stale_samples(monkeypatch, tmp_path, fault):
    from teleop import tcp_calibration as capture

    clock = [0.]
    monkeypatch.setattr(capture.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(capture.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    session = make_session(tmp_path)
    arm = SimpleNamespace(get_joint_positions=lambda: np.full(7, clock[0] if fault == "moving" else 0))
    camera = SimpleNamespace(get_latest=lambda: None)
    with pytest.raises(TimeoutError):
        session.capture(arm, None, camera, None, timeout_s=.8)
    assert not list((session.directory / "samples").glob("*.npz"))


def test_clicked_point_uses_exact_depth_and_rejects_missing_depth(tmp_path):
    session = make_session(tmp_path)
    depth = np.full((12, 16), 600, np.uint16)
    np.testing.assert_allclose(point_from_pixel(depth, [10, 5], session.metadata["camera"]), [.012, -.006, .6])
    depth[5, 10] = 0
    with pytest.raises(ValueError, match="valid depth"):
        point_from_pixel(depth, [10, 5], session.metadata["camera"])


def test_saved_rgbd_clicks_to_exported_calibration(tmp_path):
    session = make_session(tmp_path)
    expected_camera, offset, _, poses = synthetic_samples()
    annotations = {}
    rng = np.random.default_rng(9)
    for index, pose in enumerate(poses):
        pixel = rng.integers([0, 0], [16, 12]).tolist()
        depth = np.full((12, 16), rng.integers(400, 1000), np.uint16)
        point = point_from_pixel(depth, pixel, session.metadata["camera"])
        pose[:3, 3] = (expected_camera[:3, :3] @ point + expected_camera[:3, 3]
                       - pose[:3, :3] @ offset)
        name = f"sample-{index + 1:04d}.npz"
        np.savez_compressed(session.directory / "samples" / name, depth=depth, base_from_flange=pose)
        # Solver must reconstruct from the pixel/depth, not trust cached 3D annotations.
        annotations[name] = {"pixel_uv": pixel, "point_camera_m": [999, 999, 999]}
    write_json(session.directory / "annotations.json", annotations)
    _, names, points, _ = load_observations(session.directory)
    assert len(points) == len(poses) == len(names)
    result = solve_session(session.directory)
    np.testing.assert_allclose(result["base_from_camera"], expected_camera, atol=1e-8)
    np.testing.assert_allclose(np.asarray(result["flange_from_tcp"])[:3, 3], offset, atol=1e-8)
    exported = yaml.safe_load((session.directory / "calibration.yaml").read_text())
    assert exported["fit"]["sample_count"] == len(poses)
    # An inconsistent click must produce a failed fit without replacing a good file.
    previous = (session.directory / "calibration.yaml").read_bytes()
    annotations[names[0]]["pixel_uv"] = [15, 11]
    write_json(session.directory / "annotations.json", annotations)
    with pytest.raises(ValueError, match="RMS"):
        solve_session(session.directory, max_rms_mm=.00001)
    assert (session.directory / "calibration.yaml").read_bytes() == previous


def test_collection_requires_execute_before_connect(monkeypatch, tmp_path):
    from teleop import controller

    monkeypatch.setattr(controller, "run", lambda *_a, **_kw: pytest.fail("Hardware was touched"))
    with pytest.raises(SystemExit, match="without --execute"):
        main(["collect", str(tmp_path / "new"), "--config", str(ROOT / "configs/pico4_nero_l20.yaml")])
