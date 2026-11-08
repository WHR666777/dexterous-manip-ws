"""独立启动 TCP 配置的几何和回归检查，不连接真实设备。"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from cloth_agent.config import DEFAULT_CONFIG, load_config, validate_hardware_config
from cloth_agent.geometry import transform
from cloth_agent.robo import Robo


@pytest.fixture
def startup_config_path(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["calibration"]["file"] = str(DEFAULT_CONFIG.parent / raw["calibration"]["file"])
    for settings in raw["arms"].values():
        settings["workspace"]["start_pose_file"] = str(
            DEFAULT_CONFIG.parent / settings["workspace"]["start_pose_file"])
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_startup_mirrors_position_with_equal_orientation_and_keeps_right_target():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config = load_config()
    validate_hardware_config(config)
    right = transform(config["arms"]["right"]["initial_grasp_tcp"])
    left = transform(config["arms"]["left"]["initial_grasp_tcp"])
    plane = raw["startup"]["tcp_poses"]["left"]["plane_y_m"]
    np.testing.assert_allclose(right, transform(raw["startup"]["tcp_poses"]["right"]), atol=1e-15)
    np.testing.assert_allclose(left[:3, 3], [right[0, 3], 2 * plane - right[1, 3], right[2, 3]], atol=1e-15)
    np.testing.assert_array_equal(left[:3, :3], right[:3, :3])
    assert left[2, 3] == right[2, 3]
    assert np.linalg.det(left[:3, :3]) == pytest.approx(1.)
    np.testing.assert_allclose(right[:3, 3], [-.6379674577012566, .03042760698096446, .00772207748671605])
    # 左基座在 task 中有约 -5 cm 的 Z 偏移，等高仍应在 task 中成立。
    assert abs(config["arms"]["left"]["task_from_base"][2][3]) > .05


def test_reference_angles_do_not_change_startup_targets(startup_config_path):
    before = load_config(startup_config_path)
    raw = yaml.safe_load(startup_config_path.read_text())
    calibration = yaml.safe_load(Path(raw["calibration"]["file"]).read_text())
    for values in calibration["arms"].values():
        values["reference_roll_pitch_deg"] = [25., -30.]
    calibration_path = startup_config_path.with_name("calibration.yaml")
    calibration_path.write_text(yaml.safe_dump(calibration))
    raw["calibration"]["file"] = calibration_path.name
    startup_config_path.write_text(yaml.safe_dump(raw))
    after = load_config(startup_config_path)
    np.testing.assert_array_equal(np.asarray(after["arms"]["left"]["initial_grasp_tcp"])[:3, :3],
                                  np.asarray(after["arms"]["right"]["initial_grasp_tcp"])[:3, :3])
    for side in ("left", "right"):
        assert after["arms"][side]["reference_roll_pitch_deg"] == [25., -30.]
        np.testing.assert_array_equal(after["arms"][side]["initial_grasp_tcp"], before["arms"][side]["initial_grasp_tcp"])
        np.testing.assert_array_equal(after["arms"][side]["reference_tcp_rotation"], before["arms"][side]["reference_tcp_rotation"])


def test_changing_right_pose_updates_both_startup_targets_without_changing_calibration(startup_config_path):
    before = load_config(startup_config_path)
    raw = yaml.safe_load(startup_config_path.read_text())
    calibration_path = Path(raw["calibration"]["file"])
    original_bytes = calibration_path.read_bytes()
    raw["startup"]["tcp_poses"]["right"] = [-.6, .04, .12, 2.9, .1, -1.4]
    startup_config_path.write_text(yaml.safe_dump(raw))
    after = load_config(startup_config_path)
    np.testing.assert_array_equal(np.asarray(after["arms"]["left"]["initial_grasp_tcp"])[:3, :3],
                                  np.asarray(after["arms"]["right"]["initial_grasp_tcp"])[:3, :3])
    assert calibration_path.read_bytes() == original_bytes
    for side in ("left", "right"):
        assert after["arms"][side]["initial_grasp_tcp"][2][3] == .12
        for key in ("task_from_base", "flange_from_tcp", "reference_roll_pitch_deg"):
            np.testing.assert_array_equal(after["arms"][side][key], before["arms"][side][key])
    robo = Robo(None, after)
    for side in ("left", "right"):
        actual = robo.top_down_pose([0., 0., .2], arm=side)
        np.testing.assert_allclose(actual[:3, :3], np.asarray(after["arms"][side]["initial_grasp_tcp"])[:3, :3])


@pytest.mark.parametrize("invalid", ["missing", "self_mirror", "unknown_source", "nan_plane", "unknown_arm"])
def test_invalid_startup_pose_config_is_rejected_before_device_work(startup_config_path, invalid):
    raw = yaml.safe_load(startup_config_path.read_text())
    poses = raw["startup"]["tcp_poses"]
    if invalid == "missing":
        del poses["left"]
    elif invalid == "self_mirror":
        poses["left"]["mirror_position_of"] = "left"
    elif invalid == "unknown_source":
        poses["left"]["mirror_position_of"] = "other"
    elif invalid == "nan_plane":
        poses["left"]["plane_y_m"] = float("nan")
    else:
        poses["other"] = poses["right"]
    startup_config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="[Ss]tartup"):
        load_config(startup_config_path)


def test_disabled_startup_does_not_synthesize_target_from_calibration_angles(startup_config_path):
    raw = yaml.safe_load(startup_config_path.read_text())
    raw["startup"] = {"enabled": False}
    startup_config_path.write_text(yaml.safe_dump(raw))
    config = load_config(startup_config_path)
    validate_hardware_config(config)
    assert all("initial_grasp_tcp" not in arm for arm in config["arms"].values())
