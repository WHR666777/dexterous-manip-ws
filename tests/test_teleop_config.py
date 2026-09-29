from pathlib import Path

import pytest
import yaml

from teleop.config import load_config


def test_old_configuration_keeps_camera_optional(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/quest3_nero_l20.yaml")
    assert config["camera"]["enabled"] is True
    assert (config["camera"]["width"], config["camera"]["height"]) == (640, 480)
    assert config["control_hz"] == 20
    config.pop("camera")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["camera"]["enabled"] is False


@pytest.mark.parametrize("key,value", [("fps", 20), ("align_depth_to_color", False),
                                      ("serial", 1234), ("max_age_s", float("nan"))])
def test_invalid_rgbd_configuration_is_rejected(tmp_path, key, value):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/quest3_nero_l20.yaml")
    config["camera"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
