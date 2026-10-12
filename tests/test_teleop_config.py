from pathlib import Path

import pytest
import yaml

from teleop.config import load_config, require_site_configuration


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


def test_pico_configuration_only_changes_input_and_device_calibration():
    root = Path(__file__).resolve().parents[1]
    quest = load_config(root / "configs/quest3_nero_l20.yaml")
    pico = load_config(root / "configs/pico4_nero_l20.yaml")
    for section in ("arm", "tracking", "safety", "camera", "recording"):
        assert pico[section] == quest[section]
    assert pico["control_hz"] == quest["control_hz"]
    assert pico["hand"]["retarget_config"].endswith("pico4/pico4_linker_l20.yaml")
    for key in quest["hand"]:
        if key != "retarget_config":
            assert pico["hand"][key] == quest["hand"][key]
    with pytest.raises(RuntimeError, match="R_base_pico4"):
        require_site_configuration(pico, control="both")


@pytest.mark.parametrize("section,key,value", [
    (None, "input_device", "other"), ("pico4", "side", "left"),
    ("pico4", "relay_host", "0.0.0.0"), ("pico4", "relay_port", 0),
    ("pico4", "input_mode", "udp"), ("pico4", "port", 65536),
    ("calibration", "R_base_pico4", [[0, 0, 0]]),
])
def test_invalid_pico_configuration_is_rejected(tmp_path, section, key, value):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/pico4_nero_l20.yaml")
    (config if section is None else config[section])[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
