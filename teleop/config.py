"""Load and validate the single-arm teleoperation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


PLACEHOLDER = "CHANGE_ME"


def input_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["pico4" if config.get("input_device", "quest3") == "pico4" else "quest"]


def calibration_rotation(config: dict[str, Any]) -> np.ndarray:
    key = "R_base_pico4" if config.get("input_device", "quest3") == "pico4" else "R_base_quest"
    return np.asarray(config["calibration"].get(key), dtype=float)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    device = config.setdefault("input_device", "quest3")
    if device not in ("quest3", "pico4"):
        raise ValueError("input_device must be 'quest3' or 'pico4'.")
    section = "pico4" if device == "pico4" else "quest"
    required = (section, "calibration", "tracking", "safety", "arm", "hand", "recording")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    source = input_config(config)
    if source.get("transport") != "tcp":
        raise ValueError(f"{section}.transport must be 'tcp'.")
    if source.get("input_mode", "direct") not in ("direct", "relay"):
        raise ValueError(f"{section}.input_mode must be 'direct' or 'relay'.")
    if source.get("input_mode") == "relay":
        if source.get("relay_host") not in ("127.0.0.1", "localhost"):
            raise ValueError(f"{section}.relay_host must be localhost-only.")
        try:
            relay_port = int(source["relay_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{section}.relay_port must be a TCP port number.") from exc
        if not 1 <= relay_port <= 65535:
            raise ValueError(f"{section}.relay_port must be a TCP port number.")
    if device == "pico4":
        for key in ("port", "broadcast_port"):
            try:
                port = int(source[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"pico4.{key} must be a port number.") from exc
            if not 1 <= port <= 65535:
                raise ValueError(f"pico4.{key} must be a port number.")
    if source.get("side") not in ("left", "right"):
        raise ValueError(f"{section}.side must be 'left' or 'right'.")
    if config["hand"].get("type") != source.get("side"):
        raise ValueError(f"hand.type and {section}.side must select the same side.")
    if float(config.get("control_hz", 0)) <= 0:
        raise ValueError("control_hz must be positive.")
    for key, default in {'feedback_poll_hz': 20.0, 'feedback_max_age_s': .25}.items():
        try:
            value = float(config['hand'].setdefault(key, default))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'hand.{key} must be finite and positive.') from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'hand.{key} must be finite and positive.')
        config['hand'][key] = value
    camera = config.setdefault("camera", {})
    if not isinstance(camera, dict):
        raise ValueError("camera must be a mapping.")
    for key, value in {
        "enabled": False, "model": "L515", "serial": "", "width": 640,
        "height": 480, "fps": 30, "align_depth_to_color": True, "max_age_s": 0.25,
    }.items():
        camera.setdefault(key, value)
    if not isinstance(camera["enabled"], bool):
        raise ValueError("camera.enabled must be true or false.")
    if camera["enabled"]:
        if camera["model"] != "L515" or camera["fps"] != 30:
            raise ValueError("The L515 RGB-D configuration requires model L515 and fps 30.")
        if not isinstance(camera["serial"], str):
            raise ValueError("camera.serial must be a quoted string.")
        if camera["align_depth_to_color"] is not True:
            raise ValueError("RGB-D recording requires camera.align_depth_to_color: true.")
        for key in ("width", "height"):
            value = camera[key]
            if type(value) is not int or value <= 0 or value % 2:
                raise ValueError(f"camera.{key} must be a positive even integer.")
        camera["max_age_s"] = float(camera["max_age_s"])
        if not np.isfinite(camera["max_age_s"]) or camera["max_age_s"] <= 0:
            raise ValueError("camera.max_age_s must be finite and positive.")
    try:
        tcp_cube_side = float(config["safety"]["tcp_workspace_cube_side_m"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "safety.tcp_workspace_cube_side_m must be a finite positive number."
        ) from exc
    if not np.isfinite(tcp_cube_side) or tcp_cube_side <= 0:
        raise ValueError(
            "safety.tcp_workspace_cube_side_m must be a finite positive number."
        )

    rotation = calibration_rotation(config)
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError(f"calibration.R_base_{'pico4' if device == 'pico4' else 'quest'} must be a finite rotation matrix.")

    project_root = Path(__file__).resolve().parents[1]
    retarget_path = Path(config["hand"]["retarget_config"])
    if not retarget_path.is_absolute():
        retarget_path = project_root / retarget_path
    if not retarget_path.is_file():
        raise ValueError(f"AnyDex retarget config does not exist: {retarget_path}")
    config["hand"]["retarget_config"] = str(retarget_path.resolve())

    start_pose_path = Path(config["arm"]["start_pose_file"])
    if not start_pose_path.is_absolute():
        start_pose_path = project_root / start_pose_path
    config["arm"]["start_pose_file"] = str(start_pose_path.resolve())
    config["_path"] = str(config_path)
    return config


def require_site_configuration(
    config: dict[str, Any],
    *,
    control: str,
    require_calibration: bool = True,
    require_start_pose: bool = False,
) -> None:
    unresolved: list[str] = []
    if control in ("arm", "both") and config["arm"].get("can_channel") == PLACEHOLDER:
        unresolved.append("arm.can_channel")
    if control in ("hand", "both") and config["hand"].get("can_channel") == PLACEHOLDER:
        unresolved.append("hand.can_channel")
    if (
        require_calibration
        and control in ("arm", "both")
        and not bool(config["calibration"].get("calibrated"))
    ):
        rotation_key = "R_base_pico4" if config.get("input_device", "quest3") == "pico4" else "R_base_quest"
        unresolved.append(f"calibration.calibrated/{rotation_key}")
    if (
        require_start_pose
        and control in ("arm", "both")
        and not Path(config["arm"]["start_pose_file"]).is_file()
    ):
        unresolved.append("arm.start_pose_file")
    if unresolved:
        raise RuntimeError(
            "Site configuration is incomplete: " + ", ".join(unresolved)
        )
