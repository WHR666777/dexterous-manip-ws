"""Load single-side or Pico4 bimanual teleoperation configurations."""

from __future__ import annotations

from copy import deepcopy
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

    if "sides" not in config:
        return _validate_config(config, config_path)
    if config.get("input_device") != "pico4":
        raise ValueError("Bimanual teleoperation requires input_device: pico4.")
    missing = [key for key in ("pico4", "tracking", "safety", "recording")
               if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if not isinstance(config["sides"], dict) or set(config["sides"]) != {"left", "right"}:
        raise ValueError("sides must contain left and right configurations.")
    for side, settings in config["sides"].items():
        if not isinstance(settings, dict) or set(settings) != {"arm", "hand", "calibration"}:
            raise ValueError(f"sides.{side} must contain arm, hand and calibration.")
    # Validate each side with the existing single-side rules and path handling.
    validated = {side: _validate_config(settings, config_path)
                 for side, settings in select_side_configs(config).items()}
    for key in ("camera", "recording"):
        config[key] = deepcopy(validated["left"][key])
    config["sides"] = {
        side: {key: settings[key] for key in ("arm", "hand", "calibration")}
        for side, settings in validated.items()
    }
    channels = [settings[device]["can_channel"]
                for settings in validated.values() for device in ("arm", "hand")]
    if len(set(channels)) != 4:
        raise ValueError("Bimanual arm/hand configurations require four distinct CAN channels.")
    config["_path"] = str(config_path)
    return config


def select_side_configs(
    config: dict[str, Any], side: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve selected sides into independent, existing single-side configs."""
    if side not in (None, "left", "right", "both"):
        raise ValueError("side must be left, right or both.")
    if "sides" not in config:
        configured_side = input_config(config)["side"]
        if side not in (None, configured_side):
            raise ValueError(f"This configuration only defines the {configured_side} side.")
        return {configured_side: config}
    selected = ("left", "right") if side in (None, "both") else (side,)
    shared = {key: value for key, value in config.items() if key != "sides"}
    result = {}
    for name in selected:
        settings = deepcopy(shared)
        settings.update(deepcopy(config["sides"][name]))
        input_config(settings)["side"] = name
        result[name] = settings
    return result


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
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
    if not np.isfinite(float(config.get("control_hz", 0))) or float(config.get("control_hz", 0)) <= 0:
        raise ValueError("control_hz must be finite and positive.")
    if not np.isfinite(float(source["max_age_s"])) or float(source["max_age_s"]) <= 0:
        raise ValueError(f"{section}.max_age_s must be finite and positive.")
    recording = config["recording"]
    lerobot = recording.get("lerobot")
    if lerobot is not None:
        if not isinstance(lerobot, dict):
            raise ValueError("recording.lerobot must be a mapping.")
        for key in ("python", "repo_id", "task"):
            if not isinstance(lerobot.get(key), str) or not lerobot[key].strip():
                raise ValueError(f"recording.lerobot.{key} must be a nonempty string.")
        if len(lerobot["repo_id"].split("/")) != 2 or not all(lerobot["repo_id"].split("/")):
            raise ValueError("recording.lerobot.repo_id must have the form namespace/name.")
        hz = float(config["control_hz"])
        if not np.isfinite(hz) or hz != int(hz):
            raise ValueError("LeRobot recording requires a positive integer control_hz.")
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

    replay_path = Path(config["hand"].get("replay_clip_file", "configs/l20_thumb_index_9_10s.csv"))
    if not replay_path.is_absolute():
        replay_path = project_root / replay_path
    config["hand"]["replay_clip_file"] = str(replay_path.resolve())

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
    if "sides" in config:
        for side, settings in select_side_configs(config).items():
            try:
                require_site_configuration(
                    settings, control=control, require_calibration=require_calibration,
                    require_start_pose=require_start_pose,
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{side}: {exc}") from exc
        return
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
