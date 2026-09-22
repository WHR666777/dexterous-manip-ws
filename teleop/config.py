"""Load and validate the single-arm teleoperation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


PLACEHOLDER = "CHANGE_ME"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    required = ("quest", "calibration", "tracking", "safety", "arm", "hand", "recording")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    quest = config["quest"]
    if quest.get("transport") != "tcp":
        raise ValueError("Quest transport is fixed to 'tcp' for this project.")
    if quest.get("side") not in ("left", "right"):
        raise ValueError("quest.side must be 'left' or 'right'.")
    if config["hand"].get("type") != quest.get("side"):
        raise ValueError("hand.type and quest.side must select the same side.")
    if float(config.get("control_hz", 0)) <= 0:
        raise ValueError("control_hz must be positive.")
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

    rotation = np.asarray(config["calibration"].get("R_base_quest"), dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError("calibration.R_base_quest must be a finite rotation matrix.")

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
        unresolved.append("calibration.calibrated/R_base_quest")
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
