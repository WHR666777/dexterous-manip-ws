"""Configuration loading, with hardware calibration deliberately left explicit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .geometry import transform


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/cloth_agent.yaml"


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def joint_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain seven finite rad values.")
    return result


def load_config(path=DEFAULT_CONFIG) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["config_dir"] = str(path.parent)
    if config["backend"] not in ("mock", "hardware"):
        raise ValueError("backend must be mock or hardware.")
    if config["backend"] == "hardware" and config["observations"]["recorded_episode"]:
        raise ValueError("Recorded observations are for mock execution only.")
    positive(config["control_hz"], "control_hz")
    for name in ("joint_tolerance_rad", "completion_timeout_s", "max_ik_step_rad",
                 "position_step_m", "rotation_step_rad", "joint_step_rad"):
        positive(config["motion"][name], name)
    if not config["arms"] or any(name not in ("left", "right") for name in config["arms"]):
        raise ValueError("Configure left and/or right arms.")
    return config


def effective_arm(config, arm):
    result = deepcopy(config["arms"][arm])
    if config["backend"] == "mock":
        # Synthetic transforms and limits are confined to the mock backend.
        result.update(task_from_base=np.eye(4).tolist(), flange_from_tcp=np.eye(4).tolist(),
                      velocity_limits=config["mock"]["velocity_limits"],
                      acceleration_limits=config["mock"]["acceleration_limits"],
                      workspace=config["mock"]["workspace"])
    return result


def validate_hardware_config(config):
    missing = []
    for arm, settings in config["arms"].items():
        for name in ("can_channel", "hand_can_channel", "task_from_base", "flange_from_tcp",
                     "velocity_limits", "acceleration_limits", "workspace"):
            if settings.get(name) is None:
                missing.append(f"arms.{arm}.{name}")
        for name in ("open", "closed"):
            if settings.get("hand_poses", {}).get(name) is None:
                missing.append(f"arms.{arm}.hand_poses.{name}")
    if missing:
        raise ValueError("Hardware parameters are not filled: " + ", ".join(missing))
    channels = []
    for arm, settings in config["arms"].items():
        channels.extend([settings["can_channel"], settings["hand_can_channel"]])
        transform(settings["task_from_base"])
        transform(settings["flange_from_tcp"])
        for name in ("velocity_limits", "acceleration_limits"):
            if np.any(joint_vector(settings[name], name) <= 0):
                raise ValueError(f"{arm}.{name} must be positive.")
    if len(set(channels)) != len(channels):
        raise ValueError("Each configured device needs its own explicit CAN channel.")
