"""Configuration loading and execution-time validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .geometry import valid_rotation


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/quest_teleop.yaml"


def load_config(path=None) -> dict:
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Quest teleop config must be a mapping")
    config = deepcopy(config)
    for side in ("left", "right"):
        side_config = config.get("sides", {}).get(side)
        if not isinstance(side_config, dict):
            raise ValueError(f"missing sides.{side} configuration")
        rotation = np.asarray(side_config.get("base_from_quest_rotation"), dtype=float)
        if not valid_rotation(rotation):
            raise ValueError(f"sides.{side}.base_from_quest_rotation must be SO(3)")
        side_config["base_from_quest_rotation"] = rotation
    return config


def validate_hardware_config(config: dict, sides, include_hands: bool) -> None:
    channels = []
    for side in sides:
        side_config = config["sides"][side]
        arm_channel = side_config.get("arm_can")
        if not arm_channel or str(arm_channel).startswith("SET_"):
            raise ValueError(f"configure sides.{side}.arm_can before --execute")
        channels.append(str(arm_channel))
        if include_hands:
            hand_channel = side_config.get("hand_can")
            if not hand_channel or str(hand_channel).startswith("SET_"):
                raise ValueError(f"configure sides.{side}.hand_can before --execute")
            channels.append(str(hand_channel))
    if len(channels) != len(set(channels)):
        raise ValueError(f"hardware CAN channels must be unique, got {channels}")


def serializable_config(config):
    if isinstance(config, np.ndarray):
        return config.tolist()
    if isinstance(config, dict):
        return {str(key): serializable_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [serializable_config(value) for value in config]
    return config
