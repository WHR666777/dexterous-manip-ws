"""独立规划/标定配置的读取与校验，不创建设备连接。

新 TaskSession 当前不加载这些配置；保留给几何与 TOPPRA 规划使用。
其中 mock 配置只提供离线规划数值，不再对应一个假设备执行后端。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .geometry import transform


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/cloth_agent.yaml"
MOCK_CONFIG = DEFAULT_CONFIG.with_name("cloth_agent_mock.yaml")


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


def _relative_file(config, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(config["config_dir"]) / path).resolve()


def _load_calibration(config):
    reference = config.get("calibration")
    if reference is None:
        return
    path = _relative_file(config, reference["file"])
    with path.open() as stream:
        calibration = yaml.safe_load(stream)
    arm = reference["arm"]
    if arm not in config["arms"]:
        raise ValueError(f"Calibration arm {arm!r} is not configured.")
    if calibration["hand_side"] != arm:
        raise ValueError(f"Calibration hand_side must match configured arm {arm!r}.")
    settings = config["arms"][arm]
    task_base = transform(settings["task_from_base"])
    flange_tcp = transform(calibration["flange_from_tcp"])
    config["observations"]["task_from_camera"] = (
        task_base @ transform(calibration["base_from_camera"])).tolist()
    settings["flange_from_tcp"] = flange_tcp.tolist()
    settings.setdefault("hand_poses", {})["closed"] = calibration["hand_pose_raw"]
    roll_pitch = np.asarray(calibration["reference_roll_pitch_deg"], dtype=float)
    if roll_pitch.shape != (2,) or not np.isfinite(roll_pitch).all():
        raise ValueError("Calibration reference_roll_pitch_deg must contain two finite angles.")
    base_reference = Rotation.from_euler("xyz", [*roll_pitch, 0], degrees=True).as_matrix()
    settings["reference_tcp_rotation"] = (task_base[:3, :3] @ base_reference @ flange_tcp[:3, :3]).tolist()
    reference["file"] = str(path)
    reference["fit"] = calibration.get("fit")


def _resolve_workspaces(config):
    for settings in config["arms"].values():
        workspace = settings.get("workspace")
        if not isinstance(workspace, dict):
            continue  # Explicit [min_xyz, max_xyz] bounds remain supported.
        from robot_control.nero_ik import NeroIK

        start_file = _relative_file(config, workspace["start_pose_file"])
        with start_file.open() as stream:
            start = yaml.safe_load(stream)
        joints = joint_vector(start["joint_positions_rad"], "workspace start joints")
        half_side = positive(workspace["cube_side_m"], "workspace.cube_side_m") / 2
        tcp = (transform(settings["task_from_base"]) @ NeroIK().forward(joints)
               @ transform(settings["flange_from_tcp"]))
        center = tcp[:3, 3]
        settings["workspace"] = [(center - half_side).tolist(), (center + half_side).tolist()]


def load_config(path=DEFAULT_CONFIG) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["config_dir"] = str(path.parent)
    if config["backend"] not in ("mock", "hardware"):
        raise ValueError("backend must be mock or hardware.")
    positive(config["control_hz"], "control_hz")
    # 此处仅校验独立规划器仍使用的参数；旧执行器的到位/录制参数已随主流程删除。
    for name in ("max_ik_step_rad", "position_step_m", "rotation_step_rad", "joint_step_rad"):
        positive(config["motion"][name], name)
    if not config["arms"] or any(name not in ("left", "right") for name in config["arms"]):
        raise ValueError("Configure left and/or right arms.")
    _load_calibration(config)
    _resolve_workspaces(config)
    if config["observations"].get("task_from_camera") is not None:
        transform(config["observations"]["task_from_camera"])
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
        workspace = np.asarray(settings["workspace"], dtype=float)
        if (workspace.shape != (2, 3) or not np.isfinite(workspace).all()
                or np.any(workspace[0] >= workspace[1])):
            raise ValueError(f"{arm}.workspace must contain finite lower/upper XYZ bounds.")
    if len(set(channels)) != len(channels):
        raise ValueError("Each configured device needs its own explicit CAN channel.")
