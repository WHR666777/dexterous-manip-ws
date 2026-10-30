"""独立规划/标定配置的读取与校验，不创建设备连接。

TaskSession 启动时显式加载标定与运动配置；只有控制进程可以连接设备。
其中 mock 配置只提供离线规划数值，不再对应一个假设备执行后端。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .geometry import transform
from .paths import DEFAULT_CONFIG


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
    # 此处只加载示教倾角；实际抓取零点必须结合记录起始位姿的 yaw 再构造。
    settings["reference_roll_pitch_deg"] = roll_pitch.tolist()
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
        base_flange = NeroIK().forward(joints)
        task_base = transform(settings["task_from_base"])
        flange_tcp = transform(settings["flange_from_tcp"])
        tcp = task_base @ base_flange @ flange_tcp
        center = tcp[:3, 3]
        settings["workspace"] = [(center - half_side).tolist(), (center + half_side).tolist()]
        if "reference_roll_pitch_deg" in settings:
            # 与 wrist-only 一样：roll/pitch 采用示教参考，yaw 保留记录起始关节
            # 经 FK 得到的法兰 yaw。标定内部为了计算工具旋转而补入的 yaw=0
            # 不是现场的初始航向，不能直接作为 agent 的零点。
            yaw = float(Rotation.from_matrix(base_flange[:3, :3]).as_euler("xyz")[2])
            initial_flange = base_flange.copy()
            initial_flange[:3, :3] = Rotation.from_euler(
                "xyz", [*np.deg2rad(settings["reference_roll_pitch_deg"]), yaw]).as_matrix()
            initial_tcp = task_base @ initial_flange @ flange_tcp
            # 初始抓取位姿保留记录的法兰位置；TCP 位置因固定工具偏移随倾角改变。
            # 后续 top_down_pose 的 yaw 是绕 task +Z 左乘的增量，不再是绝对法兰 yaw。
            settings["start_pose_file"] = str(start_file)
            settings["start_joint_positions"] = joints.tolist()
            settings["initial_grasp_tcp"] = initial_tcp.tolist()
            settings["reference_flange_yaw_rad"] = yaw
            settings["reference_tcp_rotation"] = initial_tcp[:3, :3].tolist()


def load_config(path=DEFAULT_CONFIG) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["config_dir"] = str(path.parent)
    # 模型参数只交给 Codex，不影响机器人控制。允许 null 沿用本机设置；
    # 不在这里硬编码模型列表或各模型支持的强度，具体支持情况由 Codex 校验。
    for name in ("model", "model_reasoning_effort"):
        value = config.setdefault(name, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} 必须为非空字符串或 null Must be a non-empty string or null")
    if config["backend"] not in ("mock", "hardware"):
        raise ValueError("backend must be mock or hardware.")
    positive(config["control_hz"], "control_hz")
    # 此处仅校验独立规划器仍使用的参数；旧执行器的到位/录制参数已随主流程删除。
    for name in ("max_ik_step_rad", "position_step_m", "rotation_step_rad", "joint_step_rad"):
        positive(config["motion"][name], name)
    # 反馈到位与暂停收尾仍由设备执行层判断；默认值只补到内存，不改写现场标定文件。
    # joint_tolerance_rad：理想精度（超过此值发出警告但继续执行）。
    # joint_arrival_tolerance_rad：到位失败阈值（固件报到位后关节偏差超过此值立即终止）。
    for name, default in (("joint_tolerance_rad", .005), ("joint_arrival_tolerance_rad", .015),
                          ("completion_timeout_s", 30.)):
        config["motion"][name] = positive(config["motion"].get(name, default), name)
    # 普通 J 的固件速度百分比，与 TOPPRA 使用的 rad/s 上限不是同一物理量。
    # 使用本次实机单目标检查已验证的 10%；不由 agent cell 临时覆盖。
    speed = config["motion"].setdefault("joint_speed_percent", 10)
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 100:
        raise ValueError("motion.joint_speed_percent must be an integer in [1, 100].")
    # Nero 1.11 的实机对照使用 1 ms 主机帧节拍，避免七轴目标的四帧集中
    # 下发后整组目标被忽略。0 仅供反向对照；它不是几何分段停稳时间。
    gap = float(config["motion"].get("joint_frame_interval_s", .001))
    if not np.isfinite(gap) or gap < 0:
        raise ValueError("motion.joint_frame_interval_s must be finite and nonnegative.")
    config["motion"]["joint_frame_interval_s"] = gap
    if not config["arms"] or any(name not in ("left", "right") for name in config["arms"]):
        raise ValueError("Configure left and/or right arms.")
    _load_calibration(config)
    _resolve_workspaces(config)
    startup = config.setdefault("startup", {"enabled": False})
    if not isinstance(startup.get("enabled"), bool):
        raise ValueError("startup.enabled must be a boolean.")
    startup["timeout_s"] = positive(startup.get("timeout_s", 120.), "startup.timeout_s")
    if startup["enabled"] and any("initial_grasp_tcp" not in settings for settings in config["arms"].values()):
        raise ValueError("Startup requires calibrated reference angles and workspace.start_pose_file.")
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
        for name in ("can_channel", "hand_can_channel", "task_from_base", "flange_from_tcp", "reference_tcp_rotation",
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
