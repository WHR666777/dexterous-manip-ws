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
    if "arms" in calibration:
        config["observations"]["task_from_camera"] = transform(calibration["task_from_camera"]).tolist()
        for arm, settings in config["arms"].items():
            if arm not in calibration["arms"]:
                raise ValueError(f"Combined calibration is missing arm {arm!r}.")
            values = calibration["arms"][arm]
            for name in ("task_from_base", "flange_from_tcp"):
                settings[name] = transform(values[name]).tolist()
            angles = np.asarray(values["reference_roll_pitch_deg"], dtype=float)
            if angles.shape != (2,) or not np.isfinite(angles).all():
                raise ValueError(f"Invalid reference_roll_pitch_deg for {arm}.")
            settings["reference_roll_pitch_deg"] = angles.tolist()
            settings["calibration_hand_pose_raw"] = deepcopy(values["hand_pose_raw"])
            settings.setdefault("hand_poses", {}).update(deepcopy(calibration["hand_poses"]))
        reference["file"] = str(path)
        reference["fit"] = {arm: values.get("fit") for arm, values in calibration["arms"].items()}
        return
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
    # 标定参考角只描述工具轴的定义；启动目标从 startup.tcp_poses 单独读取。
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
        settings["start_pose_file"] = str(start_file)
        settings["start_joint_positions"] = joints.tolist()
        settings["reference_flange_yaw_rad"] = float(Rotation.from_matrix(base_flange[:3, :3]).as_euler("xyz")[2])
        # 禁用启动且没有显式目标时，计算接口使用记录姿态的工具朝向。
        # 启用启动时，下方必须用独立配置的 TCP 目标覆盖，不能借用标定参考角。
        settings["reference_tcp_rotation"] = tcp[:3, :3].tolist()


def _resolve_startup_poses(config):
    startup = config["startup"]
    poses = startup.get("tcp_poses", {})
    if not isinstance(poses, dict) or any(name not in config["arms"] for name in poses):
        raise ValueError("startup.tcp_poses must map configured arms to task TCP poses.")
    reflection = np.diag([1., -1., 1.])
    for name, value in poses.items():
        if isinstance(value, dict):
            source = value.get("mirror_position_of")
            if (not isinstance(source, str) or source not in poses or source == name
                    or isinstance(poses[source], dict)):
                raise ValueError(f"startup.tcp_poses.{name}.mirror_position_of must name an explicit TCP pose.")
            plane_y = float(value.get("plane_y_m", np.nan))
            if not np.isfinite(plane_y):
                raise ValueError("startup mirror plane_y_m must be finite.")
            original = transform(poses[source])
            target = np.eye(4)
            target[:3, 3] = reflection @ original[:3, 3] + [0., 2 * plane_y, 0.]
            # 只镜像位置；公共 task 中两侧 TCP 朝向完全相同。
            target[:3, :3] = original[:3, :3]
        else:
            target = transform(value)
        settings = config["arms"][name]
        settings["initial_grasp_tcp"] = target.tolist()
        # top_down_pose 的 yaw=0 以显式启动 TCP 朝向为准，与标定参考角独立。
        settings["reference_tcp_rotation"] = target[:3, :3].tolist()
    if startup["enabled"] and any(
            "initial_grasp_tcp" not in settings or "start_joint_positions" not in settings
            for settings in config["arms"].values()):
        raise ValueError("Startup requires startup.tcp_poses for every arm and workspace.start_pose_file.")


def _resolve_act(config):
    """ACT 是可选能力；只解析配置，不加载模型、不创建设备或子进程。"""
    act = config.get("act")
    if act is None:
        return
    if not isinstance(act, dict):
        raise ValueError("act must be a mapping or null.")
    if set(config["arms"]) != {"left", "right"}:
        raise ValueError("ACT requires both left and right arms.")
    camera = config.get("camera", {})
    if not camera.get("enabled") or (camera.get("height"), camera.get("width")) != (480, 640):
        raise ValueError("ACT requires camera.enabled=true and 640x480 RGB.")
    for name in ("python", "script"):
        if not isinstance(act.get(name), str) or not act[name].strip():
            raise ValueError(f"act.{name} must be a non-empty path.")
        act[name] = str(_relative_file(config, act[name]))
    policies = act.get("policies")
    if (not isinstance(policies, dict) or not policies
            or any(not isinstance(name, str) or not name.strip()
                   or not isinstance(path, str) or not path.strip() for name, path in policies.items())):
        raise ValueError("act.policies must map policy names to checkpoint directories.")
    act["policies"] = {name: str(_relative_file(config, path)) for name, path in policies.items()}
    if act.setdefault("device", "cuda") not in {"cuda", "cpu"}:
        raise ValueError("act.device must be cuda or cpu.")
    for name, default in (("startup_timeout_s", 120.), ("inference_timeout_s", 5.),
                          ("control_hz", 20.), ("hand_feedback_poll_hz", 20.),
                          ("hand_feedback_max_age_s", .25)):
        act[name] = positive(act.get(name, default), f"act.{name}")
    if act["control_hz"] != float(config["control_hz"]):
        raise ValueError("act.control_hz must match the cloth control_hz and policy recording rate.")
    step = act.setdefault("hand_max_raw_step", 10)
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= 255:
        raise ValueError("act.hand_max_raw_step must be an integer in [1, 255].")


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
    # 空载手指到位监督：重试次数不含首次发送；仅重发原预设，不加力或改目标。
    gripper = config.setdefault("gripper", {})
    for name, default in (("timeout_s", 2.), ("poll_interval_s", .1)):
        gripper[name] = positive(gripper.get(name, default), f"gripper.{name}")
    retries = gripper.setdefault("max_retries", 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("gripper.max_retries must be a nonnegative integer.")
    tolerance = gripper.setdefault("tolerance_raw", 5)
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or not 0 <= tolerance <= 255:
        raise ValueError("gripper.tolerance_raw must be an integer in [0, 255].")
    # 此处仅校验独立规划器仍使用的参数；旧执行器的到位/录制参数已随主流程删除。
    for name in ("max_ik_step_rad", "position_step_m", "rotation_step_rad", "joint_step_rad"):
        positive(config["motion"][name], name)
    # 反馈到位与暂停收尾仍由设备执行层判断；默认值只补到内存，不改写现场标定文件。
    # joint_tolerance_rad：理想精度（超过此值发出警告但继续执行）。
    # joint_arrival_tolerance_rad：实测关节到位阈值，必须与新鲜 J/0 同时满足。
    # J/0 也可能出现在运动启动前；误差尚大时继续监督，超过执行超时才失败。
    for name, default in (("joint_tolerance_rad", .005), ("joint_arrival_tolerance_rad", .015),
                          ("completion_timeout_s", 30.)):
        config["motion"][name] = positive(config["motion"].get(name, default), name)
    # 厂家 L 采用 check_move_linear_zbx.py 已实测的空闲等待、启动确认和
    # TCP 到位参数。注意位置阈值为 0.02 m（2 cm），不是旧 IK 的 2 mm。
    for name, default in (("linear_pre_idle_timeout_s", 10.), ("linear_start_timeout_s", 1.),
                          ("linear_position_tolerance_m", .02),
                          ("linear_angle_tolerance_rad", float(np.deg2rad(2)))):
        config["motion"][name] = positive(config["motion"].get(name, default), name)
    attempts = config["motion"].setdefault("linear_max_attempts", 3)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("motion.linear_max_attempts must be a positive integer.")
    # J/L 共用现有固件速度百分比，与 TOPPRA 使用的 rad/s 上限不是同一物理量。
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
    # YAML 的公共参数可用 anchor，但每侧运行参数必须各自拥有可变数据。
    config["arms"] = {name: deepcopy(settings) for name, settings in config["arms"].items()}
    _load_calibration(config)
    _resolve_workspaces(config)
    startup = config.setdefault("startup", {"enabled": False})
    if not isinstance(startup.get("enabled"), bool):
        raise ValueError("startup.enabled must be a boolean.")
    startup["timeout_s"] = positive(startup.get("timeout_s", 120.), "startup.timeout_s")
    _resolve_startup_poses(config)
    _resolve_act(config)
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
