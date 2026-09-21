"""NERO flange control session with explicit proposal/commit semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from robot_control.nero_ik import NeroIK, pose_error
from .geometry import pose6_to_matrix, pose_distance, relative_flange_target, valid_transform


@dataclass
class ArmFeedback:
    joints: np.ndarray
    flange: np.ndarray
    tcp: np.ndarray
    torques: np.ndarray
    status: dict
    enabled: tuple[bool, ...]
    fresh: bool
    sampled_monotonic: float = field(default_factory=time.monotonic)
    joint_timestamps: np.ndarray = field(default_factory=lambda: np.zeros(7))
    state_timestamps: np.ndarray = field(default_factory=lambda: np.zeros(8))
    flange_timestamps: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ArmProposal:
    side: str
    code: str
    target_flange: np.ndarray | None = None
    ik_joints: np.ndarray | None = None
    command_joints: np.ndarray | None = None
    fk_position_error_m: float | None = None
    fk_rotation_error_rad: float | None = None
    generated_monotonic: float = field(default_factory=time.monotonic)

    @property
    def ready(self) -> bool:
        return self.code == "READY" and self.command_joints is not None


def _status_ready(status: dict, enabled) -> bool:
    return (
        status.get("arm_status") == 0
        and status.get("ctrl_mode") == 1
        and not any((status.get("err_status") or {}).values())
        and len(enabled) == 7
        and all(enabled)
    )


def _timestamps_are_fresh(stamps, now_wall, max_age_s, max_skew_s) -> bool:
    stamps = np.asarray(stamps, dtype=np.float64)
    if not stamps.size or not np.isfinite(stamps).all() or np.any(stamps <= 0):
        return False
    ages = now_wall - stamps
    return bool(
        np.min(ages) >= -0.01
        and np.max(ages) <= max_age_s
        and np.ptp(stamps) <= max_skew_s
    )


def read_arm_feedback(arm, max_age_s: float = 0.25, max_skew_s: float = 0.10) -> ArmFeedback:
    """Read one validated snapshot without pretending SDK reads are atomic."""
    messages = arm.get_raw_motor_states()
    joints = np.asarray([message.msg.position for message in messages], dtype=np.float64)
    torques = np.asarray([message.msg.torque for message in messages], dtype=np.float64)
    stamps = np.asarray([message.timestamp for message in messages], dtype=np.float64)
    now_wall = time.time()
    raw_status = arm.get_raw_arm_status()
    status = arm.get_arm_status()
    driver_states = [arm._driver.get_driver_states(i) for i in range(1, 8)]
    state_stamps = np.asarray(
        [raw_status.timestamp, *[message.timestamp for message in driver_states]],
        dtype=np.float64,
    )
    enabled = tuple(bool(arm._driver.get_joint_enable_status(i)) for i in range(1, 8))
    flange = pose6_to_matrix(arm.get_flange_pose())
    parser = getattr(arm._driver, "_parser", None)
    frame_names = ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz")
    if parser is None or not all(getattr(parser, name, None) is not None for name in frame_names):
        raise RuntimeError("NERO V1.11 flange component timestamps are unavailable")
    flange_stamps = np.asarray(
        [getattr(parser, name).timestamp for name in frame_names], dtype=np.float64
    )
    try:
        tcp = pose6_to_matrix(arm.get_tcp_pose())
    except Exception:
        tcp = np.full((4, 4), np.nan)
    numeric_ok = (
        joints.shape == (7,)
        and torques.shape == (7,)
        and stamps.shape == (7,)
        and np.isfinite(np.r_[joints, torques, stamps]).all()
    )
    fresh = bool(
        numeric_ok
        and valid_transform(flange)
        and _timestamps_are_fresh(stamps, now_wall, max_age_s, max_skew_s)
        and _timestamps_are_fresh(state_stamps, now_wall, max_age_s, max_skew_s)
        and _timestamps_are_fresh(flange_stamps, now_wall, max_age_s, max_skew_s)
        and _status_ready(status, enabled)
    )
    return ArmFeedback(
        joints, flange, tcp, torques, status, enabled, fresh,
        joint_timestamps=stamps,
        state_timestamps=state_stamps,
        flange_timestamps=flange_stamps,
    )


def apply_sdk_joint_limits(ik: NeroIK, sdk_config: dict) -> None:
    """Require identical joint order and use the URDF/SDK limit intersection."""
    names = list(sdk_config.get("joint_names", []))
    if names != ik.joint_names:
        raise ValueError(f"SDK/URDF joint order mismatch: {names} != {ik.joint_names}")
    limits = sdk_config.get("joint_limits", {})
    try:
        sdk_limits = np.asarray([limits[name] for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SDK joint limits are incomplete") from exc
    if (
        sdk_limits.shape != (7, 2)
        or not np.isfinite(sdk_limits).all()
        or np.any(sdk_limits[:, 0] >= sdk_limits[:, 1])
    ):
        raise ValueError("SDK joint limits must be seven finite lower/upper pairs")
    intersection = np.column_stack((
        np.maximum(ik.limits[:, 0], sdk_limits[:, 0]),
        np.minimum(ik.limits[:, 1], sdk_limits[:, 1]),
    ))
    if np.any(intersection[:, 0] >= intersection[:, 1]):
        raise ValueError("SDK and URDF joint limits have an empty intersection")
    ik.limits = intersection


class VirtualArm:
    """Dry-run arm model; it never imports or constructs the hardware SDK."""

    def __init__(self, ik: NeroIK, joints=None):
        self.ik = ik
        self.joints = np.zeros(7) if joints is None else np.asarray(joints, dtype=float)
        if not ik.in_limits(self.joints):
            midpoint = np.mean(ik.limits, axis=1)
            self.joints = midpoint
        self.sent: list[np.ndarray] = []

    def feedback(self) -> ArmFeedback:
        flange = self.ik.forward(self.joints)
        now_wall = time.time()
        return ArmFeedback(
            joints=self.joints.copy(),
            flange=flange,
            tcp=flange.copy(),
            torques=np.zeros(7),
            status={"arm_status": 0, "ctrl_mode": 1, "motion_status": 0, "err_status": {}},
            enabled=(True,) * 7,
            fresh=True,
            joint_timestamps=np.full(7, now_wall),
            state_timestamps=np.full(8, now_wall),
            flange_timestamps=np.full(3, now_wall),
        )

    def move_js(self, joints):
        joints = np.asarray(joints, dtype=float)
        if not self.ik.in_limits(joints):
            raise ValueError("dry-run command is outside IK limits")
        self.joints = joints.copy()
        self.sent.append(joints.copy())


class ArmControlSession:
    """One side of differential Quest wrist to NERO flange control."""

    def __init__(
        self,
        side: str,
        base_from_quest_rotation,
        *,
        position_scale: float = 0.5,
        max_tracker_position_step_m: float = 0.04,
        max_tracker_rotation_step_deg: float = 12.0,
        max_target_position_step_m: float = 0.01,
        max_target_rotation_step_deg: float = 5.0,
        max_joint_step_deg: float = 5.0,
        ik: NeroIK | None = None,
    ):
        self.side = side
        self.mapping = np.asarray(base_from_quest_rotation, dtype=float)
        self.position_scale = float(position_scale)
        self.max_tracker_position_step_m = float(max_tracker_position_step_m)
        self.max_tracker_rotation_step_rad = np.deg2rad(max_tracker_rotation_step_deg)
        self.max_target_position_step_m = float(max_target_position_step_m)
        self.max_target_rotation_step_rad = np.deg2rad(max_target_rotation_step_deg)
        self.max_joint_step_rad = np.deg2rad(max_joint_step_deg)
        self.ik = NeroIK() if ik is None else ik
        self.wrist_anchor = self.flange_anchor = self.previous_wrist = None
        self.previous_target = self.previous_command = None
        self.armed = False
        self.paused = True
        self.fault_latched = False
        self.pause_reason = "NOT_ANCHORED"

    @property
    def anchored(self) -> bool:
        return self.wrist_anchor is not None

    def reanchor(self, wrist_world, feedback: ArmFeedback) -> bool:
        if (
            not valid_transform(wrist_world)
            or not feedback.fresh
            or not valid_transform(feedback.flange)
            or not self.ik.in_limits(feedback.joints)
            or feedback.status.get("motion_status") != 0
            or self.fault_latched
        ):
            return False
        fk_position, fk_rotation = pose_error(self.ik.forward(feedback.joints), feedback.flange)
        if fk_position > 0.01 or fk_rotation > np.deg2rad(5):
            return False
        self.wrist_anchor = np.asarray(wrist_world).copy()
        self.flange_anchor = feedback.flange.copy()
        self.previous_wrist = self.wrist_anchor.copy()
        self.previous_target = self.flange_anchor.copy()
        self.previous_command = feedback.joints.copy()
        self.paused = True
        self.armed = True
        self.pause_reason = "READY_TO_ENABLE"
        return True

    def toggle(self) -> bool:
        if not self.anchored or self.fault_latched:
            return False
        if self.paused:
            if not self.armed:
                return False
            self.paused = False
            self.pause_reason = ""
            return True
        self.pause("OPERATOR_PAUSE", require_reanchor=True)
        return False

    def pause(self, reason: str, *, require_reanchor: bool = True, fault: bool = False):
        self.paused = True
        self.pause_reason = reason
        if require_reanchor:
            self.armed = False
        if fault:
            self.fault_latched = True

    def propose(self, wrist_world, feedback: ArmFeedback) -> ArmProposal:
        if self.fault_latched:
            return ArmProposal(self.side, "FAULT_LATCHED")
        if not feedback.fresh or not self.ik.in_limits(feedback.joints):
            self.pause("ROBOT_FEEDBACK_INVALID", fault=True)
            return ArmProposal(self.side, "ROBOT_FEEDBACK_INVALID")
        if self.paused:
            return ArmProposal(self.side, "PAUSED")
        if not valid_transform(wrist_world):
            self.pause("TRACKING_INVALID")
            return ArmProposal(self.side, "TRACKING_INVALID")
        tracker_position, tracker_rotation = pose_distance(wrist_world, self.previous_wrist)
        self.previous_wrist = np.asarray(wrist_world).copy()
        if (
            tracker_position > self.max_tracker_position_step_m
            or tracker_rotation > self.max_tracker_rotation_step_rad
        ):
            self.pause("TRACKER_JUMP")
            return ArmProposal(self.side, "TRACKER_JUMP")
        target = relative_flange_target(
            self.wrist_anchor,
            wrist_world,
            self.flange_anchor,
            self.mapping,
            self.position_scale,
        )
        position_step, rotation_step = pose_distance(target, self.previous_target)
        if position_step > self.max_target_position_step_m or rotation_step > self.max_target_rotation_step_rad:
            self.pause("TARGET_JUMP")
            return ArmProposal(self.side, "TARGET_JUMP", target_flange=target)
        solution = self.ik.solve(target, feedback.joints)
        if solution is None:
            return ArmProposal(self.side, self.ik.last_diagnostics.get("code", "IK_FAILED"), target_flange=target)
        command = self.previous_command + np.clip(
            solution - self.previous_command, -self.max_joint_step_rad, self.max_joint_step_rad
        )
        if (
            not self.ik.in_limits(command)
            or np.max(np.abs(command - feedback.joints)) > self.max_joint_step_rad + 1e-12
        ):
            return ArmProposal(self.side, "COMMAND_JUMP", target_flange=target, ik_joints=solution)
        fk_position, fk_rotation = pose_error(self.ik.forward(solution), target)
        return ArmProposal(
            self.side,
            "READY",
            target_flange=target,
            ik_joints=solution,
            command_joints=command,
            fk_position_error_m=fk_position,
            fk_rotation_error_rad=fk_rotation,
        )

    def commit(self, proposal: ArmProposal):
        if not proposal.ready:
            raise ValueError("only READY proposals can be committed")
        self.previous_target = proposal.target_flange.copy()
        self.previous_command = proposal.command_joints.copy()

    def validate_presend(
        self, proposal: ArmProposal, feedback: ArmFeedback, max_target_age_s: float = 0.25
    ) -> None:
        if not proposal.ready:
            raise ValueError("proposal is not READY")
        if time.monotonic() - proposal.generated_monotonic > max_target_age_s:
            raise RuntimeError("target expired before dispatch")
        if not feedback.fresh or not self.ik.in_limits(feedback.joints):
            raise RuntimeError("NERO feedback/state became invalid before dispatch")
        if np.max(np.abs(proposal.command_joints - feedback.joints)) > self.max_joint_step_rad + 1e-12:
            raise ValueError("command exceeds the joint step from current feedback")


def dispatch_arm_proposals(items) -> None:
    """Prevalidate all sides, then dispatch; any partial failure is explicit."""
    items = list(items)
    if not items or not all(proposal.ready for _, _, proposal in items):
        raise ValueError("all arm proposals must be READY before dispatch")
    sent = []
    try:
        # Re-read and validate every side after IK, before sending either side.
        # This prevents a stale pre-IK snapshot from authorizing a command.
        for session, arm, proposal in items:
            feedback = arm.feedback() if isinstance(arm, VirtualArm) else read_arm_feedback(arm)
            session.validate_presend(proposal, feedback)
            validator = getattr(arm, "validate_joint_command", None)
            if validator is not None:
                validator(proposal.command_joints, max_joint_delta=session.max_joint_step_rad)
        for session, arm, proposal in items:
            arm.move_js(proposal.command_joints.copy())
            sent.append(session.side)
        for session, _arm, proposal in items:
            session.commit(proposal)
    except Exception as exc:
        for session, _arm, _proposal in items:
            session.pause("PARTIAL_SEND_FAILURE", fault=True)
        raise RuntimeError(f"arm dispatch failed after sides={sent}: {exc}") from exc
