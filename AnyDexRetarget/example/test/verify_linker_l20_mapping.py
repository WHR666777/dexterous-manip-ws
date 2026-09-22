"""Verify Linker L20 retarget-qpos, MuJoCo actuator and model kinematics.

This is intentionally name based.  Pinocchio and MuJoCo do not enumerate the
L20 joints in the same order, so a positional slice can look plausible while
moving the wrong finger/joint.

Run from the repository root:

    python example/test/verify_linker_l20_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    validate_mujoco_actuator_mapping,
)

CONFIG_PATH = EXAMPLE_ROOT / "config/adaptive/pico4/pico4_linker_l20.yaml"
MIMIC_JOINTS = {
    "thumb_ip": ("thumb_mcp", 1.03),
    "index_dip": ("index_pip", 0.89),
    "middle_dip": ("middle_pip", 0.89),
    "ring_dip": ("ring_pip", 0.89),
    "pinky_dip": ("pinky_pip", 0.89),
}
POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-5


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _mujoco_body_pose(model, data, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"MuJoCo body not found: {body_name}")
    return _homogeneous(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])


def _pinocchio_link_pose(robot, link_name: str) -> np.ndarray:
    link_id = robot.get_link_index(link_name)
    return np.asarray(robot.get_link_pose(link_id), dtype=np.float64)


def _set_mujoco_joint_qpos(model, data, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"MuJoCo joint not found: {joint_name}")
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_address] = value


def _rotation_error_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def _common_body_names(robot, model) -> list[str]:
    pinocchio_frames = {frame.name for frame in robot.model.frames}
    mujoco_bodies = {
        name
        for body_id in range(1, model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
    }
    return sorted(pinocchio_frames & mujoco_bodies)


def verify_side(side: str) -> tuple[float, float]:
    hand_cfg = ROBOT_HAND_CONFIGS["linker_l20"]
    retargeter = Retargeter.from_yaml(str(CONFIG_PATH), side)
    robot = retargeter.optimizer.robot
    pin_joint_names = [str(name) for name in robot.dof_joint_names]
    if len(pin_joint_names) != len(set(pin_joint_names)):
        raise AssertionError(f"Duplicate Pinocchio qpos names: {pin_joint_names}")

    model = mujoco.MjModel.from_xml_path(hand_cfg["model_path"](side))
    data = mujoco.MjData(model)
    actuator_joint_names = validate_mujoco_actuator_mapping(model, hand_cfg)

    print(f"\n[{side}] Pinocchio qpos order ({len(pin_joint_names)}):")
    for qpos_id, name in enumerate(pin_joint_names):
        print(f"  qpos[{qpos_id:02d}] {name}")
    print(f"[{side}] MuJoCo ctrl order ({len(actuator_joint_names)}):")
    for ctrl_id, name in enumerate(actuator_joint_names):
        print(f"  ctrl[{ctrl_id:02d}] {name}")

    # Every independent qpos pulse must land only on the same-named actuator.
    for expected_ctrl_id, joint_name in enumerate(actuator_joint_names):
        qpos = np.zeros(len(pin_joint_names), dtype=np.float64)
        qpos[pin_joint_names.index(joint_name)] = 0.25
        ctrl = map_retarget_qpos(qpos, hand_cfg, pin_joint_names)
        nonzero = np.flatnonzero(np.abs(ctrl) > 1e-12).tolist()
        if nonzero != [expected_ctrl_id] or not np.isclose(ctrl[expected_ctrl_id], 0.25):
            raise AssertionError(
                f"{side}: {joint_name} qpos pulse mapped to ctrl indices {nonzero}, "
                f"expected [{expected_ctrl_id}]"
            )

    # Drive each MuJoCo ctrl channel independently.  Only the joint bound to
    # that channel may move among the 16 independent joints (mimic joints are
    # intentionally not included in this list).
    for ctrl_id, joint_name in enumerate(actuator_joint_names):
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0.0
        data.ctrl[ctrl_id] = min(0.2, float(model.actuator_ctrlrange[ctrl_id, 1]) * 0.8)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        moved = []
        for candidate in actuator_joint_names:
            candidate_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, candidate
            )
            value = float(data.qpos[int(model.jnt_qposadr[candidate_id])])
            if abs(value) > 1e-3:
                moved.append(candidate)
        if moved != [joint_name]:
            raise AssertionError(
                f"{side}: ctrl[{ctrl_id}] for {joint_name} moved {moved}"
            )
    print(f"[{side}] all 16 independent MuJoCo actuator pulses passed")

    origin_name = retargeter.optimizer.origin_link_name
    common_bodies = _common_body_names(robot, model)
    if origin_name not in common_bodies:
        raise AssertionError(f"Common origin body missing: {origin_name}")

    max_position_error = 0.0
    max_rotation_error = 0.0
    worst_position = None
    worst_rotation = None
    qpos_index = {name: index for index, name in enumerate(pin_joint_names)}

    # Compare URDF and MJCF forward kinematics for each independent actuator.
    for driven_joint in actuator_joint_names:
        pin_qpos = np.zeros(len(pin_joint_names), dtype=np.float64)
        pin_qpos[qpos_index[driven_joint]] = 0.3
        for mimic_name, (source_name, multiplier) in MIMIC_JOINTS.items():
            pin_qpos[qpos_index[mimic_name]] = pin_qpos[qpos_index[source_name]] * multiplier
        robot.compute_forward_kinematics(pin_qpos)

        mujoco.mj_resetData(model, data)
        _set_mujoco_joint_qpos(model, data, driven_joint, 0.3)
        for mimic_name, (source_name, multiplier) in MIMIC_JOINTS.items():
            source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, source_name)
            source_value = data.qpos[int(model.jnt_qposadr[source_id])]
            _set_mujoco_joint_qpos(model, data, mimic_name, source_value * multiplier)
        mujoco.mj_forward(model, data)

        pin_origin_inv = np.linalg.inv(_pinocchio_link_pose(robot, origin_name))
        mj_origin_inv = np.linalg.inv(_mujoco_body_pose(model, data, origin_name))
        for body_name in common_bodies:
            pin_relative = pin_origin_inv @ _pinocchio_link_pose(robot, body_name)
            mj_relative = mj_origin_inv @ _mujoco_body_pose(model, data, body_name)
            position_error = float(
                np.linalg.norm(pin_relative[:3, 3] - mj_relative[:3, 3])
            )
            rotation_error = _rotation_error_angle(
                pin_relative[:3, :3], mj_relative[:3, :3]
            )
            if position_error > max_position_error:
                max_position_error = position_error
                worst_position = (driven_joint, body_name)
            if rotation_error > max_rotation_error:
                max_rotation_error = rotation_error
                worst_rotation = (driven_joint, body_name)

    print(
        f"[{side}] FK max position error: {max_position_error:.3e} m "
        f"at {worst_position}"
    )
    print(
        f"[{side}] FK max rotation error: {max_rotation_error:.3e} rad "
        f"at {worst_rotation}"
    )
    if max_position_error >= POSITION_TOLERANCE_M:
        raise AssertionError(
            f"{side}: URDF/MJCF position mismatch {max_position_error:.3e} m"
        )
    if max_rotation_error >= ROTATION_TOLERANCE_RAD:
        raise AssertionError(
            f"{side}: URDF/MJCF rotation mismatch {max_rotation_error:.3e} rad"
        )
    return max_position_error, max_rotation_error


def main() -> None:
    results = {side: verify_side(side) for side in ("right", "left")}
    print("\nLinker L20 mapping and FK verification PASSED")
    for side, (position_error, rotation_error) in results.items():
        print(
            f"  {side}: position={position_error:.3e} m, "
            f"rotation={rotation_error:.3e} rad"
        )


if __name__ == "__main__":
    main()
