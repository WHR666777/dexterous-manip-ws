"""Manual joint-angle control for MuJoCo simulation.

Launches a MuJoCo passive viewer and a PyQt control panel.  The control panel
uses degree values rounded to one decimal place, while MuJoCo receives radians.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import numpy as np
import yaml

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QDoubleValidator
    from PyQt5.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise ImportError(
        "angle_sim.py requires PyQt5. Install it in this environment with "
        "`pip install PyQt5` or run from an environment that already includes it."
    ) from exc


EXAMPLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_ROOT.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from output.sim.mujoco_output import (  # noqa: E402
    ROBOT_HAND_CONFIGS,
    validate_mujoco_actuator_mapping,
)


DEG_SCALE = 10
FALLBACK_RANGE_RAD = (-np.pi, np.pi)


@dataclass(frozen=True)
class ControlJoint:
    actuator_id: int
    joint_id: int
    name: str
    qpos_addr: int
    min_rad: float
    max_rad: float


@dataclass(frozen=True)
class RawJoint:
    joint_id: int
    name: str
    qpos_addr: int
    min_rad: float | None
    max_rad: float | None


class SharedControlState:
    def __init__(self, target_ctrl: np.ndarray, raw_count: int):
        self.lock = threading.Lock()
        self.target_ctrl = target_ctrl.astype(np.float64, copy=True)
        self.raw_qpos = np.zeros(raw_count, dtype=np.float64)
        self.running = True

    def set_target(self, actuator_id: int, value_rad: float) -> None:
        with self.lock:
            self.target_ctrl[actuator_id] = value_rad

    def get_targets(self) -> np.ndarray:
        with self.lock:
            return self.target_ctrl.copy()

    def update_raw_qpos(self, values: np.ndarray) -> None:
        with self.lock:
            self.raw_qpos[:] = values

    def get_raw_qpos(self) -> np.ndarray:
        with self.lock:
            return self.raw_qpos.copy()

    def stop(self) -> None:
        with self.lock:
            self.running = False

    def is_running(self) -> bool:
        with self.lock:
            return self.running


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def _apply_camera_config(
    camera,
    cam_cfg: dict,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: list[float] | None = None,
) -> None:
    camera.azimuth = cam_cfg.get("azimuth", 135) if azimuth is None else azimuth
    camera.elevation = cam_cfg.get("elevation", -20) if elevation is None else elevation
    camera.distance = cam_cfg.get("distance", 0.5) if distance is None else distance
    camera.lookat[:] = cam_cfg.get("lookat", [0, 0, 0.05]) if lookat is None else lookat


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        or f"joint[{joint_id}]"
    )


def _actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    return (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        or f"actuator[{actuator_id}]"
    )


def _single_dof_joint(model: mujoco.MjModel, joint_id: int) -> bool:
    joint_type = int(model.jnt_type[joint_id])
    return joint_type in {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }


def _range_from_actuator_or_joint(
    model: mujoco.MjModel,
    actuator_id: int,
    joint_id: int,
) -> tuple[float, float]:
    if model.actuator_ctrllimited[actuator_id]:
        lower, upper = model.actuator_ctrlrange[actuator_id]
    elif model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
    else:
        lower, upper = FALLBACK_RANGE_RAD
    return float(lower), float(upper)


def _range_from_joint(model: mujoco.MjModel, joint_id: int) -> tuple[float, float] | None:
    if model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
        return float(lower), float(upper)
    return None


def _extract_control_joints(model: mujoco.MjModel, hand_cfg: dict) -> list[ControlJoint]:
    validate_mujoco_actuator_mapping(model, hand_cfg)
    joint_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    joints: list[ControlJoint] = []
    for actuator_id in range(model.nu):
        trn_type = int(model.actuator_trntype[actuator_id])
        if trn_type not in joint_transmissions:
            raise ValueError(
                f"{_actuator_name(model, actuator_id)!r} is not a joint actuator."
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if not _single_dof_joint(model, joint_id):
            raise ValueError(
                f"{_joint_name(model, joint_id)!r} is not a 1-DoF hinge/slide joint."
            )
        lower, upper = _range_from_actuator_or_joint(model, actuator_id, joint_id)
        joints.append(
            ControlJoint(
                actuator_id=actuator_id,
                joint_id=joint_id,
                name=_joint_name(model, joint_id),
                qpos_addr=int(model.jnt_qposadr[joint_id]),
                min_rad=lower,
                max_rad=upper,
            )
        )
    return joints


def _extract_raw_joints(model: mujoco.MjModel) -> list[RawJoint]:
    joints: list[RawJoint] = []
    for joint_id in range(model.njnt):
        if not _single_dof_joint(model, joint_id):
            continue
        joint_range = _range_from_joint(model, joint_id)
        lower = joint_range[0] if joint_range is not None else None
        upper = joint_range[1] if joint_range is not None else None
        joints.append(
            RawJoint(
                joint_id=joint_id,
                name=_joint_name(model, joint_id),
                qpos_addr=int(model.jnt_qposadr[joint_id]),
                min_rad=lower,
                max_rad=upper,
            )
        )
    return joints


def _rad_to_deg(value: float) -> float:
    return float(np.rad2deg(value))


def _deg_to_rad(value: float) -> float:
    return float(np.deg2rad(value))


def _round_deg(value_rad: float) -> float:
    return round(_rad_to_deg(value_rad), 1)


def _deg_slider_value(value_deg: float) -> int:
    return int(round(value_deg * DEG_SCALE))


def _initial_ctrl(model: mujoco.MjModel, control_joints: list[ControlJoint]) -> np.ndarray:
    target = np.zeros(model.nu, dtype=np.float64)
    for joint in control_joints:
        target[joint.actuator_id] = float(np.clip(0.0, joint.min_rad, joint.max_rad))
    return target


class JointControlRow(QWidget):
    def __init__(
        self,
        joint: ControlJoint,
        target_rad: float,
        on_change,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.joint = joint
        self._on_change = on_change
        self._updating = False

        min_deg = _round_deg(joint.min_rad)
        max_deg = _round_deg(joint.max_rad)
        target_deg = _round_deg(target_rad)

        self.name_label = QLabel(joint.name)
        self.name_label.setMinimumWidth(170)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(_deg_slider_value(min_deg), _deg_slider_value(max_deg))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setValue(_deg_slider_value(target_deg))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_deg, max_deg)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.1)
        self.spin.setValue(target_deg)
        self.spin.setSuffix(" deg")
        self.spin.setMinimumWidth(110)

        self.raw_label = QLabel("0.0 deg")
        self.raw_label.setMinimumWidth(90)
        self.range_label = QLabel(f"{min_deg:.1f} .. {max_deg:.1f}")
        self.range_label.setMinimumWidth(105)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)
        layout.addWidget(self.raw_label)
        layout.addWidget(self.range_label)

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _set_target_deg(self, value_deg: float, source: str) -> None:
        if self._updating:
            return
        self._updating = True
        value_deg = round(float(value_deg), 1)
        if source != "slider":
            self.slider.setValue(_deg_slider_value(value_deg))
        if source != "spin":
            self.spin.setValue(value_deg)
        self._updating = False
        value_rad = float(np.clip(_deg_to_rad(value_deg), self.joint.min_rad, self.joint.max_rad))
        self._on_change(self.joint.actuator_id, value_rad)

    def _slider_changed(self, value: int) -> None:
        self._set_target_deg(value / DEG_SCALE, "slider")

    def _spin_changed(self, value: float) -> None:
        self._set_target_deg(value, "spin")

    def set_target_rad(self, value_rad: float) -> None:
        value_deg = _round_deg(value_rad)
        self._updating = True
        self.slider.setValue(_deg_slider_value(value_deg))
        self.spin.setValue(value_deg)
        self._updating = False

    def set_raw_rad(self, value_rad: float) -> None:
        self.raw_label.setText(f"{_round_deg(value_rad):.1f} deg")


class RawJointReadout(QWidget):
    def __init__(self, joint: RawJoint, parent: QWidget | None = None):
        super().__init__(parent)
        self.joint = joint
        self.value = QLineEdit("0.0")
        self.value.setReadOnly(True)
        self.value.setAlignment(Qt.AlignRight)
        self.value.setMinimumWidth(80)
        validator = QDoubleValidator(self)
        validator.setDecimals(1)
        self.value.setValidator(validator)

        range_text = ""
        if joint.min_rad is not None and joint.max_rad is not None:
            range_text = f"{_round_deg(joint.min_rad):.1f} .. {_round_deg(joint.max_rad):.1f}"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        name_label = QLabel(joint.name)
        name_label.setMinimumWidth(170)
        unit_label = QLabel("deg")
        unit_label.setMinimumWidth(30)
        range_label = QLabel(range_text)
        range_label.setMinimumWidth(105)
        layout.addWidget(name_label)
        layout.addWidget(self.value)
        layout.addWidget(unit_label)
        layout.addWidget(range_label)

    def set_raw_rad(self, value_rad: float) -> None:
        self.value.setText(f"{_round_deg(value_rad):.1f}")


class AngleControlWindow(QMainWindow):
    def __init__(
        self,
        state: SharedControlState,
        control_joints: list[ControlJoint],
        raw_joints: list[RawJoint],
        initial_ctrl: np.ndarray,
        stop_callback,
        title: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.state = state
        self.control_joints = control_joints
        self.raw_joints = raw_joints
        self.stop_callback = stop_callback
        self.control_rows: dict[int, JointControlRow] = {}
        self.raw_rows: list[RawJointReadout] = []

        self.setWindowTitle(title)
        self.resize(980, 760)

        central = QWidget()
        root = QVBoxLayout(central)

        header = QLabel("Target controls use degrees; raw readouts mirror MuJoCo qpos.")
        root.addWidget(header)

        controls_group = QGroupBox("Controllable joints")
        controls_layout = QVBoxLayout(controls_group)
        heading = QHBoxLayout()
        for label, width in (
            ("Joint", 170),
            ("Target", 360),
            ("Value", 110),
            ("Raw", 90),
            ("Range", 105),
        ):
            widget = QLabel(label)
            widget.setMinimumWidth(width)
            heading.addWidget(widget)
        controls_layout.addLayout(heading)

        for joint in control_joints:
            row = JointControlRow(
                joint=joint,
                target_rad=float(initial_ctrl[joint.actuator_id]),
                on_change=self._target_changed,
            )
            controls_layout.addWidget(row)
            self.control_rows[joint.qpos_addr] = row
        root.addWidget(controls_group)

        buttons = QHBoxLayout()
        zero_button = QPushButton("Zero")
        center_button = QPushButton("Center")
        close_button = QPushButton("Close")
        zero_button.clicked.connect(self._zero_targets)
        center_button.clicked.connect(self._center_targets)
        close_button.clicked.connect(self.close)
        buttons.addWidget(zero_button)
        buttons.addWidget(center_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

        raw_group = QGroupBox("All raw joints")
        raw_layout = QGridLayout(raw_group)
        columns = 2
        for index, joint in enumerate(raw_joints):
            row_widget = RawJointReadout(joint)
            self.raw_rows.append(row_widget)
            raw_layout.addWidget(row_widget, index // columns, index % columns)

        scroll = QScrollArea()
        scroll.setWidget(raw_group)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)

        self.status_label = QLabel("Starting simulation...")
        root.addWidget(self.status_label)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_readouts)
        self.timer.start(50)

    def _target_changed(self, actuator_id: int, value_rad: float) -> None:
        self.state.set_target(actuator_id, value_rad)

    def _zero_targets(self) -> None:
        for joint in self.control_joints:
            value_rad = float(np.clip(0.0, joint.min_rad, joint.max_rad))
            self.state.set_target(joint.actuator_id, value_rad)
            self.control_rows[joint.qpos_addr].set_target_rad(value_rad)

    def _center_targets(self) -> None:
        for joint in self.control_joints:
            value_rad = 0.5 * (joint.min_rad + joint.max_rad)
            self.state.set_target(joint.actuator_id, value_rad)
            self.control_rows[joint.qpos_addr].set_target_rad(value_rad)

    def _refresh_readouts(self) -> None:
        raw_values = self.state.get_raw_qpos()
        if len(raw_values) != len(self.raw_joints):
            return
        raw_by_addr = {
            joint.qpos_addr: raw_values[index]
            for index, joint in enumerate(self.raw_joints)
        }
        for qpos_addr, row in self.control_rows.items():
            if qpos_addr in raw_by_addr:
                row.set_raw_rad(float(raw_by_addr[qpos_addr]))
        for index, row in enumerate(self.raw_rows):
            row.set_raw_rad(float(raw_values[index]))

        self.status_label.setText(
            "Simulation running" if self.state.is_running() else "Simulation stopped"
        )
        if not self.state.is_running():
            self.timer.stop()

    def closeEvent(self, event) -> None:
        self.stop_callback()
        super().closeEvent(event)


def _simulation_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: SharedControlState,
    raw_joints: list[RawJoint],
    cam_cfg: dict,
    camera_azimuth: float | None,
    camera_elevation: float | None,
    camera_distance: float | None,
) -> None:
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        _apply_camera_config(
            viewer.cam,
            cam_cfg,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance=camera_distance,
        )

        sim_dt = float(model.opt.timestep)
        n_substeps = max(1, int(round((1.0 / 60.0) / sim_dt)))
        render_interval = sim_dt * n_substeps

        while state.is_running():
            loop_start = time.time()
            if viewer is not None and not viewer.is_running():
                break

            targets = state.get_targets()
            if model.nu > 0:
                data.ctrl[:] = targets[: model.nu]

            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            state.update_raw_qpos(
                np.array([data.qpos[joint.qpos_addr] for joint in raw_joints], dtype=np.float64)
            )

            if viewer is not None:
                viewer.sync()

            sleep_time = render_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
    finally:
        state.stop()
        if viewer is not None:
            viewer.close()


def _default_config_path(robot: str, optimizer: str) -> str:
    robot_name_map = {
        "shadow": "shadow_hand",
        "wuji": "wuji_hand",
        "allegro": "allegro_hand",
        "leap": "leap_hand",
        "inspire": "inspire_hand",
        "ability": "ability_hand",
        "svh": "svh_hand",
        "rohand": "rohand",
        "linkerhand_l21": "linkerhand_l21",
        "linker_l20": "linker_l20",
        "unitree_dex5": "unitree_dex5_hand",
        "sharpa": "sharpa_hand",
        "gaia": "gaia_hand20",
    }
    robot_file = robot_name_map.get(robot, robot)
    return f"config/{optimizer}/mediapipe/mediapipe_{robot_file}.yaml"


def run_angle_sim(
    hand_side: str = "right",
    config_path: str = "config/adaptive/mediapipe/mediapipe_linker_l20.yaml",
    camera_azimuth: float | None = None,
    camera_elevation: float | None = None,
    camera_distance: float | None = None,
) -> int:
    hand_side = hand_side.lower()
    if hand_side not in {"right", "left"}:
        raise ValueError("hand_side must be 'right' or 'left'")

    config_file = _resolve_config_path(config_path)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_type = config.get("robot", {}).get("type", "linker_l20")
    if robot_type not in ROBOT_HAND_CONFIGS:
        raise ValueError(
            f"Unknown robot type: {robot_type}. Supported: {list(ROBOT_HAND_CONFIGS.keys())}"
        )

    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
    model_path = Path(hand_cfg["model_path"](hand_side))
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    control_joints = _extract_control_joints(model, hand_cfg)
    if not control_joints:
        raise ValueError(f"Model has no controllable actuated joints: {model_path}")
    raw_joints = _extract_raw_joints(model)
    initial_ctrl = _initial_ctrl(model, control_joints)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)

    state = SharedControlState(initial_ctrl, len(raw_joints))
    state.update_raw_qpos(
        np.array([data.qpos[joint.qpos_addr] for joint in raw_joints], dtype=np.float64)
    )

    print("Starting manual angle simulation...")
    print(f"  Config: {config_path}")
    print(f"  Robot: {robot_type}")
    print(f"  Hand: {hand_side}")
    print(f"  Model: {model_path}")
    print(f"  Controls: {len(control_joints)} actuated joints")
    print(f"  Raw qpos readouts: {len(raw_joints)} joints")
    print("=" * 50)

    cam_cfg = config.get("render", {}).get("camera", {})
    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(
            model,
            data,
            state,
            raw_joints,
            cam_cfg,
            camera_azimuth,
            camera_elevation,
            camera_distance,
        ),
        daemon=True,
    )
    sim_thread.start()

    app = QApplication.instance() or QApplication(sys.argv)

    def stop() -> None:
        state.stop()

    title = f"Angle Sim - {robot_type} {hand_side}"
    window = AngleControlWindow(
        state=state,
        control_joints=control_joints,
        raw_joints=raw_joints,
        initial_ctrl=initial_ctrl,
        stop_callback=stop,
        title=title,
    )
    window.show()
    exit_code = app.exec_()

    state.stop()
    sim_thread.join(timeout=2.0)
    return int(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual degree-angle control with MuJoCo simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (overrides --robot and --optimizer)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adaptive",
        choices=["adaptive", "vector"],
        help="Optimizer type for default config path (default: adaptive)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="linker_l20",
        choices=[
            "shadow",
            "wuji",
            "allegro",
            "leap",
            "inspire",
            "ability",
            "svh",
            "rohand",
            "linkerhand_l21",
            "linker_l20",
            "unitree_dex5",
            "sharpa",
            "gaia",
        ],
        help="Robot hand type for default config path (default: linker_l20)",
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Hand side (default: right)",
    )
    parser.add_argument("--cam-azimuth", type=float, default=None, help="Override camera azimuth")
    parser.add_argument("--cam-elevation", type=float, default=None, help="Override camera elevation")
    parser.add_argument("--cam-distance", type=float, default=None, help="Override camera distance")

    args = parser.parse_args()
    config_path = args.config or _default_config_path(args.robot, args.optimizer)
    raise SystemExit(
        run_angle_sim(
            hand_side=args.hand,
            config_path=config_path,
            camera_azimuth=args.cam_azimuth,
            camera_elevation=args.cam_elevation,
            camera_distance=args.cam_distance,
        )
    )


if __name__ == "__main__":
    main()
