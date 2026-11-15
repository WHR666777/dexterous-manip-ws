"""双臂执行边界测试：以无硬件反馈对象驱动实际执行器，不打开 CAN/相机。"""

from threading import Event, Lock
from types import SimpleNamespace
import time

import numpy as np
import pytest
import yaml

from cloth_agent.bimanual import RobotController
from cloth_agent.config import DEFAULT_CONFIG, load_config, validate_hardware_config
from cloth_agent.geometry import transform
from cloth_agent.planning import Planner
from cloth_agent.robo import MotionResult, Robo
from robot_control.errors import CommandRejected
from robot_control.nero_ik import NeroIK
from scripts.calibrate_tcp_camera import merge_calibrations


def test_combined_calibration_loads_two_arms_and_independent_shared_presets():
    config = load_config()
    validate_hardware_config(config)
    assert set(config["arms"]) == {"left", "right"}
    right, left = config["arms"]["right"], config["arms"]["left"]
    assert left["hand_poses"] == right["hand_poses"]
    left["hand_poses"]["open"][0] = 0
    assert right["hand_poses"]["open"][0] == 255
    assert left["calibration_hand_pose_raw"] == right["hand_poses"]["replay_start"]
    assert left["reference_roll_pitch_deg"] == [-87.78180501724812, -20.890762262087105]
    assert left["hand_can_channel"] != right["hand_can_channel"]


def test_merge_preserves_fitted_left_point_and_composes_camera_frames(tmp_path):
    right_camera = transform([.1, .3, .5, .1, .2, -.3])
    left_camera = transform([.2, .8, .6, .12, .18, -.28])
    tools = {"right": transform([.2, .1, .02, -.4, .2, .3]),
             "left": transform([.23, .09, -.07, .3, -.5, .2])}
    for side, camera in (("right", right_camera), ("left", left_camera)):
        payload = {"hand_side": side, "base_from_camera": camera.tolist(),
                   "flange_from_tcp": tools[side].tolist(),
                   "reference_roll_pitch_deg": [-98., 16.] if side == "right" else [-87., -20.],
                   "hand_pose_raw": [12 if side == "right" else 34] * 20,
                   "fit": {"rms_m": .004}, "camera_frame": "color optical"}
        (tmp_path / f"{side}.yaml").write_text(yaml.safe_dump(payload))
    result = merge_calibrations(tmp_path / "right.yaml", tmp_path / "left.yaml", DEFAULT_CONFIG,
                                tmp_path / "combined.yaml")
    left = result["arms"]["left"]
    np.testing.assert_allclose(transform(left["task_from_base"]) @ left_camera, right_camera, atol=1e-12)
    np.testing.assert_allclose(left["flange_from_tcp"], tools["left"])
    np.testing.assert_allclose(result["arms"]["right"]["flange_from_tcp"], tools["right"])
    assert left["reference_roll_pitch_deg"] == [-87., -20.]
    assert left["hand_pose_raw"] == [34] * 20
    assert result["hand_poses"]["closed"] == [12] * 20


class FeedbackArm:
    def __init__(self, name, q, calls):
        self.name, self.q, self.calls = name, np.array(q), calls
        self.mode, self.moving = 1, 0
        self.reach = True

    def get_joint_positions(self):
        return self.q.copy()

    def get_raw_arm_status(self):
        return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
            ctrl_mode=1, arm_status=0, mode_feedback=self.mode, motion_status=self.moving))

    def move_joints(self, q, **_kwargs):
        self.calls.append((self.name, "J", np.asarray(q).copy()))
        if self.reach:
            self.q = np.asarray(q).copy()
        self.mode, self.moving = 1, 0

    def move_linear(self, pose, **_kwargs):
        self.calls.append((self.name, "L", np.asarray(pose).copy()))
        self.mode, self.moving = 2, 1

    def hold_target_from_feedback(self, q):
        return q.copy()

    def disconnect(self):
        pass


@pytest.fixture
def robot(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=1000.)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "time", lambda: clock.now)
    config = load_config()
    robot = RobotController(config, tmp_path, Event(), Event(), False)
    robot.test_calls, robot.test_clock = [], clock
    for name, arm in robot.arms.items():
        q = config["arms"][name]["start_joint_positions"]
        arm.arm = FeedbackArm(name, q, robot.test_calls)
        arm.validator = Planner(arm.config)
        arm._feedback_ik = NeroIK()
        arm.connected = True
        arm.hand_feedback = SimpleNamespace(get_latest=lambda: SimpleNamespace(
            positions_raw=np.zeros(20, dtype=int), query_finished_at=clock.now), stop=lambda: None)
        arm._measure()
    robot.dispatch({"operation": "begin_cell", "cell_id": 1}, owner=True)
    yield robot
    robot.close()


def ptp_payload(robot, side):
    arm = robot.arms[side]
    target = arm.arm.q.copy()
    target[6] += .04
    return arm.validator.plan_ptp(arm.arm.q, target_joints=target, arm=side).as_dict()


def tick(robot, count=1):
    for _ in range(count):
        robot.test_clock.now += .05
        robot.tick(.05)


def test_batch_publishes_both_before_same_cycle_sends_and_independent_completion(robot):
    plans = {name: ptp_payload(robot, name) for name in ("left", "right")}
    robot.arms["right"].arm.reach = False
    samples = []
    robot.on_control_sample = samples.append
    started = robot.dispatch({"operation": "start_motions", "cell_id": 1, "plans": plans}, owner=False)
    assert robot.test_calls == []
    assert started["left"]["motion_id"] != started["right"]["motion_id"]
    assert started["left"]["diagnostics"]["software_start_at"] == started["right"]["diagnostics"]["software_start_at"]
    tick(robot)
    assert {(side, mode) for side, mode, _ in robot.test_calls} == {("left", "J"), ("right", "J")}
    assert {s["arm"] for s in samples if s["command_sent"]} == {"left", "right"}
    tick(robot, 7)
    assert robot.arms["left"].motion["status"] == "completed"
    assert robot.arms["right"].motion["status"] == "running"
    assert not robot.snapshot()["quiescent"]


@pytest.mark.parametrize("failure", ["workspace", "seed", "busy", "firmware_busy", "arm_mismatch"])
def test_rejected_batch_starts_neither_arm(robot, failure):
    plans = {name: ptp_payload(robot, name) for name in ("left", "right")}
    if failure == "workspace":
        robot.arms["right"].validator.arms["right"]["workspace"] = [[10, 10, 10], [11, 11, 11]]
    elif failure == "seed":
        robot.arms["right"].arm.q[0] += .03
    elif failure == "busy":
        robot.arms["right"].motion = {"status": "running"}
    elif failure == "firmware_busy":
        robot.arms["right"].arm.moving = 1
    else:
        plans["right"]["arm"] = "left"
    with pytest.raises(CommandRejected):
        robot.dispatch({"operation": "start_motions", "cell_id": 1, "plans": plans}, owner=False)
    assert robot.arms["left"].motion is None
    assert robot.test_calls == []


def test_single_arm_requests_do_not_occupy_other_arm_and_handles_route_by_side(robot):
    for side in ("left", "right"):
        robot.dispatch({"operation": "start_motion", "cell_id": 1, "arm": side,
                        "plan": ptp_payload(robot, side)}, owner=False)
    left_id = robot.arms["left"].motion["motion_id"]
    assert robot.dispatch({"operation": "motion_status", "cell_id": 1, "arm": "left",
                           "motion_id": left_id}, owner=False)["motion_id"] == left_id
    with pytest.raises(CommandRejected, match="Unknown motion handle"):
        robot.dispatch({"operation": "motion_status", "cell_id": 1, "arm": "right",
                        "motion_id": left_id}, owner=False)
    assert robot.snapshot()["arms"]["right"]["motion"]["status"] == "running"


def test_native_linear_batch_uses_each_side_tool_and_sends_both_in_one_tick(robot):
    plans = {name: arm.validator.plan_linear(arm.arm.q, arm._tcp, arm=name).as_dict()
             for name, arm in robot.arms.items()}
    robot.dispatch({"operation": "start_motions", "cell_id": 1, "plans": plans}, owner=False)
    tick(robot)
    assert {(side, mode) for side, mode, _ in robot.test_calls} == {("left", "L"), ("right", "L")}
    for name, _, flange in robot.test_calls:
        expected = NeroIK().forward(robot.arms[name].arm.q)
        np.testing.assert_allclose(transform(flange), expected, atol=1e-9)


def test_cell_end_holds_both_and_keeps_ticking_both_until_quiescent(robot):
    plans = {name: ptp_payload(robot, name) for name in robot.arms}
    robot.dispatch({"operation": "start_motions", "cell_id": 1, "plans": plans}, owner=False)
    tick(robot)
    state = robot.dispatch({"operation": "cell_end"}, owner=True)
    assert state["cell_interrupted"]
    assert {s["motion"]["status"] for s in state["arms"].values()} == {"paused"}
    tick(robot, 8)
    assert robot.snapshot()["quiescent"]


def test_manual_hold_captures_both_once_after_shared_event_is_latched(robot):
    before = {side: arm.arm.q.copy() for side, arm in robot.arms.items()}
    state = robot.manual_hold("operator")
    assert robot.manual_stop.is_set()
    assert state["manual_hold_report"]["left"] == "hold_target_sent"
    assert state["manual_hold_report"]["right"] == "hold_target_sent"
    for arm in robot.arms.values():
        arm.arm.q[0] += .01
    robot.manual_hold("repeat")
    for side, arm in robot.arms.items():
        np.testing.assert_array_equal(arm._manual_target, before[side])
    assert len(robot.test_calls) == 2
    with pytest.raises(CommandRejected):
        robot.dispatch({"operation": "state", "cell_id": 1}, owner=False)


class LocalConnection:
    def __init__(self, robot):
        self.robot, self.operations = robot, []

    def send(self, request):
        self.operations.append(request["operation"])
        self.reply = {"ok": True, "result": self.robot.dispatch(request, owner=False)}

    def recv(self):
        return self.reply


def test_robo_batch_returns_handles_for_each_arm(robot):
    connection = LocalConnection(robot)
    robo = Robo(connection, robot.config)
    robo._begin_cell(1)
    targets = {name: arm._tcp for name, arm in robot.arms.items()}
    handles = robo.move_both(left=targets["left"], right=targets["right"], wait=False)
    assert handles["left"].status()["motion_id"] != handles["right"].status()["motion_id"]
    assert connection.operations.count("start_motions") == 1
    robo._begin_cell(2)
    with pytest.raises(RuntimeError, match="earlier cell"):
        handles["left"].status()


def test_robo_failed_second_plan_does_not_submit_first(robot):
    connection = LocalConnection(robot)
    robo = Robo(connection, robot.config)
    robo._begin_cell(1)
    target = transform(robot.arms["right"]._tcp)
    target[0, 3] = 100.
    results = robo.move_both(left=robot.arms["left"]._tcp, right=target, wait=False)
    assert all(result.status == "planning_failed" for result in results.values())
    assert "start_motions" not in connection.operations
    assert robot.test_calls == []


def test_execution_failure_pauses_peer_and_both_hold_loops_continue(robot):
    robot.arms["right"].arm.reach = False
    plans = {name: ptp_payload(robot, name) for name in robot.arms}
    robot.dispatch({"operation": "start_motions", "cell_id": 1, "plans": plans}, owner=False)
    tick(robot)
    robot.arms["left"]._fail_motion(robot.arms["left"].motion, "arrival failed")
    tick(robot)
    assert robot.arms["right"].motion["status"] == "paused"
    assert robot.arms["left"].motion["status"] == "failed"
    tick(robot, 8)
    assert robot.snapshot()["quiescent"]


def test_mirrored_startup_targets_pass_ik_fk_and_path_checks():
    from robot_control.nero_ik import pose_error, MAX_FK_POSITION_ERROR_M, MAX_FK_ROTATION_ERROR_DEG

    config = load_config()
    planner = Planner(config)
    for side, settings in config["arms"].items():
        plan = planner.plan_startup_ptp(settings["start_joint_positions"],
                                        target_tcp=settings["initial_grasp_tcp"], arm=side)
        actual = (transform(settings["task_from_base"]) @ planner.ik[side].forward(plan.positions[-1])
                  @ transform(settings["flange_from_tcp"]))
        position, angle = pose_error(actual, transform(settings["initial_grasp_tcp"]))
        assert position < MAX_FK_POSITION_ERROR_M
        assert np.rad2deg(angle) < MAX_FK_ROTATION_ERROR_DEG


def test_startup_cell_prepares_and_verifies_both_arms(tmp_path):
    from cloth_agent.session import TaskSession
    from cloth_agent.worker import _execute, _namespace

    session = TaskSession.__new__(TaskSession)
    session.robot_config = load_config()
    session.directory, session.status = tmp_path, "running"
    session._state_lock = Lock()
    session.startup_prepared = False
    session.control = SimpleNamespace(ready={"hardware_connected": True}, manual_hold=Event(),
                                      request=lambda *_: {"quiescent": True})
    session.log = lambda *_args, **_kwargs: None
    actions, actual = [], {}

    def startup_move_ptp(*, arm, target_joints=None, target_tcp=None):
        actions.append((arm, "joints" if target_joints is not None else "tcp"))
        if target_tcp is not None:
            actual[arm] = {"tcp": target_tcp}
        return MotionResult("completed", True)

    def state(*, arm):
        joints = np.array(session.robot_config["arms"][arm]["start_joint_positions"])
        return {"joints": joints + (.05 if arm == "left" else 0.)}

    def execute(source, *, _startup):
        assert _startup
        namespace = _namespace(SimpleNamespace(state=state))
        namespace["startup_open_hand"] = lambda *, arm: actions.append((arm, "open"))
        namespace["startup_move_ptp"] = startup_move_ptp
        result = _execute(source, namespace, 0)
        result["control"] = {"arms": actual}
        return result

    session.execute = execute
    session._prepare_robot()
    assert session.startup_prepared
    assert actions[:2] == [("right", "open"), ("left", "open")]
    assert ("left", "joints") in actions
    assert {arm for arm, kind in actions if kind == "tcp"} == {"left", "right"}


def test_spawned_session_exposes_both_arms_without_connecting_hardware(tmp_path):
    from cloth_agent.session import TaskSession

    with TaskSession(tmp_path / "session", config_path=DEFAULT_CONFIG, allow_hardware=False) as session:
        result = session.execute("state = robo.state()\nprint(sorted(state['arms']))\nrobo.state(arm='left')")
        assert result["status"] == "completed"
        assert "['left', 'right']" in result["stdout"]
        assert set(result["control"]["arms"]) == {"left", "right"}
        assert not result["control"]["hardware_connected"]
        rejected = session.execute("robo.move_both(left=np.eye(4), right=np.eye(4))")
        assert rejected["status"] == "error"
        assert "Hardware not connected" in rejected["exception"]
        assert session.status == "running"
