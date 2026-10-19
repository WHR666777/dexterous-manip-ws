from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pytest

from cloth_agent.backends import MockBackend
from cloth_agent.config import MOCK_CONFIG, load_config
from cloth_agent.demo import RECOVERY_PROGRAM, run_demo
from cloth_agent.executor import MotionFailed, MotionInterrupted
from cloth_agent.runtime import Runtime


@pytest.fixture
def runtime(tmp_path):
    config = load_config(MOCK_CONFIG)
    config["control_hz"] = 100
    config["motion"]["completion_timeout_s"] = .15
    with Runtime(config, tmp_path / "run") as value:
        yield value


def plan_joint(runtime, arm="left", delta=.04):
    target = runtime.backend.get_joints(arm)
    target[0] += delta
    result = runtime.plan_motion(arm, target_joints=target.tolist())
    assert result["ok"], result
    return result, target


def event_of_type(runtime, kinds, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = runtime.wait_event(.1)
        if event["type"] in kinds:
            return event
    pytest.fail(f"No event in {kinds}; {runtime.summary()}")


def test_single_arm_execution_and_command_owner(runtime):
    plan, target = plan_joint(runtime)
    other_plan, _ = plan_joint(runtime, "right")
    other_start = runtime.backend.get_joints("right")
    motion = runtime.execute(plan["plan_id"])
    with pytest.raises(RuntimeError, match="one arm"):
        runtime.execute(other_plan["plan_id"])
    result = runtime.executor.wait(motion["motion_id"], 5)
    assert result["status"] == "completed"
    np.testing.assert_allclose(runtime.backend.get_joints("left"), target)
    np.testing.assert_allclose(runtime.backend.get_joints("right"), other_start)
    assert {item[3] for item in runtime.backend.commands} == {"cloth-executor"}


def test_pause_stops_progress_and_remains_a_distinct_result(runtime):
    plan, _ = plan_joint(runtime, delta=.1)
    motion = runtime.execute(plan["plan_id"])
    time.sleep(.05)
    result = runtime.executor.pause_and_hold(motion["motion_id"])
    held = runtime.backend.get_joints("left")
    elapsed = result["elapsed"]
    time.sleep(.07)
    np.testing.assert_array_equal(runtime.backend.get_joints("left"), held)
    assert runtime.executor.status(motion["motion_id"])["elapsed"] == elapsed
    with pytest.raises(MotionInterrupted):
        runtime.executor.wait(motion["motion_id"], 1)


def test_motion_that_does_not_arrive_fails_instead_of_completing(runtime):
    runtime.backend.frozen.add("left")
    plan, _ = plan_joint(runtime, delta=.05)
    motion = runtime.execute(plan["plan_id"])
    with pytest.raises(MotionFailed, match="timed out"):
        runtime.executor.wait(motion["motion_id"], 5)


def test_same_script_recovers_with_new_parameters_and_no_source_replacement(runtime):
    plan, target = plan_joint(runtime, delta=.06)
    submitted = runtime.submit_subtask(RECOVERY_PROGRAM, {"plan_id": plan["plan_id"], "arm": "left"})
    event = event_of_type(runtime, {"recovery_requested", "subtask_failed"})
    assert event["type"] == "recovery_requested", event
    program = Path(submitted["directory"]) / "program.py"
    original_source = program.read_bytes()
    pid = runtime._subtask.process.pid
    held = runtime.backend.get_joints("left")
    time.sleep(.06)
    np.testing.assert_array_equal(runtime.backend.get_joints("left"), held)
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        runtime.submit_subtask("raise AssertionError('replacement')")
    assert event["observation"]["images"]["available"]
    runtime.reply_recovery(event["request_id"], {"action": "continue", "target_joints": target.tolist()})
    completed = event_of_type(runtime, {"subtask_completed", "subtask_failed"})
    assert completed["type"] == "subtask_completed", completed
    assert runtime._subtask.process.pid == pid
    assert program.read_bytes() == original_source
    np.testing.assert_allclose(runtime.backend.get_joints("left"), target)
    assert completed["result"]["recovered"]


def test_program_crash_pauses_active_motion_and_cannot_finish_successfully(runtime):
    plan, _ = plan_joint(runtime, delta=.1)
    runtime.submit_subtask('''from cloth_agent.tools import execute_trajectory, parameters
execute_trajectory(parameters()["plan_id"], wait=False)
raise RuntimeError("injected program crash")
''', {"plan_id": plan["plan_id"]})
    event = event_of_type(runtime, {"subtask_failed"})
    assert "code" in event["reason"]
    assert runtime.executor.active() is None
    held = runtime.backend.get_joints("left")
    time.sleep(.05)
    np.testing.assert_array_equal(runtime.backend.get_joints("left"), held)
    with pytest.raises(RuntimeError, match="failed/aborted"):
        runtime.finish_task()
    with pytest.raises(RuntimeError, match="ended"):
        runtime.submit_subtask("print('edited retry')")


def test_early_program_exit_is_not_a_completed_motion(runtime):
    plan, _ = plan_joint(runtime, delta=.1)
    runtime.submit_subtask('''from cloth_agent.tools import execute_trajectory, parameters
execute_trajectory(parameters()["plan_id"], wait=False)
''', {"plan_id": plan["plan_id"]})
    event = event_of_type(runtime, {"subtask_failed"})
    assert "before its motion finished" in event["reason"]


def test_abandonment_ends_waiting_program_and_prevents_a_new_attempt(runtime):
    runtime.submit_subtask('''from cloth_agent.tools import request_agent, report_result
reply = request_agent("need unavailable input")
report_result("aborted")
''')
    request = event_of_type(runtime, {"recovery_requested"})
    runtime.reply_recovery(request["request_id"], {"action": "abort"})
    event_of_type(runtime, {"subtask_aborted"})
    assert runtime.task_status == "aborted"
    with pytest.raises(RuntimeError, match="ended"):
        runtime.submit_subtask("print('new attempt')")


def test_program_reported_abandonment_also_ends_the_task(runtime):
    runtime.submit_subtask('''from cloth_agent.tools import report_result
report_result("aborted", reason="existing recovery branch gives up")
''')
    event_of_type(runtime, {"subtask_aborted"})
    assert runtime.task_status == "aborted"
    with pytest.raises(RuntimeError, match="ended"):
        runtime.submit_subtask("print('another attempt')")


def test_deterministic_demo_runs_two_programs_and_one_recovery(runtime):
    result = run_demo(runtime, progress=lambda *_: None)
    assert result["status"] == "completed"
    assert result["result"]["stages"] == 2
    assert result["result"]["recoveries"] == 1
    assert result["result"]["physical_fold_verified"] is False
    assert len(result["subtasks"]) == 2


def test_blank_hand_pose_is_not_silently_replaced_by_a_preset(runtime):
    with pytest.raises(ValueError, match="not been determined"):
        runtime.set_hand_pose("left", "closed")


def test_hardware_gate_happens_before_device_creation(tmp_path):
    config = deepcopy(load_config(MOCK_CONFIG))
    config["backend"] = "hardware"
    with pytest.raises(ValueError, match="startup gate"):
        Runtime(config, tmp_path / "hardware")


def test_device_failure_does_not_leave_an_abandoned_program_running(runtime, monkeypatch):
    runtime.submit_subtask("import time\ntime.sleep(30)\n")

    def disconnected(*_):
        raise RuntimeError("injected device disconnection")

    monkeypatch.setattr(runtime.backend, "send_joints", disconnected)
    event_of_type(runtime, {"executor_failed"})
    result = runtime.abort_task("device disconnected")
    assert runtime._subtask.process.poll() is not None
    assert result["status"] == "aborted"
    assert "hold_error" in result["result"]
