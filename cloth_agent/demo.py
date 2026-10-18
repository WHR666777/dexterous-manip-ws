"""A deterministic script/recovery exercise, explicitly without physical cloth."""

from __future__ import annotations

import json


RECOVERY_PROGRAM = '''from cloth_agent.tools import (
    parameters, execute_trajectory, observe, request_agent,
    plan_from_current, wait_next_observation, report_result,
)

params = parameters()
nominal_plan = params["plan_id"]
motion = execute_trajectory(nominal_plan, wait=False)
injected = False
while not motion.done():
    state = observe()
    if not injected and motion.status()["elapsed"] >= 0.15:
        # Fault injection exercises communication; no grasp detector is claimed.
        injected = True
        motion.pause_and_hold("mock supervision fault injection")
        reply = request_agent("mock supervision fault injection", state)
        if reply["action"] == "abort":
            report_result("aborted", reason="agent abandoned recovery")
            break
        recovery_plan = plan_from_current(arm=params["arm"],
                                         target_joints=reply["target_joints"])
        motion = execute_trajectory(recovery_plan, wait=False)
    wait_next_observation()
else:
    motion.wait()
    report_result(stage="mock-recovery", recovered=injected)
'''


NORMAL_PROGRAM = '''from cloth_agent.tools import parameters, execute_trajectory, report_result

params = parameters()
execute_trajectory(params["plan_id"])
report_result(stage="mock-second-stage")
'''


SMOKE_TASK = """Run a software integration smoke test using ONLY mock devices.
Do not attempt to fold or grasp real cloth. Observe the current joints. Stage 1:
plan a left-arm joint motion with joint1 increased by 0.06 rad. Submit an ordinary
script that executes the preplanned motion asynchronously, polls observations,
injects exactly one supervision fault after elapsed >= 0.15 seconds, pauses and
holds, and calls request_agent. Its existing recovery branch must expect
reply.action and reply.target_joints, replan from feedback, continue in the same
process, wait for completion and report_result. Reply through
cloth_reply_recovery with the original target_joints. Do not replace the script.
Stage 2: after stage 1 completes, plan and submit a right-arm motion with joint1
increased by 0.03 rad. Wait for its completion. Finish using cloth_finish_task,
reporting two completed software stages and one recovery; explicitly report
physical_fold_verified=false. Use no direct devices or shell program execution.
"""


def run_demo(runtime, progress=print):
    if runtime.config["backend"] != "mock":
        raise ValueError("The fault-injection demo requires backend: mock.")
    recoveries = 0
    for arm, delta, source in [("left", .06, RECOVERY_PROGRAM), ("right", .03, NORMAL_PROGRAM)]:
        target = runtime.backend.get_joints(arm)
        target[0] += delta
        plan = runtime.plan_motion(arm, target_joints=target.tolist())
        if not plan["ok"]:
            raise RuntimeError(json.dumps(plan))
        task = runtime.submit_subtask(source, {"arm": arm, "plan_id": plan["plan_id"]})
        while True:
            event = runtime.wait_event(10)
            progress(json.dumps(event, ensure_ascii=False))
            if event["type"] == "recovery_requested":
                recoveries += 1
                runtime.reply_recovery(event["request_id"],
                                       {"action": "continue", "target_joints": target.tolist()})
            elif event["type"] in ("subtask_completed", "subtask_failed", "subtask_aborted"):
                if event["subtask_id"] != task["subtask_id"]:
                    continue
                if event["type"] != "subtask_completed":
                    raise RuntimeError(f"Demo script failed: {event}")
                break
    return runtime.finish_task({"kind": "mock_execution_demo", "stages": 2,
                                "recoveries": recoveries, "physical_fold_verified": False})
