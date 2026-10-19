"""Import these functions in a normal generated script; no callback template.

The launcher provides a session for the local device host. A script cannot
replace itself or launch another subtask through this interface.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time
from urllib.request import Request, urlopen

from .executor import MotionFailed, MotionInterrupted


def _call(method, **params):
    if not os.environ.get("CLOTH_AGENT_URL"):
        raise RuntimeError("Run this script through cloth_agent.cli script or the agent host.")
    body = {"subtask_id": os.environ["CLOTH_AGENT_SUBTASK"], "method": method, "params": params}
    request = Request(os.environ["CLOTH_AGENT_URL"],
                      data=json.dumps(body, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + os.environ["CLOTH_AGENT_TOKEN"]})
    # Long recovery waits intentionally keep this same Python process alive.
    with urlopen(request, timeout=None) as response:
        result = json.load(response)
    if not result["ok"]:
        errors = {"MotionInterrupted": MotionInterrupted, "MotionFailed": MotionFailed,
                  "ValueError": ValueError, "TimeoutError": TimeoutError,
                  "PermissionError": PermissionError}
        raise errors.get(result["error"], RuntimeError)(result["message"])
    return result["result"]


def parameters():
    return json.loads(Path(os.environ["CLOTH_AGENT_PARAMS"]).read_text())


def observe(include_images=True):
    return _call("observe", include_images=include_images)


def plan_from_current(waypoints=None, *, arm="left", mode="cartesian", target_joints=None):
    result = _call("plan_motion", arm=arm, waypoints=waypoints, mode=mode,
                   target_joints=target_joints)
    if not result["ok"]:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result["plan_id"]


class MotionHandle:
    def __init__(self, motion_id):
        self.id = motion_id

    def status(self):
        return _call("motion_status", motion_id=self.id)

    def done(self):
        return self.status()["status"] != "running"

    def wait(self, timeout=None):
        return _call("wait_motion", motion_id=self.id, timeout=timeout)

    def pause_and_hold(self, reason="程序检测到异常 program detected an anomaly"):
        return _call("pause_and_hold", motion_id=self.id, reason=reason)


def execute_trajectory(plan, *, wait=True):
    plan_id = plan["plan_id"] if isinstance(plan, dict) else plan
    handle = MotionHandle(_call("execute", plan_id=plan_id)["motion_id"])
    if wait:
        handle.wait()
    return handle


def move_J(arm, target_joints, *, wait=True):
    return execute_trajectory(plan_from_current(arm=arm, target_joints=target_joints), wait=wait)


def move_L(arm, target_tcp, *, wait=True):
    return execute_trajectory(plan_from_current([target_tcp], arm=arm, mode="cartesian"), wait=wait)


def set_hand_pose(pose, *, arm="left"):
    return _call("set_hand_pose", arm=arm, pose=pose)


def pause_and_hold(reason="程序请求暂停 program requested pause"):
    return _call("pause_and_hold", reason=reason)


def request_agent(reason, observation=None):
    return _call("request_agent", reason=reason, observation=observation)


def report_result(status="completed", **result):
    return _call("report_result", status=status, result=result)


def wait_next_observation(period_s=.05):
    if not 0 < period_s <= 60:
        raise ValueError("Observation wait must be positive and at most 60 seconds.")
    time.sleep(period_s)


def geometry(operation, **values):
    return _call("calculate", operation=operation, **values)
