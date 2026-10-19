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
        # 注入异常 fault injection：只验证通信，不代表实现了抓布检测。
        injected = True
        motion.pause_and_hold("模拟监督异常 mock supervision fault injection")
        reply = request_agent("模拟监督异常 mock supervision fault injection", state)
        if reply["action"] == "abort":
            report_result("aborted", reason="agent 放弃恢复 abandoned recovery")
            break
        recovery_plan = plan_from_current(arm=params["arm"],
                                         target_joints=reply["target_joints"])
        motion = execute_trajectory(recovery_plan, wait=False)
    wait_next_observation()
else:
    motion.wait()
    report_result(stage="模拟异常恢复 mock-recovery", recovered=injected)
'''


NORMAL_PROGRAM = '''from cloth_agent.tools import parameters, execute_trajectory, report_result

params = parameters()
execute_trajectory(params["plan_id"])
report_result(stage="模拟第二阶段 mock-second-stage")
'''


SMOKE_TASK = """仅使用 mock 设备运行软件联调 smoke test，不操作真实布料。
请用中文说明计划和结果，保留 API/变量名；生成程序的注释与日志也采用中英夹杂形式。
先观察当前 joints。第一阶段 Stage 1：规划 left 臂 joint1 增加 0.06 rad 的运动，
提交普通 Python 程序异步执行预规划轨迹、轮询 observation，并在 elapsed >= 0.15 s
后恰好注入一次监督异常。暂停保持后调用 request_agent；程序中预先写好的恢复分支
接收 reply.action 和 reply.target_joints，根据实际 feedback 重新规划，在同一进程
继续执行，等待完成并 report_result。通过 cloth_reply_recovery 回复原 target_joints，
不要替换程序源码。
第二阶段 Stage 2：第一阶段完成后，规划并提交 right 臂 joint1 增加 0.03 rad 的程序。
等待完成，再调用 cloth_finish_task；报告两个软件 stage 完成、一次 recovery，明确写出
physical_fold_verified=false。不得直接打开设备或通过 shell 执行控制程序。
"""


def run_demo(runtime, progress=print):
    if runtime.config["backend"] != "mock":
        raise ValueError("异常注入 demo 仅支持 backend: mock。")
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
                    raise RuntimeError(f"演示程序失败 Demo script failed: {event}")
                break
    return runtime.finish_task({"kind": "mock_execution_demo", "stages": 2,
                                "recoveries": recoveries, "physical_fold_verified": False,
                                "description": "两个模拟阶段和一次恢复 software demo；未验证实机折布。"})
