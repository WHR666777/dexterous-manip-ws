"""中英运行说明 Bilingual labels; machine-readable identifiers stay unchanged."""


EVENT_TEXT = {
    "task_started": "任务开始 task started",
    "task_completed": "任务完成 task completed",
    "task_aborted": "任务放弃 task aborted",
    "planning_completed": "轨迹规划完成 planning completed",
    "planning_failed": "轨迹规划失败 planning failed",
    "motion_started": "运动开始 motion started",
    "motion_completed": "运动完成 motion completed",
    "motion_paused": "运动暂停并保持 motion paused",
    "motion_failed": "运动失败 motion failed",
    "hand_command": "发送手型指令 hand command",
    "subtask_started": "子任务开始 subtask started",
    "subtask_completed": "子任务完成 subtask completed",
    "subtask_failed": "子任务失败 subtask failed",
    "subtask_aborted": "子任务放弃 subtask aborted",
    "recovery_requested": "请求 agent 恢复参数 recovery requested",
    "recovery_replied": "收到 agent 恢复回复 recovery replied",
    "agent_tool_called": "Codex 调用工具 tool called",
    "executor_failed": "执行器故障 executor failed",
    "running": "等待程序事件 waiting for program event",
}

STATUS_TEXT = {
    "running": "执行中 running",
    "completed": "已完成 completed",
    "paused": "已暂停 paused",
    "failed": "失败 failed",
    "aborted": "已放弃 aborted",
    "holding": "保持当前姿态 holding",
}


def status_text(status):
    return STATUS_TEXT.get(status, status)


def event_description(event):
    text = EVENT_TEXT.get(event["type"], event["type"])
    if event.get("arm"):
        text += "；机械臂 arm=" + event["arm"]
    if event.get("reason"):
        text += "；原因 reason=" + event["reason"]
    return text
