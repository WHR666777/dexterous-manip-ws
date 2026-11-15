"""已知可返回给 agent 的操作失败；未标记的控制异常仍按故障处理。"""


class CommandRejected(ValueError):
    """输入或动作准入被拒绝；发送前的目标拒绝不回滚此前已执行的动作。"""


class OperationError(RuntimeError):
    """观测或手部反馈不可用，不代表机械臂控制通道已经失效。"""


class GripperTimeout(TimeoutError):
    """手部位置反馈未达到目标；与底层通信调用自身的超时区分。"""
