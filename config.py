"""研究控制示例的共享配置。

``NERO_MAX_JOINT_DELTA`` 的单位为 rad；``None`` 表示不额外限制
Wrapper 已验证的单步关节变化。此文件不得保存 sudo 密码或任何其他凭据，
应在运行前由操作者完成所需的 CAN 权限与接口配置。
"""

from typing import Optional


NERO_CAN_INTERFACE: str = "socketcan"
NERO_CAN_CHANNEL: str = "can0"
NERO_FIRMWARE: str = "1.11"
NERO_MAX_JOINT_DELTA: Optional[float] = None
NERO_SPEED_PERCENT: int = 10

L20_HAND_TYPE: str = "right"
L20_HAND_MODEL: str = "L20"
L20_CAN_CHANNEL: str = "can1"

CONTROL_HZ: int = 20
