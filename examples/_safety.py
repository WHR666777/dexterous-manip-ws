"""示例共享的最小化清理报告辅助函数。"""

from __future__ import annotations

import sys
from typing import Any


def _disconnect_or_report(device: Any, label: str) -> bool:
    """尝试断开设备，并将清理失败报告到标准错误流。

    Parameters
    ----------
    device : object
        提供公开 ``disconnect()`` 方法的已构造 Wrapper。
    label : str
        用于操作者诊断的设备名称。

    Returns
    -------
    bool
        断开成功时为 ``True``，断开抛出普通异常时为 ``False``。

    Notes
    -----
    此函数只处理断开本身的普通异常；调用方负责在已有主异常时保留该异常，或在
    清理是唯一失败时选择非零退出码。
    """
    try:
        device.disconnect()
    except Exception as error:
        print("{0} disconnect failed: {1}".format(label, error), file=sys.stderr)
        return False
    return True
