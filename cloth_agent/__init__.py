"""Cloth agent：Codex exec、MCP 与独立持久 Python worker / 控制进程。

统一使用 python -m cloth_agent；robo 通过独立控制进程访问右侧 Nero/L20。
导入本包不会连接设备、启动进程或发起模型请求。
"""

