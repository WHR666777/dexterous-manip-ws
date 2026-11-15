"""CLI 共用的本轮记录目录；人工保持/监控只读 YAML，不加载机器人标定或数值库。"""

from __future__ import annotations

from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/cloth_agent.yaml"


def configured_output_dir(config_path=DEFAULT_CONFIG) -> Path:
    """读取 session.output_dir；相对路径以配置文件所在目录为基准。"""
    import yaml

    path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    session = config.get("session") if isinstance(config, dict) else None
    value = session.get("output_dir") if isinstance(session, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"请在配置中指定本轮目录 Set session.output_dir in {path}")
    directory = Path(value).expanduser()
    # 与标定路径一致，以配置文件为锚点；不同终端的工作目录不会改变命令目标。
    return (directory if directory.is_absolute() else path.parent / directory).resolve()
