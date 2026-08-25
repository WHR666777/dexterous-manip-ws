"""Project-level configuration and runtime dependency contracts."""

from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_v111_and_separate_can_channels():
    """The documented default wiring must select the supported devices."""
    assert config.NERO_FIRMWARE == "1.11"
    assert config.NERO_CAN_INTERFACE == "socketcan"
    assert config.NERO_CAN_CHANNEL != config.L20_CAN_CHANNEL
    assert config.L20_HAND_MODEL == "L20"
    assert config.CONTROL_HZ == 20


def test_runtime_requirements_are_minimal():
    """A fresh runtime install must contain only the wrapper's direct dependencies."""
    lines = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert lines == {
        "numpy",
        "python-can>=3.3.4",
        "PyYAML",
        "typing-extensions>=3.7.4.3",
    }
