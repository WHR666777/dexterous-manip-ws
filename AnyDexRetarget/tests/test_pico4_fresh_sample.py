from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from input.pico4 import Pico4  # noqa: E402


def _offline_pico() -> Pico4:
    pico = Pico4.__new__(Pico4)
    pico._lock = threading.Lock()
    pico._left_hand = np.ones((21, 3), dtype=np.float32)
    pico._right_hand = np.full((21, 3), 2.0, dtype=np.float32)
    pico._last_update = 3.0
    pico._left_update = 1.0
    pico._right_update = 2.0
    return pico


def test_pico_sample_uses_selected_hand_timestamp() -> None:
    pico = _offline_pico()

    data, right_timestamp = pico.get_fingers_sample("right")
    _, left_timestamp = pico.get_fingers_sample("left")
    _, any_timestamp = pico.get_fingers_sample()

    assert right_timestamp == 2.0
    assert left_timestamp == 1.0
    assert any_timestamp == 3.0
    assert np.all(data["right_fingers"] == 2.0)


def test_pico_sample_returns_copies_and_validates_side() -> None:
    pico = _offline_pico()
    data, _ = pico.get_fingers_sample("right")
    data["right_fingers"].fill(99.0)

    assert np.all(pico._right_hand == 2.0)
    with pytest.raises(ValueError, match="hand_side"):
        pico.get_fingers_sample("middle")
