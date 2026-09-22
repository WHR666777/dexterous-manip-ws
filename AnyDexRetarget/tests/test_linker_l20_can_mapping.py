from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from output.real.drivers_linker_l20 import _L20_ACTIVE_JOINT_ORDER  # noqa: E402
from output.real.drivers_linker_l20_can import l20_qpos_to_can_slots  # noqa: E402


DEG_TO_RAD = np.pi / 180.0


class LinkerL20CANMappingTests(unittest.TestCase):
    def _qpos(self, **degrees: float) -> np.ndarray:
        qpos = np.zeros(len(_L20_ACTIVE_JOINT_ORDER), dtype=np.float64)
        indices = {name: index for index, name in enumerate(_L20_ACTIVE_JOINT_ORDER)}
        for name, value_deg in degrees.items():
            qpos[indices[name]] = float(value_deg) * DEG_TO_RAD
        return qpos

    def test_zero_degrees_maps_to_documented_open_slots(self):
        slots = l20_qpos_to_can_slots(self._qpos(), _L20_ACTIVE_JOINT_ORDER)

        np.testing.assert_array_equal(
            slots,
            np.array([
                255, 255, 255, 255, 255,
                255, 128, 128, 128, 128,
                255, 255, 255, 255, 255,
                255, 255, 255, 255, 255,
            ], dtype=np.int64),
        )

    def test_vendor_max_degrees_map_to_documented_closed_slots(self):
        slots = l20_qpos_to_can_slots(
            self._qpos(
                THUMB_CMC_PITCH=45.0,
                INDEX_MCP_PITCH=70.0,
                MIDDLE_MCP_PITCH=70.0,
                RING_MCP_PITCH=70.0,
                PINKY_MCP_PITCH=70.0,
                THUMB_CMC_ROLL=77.6,
                INDEX_MCP_ROLL=13.0,
                MIDDLE_MCP_ROLL=13.0,
                RING_MCP_ROLL=13.0,
                PINKY_MCP_ROLL=13.0,
                THUMB_CMC_YAW=91.0,
                THUMB_MCP=70.0,
                INDEX_PIP=99.0,
                MIDDLE_PIP=99.0,
                RING_PIP=99.0,
                PINKY_PIP=99.0,
            ),
            _L20_ACTIVE_JOINT_ORDER,
            reserved=[11, 12, 13, 14],
        )

        np.testing.assert_array_equal(
            slots,
            np.array([
                0, 0, 0, 0, 0,
                0, 255, 255, 255, 255,
                0, 11, 12, 13, 14,
                0, 0, 0, 0, 0,
            ], dtype=np.int64),
        )

    def test_midrange_degrees_map_to_raw_128(self):
        slots = l20_qpos_to_can_slots(
            self._qpos(
                THUMB_CMC_PITCH=22.5,
                INDEX_MCP_PITCH=35.0,
                MIDDLE_MCP_PITCH=35.0,
                RING_MCP_PITCH=35.0,
                PINKY_MCP_PITCH=35.0,
                THUMB_CMC_ROLL=38.8,
                INDEX_MCP_ROLL=0.0,
                MIDDLE_MCP_ROLL=0.0,
                RING_MCP_ROLL=0.0,
                PINKY_MCP_ROLL=0.0,
                THUMB_CMC_YAW=45.5,
                THUMB_MCP=35.0,
                INDEX_PIP=49.5,
                MIDDLE_PIP=49.5,
                RING_PIP=49.5,
                PINKY_PIP=49.5,
            ),
            _L20_ACTIVE_JOINT_ORDER,
        )

        np.testing.assert_array_equal(
            slots,
            np.array([128] * 11 + [255] * 4 + [128] * 5, dtype=np.int64),
        )

    def test_negative_roll_limit_maps_to_zero(self):
        slots = l20_qpos_to_can_slots(
            self._qpos(
                INDEX_MCP_ROLL=-13.0,
                MIDDLE_MCP_ROLL=-13.0,
                RING_MCP_ROLL=-13.0,
                PINKY_MCP_ROLL=-13.0,
            ),
            _L20_ACTIVE_JOINT_ORDER,
        )

        np.testing.assert_array_equal(slots[6:10], np.array([0, 0, 0, 0]))

    def test_pinch_context_does_not_change_direct_angle_mapping(self):
        qpos = self._qpos(INDEX_MCP_PITCH=35.0, INDEX_PIP=49.5)

        without_pinch = l20_qpos_to_can_slots(qpos, _L20_ACTIVE_JOINT_ORDER)
        with_pinch = l20_qpos_to_can_slots(
            qpos,
            _L20_ACTIVE_JOINT_ORDER,
            pinch_context=("INDEX", 1.0),
        )

        np.testing.assert_array_equal(with_pinch, without_pinch)


if __name__ == "__main__":
    unittest.main()
