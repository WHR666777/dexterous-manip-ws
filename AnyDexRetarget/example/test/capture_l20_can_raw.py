#!/usr/bin/env python3
"""Capture current Linker L20 CAN raw slots for manual pose calibration.

Run from example/:

    python test/capture_l20_can_raw.py --hand right --can-channel can0 --preset INDEX

Pose the real hand with the official GUI or by safe backdriving, then press
Enter. The script prints the 20-slot raw array and, with --write, updates the
selected pinch preset in output/real/drivers_linker_l20_can.py.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT, WORKSPACE_ROOT):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from robot_control import LinkerHandL20

DRIVER_PATH = EXAMPLE_ROOT / "output" / "real" / "drivers_linker_l20_can.py"
PRESET_NAMES = ("INDEX", "MIDDLE", "RING", "PINKY")
SLOT_NAMES = [
    "thumb_base", "index_base", "middle_base", "ring_base", "pinky_base",
    "thumb_abduction", "index_abduction", "middle_abduction",
    "ring_abduction", "pinky_abduction", "thumb_yaw",
    "reserved_11", "reserved_12", "reserved_13", "reserved_14",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
]


def _format_array(values: np.ndarray) -> str:
    items = [int(v) for v in values.tolist()]
    rows = [items[i:i + 10] for i in range(0, len(items), 10)]
    return "[\n" + "\n".join(
        "        " + ", ".join(f"{value:d}" for value in row) + ","
        for row in rows
    ) + "\n    ]"


def _write_preset(preset: str, values: np.ndarray) -> None:
    text = DRIVER_PATH.read_text()
    pattern = re.compile(
        rf'(    "{re.escape(preset)}": np\.array\()\[.*?\](, dtype=np\.float64\),)',
        re.DOTALL,
    )
    replacement = "\\1" + _format_array(values) + "\\2"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find {preset} preset in {DRIVER_PATH}")
    DRIVER_PATH.write_text(new_text)


def _read_samples(hand: LinkerHandL20, samples: int, interval: float) -> np.ndarray:
    readings = []
    for index in range(samples):
        raw = hand.get_joint_positions_raw(fresh=True)
        readings.append(np.asarray(raw, dtype=np.int64))
        if index + 1 < samples:
            time.sleep(interval)
    stacked = np.stack(readings, axis=0)
    return np.rint(np.median(stacked, axis=0)).astype(np.int64)


def _validate_raw_byte(value: int | None, name: str) -> None:
    if value is not None and not 0 <= value <= 255:
        raise ValueError(f"--{name} must be in [0, 255]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture current Linker L20 CAN raw slots")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="INDEX")
    parser.add_argument("--samples", type=int, default=5, help="number of fresh reads to median")
    parser.add_argument("--interval", type=float, default=0.05, help="seconds between samples")
    parser.add_argument("--write", action="store_true", help="write captured raw to the selected preset")
    parser.add_argument(
        "--relax-torque",
        type=int,
        default=None,
        help="set five-finger torque before capture, e.g. 0 to allow manual posing",
    )
    parser.add_argument(
        "--restore-torque",
        type=int,
        default=None,
        help="set five-finger torque after capture before disconnect",
    )
    args = parser.parse_args(argv)

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")
    _validate_raw_byte(args.relax_torque, "relax-torque")
    _validate_raw_byte(args.restore_torque, "restore-torque")

    print(f"Connecting Linker L20: hand={args.hand}, can={args.can_channel}")
    hand = LinkerHandL20(hand_type=args.hand, can_channel=args.can_channel)
    hand.connect()
    try:
        if args.relax_torque is not None:
            torque = [int(args.relax_torque)] * 5
            print(f"\nSetting torque to {torque} before capture...")
            hand.set_torque(torque)
            time.sleep(0.2)

        print("\nPose the real hand to the desired pinch now.")
        print("Recommended: use low torque; do not force the hand if it still resists.")
        input("Press Enter to read current raw slots...")
        raw = _read_samples(hand, args.samples, args.interval)
    finally:
        if args.restore_torque is not None:
            try:
                torque = [int(args.restore_torque)] * 5
                print(f"\nRestoring torque to {torque}...")
                hand.set_torque(torque)
                time.sleep(0.2)
            except Exception as exc:
                print(f"Warning: failed to restore torque: {exc!r}")
        hand.disconnect()

    print("\nCaptured L20 raw slots:")
    for index, (name, value) in enumerate(zip(SLOT_NAMES, raw.tolist())):
        print(f"  slot {index:02d} {name:18s}: {int(value):3d}")
    print("\nRAW:", raw.tolist())
    print(f"\nPreset snippet for {args.preset}:")
    print(f'    "{args.preset}": np.array({_format_array(raw)}, dtype=np.float64),')

    if args.write:
        _write_preset(args.preset, raw)
        print(f"\nWrote {args.preset} preset to {DRIVER_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
