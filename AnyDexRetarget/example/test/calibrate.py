#!/usr/bin/env python3
"""Unified calibration entrypoint.

Use one command and select the calibration behavior by the first argument.  All
remaining arguments are passed through to the selected calibration script.

Examples:
    python example/test/calibrate.py rotation --robot linker_l20 --input pico4 --hand right
    python example/test/calibrate.py scaling --robot linker_l20 --input pico4 --hand right --write
    python example/test/calibrate.py pinch --robot linker_l20 --input pico4 --hand right --write
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODES = {
    "rotation": "calibrate_rotation.py",
    "rot": "calibrate_rotation.py",
    "scaling": "calibrate_scaling.py",
    "scale": "calibrate_scaling.py",
    "pinch_scaling": "calibrate_pinch_scaling.py",
    "pinch": "calibrate_pinch_scaling.py",
}


def _print_help() -> None:
    modes = ", ".join(sorted(MODES))
    print("Usage:")
    print("  python example/test/calibrate.py <mode> [mode args...]")
    print()
    print("Modes:")
    print("  rotation | rot       标定 mediapipe_rotation")
    print("  scaling  | scale     标定 segment_scaling")
    print("  pinch_scaling | pinch 标定 pinch_scaling")
    print()
    print("Examples:")
    print("  python example/test/calibrate.py rotation --robot linker_l20 --input pico4 --hand right")
    print("  python example/test/calibrate.py scaling --robot linker_l20 --input pico4 --hand right --write")
    print("  python example/test/calibrate.py pinch --robot linker_l20 --input pico4 --hand right --write")
    print()
    print(f"Accepted mode names: {modes}")
    print("Use `python example/test/calibrate.py <mode> --help` for mode-specific options.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return

    mode = sys.argv[1]
    script_name = MODES.get(mode)
    if script_name is None:
        print(f"Unknown calibration mode: {mode!r}\n", file=sys.stderr)
        _print_help()
        raise SystemExit(2)

    script_path = SCRIPT_DIR / script_name
    sys.argv = [str(script_path)] + sys.argv[2:]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
