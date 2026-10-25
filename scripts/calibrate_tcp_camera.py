"""Joint RGB-D camera -> Nero base and fixed pinch TCP calibration.

collect commands hardware through the existing teleop; annotate and solve are offline.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_control.tcp_calibration import solve_camera_tcp
from teleop.tcp_calibration import annotate, load_observations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="teleop full wrist, freeze replay fingers, C saves RGB-D/FK")
    collect.add_argument("directory", type=Path, help="new session directory")
    collect.add_argument("--config", default="configs/pico4_nero_l20.yaml")
    collect.add_argument("--hand-frame", type=int, default=-1, help="0-based replay frame; default -1 is the last frame")
    collect.add_argument("--execute", action="store_true")
    label = commands.add_parser("annotate", help="offline click contact point in each RGB-D sample")
    label.add_argument("directory", type=Path)
    label.add_argument("--scale", type=float, default=1., help="display magnification, e.g. 2")
    solve = commands.add_parser("solve", help="offline joint least-squares calibration")
    solve.add_argument("directory", type=Path)
    solve.add_argument("--output", type=Path, help="defaults to DIRECTORY/calibration.yaml")
    solve.add_argument("--max-rms-mm", type=float, default=7., help="do not publish a result above this fit RMS")
    return parser


def solve_session(directory, output=None, max_rms_mm=5.):
    if not np.isfinite(max_rms_mm) or max_rms_mm <= 0:
        raise ValueError("--max-rms-mm must be finite and positive.")
    metadata, names, points, poses = load_observations(directory)
    result = solve_camera_tcp(points, poses)
    export = result.as_dict(np.asarray(metadata["flange_from_tcp_rotation"]))
    distances = np.asarray(export["fit"]["per_sample_m"])
    for name, distance in zip(names, distances):
        print(f"{name}: {distance * 1000:.3f} mm")
    print(f"Fit RMS={export['fit']['rms_m'] * 1000:.3f} mm; max={distances.max() * 1000:.3f} mm; "
          f"condition={result.condition_number:.1f}")
    if export["fit"]["rms_m"] * 1000 > max_rms_mm:
        raise ValueError("Fit RMS exceeds --max-rms-mm; inspect clicks/depth/FK and recapture before using calibration.")
    export["session"] = str(Path(directory).expanduser().resolve())
    export["sample_names"] = names
    export["hand_side"] = metadata["hand_side"]
    export["hand_pose_raw"] = metadata["hand_pose_raw"]
    export["reference_roll_pitch_deg"] = metadata["reference_roll_pitch_deg"]
    export["tcp_axes"] = metadata["tcp_axes"]
    output = Path(output).expanduser() if output is not None else Path(directory).expanduser() / "calibration.yaml"
    # A failed fit never replaces a previous calibration. Successful reruns replace it.
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(export, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {output.resolve()} (fit residuals are not independent accuracy measurements).")
    return export


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "annotate":
        annotate(args.directory, args.scale)
    elif args.command == "solve":
        solve_session(args.directory, args.output, args.max_rms_mm)
    else:
        # Offline commands never import the CAN/teleop controller or initialize hardware.
        from teleop.config import load_config, require_site_configuration
        from teleop.controller import run

        config = load_config(args.config)
        require_site_configuration(config, control="both", require_start_pose=True)
        if not config["camera"]["enabled"]:
            raise ValueError("Enable camera.enabled in the teleop config before collection.")
        if not args.execute:
            raise SystemExit("Refusing to command hardware without --execute.")
        run(config, "both", None, hand_mode="calibration", calibration_dir=str(args.directory),
            calibration_hand_frame=args.hand_frame)


if __name__ == "__main__":
    main()
