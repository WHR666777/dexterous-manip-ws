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
from cloth_agent.geometry import transform
from teleop.tcp_calibration import annotate, load_observations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="teleop full wrist, freeze replay fingers, C saves RGB-D/FK")
    collect.add_argument("directory", type=Path, help="new session directory")
    collect.add_argument("--config", default="configs/pico4_nero_l20.yaml")
    collect.add_argument("--hand-frame", type=int, default=0,
                         help="0-based replay frame (default: 0); use -1 for the final closed pose")
    collect.add_argument("--execute", action="store_true")
    label = commands.add_parser("annotate", help="offline click contact point in each RGB-D sample")
    label.add_argument("directory", type=Path)
    label.add_argument("--scale", type=float, default=1., help="display magnification, e.g. 2")
    solve = commands.add_parser("solve", help="offline joint least-squares calibration")
    solve.add_argument("directory", type=Path)
    solve.add_argument("--output", type=Path, help="defaults to DIRECTORY/calibration.yaml")
    solve.add_argument("--max-rms-mm", type=float, default=10., help="do not publish a result above this fit RMS")
    merge = commands.add_parser("merge", help="combine two calibrations in the right-base task frame")
    merge.add_argument("right", type=Path)
    merge.add_argument("left", type=Path)
    merge.add_argument("--config", type=Path, default=ROOT / "configs/cloth_agent.yaml",
                       help="read existing shared open/replay hand presets")
    merge.add_argument("--output", type=Path, required=True)
    return parser


def merge_calibrations(right_path, left_path, config_path, output):
    """离线合并到公共 task，完整保留两侧原始工具标定及参考角。"""
    sources = {"right": Path(right_path).resolve(), "left": Path(left_path).resolve()}
    values = {side: yaml.safe_load(path.read_text()) for side, path in sources.items()}
    for side, calibration in values.items():
        if calibration["hand_side"] != side:
            raise ValueError(f"Expected {side} hand calibration")
    config = yaml.safe_load(Path(config_path).read_text())
    presets = dict(config["arms"]["right"]["hand_poses"])
    presets["closed"] = values["right"]["hand_pose_raw"]
    right_camera = transform(values["right"]["base_from_camera"])
    arms = {}
    for side, calibration in values.items():
        reference = calibration["reference_roll_pitch_deg"]
        tool = transform(calibration["flange_from_tcp"])
        task_base = (np.eye(4) if side == "right" else
                     right_camera @ np.linalg.inv(transform(calibration["base_from_camera"])))
        arms[side] = {"task_from_base": task_base.tolist(), "flange_from_tcp": tool.tolist(),
                      "reference_roll_pitch_deg": reference,
                      "hand_pose_raw": calibration["hand_pose_raw"],
                      "fit": calibration["fit"],
                      "source_file": str(sources[side].relative_to(ROOT)) if sources[side].is_relative_to(ROOT)
                                     else str(sources[side])}
    combined = {"version": 2, "transform_convention": "a_from_b maps coordinates from b to a",
                "units": {"translation": "m", "angles": "rad", "reference_roll_pitch": "deg"},
                "task_frame": "right Nero base_link", "camera_frame": values["right"]["camera_frame"],
                "task_from_camera": right_camera.tolist(), "hand_poses": presets, "arms": arms,
                "notes": ["Both sides were calibrated in first-frame hand poses, as specified by the operator.",
                          "Shared open/closed/replay_start commands use the same raw values on both hands.",
                          "Both flange_from_tcp transforms and reference angles are preserved from source calibrations.",
                          "Camera and both bases must remain fixed between the source calibrations."]}
    output = Path(output)
    output.write_text(yaml.safe_dump(combined, sort_keys=False, allow_unicode=True))
    print(f"Saved combined calibration {output.resolve()}")
    return combined


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
    elif args.command == "merge":
        merge_calibrations(args.right, args.left, args.config, args.output)
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
