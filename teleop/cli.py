"""Command-line entry point for Pico 4 / Quest 3 -> Nero + L20 teleoperation."""

from __future__ import annotations

import argparse

from .config import load_config, require_site_configuration, select_side_configs
from .controller import (
    enable_arm,
    input_check,
    preflight,
    record_start_pose,
    reset_arm,
    run,
)
from .quest_listener import serve as serve_quest_listener


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pico4_nero_l20.yaml")
    parser.add_argument("--side", choices=("left", "right", "both"),
                        help="defaults to the side(s) defined by the configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("input-check", help="VR + AnyDex only; never opens CAN")
    subparsers.add_parser(
        "quest-listener",
        help="persistently receive HTS on the Quest TCP port and relay frames locally",
    )
    preflight_parser = subparsers.add_parser("preflight", help="read-only hardware/FK check")
    preflight_parser.add_argument("--control", choices=("arm", "hand", "both"), default="both")
    record_parser = subparsers.add_parser(
        "record-start-pose",
        help="save the current Nero joint angles without enabling or moving",
    )
    record_parser.add_argument("--output", help="defaults to arm.start_pose_file")
    record_parser.add_argument("--overwrite", action="store_true")
    enable_parser = subparsers.add_parser("arm-enable", help="enable Nero without a motion target")
    enable_parser.add_argument("--execute", action="store_true", help="required acknowledgement")
    reset_parser = subparsers.add_parser(
        "arm-reset",
        help="disable Nero, then reset its control state without a motion target",
    )
    reset_parser.add_argument("--execute", action="store_true", help="required acknowledgement")
    run_parser = subparsers.add_parser("run", help="stream commands after explicit confirmation")
    run_parser.add_argument("--control", choices=("arm", "hand", "both"), required=True)
    run_parser.add_argument("--execute", action="store_true", help="required acknowledgement")
    run_parser.add_argument("--record-dir")
    run_parser.add_argument(
        "--hand-mode", choices=("tracking", "wrist-only"), default="tracking",
        help="tracking: follow VR fingers; wrist-only: track position/yaw, lock roll/pitch, P=replay, O=reset",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    try:
        selected = select_side_configs(config, args.side)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bimanual = len(selected) == 2
    single = next(iter(selected.values()))
    if args.command == "input-check":
        if bimanual:
            from .bimanual_controller import input_check as bimanual_input_check

            bimanual_input_check(config)
        else:
            input_check(single)
        return
    if args.command == "quest-listener":
        if config["input_device"] != "quest3":
            raise SystemExit("quest-listener requires --config configs/quest3_nero_l20.yaml; "
                             "Pico 4 uses AnyDexRetarget/example/input/pico4_daemon.py.")
        serve_quest_listener(config)
        return
    if args.command == "record-start-pose":
        if bimanual:
            raise SystemExit("record-start-pose requires --side left or --side right.")
        require_site_configuration(
            single,
            control="arm",
            require_calibration=False,
            require_start_pose=False,
        )
        record_start_pose(single, args.output, overwrite=args.overwrite)
        return
    if args.command == "run" and bimanual and args.hand_mode != "tracking":
        raise SystemExit("Bimanual run supports --hand-mode tracking; select one side for wrist-only.")
    control = "arm" if args.command in ("arm-enable", "arm-reset") else args.control
    for side, settings in selected.items():
        try:
            require_site_configuration(
                settings, control=control, require_calibration=args.command == "run",
                require_start_pose=args.command == "run",
            )
        except RuntimeError as exc:
            raise SystemExit(f"{side}: {exc}") from exc
    if args.command == "preflight":
        for side, settings in selected.items():
            print(f"[{side}] preflight")
            preflight(settings, args.control)
        return
    if not args.execute:
        raise SystemExit("Refusing to command hardware without --execute.")
    actions = {
        "arm-enable": "enable Nero",
        "arm-reset": "disable and reset Nero",
        "run": "start command streaming",
    }
    action = actions[args.command]
    if input(f"Type EXECUTE to {action} ({', '.join(selected)}): ").strip() != "EXECUTE":
        raise SystemExit("Confirmation did not match; no commands were sent.")
    if args.command == "arm-enable":
        for settings in selected.values():
            enable_arm(settings)
        return
    if args.command == "arm-reset":
        for settings in selected.values():
            reset_arm(settings)
        return
    if bimanual:
        from .bimanual_controller import run as bimanual_run

        bimanual_run(config, args.control, args.record_dir)
    else:
        if "sides" in config:
            # Selecting one side produces a 23-D dataset, with its own identity/root.
            side = next(iter(selected))
            single["recording"]["output_dir"] += f"_{side}"
            if single["recording"].get("lerobot") is not None:
                single["recording"]["lerobot"]["repo_id"] += f"_{side}"
        run(single, args.control, args.record_dir, hand_mode=args.hand_mode)


if __name__ == "__main__":
    main()
