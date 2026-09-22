"""Command-line entry point for Quest 3 -> Nero + L20 teleoperation."""

from __future__ import annotations

import argparse

from .config import load_config, require_site_configuration
from .controller import enable_arm, input_check, preflight, run, reset_arm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quest3_nero_l20.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("input-check", help="Quest + AnyDex only; never opens CAN")
    preflight_parser = subparsers.add_parser("preflight", help="read-only hardware/FK check")
    preflight_parser.add_argument("--control", choices=("arm", "hand", "both"), default="both")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "input-check":
        input_check(config)
        return
    control = "arm" if args.command in ("arm-enable", "arm-reset") else args.control
    require_site_configuration(
        config,
        control=control,
        require_calibration=args.command == "run",
    )
    if args.command == "preflight":
        preflight(config, args.control)
        return
    if not args.execute:
        raise SystemExit("Refusing to command hardware without --execute.")
    actions = {
        "arm-enable": "enable Nero",
        "arm-reset": "disable and reset Nero",
        "run": "start command streaming",
    }
    action = actions[args.command]
    if input(f"Type EXECUTE to {action}: ").strip() != "EXECUTE":
        raise SystemExit("Confirmation did not match; no commands were sent.")
    if args.command == "arm-enable":
        enable_arm(config)
        return
    if args.command == "arm-reset":
        reset_arm(config)
        return
    run(config, args.control, args.record_dir)


if __name__ == "__main__":
    main()
