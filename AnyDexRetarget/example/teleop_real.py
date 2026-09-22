"""Teleoperation with real robot hand hardware.

Uses the Retargeter interface to map hand tracking input to joint targets and
send them to real hardware via:
- wujihandpy (Wuji Hand)
- TCP socket bridge (Shadow Hand)
- direct serial protocol (Inspire RH56DFX)
- Gaia HandSDK (GaiaHand20 / Gaia20)
- RS485 Modbus or SocketCAN (Linker L20)
"""

import argparse
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter
from output.real import (
    GaiaHand20Output,
    InspireSerialOutput,
    LinkerL20CANOutput,
    LinkerL20Output,
    ShadowTCPOutput,
    WujiOutput,
)
from output.real.drivers_linker_l20_can import (
    l20_qpos_to_can_slots
)


# -------------------- Teleoperation --------------------

def run_teleop(
    robot_type: str = "wuji",
    hand_side: str = "right",
    config_path: str = "config/mediapipe/mediapipe_wuji_hand.yaml",
    input_device_type: str = "mediapipe_replay",
    visionpro_ip: str = "192.168.50.127",
    quest3_port: int = 9000,
    quest3_protocol: str = "udp",
    pico4_mode: str = "relay",
    pico4_relay_host: str = "127.0.0.1",
    pico4_relay_port: int = 63902,
    pico4_port: int = 63901,
    pico4_broadcast_port: int = 29888,
    noitom_local_ip: str = "192.168.5.25",
    noitom_local_port: int = 8000,
    noitom_server_ip: str = "192.168.5.33",
    noitom_server_port: int = 9000,
    mediapipe_replay_path: str = "data/avp1.pkl",
    video_path: str = "data/right.mp4",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    show_video: bool = False,
    video_depth_scale: float = 1.25,
    docker_ip: str = "localhost",
    docker_port: int = 5555,
    inspire_port: str = "/dev/ttyUSB0",
    inspire_baudrate: int = 115200,
    inspire_hand_id: int = 1,
    gaia_port: str = "/dev/ttyACM0",
    gaia_baudrate: int = 921600,
    gaia_use_slcan: bool = True,
    gaia_has_main_board: bool = True,
    gaia_speed: float = 1.0,
    gaia_use_broadcast: bool = True,
    gaia_lpf_level: int = 3,
    gaia_pip_scale: float = 1.5,
    gaia_thumb_pip_scale: float = 1.1,
    gaia_enable_delay: float = 1.0,
    gaia_command_hz: float = 100.0,
    gaia_zero_on_close: bool = False,
    l20_transport: str = "rs485",
    l20_port: str = "/dev/ttyUSB0",
    l20_baudrate: int = 460800,
    l20_slave_id: int | None = None,
    l20_speed: int = 200,
    l20_current_limit: int = 200,
    l20_clear_faults: bool = False,
    l20_open_on_exit: bool = False,
    l20_print_registers: bool = False,
    l20_dry_run: bool = False,
    l20_can_channel: str = "can0",
    l20_can_speed: int = 30,
    l20_can_command_hz: float = 60.0,
    l20_can_max_step: int = 10,
):
    """Run teleoperation with real hardware.

    Input acquisition + retargeting runs in a background thread, while the main
    thread sends the latest joint target to hardware at a steady control rate.
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    # Import only the selected input backend. Camera/video backends require
    # OpenCV, which should not be mandatory for headset-based teleoperation.
    if input_device_type == "visionpro":
        from input.visionpro import VisionPro
        input_device = VisionPro(ip=visionpro_ip)
    elif input_device_type == "quest3":
        from input.quest3 import Quest3
        input_device = Quest3(port=quest3_port, protocol=quest3_protocol)
    elif input_device_type == "pico4":
        from input.pico4 import Pico4
        input_device = Pico4(
            mode=pico4_mode,
            relay_host=pico4_relay_host,
            relay_port=pico4_relay_port,
            port=pico4_port,
            broadcast_port=pico4_broadcast_port,
        )
    elif input_device_type == "noitom":
        from input.noitom import NoitomInput
        input_device = NoitomInput(
            local_ip=noitom_local_ip,
            local_port=noitom_local_port,
            server_ip=noitom_server_ip,
            server_port=noitom_server_port,
        )
    elif input_device_type == "mediapipe_replay":
        if not mediapipe_replay_path:
            raise ValueError(
                "mediapipe_replay_path is required for mediapipe_replay mode"
            )
        from input.mediapipe_replay import MediaPipeReplay
        input_device = MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        )
    elif input_device_type == "camera":
        from input.camera import Camera
        input_device = Camera(camera_id=0, show_preview=True)
    elif input_device_type == "realsense":
        from input.realsense import Realsense
        input_device = Realsense(hand_side=hand_side, show_video=show_video)
    elif input_device_type == "video":
        from input.video import Video
        input_device = Video(
            video_path=video_path,
            hand_side=hand_side,
            show_video=show_video,
            playback_speed=playback_speed,
            loop=playback_loop,
            depth_scale=video_depth_scale,
        )
    else:
        raise ValueError(f"Unknown input device type: {input_device_type}")

    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Get joint names from retargeter for name-aware hardware mappings.
    joint_names = retargeter.optimizer.robot.dof_joint_names

    # Initialize hardware only after input/config validation. This avoids
    # leaving an enabled hand behind when an input dependency or YAML is bad.
    if robot_type == "wuji":
        output = WujiOutput()
    elif robot_type == "shadow":
        output = ShadowTCPOutput(docker_ip=docker_ip, port=docker_port)
    elif robot_type == "inspire":
        output = InspireSerialOutput(
            port_name=inspire_port,
            baudrate=inspire_baudrate,
            hand_id=inspire_hand_id,
        )
    elif robot_type == "gaia":
        output = GaiaHand20Output(
            port_name=gaia_port,
            hand_side=hand_side,
            baudrate=gaia_baudrate,
            use_slcan=gaia_use_slcan,
            has_main_board=gaia_has_main_board,
            speed=gaia_speed,
            use_broadcast=gaia_use_broadcast,
            lpf_level=gaia_lpf_level,
            pip_scale=gaia_pip_scale,
            thumb_pip_scale=gaia_thumb_pip_scale,
            enable_delay=gaia_enable_delay,
            command_hz=gaia_command_hz,
            zero_on_close=gaia_zero_on_close,
        )
    elif robot_type == "linker_l20":
        if l20_transport == "can":
            if l20_open_on_exit:
                raise ValueError("--l20-open-on-exit is only supported by the RS485 transport")
            output = LinkerL20CANOutput(
                hand_side=hand_side,
                can_channel=l20_can_channel,
                speed=l20_can_speed,
                command_hz=l20_can_command_hz,
                max_step=l20_can_max_step,
                clear_faults=l20_clear_faults,
                print_registers=l20_print_registers,
                dry_run=l20_dry_run,
            )
        elif l20_transport == "rs485":
            output = LinkerL20Output(
                port=l20_port,
                baudrate=l20_baudrate,
                slave_id=l20_slave_id if l20_slave_id is not None else (0x29 if hand_side == "right" else 0x2A),
                speed=l20_speed,
                current_limit=l20_current_limit,
                clear_faults=l20_clear_faults,
                open_on_exit=l20_open_on_exit,
                print_registers=l20_print_registers,
                dry_run=l20_dry_run,
            )
        else:
            raise ValueError("l20_transport must be 'rs485' or 'can'")
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    if input_device_type in ("mediapipe_replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    input_data_log = [] if enable_recording else None
    start_time = time.time()

    latest_qpos = np.zeros(retargeter.num_joints, dtype=np.float32)
    latest_fingers_pose = np.zeros((21, 3), dtype=np.float32)
    qpos_lock = threading.Lock()
    qpos_ready = False
    stop_event = threading.Event()
    input_frame_count = 0
    control_frame_count = 0
    input_thread_error = None

    def input_thread_fn():
        nonlocal qpos_ready, input_frame_count, input_thread_error
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception as exc:
                input_thread_error = exc
                break

            fingers_pose = fingers_data[f"{hand_side}_fingers"]
            if np.allclose(fingers_pose, 0):
                if (
                    input_device_type in ("mediapipe_replay", "video")
                    and not playback_loop
                    and getattr(input_device, "_finished", False)
                ):
                    break
                time.sleep(0.01)
                continue

            if enable_recording and input_data_log is not None:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": fingers_data["left_fingers"].copy(),
                    "right_fingers": fingers_data["right_fingers"].copy(),
                })

            qpos = retargeter.retarget(fingers_pose)
            if robot_type == "linker_l20" and l20_transport == "can":
                # 预留槽位先按官方 GUI 的值填 255
                reserved = np.array([255, 255, 255, 255], dtype=np.int64)

                slots = l20_qpos_to_can_slots(
                    qpos,
                    joint_names,
                    reserved,
                )

                slot_names = [
                    "拇指根部",
                    "食指根部",
                    "中指根部",
                    "无名指根部",
                    "小指根部",

                    "拇指侧摆",
                    "食指侧摆",
                    "中指侧摆",
                    "无名指侧摆",
                    "小指侧摆",

                    "拇指横摆",

                    "预留11",
                    "预留12",
                    "预留13",
                    "预留14",

                    "拇指尖部",
                    "食指末端",
                    "中指末端",
                    "无名指末端",
                    "小指末端",
                ]

                print("\n========== L20 ALL RAW ==========")

                for i, (name, value) in enumerate(zip(slot_names, slots)):
                    print(f"slot {i:02d}  {name:8s}: {int(value):3d}")

                print("RAW:", slots.tolist())
                print("=================================\n")
            with qpos_lock:
                latest_qpos[:] = qpos
                latest_fingers_pose[:] = fingers_pose
                qpos_ready = True
            input_frame_count += 1

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print("Starting teleoperation...")
        print(f"  Robot: {robot_type}")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        if input_device_type == "pico4":
            if pico4_mode == "direct":
                print(f"  Pico4 mode: direct (tcp={pico4_port}, udp_broadcast={pico4_broadcast_port})")
            else:
                print(f"  Pico4 mode: relay ({pico4_relay_host}:{pico4_relay_port})")
        if robot_type == "shadow":
            print(f"  Shadow bridge: {docker_ip}:{docker_port}")
        elif robot_type == "inspire":
            print(f"  Inspire serial: {inspire_port} @ {inspire_baudrate} (id={inspire_hand_id})")
        elif robot_type == "gaia":
            print(
                f"  Gaia HandSDK: {gaia_port} @ {gaia_baudrate} "
                f"(slcan={gaia_use_slcan}, main_board={gaia_has_main_board}, "
                f"command_hz={gaia_command_hz:g}, pip_scale={gaia_pip_scale:g}, "
                f"thumb_pip_scale={gaia_thumb_pip_scale:g})"
            )
        elif robot_type == "linker_l20":
            if l20_transport == "can":
                print(
                    f"  Linker L20 CAN: {l20_can_channel} "
                    f"(speed={l20_can_speed}, command_hz={l20_can_command_hz:g}, "
                    f"max_step={l20_can_max_step}, dry_run={l20_dry_run})"
                )
            else:
                effective_l20_slave_id = l20_slave_id if l20_slave_id is not None else (0x29 if hand_side == "right" else 0x2A)
                print(
                    f"  Linker L20 RS485: {l20_port} @ {l20_baudrate} "
                    f"(slave_id={effective_l20_slave_id}, speed={l20_speed}, "
                    f"current_limit={l20_current_limit}, dry_run={l20_dry_run})"
                )
        print("=" * 50)

        input_thread.start()

        # The official G20 API sends five blocking CAN frames per hand target.
        # Poll faster than the requested rate so 50-60 Hz targets are not
        # quantized down by the generic 100 Hz outer loop.
        control_hz = (
            max(100.0, l20_can_command_hz * 10.0)
            if robot_type == "linker_l20" and l20_transport == "can"
            else 100.0
        )
        control_dt = 1.0 / control_hz
        fps_start_time = time.time()

        while True:
            loop_start = time.time()

            with qpos_lock:
                if qpos_ready:
                    qpos_to_send = latest_qpos.copy()
                    fingers_pose_to_send = latest_fingers_pose.copy()
                else:
                    qpos_to_send = None
                    fingers_pose_to_send = None

            if qpos_to_send is not None:
                if hasattr(output, "set_tracking_keypoints"):
                    output.set_tracking_keypoints(fingers_pose_to_send)
                output.send(qpos_to_send, joint_names)
                control_frame_count += 1

            if control_frame_count > 0 and control_frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                control_fps = control_frame_count / elapsed
                input_fps = input_frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Control FPS: {control_fps:.1f}  |  Input FPS: {input_fps:.1f}")

            if input_thread_error is not None and not input_thread.is_alive():
                raise RuntimeError("Input/retargeting thread stopped unexpectedly") from input_thread_error

            if (
                input_device_type in ("mediapipe_replay", "video")
                and not playback_loop
                and getattr(input_device, "_finished", False)
                and not input_thread.is_alive()
            ):
                break

            sleep_time = control_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        output.close()

    return input_data_log


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Teleoperation with Real Robot Hand Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wuji Hand with replay data
  python teleop_real.py --robot wuji --play data/avp1.pkl

  # Shadow Hand via TCP to Docker ROS bridge
  python teleop_real.py --robot shadow --play data/avp1.pkl --docker-ip localhost --docker-port 5555

  # Live VisionPro input with Shadow Hand
  python teleop_real.py --robot shadow --input visionpro --ip <your-vision-pro-ip>

  # Inspire Hand via direct serial control
  python teleop_real.py --robot inspire --input noitom --hand right --noitom-local-ip 192.168.5.25 --inspire-port /dev/ttyUSB0

  # Pico 4 direct mode with PC broadcast discovery
  python teleop_real.py --robot inspire --input pico4 --hand right --pico4-mode direct --inspire-port /dev/ttyUSB0

  # GaiaHand20 through HandSDK using the local Pico relay daemon
  python teleop_real.py --robot gaia --input pico4 --hand right --pico4-mode relay --gaia-port /dev/ttyACM0

  # Linker L20 through RS485 Modbus
  python teleop_real.py --robot linker_l20 --input pico4 --hand right --pico4-mode relay --l20-port /dev/ttyUSB0

  # Linker L20 right hand through SocketCAN (start at the calibrated 60 Hz default)
  python teleop_real.py --robot linker_l20 --input camera --hand right --l20-transport can --l20-can-channel can0 --l20-can-command-hz 60

  # Linker L20 dry-run register print without opening serial
  python teleop_real.py --robot linker_l20 --play data/avp1.pkl --l20-dry-run --l20-print-registers

  # GaiaHand20 without main board (direct serial)
  python teleop_real.py --robot gaia --play data/avp1.pkl --gaia-port /dev/ttyUSB0 --gaia-baudrate 230400 --no-gaia-use-slcan --no-gaia-has-main-board

  # RealSense camera input
  python teleop_real.py --robot shadow --realsense

  # Video file input
  python teleop_real.py --robot shadow --video data/right.mp4

  # Record input data while using VisionPro
  python teleop_real.py --input visionpro --record
        """,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", type=str, default="adaptive",
                        choices=["adaptive", "vector"],
                        help="Optimizer type: adaptive (default) or vector (KeyVectorOptimizer)")
    parser.add_argument("--robot", type=str, default="wuji",
                        choices=["wuji", "shadow", "inspire", "gaia", "linker_l20"],
                        help="Robot hand type (default: wuji)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Hand side (default: right)")

    parser.add_argument("--input", type=str, default=None,
                        choices=["visionpro", "quest3", "pico4", "noitom", "mediapipe_replay", "camera", "realsense", "video"],
                        help="Input device type")
    parser.add_argument("--realsense", action="store_true",
                        help="Use RealSense camera (shortcut for --input realsense)")
    parser.add_argument("--show-video", action="store_true",
                        help="Show video with MediaPipe landmarks overlay")
    parser.add_argument("--video-depth-scale", type=float, default=1.25,
                        help="Extra scale applied to MediaPipe z/depth for video input (default: 1.25)")

    parser.add_argument("--play", type=str, default=None, metavar="FILE",
                        help="Play MediaPipe recording file (shortcut for --input mediapipe_replay)")
    parser.add_argument("--video", type=str, default=None, metavar="FILE",
                        help="Play MP4/AVI video file with MediaPipe detection (shortcut for --input video)")

    parser.add_argument("--ip", type=str, default="192.168.50.127",
                        help="VisionPro IP address (default: 192.168.50.127)")
    parser.add_argument("--quest3-port", type=int, default=9000,
                        help="Quest 3 HTS listener port (default: 9000)")
    parser.add_argument("--quest3-protocol", type=str, default="udp", choices=["udp", "tcp"],
                        help="Quest 3 HTS transport protocol (default: udp)")

    parser.add_argument("--pico4-mode", type=str, default="relay", choices=["relay", "direct"],
                        help="Pico 4 input mode: relay daemon (default) or direct TCP server")
    parser.add_argument("--pico4-relay-host", type=str, default="127.0.0.1",
                        help="Pico 4 relay daemon host (default: 127.0.0.1)")
    parser.add_argument("--pico4-relay-port", type=int, default=63902,
                        help="Pico 4 relay daemon port (default: 63902)")
    parser.add_argument("--pico4-port", type=int, default=63901,
                        help="Pico 4 TCP listen port (default: 63901)")
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888,
                        help="Pico 4 direct-mode UDP broadcast port (default: 29888)")

    parser.add_argument("--noitom-local-ip", type=str, default="192.168.5.25",
                        help="Noitom: Linux IP (must match Axis Studio destination, default: 192.168.5.25)")
    parser.add_argument("--noitom-local-port", type=int, default=8000,
                        help="Noitom: local UDP port (default: 7012)")
    parser.add_argument("--noitom-server-ip", type=str, default="192.168.5.33",
                        help="Noitom: Windows Axis Studio IP (default: 192.168.5.33)")
    parser.add_argument("--noitom-server-port", type=int, default=9000,
                        help="Noitom: Axis Studio BVH broadcast port (default: 9000)")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed for replay mode (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable looping for replay/video mode")
    parser.add_argument("--record", action="store_true",
                        help="Record input data to file")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Output file for recording (default: auto-generated)")

    # Shadow Hand TCP options
    parser.add_argument("--docker-ip", type=str, default="localhost",
                        help="Docker ROS bridge IP (default: localhost, for --robot shadow)")
    parser.add_argument("--docker-port", type=int, default=5555,
                        help="Docker ROS bridge port (default: 5555, for --robot shadow)")
    parser.add_argument("--inspire-port", type=str, default="/dev/ttyUSB0",
                        help="Inspire serial port (default: /dev/ttyUSB0, for --robot inspire)")
    parser.add_argument("--inspire-baudrate", type=int, default=115200,
                        help="Inspire serial baudrate (default: 115200, for --robot inspire)")
    parser.add_argument("--inspire-hand-id", type=int, default=1,
                        help="Inspire hand ID in serial protocol (default: 1, for --robot inspire)")

    # GaiaHand20 / HandSDK options
    parser.add_argument("--gaia-port", type=str, default="/dev/ttyACM0",
                        help="GaiaHand20 serial/SLCAN port (default: /dev/ttyACM0)")
    parser.add_argument("--gaia-baudrate", type=int, default=921600,
                        help="GaiaHand20 baudrate (default: 921600; use 230400 without main board)")
    parser.add_argument("--gaia-use-slcan", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Gaia HandSDK SLCAN transport (default: enabled)")
    parser.add_argument("--gaia-has-main-board", action=argparse.BooleanOptionalAction, default=True,
                        help="Declare that the Gaia hand has a main board (default: enabled)")
    parser.add_argument("--gaia-speed", type=float, default=1.0,
                        help="Gaia movement speed in (0, 1] (default/maximum: 1.0)")
    parser.add_argument("--gaia-use-broadcast", action=argparse.BooleanOptionalAction, default=True,
                        help="Use HandSDK broadcast joint commands (default: enabled)")
    parser.add_argument("--gaia-lpf-level", type=int, choices=range(0, 6), default=3,
                        help="Gaia motor position LPF level, 0-5 (default: 3)")
    parser.add_argument("--gaia-pip-scale", type=float, default=1.5,
                        help="Gaia non-thumb PIP command scale (default: 1.5; DIP follows mechanically)")
    parser.add_argument("--gaia-thumb-pip-scale", type=float, default=1.1,
                        help="Gaia thumb PIP (joint_3) command scale (default: 1.1)")
    parser.add_argument("--gaia-enable-delay", type=float, default=1.0,
                        help="Seconds to wait after enabling Gaia motors (default: 1.0)")
    parser.add_argument("--gaia-command-hz", type=float, default=100.0,
                        help="Maximum Gaia command rate in Hz (default/effective maximum: 100)")
    parser.add_argument("--gaia-zero-on-close", action="store_true",
                        help="Return GaiaHand20 to zero before disabling it on exit")

    # Linker L20 transport options (RS485 remains the default)
    parser.add_argument("--l20-transport", choices=["rs485", "can"], default="rs485",
                        help="Linker L20 transport (default: rs485; use can for SocketCAN)")
    parser.add_argument("--l20-port", type=str, default="/dev/ttyUSB0",
                        help="Linker L20 serial/RS485 port (default: /dev/ttyUSB0)")
    parser.add_argument("--l20-baudrate", type=int, default=460800,
                        help="Linker L20 baudrate (default: 460800)")
    parser.add_argument("--l20-slave-id", type=lambda x: int(x, 0), default=None,
                        help="Linker L20 Modbus slave id (default: 0x29 right, 0x2A left)")
    parser.add_argument("--l20-speed", type=int, default=200,
                        help="Linker L20 speed register value, 0-255 (default: 200)")
    parser.add_argument("--l20-current-limit", type=int, default=200,
                        help="Linker L20 current-limit register value, 0-255 (default: 200)")
    parser.add_argument("--l20-clear-faults", action="store_true",
                        help="Clear Linker L20 fault registers during startup")
    parser.add_argument("--l20-open-on-exit", action="store_true",
                        help="Return Linker L20 to open-palm registers before closing")
    parser.add_argument("--l20-print-registers", action="store_true",
                        help="Print Linker L20 register commands")
    parser.add_argument("--l20-dry-run", action="store_true",
                        help="Compute and print Linker L20 commands without opening hardware")
    parser.add_argument("--l20-can-channel", type=str, default="can0",
                        help="Linker L20 SocketCAN channel (default: can0)")
    parser.add_argument("--l20-can-speed", type=int, default=30,
                        help="Linker L20 CAN five-finger speed value, 0-255 (default: 30)")
    parser.add_argument("--l20-can-command-hz", type=float, default=60.0,
                        help="Maximum Linker L20 CAN command rate, 1-200 Hz (default: 60)")
    parser.add_argument("--l20-can-max-step", type=int, default=10,
                        help="Maximum raw position change per CAN command, 1-255 (default: 10)")


    args = parser.parse_args(argv)

    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = "data/avp1.pkl"

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")
    if input_device_type == "video" and not video_path:
        parser.error("--video FILE is required for video mode")

    config_path = args.config
    if config_path is None:
        robot_name_map = {
            "wuji": "wuji_hand",
            "shadow": "shadow_hand",
            "inspire": "inspire_hand",
            "gaia": "gaia_hand20",
            "linker_l20": "linker_l20",
        }
        input_to_dir = {
            "quest3": "quest3",
            "visionpro": "avp",
            "noitom": "noitom",
            "pico4": "pico4",
        }
        config_dir = input_to_dir.get(input_device_type, "mediapipe")
        robot_file = robot_name_map.get(args.robot, args.robot)
        config_path = f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"

    log = run_teleop(
        robot_type=args.robot,
        hand_side=args.hand,
        config_path=config_path,
        input_device_type=input_device_type,
        visionpro_ip=args.ip,
        quest3_port=args.quest3_port,
        quest3_protocol=args.quest3_protocol,
        pico4_mode=args.pico4_mode,
        pico4_relay_host=args.pico4_relay_host,
        pico4_relay_port=args.pico4_relay_port,
        pico4_port=args.pico4_port,
        pico4_broadcast_port=args.pico4_broadcast_port,
        noitom_local_ip=args.noitom_local_ip,
        noitom_local_port=args.noitom_local_port,
        noitom_server_ip=args.noitom_server_ip,
        noitom_server_port=args.noitom_server_port,
        mediapipe_replay_path=mediapipe_replay_path,
        video_path=video_path,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
        show_video=args.show_video,
        video_depth_scale=args.video_depth_scale,
        docker_ip=args.docker_ip,
        docker_port=args.docker_port,
        inspire_port=args.inspire_port,
        inspire_baudrate=args.inspire_baudrate,
        inspire_hand_id=args.inspire_hand_id,
        gaia_port=args.gaia_port,
        gaia_baudrate=args.gaia_baudrate,
        gaia_use_slcan=args.gaia_use_slcan,
        gaia_has_main_board=args.gaia_has_main_board,
        gaia_speed=args.gaia_speed,
        gaia_use_broadcast=args.gaia_use_broadcast,
        gaia_lpf_level=args.gaia_lpf_level,
        gaia_pip_scale=args.gaia_pip_scale,
        gaia_thumb_pip_scale=args.gaia_thumb_pip_scale,
        gaia_enable_delay=args.gaia_enable_delay,
        gaia_command_hz=args.gaia_command_hz,
        gaia_zero_on_close=args.gaia_zero_on_close,
        l20_transport=args.l20_transport,
        l20_port=args.l20_port,
        l20_baudrate=args.l20_baudrate,
        l20_slave_id=args.l20_slave_id,
        l20_speed=args.l20_speed,
        l20_current_limit=args.l20_current_limit,
        l20_clear_faults=args.l20_clear_faults,
        l20_open_on_exit=args.l20_open_on_exit,
        l20_print_registers=args.l20_print_registers,
        l20_dry_run=args.l20_dry_run,
        l20_can_channel=args.l20_can_channel,
        l20_can_speed=args.l20_can_speed,
        l20_can_command_hz=args.l20_can_command_hz,
        l20_can_max_step=args.l20_can_max_step,
    )

    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"

        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
