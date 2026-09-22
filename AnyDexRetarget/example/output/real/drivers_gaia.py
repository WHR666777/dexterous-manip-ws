"""Output driver for GaiaHand20 through the official Gaia HandSDK."""

import time

import numpy as np

from .base import HandOutput

_GAIA_FINGER_JOINT_COUNTS = (
    ("thumb", 4),
    ("index", 3),
    ("middle", 3),
    ("ring", 3),
    ("little", 3),
)
_GAIA_MAX_CONSECUTIVE_SEND_FAILURES = 5


def _gaia_retarget_to_real(
    retarget_output: np.ndarray,
    joint_names: list[str],
    hand_side: str,
    pip_scale: float = 1.0,
    thumb_pip_scale: float = 1.0,
) -> list[float]:
    """Map Gaia's 20-joint retarget output to the HandSDK 16-joint order.

    The retargeting URDF contains four revolute joints per finger. For the four
    non-thumb fingers, joint 4 is passive and coupled to joint 3 on the real
    GaiaHand20. HandSDK therefore expects only the active joints in this order:
    thumb(4), index(3), middle(3), ring(3), little(3).
    """
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.ndim != 1:
        raise ValueError(f"Expected 1-D Gaia retarget output, got shape {output.shape}")
    if output.shape[0] != len(joint_names):
        raise ValueError(
            "Gaia qpos/joint-name length mismatch: "
            f"{output.shape[0]} positions vs {len(joint_names)} names"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Gaia retarget output: contains NaN/Inf")
    if not np.isfinite(pip_scale) or pip_scale <= 0.0:
        raise ValueError(f"Gaia PIP scale must be positive, got {pip_scale}")
    if not np.isfinite(thumb_pip_scale) or thumb_pip_scale <= 0.0:
        raise ValueError(
            f"Gaia thumb PIP scale must be positive, got {thumb_pip_scale}"
        )

    name_to_position = dict(zip(joint_names, output))
    if len(name_to_position) != len(joint_names):
        raise ValueError("Gaia retarget joint names contain duplicates")

    sdk_positions: list[float] = []
    missing_names: list[str] = []
    for finger, joint_count in _GAIA_FINGER_JOINT_COUNTS:
        for joint_index in range(1, joint_count + 1):
            name = f"{hand_side}_{finger}_joint_{joint_index}"
            if name not in name_to_position:
                missing_names.append(name)
                continue

            value = float(name_to_position[name])
            if joint_index == 3:
                # Compensate only the real active PIP commands; simulation and
                # retargeting remain unchanged. The four non-thumb DIPs follow
                # their PIPs mechanically, while the thumb DIP remains an
                # independently commanded joint and is not scaled here.
                scale = thumb_pip_scale if finger == "thumb" else pip_scale
                value = float(np.clip(value * scale, 0.0, np.pi / 2.0))
            sdk_positions.append(value)

    if missing_names:
        raise ValueError(
            "Gaia retarget output is missing active joint(s): "
            + ", ".join(missing_names)
        )
    return sdk_positions


class GaiaHand20Output(HandOutput):
    """Output driver for GaiaHand20 through the official Gaia HandSDK."""

    def __init__(
        self,
        port_name: str,
        hand_side: str,
        baudrate: int = 921600,
        use_slcan: bool = True,
        has_main_board: bool = True,
        speed: float = 1.0,
        use_broadcast: bool = True,
        lpf_level: int = 3,
        pip_scale: float = 1.5,
        thumb_pip_scale: float = 1.1,
        enable_delay: float = 1.0,
        command_hz: float = 100.0,
        zero_on_close: bool = False,
    ):
        try:
            from hand import create_hand
        except ImportError as exc:
            raise ImportError(
                "Gaia HandSDK is required for GaiaHand20 control. Install the "
                "handsdk wheel matching your Python/platform from "
                "gaia_hand/02.HandSDK/packages, then verify with "
                "`python -c \"import hand\"`."
            ) from exc

        if not 0.0 < speed <= 1.0:
            raise ValueError(f"Gaia speed must be in (0, 1], got {speed}")
        if not 0 <= lpf_level <= 5:
            raise ValueError(f"Gaia LPF level must be in [0, 5], got {lpf_level}")
        if not np.isfinite(pip_scale) or pip_scale <= 0.0:
            raise ValueError(f"Gaia PIP scale must be positive, got {pip_scale}")
        if not np.isfinite(thumb_pip_scale) or thumb_pip_scale <= 0.0:
            raise ValueError(
                f"Gaia thumb PIP scale must be positive, got {thumb_pip_scale}"
            )
        if command_hz <= 0:
            raise ValueError(f"Gaia command rate must be positive, got {command_hz}")

        self._port_name = port_name
        self._hand_side = hand_side
        self._baudrate = int(baudrate)
        self._use_slcan = bool(use_slcan)
        self._has_main_board = bool(has_main_board)
        self._speed = float(speed)
        self._use_broadcast = bool(use_broadcast)
        self._pip_scale = float(pip_scale)
        self._thumb_pip_scale = float(thumb_pip_scale)
        self._zero_on_close = bool(zero_on_close)
        self._min_command_period = 1.0 / float(command_hz)
        self._last_command_time = -float("inf")
        self._consecutive_send_failures = 0
        self._connected = False
        self._enabled = False

        self.hand = create_hand(
            "gaia20",
            hand_side,
            port=port_name,
            baudrate=self._baudrate,
            use_slcan=self._use_slcan,
            has_main_board=self._has_main_board,
        )

        try:
            if not self.hand.connect():
                raise ConnectionError(
                    f"Failed to connect to GaiaHand20 at {port_name}"
                )
            self._connected = True
            print(
                f"Connected to GaiaHand20 ({hand_side}) at {port_name} "
                f"@ {self._baudrate} baud "
                f"(slcan={self._use_slcan}, main_board={self._has_main_board})."
            )

            self.hand.config_pos_lpf_lv(device_id=255, level=int(lpf_level))
            time.sleep(0.5)

            if not self.hand.enable_all_motors_broadcast(True):
                raise RuntimeError("Failed to enable all GaiaHand20 motors")
            self._enabled = True
            time.sleep(max(0.0, float(enable_delay)))
        except Exception:
            self.close()
            raise

    def send(self, qpos, joint_names):
        now = time.monotonic()
        if now - self._last_command_time < self._min_command_period:
            return

        positions = _gaia_retarget_to_real(
            qpos,
            joint_names,
            self._hand_side,
            pip_scale=self._pip_scale,
            thumb_pip_scale=self._thumb_pip_scale,
        )
        success = self.hand.move_joints_pos(
            positions,
            speed=self._speed,
            use_broadcast=self._use_broadcast,
        )
        self._last_command_time = now

        if success:
            self._consecutive_send_failures = 0
            return

        self._consecutive_send_failures += 1
        if self._consecutive_send_failures >= _GAIA_MAX_CONSECUTIVE_SEND_FAILURES:
            raise RuntimeError(
                "GaiaHand20 rejected "
                f"{self._consecutive_send_failures} consecutive joint commands"
            )
        print(
            "Warning: GaiaHand20 joint command failed "
            f"({self._consecutive_send_failures}/"
            f"{_GAIA_MAX_CONSECUTIVE_SEND_FAILURES})."
        )

    def close(self):
        hand = getattr(self, "hand", None)
        if hand is None:
            return

        try:
            if self._connected and self._zero_on_close:
                print("Returning GaiaHand20 to zero position...")
                hand.hand_zero()
                time.sleep(1.0)
        except Exception as exc:
            print(f"Warning: GaiaHand20 zero-on-close failed: {exc}")

        try:
            if self._connected and self._enabled:
                hand.enable_all_motors_broadcast(False)
        except Exception as exc:
            print(f"Warning: failed to disable GaiaHand20 motors: {exc}")
        finally:
            self._enabled = False

        try:
            hand.close()
        except Exception as exc:
            print(f"Warning: failed to close GaiaHand20 connection: {exc}")
        finally:
            self._connected = False
            self.hand = None
            print(f"Closed GaiaHand20 connection at {self._port_name}.")


__all__ = ["GaiaHand20Output", "_gaia_retarget_to_real"]
