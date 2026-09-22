"""Direct serial protocol driver for Inspire RH56DFX hand."""

import time

import numpy as np

from .base import HandOutput

_INSPIRE_CHANNEL_INDICES = [4, 6, 2, 0, 9, 8]
_INSPIRE_CHANNEL_MAX_RAD = [1.47, 1.47, 1.47, 1.47, 0.6, 1.308]
_INSPIRE_CHANNEL_INVERT = [True, True, True, True, True, True]
_INSPIRE_SERIAL_RESPONSE_TIMEOUT_S = 0.1


def _inspire_retarget_to_real(retarget_output: np.ndarray) -> list[int]:
    """Map 12-D Inspire retarget output (rad) to 6 serial control channels."""
    output = np.asarray(retarget_output, dtype=np.float64)
    if output.shape[0] < 12:
        raise ValueError(f"Expected at least 12 Inspire joints, got shape {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError("Invalid Inspire retarget output: contains NaN/Inf")

    result: list[int] = []
    for idx, max_rad, invert in zip(
        _INSPIRE_CHANNEL_INDICES,
        _INSPIRE_CHANNEL_MAX_RAD,
        _INSPIRE_CHANNEL_INVERT,
    ):
        value = float(np.clip(output[idx] / max_rad, 0.0, 1.0))
        if invert:
            value = 1.0 - value
        result.append(int(value * 2000))
    return result


class InspireSerialOutput(HandOutput):
    """Direct serial controller for Inspire RH56DFX hand."""

    def __init__(self, port_name: str, baudrate: int = 115200, hand_id: int = 1):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for Inspire hand control. "
                "Install it with `pip install pyserial`."
            ) from exc

        self._port = serial.Serial(port_name, baudrate, timeout=0.001)
        self._hand_id = int(hand_id)
        self._port_name = port_name
        self._baudrate = int(baudrate)
        self.send_count = 0
        print(f"Connected to Inspire hand serial at {port_name} @ {baudrate} baud.")

    def _read_response(self) -> bytes:
        deadline = time.time() + _INSPIRE_SERIAL_RESPONSE_TIMEOUT_S
        input_bytes = bytearray()
        while time.time() < deadline:
            chunk = self._port.read(self._port.in_waiting or 1)
            if chunk:
                input_bytes += chunk
            else:
                break
        return bytes(input_bytes)

    @staticmethod
    def _encode_channels(channels: list[int]) -> list[int]:
        if len(channels) != 6:
            raise ValueError(f"Inspire hand expects 6 channels, got {len(channels)}")
        return [int(np.clip(round(ch / 2.0), 0, 1000)) for ch in channels]

    def send(self, qpos, joint_names):
        channels = _inspire_retarget_to_real(qpos)
        encoded = self._encode_channels(channels)
        packet = bytearray([0xEB, 0x90, self._hand_id, 0x0F, 0x12, 0xCE, 0x05])
        for angle in encoded:
            packet.append(angle & 0xFF)
            packet.append((angle >> 8) & 0xFF)
        checksum = sum(packet[2:2 + 0x0F + 3])
        packet.append(checksum & 0xFF)
        self._port.write(packet)
        self._read_response()
        self.send_count += 1

    def close(self):
        if self._port.is_open:
            self._port.close()
            print(f"Closed Inspire hand serial at {self._port_name} @ {self._baudrate} baud.")


__all__ = ["InspireSerialOutput", "_inspire_retarget_to_real"]
