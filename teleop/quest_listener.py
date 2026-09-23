"""Persistent Quest TCP listener with a localhost frame relay.

The Hand Tracking Streamer connects to the AnyDex Quest3 listener only once on
some headset versions.  Keeping that listener in this small, hardware-free
process lets arm/hand control processes be restarted without disturbing the
headset connection.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


def _make_anydex_quest(config: dict[str, Any]):
    """Create the project's sole Quest TCP/CSV/coordinate-conversion input."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root / "AnyDexRetarget", project_root / "AnyDexRetarget" / "example"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from input.quest3 import Quest3

    quest = config["quest"]
    return Quest3(host=quest["listen_host"], port=quest["port"], protocol=quest["transport"])


def _encode_frame(frame) -> bytes:
    if frame is None:
        return b"null\n"
    payload = {
        "side": frame.side,
        "wrist_position": np.asarray(frame.wrist_position, dtype=float).tolist(),
        "wrist_quat": np.asarray(frame.wrist_quat, dtype=float).tolist(),
        "landmarks": np.asarray(frame.landmarks, dtype=float).tolist(),
        "received_at": float(frame.received_at),
        "pair_skew": float(frame.pair_skew),
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class QuestRelayClient:
    """Read the newest complete Quest frame from the local relay."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.15) -> None:
        self.host, self.port, self.timeout_s = host, int(port), float(timeout_s)

    def _request(self, request: str) -> bytes | None:
        try:
            with socket.create_connection((self.host, self.port), self.timeout_s) as conn:
                conn.settimeout(self.timeout_s)
                conn.sendall((request + "\n").encode("ascii"))
                chunks: list[bytes] = []
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    chunks.append(data)
                    if b"\n" in data:
                        break
        except OSError:
            return None
        return b"".join(chunks).split(b"\n", 1)[0]

    def is_available(self) -> bool:
        """Return whether the persistent listener, not necessarily HTS, is up."""
        return self._request("ping") == b'{"status":"ok"}'

    def get_hand_frame(self, side: str):
        response = self._request(side.lower())
        if response is None:
            return None
        try:
            payload = json.loads(response)
            if payload is None:
                return None
            return SimpleNamespace(
                side=payload["side"],
                wrist_position=np.asarray(payload["wrist_position"], dtype=np.float64),
                wrist_quat=np.asarray(payload["wrist_quat"], dtype=np.float64),
                landmarks=np.asarray(payload["landmarks"], dtype=np.float64),
                received_at=float(payload["received_at"]),
                pair_skew=float(payload["pair_skew"]),
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def stop(self) -> None:
        """Match the direct Quest input lifecycle; clients own no resources."""


def serve(config: dict[str, Any]) -> None:
    """Bind the headset listener and serve frame snapshots until Ctrl+C."""
    quest = config["quest"]
    relay_host = quest["relay_host"]
    relay_port = int(quest["relay_port"])
    source = _make_anydex_quest(config)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((relay_host, relay_port))
        server.listen(16)
        server.settimeout(0.5)
        print(
            f"Quest listener persistent: HTS TCP {quest['listen_host']}:{quest['port']}; "
            f"local relay {relay_host}:{relay_port}. Ctrl+C stops both."
        )
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            with conn:
                conn.settimeout(0.2)
                try:
                    side = conn.recv(16).decode("ascii").strip().lower()
                except (OSError, UnicodeDecodeError):
                    continue
                if side == "ping":
                    response = b'{"status":"ok"}\n'
                else:
                    frame = source.get_hand_frame(side) if side in ("left", "right") else None
                    response = _encode_frame(frame)
                try:
                    conn.sendall(response)
                except OSError:
                    pass
    except OSError as exc:
        raise RuntimeError(
            f"Cannot start persistent Quest listener on {relay_host}:{relay_port}: {exc}"
        ) from exc
    finally:
        server.close()
        source.stop()
