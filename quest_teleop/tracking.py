"""Thread-safe adapter around the official Hand Tracking Streamer SDK."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np

from .geometry import unity_landmarks_to_flu, unity_pose_to_flu


@dataclass(frozen=True)
class TrackedHand:
    side: str
    wrist_world: np.ndarray
    landmarks_local: np.ndarray
    recv_monotonic_ns: int
    recv_unix_ns: Optional[int]
    source_ts_ns: Optional[int]
    sequence_id: int
    source_frame_seq: Optional[int]

    def age_s(self, now_ns: Optional[int] = None) -> float:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return max(0.0, (now_ns - self.recv_monotonic_ns) / 1e9)


@dataclass(frozen=True)
class TrackedHead:
    pose_world: np.ndarray
    recv_monotonic_ns: int
    recv_unix_ns: Optional[int]
    source_ts_ns: Optional[int]
    sequence_id: int
    source_frame_seq: Optional[int]


@dataclass(frozen=True)
class TrackingSnapshot:
    left: Optional[TrackedHand]
    right: Optional[TrackedHand]
    head: Optional[TrackedHead]
    error: Optional[str] = None
    sequence_reset: bool = False

    def hand(self, side: str) -> Optional[TrackedHand]:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError("side must be left or right")

    def valid(self, sides=("left", "right"), max_age_s: float = 0.15, max_skew_s: float = 0.04) -> bool:
        now = time.monotonic_ns()
        hands = [self.hand(side) for side in sides]
        if any(hand is None or hand.age_s(now) > max_age_s for hand in hands):
            return False
        receive_times = [hand.recv_monotonic_ns for hand in hands if hand is not None]
        return not receive_times or (max(receive_times) - min(receive_times)) / 1e9 <= max_skew_s


def _copy_hand(hand: Optional[TrackedHand]) -> Optional[TrackedHand]:
    if hand is None:
        return None
    return TrackedHand(
        side=hand.side,
        wrist_world=hand.wrist_world.copy(),
        landmarks_local=hand.landmarks_local.copy(),
        recv_monotonic_ns=hand.recv_monotonic_ns,
        recv_unix_ns=hand.recv_unix_ns,
        source_ts_ns=hand.source_ts_ns,
        sequence_id=hand.sequence_id,
        source_frame_seq=hand.source_frame_seq,
    )


class QuestTrackingReceiver:
    """Receive complete HTS frames in a daemon thread; never replay stale data."""

    def __init__(self, transport="tcp", host="0.0.0.0", port=8000, autostart=True):
        self.transport = transport
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = {"left": None, "right": None, "head": None}
        self._error: Optional[str] = None
        self._sequence_reset = False
        self._thread: Optional[threading.Thread] = None
        self._transport_receiver = None
        if autostart:
            self.start()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hts-receiver", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        # HTSClient owns the transport inside iter_events(). Keep the concrete
        # receiver so stop() can unblock a quiet UDP/TCP socket immediately.
        with self._lock:
            transport_receiver = self._transport_receiver
        if transport_receiver is not None:
            try:
                transport_receiver.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def snapshot(self) -> TrackingSnapshot:
        with self._lock:
            head = self._latest["head"]
            head_copy = None if head is None else TrackedHead(
                pose_world=head.pose_world.copy(),
                recv_monotonic_ns=head.recv_monotonic_ns,
                recv_unix_ns=head.recv_unix_ns,
                source_ts_ns=head.source_ts_ns,
                sequence_id=head.sequence_id,
                source_frame_seq=head.source_frame_seq,
            )
            return TrackingSnapshot(
                left=_copy_hand(self._latest["left"]),
                right=_copy_hand(self._latest["right"]),
                head=head_copy,
                error=self._error,
                sequence_reset=self._sequence_reset,
            )

    def clear_sequence_reset(self):
        with self._lock:
            self._sequence_reset = False

    def ingest(self, event) -> None:
        """Ingest an SDK frame. Public for deterministic adapter tests."""
        side_value = str(getattr(getattr(event, "side", None), "value", getattr(event, "side", ""))).lower()
        if side_value in ("left", "right"):
            wrist = event.wrist
            hand = TrackedHand(
                side=side_value,
                wrist_world=unity_pose_to_flu(
                    [wrist.x, wrist.y, wrist.z], [wrist.qx, wrist.qy, wrist.qz, wrist.qw]
                ),
                landmarks_local=unity_landmarks_to_flu(event.landmarks.points),
                recv_monotonic_ns=int(event.recv_ts_ns),
                recv_unix_ns=getattr(event, "recv_time_unix_ns", None),
                source_ts_ns=getattr(event, "source_ts_ns", None),
                sequence_id=int(event.sequence_id),
                source_frame_seq=getattr(event, "source_frame_seq", None),
            )
            with self._lock:
                previous = self._latest[side_value]
                if previous is not None and hand.sequence_id <= previous.sequence_id and hand.recv_monotonic_ns > previous.recv_monotonic_ns:
                    self._sequence_reset = True
                    self._latest[side_value] = hand
                elif previous is None or hand.sequence_id > previous.sequence_id:
                    self._latest[side_value] = hand
            return
        if side_value == "head":
            pose = event.head
            head = TrackedHead(
                pose_world=unity_pose_to_flu(
                    [pose.x, pose.y, pose.z], [pose.qx, pose.qy, pose.qz, pose.qw]
                ),
                recv_monotonic_ns=int(event.recv_ts_ns),
                recv_unix_ns=getattr(event, "recv_time_unix_ns", None),
                source_ts_ns=getattr(event, "source_ts_ns", None),
                sequence_id=int(event.sequence_id),
                source_frame_seq=getattr(event, "source_frame_seq", None),
            )
            with self._lock:
                previous = self._latest["head"]
                if previous is not None and head.sequence_id <= previous.sequence_id and head.recv_monotonic_ns > previous.recv_monotonic_ns:
                    self._sequence_reset = True
                    self._latest["head"] = head
                elif previous is None or head.sequence_id > previous.sequence_id:
                    self._latest["head"] = head

    def _run(self):
        try:
            from hand_tracking_sdk import (
                ErrorPolicy,
                HTSClient,
                HTSClientConfig,
                StreamOutput,
                TCPClientConfig,
                TCPClientLineReceiver,
                TCPServerConfig,
                TCPServerLineReceiver,
                TransportMode,
                UDPLineReceiver,
                UDPReceiverConfig,
            )
            modes = {
                "udp": TransportMode.UDP,
                "tcp": TransportMode.TCP_SERVER,
                "tcp_server": TransportMode.TCP_SERVER,
                "tcp_client": TransportMode.TCP_CLIENT,
            }
            if self.transport not in modes:
                raise ValueError("transport must be udp, tcp/tcp_server, or tcp_client")
            sdk_config = HTSClientConfig(
                    transport_mode=modes[self.transport],
                    host=self.host,
                    port=self.port,
                    timeout_s=0.5,
                    output=StreamOutput.FRAMES,
                    error_policy=ErrorPolicy.TOLERANT,
                    include_wall_time=True,
                )

            def receiver_factory(config):
                if config.transport_mode == TransportMode.UDP:
                    receiver = UDPLineReceiver(UDPReceiverConfig(
                        host=config.host, port=config.port, timeout_s=config.timeout_s,
                    ))
                elif config.transport_mode == TransportMode.TCP_SERVER:
                    receiver = TCPServerLineReceiver(TCPServerConfig(
                        host=config.host, port=config.port,
                        # iter_lines() treats timeout as transient, so a short
                        # accept poll keeps shutdown bounded without making
                        # headset startup time-limited.
                        accept_timeout_s=config.timeout_s,
                        read_timeout_s=config.timeout_s,
                    ))
                else:
                    receiver = TCPClientLineReceiver(TCPClientConfig(
                        host=config.host, port=config.port,
                        connect_timeout_s=config.timeout_s,
                        read_timeout_s=config.timeout_s,
                        reconnect_delay_s=config.reconnect_delay_s,
                    ))
                with self._lock:
                    self._transport_receiver = receiver
                return receiver

            client = HTSClient(sdk_config, receiver_factory=receiver_factory)
            for event in client.iter_events():
                if self._stop.is_set():
                    break
                self.ingest(event)
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._transport_receiver = None


class WorldResetDetector:
    """Detect a large common rigid jump of head and both wrists."""

    def __init__(self, minimum_position_m=0.03, minimum_rotation_deg=8.0, agreement_position_m=0.015, agreement_rotation_deg=5.0):
        self.minimum_position_m = float(minimum_position_m)
        self.minimum_rotation_rad = np.deg2rad(minimum_rotation_deg)
        self.agreement_position_m = float(agreement_position_m)
        self.agreement_rotation_rad = np.deg2rad(agreement_rotation_deg)
        self.previous = None

    def update(self, snapshot: TrackingSnapshot) -> bool:
        if snapshot.left is None or snapshot.right is None or snapshot.head is None:
            return False
        current = {
            "left": snapshot.left.wrist_world,
            "right": snapshot.right.wrist_world,
            "head": snapshot.head.pose_world,
        }
        if self.previous is None:
            self.previous = {key: value.copy() for key, value in current.items()}
            return False
        deltas = {key: current[key] @ np.linalg.inv(self.previous[key]) for key in current}
        self.previous = {key: value.copy() for key, value in current.items()}
        from .geometry import pose_distance
        head_position, head_rotation = pose_distance(deltas["head"], np.eye(4))
        if head_position < self.minimum_position_m and head_rotation < self.minimum_rotation_rad:
            return False
        for side in ("left", "right"):
            position, rotation = pose_distance(deltas[side], deltas["head"])
            if position > self.agreement_position_m or rotation > self.agreement_rotation_rad:
                return False
        return True
