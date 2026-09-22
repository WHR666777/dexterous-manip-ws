"""Noitom Perception Neuron Studio (PNS-G) dual hand glove input device.

Receives hand tracking data from Axis Studio (Windows) via the MocapApi SDK
over UDP.  Data pipeline:
    PNS-G gloves → Axis Studio (Windows) → UDP broadcast → Linux (MocapApi)

Network defaults match the configuration in mocap_main_base.py:
    local_ip:local_port  = 192.168.5.25:7012  (Linux, this machine)
    server_ip:server_port = 192.168.5.33:9000  (Windows running Axis Studio)

Usage:
    python teleop_real.py --input noitom
    python teleop_real.py --input noitom --noitom-local-ip 192.168.5.25 --noitom-local-port 7012
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Path to the mocapapi_python package bundled in third_party
_MOCAP_API_DIR = Path(__file__).parent / "third_party" / "mocapapi_python"


def _ensure_mocap_api_on_path() -> None:
    path = str(_MOCAP_API_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Quaternion utilities (Noitom uses (w, x, y, z) convention)
# ---------------------------------------------------------------------------

def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    qxyz = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qxyz, v)
    return v + w * t + np.cross(qxyz, t)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q1 * q2, both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Noitom → MediaPipe joint mapping
# ---------------------------------------------------------------------------

# Wrist joint name for each side
_WRIST_NAME: Dict[str, str] = {
    "right": "RightHand",
    "left":  "LeftHand",
}

# FK traversal chains (full chains including InHand* metacarpal joints).
# InHand* joints are needed for correct FK position computation of Hand*1,
# but are NOT assigned to any MediaPipe landmark slot.
_FK_CHAINS: Dict[str, Dict[str, list]] = {
    "right": {
        "thumb":  ["RightHandThumb1",   "RightHandThumb2",   "RightHandThumb3"],
        "index":  ["RightInHandIndex",  "RightHandIndex1",   "RightHandIndex2",  "RightHandIndex3"],
        "middle": ["RightInHandMiddle", "RightHandMiddle1",  "RightHandMiddle2", "RightHandMiddle3"],
        "ring":   ["RightInHandRing",   "RightHandRing1",    "RightHandRing2",   "RightHandRing3"],
        "pinky":  ["RightInHandPinky",  "RightHandPinky1",   "RightHandPinky2",  "RightHandPinky3"],
    },
    "left": {
        "thumb":  ["LeftHandThumb1",   "LeftHandThumb2",   "LeftHandThumb3"],
        "index":  ["LeftInHandIndex",  "LeftHandIndex1",   "LeftHandIndex2",  "LeftHandIndex3"],
        "middle": ["LeftInHandMiddle", "LeftHandMiddle1",  "LeftHandMiddle2", "LeftHandMiddle3"],
        "ring":   ["LeftInHandRing",   "LeftHandRing1",    "LeftHandRing2",   "LeftHandRing3"],
        "pinky":  ["LeftInHandPinky",  "LeftHandPinky1",   "LeftHandPinky2",  "LeftHandPinky3"],
    },
}

# MediaPipe assignment chains: only the 3 joints that map to MP landmarks.
# All 5 fingers use Hand*1 (MCP), Hand*2 (PIP), Hand*3 (DIP), plus an
# extrapolated fingertip → MP[4/8/12/16/20].
#   thumb  → MP[1,2,3] + tip → MP[4]
#   index  → MP[5,6,7] + tip → MP[8]
#   middle → MP[9,10,11] + tip → MP[12]
#   ring   → MP[13,14,15] + tip → MP[16]
#   pinky  → MP[17,18,19] + tip → MP[20]
_MP_ASSIGN_CHAINS: Dict[str, Dict[str, list]] = {
    "right": {
        "thumb":  ["RightHandThumb1",  "RightHandThumb2",  "RightHandThumb3"],
        "index":  ["RightHandIndex1",  "RightHandIndex2",  "RightHandIndex3"],
        "middle": ["RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3"],
        "ring":   ["RightHandRing1",   "RightHandRing2",   "RightHandRing3"],
        "pinky":  ["RightHandPinky1",  "RightHandPinky2",  "RightHandPinky3"],
    },
    "left": {
        "thumb":  ["LeftHandThumb1",  "LeftHandThumb2",  "LeftHandThumb3"],
        "index":  ["LeftHandIndex1",  "LeftHandIndex2",  "LeftHandIndex3"],
        "middle": ["LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3"],
        "ring":   ["LeftHandRing1",   "LeftHandRing2",   "LeftHandRing3"],
        "pinky":  ["LeftHandPinky1",  "LeftHandPinky2",  "LeftHandPinky3"],
    },
}

_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Pre-built parent maps from the full FK chains: joint_name → parent_name
def _build_parent_map(side: str) -> Dict[str, str]:
    pm: Dict[str, str] = {}
    wrist = _WRIST_NAME[side]
    for chain in _FK_CHAINS[side].values():
        pm[chain[0]] = wrist
        for i in range(1, len(chain)):
            pm[chain[i]] = chain[i - 1]
    return pm


_PARENT_MAP: Dict[str, Dict[str, str]] = {
    "right": _build_parent_map("right"),
    "left":  _build_parent_map("left"),
}


def _avatar_to_hand_landmarks(avatar, side: str) -> np.ndarray:
    """Compute MediaPipe-style (21, 3) landmarks for one hand via FK.

    Two-pass approach:
      Pass 1 — FK traversal over _FK_CHAINS (includes InHand* metacarpal joints)
               to compute correct world positions for all joints.
      Pass 2 — MediaPipe assignment from _MP_ASSIGN_CHAINS:
               Hand*1 (true MCP knuckle) → MP[5/9/13/17],
               Hand*2 (PIP)              → MP[6/10/14/18],
               Hand*3 (DIP)              → MP[7/11/15/19],
               extrapolated tip          → MP[8/12/16/20].
               Thumb follows the same 3-joint + extrapolated-tip pattern.

    InHand* (metacarpal) joints are used in FK only; they are NOT assigned to
    any MediaPipe slot.  This ensures MP[5] is the true index-finger MCP knuckle,
    which is required by apply_mediapipe_transformations() for frame estimation.

    Args:
        avatar: MCPAvatar object from an AvatarUpdated event.
        side:   "right" or "left".

    Returns:
        np.ndarray (21, 3), float32, in metres, wrist at origin.
        Returns zeros if the hand joints cannot be read.
    """
    wrist_name = _WRIST_NAME[side]
    parent_map = _PARENT_MAP[side]
    fk_chains = _FK_CHAINS[side]
    mp_chains = _MP_ASSIGN_CHAINS[side]

    # Collect all joint objects (full FK chains) up-front
    all_names = [j for chain in fk_chains.values() for j in chain]
    joints: Dict[str, object] = {}
    for name in all_names:
        try:
            joints[name] = avatar.get_joint_by_name(name)
        except Exception:
            pass

    if not joints:
        return np.zeros((21, 3), dtype=np.float32)

    # Pass 1: FK accumulation over full chains (wrist = origin, identity rotation)
    _id_rot = np.array([1.0, 0.0, 0.0, 0.0])
    world_pos: Dict[str, np.ndarray] = {wrist_name: np.zeros(3)}
    world_rot: Dict[str, np.ndarray] = {wrist_name: _id_rot.copy()}

    for jname in all_names:
        if jname not in joints:
            continue
        jnt = joints[jname]
        pname = parent_map[jname]

        lp_raw = jnt.get_local_position()
        if lp_raw is None:
            world_pos[jname] = world_pos.get(pname, np.zeros(3))
            world_rot[jname] = world_rot.get(pname, _id_rot.copy())
            continue

        try:
            lr_raw = jnt.get_local_rotation()   # (w, x, y, z)
        except Exception:
            lr_raw = (1.0, 0.0, 0.0, 0.0)

        lp = np.array(lp_raw, dtype=float)
        lr = np.array(lr_raw, dtype=float)

        p_pos = world_pos.get(pname, np.zeros(3))
        p_rot = world_rot.get(pname, _id_rot.copy())

        world_pos[jname] = p_pos + _quat_rotate(p_rot, lp)
        world_rot[jname] = _quat_mul(p_rot, lr)

    # Pass 2: assign MediaPipe slots from _MP_ASSIGN_CHAINS
    # MP[0] = wrist (already zero).  Each finger: 3 joints + extrapolated tip.
    landmarks = np.zeros((21, 3), dtype=np.float32)
    mp_idx = 1

    for finger in _FINGER_ORDER:
        chain = mp_chains[finger]   # always 3 joints

        for jname in chain:
            if mp_idx >= 21:
                break
            if jname in world_pos:
                landmarks[mp_idx] = (world_pos[jname] * 0.01).astype(np.float32)  # cm→m
            mp_idx += 1

        # Extrapolate fingertip using DIP's world rotation so that
        # independent DIP flexion is reflected in the tip position.
        # Correct formula:
        #   1. Convert PIP→DIP (world frame) back to PIP's local frame via inv(world_rot[PIP])
        #   2. Rotate the local bone direction by world_rot[DIP] to get the actual tip direction
        if mp_idx < 21:
            j1, j2 = chain[-2], chain[-1]  # j1=PIP, j2=DIP
            if j1 in world_pos and j2 in world_pos and j1 in world_rot and j2 in world_rot:
                p1 = world_pos[j1]
                p2 = world_pos[j2]
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                if seg_len > 1e-6:
                    # inv(q) for unit quaternion = (w, -x, -y, -z)
                    wr_pip = world_rot[j1]
                    wr_pip_inv = np.array([wr_pip[0], -wr_pip[1], -wr_pip[2], -wr_pip[3]])
                    # bone direction in PIP's local frame
                    e_local = _quat_rotate(wr_pip_inv, seg / seg_len)
                    # tip direction in world frame using DIP's accumulated rotation
                    dip_fwd = _quat_rotate(world_rot[j2], e_local)
                    landmarks[mp_idx] = ((p2 + dip_fwd * seg_len) * 0.01).astype(np.float32)
                else:
                    landmarks[mp_idx] = (p2 * 0.01).astype(np.float32)
            mp_idx += 1

    return landmarks


# ---------------------------------------------------------------------------
# NoitomInput class
# ---------------------------------------------------------------------------

class NoitomInput:
    """Noitom PNS-G dual hand glove input via MocapApi SDK.

    Starts a daemon thread that polls AvatarUpdated events from Axis Studio,
    runs forward kinematics for both hands, and stores the latest (21, 3)
    MediaPipe-style landmarks.

    Args:
        local_ip:    IP address of this Linux machine.  Must match the
                     "destination address" configured in Axis Studio BVH
                     broadcast settings.
        local_port:  UDP port this machine listens on.
        server_ip:   IP of the Windows machine running Axis Studio.
        server_port: UDP port Axis Studio broadcasts from.
    """

    def __init__(
        self,
        local_ip: str = "192.168.5.25",
        local_port: int = 8000,
        server_ip: str = "192.168.5.33",
        server_port: int = 9000,
    ) -> None:
        _ensure_mocap_api_on_path()
        from mocap_api import (
            MCPApplication,
            MCPAvatar,
            MCPBvhData,
            MCPBvhDisplacement,
            MCPBvhRotation,
            MCPEventType,
            MCPSettings,
        )

        self._MCPAvatar = MCPAvatar
        self._MCPEventType = MCPEventType

        # Initialise and open the SDK connection
        self._app = MCPApplication()
        settings = MCPSettings()
        settings.set_bvh_data(MCPBvhData.Binary)
        settings.set_bvh_transformation(MCPBvhDisplacement.Enable)
        settings.set_bvh_rotation(MCPBvhRotation.YXZ)
        settings.SetSettingsUDPEx(local_ip, local_port)
        settings.SetSettingsUDPServer(server_ip, server_port)
        self._app.set_settings(settings)
        opened, err_msg = self._app.open()
        if not opened:
            raise RuntimeError(f"MocapApi open() failed: {err_msg}")

        # Shared state (protected by lock)
        self._lock = threading.Lock()
        self._left  = np.zeros((21, 3), dtype=np.float32)
        self._right = np.zeros((21, 3), dtype=np.float32)
        self._last_update: float = 0.0

        # Daemon polling thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        logger.info(
            "NoitomInput started — local %s:%d  server %s:%d",
            local_ip, local_port, server_ip, server_port,
        )
        print(
            f"[NoitomInput] Listening on {local_ip}:{local_port} "
            f"(Axis Studio at {server_ip}:{server_port})"
        )

    # ── public interface ────────────────────────────────────────────────────

    def get_fingers_data(self) -> dict:
        """Return both hand landmarks in MediaPipe (21, 3) format.

        Returns:
            dict with keys:
                "left_fingers":  np.ndarray (21, 3), float32
                "right_fingers": np.ndarray (21, 3), float32
        """
        with self._lock:
            return {
                "left_fingers":  self._left.copy(),
                "right_fingers": self._right.copy(),
            }

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to exit."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── internal polling ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        MCPEventType = self._MCPEventType
        while not self._stop.is_set():
            try:
                evts = self._app.poll_next_event()
            except Exception as exc:
                logger.warning("poll_next_event error: %s", exc)
                time.sleep(0.05)
                continue

            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    try:
                        avatar = self._MCPAvatar(evt.event_data.avatar_handle)
                        left  = _avatar_to_hand_landmarks(avatar, "left")
                        right = _avatar_to_hand_landmarks(avatar, "right")
                        with self._lock:
                            self._left  = left
                            self._right = right
                            self._last_update = time.monotonic()
                    except Exception as exc:
                        logger.debug("Avatar processing error: %s", exc)

            time.sleep(0.01)  # poll at ~100 Hz
