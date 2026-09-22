"""Output driver for Shadow Hand via TCP socket to docker_ros_bridge."""

import json
import socket
import time

from .base import HandOutput


class ShadowTCPOutput(HandOutput):
    """Output driver for Shadow Hand via TCP socket to docker_ros_bridge."""

    def __init__(self, docker_ip="localhost", port=5555):
        self.docker_ip = docker_ip
        self.port = port
        self.sock = self._connect()

    def _connect(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.docker_ip, self.port))
                print(f"Connected to Shadow Hand ROS bridge at {self.docker_ip}:{self.port}")
                return s
            except ConnectionRefusedError:
                print(f"Cannot connect to {self.docker_ip}:{self.port}, retrying in 2s...")
                time.sleep(2)

    def send(self, qpos, joint_names):
        msg = json.dumps({
            "joint_names": joint_names,
            "positions": qpos.tolist(),
        }) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("Connection lost, reconnecting...")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = self._connect()
            self.sock.sendall(msg.encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


__all__ = ["ShadowTCPOutput"]
