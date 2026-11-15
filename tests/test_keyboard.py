"""Exercise the actual terminal buffering without opening robot hardware."""

import os
import pty
import sys
import termios

from teleop.keyboard import Keyboard


def test_queued_keys_remain_readable_and_terminal_is_restored(monkeypatch):
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "r") as terminal:
            monkeypatch.setattr(sys, "stdin", terminal)
            previous = termios.tcgetattr(terminal.fileno())
            with Keyboard() as keyboard:
                assert keyboard.read() is None
                os.write(master, b"bR\nq")
                assert [keyboard.read() for _ in range(4)] == ["b", "r", "\n", "q"]
                assert keyboard.read() is None
            assert termios.tcgetattr(terminal.fileno()) == previous
    finally:
        os.close(master)
