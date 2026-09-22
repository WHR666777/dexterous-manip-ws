"""Small non-blocking terminal keyboard helper for the Linux experiment PC."""

from __future__ import annotations

import select
import sys
import termios
import tty


class Keyboard:
    def __enter__(self) -> "Keyboard":
        if not sys.stdin.isatty():
            raise RuntimeError("Teleoperation requires an interactive terminal.")
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1).lower() if ready else None

