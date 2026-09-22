"""Common interface for real-hand hardware output drivers."""

from abc import ABC, abstractmethod

import numpy as np


class HandOutput(ABC):
    """Base class for drivers that send retargeted qpos to real hand hardware."""

    @abstractmethod
    def send(self, qpos: np.ndarray, joint_names: list[str]) -> None:
        """Send a retargeted joint target to the hardware."""

    @abstractmethod
    def close(self) -> None:
        """Release the hardware connection."""


__all__ = ["HandOutput"]
