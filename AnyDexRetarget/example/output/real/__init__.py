"""Real-hand output drivers, one script per hand type."""

from .drivers_gaia import GaiaHand20Output
from .drivers_inspire import InspireSerialOutput
from .drivers_linker_l20 import LinkerL20Output
from .drivers_linker_l20_can import LinkerL20CANOutput
from .drivers_shadow import ShadowTCPOutput
from .drivers_wuji import WujiOutput

__all__ = [
    "WujiOutput",
    "ShadowTCPOutput",
    "InspireSerialOutput",
    "GaiaHand20Output",
    "LinkerL20Output",
    "LinkerL20CANOutput",
]
