"""Output driver for Wuji Hand via wujihandpy."""

import time

from .base import HandOutput


class WujiOutput(HandOutput):
    """Output driver for Wuji Hand via wujihandpy."""

    def __init__(self):
        import wujihandpy
        self.hand = wujihandpy.Hand()
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=False,
            filter=wujihandpy.filter.LowPass(cutoff_freq=5.0),
        )
        time.sleep(0.5)

    def send(self, qpos, joint_names):
        self.controller.set_joint_target_position(qpos.reshape(5, 4))

    def close(self):
        self.hand.write_joint_enabled(False)


__all__ = ["WujiOutput"]
