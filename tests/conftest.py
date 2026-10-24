"""Keep controller ordering tests deterministic; spawn is tested separately."""

import pytest


@pytest.fixture
def inline_compute(monkeypatch):
    from teleop import controller

    class InlineCompute:
        def __init__(self, config, control, hand_mode="tracking"):
            self.ik = (controller.NeroIK(max_joint_delta_rad=config["arm"]["max_joint_delta_rad"])
                       if control in ("arm", "both") else None)
            self.retargeter = (controller._retargeter(config)
                              if control in ("hand", "both") and hand_mode == "tracking" else None)
            self.timings = {}

        def start(self): pass
        def close(self): pass
        def check_health(self): pass
        def begin_cycle(self): pass

    monkeypatch.setattr(controller, "_make_compute", InlineCompute)
    return InlineCompute
