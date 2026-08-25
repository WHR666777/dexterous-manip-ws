# Nero + LinkerHand L20 Research Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, tested Python research-control layer over the official Nero v1.11 and LinkerHand L20 SDKs, with stable NumPy action/observation interfaces and safe read-only-by-default examples.

**Architecture:** Keep both official SDK repositories as untouched sibling source checkouts. `NeroArm` and `LinkerHandL20` adapt only verified SDK behavior, while `RobotSystem` composes them without exposing SDK message types. Hardware-independent tests use injected fake drivers/APIs; real SDK imports are checked separately without claiming hardware validation.

**Tech Stack:** Python 3.8, NumPy, python-can, PyYAML, typing-extensions, pytest, official `pyAgxArm`, official LinkerHand Python SDK.

**Spec:** `docs/superpowers/specs/2026-08-25-nero-l20-research-wrapper-design.md`

## Global Constraints

- Nero firmware is exactly `v1.11`; create its Driver with `NeroFW.V111`, never `NeroFW.DEFAULT`, `V112`, or `V120`.
- Do not modify either official SDK, reimplement CAN, add IK, expose MIT/CPV/JS, or invent feedback values.
- Nero joint commands are shape `(7,)`, radians, finite, and checked against the official SDK config limits.
- Nero v1.11 has no trustworthy joint velocity feedback; do not expose `joint_velocity` or synthesize `dq`.
- L20 position is shape `(20,)`; raw values are integral `[0, 255]`; reserved indices `11..14` stay present.
- L20 speed/current/fault are shape `(5,)`; torque and embedded version are unsupported and must not appear as valid APIs.
- Canonical `RobotSystem.step()` hand action is shape `(20,)`, normalized `[-1, 1]`.
- Public functions and methods have Chinese NumPy-style docstrings covering type, shape, order, unit, range, return, exception, SDK mapping, and blocking behavior.
- Every Python module that uses PEP 585 built-in generic annotations starts with `from __future__ import annotations`, preserving Python 3.8 compatibility.
- Examples are read-only unless `--execute` is explicitly passed and the terminal confirmation succeeds.
- Runtime requirements stay minimal; no ROS, GUI, Web, Hydra, Docker, learning algorithm, or camera dependency.
- A passing import/unit test is reported as static verification only; hardware validation remains explicitly unperformed.

---

### Task 1: Establish the reproducible SDK and test baseline

**Files:**
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_sdk_contract.py`
- Preserve: `docs/superpowers/specs/2026-08-25-nero-l20-research-wrapper-design.md`
- Preserve: `docs/superpowers/plans/2026-08-25-nero-l20-research-wrapper-implementation.md`

**Interfaces:**
- Consumes: official GitHub repositories at the commits recorded in the spec.
- Produces: untouched `pyAgxArm/` and `linkerhand-python-sdk/` source checkouts; a parent Git repository that ignores both nested SDK repositories; executable pytest baseline.

- [ ] **Step 1: Initialize only the parent research repository metadata**

Run:

```bash
git init
```

Create `.gitignore` with:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
pyAgxArm/
linkerhand-python-sdk/
```

The SDK directories are ignored because they retain their own Git histories and must not become accidental embedded-repository entries in the research repository.

- [ ] **Step 2: Clone and pin both official SDK checkouts**

Run:

```bash
git clone https://github.com/agilexrobotics/pyAgxArm.git pyAgxArm
git -C pyAgxArm checkout 8cd90f9106219a156c3c0d7e58ee36d838a89baf
git clone https://github.com/linker-bot/linkerhand-python-sdk.git linkerhand-python-sdk
git -C linkerhand-python-sdk checkout 0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8
git -C pyAgxArm status --short
git -C linkerhand-python-sdk status --short
```

Expected: both `status --short` outputs are empty.

- [ ] **Step 3: Install test-time and verified runtime dependencies**

Run:

```bash
python3 -m pip install --user pytest numpy 'python-can>=3.3.4' PyYAML 'typing-extensions>=3.7.4.3'
python3 -m pip install --user -e ./pyAgxArm
```

Expected: installation succeeds under Python 3.8.10 without installing the LinkerHand GUI requirements.

- [ ] **Step 4: Write the SDK contract test**

Create `tests/__init__.py` as an empty file. Create `tests/test_sdk_contract.py`:

```python
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKER_SDK = ROOT / "linkerhand-python-sdk"


def test_nero_v111_public_contract_exists():
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    assert ArmModel.NERO == "nero"
    assert NeroFW.V111 == "v111"
    assert callable(create_agx_arm_config)
    assert callable(AgxArmFactory.create_arm)

    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v111.driver import Driver

    expected = {
        "connect", "disconnect", "enable", "disable", "reset",
        "electronic_emergency_stop", "get_joint_angles", "get_motor_states",
        "get_flange_pose", "get_tcp_pose", "get_firmware", "get_arm_status", "move_j",
        "move_p", "move_l", "set_speed_percent", "set_joint_limits_enabled",
    }
    assert expected <= set(dir(Driver))
    assert "timeout" not in inspect.signature(Driver.enable).parameters
    assert "get_ik_joint_angles" not in Driver.__dict__


def test_l20_public_contract_and_source_limitations_exist():
    sys.path.insert(0, str(LINKER_SDK))
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
        from LinkerHand.core.can.linker_hand_l20_can import LinkerHandL20Can
    finally:
        sys.path.remove(str(LINKER_SDK))

    expected_api = {
        "finger_move", "get_state", "get_state_for_pub", "set_speed",
        "get_speed", "set_current", "get_current", "get_temperature",
        "get_fault", "clear_faults",
    }
    assert expected_api <= set(dir(LinkerHandApi))
    assert hasattr(LinkerHandL20Can, "close_can_interface")

    torque_source = inspect.getsource(LinkerHandL20Can.get_torque)
    version_source = inspect.getsource(LinkerHandL20Can.get_version)
    assert "[0] * 5" in torque_source
    assert "[0] * 5" in version_source
```

- [ ] **Step 5: Run the contract test**

Run:

```bash
python3 -m pytest tests/test_sdk_contract.py -v
```

Expected: 2 tests pass. This verifies source/API presence only, not CAN communication.

- [ ] **Step 6: Commit the reproducible baseline**

Run:

```bash
git add .gitignore tests/__init__.py tests/test_sdk_contract.py docs/superpowers
git commit -m "chore: establish pinned SDK contract baseline"
```

---

### Task 2: Implement NeroArm with v1.11-safe behavior

**Files:**
- Create: `robot_control/__init__.py`
- Create: `robot_control/nero.py`
- Create: `tests/test_nero.py`

**Interfaces:**
- Consumes: `pyAgxArm.create_agx_arm_config`, `AgxArmFactory.create_arm`, `ArmModel.NERO`, `NeroFW.V111`, and the public V111 Driver methods listed in Task 1.
- Produces: `NeroArm` with lifecycle, status (including reported firmware), raw debugging, validation, observation, and non-blocking motion methods defined by spec sections 6.1–6.5.

- [ ] **Step 1: Write fake SDK objects and failing construction tests**

Create `tests/test_nero.py` beginning with:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from robot_control.nero import NeroArm


class Message:
    def __init__(self, msg, timestamp=123.0, hz=100.0):
        self.msg = msg
        self.timestamp = timestamp
        self.hz = hz


class FakeNeroDriver:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.enable_after = 1
        self.enable_calls = 0
        self.disable_calls = 0
        self.sent_joints = []
        self.sent_poses = []
        self.sent_linear = []
        self.speed = None
        self.limits_enabled = False
        self.joint_limits = {
            "joint1": [-2.705261, 2.705261],
            "joint2": [-1.745330, 1.745330],
            "joint3": [-2.757621, 2.757621],
            "joint4": [-1.012291, 2.146755],
            "joint5": [-2.757621, 2.757621],
            "joint6": [-0.733039, 0.959932],
            "joint7": [-1.570797, 1.570797],
        }
        self.q = [0.0] * 7
        self.tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.flange = [0.1, 0.2, 0.3, 0.0, 0.1, 0.2]
        self.status = SimpleNamespace(
            ctrl_mode=1, arm_status=0, mode_feedback=1, teach_status=0,
            motion_status=0, trajectory_num=0,
            err_status=SimpleNamespace(joint_1_angle_limit=False),
        )

    def get_config(self):
        return {"joint_limits": self.joint_limits}

    def set_joint_limits_enabled(self, enabled):
        self.limits_enabled = enabled

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def is_ok(self):
        return self.connected

    def enable(self):
        self.enable_calls += 1
        if self.enable_calls >= self.enable_after:
            self.enabled = True
        return self.enabled

    def disable(self):
        self.disable_calls += 1
        self.enabled = False
        return True

    def get_joint_enable_status(self, index):
        return self.enabled

    def get_joint_angles(self):
        return Message(list(self.q))

    def get_motor_states(self, index):
        return Message(SimpleNamespace(
            position=self.q[index - 1], velocity=0.0,
            current=0.0, torque=self.tau[index - 1],
        ))

    def get_flange_pose(self):
        return Message(list(self.flange))

    def get_tcp_pose(self):
        return Message(list(self.flange))

    def get_arm_status(self):
        return Message(self.status)

    def get_firmware(self):
        return {"software_version": "1.11"}

    def set_speed_percent(self, speed):
        self.speed = speed

    def move_j(self, joints):
        self.sent_joints.append(list(joints))

    def move_p(self, pose):
        self.sent_poses.append(list(pose))

    def move_l(self, pose):
        self.sent_linear.append(list(pose))

    def electronic_emergency_stop(self):
        self.estopped = True

    def reset(self):
        self.reset_called = True


def make_connected_arm(enabled=True, **kwargs):
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver, **kwargs)
    arm.connect()
    driver.enabled = enabled
    return arm, driver


def test_constructor_enables_official_software_joint_limits():
    driver = FakeNeroDriver()
    NeroArm(driver=driver)
    assert driver.limits_enabled is True


def test_real_factory_is_called_with_v111(monkeypatch):
    calls = {}
    driver = FakeNeroDriver()
    symbols = SimpleNamespace(
        ArmModel=SimpleNamespace(NERO="nero"),
        NeroFW=SimpleNamespace(V111="v111"),
        create_agx_arm_config=lambda **kwargs: calls.setdefault("config", kwargs),
        AgxArmFactory=SimpleNamespace(create_arm=lambda config: driver),
    )
    monkeypatch.setattr("robot_control.nero._load_nero_sdk", lambda: symbols)
    NeroArm(can_interface="socketcan", can_channel="can7")
    assert calls["config"] == {
        "robot": "nero", "firmeware_version": "v111",
        "interface": "socketcan", "channel": "can7",
    }
```

- [ ] **Step 2: Run construction tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_nero.py::test_constructor_enables_official_software_joint_limits tests/test_nero.py::test_real_factory_is_called_with_v111 -v
```

Expected: collection or import fails because `robot_control.nero.NeroArm` does not exist.

- [ ] **Step 3: Implement SDK loading, construction, connection state, and official limits**

Create `robot_control/__init__.py` exporting `NeroArm`. Create `robot_control/nero.py` with:

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np


NERO_DOF = 7


def _load_nero_sdk():
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
    return type("NeroSdk", (), {
        "AgxArmFactory": AgxArmFactory,
        "ArmModel": ArmModel,
        "NeroFW": NeroFW,
        "create_agx_arm_config": staticmethod(create_agx_arm_config),
    })


class NeroArm:
    def __init__(
        self,
        can_interface: str = "socketcan",
        can_channel: str = "can0",
        max_joint_delta: Optional[float] = None,
        driver: Optional[Any] = None,
    ) -> None:
        if driver is None:
            sdk = _load_nero_sdk()
            config = sdk.create_agx_arm_config(
                robot=sdk.ArmModel.NERO,
                firmeware_version=sdk.NeroFW.V111,
                interface=can_interface,
                channel=can_channel,
            )
            driver = sdk.AgxArmFactory.create_arm(config)
        self._driver = driver
        self._max_joint_delta = self._validate_delta_setting(max_joint_delta)
        self._driver.set_joint_limits_enabled(True)
        config = self._driver.get_config()
        limits = config.get("joint_limits", {})
        names = [f"joint{i}" for i in range(1, NERO_DOF + 1)]
        if any(name not in limits for name in names):
            raise RuntimeError("Nero SDK config does not contain all 7 joint limits.")
        self._joint_limits = np.asarray([limits[name] for name in names], dtype=np.float64)
```

Add complete Chinese docstrings while implementing, including the injected `driver` test seam.

- [ ] **Step 4: Run construction tests and verify GREEN**

Run the command from Step 2.

Expected: both tests pass.

- [ ] **Step 5: Write failing lifecycle and timeout tests**

Append tests:

```python
def test_connect_disconnect_are_idempotent():
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver)
    arm.connect()
    arm.connect()
    assert arm.is_connected()
    arm.disconnect()
    arm.disconnect()
    assert not arm.is_connected()


def test_enable_retries_until_success():
    arm, driver = make_connected_arm(enabled=False)
    driver.enable_after = 3
    arm.enable(timeout=0.2, poll_interval=0.001)
    assert driver.enable_calls == 3
    assert arm.is_enabled()


def test_enable_timeout_is_finite():
    arm, driver = make_connected_arm(enabled=False)
    driver.enable_after = 10_000
    with pytest.raises(TimeoutError, match="Nero enable timed out"):
        arm.enable(timeout=0.01, poll_interval=0.001)


def test_disable_timeout_is_finite():
    arm, driver = make_connected_arm(enabled=True)
    driver.disable = lambda: False
    with pytest.raises(TimeoutError, match="Nero disable timed out"):
        arm.disable(timeout=0.01, poll_interval=0.001)


def test_commands_require_connection_and_enable():
    driver = FakeNeroDriver()
    arm = NeroArm(driver=driver)
    with pytest.raises(RuntimeError, match="not connected"):
        arm.command_joint_positions([0.0] * 7)
    arm.connect()
    with pytest.raises(RuntimeError, match="not enabled"):
        arm.command_joint_positions([0.0] * 7)
```

- [ ] **Step 6: Run lifecycle tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_nero.py -k 'connect or enable or commands_require' -v
```

Expected: failures identify missing lifecycle methods.

- [ ] **Step 7: Implement lifecycle methods minimally**

Implement `connect`, `disconnect`, `is_connected`, `is_enabled`, `is_ok`, `_require_connected`, `_require_enabled`, `enable`, and `disable`. The retry body is:

```python
deadline = time.monotonic() + timeout
while True:
    if bool(operation()):
        return
    if time.monotonic() >= deadline:
        raise TimeoutError(message)
    time.sleep(poll_interval)
```

Validate `timeout >= 0` and `poll_interval > 0`; for `timeout == 0`, call the SDK operation exactly once before timing out.

- [ ] **Step 8: Run lifecycle tests and verify GREEN**

Run the Step 6 command. Expected: all selected tests pass.

- [ ] **Step 9: Write failing validation, state, raw, and motion tests**

Append parameterized tests:

```python
@pytest.mark.parametrize("bad", [
    [0.0] * 6,
    [[0.0] * 7],
    [0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
])
def test_joint_validation_rejects_bad_shape_or_nonfinite(bad):
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError):
        arm.validate_joint_command(bad)


def test_joint_validation_rejects_official_limit_violation():
    arm, _ = make_connected_arm()
    with pytest.raises(ValueError, match="joint 1"):
        arm.validate_joint_command([2.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_joint_validation_rejects_excessive_delta():
    arm, _ = make_connected_arm(max_joint_delta=0.05)
    with pytest.raises(ValueError, match="max_joint_delta"):
        arm.validate_joint_command([0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_state_arrays_have_verified_shapes_units_and_copies(monkeypatch):
    arm, driver = make_connected_arm()
    assert arm.get_joint_positions().shape == (7,)
    assert arm.get_joint_torques().shape == (7,)
    assert arm.get_flange_pose().shape == (6,)
    assert arm.get_tcp_pose().shape == (6,)
    monkeypatch.setattr("robot_control.nero.time.time", lambda: 456.0)
    observation = arm.get_observation()
    assert set(observation) == {"joint_position", "joint_torque", "tcp_pose", "timestamp"}
    assert observation["timestamp"] == 456.0
    assert arm.get_state_vector().shape == (7,)
    assert "joint_velocity" not in observation
    q = arm.get_joint_positions()
    q[0] = 99.0
    assert driver.q[0] == 0.0


def test_reported_firmware_is_plain_data():
    arm, _ = make_connected_arm()
    firmware = arm.get_firmware()
    assert firmware == {"software_version": "1.11"}
    assert isinstance(firmware, dict)


def test_raw_methods_return_sdk_messages():
    arm, _ = make_connected_arm()
    assert isinstance(arm.get_raw_joint_positions(), Message)
    assert len(arm.get_raw_motor_states()) == 7
    assert isinstance(arm.get_raw_flange_pose(), Message)
    assert isinstance(arm.get_raw_arm_status(), Message)


def test_motion_methods_map_to_verified_sdk_calls():
    arm, driver = make_connected_arm()
    arm.command_joint_positions([0.01] * 7)
    arm.move_joints([0.02] * 7, speed_percent=10)
    arm.move_pose([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], speed_percent=9)
    arm.move_linear([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], speed_percent=8)
    assert driver.sent_joints == [[0.01] * 7, [0.02] * 7]
    assert driver.sent_poses == [[0.1, 0.2, 0.3, 0.0, 0.1, 0.2]]
    assert driver.sent_linear == [[0.1, 0.2, 0.3, 0.0, 0.1, 0.2]]
    assert driver.speed == 8


def test_status_and_safety_methods_map_without_message_leakage():
    arm, driver = make_connected_arm()
    status = arm.get_arm_status()
    assert status["arm_status"] == 0
    assert status["err_status"]["joint_1_angle_limit"] is False
    arm.emergency_stop()
    arm.reset()
    assert driver.estopped is True
    assert driver.reset_called is True
```

- [ ] **Step 10: Run new behavior tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_nero.py -v
```

Expected: failures identify missing validation, state conversion, raw methods, movement, and safety mappings.

- [ ] **Step 11: Implement the remaining NeroArm behavior**

Use these exact conversion rules:

```python
array = np.asarray(values, dtype=np.float64)
if array.shape != expected_shape:
    raise ValueError(f"Expected shape {expected_shape}, got {array.shape}.")
if not np.all(np.isfinite(array)):
    raise ValueError("Values must all be finite.")
return array.copy()
```

For `get_joint_torques()`, call `get_motor_states(index)` for indices `1..7` and extract `.msg.torque`. For pose validation require shape `(6,)`, finite values, `roll/yaw` in `[-pi, pi]`, and `pitch` in `[-pi/2, pi/2]` before calling the SDK. Validate speed as an actual integer in `[0, 100]`, rejecting bool and fractional values.

For `get_firmware()` and status serialization, recurse over dict/list/tuple and public `vars(value)` keys that do not start with `_`; pass through scalar `str/int/float/bool/None` values. Return fresh ordinary-Python structures so neither method leaks mutable SDK objects. If the SDK returns `None` for firmware because feedback is unavailable, raise `RuntimeError` instead of manufacturing a version.

- [ ] **Step 12: Run Nero tests and the full baseline**

Run:

```bash
python3 -m pytest tests/test_nero.py tests/test_sdk_contract.py -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 13: Commit NeroArm**

Run:

```bash
git add robot_control/__init__.py robot_control/nero.py tests/test_nero.py
git commit -m "feat: add Nero v1.11 research wrapper"
```

---

### Task 3: Implement the LinkerHandL20 adapter

**Files:**
- Create: `robot_control/l20.py`
- Modify: `robot_control/__init__.py`
- Create: `tests/test_l20.py`

**Interfaces:**
- Consumes: official `LinkerHandApi` methods `finger_move`, `get_state`, `get_state_for_pub`, speed/current/fault/temperature methods, and current L20 teardown fields.
- Produces: `LinkerHandL20`, `L20_JOINT_NAMES`, `L20_ACTIVE_POSITION_INDICES`, explicit raw/normalized position APIs, supported five-motor data, SDK version, presets, and idempotent disconnect.

- [ ] **Step 1: Write the failing L20 constant and lifecycle tests**

Create `tests/test_l20.py`:

```python
import numpy as np
import pytest

from robot_control.l20 import (
    L20_ACTIVE_POSITION_INDICES,
    L20_JOINT_NAMES,
    LinkerHandL20,
)


class FakeReceiveThread:
    def __init__(self, alive=False):
        self.join_timeout = None
        self.alive = alive

    def join(self, timeout=None):
        self.join_timeout = timeout

    def is_alive(self):
        return self.alive


class FakeLowLevelHand:
    def __init__(self):
        self.running = True
        self.closed = False
        self.receive_thread = FakeReceiveThread()

    def close_can_interface(self):
        self.closed = True


class FakeLinkerApi:
    def __init__(self):
        self.hand = FakeLowLevelHand()
        self.version = "3.1.1"
        self.positions = list(range(20))
        self.commands = []
        self.speed = [10, 20, 30, 40, 50]
        self.current = [50, 40, 30, 20, 10]
        self.fault = [0, 1, 2, 3, 4]
        self.temperature = list(range(20, 40))

    def finger_move(self, pose):
        self.commands.append(list(pose))

    def get_state(self):
        return list(self.positions)

    def get_state_for_pub(self):
        return list(self.positions)

    def set_speed(self, speed):
        self.speed = list(speed)

    def get_speed(self):
        return list(self.speed)

    def set_current(self, current):
        self.current = list(current)

    def get_current(self):
        return list(self.current)

    def get_fault(self):
        return list(self.fault)

    def clear_faults(self):
        self.fault = [0] * 5

    def get_temperature(self):
        return list(self.temperature)


def test_l20_joint_order_and_reserved_indices_are_fixed():
    assert len(L20_JOINT_NAMES) == 20
    assert L20_JOINT_NAMES[10] == "thumb_roll"
    assert L20_JOINT_NAMES[11:15] == (
        "reserved_11", "reserved_12", "reserved_13", "reserved_14"
    )
    assert L20_ACTIVE_POSITION_INDICES == tuple(range(11)) + tuple(range(15, 20))


def test_connect_is_lazy_and_disconnect_stops_only_the_sdk_bus():
    created = []
    api = FakeLinkerApi()
    hand = LinkerHandL20(api_factory=lambda **kwargs: created.append(kwargs) or api)
    assert not hand.is_connected()
    hand.connect()
    assert created == [{
        "hand_type": "right", "hand_joint": "L20",
        "modbus": "None", "can": "can1",
    }]
    hand.disconnect()
    assert api.hand.running is False
    assert api.hand.closed is True
    assert not hand.is_connected()


def test_disconnect_thread_timeout_is_bounded_and_retryable():
    api = FakeLinkerApi()
    api.hand.receive_thread.alive = True
    hand = LinkerHandL20(api_factory=lambda **kwargs: api, join_timeout=0.01)
    hand.connect()
    with pytest.raises(TimeoutError, match="receive thread"):
        hand.disconnect()
    assert api.hand.receive_thread.join_timeout == 0.01
    api.hand.receive_thread.alive = False
    hand.disconnect()
    assert not hand.is_connected()
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_l20.py -k 'joint_order or connect' -v
```

Expected: import fails because `robot_control.l20` does not exist.

- [ ] **Step 3: Implement constants, SDK path loading, and lifecycle**

Create `robot_control/l20.py` with these immutable constants:

```python
L20_JOINT_NAMES = (
    "thumb_base", "index_base", "middle_base", "ring_base", "little_base",
    "thumb_abduction", "index_abduction", "middle_abduction",
    "ring_abduction", "little_abduction", "thumb_roll",
    "reserved_11", "reserved_12", "reserved_13", "reserved_14",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
)
L20_ACTIVE_POSITION_INDICES = tuple(range(11)) + tuple(range(15, 20))
```

Implement `_load_linker_api()` using `importlib.import_module`. On the first `ModuleNotFoundError`, insert `<project_root>/linkerhand-python-sdk` into `sys.path` only if the directory and `LinkerHand/linker_hand_api.py` both exist, then retry. Raise an actionable `ImportError` without connecting hardware.

`LinkerHandL20.__init__` stores configuration and factory only. `connect()` creates the official API exactly once. `disconnect()` sets low-level `running=False`, calls `close_can_interface()`, and joins with `join_timeout`. It raises `TimeoutError` if the thread is still alive and retains the low-level reference so a later `disconnect()` can retry the bounded join; once the thread is confirmed stopped, it clears the API reference. Even if bus shutdown raises, it still attempts the join before reporting the collected teardown error.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

Run the Step 2 command. Expected: selected tests pass.

- [ ] **Step 5: Write failing raw/normalized position tests**

Append:

```python
def connected_hand():
    api = FakeLinkerApi()
    hand = LinkerHandL20(api_factory=lambda **kwargs: api)
    hand.connect()
    return hand, api


@pytest.mark.parametrize("bad", [
    [0] * 19,
    [0] * 21,
    [0] * 19 + [256],
    [0] * 19 + [-1],
    [0] * 19 + [1.5],
    [0] * 19 + [np.nan],
])
def test_raw_position_rejects_bad_shape_range_or_fraction(bad):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        hand.set_joint_positions_raw(bad)


@pytest.mark.parametrize("bad", [
    [0.0] * 19,
    [0.0] * 19 + [1.01],
    [0.0] * 19 + [np.inf],
])
def test_normalized_position_rejects_invalid_values(bad):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        hand.set_joint_positions_normalized(bad)


def test_normalized_position_maps_endpoints_and_midpoint_half_up():
    hand, api = connected_hand()
    action = [-1.0, 0.0, 1.0] + [0.0] * 17
    hand.set_joint_positions_normalized(action)
    assert api.commands[-1] == [0, 128, 255] + [128] * 17


def test_fresh_and_cached_positions_are_explicit_copies():
    hand, api = connected_hand()
    fresh = hand.get_joint_positions_raw(fresh=True)
    cached = hand.get_cached_joint_positions_raw()
    assert fresh.shape == (20,)
    assert cached.shape == (20,)
    fresh[0] = 999
    assert api.positions[0] == 0
    assert hand.get_active_joint_indices() == L20_ACTIVE_POSITION_INDICES


def test_invalid_sdk_position_feedback_is_not_padded_or_fabricated():
    hand, api = connected_hand()
    api.positions = []
    with pytest.raises(RuntimeError, match="position feedback"):
        hand.get_joint_positions_raw(fresh=True)
```

- [ ] **Step 6: Run position tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_l20.py -k 'position or cached' -v
```

Expected: failures identify missing position APIs.

- [ ] **Step 7: Implement exact position validation and conversion**

Raw validation uses float64 for finite/integral checking, then converts to integer:

```python
values = np.asarray(positions, dtype=np.float64)
if values.shape != (20,):
    raise ValueError(f"L20 position must have shape (20,), got {values.shape}.")
if not np.all(np.isfinite(values)):
    raise ValueError("L20 position values must all be finite.")
if not np.all(values == np.floor(values)):
    raise ValueError("L20 raw position values must be integral.")
if np.any(values < 0) or np.any(values > 255):
    raise ValueError("L20 raw position values must be in [0, 255].")
return values.astype(np.uint8).astype(np.int64)
```

Normalized mapping is exactly:

```python
raw = np.floor((values + 1.0) * 127.5 + 0.5).astype(np.int64)
```

Send the validated list through `finger_move(pose=raw.tolist())`.

- [ ] **Step 8: Run position tests and verify GREEN**

Run the Step 6 command. Expected: all selected tests pass.

- [ ] **Step 9: Write failing supported-state and preset tests**

Append:

```python
def test_supported_five_motor_data_and_temperature_shapes():
    hand, api = connected_hand()
    hand.set_speed([1, 2, 3, 4, 5])
    hand.set_current([5, 4, 3, 2, 1])
    assert hand.get_speed().tolist() == [1, 2, 3, 4, 5]
    assert hand.get_current().tolist() == [5, 4, 3, 2, 1]
    assert hand.get_fault().shape == (5,)
    assert hand.get_temperature().shape == (20,)
    hand.clear_faults()
    assert api.fault == [0] * 5
    assert hand.get_sdk_version() == "3.1.1"


@pytest.mark.parametrize("method,value", [
    ("set_speed", [1, 2, 3, 4]),
    ("set_current", [1, 2, 3, 4, 256]),
])
def test_five_motor_setters_validate_exact_shape_and_range(method, value):
    hand, _ = connected_hand()
    with pytest.raises(ValueError):
        getattr(hand, method)(value)


def test_official_open_and_close_presets_are_used_exactly():
    hand, api = connected_hand()
    hand.open_hand()
    assert api.commands[-1] == [
        255, 255, 255, 255, 255, 255, 10, 100, 180, 240,
        245, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ]
    hand.close_hand()
    assert api.commands[-1] == [
        40, 0, 0, 0, 0, 131, 10, 100, 180, 240,
        19, 255, 255, 255, 255, 135, 0, 0, 0, 0,
    ]


def test_fake_torque_and_embedded_version_are_not_public_wrapper_methods():
    assert not hasattr(LinkerHandL20, "get_torque")
    assert not hasattr(LinkerHandL20, "set_torque")
    assert not hasattr(LinkerHandL20, "get_version")
```

- [ ] **Step 10: Run supported-state tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_l20.py -v
```

Expected: missing supported-state and preset methods fail.

- [ ] **Step 11: Implement supported state, setters, and official presets**

Use one private `_validate_five_raw(values, name)` for speed/current. Validate getter outputs independently; empty or wrong-sized SDK feedback raises `RuntimeError` rather than padding. `get_temperature()` requires exactly 20 finite values but does not assign a physical unit.

Define the two preset tuples directly above the class with comments naming official file `example/gui_control/config/constants.py`, L20 action names, and commit `0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8`.

- [ ] **Step 12: Run all L20 and SDK tests**

Run:

```bash
python3 -m pytest tests/test_l20.py tests/test_sdk_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 13: Export and commit LinkerHandL20**

Add `LinkerHandL20`, `L20_JOINT_NAMES`, and `L20_ACTIVE_POSITION_INDICES` to `robot_control/__init__.py`.

Run:

```bash
git add robot_control/l20.py robot_control/__init__.py tests/test_l20.py
git commit -m "feat: add LinkerHand L20 research wrapper"
```

---

### Task 4: Compose the two devices in RobotSystem

**Files:**
- Create: `robot_control/robot_system.py`
- Modify: `robot_control/__init__.py`
- Create: `tests/test_robot_system.py`

**Interfaces:**
- Consumes: `NeroArm` and `LinkerHandL20` public methods from Tasks 2 and 3.
- Produces: `RobotSystem.connect/enable/disable/disconnect/get_observation/step/self_check`, canonical two-device action and nested observation.

- [ ] **Step 1: Write failing composition and rollback tests**

Create `tests/test_robot_system.py`:

```python
import numpy as np
import pytest

from robot_control.robot_system import RobotSystem


class FakeArm:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.commands = []
        self.disconnected = 0
        self.fail_disconnect = False

    def connect(self): self.connected = True
    def disconnect(self):
        self.connected = False
        self.disconnected += 1
        if self.fail_disconnect:
            raise RuntimeError("arm disconnect failed")
    def enable(self, timeout=5.0): self.enabled = True
    def disable(self, timeout=5.0): self.enabled = False
    def is_connected(self): return self.connected
    def is_enabled(self): return self.enabled
    def is_ok(self): return self.connected
    def validate_joint_command(self, value):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError("bad arm action")
        return array
    def command_joint_positions(self, value): self.commands.append(list(value))
    def get_observation(self):
        return {
            "joint_position": np.zeros(7), "joint_torque": np.ones(7),
            "tcp_pose": np.zeros(6), "timestamp": 1.0,
        }
    def get_firmware(self): return {"software_version": "1.11"}
    def get_arm_status(self): return {"arm_status": 0}


class FakeHand:
    def __init__(self, fail_connect=False):
        self.connected = False
        self.fail_connect = fail_connect
        self.commands = []
        self.fault = np.zeros(5, dtype=np.int64)
        self.disconnected = 0
        self.fail_disconnect = False
    def connect(self):
        if self.fail_connect: raise RuntimeError("hand failed")
        self.connected = True
    def disconnect(self):
        self.connected = False
        self.disconnected += 1
        if self.fail_disconnect:
            raise RuntimeError("hand disconnect failed")
    def is_connected(self): return self.connected
    def get_joint_positions_raw(self, fresh=True): return np.arange(20)
    def get_fault(self): return self.fault.copy()
    def set_joint_positions_normalized(self, value): self.commands.append(list(value))


def test_connect_rolls_back_arm_when_hand_fails():
    arm = FakeArm()
    robot = RobotSystem(arm=arm, hand=FakeHand(fail_connect=True))
    with pytest.raises(RuntimeError, match="L20"):
        robot.connect()
    assert arm.disconnected == 1


def test_enable_and_disable_apply_only_to_arm():
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.enable(timeout=0.2)
    assert arm.enabled
    robot.disable(timeout=0.2)
    assert not arm.enabled


def test_disconnect_is_idempotent_for_both_devices():
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.disconnect()
    robot.disconnect()
    assert not arm.connected
    assert not hand.connected


def test_disconnect_attempts_both_and_aggregates_failures():
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    arm.fail_disconnect = True
    hand.fail_disconnect = True
    with pytest.raises(RuntimeError) as caught:
        robot.disconnect()
    assert "hand disconnect failed" in str(caught.value)
    assert "arm disconnect failed" in str(caught.value)
    assert arm.disconnected == 1
    assert hand.disconnected == 1
```

- [ ] **Step 2: Run composition tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_robot_system.py -k 'connect or enable' -v
```

Expected: import fails because `RobotSystem` does not exist.

- [ ] **Step 3: Implement construction and lifecycle**

`RobotSystem.__init__(arm=None, hand=None)` creates default wrappers only when arguments are absent. Implement connect rollback exactly as:

```python
self.arm.connect()
try:
    self.hand.connect()
except Exception as exc:
    self.arm.disconnect()
    raise RuntimeError(f"Failed to connect L20 after Nero connected: {exc}") from exc
```

`disconnect()` always attempts hand then arm, preserves both exception messages if both fail, and remains idempotent.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

Run the Step 2 command. Expected: selected tests pass.

- [ ] **Step 5: Write failing observation, step, partial-send, and self-check tests**

Append:

```python
def connected_enabled_robot():
    arm, hand = FakeArm(), FakeHand()
    robot = RobotSystem(arm=arm, hand=hand)
    robot.connect()
    robot.enable()
    return robot, arm, hand


def test_observation_has_fixed_nested_shapes(monkeypatch):
    robot, _, _ = connected_enabled_robot()
    monkeypatch.setattr("robot_control.robot_system.time.time", lambda: 789.0)
    observation = robot.get_observation()
    assert set(observation) == {"arm", "hand", "timestamp"}
    assert observation["arm"]["joint_position"].shape == (7,)
    assert observation["arm"]["joint_torque"].shape == (7,)
    assert observation["arm"]["tcp_pose"].shape == (6,)
    assert "timestamp" not in observation["arm"]
    assert observation["hand"]["joint_position_raw"].shape == (20,)
    assert observation["timestamp"] == 789.0


def test_step_validates_both_actions_before_sending():
    robot, arm, hand = connected_enabled_robot()
    with pytest.raises(ValueError, match="hand_joint_position"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(19),
        })
    assert arm.commands == []
    assert hand.commands == []


def test_step_rejects_missing_or_unknown_keys():
    robot, _, _ = connected_enabled_robot()
    with pytest.raises(ValueError, match="keys"):
        robot.step({"arm_joint_position": np.zeros(7)})
    with pytest.raises(ValueError, match="keys"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
            "typo": 1,
        })


def test_step_sends_once_to_arm_then_hand():
    robot, arm, hand = connected_enabled_robot()
    robot.step({
        "arm_joint_position": np.full(7, 0.01),
        "hand_joint_position": np.zeros(20),
    })
    assert arm.commands == [[0.01] * 7]
    assert hand.commands == [[0.0] * 20]


def test_hand_failure_reports_possible_partial_arm_send():
    robot, arm, hand = connected_enabled_robot()
    def fail(_): raise RuntimeError("CAN send failed")
    hand.set_joint_positions_normalized = fail
    with pytest.raises(RuntimeError, match="may already have been sent"):
        robot.step({
            "arm_joint_position": np.zeros(7),
            "hand_joint_position": np.zeros(20),
        })
    assert len(arm.commands) == 1


def test_self_check_is_read_only_structured_and_prints_summary(capsys):
    robot, arm, hand = connected_enabled_robot()
    result = robot.self_check()
    assert result["nero"]["firmware_config"] == "V111"
    assert result["nero"]["firmware_reported"] == "1.11"
    assert result["l20"]["fault"] == [0, 0, 0, 0, 0]
    assert "[OK]" in capsys.readouterr().out
    assert arm.commands == []
    assert hand.commands == []
```

- [ ] **Step 6: Run behavior tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_robot_system.py -v
```

Expected: observation, step, and self-check tests fail for missing methods.

- [ ] **Step 7: Implement observation, full prevalidation, step, and self-check**

Before either command is sent, validate:

```python
expected_keys = {"arm_joint_position", "hand_joint_position"}
if set(action) != expected_keys:
    raise ValueError(
        f"Action keys must be exactly {sorted(expected_keys)}, got {sorted(action)}."
    )
arm_value = self.arm.validate_joint_command(action["arm_joint_position"])
hand_value = np.asarray(action["hand_joint_position"], dtype=np.float64)
if hand_value.shape != (20,) or not np.all(np.isfinite(hand_value)):
    raise ValueError("hand_joint_position must be a finite array with shape (20,).")
if np.any(hand_value < -1.0) or np.any(hand_value > 1.0):
    raise ValueError("hand_joint_position values must be in [-1, 1].")
```

Then send arm followed by hand. Wrap only a hand-send failure with the partial-send warning; preserve an arm-send failure directly because no hand command was attempted.

`get_observation()` removes the arm adapter's inner timestamp and creates one top-level Unix timestamp after acquiring both devices, so the canonical nested schema contains exactly the documented fields. `self_check()` returns ordinary Python values only and prints one `[OK]` or `[FAIL]` line per checked device/category. It may request state/fault frames but must not call any movement or preset method.

- [ ] **Step 8: Run RobotSystem and full unit tests**

Run:

```bash
python3 -m pytest tests/test_robot_system.py tests/test_nero.py tests/test_l20.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Export and commit RobotSystem**

Export `RobotSystem` from `robot_control/__init__.py`.

Run:

```bash
git add robot_control/robot_system.py robot_control/__init__.py tests/test_robot_system.py
git commit -m "feat: compose Nero and L20 control system"
```

---

### Task 5: Add read-only-by-default examples

**Files:**
- Create: `config.py`
- Create: `examples/__init__.py`
- Create: `examples/nero_example.py`
- Create: `examples/l20_example.py`
- Create: `examples/nero_l20_example.py`
- Create: `tests/test_examples.py`

**Interfaces:**
- Consumes: `NeroArm`, `LinkerHandL20`, `RobotSystem`, and the configuration contract in approved spec section 5.
- Produces: three CLI examples whose default paths send no motion and whose execute paths require both flag and confirmation.

- [ ] **Step 1: Write failing example safety tests**

Create `tests/test_examples.py` using injected factories rather than hardware. Begin by locking the shared configuration values:

```python
import pytest

import config
from examples import l20_example, nero_example, nero_l20_example


def test_example_configuration_is_v111_and_uses_separate_buses():
    assert config.NERO_FIRMWARE == "1.11"
    assert config.NERO_CAN_INTERFACE == "socketcan"
    assert config.NERO_CAN_CHANNEL == "can0"
    assert config.L20_CAN_CHANNEL == "can1"
    assert config.NERO_CAN_CHANNEL != config.L20_CAN_CHANNEL
    assert config.CONTROL_HZ == 20


@pytest.mark.parametrize("module", [nero_example, l20_example, nero_l20_example])
def test_parser_defaults_to_read_only(module):
    args = module.build_parser().parse_args([])
    assert args.execute is False


@pytest.mark.parametrize("module", [nero_example, l20_example, nero_l20_example])
def test_execute_requires_exact_confirmation(module, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert module.confirm_execution() is False
    monkeypatch.setattr("builtins.input", lambda prompt: "EXECUTE")
    assert module.confirm_execution() is True


def test_nero_delta_has_conservative_cli_bound():
    parser = nero_example.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--execute", "--delta-rad", "0.2"])


def test_l20_delta_has_conservative_cli_bound():
    parser = l20_example.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--execute", "--delta-raw", "100"])
```

Use argparse `choices` for bounded deltas: Nero choices `(0.005, 0.01, 0.02)` rad and L20 choices `(1, 2, 5, 10)` raw units.

- [ ] **Step 2: Run example tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_examples.py -v
```

Expected: imports fail because the example modules do not exist.

- [ ] **Step 3: Implement shared CLI safety pattern in each small example**

First create `config.py` with the exact typed constants from approved spec section 5:

```python
from typing import Optional


NERO_CAN_INTERFACE: str = "socketcan"
NERO_CAN_CHANNEL: str = "can0"
NERO_FIRMWARE: str = "1.11"
NERO_MAX_JOINT_DELTA: Optional[float] = None
NERO_SPEED_PERCENT: int = 10

L20_HAND_TYPE: str = "right"
L20_HAND_MODEL: str = "L20"
L20_CAN_CHANNEL: str = "can1"

CONTROL_HZ: int = 20
```

The module docstring explains in Chinese that `NERO_MAX_JOINT_DELTA` is in rad and that sudo passwords must never be stored here. Each example imports its defaults from this module and defines complete functions with these signatures:

```python
def build_parser() -> argparse.ArgumentParser:
    """构建只读默认值的命令行解析器。"""
    parser = argparse.ArgumentParser()
    return parser

def confirm_execution() -> bool:
    return input("Type EXECUTE to send a small robot command: ").strip() == "EXECUTE"

def main() -> int:
    """运行示例并返回进程退出码。"""
    args = build_parser().parse_args()
    return run(args)
```

Each module adds its task-specific arguments before returning the parser and implements `run(args: argparse.Namespace) -> int`; the following flows define that behavior.

Nero example flow:

```text
construct -> connect -> print status/q -> if --execute: confirm -> enable ->
copy current q -> apply selected joint delta -> move_joints at configured speed ->
print resulting state -> finally disconnect
```

L20 example flow:

```text
construct -> connect -> print SDK version/position/speed/current/temperature/fault ->
if --execute: confirm -> copy current raw position -> apply bounded active-index delta ->
set_joint_positions_raw -> finally disconnect
```

Joint arguments use zero-based indices consistent with NumPy arrays. L20 `--joint-index` choices are exactly `L20_ACTIVE_POSITION_INDICES`.

Joint example flow:

```text
RobotSystem.connect -> self_check -> observation -> print ->
if --execute: confirm -> enable -> construct normalized hand action from current raw state ->
apply bounded changes -> step once -> demonstrate a finite loop with policy comment ->
finally disconnect
```

The loop comment is exactly:

```python
# Future integration point: action = policy(observation)
```

Do not put a movement command before the execute/confirmation checks.

- [ ] **Step 4: Run example tests and CLI help smoke tests**

Run:

```bash
python3 -m pytest tests/test_examples.py -v
python3 examples/nero_example.py --help
python3 examples/l20_example.py --help
python3 examples/nero_l20_example.py --help
```

Expected: pytest passes; each help command exits 0 without connecting hardware.

- [ ] **Step 5: Commit examples**

Run:

```bash
git add config.py examples tests/test_examples.py
git commit -m "feat: add safe Nero and L20 examples"
```

---

### Task 6: Add minimal configuration, dependency contract, and README

**Files:**
- Modify: `config.py`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/test_project_contract.py`

**Interfaces:**
- Consumes: every public API and command from Tasks 2–5 plus the approved spec.
- Produces: verified default configuration, minimal runtime dependency list, and primary Chinese user documentation with exact commands and action/observation definitions.

- [ ] **Step 1: Write failing project contract tests**

Create `tests/test_project_contract.py`:

```python
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_v111_and_separate_can_channels():
    assert config.NERO_FIRMWARE == "1.11"
    assert config.NERO_CAN_INTERFACE == "socketcan"
    assert config.NERO_CAN_CHANNEL != config.L20_CAN_CHANNEL
    assert config.L20_HAND_MODEL == "L20"
    assert config.CONTROL_HZ == 20


def test_runtime_requirements_are_minimal():
    lines = {
        line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert lines == {
        "numpy", "python-can>=3.3.4", "PyYAML",
        "typing-extensions>=3.7.4.3",
    }


def test_readme_contains_required_safety_and_interface_sections():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = {
        "Nero v1.11 compatibility", "运行前检查", "控制冲突",
        "Action / Observation", "Diffusion Policy", "VLA",
        "静态验证", "真机验证", "disable", "NeroFW.V111",
        "can0", "can1", "candump", "bitrate 1000000",
    }
    missing = sorted(value for value in required if value not in text)
    assert missing == []
```

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_project_contract.py -v
```

Expected: failures identify the still-missing requirements and README; configuration assertions already pass from Task 5.

- [ ] **Step 3: Verify config and create exact minimal requirements**

Recheck that `config.py` still contains the typed constants from spec section 5 and its Chinese module documentation. Create `requirements.txt` with exactly:

```text
numpy
python-can>=3.3.4
PyYAML
typing-extensions>=3.7.4.3
```

- [ ] **Step 4: Write README as the primary operations manual**

Use these top-level sections in this order:

```text
项目目的
系统架构
文件结构
官方 SDK 来源与固定提交
环境安装
CAN 配置
Nero v1.11 compatibility
L20 配置与控制冲突
运行前检查
单独运行 Nero
单独运行 L20
联合运行
API 说明
Action / Observation
单位与维度
Diffusion Policy / ACT 接入
VLA / Teleoperation 接入
常见错误
安全注意事项
验证状态
建议的第一次真机测试顺序
```

Include exact install commands:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./pyAgxArm
export PYTHONPATH="$PWD/linkerhand-python-sdk:$PYTHONPATH"
```

Include read-only CAN checks and manual activation examples:

```bash
lsusb
ip link show
ip -details link show can0
ip -details link show can1
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
candump can0
candump can1
```

State that `candump` listens and normal control should not run alongside any process that sends L20 commands. Explain the official LinkerHand password setting and instruct users to pre-activate CAN rather than store a real sudo password in this project.

The Action/Observation table must match the spec exactly and explicitly omit Nero joint velocity and L20 torque/version. Document the L20 fresh-state latency limitation and the fact that `step()` is not atomic across devices.

Example commands are:

```bash
python3 examples/nero_example.py
python3 examples/nero_example.py --execute
python3 examples/l20_example.py
python3 examples/l20_example.py --execute
python3 examples/nero_l20_example.py
python3 examples/nero_l20_example.py --execute
```

- [ ] **Step 5: Run project contracts and all tests**

Run:

```bash
python3 -m pytest tests/test_project_contract.py -v
python3 -m pytest -q
```

Expected: all tests pass; no hardware connection is attempted.

- [ ] **Step 6: Commit documentation and configuration**

Run:

```bash
git add config.py requirements.txt README.md tests/test_project_contract.py
git commit -m "docs: add setup safety and ML interface guide"
```

---

### Task 7: Perform completion verification and reconcile the final API

**Files:**
- Verify: all created source, examples, tests, config, requirements, and README
- Modify only if verification exposes a tested defect or documentation mismatch

**Interfaces:**
- Consumes: complete implementation and pinned SDK checkouts.
- Produces: evidence-backed static verification report and explicit list of true-hardware checks still outstanding.

- [ ] **Step 1: Inspect repository and official SDK cleanliness**

Run:

```bash
git status --short
git -C pyAgxArm status --short
git -C linkerhand-python-sdk status --short
```

Expected: parent is clean before verification edits; both SDK outputs are empty throughout.

- [ ] **Step 2: Compile every Python file**

Run:

```bash
python3 -m compileall -q robot_control examples tests config.py
```

Expected: exit 0 and no output.

- [ ] **Step 3: Run the complete test suite from a clean process**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass, no warnings, no hardware access.

- [ ] **Step 4: Verify public imports and exact firmware factory selection**

Run:

```bash
python3 -c 'from robot_control import NeroArm, LinkerHandL20, RobotSystem, L20_JOINT_NAMES; print(len(L20_JOINT_NAMES))'
python3 -c 'from pyAgxArm import NeroFW; assert NeroFW.V111 == "v111"; print(NeroFW.V111)'
PYTHONPATH="$PWD/linkerhand-python-sdk${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from LinkerHand.linker_hand_api import LinkerHandApi; print(LinkerHandApi.__name__)'
```

Expected output includes `20`, `v111`, and `LinkerHandApi`. Importing classes must not instantiate hardware.

- [ ] **Step 5: Verify examples remain read-only at argument parsing time**

Run:

```bash
python3 examples/nero_example.py --help >/dev/null
python3 examples/l20_example.py --help >/dev/null
python3 examples/nero_l20_example.py --help >/dev/null
```

Expected: exit 0. Do not run example `main()` without connected hardware because even read-only mode intentionally opens CAN.

- [ ] **Step 6: Audit forbidden and misleading APIs**

Run:

```bash
rg -n 'NeroFW\.(DEFAULT|V112|V120)|move_mit|move_js|move_cpv|get_ik_joint_angles|def get_joint_velocities|def get_torque|def set_torque|def get_version' robot_control examples README.md
```

Expected: no production API use. README may mention excluded names only in compatibility explanations; inspect every match manually.

- [ ] **Step 7: Audit dimensions, units, and documentation alignment**

Run:

```bash
rg -n 'shape.*\(7,\)|shape.*\(20,\)|rad|0.*255|\[-1, 1\]|timestamp|NeroFW.V111' robot_control README.md
```

Confirm each public method docstring matches tests and README. Confirm no L20 raw value is described as rad and no Nero angle is described as degrees.

- [ ] **Step 8: Fix any discovered defect through a fresh RED/GREEN cycle**

For each actual defect, add one focused failing regression test, run it to observe the expected failure, patch the smallest production/documentation unit, then rerun the focused and complete suites. Do not make untested behavioral edits during this step.

- [ ] **Step 9: Commit verification-only fixes if any**

If Step 8 changed files:

```bash
git add robot_control examples tests config.py requirements.txt README.md
git commit -m "test: reconcile wrapper verification findings"
```

If Step 8 made no changes, do not create an empty commit.

- [ ] **Step 10: Prepare final report without claiming hardware success**

Report:

- created/modified files;
- Nero, L20, and RobotSystem APIs;
- V111 selection and excluded firmware features;
- exact Action/Observation shapes and units;
- install, CAN, and example commands;
- test/compile/import command outputs;
- both official SDK worktrees remaining unchanged;
- all true-hardware checks from spec section 15;
- the required first-test sequence from environment/import through DP/VLA.
