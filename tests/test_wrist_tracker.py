import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleop.wrist_tracker import WristTracker


def quat(rpy_deg):
    return Rotation.from_euler("xyz", rpy_deg, degrees=True).as_quat()


def test_fixed_roll_pitch_ignores_vr_tilt_tracks_translation_and_wrapped_yaw():
    fixed = np.deg2rad([-97.5, -6.3])
    tracker = WristTracker(np.array([.1, .2, .3]), quat([-97.5, -6.3, -20]),
                           fixed_roll_pitch=fixed, ema_alpha=1.0)
    tracker.update(np.zeros(3), quat([20, 30, 170]))
    position, orientation = tracker.update(np.array([.02, -.03, .01]), quat([-50, -10, -179]))
    np.testing.assert_allclose(position, [.12, .17, .31])
    np.testing.assert_allclose(Rotation.from_quat(orientation).as_euler("xyz", degrees=True),
                               [-97.5, -6.3, -9], atol=1e-9)
    # A different tilt at the same yaw must leave the robot's whole orientation unchanged.
    _, tilted = tracker.update(np.array([.02, -.03, .01]), quat([65, 40, -179]))
    np.testing.assert_allclose(Rotation.from_quat(tilted).as_matrix(),
                               Rotation.from_quat(orientation).as_matrix(), atol=1e-12)


def test_initial_tilt_converges_in_small_steps_without_a_first_frame_jump():
    tracker = WristTracker(np.zeros(3), quat([0, 0, 30]),
                           fixed_roll_pitch=np.deg2rad([-15, 20]), roll_pitch_step_deg=1)
    _, anchor = tracker.update(np.zeros(3), quat([50, 10, 70]))
    previous = Rotation.from_quat(anchor)
    np.testing.assert_allclose(previous.as_euler("xyz", degrees=True), [0, 0, 30], atol=1e-9)
    for _ in range(30):
        _, orientation = tracker.update(np.zeros(3), quat([50, 10, 70]))
        current = Rotation.from_quat(orientation)
        assert np.rad2deg((current * previous.inv()).magnitude()) <= 1.000001
        previous = current
    np.testing.assert_allclose(current.as_euler("xyz", degrees=True), [-15, 20, 30], atol=1e-9)


def test_reanchor_resets_yaw_reference_but_preserves_demonstrated_tilt():
    tracker = WristTracker(np.zeros(3), quat([-15, 20, 30]),
                           fixed_roll_pitch=np.deg2rad([-15, 20]))
    tracker.update(np.zeros(3), quat([0, 0, 0]))
    tracker.update(np.zeros(3), quat([10, 25, 40]))
    tracker.reanchor(np.array([1, 2, 3]), quat([-15, 20, 80]))
    tracker.update(np.zeros(3), quat([40, 20, 100]))
    position, orientation = tracker.update(np.zeros(3), quat([-30, -10, 110]))
    np.testing.assert_allclose(position, [1, 2, 3])
    np.testing.assert_allclose(Rotation.from_quat(orientation).as_euler("xyz", degrees=True),
                               [-15, 20, 90], atol=1e-9)


@pytest.mark.parametrize("fixed", [[1], [1, 2, 3], [float("nan"), 0]])
def test_fixed_tilt_requires_two_finite_angles(fixed):
    with pytest.raises(ValueError, match="fixed_roll_pitch"):
        WristTracker(np.zeros(3), quat([0, 0, 0]), fixed_roll_pitch=fixed)
