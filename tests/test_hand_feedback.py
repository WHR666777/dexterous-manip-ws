from threading import Event, current_thread
import time
from types import SimpleNamespace

import numpy as np
import pytest

from teleop.hand_feedback import HandFeedbackPoller


def test_background_queries_publish_copied_timestamped_snapshots_at_requested_rate(capsys):
    calls, ready = [], Event()

    def query(*, fresh):
        calls.append((time.monotonic(), fresh, current_thread().name))
        if len(calls) >= 3:
            ready.set()
        return np.full(20, len(calls), dtype=int)

    poller = HandFeedbackPoller(SimpleNamespace(get_joint_positions_raw=query), poll_hz=25.)
    try:
        poller.start()
        assert ready.wait(1.)
        snapshot = poller.get_latest()
        assert snapshot.query_finished_at >= snapshot.query_started_at
        assert snapshot.query_sequence >= 2
        np.testing.assert_array_equal(snapshot.positions_raw, snapshot.query_sequence)
        snapshot.positions_raw[:] = 255
        assert np.all(poller.get_latest().positions_raw < 255)
        assert all(fresh and name == 'l20-feedback' for _, fresh, name in calls)
        assert calls[2][0] - calls[0][0] >= .06
        poller.report()
        assert '[hand feedback]' in capsys.readouterr().out
    finally:
        poller.stop()
    count = len(calls)
    assert poller._thread is None
    poller.stop()
    assert len(calls) == count


def test_stale_query_is_rejected_without_claiming_hardware_packet_age(monkeypatch):
    poller = HandFeedbackPoller(SimpleNamespace(
        get_joint_positions_raw=lambda **_: np.zeros(20, dtype=int)), max_age_s=.1)
    try:
        poller.start()
        snapshot = poller.get_latest()
    finally:
        poller.stop()
    monkeypatch.setattr(time, 'monotonic', lambda: snapshot.query_finished_at + .11)
    with pytest.raises(RuntimeError, match='stale'):
        poller.get_latest()


def test_query_errors_are_reported_and_worker_stops():
    def fail(**kwargs):
        raise ValueError('bad SDK feedback')

    poller = HandFeedbackPoller(SimpleNamespace(get_joint_positions_raw=fail))
    try:
        with pytest.raises(RuntimeError, match='query failed') as caught:
            poller.start()
        assert isinstance(caught.value.__cause__, ValueError)
    finally:
        poller.stop()


def test_stop_timeout_keeps_running_query_available_for_retry():
    entered, release = Event(), Event()

    def blocked(**kwargs):
        entered.set()
        release.wait(1.)
        return np.zeros(20, dtype=int)

    poller = HandFeedbackPoller(SimpleNamespace(get_joint_positions_raw=blocked))
    try:
        with pytest.raises(TimeoutError, match='startup'):
            poller.start(timeout_s=.01)
        assert entered.wait(.5)
        with pytest.raises(TimeoutError, match='keep SDK connected'):
            poller.stop(timeout_s=.01)
    finally:
        release.set()
        poller.stop(timeout_s=1.)
