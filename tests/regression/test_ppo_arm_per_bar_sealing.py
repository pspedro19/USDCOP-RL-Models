"""Clock-controlled unit tests. These records are not real forward decisions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.llm_forward.arms import ppo_stream

UTC = UTC
CLOSE = datetime(2023, 6, 1, 13, 5, tzinfo=UTC)


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        instant = CLOSE + timedelta(seconds=1)

        @classmethod
        def now(cls, tz=None):
            return cls.instant.astimezone(tz)

    monkeypatch.setattr(ppo_stream, "datetime", Clock)
    return Clock


class Model:
    def __init__(self, action=2, after=None):
        self.calls = 0
        self.action = action
        self.after = after

    def predict(self, observation, deterministic=True):
        self.calls += 1
        if self.after:
            self.after()
        return np.asarray(self.action), None


def decide(model, **overrides):
    runner = ppo_stream.PPOStreamingRunner(
        model, arm_id="fixture", model_id="no-model-fit", preregistration_sha256="a" * 64,
    )
    args = {
        "session_date": "2023-06-01", "bar_index": 0,
        "partial": SimpleNamespace(bars_received=1, market=np.zeros((1, 25)),
                                   context=np.zeros(7), spread_pips=3.0),
        "previous_weight": 0.0, "bars_in_position": 0, "unrealized": 0.0,
        "drawdown": 0.0, "n_changes": 0, "bar_close_utc": CLOSE,
        "bar_received_at_utc": CLOSE + timedelta(milliseconds=500),
    }
    args.update(overrides)
    return runner.decide(**args)


@pytest.mark.parametrize("delay,expected", [(2, True), (299.999999, True), (300, False), (310, False)])
def test_sealing_uses_inference_completion_not_receipt(clock, delay, expected):
    finish = CLOSE + timedelta(seconds=delay)
    model = Model(after=lambda: setattr(clock, "instant", finish))
    record = decide(model)
    assert datetime.fromisoformat(record.emitted_at_utc) == finish
    assert record.sealed_before_next_bar is expected
    assert datetime.fromisoformat(record.bar_received_at_utc) == CLOSE + timedelta(milliseconds=500)


@pytest.mark.parametrize("field,value", [
    ("bar_close_utc", CLOSE.replace(tzinfo=None)),
    ("bar_received_at_utc", CLOSE.replace(tzinfo=None)),
    ("bar_received_at_utc", CLOSE - timedelta(microseconds=1)),
    ("bar_received_at_utc", CLOSE + timedelta(seconds=2)),
    ("bar_close_utc", CLOSE - timedelta(minutes=5)),
    ("session_date", "2023-06-02"), ("bar_index", True), ("bar_index", 0.0),
    ("spread_pips", float("nan")), ("spread_pips", -1.0), ("spread_pips", True),
])
def test_bad_metadata_rejected_before_inference(clock, field, value):
    model = Model()
    with pytest.raises(ValueError):
        decide(model, **{field: value})
    assert model.calls == 0


@pytest.mark.parametrize("action", [2.5, True, "2", [], [1, 2], float("nan"), 2+0j, -1, 5])
def test_invalid_actions_are_not_coerced(clock, action):
    with pytest.raises(ValueError):
        decide(Model(action))


def test_backwards_clock_during_inference_rejected(clock):
    model = Model(after=lambda: setattr(clock, "instant", CLOSE))
    with pytest.raises(ValueError):
        decide(model)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_observation_never_reaches_model(clock, value):
    model = Model()
    partial = SimpleNamespace(bars_received=1, market=np.full((1, 25), value),
                              context=np.zeros(7), spread_pips=3.0)
    with pytest.raises(ValueError):
        decide(model, partial=partial)
    assert model.calls == 0
