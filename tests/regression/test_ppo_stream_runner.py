from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.research.llm_forward.arms.ppo_stream import PPOStreamingRunner


class _Model:
    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert observation.dtype == np.float32
        return np.array(2), None  # frozen action-space middle/flat level


class _Partial:
    bars_received = 1
    market = np.zeros((1, 2), dtype=np.float32)
    context = np.zeros(2, dtype=np.float32)
    spread_pips = 3.0


def test_runner_emits_one_bar_record_and_marks_on_time_seal(monkeypatch) -> None:
    from src.research.llm_forward.arms import ppo_stream

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 11, 13, 5, 31, tzinfo=UTC)

    monkeypatch.setattr(ppo_stream, "datetime", Clock)
    runner = PPOStreamingRunner(_Model(), arm_id="ppo_stream_v1",
                                model_id="ppo.zip", preregistration_sha256="a" * 64)
    record = runner.decide(
        session_date="2026-09-11", bar_index=0, partial=_Partial(),
        previous_weight=0.0, bars_in_position=0, unrealized=0.0,
        drawdown=0.0, n_changes=0,
        bar_close_utc=datetime(2026, 9, 11, 13, 5, tzinfo=UTC),
        bar_received_at_utc=datetime(2026, 9, 11, 13, 5, 30, tzinfo=UTC),
    )
    assert record.decision_id == "2026-09-11::ppo_stream_v1::b00"
    assert record.bar_index == 0
    assert record.sealed_before_next_bar is True
    assert record.decision.score == 0.0
    assert record.session_open_utc == "2026-09-11T13:00:00+00:00"


def test_runner_rejects_prefix_that_contains_an_unaccounted_future_bar() -> None:
    runner = PPOStreamingRunner(_Model(), arm_id="ppo_stream_v1",
                                model_id="ppo.zip", preregistration_sha256="a" * 64)
    with pytest.raises(ValueError, match="prefix length"):
        runner.decide(
            session_date="2026-09-11", bar_index=0, partial=type("P", (), {
                "bars_received": 2, "market": np.zeros((2, 2), dtype=np.float32),
                "context": np.zeros(2, dtype=np.float32), "spread_pips": 3.0,
            })(),
            previous_weight=0.0, bars_in_position=0, unrealized=0.0,
            drawdown=0.0, n_changes=0,
            bar_close_utc="2026-09-11T13:05:00+00:00",
            bar_received_at_utc="2026-09-11T13:05:30+00:00",
        )
