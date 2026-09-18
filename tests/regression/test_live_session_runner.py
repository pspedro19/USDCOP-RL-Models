from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from src.research.llm_forward.stream_runner import LiveSessionRunner, StreamState


class _Model:
    def predict(self, observation, deterministic=True):
        return np.array(2), None


class _Partial:
    def __init__(self, bars_received):
        self.date = "2026-09-11"
        self.bars_received = bars_received
        self.market = np.zeros((bars_received, 2), dtype=np.float32)
        self.context = np.zeros(2, dtype=np.float32)
        self.spread_pips = 3.0
        self.closes = np.full(bars_received, 4000.0)


def test_live_runner_waits_without_advancing_until_next_prefix_exists(tmp_path):
    state_path = tmp_path / "state.json"
    ledger_path = tmp_path / "decisions.jsonl"
    runner = LiveSessionRunner(
        model=_Model(), arm_id="ppo_stream_v1", model_id="ppo.zip",
        preregistration_sha256="a" * 64, ledger_path=ledger_path, state_path=state_path,
    )
    assert runner.step(
        session_date="2026-09-11", partial=_Partial(2),
        bar_close_utc=datetime(2026, 9, 11, 13, 10, tzinfo=UTC),
        bar_received_at_utc=datetime(2026, 9, 11, 13, 10, tzinfo=UTC),
    ) is None
    assert not state_path.exists()
    assert runner.step(
        session_date="2026-09-11", partial=_Partial(1),
        bar_close_utc=datetime(2026, 9, 11, 13, 5, tzinfo=UTC),
        bar_received_at_utc=datetime(2026, 9, 11, 13, 5, tzinfo=UTC),
    ) is not None
    state = StreamState.load_or_create(state_path, "2026-09-11", "ppo_stream_v1")
    assert state.next_bar == 1
    assert len(list(runner.ledger)) == 1
