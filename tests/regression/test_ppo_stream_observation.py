from __future__ import annotations

import numpy as np

from src.research.llm_forward.arms.ppo_arm import observation_for_closed_bar


class _Partial:
    bars_received = 2
    market = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    context = np.asarray([0.5, -0.5], dtype=np.float32)


def test_stream_observation_uses_only_last_closed_prefix_row() -> None:
    obs = observation_for_closed_bar(
        _Partial(), previous_weight=0.5, bars_in_position=1,
        unrealized=0.001, drawdown=-0.002, n_changes=1,
    )
    assert obs[:2].tolist() == [3.0, 4.0]
    assert obs[-2:].tolist() == [0.5, -0.5]


def test_stream_observation_rejects_inconsistent_prefix() -> None:
    partial = _Partial()
    partial.bars_received = 3
    try:
        observation_for_closed_bar(
            partial, previous_weight=0.0, bars_in_position=0,
            unrealized=0.0, drawdown=0.0, n_changes=0,
        )
    except ValueError as exc:
        assert "consistent" in str(exc)
    else:  # pragma: no cover - assertion makes the failure explicit
        raise AssertionError("future/inconsistent prefix was accepted")
