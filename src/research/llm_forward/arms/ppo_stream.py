"""Causal per-bar runner for the native frozen PPO forward arm.

The runner deliberately separates inference from ledger I/O.  A live adapter supplies the
closed-bar prefix and persisted position state; this module returns one immutable
``DecisionRecord`` which the adapter may append.  It never loads the full target session.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from numbers import Integral, Real

import numpy as np

from src.research.llm_forward.arms.ppo_arm import observation_for_closed_bar
from src.research.llm_forward.canonical import sha256_text
from src.research.llm_forward.corpus import session_cutoff_utc
from src.research.llm_forward.schema import Decision, DecisionRecord
from src.research.observation_contract import (
    LEGACY_VERSION,
    observation_contract,
    require_model_contract,
)
from src.research.session_env import EXPOSURE_LEVELS
from src.research.session_gym import OPERABLE_RETURNS


def _direction(weight: float) -> str:
    return "flat" if weight == 0.0 else ("long" if weight > 0 else "short")


def _as_utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("stream timestamps must have an explicit timezone")
    return value.astimezone(UTC)


def _validated_spread(value) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ValueError("spread must be a finite nonnegative real number")
    if not np.isfinite(value) or value < 0:
        raise ValueError("spread must be a finite nonnegative real number")
    return float(value)


def _utc_now() -> datetime:
    return _as_utc(datetime.now(UTC))


class PPOStreamingRunner:
    """Produce one sealed record for each closed bar of a session."""

    def __init__(self, model, *, arm_id: str, model_id: str, preregistration_sha256: str,
                 observation_version: str = LEGACY_VERSION):
        require_model_contract(model, observation_version)
        self.observation_version = observation_version
        self.model = model
        self.arm_id = arm_id
        self.model_id = model_id
        self.preregistration_sha256 = preregistration_sha256

    def decide(
        self,
        *,
        session_date: str,
        bar_index: int,
        partial,
        previous_weight: float,
        bars_in_position: int,
        unrealized: float,
        drawdown: float,
        n_changes: int,
        bar_close_utc: datetime | str,
        bar_received_at_utc: datetime | str,
        spread_pips: float | None = None,
    ) -> DecisionRecord:
        """Infer from a prefix ending at ``bar_index`` and return an unappended record."""
        if getattr(partial, "observation_version", LEGACY_VERSION) != self.observation_version:
            raise ValueError("stream observation version differs from the frozen model")
        if self.observation_version != LEGACY_VERSION:
            observation_contract(self.observation_version).validate_arrays(
                partial.market, partial.context, bars=partial.bars_received)
        if (isinstance(bar_index, bool | np.bool_) or not isinstance(bar_index, Integral)
                or not 0 <= bar_index < OPERABLE_RETURNS):
            raise ValueError(f"bar_index must be in [0, {OPERABLE_RETURNS - 1}]")
        if partial.bars_received != bar_index + 1:
            raise ValueError("partial prefix length must equal bar_index + 1")
        close_time = _as_utc(bar_close_utc)
        received = _as_utc(bar_received_at_utc)
        session_open = session_cutoff_utc(session_date, 8)
        if close_time != session_open + timedelta(minutes=5 * (int(bar_index) + 1)):
            raise ValueError("bar close must match the session and five-minute bar index")
        started = _utc_now()
        if not close_time <= received <= started:
            raise ValueError("bar timing must satisfy close <= received <= inference start")
        spread = _validated_spread(partial.spread_pips if spread_pips is None else spread_pips)
        # Validate before clipping: np.clip would silently turn infinity into a feature.
        for values in (partial.market, partial.context,
                       [previous_weight, bars_in_position, unrealized, drawdown, n_changes]):
            array = np.asarray(values)
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError("observation inputs must be finite real numbers")
        observation = observation_for_closed_bar(
            partial,
            previous_weight=previous_weight,
            bars_in_position=bars_in_position,
            unrealized=unrealized,
            drawdown=drawdown,
            n_changes=n_changes,
        )
        action, _ = self.model.predict(observation, deterministic=True)
        completed = _utc_now()
        if completed < started:
            raise ValueError("clock moved backwards during inference")
        action_array = np.asarray(action)
        if action_array.size != 1 or action_array.dtype.kind not in "iu":
            raise ValueError("model must return exactly one integer action index")
        action_index = int(action_array.reshape(-1)[0])
        if not 0 <= action_index < len(EXPOSURE_LEVELS):
            raise ValueError(f"model returned invalid action index {action_index}")
        weight = float(EXPOSURE_LEVELS[action_index])
        decision_id = f"{session_date}::{self.arm_id}::b{bar_index:02d}"
        return DecisionRecord(
            seq=-1,
            decision_id=decision_id,
            session_date=session_date,
            emitted_at_utc=completed.isoformat(timespec="microseconds"),
            cutoff_utc=close_time.isoformat(timespec="seconds"),
            session_open_utc=session_open.isoformat(timespec="seconds"),
            sealed_before_open=False,
            preregistration_sha256=self.preregistration_sha256,
            prompt_sha256=sha256_text(observation.tobytes().hex()),
            model=self.model_id,
            provider="rl_frozen_stream",
            temperature=0.0,
            seed=None,
            corpus=[],
            decision=Decision(
                score=weight,
                direction=_direction(weight),
                confidence=1.0,
                rationale=f"frozen PPO bar {bar_index}; causal prefix {partial.bars_received}",
            ),
            abstained=False,
            abstain_reason=None,
            usage=None,
            decision_path=None,
            spread_pips=spread,
            information_edge=("causal prefix through b; completion timestamp is inference end, "
                              "not execution or durable-storage acknowledgement"),
            sealed_before_next_bar=completed < close_time + timedelta(minutes=5),
            bar_index=bar_index,
            bar_received_at_utc=received.isoformat(timespec="microseconds"),
        )
