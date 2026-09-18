"""Persistent one-bar-at-a-time adapter for the frozen PPO forward arm."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from numbers import Real
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from src.research.cost_model import bar_cost, realized_vol_pips
from src.research.llm_forward.arms.ppo_arm import observation_for_closed_bar
from src.research.llm_forward.arms.ppo_stream import (
    PPOStreamingRunner,
    _as_utc,
    _utc_now,
    _validated_spread,
)
from src.research.llm_forward.canonical import sha256_text
from src.research.llm_forward.corpus import session_cutoff_utc
from src.research.llm_forward.ledger import Ledger, LedgerError
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS, simple_returns

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class StreamState:
    """Cache AFTER the last decision, BEFORE its next price interval is known.

    Counters/weight include that decision. PnL/drawdown are marked at its observed
    close; no future price is fabricated. Before inference all fields are checked
    against the ledger and recomputed at the newly received close.
    """
    session_date: str
    arm_id: str
    next_bar: int = 0
    previous_weight: float = 0.0
    bars_in_position: int = 0
    unrealized: float = 0.0
    drawdown: float = 0.0
    n_changes: int = 0

    @classmethod
    def load_or_create(cls, path: Path, session_date: str, arm_id: str) -> StreamState:
        if not path.is_file():
            return cls(session_date=session_date, arm_id=arm_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("session_date") != session_date or payload.get("arm_id") != arm_id:
            raise ValueError("stream state belongs to another session or arm")
        state = cls(**payload)
        for field in ("next_bar", "bars_in_position", "n_changes"):
            value = getattr(state, field)
            if type(value) is not int or not 0 <= value <= OPERABLE_RETURNS:
                raise ValueError(f"invalid stream state counter: {field}")
        for field in ("previous_weight", "unrealized", "drawdown"):
            value = getattr(state, field)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"invalid stream state number: {field}")
        if state.previous_weight not in EXPOSURE_LEVELS or state.drawdown > 0:
            raise ValueError("stream state weight/drawdown violates gym convention")
        return state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Retry only the same atomic replace, never append the decision again or
        # delete the old state. Persistent permission failures remain fatal.
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.025 * 2 ** attempt)


def _after_decision(state: StreamState, weight: float) -> StreamState:
    changed = weight != state.previous_weight
    return replace(
        state, next_bar=state.next_bar + 1, previous_weight=weight,
        n_changes=state.n_changes + int(changed),
        bars_in_position=(0 if changed else state.bars_in_position) + int(weight != 0),
        unrealized=0.0 if changed else state.unrealized,
    )


def _prefix_states(closes, weights, spread, session_date, arm_id) -> list[StreamState]:
    """Gym states BEFORE each decision, including the final received prefix close."""
    sigma = realized_vol_pips(closes)
    returns = simple_returns(closes)
    states = [StreamState(session_date=session_date, arm_id=arm_id)]
    cumulative = peak = 0.0
    entry_price = None
    for bar, weight in enumerate(weights):
        state = states[-1]
        change = weight - state.previous_weight
        cost = bar_cost(change, spread, sigma[bar], closes[bar]).cost_ret
        cumulative += weight * returns[bar] - cost
        peak = max(peak, cumulative)
        if change != 0:
            entry_price = closes[bar] if weight != 0 else None
        next_state = _after_decision(state, weight)
        next_state.unrealized = (
            float(weight * (closes[bar + 1] / entry_price - 1)) if entry_price else 0.0
        )
        next_state.drawdown = float(cumulative - peak)
        states.append(next_state)
    return states


def _validate_history_timing(row, session_date, bar):
    opened = session_cutoff_utc(session_date, 8)
    closed = opened + timedelta(minutes=5 * (bar + 1))
    received = _as_utc(row["bar_received_at_utc"])
    emitted = _as_utc(row["emitted_at_utc"])
    if (_as_utc(row["session_open_utc"]) != opened
            or _as_utc(row["cutoff_utc"]) != closed
            or not closed <= received <= emitted
            or type(row.get("sealed_before_next_bar")) is not bool
            or row["sealed_before_next_bar"] != (emitted < closed + timedelta(minutes=5))):
        raise LedgerError("stream history has inconsistent temporal metadata")


@contextmanager
def _writer_lock(path: Path):
    """Serialize cooperating stream writers; a stale crash lock needs operator review."""
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LedgerError("stream ledger already locked; no inference attempted") from exc
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


class LiveSessionRunner:
    """Advance exactly one bar when the caller supplies the next closed prefix."""

    def __init__(self, *, model, arm_id: str, model_id: str, preregistration_sha256: str,
                 ledger_path: Path, state_path: Path, observation_version: str = "research37_v2"):
        self._runner = PPOStreamingRunner(
            model, arm_id=arm_id, model_id=model_id,
            preregistration_sha256=preregistration_sha256,
            observation_version=observation_version,
        )
        self.ledger = Ledger(ledger_path, key_field="decision_id")
        self.state_path = state_path
        self.arm_id = arm_id

    def step(self, *, session_date: str, partial, bar_close_utc, bar_received_at_utc,
             spread_pips: float | None = None) -> dict | None:
        """Append one decision, or return ``None`` when the next bar is unavailable."""
        with _writer_lock(self.ledger.path):
            return self._step(session_date=session_date, partial=partial,
                              bar_close_utc=bar_close_utc,
                              bar_received_at_utc=bar_received_at_utc, spread_pips=spread_pips)

    def _step(self, *, session_date, partial, bar_close_utc, bar_received_at_utc, spread_pips):
        if str(getattr(partial, "date", None)) != session_date:
            raise ValueError("partial date must match the requested session")
        if type(partial.bars_received) is not int or not 1 <= partial.bars_received <= 60:
            raise ValueError("partial bars_received must be an integer in [1, 60]")
        cached = StreamState.load_or_create(self.state_path, session_date, self.arm_id)
        if partial.bars_received != cached.next_bar + 1:
            return None
        closes = np.asarray(partial.closes)
        if (closes.ndim != 1 or len(closes) != partial.bars_received
                or closes.dtype.kind not in "iuf" or not np.isfinite(closes).all()
                or np.any(closes <= 0)):
            raise ValueError("stream price prefix must be finite, positive and numeric")
        closes = closes.astype(float)
        spread = _validated_spread(partial.spread_pips if spread_pips is None else spread_pips)
        chain_ok, reason = self.ledger.verify()
        if not chain_ok:
            raise LedgerError(reason)
        rows = list(self.ledger)
        ids = [row["decision_id"] for row in rows]
        if (len(ids) != len(set(ids)) or any(type(r.get("seq")) is not int for r in rows)
                or [r.get("seq") for r in rows] != list(range(len(rows)))):
            raise LedgerError("ledger sequence/identity mismatch")
        stem = f"{session_date}::{self.arm_id}::"
        prior = [row for row in rows if row["decision_id"].startswith(stem)]
        if len(prior) != cached.next_bar:
            raise LedgerError("ledger/cache mismatch; operator recovery required")
        weights = []
        previous_emitted = None
        for bar, row in enumerate(prior):
            _validate_history_timing(row, session_date, bar)
            emitted = _as_utc(row["emitted_at_utc"])
            if previous_emitted is not None and emitted < previous_emitted:
                raise LedgerError("stream history emission clock moved backwards")
            previous_emitted = emitted
            if (row["decision_id"] != f"{stem}b{bar:02d}"
                    or type(row.get("bar_index")) is not int or row["bar_index"] != bar
                    or row.get("session_date") != session_date
                    or row.get("model") != self._runner.model_id
                    or row.get("provider") != "rl_frozen_stream"
                    or row.get("preregistration_sha256") != self._runner.preregistration_sha256
                    or _validated_spread(row.get("spread_pips")) != spread):
                raise LedgerError("stream history identity/order/cost contract mismatch")
            weight = row.get("decision", {}).get("score")
            if isinstance(weight, bool) or not isinstance(weight, Real) or weight not in EXPOSURE_LEVELS:
                raise LedgerError("stream history has a non-discrete weight")
            weights.append(float(weight))
        states = _prefix_states(closes, weights, spread, session_date, self.arm_id)
        expected_cache = _after_decision(states[-2], weights[-1]) if weights else states[0]
        if cached != expected_cache:
            raise LedgerError("cached position state differs from the sealed history")
        # Past feature/context or state rewrites must not silently change the experiment.
        for bar, row in enumerate(prior):
            past = SimpleNamespace(bars_received=bar + 1, market=partial.market[:bar + 1],
                                   context=partial.context)
            values = asdict(states[bar])
            observed = observation_for_closed_bar(past, **{
                key: values[key] for key in ("previous_weight", "bars_in_position",
                                             "unrealized", "drawdown", "n_changes")
            })
            if sha256_text(observed.tobytes().hex()) != row.get("prompt_sha256"):
                raise LedgerError("past observation changed; refusing to rewrite history")
        if previous_emitted is not None and _utc_now() < previous_emitted:
            raise LedgerError("clock moved backwards since the last stream decision")
        state = states[-1]
        record = self._runner.decide(
            session_date=session_date,
            bar_index=state.next_bar,
            partial=partial,
            previous_weight=state.previous_weight,
            bars_in_position=state.bars_in_position,
            unrealized=state.unrealized,
            drawdown=state.drawdown,
            n_changes=state.n_changes,
            bar_close_utc=bar_close_utc,
            bar_received_at_utc=bar_received_at_utc,
            spread_pips=spread,
        )
        written = self.ledger.append(record)
        _after_decision(state, float(record.decision.score)).save(self.state_path)
        return written
