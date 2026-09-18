"""Engineering fixtures, not market experiments or forward performance evidence."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research.features import FEATURE_ORDER, GROUPS
from src.research.llm_forward.arms import ppo_stream
from src.research.llm_forward.canonical import GENESIS_HASH, chain_hash
from src.research.llm_forward.ledger import LedgerError
from src.research.llm_forward.settle_thesis import aggregate_bar_records, settle_session
from src.research.llm_forward.stream_runner import LiveSessionRunner, StreamState
from src.research.session_env import EXPOSURE_LEVELS
from src.research.session_gym import SessionSpec, SessionTradingEnv

UTC = UTC
SESSION = "2023-06-01"
FIRST_CLOSE = datetime(2023, 6, 1, 13, 5, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        instant = FIRST_CLOSE + timedelta(seconds=1)

        @classmethod
        def now(cls, tz=None):
            return cls.instant.astimezone(tz)

    monkeypatch.setattr(ppo_stream, "datetime", Clock)
    return Clock


class Model:
    def __init__(self, actions=None):
        self.actions = iter(actions if actions is not None else [4] * 59)
        self.observations = []

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        self.observations.append(observation.copy())
        return np.asarray(next(self.actions)), None


def spec_for(closes=None):
    if closes is None:
        closes = 4000 * np.exp(np.sin(np.arange(60) / 3) * 0.018)
    n_market = len(FEATURE_ORDER) - len(GROUPS["posicion"]) - 7
    return SessionSpec(
        date=date.fromisoformat(SESSION), close=np.asarray(closes, dtype=float),
        market=np.zeros((60, n_market), dtype=np.float32),
        context=np.zeros(7, dtype=np.float32), spread_pips=3.0,
    )


def prefix(spec, bar):
    return SimpleNamespace(
        date=spec.date, bars_received=bar + 1, closes=spec.close[:bar + 1].copy(),
        market=spec.market[:bar + 1].copy(), context=spec.context.copy(),
        spread_pips=spec.spread_pips,
    )


def runner_for(tmp_path, model=None, **kwargs):
    return LiveSessionRunner(
        model=model or Model(), arm_id="ppo_test", model_id="fixture-not-trained",
        preregistration_sha256="a" * 64, ledger_path=tmp_path / "ledger.jsonl",
        state_path=tmp_path / "state.json", **kwargs,
    )


def step(runner, spec, bar, clock):
    closed = FIRST_CLOSE + timedelta(minutes=5 * bar)
    clock.instant = closed + timedelta(seconds=1)
    return runner.step(
        session_date=SESSION, partial=prefix(spec, bar), bar_close_utc=closed,
        bar_received_at_utc=closed + timedelta(milliseconds=500),
    )


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
@pytest.mark.parametrize("prices", ["fixture", "real_source"])
def test_all_59_observations_match_gym_including_restarts(tmp_path, clock, seed, prices):
    closes = None
    if prices == "real_source":
        frame = pd.read_parquet(ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet")
        dates = pd.to_datetime(frame.time).dt.tz_convert("America/Bogota").dt.date
        closes = frame.loc[dates == date.fromisoformat(SESSION), "close"].to_numpy()
        assert len(closes) == 60
    spec = spec_for(closes)
    actions = np.random.default_rng(seed).integers(0, 5, 59).tolist()
    # Holds, same-sign resizing, closing, flat holding and both sign reversals.
    actions[:12] = [4, 4, 3, 3, 0, 0, 2, 2, 1, 1, 4, 3]
    env = SessionTradingEnv([spec], shuffle=False)
    expected, _ = env.reset()
    model = Model(actions)
    runner = runner_for(tmp_path, model)
    for bar, action in enumerate(actions):
        if bar in (1, 11, 58):
            runner = runner_for(tmp_path, model)
        record = step(runner, spec, bar, clock)
        np.testing.assert_array_equal(model.observations[-1], expected)
        assert record["decision"]["score"] == EXPOSURE_LEVELS[action]
        expected, _, terminated, _, _ = env.step(action)
        assert terminated == (bar == 58)
    assert runner.ledger.verify()[0]
    rows = list(runner.ledger)
    scored = settle_session(aggregate_bar_records(rows), spec.close)
    assert scored["daily_return"] == pytest.approx(env.last_result.daily_return, abs=1e-12)
    assert scored["terminal_cost"] == pytest.approx(env.last_result.terminal_cost, abs=1e-12)


@pytest.mark.parametrize("field,value", [
    ("previous_weight", -1.0), ("bars_in_position", 8), ("n_changes", 8),
    ("unrealized", 0.75), ("drawdown", -0.75), ("next_bar", True),
    ("n_changes", False), ("previous_weight", float("nan")),
])
def test_tampered_cache_is_not_authority(tmp_path, clock, field, value):
    model = Model()
    runner = runner_for(tmp_path, model)
    spec = spec_for()
    step(runner, spec, 0, clock)
    state = StreamState.load_or_create(runner.state_path, SESSION, "ppo_test")
    runner.state_path.write_text(json.dumps(asdict(replace(state, **{field: value}))),
                                 encoding="utf-8")
    before = runner.ledger.path.read_bytes()
    with pytest.raises((ValueError, LedgerError)):
        step(runner, spec, 1, clock)
    assert len(model.observations) == 1
    assert runner.ledger.path.read_bytes() == before


def test_chain_tampering_blocks_before_model(tmp_path, clock):
    model = Model()
    runner = runner_for(tmp_path, model)
    spec = spec_for()
    step(runner, spec, 0, clock)
    row = next(iter(runner.ledger))
    row["decision"]["score"] = -1.0
    runner.ledger.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = runner.ledger.path.read_bytes()
    with pytest.raises((ValueError, LedgerError)):
        step(runner, spec, 1, clock)
    assert len(model.observations) == 1
    assert runner.ledger.path.read_bytes() == before


@pytest.mark.parametrize("change", ["model", "spread", "past_market", "context", "missing_ledger"])
def test_history_identity_changes_block_before_model(tmp_path, clock, change):
    model = Model()
    runner = runner_for(tmp_path, model)
    spec = spec_for()
    step(runner, spec, 0, clock)
    if change == "model":
        runner._runner.model_id = "different-model"
    elif change == "spread":
        spec = replace(spec, spread_pips=4.0)
    elif change == "past_market":
        spec.market[0, 0] = 0.25
    elif change == "context":
        spec.context[0] = 0.25
    else:
        runner.ledger.path.unlink()
    with pytest.raises((ValueError, LedgerError)):
        step(runner, spec, 1, clock)
    assert len(model.observations) == 1


def test_append_without_state_save_is_not_silently_replayed(tmp_path, clock, monkeypatch):
    runner = runner_for(tmp_path)
    spec = spec_for()
    original = StreamState.save

    def crash(self, path):
        raise OSError("simulated crash after durable append")

    monkeypatch.setattr(StreamState, "save", crash)
    with pytest.raises(OSError):
        step(runner, spec, 0, clock)
    monkeypatch.setattr(StreamState, "save", original)
    model = Model()
    restarted = runner_for(tmp_path, model)
    with pytest.raises((ValueError, LedgerError)):
        step(restarted, spec, 0, clock)
    assert model.observations == []
    assert len(list(runner.ledger)) == 1


@pytest.mark.parametrize("bad", [0.0, -4.0, float("nan"), float("inf"), True, "4000"])
def test_invalid_price_prefix_blocks_before_model(tmp_path, clock, bad):
    model = Model()
    runner = runner_for(tmp_path, model)
    part = prefix(spec_for(), 0)
    part.closes = np.array([bad])
    with pytest.raises(ValueError):
        runner.step(session_date=SESSION, partial=part, bar_close_utc=FIRST_CLOSE,
                    bar_received_at_utc=FIRST_CLOSE)
    assert model.observations == []


def test_existing_writer_lock_prevents_inference(tmp_path, clock):
    model = Model()
    runner = runner_for(tmp_path, model)
    lock_path = runner.ledger.path.with_suffix(".jsonl.lock")
    lock_path.touch()
    with pytest.raises(LedgerError):
        step(runner, spec_for(), 0, clock)
    assert model.observations == []
    assert lock_path.exists()


@pytest.mark.parametrize("field,value", [
    ("seq", False), ("cutoff_utc", "2022-06-01T13:05:00+00:00"),
    ("bar_received_at_utc", "2023-06-01T13:04:59+00:00"),
    ("emitted_at_utc", "2023-06-01T14:00:00+00:00"),
    ("session_open_utc", "2023-06-01T12:00:00+00:00"),
    ("sealed_before_next_bar", 1),
])
def test_rehashed_but_semantically_invalid_history_is_rejected(tmp_path, clock, field, value):
    model = Model()
    runner = runner_for(tmp_path, model)
    spec = spec_for()
    step(runner, spec, 0, clock)
    row = next(iter(runner.ledger))
    row[field] = value
    payload = {key: value for key, value in row.items() if key not in ("prev_hash", "record_hash")}
    row["record_hash"] = chain_hash(GENESIS_HASH, payload)
    runner.ledger.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert runner.ledger.verify()[0]
    with pytest.raises((ValueError, LedgerError)):
        step(runner, spec, 1, clock)
    assert len(model.observations) == 1


def test_partial_date_must_equal_request_date(tmp_path, clock):
    runner = runner_for(tmp_path)
    spec = replace(spec_for(), date=date(2023, 6, 2))
    with pytest.raises(ValueError):
        step(runner, spec, 0, clock)


def test_transient_atomic_replace_error_retries_without_duplicate_decision(tmp_path, clock, monkeypatch):
    actual_replace = Path.replace
    attempts = []

    def temporary_denial(path, target):
        attempts.append((path, target))
        if len(attempts) == 1:
            raise PermissionError("temporary sharing denial")
        return actual_replace(path, target)

    monkeypatch.setattr(Path, "replace", temporary_denial)
    runner = runner_for(tmp_path)
    step(runner, spec_for(), 0, clock)
    assert len(attempts) == 2
    assert len(list(runner.ledger)) == 1
    assert StreamState.load_or_create(runner.state_path, SESSION, "ppo_test").next_bar == 1


def test_permanent_replace_error_leaves_evidence_and_blocks_restart(tmp_path, clock, monkeypatch):
    actual_replace = Path.replace
    attempts = []

    def permanent_denial(path, target):
        attempts.append((path, target))
        raise PermissionError("permanent sharing denial")

    monkeypatch.setattr(Path, "replace", permanent_denial)
    runner = runner_for(tmp_path)
    with pytest.raises(PermissionError):
        step(runner, spec_for(), 0, clock)
    assert 1 <= len(attempts) <= 5
    assert len(list(runner.ledger)) == 1
    monkeypatch.setattr(Path, "replace", actual_replace)
    with pytest.raises(LedgerError):
        step(runner, spec_for(), 0, clock)


def test_complete_stream_reaches_settlement_command_once(tmp_path, clock, monkeypatch):
    from src.research.llm_forward import settle_thesis

    runner = runner_for(tmp_path)
    spec = spec_for()
    for bar in range(59):
        step(runner, spec, bar, clock)
    monkeypatch.setattr(settle_thesis, "DECISIONS_PATH", runner.ledger.path)
    monkeypatch.setattr(settle_thesis, "SETTLEMENTS_PATH", tmp_path / "settled.jsonl")
    assert settle_thesis.run({SESSION: spec.close}) == 0
    from src.research.llm_forward.ledger import Ledger
    settlements = Ledger(tmp_path / "settled.jsonl", "decision_id")
    assert settlements.verify()[0]
    assert len(list(settlements)) == 1
    first_bytes = settlements.path.read_bytes()
    assert settle_thesis.run({SESSION: spec.close}) == 0
    assert settlements.path.read_bytes() == first_bytes


@pytest.mark.parametrize("timestamp", ["2023-06-01T08:00:00-05:00", "2023-06-01T13:00:00+00:00"])
def test_cli_uses_bar_open_plus_five_minutes_and_early_receipt(tmp_path, monkeypatch, timestamp):
    import sys

    import stable_baselines3

    import scripts.analysis.run_ppo_stream_bar as cli

    bars_path = tmp_path / "prefix.csv"
    bars_path.write_text("time,close\n" + timestamp + ",4000\n", encoding="utf-8")
    model_path = tmp_path / "fixture.zip"
    model_path.touch()
    captured = {}

    class Clock(datetime):
        instant = FIRST_CLOSE + timedelta(seconds=1)

        @classmethod
        def now(cls, tz=None):
            return cls.instant

    class StubRunner:
        def __init__(self, **kwargs):
            pass

        def step(self, **kwargs):
            captured.update(kwargs)
            return {"test_record": True}

    def build(day, bars):
        Clock.instant += timedelta(hours=1)
        return prefix(spec_for(), 0)

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "build_live_spec_partial", build)
    monkeypatch.setattr(cli, "load_preregistration", lambda path: ({}, "a" * 64))
    monkeypatch.setattr(cli, "arm_spec", lambda *args: {"kind": "rl_frozen", "decisions_per_session": 59})
    monkeypatch.setattr(cli, "LiveSessionRunner", StubRunner)
    monkeypatch.setattr(stable_baselines3.PPO, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(sys, "argv", [
        "run_ppo_stream_bar", "--session-date", SESSION, "--bar-index", "0",
        "--bars-path", str(bars_path), "--model", str(model_path), "--model-id", "fixture",
        "--arm-id", "fixture", "--ledger", str(tmp_path / "not-written.jsonl"),
        "--state", str(tmp_path / "not-written-state.json"),
    ])
    assert cli.main() == 0
    assert captured["bar_close_utc"] == FIRST_CLOSE
    assert datetime.fromisoformat(captured["bar_received_at_utc"]) == FIRST_CLOSE + timedelta(seconds=1)


def test_clock_rollback_between_calls_blocks_before_next_inference(tmp_path, clock):
    class SlowModel(Model):
        def predict(self, observation, deterministic=True):
            result = super().predict(observation, deterministic)
            clock.instant += timedelta(minutes=20)
            return result

    model = SlowModel()
    runner = runner_for(tmp_path, model)
    spec = spec_for()
    first = step(runner, spec, 0, clock)
    assert first["sealed_before_next_bar"] is False
    with pytest.raises(LedgerError, match="clock moved backwards"):
        step(runner, spec, 1, clock)
    assert len(model.observations) == 1


@pytest.mark.parametrize("count", [True, 1.0])
def test_prefix_count_is_strict_integer(tmp_path, clock, count):
    runner = runner_for(tmp_path)
    part = prefix(spec_for(), 0)
    part.bars_received = count
    with pytest.raises(ValueError):
        runner.step(session_date=SESSION, partial=part, bar_close_utc=FIRST_CLOSE,
                    bar_received_at_utc=FIRST_CLOSE)
