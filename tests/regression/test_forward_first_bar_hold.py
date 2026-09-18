"""C049 controls: no model training, provider calls or prospective observations."""

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.llm_forward.arms import ppo_arm, ppo_stream
from src.research.llm_forward.canonical import GENESIS_HASH, chain_hash
from src.research.llm_forward.settlement_accounting import candidate_from_sources

DAY = "2023-06-01"
OPEN = datetime(2023, 6, 1, 13, tzinfo=UTC)
CLOSE = OPEN + timedelta(minutes=5)


def rehash(row):
    row["prev_hash"] = GENESIS_HASH
    row["record_hash"] = chain_hash(GENESIS_HASH, {
        k: v for k, v in row.items() if k not in {"prev_hash", "record_hash"}
    })
    return row


def held_row(**changes):
    row = {
        "seq": 0, "decision_id": f"{DAY}::unit_held", "session_date": DAY,
        "emitted_at_utc": (CLOSE + timedelta(seconds=2)).isoformat(),
        "cutoff_utc": CLOSE.isoformat(), "session_open_utc": OPEN.isoformat(),
        "sealed_before_open": False, "preregistration_sha256": "a" * 64,
        "prompt_sha256": "b" * 64, "model": "unit-no-fit", "provider": "rl_frozen_stream",
        "temperature": 0.0, "seed": None, "corpus": [],
        "decision": {"score": 0.5, "direction": "long", "confidence": 1.0, "rationale": "unit"},
        "abstained": False, "abstain_reason": None, "usage": None, "spread_pips": 3.0,
        "decision_path": [0.5] * 59, "sealed_before_next_bar": True, "bar_index": 0,
        "bar_received_at_utc": (CLOSE + timedelta(seconds=1)).isoformat(),
        "decision_schedule": "first_bar_hold",
    }
    row.update(changes)
    return rehash(row)


def test_explicit_held_record_is_one_candidate_and_preserves_source():
    row = held_row()
    original = deepcopy(row)
    candidate = candidate_from_sources([row])
    assert candidate == original == row
    assert candidate["decision_path"] == [0.5] * 59


@pytest.mark.parametrize("changes", [
    {"decision_schedule": None}, {"decision_schedule": "unknown"},
    {"decision_schedule": True}, {"decision_schedule": []}, {"decision_schedule": False},
    {"decision_schedule": 0}, {"decision_schedule": ""}, {"decision_schedule": {}},
    {"decision_path": [0.5] * 58}, {"decision_path": None},
    {"decision_path": [0.5] * 58 + [-0.5]}, {"decision_path": [True] * 59},
    {"decision_path": ["0.5"] * 59}, {"decision_path": [0.0] * 59},
    {"bar_index": 0.0}, {"bar_index": True}, {"bar_index": None}, {"bar_index": 1},
    {"decision_id": f"{DAY}::unit_held::b00"}, {"provider": "deepseek"},
    {"provider": "stub"}, {"sealed_before_open": True},
    {"cutoff_utc": OPEN.isoformat()}, {"cutoff_utc": CLOSE.replace(tzinfo=None).isoformat()},
    {"bar_received_at_utc": (CLOSE - timedelta(microseconds=1)).isoformat()},
    {"emitted_at_utc": (CLOSE + timedelta(minutes=5)).isoformat()},
    {"sealed_before_next_bar": None}, {"sealed_before_next_bar": "True"},
    {"session_open_utc": CLOSE.isoformat()}, {"abstained": True},
])
def test_held_admission_cannot_be_inferred_coerced_or_backdated(changes):
    with pytest.raises((ValueError, TypeError)):
        candidate_from_sources([held_row(**changes)])


def test_unknown_schedule_does_not_fall_back_to_preopen_daily():
    row = held_row(
        decision_schedule="unknown", bar_index=None, decision_path=None,
        emitted_at_utc=(OPEN - timedelta(seconds=2)).isoformat(),
        cutoff_utc=OPEN.isoformat(), sealed_before_open=True, sealed_before_next_bar=None,
    )
    with pytest.raises(ValueError):
        candidate_from_sources([row])


def test_held_record_cannot_be_one_member_of_a_full_stream():
    from src.research.llm_forward.settlement_accounting import aggregate
    rows = []
    for index in range(59):
        close = OPEN + timedelta(minutes=5 * (index + 1))
        rows.append(held_row(
            seq=index, decision_id=f"{DAY}::unit_held::b{index:02d}", bar_index=index,
            decision_schedule=None, decision_path=None, cutoff_utc=close.isoformat(),
            bar_received_at_utc=(close + timedelta(seconds=1)).isoformat(),
            emitted_at_utc=(close + timedelta(seconds=2)).isoformat(),
        ))
    assert aggregate(rows) is not None
    rows[0]["decision_schedule"] = "first_bar_hold"
    rehash(rows[0])
    assert aggregate(rows) is None


class Model:
    def __init__(self, after=None, action=3):
        self.calls = 0
        self.after = after
        self.action = action

    def predict(self, observation, deterministic=True):
        self.calls += 1
        assert observation.shape == (37,)
        if self.after:
            self.after()
        return np.asarray(self.action), None


@pytest.fixture
def controlled_clock(monkeypatch):
    state = SimpleNamespace(now=CLOSE + timedelta(seconds=1))
    monkeypatch.setattr(ppo_stream, "_utc_now", lambda: state.now)
    monkeypatch.setattr(ppo_arm, "_utc_now", lambda: state.now, raising=False)
    return state


def produce(model, **changes):
    kwargs = {
        "session_date": DAY, "arm_id": "unit_held", "model_id": "unit-no-fit",
        "preregistration_sha256": "a" * 64,
        "partial": SimpleNamespace(date=DAY, bars_received=1, market=np.zeros((1, 25)),
                                   context=np.zeros(7), spread_pips=3.0),
        "bar_received_at_utc": CLOSE + timedelta(seconds=1),
    }
    kwargs.update(changes)
    return ppo_arm.decide_first_bar_hold(model, **kwargs)


def test_producer_makes_one_inference_then_ledger_settlement_reader_agree(
    controlled_clock, monkeypatch, tmp_path,
):
    from src.research.llm_forward import settle_thesis, verify
    from src.research.llm_forward.ledger import Ledger
    from src.research.llm_forward.settlement_accounting import read_ledger
    from src.research.session_env import run_session

    model = Model(after=lambda: setattr(controlled_clock, "now", CLOSE + timedelta(seconds=3)))
    record = produce(model)
    assert model.calls == 1
    assert record.decision_schedule == "first_bar_hold"
    assert record.emitted_at_utc == (CLOSE + timedelta(seconds=3)).isoformat(timespec="microseconds")
    assert record.bar_received_at_utc == (CLOSE + timedelta(seconds=1)).isoformat(timespec="microseconds")
    assert record.decision_path == [0.5] * 59
    assert record.decision_id == f"{DAY}::unit_held"
    assert "fill" in record.information_edge.lower()
    assert record.prompt_sha256 != "b" * 64

    decisions, settlements = tmp_path / "decisions.jsonl", tmp_path / "settlements.jsonl"
    Ledger(decisions, "decision_id").append(record)
    for module in (settle_thesis, verify):
        monkeypatch.setattr(module, "DECISIONS_PATH", decisions)
        monkeypatch.setattr(module, "SETTLEMENTS_PATH", settlements)
    closes = np.linspace(4000.0, 4100.0, 60)  # accounting fixture, not a market result
    assert settle_thesis.run({DAY: closes}) == 0
    before = settlements.read_bytes()
    assert settle_thesis.run({DAY: closes}) == 0
    assert settlements.read_bytes() == before
    report = verify.audit_snapshot(read_ledger(decisions), read_ledger(settlements))
    assert report["admissible_session_arms"] == report["net_verified_sessions"] == 1
    assert report["excluded_decision_records"] == 0
    assert report["scientific_ready"] is False
    expected = run_session(closes, np.full(59, 0.5), 3.0, date=DAY)
    assert report["daily_returns"][0]["net_return"] == expected.daily_return


@pytest.mark.parametrize("delay,valid", [(299.999999, True), (300, False), (310, False)])
def test_held_deadline_is_real_inference_end(controlled_clock, delay, valid):
    model = Model(after=lambda: setattr(controlled_clock, "now", CLOSE + timedelta(seconds=delay)))
    record = produce(model)
    assert record.sealed_before_next_bar is valid
    row = rehash(asdict(record))
    if valid:
        assert candidate_from_sources([row]) == row
    else:
        with pytest.raises(ValueError):
            candidate_from_sources([row])


def test_held_rejects_two_bar_prefix_before_predict(controlled_clock):
    model = Model()
    with pytest.raises(ValueError):
        produce(model, partial=SimpleNamespace(bars_received=2, market=np.zeros((2, 25)),
                                              context=np.zeros(7), spread_pips=3.0))
    assert model.calls == 0


def test_now_override_is_not_a_production_backdating_interface(monkeypatch):
    monkeypatch.setattr(ppo_arm, "load_preregistration", lambda *a: pytest.fail("must reject before IO"))
    with pytest.raises(ValueError, match="now_override"):
        ppo_arm.run(DAY, now_override=CLOSE + timedelta(seconds=1))


@pytest.fixture
def source_adapter(controlled_clock, monkeypatch, tmp_path):
    import pandas as pd
    from stable_baselines3 import PPO

    model = Model()
    checkpoint = tmp_path / "unit.zip"
    checkpoint.write_bytes(b"NOT A CHECKPOINT: PPO.load is replaced in this test")
    source = pd.DataFrame({
        "time": [OPEN + timedelta(minutes=5 * index) for index in range(60)],
        "symbol": ["USDCOP"] * 60,
        "open": [4000.0] * 60, "high": [4001.0] * 60,
        "low": [3999.0] * 60, "close": [4000.0] * 60,
    })
    state = SimpleNamespace(source=source, model=model, loads=0, prefixes=[],
                            arm={"kind": "rl_frozen", "decisions_per_session": 1, "model": str(checkpoint)})
    monkeypatch.setattr(ppo_arm, "load_preregistration", lambda *a: ({}, "a" * 64))
    monkeypatch.setattr(ppo_arm, "arm_spec", lambda *a: state.arm)
    monkeypatch.setattr(ppo_arm, "DECISIONS_PATH", tmp_path / "adapter_decisions.jsonl")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: state.source.copy())

    def load(*args, **kwargs):
        state.loads += 1
        return model

    def partial(day, prefix):
        assert day == DAY
        assert len(prefix) == 1
        assert prefix.iloc[0]["time"] == OPEN
        state.prefixes.append(prefix.copy())
        return SimpleNamespace(date=DAY, bars_received=1, market=np.zeros((1, 25)),
                               context=np.zeros(7), spread_pips=3.0)

    monkeypatch.setattr(PPO, "load", load)
    monkeypatch.setattr(ppo_arm, "build_live_spec_partial", partial)
    for name in ("build_live_spec", "decide_weights", "decide_first_weight"):
        monkeypatch.setattr(ppo_arm, name, lambda *a, **kw: pytest.fail("batch route used"), raising=False)
    return state


def test_run_does_not_read_full_session_features_and_ignores_other_symbols(source_adapter):
    import pandas as pd

    from src.research.llm_forward.settlement_accounting import read_ledger

    state = source_adapter
    other = state.source.copy()
    other["symbol"] = "XAUUSD"
    other["close"] = 999999.0
    state.source = pd.concat([other, state.source.iloc[::-1]], ignore_index=True)
    assert ppo_arm.run(DAY, arm_id="unit_held") == 0
    assert state.loads == state.model.calls == len(state.prefixes) == 1
    rows = read_ledger(ppo_arm.DECISIONS_PATH)
    assert len(rows) == 1
    assert candidate_from_sources(rows)["decision_schedule"] == "first_bar_hold"


@pytest.mark.parametrize("case", ["missing", "duplicate", "naive"])
def test_bad_source_does_not_reach_model(source_adapter, case):
    import pandas as pd
    state = source_adapter
    if case == "missing":
        state.source = state.source.iloc[1:]
    elif case == "duplicate":
        state.source = pd.concat([state.source, state.source.iloc[:1]], ignore_index=True)
    else:
        state.source["time"] = state.source["time"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        ppo_arm.run(DAY, arm_id="unit_held")
    assert state.loads == state.model.calls == len(state.prefixes) == 0


@pytest.mark.parametrize("count", [59, 1.0, True, "1", None])
def test_batch_or_coerced_arm_never_loads_model(source_adapter, count):
    source_adapter.arm["decisions_per_session"] = count
    with pytest.raises(ValueError, match="per-bar"):
        ppo_arm.run(DAY, arm_id="unit_held")
    assert source_adapter.loads == source_adapter.model.calls == 0


@pytest.mark.parametrize("instant", [OPEN, CLOSE + timedelta(minutes=5)])
def test_call_outside_first_bar_window_never_loads_model(source_adapter, controlled_clock, instant):
    controlled_clock.now = instant
    with pytest.raises(ValueError, match="close"):
        ppo_arm.run(DAY, arm_id="unit_held")
    assert source_adapter.loads == source_adapter.model.calls == 0


def test_old_payload_without_optional_field_is_not_rehashed_or_reclassified():
    from src.research.llm_forward.settlement_accounting import check_chain
    row = held_row()
    del row["decision_schedule"]
    rehash(row)
    before = deepcopy(row)
    check_chain([row])
    with pytest.raises(ValueError):
        candidate_from_sources([row])
    assert row == before


def mixed_mode_rows():
    rows = [held_row()]
    for index in range(59):
        close = OPEN + timedelta(minutes=5 * (index + 1))
        row = held_row(
            seq=index + 1, decision_id=f"{DAY}::unit_held::b{index:02d}", bar_index=index,
            decision_schedule=None, decision_path=None, cutoff_utc=close.isoformat(),
            bar_received_at_utc=(close + timedelta(seconds=1)).isoformat(),
            emitted_at_utc=(close + timedelta(seconds=2)).isoformat(),
        )
        row["prev_hash"] = rows[-1]["record_hash"]
        row["record_hash"] = chain_hash(row["prev_hash"], {
            k: v for k, v in row.items() if k not in {"prev_hash", "record_hash"}
        })
        rows.append(row)
    return rows


@pytest.mark.parametrize("consumer", ["reader", "writer"])
def test_same_arm_cannot_be_held_and_native_even_with_valid_chains(consumer, monkeypatch, tmp_path):
    from src.research.llm_forward import settle_thesis, verify
    rows = mixed_mode_rows()
    if consumer == "reader":
        with pytest.raises(ValueError, match="schedule|modality|modalidad"):
            verify.audit_snapshot(rows, [])
    else:
        monkeypatch.setattr(settle_thesis, "DECISIONS_PATH", tmp_path / "decision.jsonl")
        monkeypatch.setattr(settle_thesis, "read_ledger", lambda path: rows if path.name == "decision.jsonl" else [])
        class RejectAppend:
            path = tmp_path / "settled.jsonl"
            def append(self, *a):
                pytest.fail("must reject before writing either modality")
        with pytest.raises(ValueError, match="schedule|modality|modalidad"):
            settle_thesis._run_locked({DAY: np.full(60, 4000.0)}, RejectAppend())


def test_held_writer_respects_stream_writer_lock_before_loading(source_adapter):
    from src.research.llm_forward.ledger import LedgerError
    lock = ppo_arm.DECISIONS_PATH.with_suffix(".jsonl.lock")
    lock.write_text("unit control owns this lock", encoding="utf-8")
    with pytest.raises(LedgerError, match="locked"):
        ppo_arm.run(DAY, arm_id="unit_held")
    assert source_adapter.loads == source_adapter.model.calls == 0
    assert lock.read_text(encoding="utf-8") == "unit control owns this lock"


@pytest.mark.parametrize("field,value", [("date", None), ("date", "2023-06-02"),
                                        ("bars_received", True), ("bars_received", 1.0)])
def test_held_prefix_identity_is_validated_before_predict(controlled_clock, field, value):
    partial = SimpleNamespace(date=DAY, bars_received=1, market=np.zeros((1, 25)),
                              context=np.zeros(7), spread_pips=3.0)
    setattr(partial, field, value)
    model = Model()
    with pytest.raises(ValueError):
        produce(model, partial=partial)
    assert model.calls == 0


def test_held_retry_is_idempotent_without_a_second_inference(source_adapter, controlled_clock):
    assert ppo_arm.run(DAY, arm_id="unit_held") == 0
    before = ppo_arm.DECISIONS_PATH.read_bytes()
    controlled_clock.now = OPEN + timedelta(days=1)
    assert ppo_arm.run(DAY, arm_id="unit_held") == 0
    assert source_adapter.model.calls == 1
    assert ppo_arm.DECISIONS_PATH.read_bytes() == before
