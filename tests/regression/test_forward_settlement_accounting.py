"""C048: accounting/admission controls, not trading experiments."""

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import numpy as np
import pytest

from src.research.llm_forward.schema import Decision, DecisionRecord
from src.research.llm_forward.settle_thesis import (
    aggregate_bar_records,
    settle_session,
    snap_to_action_space,
    weights_from_record,
)

DAY = "2023-06-01"
OPEN = datetime(2023, 6, 1, 13, tzinfo=UTC)


def daily_record(**changes):
    row = asdict(DecisionRecord(
        seq=0, decision_id=f"{DAY}::daily", session_date=DAY,
        emitted_at_utc=(OPEN - timedelta(seconds=1)).isoformat(),
        cutoff_utc=(OPEN - timedelta(seconds=2)).isoformat(),
        session_open_utc=OPEN.isoformat(), sealed_before_open=True,
        preregistration_sha256="a" * 64, prompt_sha256="b" * 64,
        model="fixture", provider="test_provider", temperature=0.0, seed=None,
        corpus=[], decision=Decision(0.5, "long", 0.5, "unit control"),
        abstained=False, abstain_reason=None, usage=None, spread_pips=3.0,
        record_hash="c" * 64,
    ))
    row.update(changes)
    return row


def stream_records():
    rows = []
    for bar in range(59):
        close = OPEN + timedelta(minutes=5 * (bar + 1))
        rows.append(daily_record(
            seq=bar, decision_id=f"{DAY}::stream_test::b{bar:02d}", bar_index=bar,
            cutoff_utc=close.isoformat(), sealed_before_open=False,
            bar_received_at_utc=(close + timedelta(seconds=1)).isoformat(),
            emitted_at_utc=(close + timedelta(seconds=2)).isoformat(),
            sealed_before_next_bar=True,
            record_hash=f"{bar + 1:064x}",
        ))
    return rows


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.5", -1.01, 1.01, None])
def test_invalid_score_never_becomes_a_position(score):
    with pytest.raises(ValueError):
        snap_to_action_space(score)


@pytest.mark.parametrize("path", [[], [0.0] * 58, [[0.0]] * 59, [True] * 59,
                                  ["0.5"] * 59, [0.2] * 59, [float("nan")] * 59])
def test_invalid_path_never_falls_back_to_daily_policy(path):
    with pytest.raises(ValueError):
        weights_from_record(daily_record(decision_path=path))


@pytest.mark.parametrize("closes", [[4000.0] * 59, [[4000.0]] * 60, [True] * 60,
                                    ["4000"] * 60, [float("nan")] * 60,
                                    [float("inf")] * 60, [0.0] * 60, [-1.0] * 60])
def test_prices_have_strict_positive_numeric_domain(closes):
    with pytest.raises(ValueError):
        settle_session(daily_record(), closes)


@pytest.mark.parametrize("spread", [True, "3", -1, float("nan"), float("inf")])
def test_spread_cannot_be_coerced_or_nonfinite(spread):
    with pytest.raises(ValueError):
        settle_session(daily_record(spread_pips=spread), np.full(60, 4000.0))


@pytest.mark.parametrize(("field", "value"), [
    ("bar_index", 1.0), ("bar_index", True), ("bar_index", "1"),
    ("spread_pips", None), ("spread_pips", -1), ("spread_pips", True),
    ("provider", "stub"), ("provider", "different"), ("model", "other_model"),
    ("preregistration_sha256", "d" * 64), ("sealed_before_next_bar", "False"),
    ("sealed_before_next_bar", None), ("abstained", "False"),
    ("emitted_at_utc", "2030-01-01T13:00:00+00:00"),
    ("cutoff_utc", "2023-06-01T13:10:00"), ("record_hash", ""),
    ("decision", {"score": float("nan")}), ("decision", {"score": True}),
    ("decision", {"score": 0.3}), ("decision_path", []),
])
def test_stream_rejects_semantic_corruption(field, value):
    rows = stream_records()
    rows[1][field] = value
    assert aggregate_bar_records(rows) is None


def test_stream_does_not_silently_discard_extra_daily_row():
    assert aggregate_bar_records([*stream_records(), daily_record()]) is None


def test_valid_complete_stream_keeps_all_source_references():
    rows = stream_records()
    original = deepcopy(rows)
    aggregate = aggregate_bar_records(list(reversed(rows)))
    assert aggregate is not None
    assert aggregate["source_decisions"] == [
        {"decision_id": row["decision_id"], "record_hash": row["record_hash"]}
        for row in rows
    ]
    assert rows == original


def test_daily_optional_none_preserves_honest_before_open_seal():
    from src.research.llm_forward.settlement_accounting import sealed
    row = daily_record()
    assert row["sealed_before_next_bar"] is None
    assert sealed(row)
    row["emitted_at_utc"] = OPEN.isoformat()
    assert not sealed(row)


def accounting_fixture():
    from src.research.llm_forward.canonical import GENESIS_HASH, chain_hash
    from src.research.llm_forward.settlement_accounting import build_accounting
    row = daily_record()
    row["prev_hash"] = GENESIS_HASH
    row["record_hash"] = chain_hash(GENESIS_HASH, {
        k: v for k, v in row.items() if k not in {"prev_hash", "record_hash"}
    })
    closes = np.linspace(4000.0, 4100.0, 60)
    accounting = build_accounting(row, closes, [row])
    outer = {
        "decision_id": row["decision_id"], "session_date": DAY,
        "settled_at_utc": (OPEN + timedelta(hours=5)).isoformat(),
        "open_price": 4000.0, "close_price": 4100.0,
        "realized_return": round(4100.0 / 4000.0 - 1, 8),
        "signed_return": round(accounting["gross_return"], 8),
        "bars_observed": 60, "accounting": accounting,
    }
    return outer, [row]


def test_accounting_replays_and_preserves_unknown_api_cost():
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer, rows = accounting_fixture()
    account = outer["accounting"]
    assert validate_accounting(outer, rows) == account["daily_return"]
    assert account["daily_return"] == account["gross_return"] - account["total_cost"]
    assert len(account["costs"]) == 60
    assert account["costs"][-1] == account["terminal_cost"] > 0
    assert account["api_cost"]["total_usd"] is None
    assert account["api_cost"]["unknown_decisions"] == 1
    assert account["evidence"] == "paper_assumed_costs"


@pytest.mark.parametrize("field", ["gross_return", "daily_return", "total_cost", "terminal_cost",
                                  "n_changes", "mean_abs_exposure", "sum_abs_dw"])
def test_accounting_rejects_scalar_tampering(field):
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer, rows = accounting_fixture()
    outer["accounting"][field] += 0.01
    with pytest.raises(ValueError):
        validate_accounting(outer, rows)


@pytest.mark.parametrize(("field", "value"), [
    ("signed_return", 0), ("realized_return", 0), ("open_price", 3999.0),
    ("close_price", 4101.0), ("bars_observed", True), ("session_date", "2023-06-02"),
    ("decision_id", "2023-06-01::other"), ("settled_at_utc", OPEN.isoformat()),
])
def test_outer_record_cannot_disagree_with_inner_accounting(field, value):
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer, rows = accounting_fixture()
    outer[field] = value
    with pytest.raises(ValueError):
        validate_accounting(outer, rows)


def test_source_reference_is_checked_against_original_decision():
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer, rows = accounting_fixture()
    rows[0]["decision"]["score"] = -0.5
    with pytest.raises(ValueError):
        validate_accounting(outer, rows)


@pytest.mark.parametrize("payload", [None, "absent"])
def test_legacy_gross_is_not_net(payload):
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer = {"signed_return": 0.02}
    if payload is None:
        outer["accounting"] = None
    assert validate_accounting(outer, []) is None


def write_daily(tmp_path, monkeypatch, rows=None):
    from src.research.llm_forward import settle_thesis, verify
    from src.research.llm_forward.ledger import Ledger
    decisions = Ledger(tmp_path / "decisions.jsonl", "decision_id")
    settlements = Ledger(tmp_path / "settlements.jsonl", "decision_id")
    for module in (settle_thesis, verify):
        monkeypatch.setattr(module, "DECISIONS_PATH", decisions.path)
        monkeypatch.setattr(module, "SETTLEMENTS_PATH", settlements.path)
    for row in rows or [daily_record()]:
        decisions.append(DecisionRecord(**row))
    return decisions, settlements


def test_daily_write_read_net_and_rerun_is_idempotent(tmp_path, monkeypatch):
    from src.research.llm_forward import settle_thesis, verify
    decisions, settlements = write_daily(tmp_path, monkeypatch)
    assert settle_thesis.run({DAY: np.linspace(4000, 4100, 60)}) == 0
    rows = list(settlements)
    assert len(rows) == 1
    assert rows[0]["accounting"]["daily_return"] < rows[0]["signed_return"]
    original = settlements.path.read_bytes()
    assert settle_thesis.run({DAY: np.linspace(4000, 4100, 60)}) == 0
    assert original == settlements.path.read_bytes()
    audit = verify.audit_snapshot(list(decisions), rows)
    assert audit["net_verified_sessions"] == 1
    assert audit["daily_returns"][0]["net_return"] == rows[0]["accounting"]["daily_return"]
    assert verify.main() == 0


def test_verify_does_not_mistake_observation_hashes_for_prompt_versions(tmp_path, monkeypatch, capsys):
    from src.research.llm_forward import settle_thesis, verify
    rows = stream_records()
    for bar, row in enumerate(rows):
        row["prompt_sha256"] = sha256(str(bar).encode()).hexdigest()
    write_daily(tmp_path, monkeypatch, rows)
    assert settle_thesis.run({DAY: np.full(60, 4000.0)}) == 0
    assert verify.main() == 0
    output = capsys.readouterr().out
    assert "treatment changed" not in output.lower()
    assert "net_verified_sessions" in output


def test_preflight_invalid_second_candidate_writes_neither(tmp_path, monkeypatch):
    from src.research.llm_forward import settle_thesis
    rows = [daily_record(), daily_record(decision_id=f"{DAY}::second", spread_pips=-1)]
    _, ledger = write_daily(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError):
        settle_thesis.run({DAY: np.full(60, 4000.0)})
    assert list(ledger) == []


def test_planned_daily_cutoff_can_equal_open_without_future_documents():
    from src.research.llm_forward.settlement_accounting import sealed
    row = daily_record(cutoff_utc=OPEN.isoformat())
    assert sealed(row)


def test_actual_future_document_is_excluded_even_with_true_flag():
    from src.research.llm_forward.settlement_accounting import sealed
    row = daily_record(corpus=[{"published_at_utc": OPEN.isoformat()}])
    assert not sealed(row)


@pytest.mark.parametrize("field", ["closes", "weights", "gross_bars", "costs"])
def test_payload_vector_tampering_is_detected(field):
    from src.research.llm_forward.settlement_accounting import validate_accounting
    outer, rows = accounting_fixture()
    outer["accounting"][field][0] += 0.01
    with pytest.raises(ValueError):
        validate_accounting(outer, rows)


def test_constant_price_accounting_has_independent_analytic_oracle():
    from src.research.llm_forward.settlement_accounting import build_accounting
    _, rows = accounting_fixture()
    result = build_accounting(rows[0], np.full(60, 4000.0), rows)
    expected_each_side = 0.5 * (3.0 / 2 + 0.5) / 4000.0
    assert result["gross_return"] == 0
    assert result["total_cost"] == 2 * expected_each_side
    assert result["terminal_cost"] == expected_each_side
    assert result["daily_return"] == -2 * expected_each_side
    assert result["sum_abs_dw"] == 1.0


@pytest.mark.parametrize("spread", [None, -3.0])
def test_admissible_count_requires_the_same_cost_gate_as_writer(tmp_path, monkeypatch, spread):
    from src.research.llm_forward import verify
    decisions, _ = write_daily(tmp_path, monkeypatch, [daily_record(spread_pips=spread)])
    result = verify.audit_snapshot(list(decisions), [])
    assert result["admissible_session_arms"] == 0
    assert result["excluded_decision_records"] == 1


def test_old_raw_row_without_accounting_survives_new_append(tmp_path, monkeypatch):
    import json

    from src.research.llm_forward import settle_thesis, verify
    from src.research.llm_forward.canonical import GENESIS_HASH, chain_hash
    from src.research.llm_forward.schema import SettlementRecord

    decisions, ledger = write_daily(tmp_path, monkeypatch, [
        daily_record(), daily_record(decision_id=f"{DAY}::second")
    ])
    legacy = asdict(SettlementRecord(0, f"{DAY}::daily", DAY,
        "2023-06-01T18:00:00+00:00", 4000.0, 4000.0, 0.0, 0.0, 60))
    legacy.pop("accounting")
    legacy["prev_hash"] = GENESIS_HASH
    legacy["record_hash"] = chain_hash(GENESIS_HASH, {
        k: v for k, v in legacy.items() if k not in {"prev_hash", "record_hash"}
    })
    ledger.path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    original = ledger.path.read_bytes()
    assert settle_thesis.run({DAY: np.full(60, 4000.0)}) == 0
    assert ledger.path.read_bytes().startswith(original)
    result = verify.audit_snapshot(list(decisions), list(ledger))
    assert result["net_verified_sessions"] == 1
    assert result["legacy_net_unknown"] == 1


@pytest.mark.parametrize("changes", [{"provider": "stub"}, {"abstained": True},
                                   {"emitted_at_utc": OPEN.isoformat()}])
def test_excluded_decisions_are_not_settled_as_zero(tmp_path, monkeypatch, changes):
    from src.research.llm_forward import settle_thesis
    _, ledger = write_daily(tmp_path, monkeypatch, [daily_record(**changes)])
    assert settle_thesis.run({DAY: np.full(60, 4000.0)}) == 0
    assert list(ledger) == []


def test_missing_legacy_accounting_is_reported_not_used_as_net(tmp_path, monkeypatch):
    from src.research.llm_forward import verify
    from src.research.llm_forward.schema import SettlementRecord
    decisions, ledger = write_daily(tmp_path, monkeypatch)
    ledger.append(SettlementRecord(
        -1, f"{DAY}::daily", DAY, "2023-06-01T18:00:00+00:00",
        4000.0, 4100.0, 0.025, 0.0125, 60,
    ))
    result = verify.audit_snapshot(list(decisions), list(ledger))
    assert result["legacy_net_unknown"] == 1
    assert result["daily_returns"] == []
    assert result["net_verified_sessions"] == 0


@pytest.mark.parametrize("decision", [None, {"score": True, "direction": "long", "confidence": 0.5},
    {"score": -1.0, "direction": "short", "confidence": 0.5},
    {"score": 0.5, "direction": "short", "confidence": 0.5}])
def test_path_summary_must_match_its_first_position(tmp_path, monkeypatch, decision):
    from src.research.llm_forward import settle_thesis
    _, ledger = write_daily(tmp_path, monkeypatch, [
        daily_record(decision=decision, decision_path=[0.5] * 59)
    ])
    with pytest.raises(ValueError):
        settle_thesis.run({DAY: np.full(60, 4000.0)})
    assert list(ledger) == []


def test_duplicate_json_key_rejected_even_if_last_value_matches_hash(tmp_path, monkeypatch):
    from src.research.llm_forward import settle_thesis, verify
    decisions, ledger = write_daily(tmp_path, monkeypatch)
    original = decisions.path.read_text(encoding="utf-8")
    decisions.path.write_text(original.replace('"spread_pips": 3.0',
        '"spread_pips": 999.0, "spread_pips": 3.0'), encoding="utf-8")
    assert decisions.path.read_text(encoding="utf-8") != original
    with pytest.raises(ValueError):
        settle_thesis.run({DAY: np.full(60, 4000.0)})
    assert verify.main() == 1
    assert list(ledger) == []


def test_source_cannot_impersonate_trusted_aggregate(tmp_path, monkeypatch):
    import json

    from src.research.llm_forward import settle_thesis
    from src.research.llm_forward.canonical import chain_hash
    decisions, ledger = write_daily(tmp_path, monkeypatch, stream_records())
    rows = list(decisions)
    forged = dict(rows[0])
    forged.update(seq=59, decision_id=f"{DAY}::stream_test::stream", bar_index=None,
                  decision_path=[0.5] * 59, source_decisions=[
                      {"decision_id": r["decision_id"], "record_hash": r["record_hash"]}
                      for r in rows], prev_hash=rows[-1]["record_hash"])
    forged["record_hash"] = chain_hash(forged["prev_hash"], {
        k: v for k, v in forged.items() if k not in {"prev_hash", "record_hash"}
    })
    with decisions.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(forged) + "\n")
    with pytest.raises(ValueError):
        settle_thesis.run({DAY: np.full(60, 4000.0)})
    assert list(ledger) == []
