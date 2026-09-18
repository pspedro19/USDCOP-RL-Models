import json
import hashlib
from datetime import date
from types import SimpleNamespace

import numpy as np

from scripts.analysis import settle_thesis_llm as settlement


def _portable(session):
    return SimpleNamespace(block=lambda name: [session])


def _write_ledger(path, n=59, weight=0.0):
    with path.open("w", encoding="utf-8") as handle:
        for bar in range(n):
            handle.write(json.dumps({
                "session_date": "2026-01-05", "bar": bar, "weight": weight,
                "valid_json": True,
            }) + "\n")


def test_complete_flat_session_uses_canonical_accounting(tmp_path, monkeypatch):
    session = SimpleNamespace(date=date(2026, 1, 5), close=np.full(60, 4000.0), spread_pips=3.0)
    monkeypatch.setattr(settlement, "load_portable", lambda path: _portable(session))
    ledger = tmp_path / "decisions.jsonl"
    _write_ledger(ledger)

    report = settlement.settle(ledger, "selection", tmp_path / "portable.pkl")

    assert report["n_sessions_settled"] == 1
    assert report["n_sessions_excluded"] == 0
    assert report["sessions"][0]["daily_return"] == 0.0
    assert report["compounded_return"] == 0.0


def test_incomplete_session_is_excluded_not_zero_filled(tmp_path, monkeypatch):
    session = SimpleNamespace(date=date(2026, 1, 5), close=np.full(60, 4000.0), spread_pips=3.0)
    monkeypatch.setattr(settlement, "load_portable", lambda path: _portable(session))
    ledger = tmp_path / "decisions.jsonl"
    _write_ledger(ledger, n=58)

    report = settlement.settle(ledger, "selection", tmp_path / "portable.pkl")

    assert report["n_sessions_settled"] == 0
    assert report["n_sessions_excluded"] == 1
    assert report["excluded"]["2026-01-05"] == "incomplete_59_decisions"


def test_strict_settlement_binds_ledger_to_portable_hash(tmp_path, monkeypatch):
    session = SimpleNamespace(date=date(2026, 1, 5), close=np.full(60, 4000.0), spread_pips=3.0)
    monkeypatch.setattr(settlement, "load_portable", lambda path: _portable(session))
    portable = tmp_path / "portable.pkl"
    portable.write_bytes(b"portable-v2")
    digest = hashlib.sha256(portable.read_bytes()).hexdigest()
    ledger = tmp_path / "decisions.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        for bar in range(59):
            handle.write(json.dumps({
                "decision_id": f"2026-01-05::llm::{bar}",
                "session_date": "2026-01-05", "bar": bar, "weight": 0.0,
                "valid_json": True, "prompt_hash": "a" * 64,
                "raw_response_sha256": "b" * 64, "model_id": "deepseek-chat",
                "prompt_version": "thesis-llm-trader-v1", "max_tokens": 256,
                "temperature": 0.1, "top_p": 0.9, "dataset_sha256": digest,
            }) + "\n")
    report = settlement.settle(ledger, "selection", portable, strict_ledger=True)
    assert report["n_sessions_settled"] == 1
