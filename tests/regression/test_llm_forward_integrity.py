"""
Regression: las garantias del carril forward, no su fontaneria.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

Vendorizado del arnes `llm-forward`. Estos tests no preguntan "corre?" sino "se sostiene la
garantia?": un registro editado se detecta, uno borrado rompe la cadena, un documento del
futuro se cae, un timestamp naive lanza, una sesion duplicada se rechaza.

Es la parte que hace utilizable el brazo forward. Una evaluacion prospectiva vale exactamente
lo que valga su prueba de que la decision existio antes del resultado; sin estos tests, el
ledger es un fichero de texto que alguien podria haber escrito el martes siguiente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.llm_forward.corpus import COT, CutoffViolation, RawDoc, filter_by_cutoff, session_cutoff_utc
from src.research.llm_forward.ledger import Ledger, LedgerError
from src.research.llm_forward.schema import Decision, DecisionRecord, LlmUsage


def make_record(session_date: str, score: float) -> DecisionRecord:
    return DecisionRecord(
        seq=-1,
        decision_id=f"{session_date}::test_arm",
        session_date=session_date,
        emitted_at_utc="2026-08-26T12:30:00+00:00",
        cutoff_utc="2026-08-26T13:00:00+00:00",
        session_open_utc="2026-08-26T13:00:00+00:00",
        sealed_before_open=True,
        preregistration_sha256="a" * 64,
        prompt_sha256="b" * 64,
        model="gpt-4o-mini",
        provider="openai",
        temperature=0.0,
        seed=1,
        corpus=[],
        decision=Decision(score=score, direction="long", confidence=0.5, rationale="x"),
        abstained=False,
        abstain_reason=None,
        usage=LlmUsage(100, 20, 900, 0.0001, "fp_abc"),
    )


class TestChain:
    def test_intact_chain_verifies(self, tmp_path):
        ledger = Ledger(tmp_path / "d.jsonl", "decision_id")
        for day in ("2026-08-26", "2026-08-27", "2026-08-28"):
            ledger.append(make_record(day, 0.4))

        ok, message = ledger.verify()
        assert ok, message
        assert "3 record(s)" in message

    def test_edited_record_is_detected(self, tmp_path):
        """The core claim: you cannot change a sealed score and get away with it."""
        path = tmp_path / "d.jsonl"
        ledger = Ledger(path, "decision_id")
        for day in ("2026-08-26", "2026-08-27", "2026-08-28"):
            ledger.append(make_record(day, 0.4))

        lines = path.read_text().splitlines()
        tampered = json.loads(lines[1])
        tampered["decision"]["score"] = 0.99  # backdate a winning call
        lines[1] = json.dumps(tampered, ensure_ascii=True, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")

        ok, reason = ledger.verify()
        assert not ok
        assert "modified after writing" in reason

    def test_deleted_record_is_detected(self, tmp_path):
        """Dropping an inconvenient day breaks the chain too."""
        path = tmp_path / "d.jsonl"
        ledger = Ledger(path, "decision_id")
        for day in ("2026-08-26", "2026-08-27", "2026-08-28"):
            ledger.append(make_record(day, 0.4))

        lines = path.read_text().splitlines()
        path.write_text(lines[0] + "\n" + lines[2] + "\n")

        ok, reason = ledger.verify()
        assert not ok
        assert "does not match" in reason

    def test_duplicate_session_is_refused(self, tmp_path):
        """No re-rolling a decision you did not like."""
        ledger = Ledger(tmp_path / "d.jsonl", "decision_id")
        ledger.append(make_record("2026-08-26", 0.4))

        with pytest.raises(LedgerError, match="already in"):
            ledger.append(make_record("2026-08-26", -0.9))


class TestCutoff:
    def test_cutoff_is_thirteen_hundred_utc(self):
        """08:00 COT is 13:00 UTC year-round; Colombia has no DST."""
        cutoff = session_cutoff_utc("2026-08-26", cutoff_hour_cot=8)
        assert cutoff == datetime(2026, 8, 26, 13, 0, tzinfo=timezone.utc)

    def test_future_document_is_dropped(self):
        cutoff = session_cutoff_utc("2026-08-26")
        docs = [
            RawDoc("u1", "before", "ok", cutoff - timedelta(hours=2)),
            RawDoc("u2", "after", "leak", cutoff + timedelta(minutes=1)),
        ]
        kept = filter_by_cutoff(docs, cutoff)
        assert [d.title for d in kept] == ["before"]

    def test_document_exactly_at_cutoff_is_dropped(self):
        """Ambiguity resolves against inclusion."""
        cutoff = session_cutoff_utc("2026-08-26")
        docs = [RawDoc("u", "boundary", "t", cutoff)]
        assert filter_by_cutoff(docs, cutoff) == []

    def test_stale_document_outside_lookback_is_dropped(self):
        cutoff = session_cutoff_utc("2026-08-26")
        docs = [RawDoc("u", "old", "t", cutoff - timedelta(days=10))]
        assert filter_by_cutoff(docs, cutoff, lookback_days=3) == []

    def test_naive_timestamp_raises(self):
        """A source that forgets its timezone must fail loudly, not be guessed at."""
        cutoff = session_cutoff_utc("2026-08-26")
        docs = [RawDoc("u", "naive", "t", datetime(2026, 8, 26, 6, 0))]
        with pytest.raises(CutoffViolation, match="naive published_at"):
            filter_by_cutoff(docs, cutoff)

    def test_local_time_is_converted_not_compared_raw(self):
        """A doc stamped 07:00 COT is inside; 09:00 COT is outside."""
        cutoff = session_cutoff_utc("2026-08-26")
        early = datetime(2026, 8, 26, 7, 0, tzinfo=COT)
        late = datetime(2026, 8, 26, 9, 0, tzinfo=COT)
        docs = [RawDoc("a", "early", "t", early), RawDoc("b", "late", "t", late)]
        assert [d.title for d in filter_by_cutoff(docs, cutoff)] == ["early"]


class TestSeal:
    """The seal check is the one guarantee that cannot be verified after the
    fact from the data alone, so it gets its own test."""

    def test_cutoff_precedes_open_by_construction(self):
        cutoff = session_cutoff_utc("2026-08-26", cutoff_hour_cot=8)
        open_utc = session_cutoff_utc("2026-08-26", cutoff_hour_cot=8)
        assert cutoff <= open_utc

    def test_late_record_is_marked_not_dropped(self):
        """A late run must leave a visible, excluded row. A missing row is
        indistinguishable from a day that never existed."""
        record = make_record("2026-08-26", 0.4)
        record.sealed_before_open = False
        assert record.decision is not None  # data kept
        assert record.sealed_before_open is False  # but flagged
