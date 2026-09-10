from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data_quality import corrections
from src.data_quality.corrections import (
    CorrectionRequest,
    MarketCorrectionError,
    QuarantineContext,
    apply_market_correction,
)
from src.market.publication import MarketPublicationResult


QID = "11111111-1111-4111-8111-111111111111"
ORIGINAL_TIME = datetime(1990, 1, 2, 13, tzinfo=timezone.utc)


class RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    class Cursor:
        def __init__(self, owner) -> None:
            self.owner = owner

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement, _params=None) -> None:
            self.owner.statements.append(" ".join(statement.split()))

    def cursor(self):
        return self.Cursor(self)


def _context(**changes) -> QuarantineContext:
    values = {
        "quarantine_id": QID,
        "status": "OPEN",
        "correction_event_id": None,
        "provider_id": "twelvedata",
        "provider_symbol": "USD/MXN",
        "interval_id": "PT5M",
        "observed_at": ORIGINAL_TIME,
        "source_uri": "dag://l0_ohlcv_backfill/USD/MXN",
        "source_record": {"time": ORIGINAL_TIME.isoformat(), "open": 180.0},
        "context_version": 1,
    }
    values.update(changes)
    return QuarantineContext(**values)


def _request(**changes) -> CorrectionRequest:
    values = {
        "revision_type": "PROVIDER_CORRECTION",
        "corrected_record": {
            "time": "1990-01-02T13:00:00Z",
            "open": 18.0,
            "high": 18.1,
            "low": 17.9,
            "close": 18.0,
            "volume": 1,
        },
        "reason": "provider decimal correction",
        "corrected_by": "operator@example",
        "compared_provider_id": "banxico",
    }
    values.update(changes)
    return CorrectionRequest(**values)


def test_correction_uses_original_rule_time_and_updates_after_canonical(monkeypatch) -> None:
    conn = RecordingConnection()
    calls: list[dict] = []
    events: list[str] = []
    monkeypatch.setattr(corrections, "_lock_quarantine", lambda *_args: _context())

    def publish(_conn, **kwargs):
        calls.append(kwargs)
        events.append("canonical")
        return MarketPublicationResult(
            accepted=(kwargs["rows"][0],),
            raw_count=1,
            canonical_count=1,
            quarantine_count=0,
        )

    monkeypatch.setattr(corrections, "publish_provider_rows", publish)
    monkeypatch.setattr(
        corrections,
        "_insert_correction_event",
        lambda *_args, **_kwargs: events.append("correction"),
    )
    monkeypatch.setattr(
        corrections,
        "_mark_corrected",
        lambda *_args, **_kwargs: events.append("resolved"),
    )

    result = apply_market_correction(conn, quarantine_id=QID, request=_request())

    assert result.canonical_count == 1 and not result.idempotent
    assert calls[0]["quality_observed_at"] == ORIGINAL_TIME
    assert calls[0]["observed_at"] > datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert events == ["canonical", "correction", "resolved"]
    assert conn.statements == [
        "SAVEPOINT c027_market_correction",
        "RELEASE SAVEPOINT c027_market_correction",
    ]


def test_invalid_correction_rolls_back_to_savepoint_with_zero_followup_writes(
    monkeypatch,
) -> None:
    conn = RecordingConnection()
    events: list[str] = []
    monkeypatch.setattr(corrections, "_lock_quarantine", lambda *_args: _context())
    monkeypatch.setattr(
        corrections,
        "publish_provider_rows",
        lambda *_args, **_kwargs: MarketPublicationResult((), 1, 0, 1),
    )
    monkeypatch.setattr(
        corrections,
        "_insert_correction_event",
        lambda *_args, **_kwargs: events.append("correction"),
    )
    monkeypatch.setattr(
        corrections,
        "_mark_corrected",
        lambda *_args, **_kwargs: events.append("resolved"),
    )

    with pytest.raises(MarketCorrectionError, match="exactly one"):
        apply_market_correction(conn, quarantine_id=QID, request=_request())

    assert events == []
    assert conn.statements == [
        "SAVEPOINT c027_market_correction",
        "ROLLBACK TO SAVEPOINT c027_market_correction",
        "RELEASE SAVEPOINT c027_market_correction",
    ]


def test_identical_retry_is_idempotent_but_different_correction_conflicts(monkeypatch) -> None:
    conn = RecordingConnection()
    request = _request()
    expected_id = corrections._correction_id(QID, request)
    monkeypatch.setattr(
        corrections,
        "_lock_quarantine",
        lambda *_args: _context(status="CORRECTED", correction_event_id=expected_id),
    )
    monkeypatch.setattr(
        corrections,
        "publish_provider_rows",
        lambda *_args, **_kwargs: pytest.fail("idempotent retry must not republish"),
    )

    result = apply_market_correction(conn, quarantine_id=QID, request=request)
    assert result.idempotent and result.correction_event_id == expected_id

    other = _request(corrected_record={**request.corrected_record, "close": 19.0})
    with pytest.raises(MarketCorrectionError, match="different correction"):
        apply_market_correction(conn, quarantine_id=QID, request=other)


def test_contextless_legacy_and_provider_correction_without_evidence_fail_closed(
    monkeypatch,
) -> None:
    with pytest.raises(MarketCorrectionError, match="compared_provider_id"):
        _request(compared_provider_id=None).validate()

    conn = RecordingConnection()
    monkeypatch.setattr(
        corrections,
        "_lock_quarantine",
        lambda *_args: _context(context_version=None, observed_at=None),
    )
    with pytest.raises(MarketCorrectionError, match="no typed replay context"):
        apply_market_correction(conn, quarantine_id=QID, request=_request())
    assert "ROLLBACK TO SAVEPOINT c027_market_correction" in conn.statements


def test_operator_cli_never_commits_an_exception_path(monkeypatch, tmp_path) -> None:
    from scripts.ops import resolve_market_quarantine as cli

    record_path = tmp_path / "corrected.json"
    record_path.write_text(
        '{"time":"1990-01-02T13:00:00Z","open":18,"high":18.1,'
        '"low":17.9,"close":18,"volume":1}',
        encoding="utf-8",
    )
    events: list[str] = []

    class Connection:
        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "_db_conn", lambda: Connection())
    monkeypatch.setattr(
        cli,
        "apply_market_correction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        cli.main(
            [
                QID,
                "--revision-type",
                "PROVIDER_CORRECTION",
                "--corrected-record",
                str(record_path),
                "--reason",
                "provider decimal correction",
                "--actor",
                "operator@example",
                "--compared-provider",
                "banxico",
            ]
        )

    assert events == ["rollback", "close"]
