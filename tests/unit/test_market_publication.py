from __future__ import annotations

import inspect
from datetime import datetime, timezone

from src.data_quality.rules import QualityDecision
from src.market import publication
from src.market.publication import _source_hash, _structurally_representable


def test_publish_provider_rows_requires_source_uri_without_default() -> None:
    parameter = inspect.signature(publication.publish_provider_rows).parameters["source_uri"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def _row(**changes):
    row = {
        "time": datetime(2026, 8, 4, 15, tzinfo=timezone.utc),
        "open": 4100.0,
        "high": 4110.0,
        "low": 4090.0,
        "close": 4105.0,
        "volume": 2.0,
    }
    row.update(changes)
    return row


def test_raw_representability_matches_applied_schema_constraints() -> None:
    assert _structurally_representable(_row())
    assert not _structurally_representable(_row(high=4080.0))
    assert not _structurally_representable(_row(volume=-1.0))
    assert not _structurally_representable(_row(open=float("nan")))
    assert not _structurally_representable(_row(time=datetime(2026, 8, 4)))


def test_source_hash_is_stable_and_commits_provider_identity() -> None:
    row = _row()
    first = _source_hash(
        row,
        provider_id="twelvedata",
        provider_symbol="USD/COP",
        interval_id="PT5M",
    )
    replay = _source_hash(
        dict(reversed(list(row.items()))),
        provider_id="twelvedata",
        provider_symbol="USD/COP",
        interval_id="PT5M",
    )
    other_provider = _source_hash(
        row,
        provider_id="other",
        provider_symbol="USD/COP",
        interval_id="PT5M",
    )

    assert first == replay
    assert first != other_provider
    assert first.startswith("sha256:")


def test_scoped_quality_uses_each_rows_event_time_not_batch_retrieval_time(
    monkeypatch,
) -> None:
    pre_cutoff = _row(time=datetime(1990, 1, 2, 13, tzinfo=timezone.utc), open=18.0,
                      high=18.1, low=17.9, close=18.0)
    post_cutoff = _row(time=datetime(2026, 8, 4, 15, tzinfo=timezone.utc), open=18.0,
                       high=18.1, low=17.9, close=18.0)
    evaluated_at: list[datetime] = []
    canonical_rows: list[dict] = []
    quarantined: list[dict] = []

    class Rules:
        version = "scoped-test-v1"

        def evaluate_provider_bar(self, _provider, _symbol, row, *, observed_at):
            evaluated_at.append(observed_at)
            if observed_at.year < 1993:
                return QualityDecision(
                    False,
                    "QUARANTINED",
                    "bar.range_scope",
                    row,
                    "no historical scoped range",
                )
            return QualityDecision(True, "VALID")

    monkeypatch.setattr(publication, "ruleset_from_spine", lambda _conn: Rules())
    monkeypatch.setattr(
        publication, "resolved_instrument_id", lambda *_args: "instrument-usdmxn"
    )
    monkeypatch.setattr(
        publication,
        "_insert_raw",
        lambda _conn, **kwargs: "raw-" + kwargs["row"]["time"].isoformat(),
    )
    monkeypatch.setattr(
        publication,
        "_insert_canonical",
        lambda _conn, **kwargs: canonical_rows.append(kwargs["row"]),
    )
    monkeypatch.setattr(
        publication,
        "record_quarantine",
        lambda _conn, **kwargs: quarantined.append(kwargs),
    )

    result = publication.publish_provider_rows(
        object(),
        provider_id="twelvedata",
        provider_symbol="USD/MXN",
        interval_id="PT5M",
        rows=[pre_cutoff, post_cutoff],
        source_uri="dag://l0_ohlcv_backfill/USD/MXN?job=twelvedata_backfill",
        observed_at=datetime(2026, 8, 4, 16, tzinfo=timezone.utc),
    )

    assert evaluated_at == [pre_cutoff["time"], post_cutoff["time"]]
    assert result.accepted == (post_cutoff,)
    assert result.raw_count == 2
    assert result.canonical_count == 1
    assert result.quarantine_count == 1
    assert canonical_rows == [post_cutoff]
    assert [item["row"] for item in quarantined] == [pre_cutoff]
    assert quarantined[0]["interval_id"] == "PT5M"
    assert quarantined[0]["observed_at"] == pre_cutoff["time"]
    assert quarantined[0]["source_uri"].startswith("dag://l0_ohlcv_backfill/")


def test_structural_quarantine_carries_the_same_typed_context(monkeypatch) -> None:
    malformed = _row(high=4080.0)
    quarantined: list[dict] = []

    class Rules:
        version = "scoped-test-v1"

        def evaluate_provider_bar(self, *_args, observed_at, **_kwargs):
            return QualityDecision(
                False, "QUARANTINED", "bar.ohlc_order", {}, "invalid OHLC"
            )

    monkeypatch.setattr(publication, "ruleset_from_spine", lambda _conn: Rules())
    monkeypatch.setattr(
        publication, "resolved_instrument_id", lambda *_args: "instrument-usdmxn"
    )
    monkeypatch.setattr(
        publication,
        "record_quarantine",
        lambda _conn, **kwargs: quarantined.append(kwargs),
    )

    result = publication.publish_provider_rows(
        object(),
        provider_id="twelvedata",
        provider_symbol="USD/MXN",
        interval_id="PT5M",
        rows=[malformed],
        source_uri="dag://l0_ohlcv_realtime/USD/MXN?job=twelvedata_multi",
        observed_at=datetime(2026, 8, 4, 16, tzinfo=timezone.utc),
    )

    assert result.quarantine_count == 1
    assert len(quarantined) == 1
    assert quarantined[0]["interval_id"] == "PT5M"
    assert quarantined[0]["observed_at"] == malformed["time"]
    assert quarantined[0]["source_uri"].startswith("dag://l0_ohlcv_realtime/")


def test_governed_correction_can_preserve_the_original_quality_instant(monkeypatch) -> None:
    row = _row(time=datetime(2026, 8, 4, 15, tzinfo=timezone.utc))
    original_instant = datetime(1990, 1, 2, 13, tzinfo=timezone.utc)
    seen: list[datetime] = []

    class Rules:
        version = "scoped-test-v1"

        def evaluate_provider_bar(self, *_args, observed_at, **_kwargs):
            seen.append(observed_at)
            return QualityDecision(True, "VALID")

    monkeypatch.setattr(publication, "ruleset_from_spine", lambda _conn: Rules())
    monkeypatch.setattr(
        publication, "resolved_instrument_id", lambda *_args: "instrument-usdmxn"
    )
    monkeypatch.setattr(publication, "_insert_raw", lambda *_args, **_kwargs: "raw-id")
    monkeypatch.setattr(publication, "_insert_canonical", lambda *_args, **_kwargs: None)

    publication.publish_provider_rows(
        object(),
        provider_id="twelvedata",
        provider_symbol="USD/MXN",
        interval_id="PT5M",
        rows=[row],
        source_uri="ops://market_correction/test",
        observed_at=datetime(2026, 8, 4, 16, tzinfo=timezone.utc),
        quality_observed_at=original_instant,
    )

    assert seen == [original_instant]
