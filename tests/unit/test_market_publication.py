from __future__ import annotations

from datetime import datetime, timezone

from src.market.publication import _source_hash, _structurally_representable


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
