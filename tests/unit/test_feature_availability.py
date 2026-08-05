from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.data_quality.feature_availability import (
    FeatureMeasurement,
    FeatureSpec,
    load_feature_specs,
    load_feature_max_age,
    measure_feature,
    persist_measurement,
    news_feature_cutoff,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTOFF = datetime(2026, 8, 4, 12, tzinfo=UTC)


class FakeCursor:
    def __init__(self, results):
        self.results = list(results)
        self.executions = []
        self.current = None

    def execute(self, query, params=()):
        self.executions.append((query, params))
        self.current = self.results.pop(0) if self.results else None

    def fetchall(self):
        return self.current

    def fetchone(self):
        return self.current


def _spec() -> FeatureSpec:
    return FeatureSpec(
        feature_id="news_articles.sentiment_score",
        table="news_articles",
        column="sentiment_score",
        time_column="published_at",
    )


def test_repository_registry_contains_only_verified_ghost_features() -> None:
    specs = load_feature_specs(REPO_ROOT / "config/quality/feature_availability.yaml")
    assert {spec.feature_id for spec in specs} == {
        "macro_banrep_forwards_monthly.forward_rate",
        "crypto_derivatives_daily.liquidations_usd",
        "crypto_derivatives_daily.open_interest",
        "crypto_derivatives_daily.long_short_ratio",
        "news_articles.sentiment_score",
        "news_articles.sentiment_label",
        "news_articles.gdelt_tone",
    }
    assert all(spec.require_variation for spec in specs)
    assert load_feature_max_age(
        REPO_ROOT / "config/quality/feature_availability.yaml"
    ).total_seconds() == 24 * 60 * 60


def test_registry_rejects_missing_or_non_positive_max_age(tmp_path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("version: '1.1.0'\nfeatures: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive max_age_hours"):
        load_feature_specs(registry)

    registry.write_text(
        "version: '1.1.0'\nmax_age_hours: 0\nfeatures: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="positive max_age_hours"):
        load_feature_specs(registry)


def test_consumer_cutoff_matches_the_final_daily_news_run() -> None:
    assert news_feature_cutoff(datetime(2026, 8, 4).date()) == datetime(
        2026, 8, 4, 18, tzinfo=UTC
    )


def test_constant_non_null_values_are_unavailable_at_exact_cutoff() -> None:
    cursor = FakeCursor([
        [("sentiment_score",), ("published_at",)],
        (92, 92, 1),
    ])
    result = measure_feature(cursor, _spec(), CUTOFF)

    assert result.status == "UNAVAILABLE"
    assert result.reason_code == "feature.constant_placeholder"
    assert result.details["non_null_rows"] == 92
    assert result.details["distinct_values"] == 1
    assert cursor.executions[1][1] == (CUTOFF,)


def test_missing_table_or_column_is_unavailable_not_zero() -> None:
    cursor = FakeCursor([[]])
    result = measure_feature(cursor, _spec(), CUTOFF)

    assert result.status == "UNAVAILABLE"
    assert result.reason_code == "feature.not_measured"
    assert result.details["non_null_rows"] == 0
    assert result.details["missing_columns"] == ["published_at", "sentiment_score"]


def test_naive_cutoff_is_rejected_before_query() -> None:
    cursor = FakeCursor([])
    with pytest.raises(ValueError, match="timezone-aware"):
        measure_feature(cursor, _spec(), datetime(2026, 8, 4))
    assert cursor.executions == []


def test_direct_spec_cannot_inject_an_sql_identifier() -> None:
    cursor = FakeCursor([])
    unsafe = FeatureSpec(
        feature_id="unsafe",
        table='news_articles"; DROP TABLE news_articles; --',
        column="sentiment_score",
        time_column="published_at",
    )
    with pytest.raises(ValueError, match="lowercase SQL identifier"):
        measure_feature(cursor, unsafe, CUTOFF)
    assert cursor.executions == []


def test_persistence_is_idempotent_but_rejects_payload_collision() -> None:
    measurement = FeatureMeasurement(
        feature_id=_spec().feature_id,
        instrument_id=None,
        status="UNAVAILABLE",
        reason_code="feature.constant_placeholder",
        observed_at=CUTOFF,
        details={"distinct_values": 1},
    )
    good = FakeCursor([None, (measurement.status, measurement.reason_code, measurement.details)])
    persist_measurement(good, measurement)
    assert "ON CONFLICT DO NOTHING" in good.executions[0][0]

    bad = FakeCursor([None, ("AVAILABLE", "feature.available", {"distinct_values": 2})])
    with pytest.raises(RuntimeError, match="immutable feature status collision"):
        persist_measurement(bad, measurement)
