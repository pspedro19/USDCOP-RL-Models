from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

from src.analysis.weekly_generator import WeeklyAnalysisGenerator
from src.analysis.prompt_templates import build_news_section
from src.data_quality.feature_availability import news_feature_cutoff


def _generator() -> WeeklyAnalysisGenerator:
    return WeeklyAnalysisGenerator.__new__(WeeklyAnalysisGenerator)


def test_daily_average_has_no_synthetic_zero_when_every_value_is_missing() -> None:
    rows = [{"tone": None}, {"sentiment_score": float("nan")}, {}]
    measured = [
        value
        for value in (WeeklyAnalysisGenerator._sentiment_value(row) for row in rows)
        if value is not None
    ]
    assert measured == []


def test_measured_neutral_is_preserved_as_a_real_value() -> None:
    assert WeeklyAnalysisGenerator._sentiment_value({"tone": 0.0}) == 0.0


def test_weekly_context_reports_unavailable_instead_of_neutral(monkeypatch) -> None:
    generator = _generator()
    articles = pd.DataFrame(
        [{
            "date": pd.Timestamp("2026-08-03"),
            "title": "BanRep policy outlook for the Colombian peso",
            "source": "example",
            "news_source": "example",
            "url": "https://example.invalid/a",
            "tone": None,
        }]
    )
    monkeypatch.setattr(generator, "_get_all_articles", lambda: articles)

    result = generator._load_news_context(date(2026, 8, 3), date(2026, 8, 7))

    assert result["avg_sentiment"] is None
    assert result["sentiment_unavailable_reason"] == "feature.not_measured"
    assert result["highlights"][0]["sentiment"] is None


def test_shared_news_cutoff_is_the_exact_final_news_run() -> None:
    assert news_feature_cutoff(date(2026, 8, 7)) == datetime(
        2026, 8, 7, 18, tzinfo=UTC
    )


def test_weekly_cutoff_is_not_replaced_by_nested_daily_generation() -> None:
    generator = _generator()
    weekly_cutoff = news_feature_cutoff(date(2026, 8, 7))
    generator._feature_cutoff = weekly_cutoff
    generator._ensure_feature_cutoff(date(2026, 8, 3))
    assert generator._feature_cutoff == weekly_cutoff


def test_prompt_names_unavailable_sentiment_instead_of_neutral() -> None:
    rendered = build_news_section({
        "article_count": 92,
        "avg_sentiment": None,
        "sentiment_unavailable_reason": "feature.constant_placeholder",
    })
    assert "NO DISPONIBLE (feature.constant_placeholder)" in rendered
    assert "sentimiento promedio: neutral" not in rendered


class _StatusCursor:
    def __init__(self, status_rows):
        self.status_rows = status_rows
        self.rows = []

    def execute(self, sql, params=None):
        if "quality.feature_status" in sql:
            self.rows = self.status_rows
        else:
            self.rows = [{
                "date": datetime(2026, 8, 7, 12, tzinfo=UTC),
                "title": "BanRep holds the policy rate",
                "source": "investing",
                "url": "https://example.invalid/stale",
                "language": "es",
                "sentiment_score": 0.42,
                "sentiment_label": "positive",
                "gdelt_tone": None,
                "category": "monetary",
            }]

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StatusConnection:
    def __init__(self, status_rows):
        self.status_rows = status_rows

    def cursor(self, *args, **kwargs):
        return _StatusCursor(self.status_rows)

    def close(self):
        return None


def test_stale_available_status_degrades_to_explicit_unavailable(monkeypatch) -> None:
    cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda *args, **kwargs: _StatusConnection([{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
            "observed_at": datetime(2026, 8, 5, 18, tzinfo=UTC),
            "created_at": datetime(2026, 8, 5, 18, tzinfo=UTC),
        }]),
    )
    monkeypatch.setattr(
        "src.analysis.weekly_generator.PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = cutoff

    frame = generator._get_all_articles()

    assert frame.iloc[0]["tone"] is None
    assert frame.iloc[0]["sentiment_unavailable_reason"] == "feature.status_stale"


def test_exact_24h_status_with_valid_creation_seal_remains_fresh(monkeypatch) -> None:
    cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)
    observed_at = datetime(2026, 8, 6, 18, tzinfo=UTC)
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda *args, **kwargs: _StatusConnection([{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
            "observed_at": observed_at,
            "created_at": observed_at,
        }]),
    )
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = cutoff

    frame = generator._get_all_articles()

    assert frame.iloc[0]["tone"] == pytest.approx(0.42)
    assert frame.iloc[0]["sentiment_unavailable_reason"] is None


def test_status_without_creation_seal_is_not_authoritative(monkeypatch) -> None:
    cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda *args, **kwargs: _StatusConnection([{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
            "observed_at": datetime(2026, 8, 7, 17, tzinfo=UTC),
        }]),
    )
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = cutoff

    frame = generator._get_all_articles()

    assert frame.iloc[0]["tone"] is None
    assert (
        frame.iloc[0]["sentiment_unavailable_reason"]
        == "feature.status_provenance_invalid"
    )


def test_missing_feature_cutoff_fails_closed_before_db_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda *args, **kwargs: pytest.fail("DB must not be opened without a cutoff"),
    )
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = None

    with pytest.raises(RuntimeError, match="feature cutoff must be set"):
        generator._get_all_articles()


def test_ungoverned_gdelt_csv_cannot_override_unavailable_sentiment(monkeypatch) -> None:
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)
    governed_null = pd.DataFrame([{
        "date": pd.Timestamp("2026-08-05"),
        "title": "BanRep holds the policy rate",
        "source": "investing",
        "news_source": "investing",
        "url": "https://example.invalid/stale",
        "tone": None,
        "sentiment_unavailable_reason": "feature.status_stale",
    }])
    monkeypatch.setattr(generator, "_get_all_articles", lambda: governed_null)
    result = generator._load_news_context(date(2026, 8, 3), date(2026, 8, 7))

    assert result["avg_sentiment"] is None
    assert result["sentiment_unavailable_reason"] == "feature.status_stale"


def test_tracked_backup_supplies_headlines_but_never_ungoverned_tone(
    monkeypatch, tmp_path
) -> None:
    backup = tmp_path / "data/backups/features/news_articles.parquet"
    backup.parent.mkdir(parents=True)
    pd.DataFrame([{
        "published_at": datetime(2026, 8, 5, 12, tzinfo=UTC),
        "title": "BanRep holds the policy rate",
        "source_id": "backup",
        "sentiment_score": 0.99,
    }]).to_parquet(backup, index=False)
    monkeypatch.setattr("src.analysis.weekly_generator.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        psycopg2, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    generator = _generator()
    generator._all_articles_cache = None
    generator._feature_cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)

    frame = generator._get_all_articles()

    assert frame.iloc[0]["title"] == "BanRep holds the policy rate"
    assert frame.iloc[0]["tone"] is None
    assert (
        frame.iloc[0]["sentiment_unavailable_reason"]
        == "feature.backup_without_status"
    )
