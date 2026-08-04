from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

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
    monkeypatch.setattr(generator, "_get_gdelt_sentiment", lambda: pd.DataFrame())

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
