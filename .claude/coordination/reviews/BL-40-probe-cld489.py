"""CLD-489 probe: the C028 consumer accepts a feature_status of ANY age.

Target hash under review: 1fb83da7 (BL-40 promotion) over 9c9b0bcd + 3042155b.

The consumer SQL in src/analysis/weekly_generator.py:1774-1784 is

    SELECT DISTINCT ON (feature_id) feature_id, status, reason_code
      FROM quality.feature_status
     WHERE feature_id IN (...) AND observed_at <= %s
     ORDER BY feature_id, observed_at DESC

`observed_at <= cutoff` is the causal upper bound (correct). There is NO lower
bound, so the "latest row at or before the cutoff" may have been measured
arbitrarily long ago. Consequence proven below:

  P1  a single AVAILABLE row, 584 days stale, is applied as authoritative:
      tone is published as MEASURED with reason=None.
  P2  the fail-closed branch `feature.status_missing` (line 1815) is only
      reachable while the table has never held a row for that feature_id.
  P3  the durable future-dated rows of the R1 run (CXD-503: observed_at
      2026-08-06T00:00Z, written by the 2026-08-04 run) satisfy `<= cutoff`
      for the 2026-08-06 analysis.
  P4  the RuntimeError guard "news feature cutoff must be set" (line 1773)
      sits inside `try/except Exception -> logger.debug`, so violating the
      invariant silently drops the DB source instead of stopping.

Run:  python -m pytest <this file> -q
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "src" / "analysis" / "weekly_generator.py").exists():
        sys.path.insert(0, str(parent))
        break
else:  # pragma: no cover
    sys.path.insert(0, r"C:\Users\USUARIO\Documents\USDCOP-RL-Models")

from src.analysis.weekly_generator import WeeklyAnalysisGenerator  # noqa: E402

ARTICLE_ROWS = [
    {
        "date": datetime(2026, 8, 5, 12, tzinfo=UTC),
        "title": "BanRep holds the policy rate",
        "source": "investing",
        "url": "https://example.invalid/1",
        "language": "es",
        "sentiment_score": 0.42,
        "sentiment_label": "positive",
        "gdelt_tone": None,
        "category": "monetary",
    }
]


class _Cursor:
    """Returns `status_rows` for the status query, ARTICLE_ROWS for the articles."""

    def __init__(self, status_rows: list[dict], calls: list[tuple]) -> None:
        self._status_rows = status_rows
        self._calls = calls
        self._pending: list[dict] = []

    def execute(self, sql, params=None):
        self._calls.append((" ".join(sql.split()), params))
        self._pending = (
            list(self._status_rows) if "quality.feature_status" in sql else list(ARTICLE_ROWS)
        )

    def fetchall(self):
        return self._pending

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Connection:
    def __init__(self, status_rows: list[dict], calls: list[tuple]) -> None:
        self._status_rows = status_rows
        self._calls = calls

    def cursor(self, *args, **kwargs):
        return _Cursor(self._status_rows, self._calls)

    def close(self):
        return None


def _run(monkeypatch, status_rows, cutoff):
    """Drive the real _get_all_articles DB branch; return (db_frame, sql_calls)."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        psycopg2, "connect", lambda *a, **k: _Connection(status_rows, calls)
    )
    gen = WeeklyAnalysisGenerator.__new__(WeeklyAnalysisGenerator)
    gen._all_articles_cache = None
    gen._feature_cutoff = cutoff
    # Neutralise the file-based sources: only the DB branch is under test.
    monkeypatch.setattr(
        "src.analysis.weekly_generator.PROJECT_ROOT", Path("Z:/nonexistent-for-probe")
    )
    frame = gen._get_all_articles()
    return frame, calls


def test_p1_a_584_day_old_available_status_is_applied_as_authoritative(monkeypatch):
    cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)
    stale = datetime(2024, 12, 31, 18, tzinfo=UTC)  # 584 days before the cutoff
    frame, calls = _run(
        monkeypatch,
        [{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
        }],
        cutoff,
    )
    assert (cutoff - stale).days == 584
    assert not frame.empty
    # The stale AVAILABLE row makes the tone a MEASURED value with no reason.
    assert frame.iloc[0]["tone"] == pytest.approx(0.42)
    assert frame.iloc[0]["sentiment_unavailable_reason"] is None
    # And the SQL that selected it carries no lower bound on observed_at.
    status_sql = next(sql for sql, _ in calls if "quality.feature_status" in sql)
    assert "observed_at <= %s" in status_sql
    assert ">=" not in status_sql and "interval" not in status_sql.lower()


def test_p2_status_missing_is_unreachable_once_any_row_exists(monkeypatch):
    """Fail-closed reason only fires on a table that never held the feature."""
    cutoff = datetime(2026, 8, 7, 18, tzinfo=UTC)

    empty, _ = _run(monkeypatch, [], cutoff)
    assert empty.iloc[0]["sentiment_unavailable_reason"] == "feature.status_missing"

    ancient, _ = _run(
        monkeypatch,
        [{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
        }],
        cutoff,
    )
    assert ancient.iloc[0]["sentiment_unavailable_reason"] is None


def test_p3_future_dated_r1_rows_satisfy_the_cutoff_of_their_own_target_day(monkeypatch):
    """CXD-503: the 08-04 run persisted observed_at=2026-08-06T00:00Z, durable."""
    cutoff = datetime(2026, 8, 6, 18, tzinfo=UTC)
    r1_observed = datetime(2026, 8, 6, 0, tzinfo=UTC)
    assert r1_observed <= cutoff  # selected by the consumer WHERE clause
    frame, _ = _run(
        monkeypatch,
        [{
            "feature_id": "news_articles.sentiment_score",
            "status": "AVAILABLE",
            "reason_code": "feature.available",
        }],
        cutoff,
    )
    assert frame.iloc[0]["sentiment_unavailable_reason"] is None


def test_p4_missing_cutoff_is_swallowed_instead_of_raising(monkeypatch):
    cutoff = None
    frame, calls = _run(monkeypatch, [], cutoff)
    # No exception escaped, and no status query was ever issued.
    assert not any("quality.feature_status" in sql for sql, _ in calls)
    assert frame.empty or "tone" not in frame.columns
