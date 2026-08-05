"""CLD-490 probe: cross-review causal de C-030 (`ad4b48b9`).

C-030 responde a CLD-489. Este probe separa lo que quedó CERRADO de lo que SIGUE ABIERTO.

CERRADO (verificado aquí, no asumido):
  A1  frontera de frescura: 24h exactas = FRESCO; 24h+1s = `feature.status_stale`.
  A2  `observed_at` naive o ausente => `feature.status_timestamp_invalid`.

ABIERTO (dos hallazgos nuevos, ambos con la misma firma: el gate se rodea por un lado
que no mira):
  A3  P3 de CLD-489 NO está cerrado. La frescura se mide contra el `observed_at` que la
      propia fila DECLARA. `quality.feature_status` (migración 073) no tiene `created_at`
      -- su tabla hermana `market.canonical_bar` sí lo tiene. Una fila con `observed_at`
      falsificado hacia el futuro (las 7 del R1: escritas el 04-ago, sellando 06-ago
      00:00Z) queda a 18h del cutoff 06-ago 18:00Z: pasa como FRESCA.
  A4  el gate gobierna la columna de DB, no el valor que llega a la UI.
      `_load_news_context` (líneas 1277-1287) cae a `data/news/gdelt_daily_sentiment.csv`
      cuando el tono gobernado es null, y publica `avg_sentiment` con
      `sentiment_unavailable_reason = None`. El CSV no tiene status, ni cutoff, ni cota de
      frescura. Y el test verde de la casa
      (`test_weekly_context_reports_unavailable_instead_of_neutral`) es verde porque
      inyecta un DataFrame VACÍO en ese mismo fallback.

Run:  python -m pytest .claude/coordination/reviews/BL-40-probe-cld490.py -q
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "analysis" / "weekly_generator.py").exists():
        sys.path.insert(0, str(_parent))
        REPO = _parent
        break

from src.analysis.weekly_generator import WeeklyAnalysisGenerator  # noqa: E402

CUTOFF = datetime(2026, 8, 7, 18, tzinfo=UTC)

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
    def __init__(self, status_rows):
        self._status_rows = status_rows
        self._pending = []

    def execute(self, sql, params=None):
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
    def __init__(self, status_rows):
        self._status_rows = status_rows

    def cursor(self, *a, **k):
        return _Cursor(self._status_rows)

    def close(self):
        return None


def _db_frame(monkeypatch, observed_at, cutoff=CUTOFF, status="AVAILABLE"):
    """Real _get_all_articles DB branch with a single status row."""
    row = {
        "feature_id": "news_articles.sentiment_score",
        "status": status,
        "reason_code": "feature.available",
    }
    if observed_at is not _MISSING:
        row["observed_at"] = observed_at
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: _Connection([row]))
    monkeypatch.setattr(
        "src.analysis.weekly_generator.PROJECT_ROOT", Path("Z:/nonexistent-for-probe")
    )
    gen = WeeklyAnalysisGenerator.__new__(WeeklyAnalysisGenerator)
    gen._all_articles_cache = None
    gen._feature_cutoff = cutoff
    return gen._get_all_articles()


_MISSING = object()


# ---------------------------------------------------------------- A1: frontera 24h
def test_a1_exactly_24h_is_still_fresh_and_24h_plus_one_second_is_stale(monkeypatch):
    fresh = _db_frame(monkeypatch, CUTOFF - timedelta(hours=24))
    assert fresh.iloc[0]["tone"] == pytest.approx(0.42)
    assert fresh.iloc[0]["sentiment_unavailable_reason"] is None

    stale = _db_frame(monkeypatch, CUTOFF - timedelta(hours=24, seconds=1))
    assert stale.iloc[0]["tone"] is None
    assert stale.iloc[0]["sentiment_unavailable_reason"] == "feature.status_stale"


# ------------------------------------------------- A2: timestamp naive o ausente
def test_a2_naive_or_missing_observed_at_is_rejected(monkeypatch):
    naive = _db_frame(monkeypatch, datetime(2026, 8, 7, 18))
    assert naive.iloc[0]["sentiment_unavailable_reason"] == "feature.status_timestamp_invalid"

    absent = _db_frame(monkeypatch, _MISSING)
    assert absent.iloc[0]["sentiment_unavailable_reason"] == "feature.status_timestamp_invalid"

    as_date = _db_frame(monkeypatch, date(2026, 8, 7))
    assert as_date.iloc[0]["sentiment_unavailable_reason"] == "feature.status_timestamp_invalid"


# ----------------------------------- A3: la frescura no detecta un observed_at falsificado
def test_a3_forged_future_observed_at_passes_as_fresh(monkeypatch):
    """Las 7 filas R1 (CXD-503) siguen consumibles: 18h < 24h del cutoff del 06-ago."""
    r1_cutoff = datetime(2026, 8, 6, 18, tzinfo=UTC)
    r1_observed = datetime(2026, 8, 6, 0, tzinfo=UTC)  # sellada por la corrida del 04-ago
    assert r1_cutoff - r1_observed == timedelta(hours=18)

    frame = _db_frame(monkeypatch, r1_observed, cutoff=r1_cutoff)
    assert frame.iloc[0]["tone"] == pytest.approx(0.42)
    assert frame.iloc[0]["sentiment_unavailable_reason"] is None  # aceptada como fresca


def test_a3b_feature_status_has_no_created_at_to_catch_the_forgery():
    """La defensa existe en la tabla hermana y falta en esta."""
    sql = (REPO / "database/migrations/073_market_quality.sql").read_text(encoding="utf-8")
    feature_status = sql.split("CREATE TABLE IF NOT EXISTS quality.feature_status")[1]
    feature_status = feature_status.split(");")[0]
    canonical = sql.split("CREATE TABLE IF NOT EXISTS market.canonical_bar")[1].split(");")[0]
    assert "created_at" in canonical and "NOW()" in canonical
    assert "created_at" not in feature_status and "inserted_at" not in feature_status


# ------------------------------- A4: el CSV de GDELT rodea el gate entero
def test_a4_gdelt_csv_fallback_republishes_a_measured_sentiment_without_reason(monkeypatch):
    """Tono gobernado null (stale) => la UI recibe un numero y reason=None."""
    gen = WeeklyAnalysisGenerator.__new__(WeeklyAnalysisGenerator)
    gen._all_articles_cache = None
    gen._feature_cutoff = CUTOFF

    governed_null = pd.DataFrame([{
        "date": pd.Timestamp("2026-08-05"),
        "title": "BanRep holds the policy rate",
        "source": "investing",
        "news_source": "investing",
        "url": "https://example.invalid/1",
        "tone": None,                                    # gate C028: no medido
        "sentiment_unavailable_reason": "feature.status_stale",
    }])
    ungoverned_csv = pd.DataFrame(
        {"tone_avg": [-3.1, -2.9]},
        index=pd.to_datetime(["2026-08-05", "2026-08-06"]),
    )
    monkeypatch.setattr(gen, "_get_all_articles", lambda: governed_null)
    monkeypatch.setattr(gen, "_get_gdelt_sentiment", lambda: ungoverned_csv)

    result = gen._load_news_context(date(2026, 8, 3), date(2026, 8, 7))

    # El gate dijo "no medido" y la superficie publica un numero sin motivo.
    assert result["avg_sentiment"] == pytest.approx(-3.0)
    assert result["sentiment_unavailable_reason"] is None


def test_a4b_the_house_green_test_disables_the_bypass_it_would_hit():
    """El test verde inyecta un DataFrame vacio en el mismo fallback."""
    src = (REPO / "tests/unit/test_weekly_sentiment_unavailable.py").read_text(encoding="utf-8")
    body = src.split("def test_weekly_context_reports_unavailable_instead_of_neutral")[1]
    body = body.split("\ndef ")[0]
    assert '_get_gdelt_sentiment", lambda: pd.DataFrame()' in body
