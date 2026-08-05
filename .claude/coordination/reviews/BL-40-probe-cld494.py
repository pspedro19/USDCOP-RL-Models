"""CLD-494 probe: el candado de provenance de C-031 rechaza TAMBIEN la fila legitima.

Hashes atacados: `25d2f4cd` + `2032ab32` + `95d434c6` (C-031).

C-031 cierra A3 (sello autodeclarado) exigiendo `created_at` de reloj DB y, en el consumidor,

    ... AND observed_at <= %s AND created_at IS NOT NULL AND created_at <= %s   (ambos = cutoff)

y en Python `observed_at > created_at or created_at > cutoff => feature.status_provenance_invalid`.

`observed_at <= created_at` es correcto: nada se crea antes de observarse. **`created_at <= cutoff`
no lo es**, y cierra la puerta a todo:

  el productor sella `observed_at = data_interval_end` (18:00:00Z EXACTO, es la clave del run),
  y `created_at` lo pone el reloj de la DB CUANDO LA TAREA CORRE -- segundos DESPUES de 18:00:00Z,
  porque Airflow dispara el run al cerrar el intervalo. Asi que toda fila real tiene
  `created_at > cutoff` y queda fuera.

Consecuencia: `quality.feature_status` no puede volver a ser consumida. El estado degrada a
`feature.status_provenance_invalid` (rama Python) o a `feature.status_missing` (el filtro SQL la
excluye antes). Y **coincide con la salida esperada hoy** --las 7 filas vivas son `UNAVAILABLE`--
asi que el gate parece funcionar mientras esta muerto: la clase exacta que llevamos dos dias
matando (K-051).

P1  fila legitima (observed 18:00:00Z, created 18:00:07Z, cutoff 18:00:00Z) => rechazada.
P2  el filtro SQL la excluiria en una DB real: lleva `created_at <= %s` con el cutoff.
P3  la frontera de 24h SIGUE bien cuando el sello pasa (regresion de A1, con provenance valida).
P4  A3 cerrado de verdad: sello futuro falsificado => rechazado incluso con created_at coherente.

Run:  python -m pytest .claude/coordination/reviews/BL-40-probe-cld494.py -q
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg2
import pytest

for _p in Path(__file__).resolve().parents:
    if (_p / "src" / "analysis" / "weekly_generator.py").exists():
        sys.path.insert(0, str(_p))
        REPO = _p
        break

from src.analysis.weekly_generator import WeeklyAnalysisGenerator  # noqa: E402

CUTOFF = datetime(2026, 8, 7, 18, tzinfo=UTC)  # news_feature_cutoff(2026-08-07)

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

_SQL_SEEN: list[str] = []


class _Cursor:
    def __init__(self, status_rows):
        self._status_rows = status_rows
        self._pending = []

    def execute(self, sql, params=None):
        if "quality.feature_status" in sql:
            _SQL_SEEN.append(" ".join(sql.split()))
            self._pending = list(self._status_rows)
        else:
            self._pending = list(ARTICLE_ROWS)

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


def _frame(monkeypatch, observed_at, created_at, cutoff=CUTOFF):
    row = {
        "feature_id": "news_articles.sentiment_score",
        "status": "AVAILABLE",
        "reason_code": "feature.available",
        "observed_at": observed_at,
        "created_at": created_at,
    }
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: _Connection([row]))
    monkeypatch.setattr(
        "src.analysis.weekly_generator.PROJECT_ROOT", Path("Z:/nonexistent-for-probe")
    )
    gen = WeeklyAnalysisGenerator.__new__(WeeklyAnalysisGenerator)
    gen._all_articles_cache = None
    gen._feature_cutoff = cutoff
    return gen._get_all_articles()


def test_p1_the_legitimate_row_of_the_18z_run_is_rejected(monkeypatch):
    """observed = clave del run; created = reloj DB al ejecutar, 7s despues."""
    observed = CUTOFF                        # data_interval_end exacto
    created = CUTOFF + timedelta(seconds=7)  # NOW() cuando la tarea corrio
    assert observed <= created               # provenance coherente
    frame = _frame(monkeypatch, observed, created)
    assert frame.iloc[0]["tone"] is None
    assert frame.iloc[0]["sentiment_unavailable_reason"] == "feature.status_provenance_invalid"


def test_p2_the_sql_filter_would_exclude_it_in_a_real_database(monkeypatch):
    _SQL_SEEN.clear()
    _frame(monkeypatch, CUTOFF, CUTOFF + timedelta(seconds=7))
    sql = next(s for s in _SQL_SEEN)
    assert "created_at <= %s" in sql, "el filtro por cutoff sobre created_at es el que cierra todo"
    assert "observed_at <= %s" in sql


def test_p3_the_24h_boundary_still_holds_when_provenance_is_valid(monkeypatch):
    """Regresion de A1: con sello valido, 24h exactas siguen frescas y 24h+1s stale."""
    observed = CUTOFF - timedelta(hours=24)
    fresh = _frame(monkeypatch, observed, observed)  # created == observed, <= cutoff
    assert fresh.iloc[0]["tone"] == pytest.approx(0.42)
    assert fresh.iloc[0]["sentiment_unavailable_reason"] is None

    observed = CUTOFF - timedelta(hours=24, seconds=1)
    stale = _frame(monkeypatch, observed, observed)
    assert stale.iloc[0]["tone"] is None
    assert stale.iloc[0]["sentiment_unavailable_reason"] == "feature.status_stale"


def test_p4_a3_is_really_closed_even_with_a_coherent_created_at(monkeypatch):
    """Sello futuro falsificado: created_at anterior a observed_at => rechazado."""
    r1_cutoff = datetime(2026, 8, 6, 18, tzinfo=UTC)
    forged_observed = datetime(2026, 8, 6, 0, tzinfo=UTC)   # sellada por la corrida del 04-ago
    real_created = datetime(2026, 8, 4, 22, tzinfo=UTC)     # cuando de verdad se escribio
    frame = _frame(monkeypatch, forged_observed, real_created, cutoff=r1_cutoff)
    assert frame.iloc[0]["tone"] is None
    assert frame.iloc[0]["sentiment_unavailable_reason"] == "feature.status_provenance_invalid"
