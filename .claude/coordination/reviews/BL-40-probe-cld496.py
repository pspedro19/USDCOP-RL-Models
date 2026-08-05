"""CLD-496 probe: mutacion independiente del limite de publicacion de C-031 (`ef34c9bd`).

Codex pidio "mutacion independiente Claude del limite 60m". La mutacion se hace **sin tocar
`config/quality/feature_availability.yaml`**: se sustituye el cargador y se le pasan registries
temporales. Mutar un config compartido mientras el otro agente tiene lease de DB/DAG es
exactamente el incidente K-050, y aqui no hace falta.

Lo que fija:
  B1  el sello real (+7s) pasa: el defecto de CLD-494 esta cerrado.
  B2  frontera: +60m exactas pasan, +60m01s da `feature.status_provenance_invalid`.
  B3  la frontera la manda el SSOT, no un literal: con lag de 1 minuto, +7s sigue pasando y
      +2m se rechaza. Es la mutacion del limite.
  B4  R1 sigue invalida: `created_at` ANTERIOR a `observed_at` se rechaza.
  B5  `created_at` naive o ausente se rechaza.
  B6  la frescura de 24h sigue encima del lag: lag valido + 25h de antiguedad = `status_stale`.
  B7  el SQL lleva la MISMA cota que Python (si no, una DB real filtraria distinto de lo que
      estos tests ejercen).
  B8  el registry rechaza lag ausente, 0, negativo y `True` (bool es subclase de int).

Run:  python -m pytest .claude/coordination/reviews/BL-40-probe-cld496.py -q
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

import src.analysis.weekly_generator as wg  # noqa: E402
from src.data_quality.feature_availability import load_feature_publish_lag  # noqa: E402

CUTOFF = datetime(2026, 8, 7, 18, tzinfo=UTC)
OBSERVED = CUTOFF  # data_interval_end exacto del run de las 18Z

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


def _frame(monkeypatch, observed_at, created_at, cutoff=CUTOFF, lag_minutes=None):
    """Ejecuta la rama DB real. `lag_minutes` sustituye el limite SIN tocar el SSOT."""
    row = {
        "feature_id": "news_articles.sentiment_score",
        "status": "AVAILABLE",
        "reason_code": "feature.available",
        "observed_at": observed_at,
        "created_at": created_at,
    }
    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: _Connection([row]))
    monkeypatch.setattr(wg, "PROJECT_ROOT", Path("Z:/nonexistent-for-probe"))
    if lag_minutes is not None:
        monkeypatch.setattr(
            wg, "load_feature_publish_lag", lambda *_a, **_k: timedelta(minutes=lag_minutes)
        )
    gen = wg.WeeklyAnalysisGenerator.__new__(wg.WeeklyAnalysisGenerator)
    gen._all_articles_cache = None
    gen._feature_cutoff = cutoff
    return gen._get_all_articles()


def _reason(frame):
    return frame.iloc[0]["sentiment_unavailable_reason"]


# ------------------------------------------------------- B1: el defecto esta cerrado
def test_b1_the_real_seven_second_seal_is_accepted(monkeypatch):
    frame = _frame(monkeypatch, OBSERVED, OBSERVED + timedelta(seconds=7))
    assert frame.iloc[0]["tone"] == pytest.approx(0.42)
    assert _reason(frame) is None


# ------------------------------------------------------------------ B2: frontera SSOT
def test_b2_the_declared_sixty_minute_boundary_is_inclusive(monkeypatch):
    ssot = load_feature_publish_lag(REPO / "config/quality/feature_availability.yaml")
    assert ssot == timedelta(minutes=60)

    exacto = _frame(monkeypatch, OBSERVED, OBSERVED + ssot)
    assert _reason(exacto) is None, "60m exactas deben pasar: 'max lag' es inclusivo"

    pasado = _frame(monkeypatch, OBSERVED, OBSERVED + ssot + timedelta(seconds=1))
    assert _reason(pasado) == "feature.status_provenance_invalid"


# --------------------------------------- B3: LA MUTACION -- el limite lo manda el SSOT
@pytest.mark.parametrize(
    "lag_minutes, retraso, espera",
    [
        (1, timedelta(seconds=7), None),
        (1, timedelta(minutes=2), "feature.status_provenance_invalid"),
        (1, timedelta(minutes=1), None),
        (600, timedelta(minutes=90), None),
    ],
)
def test_b3_the_boundary_follows_the_ssot_and_is_not_a_literal(
    monkeypatch, lag_minutes, retraso, espera
):
    frame = _frame(monkeypatch, OBSERVED, OBSERVED + retraso, lag_minutes=lag_minutes)
    assert _reason(frame) == espera


# ------------------------------------------------------------ B4/B5: provenance rota
def test_b4_the_r1_forgery_is_still_rejected(monkeypatch):
    """created_at ANTERIOR a observed_at: la fila del 04-ago sellando 06-ago."""
    frame = _frame(
        monkeypatch,
        datetime(2026, 8, 6, 0, tzinfo=UTC),
        datetime(2026, 8, 4, 22, tzinfo=UTC),
        cutoff=datetime(2026, 8, 6, 18, tzinfo=UTC),
    )
    assert _reason(frame) == "feature.status_provenance_invalid"


@pytest.mark.parametrize("created", [None, datetime(2026, 8, 7, 18, 0, 7)])
def test_b5_missing_or_naive_created_at_is_rejected(monkeypatch, created):
    frame = _frame(monkeypatch, OBSERVED, created)
    assert _reason(frame) == "feature.status_provenance_invalid"


# ------------------------------------------------- B6: la frescura sigue por encima
def test_b6_a_valid_lag_does_not_rescue_a_stale_measurement(monkeypatch):
    viejo = CUTOFF - timedelta(hours=25)
    frame = _frame(monkeypatch, viejo, viejo + timedelta(seconds=7))
    assert _reason(frame) == "feature.status_stale"


# ------------------------------------------------- B7: SQL y Python dicen lo mismo
def test_b7_the_sql_carries_the_same_bound_as_python(monkeypatch):
    _SQL_SEEN.clear()
    _frame(monkeypatch, OBSERVED, OBSERVED + timedelta(seconds=7))
    sql = _SQL_SEEN[0]
    assert "created_at <= observed_at + %s" in sql
    assert "created_at IS NOT NULL" in sql
    assert "observed_at <= %s" in sql
    assert "created_at <= %s" not in sql, "la cota contra el cutoff era el defecto de CLD-494"


# --------------------------------------------------- B8: el registry valida el limite
@pytest.mark.parametrize(
    "cuerpo",
    [
        "version: '1.1.0'\nmax_age_hours: 24\nfeatures: []\n",
        "version: '1.1.0'\nmax_age_hours: 24\nmax_publish_lag_minutes: 0\nfeatures: []\n",
        "version: '1.1.0'\nmax_age_hours: 24\nmax_publish_lag_minutes: -5\nfeatures: []\n",
        "version: '1.1.0'\nmax_age_hours: 24\nmax_publish_lag_minutes: true\nfeatures: []\n",
    ],
    ids=["ausente", "cero", "negativo", "bool"],
)
def test_b8_the_registry_rejects_an_unusable_lag(tmp_path, cuerpo):
    registry = tmp_path / "registry.yaml"
    registry.write_text(cuerpo, encoding="utf-8")
    with pytest.raises(ValueError, match="max_publish_lag_minutes"):
        load_feature_publish_lag(registry)


# ------------------- B9: una corrida LENTA desaparece y la sustituye otra medicion
def test_b9_a_late_run_is_silently_replaced_by_an_earlier_measurement(monkeypatch):
    """Observacion, no defecto de la cota: el filtro SQL borra la fila tardia.

    Escenario realista: la corrida de las 18Z tarda 70 minutos (ingesta + enriquecimiento +
    export antes de medir). Su fila queda `created_at - observed_at > 60m`, asi que el WHERE
    la excluye ANTES del `DISTINCT ON`. La siguiente candidata es la fila de las 12Z, valida
    y con 6h de antiguedad (< 24h): el consumidor la usa y devuelve AVAILABLE.

    El resultado no es falso -- 12Z fue una medicion real -- pero **el retraso de la corrida
    del cutoff objetivo se vuelve invisible**: la superficie no distingue "medido a las 18Z"
    de "las 18Z llegaron tarde y te doy las 12Z".

    Aqui se modela lo que la DB devolveria: solo la fila que pasa el WHERE.
    """
    fila_12z = datetime(2026, 8, 7, 12, tzinfo=UTC)
    frame = _frame(monkeypatch, fila_12z, fila_12z + timedelta(seconds=7))
    assert frame.iloc[0]["tone"] == pytest.approx(0.42)
    assert _reason(frame) is None
    # Y la de las 18Z, con 70m de lag, no habria sobrevivido al filtro:
    tardia = _frame(monkeypatch, OBSERVED, OBSERVED + timedelta(minutes=70))
    assert _reason(tardia) == "feature.status_provenance_invalid"
