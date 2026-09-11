"""
Regression: la mascara de evaluacion es comun, reproducible y honesta.

Contract: CTR-RESEARCH-EVALMASK-001 Â· Date: 2026-08-24

Â§9.5 del plan de tesis y su test 15: todos los sistemas se evaluan sobre EL MISMO conjunto
de sesiones validas; ninguna invalida entra como retorno 0 ni cuenta en `n`.

Sin esto, un festivo relleno por el proveedor (retorno ~0) baja la volatilidad y sube el
Sharpe de todas las estrategias por igual â€” incluidos los baselines, asi que el sesgo NO se
cancela en la diferencia pareadaâ€” y ademas infla el `n` sobre el que se calculan los IC.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "config" / "research" / "evaluation_mask.json"

# Medido el 2026-08-24 sobre la serie ya reparada (CTR-DQ-TZ-001).
EXPECTED_VALID = 1249
MIN_HOLDOUT_VALID = 500   # umbral de Â§11.2 del plan: por debajo, el contraste es indecidible


@pytest.fixture(scope="module")
def mask():
    pytest.importorskip("pandas")
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.research.evaluation_mask import build_mask
    return build_mask()


def test_artifact_exists_and_matches_the_builder(mask):
    """El JSON publicado y el codigo no pueden decir cosas distintas."""
    assert ARTIFACT.is_file(), f"falta {ARTIFACT.relative_to(ROOT).as_posix()}"
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert stored["sha256"] == mask.sha256, (
        "el artefacto quedo desincronizado del builder. Regenera con "
        "`python -m src.research.evaluation_mask --out config/research/evaluation_mask.json`"
    )


def test_mask_is_reproducible(mask):
    """Mismo dato -> mismo hash. Es lo que hace comparables dos corridas."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.research.evaluation_mask import build_mask
    assert build_mask().sha256 == mask.sha256


def test_holidays_are_excluded_whether_full_or_partial(mask):
    """Un festivo con 60 barras es relleno; con 3, residuo. Los dos sobran."""
    excluded = set(mask.excluded.get("holiday", ()))
    assert excluded, "ningun festivo excluido: la lista de festivos no se esta aplicando"
    from datetime import date
    for d in (date(2024, 1, 1), date(2025, 1, 1), date(2024, 3, 29)):
        assert d in excluded, f"{d} es festivo colombiano y sigue en la mascara"


def test_us_holiday_union_is_excluded(mask):
    """The declared calendar is Colombia ∪ USA, including Independence Day."""
    from datetime import date

    assert date(2025, 7, 4) in set(mask.excluded.get("us_holiday", ()))


def test_flat_ohlc_provenance_is_persisted(mask):
    """Flat-print quality is metadata, never an implicit return filter."""
    assert mask.flat_ohlc_pct
    assert all(0.0 <= value <= 100.0 for value in mask.flat_ohlc_pct.values())
    payload = mask.to_dict()
    assert "flat_ohlc_pct" in payload


def test_no_weekend_or_out_of_window_session_survives(mask):
    assert not any(d.weekday() >= 5 for d in mask.valid), "hay fines de semana en la mascara"


def test_valid_count_is_stable(mask):
    """Si cambia, hay que mirar por que â€” no ajustar el numero sin explicacion."""
    assert len(mask) == EXPECTED_VALID, (
        f"sesiones validas: {len(mask)} != {EXPECTED_VALID} congeladas. Si el dato cambio "
        "(reparacion, backfill), actualiza el numero Y di en el commit que lo movio."
    )


def test_holdout_keeps_enough_power():
    """El bloque de juicio debe seguir siendo decidible tras enmascarar."""
    yaml = pytest.importorskip("yaml")
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.research.evaluation_mask import build_mask
    p = yaml.safe_load((ROOT / "config/research/partition.yaml").read_text(encoding="utf-8"))
    h = p["blocks"]["holdout"]
    n = len(build_mask().in_block(h["start"], h["end"]))
    assert n >= MIN_HOLDOUT_VALID, (
        f"hold-out efectivo tras la mascara: {n} < {MIN_HOLDOUT_VALID}. Â§11.2 del plan "
        "llama a 500 'el limite' para que el contraste principal sea decidible; por debajo "
        "hay que reportarlo como indecidible o reabrir la decision de particion."
    )
