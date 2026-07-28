"""BL-20 — valida los artefactos de interpretabilidad (fase 1: DATOS).

Corre el generador real para ridge (zoo, SHAP lineal cerrado) + spx500 (rule-based,
atribucion de reglas) y valida shape, header 'nota' obligatorio y ausencia de NaN/Inf
en el JSON serializado (A.7 / safe_json_dump).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.generate_interpretability import (  # noqa: E402
    NOTA,
    generate_rule_attribution,
    generate_zoo_linear,
)


def _fail_on_constant(tok: str):
    raise AssertionError(f"JSON contiene constante no-finita: {tok}")


def _load_strict(path: Path) -> dict:
    """json.load que REVIENTA si el texto contiene NaN/Infinity (safe JSON gate)."""
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_fail_on_constant)


def _assert_all_finite(obj, where: str = "$") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"valor no finito en {where}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite(v, f"{where}[{i}]")


@pytest.fixture(scope="module")
def artifacts() -> dict[str, Path]:
    """Corre el generador UNA vez (ridge + spx500) y devuelve los paths."""
    zoo = generate_zoo_linear(("ridge",))
    rules = generate_rule_attribution(("spx500",))
    assert len(zoo) == 1 and len(rules) == 1
    return {"ridge": zoo[0], "spx500": rules[0]}


def test_paths_follow_surface_asset_model_version_layout(artifacts):
    pub = REPO / "usdcop-trading-dashboard" / "public" / "data" / "interpretability"
    ridge, spx = artifacts["ridge"], artifacts["spx500"]
    assert ridge.name == "summary.json" and spx.name == "summary.json"
    # <surface>/<asset>/<model_id>/<version>/summary.json
    assert ridge.relative_to(pub).parts[:3] == ("zoo", "usdcop", "ridge")
    assert spx.relative_to(pub).parts[:3] == ("rule_based", "spx500", "spx500_regime_gated_v1")
    assert len(ridge.relative_to(pub).parts) == 5
    assert len(spx.relative_to(pub).parts) == 5


def test_nota_header_present_and_exact(artifacts):
    for path in artifacts.values():
        d = _load_strict(path)
        assert d["nota"] == NOTA
        assert d["nota"] == ("SHAP explica el modelo, no el mercado; "
                             "solo test-folds; diagnostico 0 trials")


def test_no_nan_inf_anywhere(artifacts):
    for path in artifacts.values():
        _assert_all_finite(_load_strict(path))


def test_ridge_linear_shap_shape(artifacts):
    d = _load_strict(artifacts["ridge"])
    assert d["surface"] == "zoo" and d["asset"] == "usdcop" and d["model_id"] == "ridge"
    assert d["method"] == "linear_shap_closed_form"
    assert d["attribution_not_shap"] is False
    assert d["n_features"] == len(d["top_features"]) > 0
    for entry in d["top_features"]:
        assert set(entry) == {"rank", "feature", "coef", "mean_abs_shap", "mean_shap"}
        assert entry["mean_abs_shap"] >= 0.0
    # top_features ordenado por mean|phi| descendente
    mas = [e["mean_abs_shap"] for e in d["top_features"]]
    assert mas == sorted(mas, reverse=True)
    # agregado por anio: cada anio lista las mismas features
    assert len(d["by_year"]) >= 2
    for yr, feats in d["by_year"].items():
        assert int(yr) >= 2020
        assert len(feats) == d["n_features"]
    # fit walk-forward: train-only + purga declaradas
    assert d["fit"]["purge_days"] == d["fit"]["horizon"] == 5
    assert d["fit"]["n_train"] >= 200


def test_spx500_rule_attribution_shape(artifacts):
    d = _load_strict(artifacts["spx500"])
    assert d["surface"] == "rule_based"
    assert d["model_id"] == "spx500_regime_gated_v1"
    assert d["attribution_not_shap"] is True          # etiqueta obligatoria BL-20
    assert d["method"] == "rule_attribution"
    r = d["rules"]
    assert 0.0 <= r["pct_days_trend_on"] <= 1.0
    assert 0.0 <= r["pct_days_position_active"] <= 1.0
    dec = d["pnl_decomposition"]
    # identidad de la descomposicion: gross = beta + timing (cov(pos,ret))
    assert dec["pnl_gross"] == pytest.approx(
        dec["pnl_beta"] + dec["pnl_timing_cov_pos_ret"], abs=1e-9)
    assert dec["pnl_net"] == pytest.approx(dec["pnl_gross"] - dec["costs"], abs=1e-9)
    assert dec["n_days"] > 0
    for yr, y in d["by_year"].items():
        assert y["n_days"] > 0
        assert y["pnl_gross"] == pytest.approx(
            y["pnl_beta"] + y["pnl_timing_cov_pos_ret"], abs=1e-9)
