"""BL-20 — los PRODUCTORES se ejecutan, no solo se validan sus JSON (CXD-576).

POR QUÉ EXISTE. CODEX rechazó el flip de BL-20 con un hallazgo que yo no había visto y que es
peor de lo que él lo planteó: ningún test llamaba a `generate_zoo_hybrid()` ni a
`generate_composite_v11()`. Se juzgaba el `summary.json` publicado, no el generador — así que
rompiendo el productor y dejando los artefactos intactos, las 90 pruebas seguían verdes.

Y el corolario es lo grave: **los candados que hacen confiable todo esto viven DENTRO del
productor** — la aditividad contra la predicción del híbrido completo, la composición del
reescalado afín, y la negativa a publicar cuando `add_err` se pasa. Si nadie ejecuta el
productor, nadie ejecuta esos candados. Lo que quedaba juzgado era el JSON; lo que hace que el
JSON sea cierto, no.

QUÉ HACE ESTE FICHERO. Ejecuta ambos generadores sobre una **fixture acotada** (frame sintético,
`OUT_ROOT` a `tmp_path`) y comprueba que la garantía se sostiene *y que muerde*. Los mutantes se
inyectan en **costuras reales del productor**, no en el artefacto:

    _hybrid_linear_half   -> ceros     la mitad LINEAL del híbrido deja de ser correcta
    _tree_shap_backend    -> ceros     la mitad ÁRBOL deja de ser correcta
    np.polyfit            -> (1, 0)    el reescalado afín del wrapper se OMITE
                                       (exactamente mi bug original: daba 1e-2, no 1e-17)

En los tres casos el productor **no debe publicar** un artefacto de atribución: debe degradar a
`tree_shap_unavailable` con razón tipada. Que degrade es el criterio; que el JSON exista no lo es.

NO reimplementa la matemática: si la comprobación de aditividad del productor desapareciera, estos
tests caerían solos porque el artefacto pasaría a publicarse con valores falsos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.analysis.generate_interpretability as gi  # noqa: E402

#: Features mínimas que el frame sintético expone. No son las 21 del zoo: al productor le basta
#: con que `cfg.get_feature_columns()` y el frame coincidan, y una fixture chica hace el test
#: rápido sin desactivar ninguna de las comprobaciones que se quieren juzgar.
FEATS = ["close", "open", "high", "low", "return_1d", "volatility_5d", "rsi_14d", "ma_ratio_20d"]


def _frame(n_years: int = 4, per_year: int = 260) -> pd.DataFrame:
    """Frame sintético determinista con señal LINEAL REAL en el target.

    El detalle no es cosmético y me costó un rojo: la primera versión usaba un paseo
    aleatorio puro, así que Ridge predecía ≈0 y LightGBM salía constante. Con la parte
    lineal aportando ≈0, **anularla no rompía la aditividad** y el mutante «mitad lineal
    falseada» pasaba verde — el test no probaba nada. La fixture tiene que hacer que
    AMBAS mitades pesen, o el candado que dice medir no mide.

    Aquí el retorno a 5 días depende de un driver observable (`rsi_14d`), así que el
    término lineal del híbrido tiene señal de verdad que perder.
    """
    rng = np.random.default_rng(20260806)
    n = n_years * per_year
    dates = pd.bdate_range("2021-01-04", periods=n)
    driver = rng.normal(0, 1.0, n)                       # observable en t
    # log-retorno diario = ruido + 1/5 del driver de hace 5 barras  =>  ret_5d(t) ≈ k·driver(t)
    k = 0.01
    inc = rng.normal(0, 0.002, n)
    inc[5:] += k * driver[:-5] / 5.0
    close = 100 * np.exp(np.cumsum(inc))
    df = pd.DataFrame({"date": dates, "close": close})
    df["open"] = close + rng.normal(0, 0.1, n)
    df["high"] = close + np.abs(rng.normal(0, 0.2, n))
    df["low"] = close - np.abs(rng.normal(0, 0.2, n))
    df["return_1d"] = df["close"].pct_change().fillna(0.0)
    df["volatility_5d"] = df["return_1d"].rolling(5, min_periods=1).std().fillna(0.0)
    df["rsi_14d"] = driver          # el DRIVER del target: la parte lineal tiene que pesar
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20, min_periods=1).mean()
    return df


class _Cfg:
    """Stub de config: el productor solo le pide las columnas y la ruta para la huella."""
    _config_path = str(REPO / "config" / "forecasting_ssot.yaml")

    def get_feature_columns(self):
        return list(FEATS)


class _Loader:
    def __init__(self, cfg, project_root=None):
        self._df = _frame()

    def load_dataset(self, target_horizon=None):
        return self._df.copy(), list(FEATS)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Ejecuta los productores de verdad, pero escribiendo en `tmp_path`."""
    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    import src.forecasting.ssot_config as ssot
    import src.forecasting.dataset_loader as dl
    monkeypatch.setattr(ssot.ForecastingSSOTConfig, "load", classmethod(lambda cls, p=None: _Cfg()))
    monkeypatch.setattr(dl, "ForecastingDatasetLoader", _Loader)
    return tmp_path


def _artifacts(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("summary.json"))]


# ───────────────────────────────────────────────── híbridos: el productor se ejecuta

def test_hybrid_producer_runs_and_publishes_exact_attribution(sandbox):
    """Camino sano: el generador CORRE y publica atribución con aditividad de coma flotante."""
    gi.generate_zoo_hybrid(model_ids=("hybrid_lightgbm",), asset="usdcop")
    arts = _artifacts(sandbox)
    assert arts, "el productor no publicó nada"
    a = arts[0]
    assert a["method"] == "hybrid_shap_convex_decomposition", (
        f"esperaba atribución publicada, salió {a['method']!r} ({a.get('detail')})")
    assert a["additivity_max_abs_err"] < 1e-9, a["additivity_max_abs_err"]
    assert 0.0 <= a["hybrid_alpha"] <= 1.0


@pytest.mark.parametrize("costura,parche,motivo", [
    ("_hybrid_linear_half",
     lambda mdl, Xte: (np.zeros((len(Xte), len(FEATS))), 0.0),
     "la mitad LINEAL falseada"),
    ("_tree_shap_backend",
     lambda mid: ("stub", lambda m, X: (np.zeros((len(X), len(FEATS))), np.zeros(len(X)))),
     "la mitad ÁRBOL falseada"),
])
def test_hybrid_producer_refuses_to_publish_when_a_half_is_wrong(
        sandbox, monkeypatch, costura, parche, motivo):
    """Si cualquiera de las dos mitades miente, el productor NO publica atribución.

    Es el candado que vive dentro del generador y que hasta ahora nadie ejecutaba: la
    aditividad se mide contra la predicción del híbrido COMPLETO, así que falsear una mitad
    la rompe aunque la otra sea perfecta.
    """
    monkeypatch.setattr(gi, costura, parche)
    gi.generate_zoo_hybrid(model_ids=("hybrid_lightgbm",), asset="usdcop")
    arts = _artifacts(sandbox)
    assert arts, "no se escribió ni el artefacto degradado"
    a = arts[0]
    assert a["method"] == "tree_shap_unavailable", (
        f"con {motivo} el productor PUBLICÓ atribución (method={a['method']!r}, "
        f"add_err={a.get('additivity_max_abs_err')}): una atribución que no suma a la "
        f"predicción no explica al modelo y no puede publicarse")
    assert a["reason"] == "shap_computation_failed"


def test_hybrid_producer_refuses_when_the_affine_rescale_is_omitted(sandbox, monkeypatch):
    """El reescalado afín del wrapper NO puede ignorarse — fue mi bug original.

    El booster expone su salida cruda a TreeSHAP, pero el wrapper aplica después un
    reescalado de varianza que SÍ entra en la combinación del híbrido. Omitirlo daba
    `add_err` de 1e-2 en vez de 1e-17. Aquí se simula la omisión forzando la recta
    identidad; si el productor dejara de componerla, publicaría igual y este test caería.
    """
    real = np.polyfit

    def sin_reescalado(x, y, deg):
        return np.array([1.0, 0.0]) if deg == 1 else real(x, y, deg)

    monkeypatch.setattr(gi.np, "polyfit", sin_reescalado)
    gi.generate_zoo_hybrid(model_ids=("hybrid_xgboost",), asset="usdcop")
    arts = _artifacts(sandbox)
    assert arts, "no se escribió artefacto"
    a = arts[0]
    if a["method"] == "hybrid_shap_convex_decomposition":
        # Sólo es aceptable si en ESTA fixture el wrapper no llegó a reescalar (la rama
        # sólo dispara con std < 0.005): entonces identidad ES la transformación real.
        assert a["additivity_max_abs_err"] < 1e-9, (
            "publicó atribución con el reescalado omitido Y aditividad rota: el candado "
            "de aditividad dejó de morder")


# ───────────────────────────────────────────────── composite v11: el productor se ejecuta

def _frame25(recipe: list[str]) -> pd.DataFrame:
    """Frame con las 25 columnas REALES de la receta v11.

    El atajo de 8 features fue justo lo que dejo pasar la mentira que CODEX reprodujo
    (`n_features: 8` publicado con `scope` afirmando 25): si el camino sano no usa la
    receta canonica, la guarda de identidad nunca se ejercita.
    """
    base = _frame()
    rng = np.random.default_rng(4242)
    n = len(base)
    for c in recipe:
        if c in base.columns:
            continue
        if c == "day_of_week":
            base[c] = base["date"].dt.dayofweek
        elif c == "month":
            base[c] = base["date"].dt.month
        elif c == "is_month_end":
            base[c] = base["date"].dt.is_month_end.astype(int)
        else:
            base[c] = rng.normal(0, 1.0, n)
    return base


class _Loader25:
    def __init__(self, cfg, project_root=None):
        self._recipe = gi.yaml_safe_load_recipe()
        self._df = _frame25(self._recipe)

    def load_dataset(self, target_horizon=None):
        return self._df.copy(), list(self._recipe)


@pytest.fixture()
def sandbox25(tmp_path, monkeypatch):
    """Sandbox del composite con la receta CANONICA de 25, no un atajo."""
    recipe = gi.yaml_safe_load_recipe()
    monkeypatch.setattr(gi, "OUT_ROOT", tmp_path)
    import src.forecasting.ssot_config as ssot
    import src.forecasting.dataset_loader as dl

    class _C:
        _config_path = str(REPO / "config" / "forecasting_ssot.yaml")

        def get_feature_columns(self):
            return list(recipe)

    monkeypatch.setattr(ssot.ForecastingSSOTConfig, "load", classmethod(lambda cls, p=None: _C()))
    monkeypatch.setattr(dl, "ForecastingDatasetLoader", _Loader25)
    return tmp_path, recipe


def test_composite_producer_runs_on_the_canonical_25_and_declares_both_negatives(
        sandbox25, monkeypatch):
    """Camino sano con la receta CANONICA: 25 medidas, no 25 afirmadas.

    `yaml_safe_load_recipe` NO se parchea aqui — se usa la real, que es lo que hace que la
    guarda de identidad se ejercite de verdad.
    """
    tmp, recipe = sandbox25
    monkeypatch.setattr(
        "src.forecasting.enhance_v2.enhance_features_v2",
        lambda df, base, project_root=None, include_xlead=False: (df, list(recipe)))
    gi.generate_composite_v11()
    a = _artifacts(tmp)[0]
    assert a["surface"] == "composite" and a["model_id"] == "usdcop_ridge_br"
    assert a["n_features"] == 25 == len(recipe), (
        f"el productor publico {a['n_features']} features")
    assert f"recipe25 ({a['n_features']}" in a["scope"], (
        "el scope debe DERIVAR el conteo, no cablearlo: publicar 8 diciendo 25 fue el "
        "defecto que CXD-583 reprodujo")
    assert a["additivity_max_abs_err"] < 1e-9
    assert "NO explica" in a["scope"] and "REGLAS" in a["scope"]
    assert "recipe25" in a["scope"] and "dag_legacy23" in a["scope"]
    assert all("coef" in r for r in a["top_features"])


def test_composite_aborts_when_the_builder_returns_24_with_the_real_recipe(
        sandbox25, monkeypatch):
    """NEGATIVA 1 — receta REAL, builder incompleto: falta una feature declarada."""
    _, recipe = sandbox25
    monkeypatch.setattr(
        "src.forecasting.enhance_v2.enhance_features_v2",
        lambda df, base, project_root=None, include_xlead=False: (df, list(recipe)[:-1]))
    with pytest.raises(RuntimeError, match="builder no produjo"):
        gi.generate_composite_v11()


def test_composite_aborts_when_the_recipe_itself_is_not_the_canonical_25(
        sandbox25, monkeypatch):
    """NEGATIVA 2 — builder y receta COINCIDEN, pero la receta no es la de v11.

    Este es el agujero exacto de CXD-583: con ambos a 24 (o a 8) mi version anterior
    publicaba, porque solo exigia inclusion y nunca identidad. Ahora aborta ANTES.
    """
    _, recipe = sandbox25
    recorte = list(recipe)[:-1]
    monkeypatch.setattr(gi, "yaml_safe_load_recipe", lambda: recorte)
    monkeypatch.setattr(
        "src.forecasting.enhance_v2.enhance_features_v2",
        lambda df, base, project_root=None, include_xlead=False: (df, list(recorte)))
    with pytest.raises(RuntimeError, match="v11 declara"):
        gi.generate_composite_v11()
