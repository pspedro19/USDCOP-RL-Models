"""
Regression: ninguna feature de la tesis mira el futuro.

Contract: CTR-RESEARCH-FEATURES-001 · Date: 2026-08-25

## El test que de verdad importa

Casi cualquier error en este repo produce un número plausible. El look-ahead produce un número
**bueno**, que es peor: nadie audita un backtest que sale bien.

La comprobación central aquí es de perturbación: se corta la serie en `b`, se altera todo lo
posterior, y se recalculan las features. Si alguna cambia en `b` o antes, esa feature está
leyendo el futuro. No hace falta razonar sobre la fórmula — la propiedad se mide.

Es el mismo argumento que `test_regime_hmm_is_causal.py` aplica al HMM, ahora sobre el vector
de observación completo.

## Y el DO-NOT explícito

`CLAUDE.md`: *"Do NOT use pandas `ewm()` for RSI — use Wilder's EMA (alpha=1/period)"*. Se
comprueba contra el RSI calculado a mano sobre el ejemplo canónico de Wilder, y se comprueba
además que NO coincide con el `ewm()` por defecto: un test que solo verificase "es un RSI"
pasaría con la implementación prohibida.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.features import (  # noqa: E402
    ENDOGENOUS, EXCLUDED_GROUPS, FEATURE_ORDER, GROUPS, SCHEMA, SCHEMA_PATH,
    build_market_features, true_range, wilder_atr, wilder_rsi)

MARKET_GROUPS = ("precio", "volatilidad", "tendencia", "temporal")
MARKET_FEATURES = [f for g in MARKET_GROUPS for f in GROUPS[g]]


def synthetic_m5(n_sessions: int = 6, seed: int = 0) -> pd.DataFrame:
    """Sesiones sintéticas de 60 barras, 08:00-12:55 COT, días hábiles."""
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2023-03-06")          # lunes
    for _ in range(n_sessions):
        while day.dayofweek >= 5:
            day += pd.Timedelta(days=1)
        price = 4000.0 + rng.normal(0, 5)
        for b in range(60):
            ts = day + pd.Timedelta(hours=8, minutes=5 * b)
            step = rng.normal(0, 2.0)
            o = price
            c = price + step
            rows.append({"time": ts, "symbol": "USD/COP", "open": o, "close": c,
                         "high": max(o, c) + abs(rng.normal(0, 0.5)),
                         "low": min(o, c) - abs(rng.normal(0, 0.5))})
            price = c
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Causalidad por perturbación
# ---------------------------------------------------------------------------

def test_future_bars_cannot_change_past_features():
    """Alterar el futuro no puede mover ni una feature del pasado."""
    m5 = synthetic_m5()
    base = build_market_features(m5)

    cut = 200                                  # dentro de la 4a sesión
    tampered = m5.copy()
    tampered.loc[cut:, ["open", "high", "low", "close"]] *= 1.25

    after = build_market_features(tampered)
    assert list(base.index) == list(after.index)

    diff = (base[MARKET_FEATURES].iloc[:cut] - after[MARKET_FEATURES].iloc[:cut]).abs()
    leaking = diff.max()[diff.max() > 1e-9]
    assert leaking.empty, (
        "estas features cambian en el PASADO al alterar el futuro — look-ahead:\n"
        + leaking.to_string()
    )


def test_truncating_the_series_leaves_earlier_features_identical():
    """Calcular con 6 sesiones o con 4 debe dar lo mismo en las 4 primeras.

    Si difiere, algo se normaliza con estadísticos de toda la muestra — la fuga que
    §11 llama "normalización global" y que no deja rastro en ninguna tabla.
    """
    m5 = synthetic_m5()
    full = build_market_features(m5)
    truncated = build_market_features(m5.iloc[: 4 * 60])

    common = truncated.index
    diff = (full.loc[common, MARKET_FEATURES] - truncated[MARKET_FEATURES]).abs()
    offenders = diff.max()[diff.max() > 1e-9]
    assert offenders.empty, (
        "features que dependen de datos posteriores al truncar:\n" + offenders.to_string()
    )


def test_session_scoped_features_reset_each_day():
    """`ret_sesion_acum` y `close_pos_rango` son de sesión: en la barra 0 valen lo trivial.

    Si arrastrasen el día anterior, el agente vería la sesión previa a través de una feature
    que el plan declara intradía (§9.1 fija `w_{-1}=0`; el estado también se reinicia).
    """
    f = build_market_features(synthetic_m5())
    first_bars = f.groupby("_session").head(1)
    assert np.allclose(first_bars["ret_sesion_acum"], 0.0, atol=1e-12)
    assert ((first_bars["close_pos_rango"] >= 0.0)
            & (first_bars["close_pos_rango"] <= 1.0)).all()


def test_no_nan_or_inf_reaches_the_observation_vector():
    f = build_market_features(synthetic_m5())
    cols = [c for c in f.columns if c != "_session"]
    assert np.isfinite(f[cols].to_numpy()).all(), "hay NaN/Inf en el vector de observación"


# ---------------------------------------------------------------------------
# RSI de Wilder — DO-NOT explícito de CLAUDE.md
# ---------------------------------------------------------------------------

def test_rsi_uses_wilder_smoothing_not_pandas_default_ewm():
    """`alpha = 1/period`, no `2/(period+1)`. Para period=14: 0,0714 frente a 0,1333."""
    rng = np.random.default_rng(7)
    close = pd.Series(4000 + np.cumsum(rng.normal(0, 3, 400)))

    ours = wilder_rsi(close, 14)

    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    wilder = (100 - 100 / (1 + gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
                           / loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()))
    assert np.allclose(ours.iloc[20:], wilder.iloc[20:], atol=1e-9)

    forbidden = (100 - 100 / (1 + gain.ewm(span=14, adjust=False).mean()
                              / loss.ewm(span=14, adjust=False).mean()))
    assert not np.allclose(ours.iloc[20:], forbidden.iloc[20:], atol=1e-3), (
        "el RSI coincide con el ewm() por defecto de pandas — es el DO-NOT de CLAUDE.md"
    )


def test_rsi_is_bounded_and_reacts_to_direction():
    up = pd.Series(np.linspace(4000, 4200, 200))
    down = pd.Series(np.linspace(4200, 4000, 200))
    assert wilder_rsi(up).iloc[-1] > 95.0
    assert wilder_rsi(down).iloc[-1] < 5.0
    r = wilder_rsi(pd.Series(4000 + np.cumsum(np.random.default_rng(1).normal(0, 3, 300))))
    assert r.between(0, 100).all()


# ---------------------------------------------------------------------------
# El esquema es el contrato
# ---------------------------------------------------------------------------

def test_schema_file_matches_the_code():
    """Si el JSON publicado y el código divergen, el vector deja de ser reproducible."""
    import json

    assert SCHEMA_PATH.is_file(), "falta feature_schema.json: ejecuta write_schema()"
    on_disk = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert on_disk["sha256"] == SCHEMA.sha256, (
        "el hash del esquema en disco no coincide con el código — regenera el JSON"
    )
    assert on_disk["order"] == list(FEATURE_ORDER)
    assert on_disk["n_features"] == len(FEATURE_ORDER)


def test_feature_names_are_unique_and_grouped_exactly_once():
    assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER), "hay nombres duplicados"
    flat = [f for g in GROUPS.values() for f in g]
    assert sorted(flat) == sorted(FEATURE_ORDER)


def test_excluded_groups_state_the_criterion_that_removed_them():
    """§6.5 elimina grupos por criterio medido, no por preferencia: que quede escrito."""
    assert {"volumen", "sorpresa_macro"} <= set(EXCLUDED_GROUPS)
    for name, reason in EXCLUDED_GROUPS.items():
        assert len(reason) > 40, f"{name}: la razón de exclusión debe ser verificable"


def test_position_features_are_flagged_endogenous():
    """§10.6: dependen de la trayectoria del agente, no del mercado.

    Un modelo no secuencial no puede consumirlas sin construirse una posición que nunca tuvo,
    así que la frontera tiene que estar marcada en el esquema, no en la memoria de alguien.
    """
    assert ENDOGENOUS == set(GROUPS["posicion"])
    assert ENDOGENOUS.isdisjoint(set(MARKET_FEATURES))


def test_macro_uses_strictly_prior_observation(tmp_path, monkeypatch):
    """La fila macro de d no puede entrar en la decisión de la misma sesión."""
    import src.research.features as features_module

    dates = pd.date_range("2023-06-14", periods=2, freq="D")
    macro = pd.DataFrame({
        "COMM_OIL_BRENT_GLB_D_BRENT": [70.0, 77.0],
        "FXRT_INDEX_DXY_USA_D_DXY": [100.0, 110.0],
        "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR": [10.0, 11.0],
        "FINC_BOND_YIELD2Y_USA_D_DGS2": [4.0, 5.0],
    }, index=dates)
    path = tmp_path / "macro.parquet"
    macro.to_parquet(path)
    monkeypatch.setattr(features_module, "MACRO_CLEAN", path)

    out = features_module.attach_macro_features([pd.Timestamp("2023-06-15")])
    # The only usable DXY return is 2023-06-15? No: the session must use the
    # return observed on 2023-06-14, which needs a still earlier level; with
    # only two rows this is unavailable and remains NaN: missing macro must not
    # be silently converted into a neutral zero.
    assert pd.isna(out.loc[pd.Timestamp("2023-06-15"), "dxy_ret_prev"])


def test_macro_artifact_missing_fails_closed(monkeypatch, tmp_path):
    import src.research.features as features_module

    monkeypatch.setattr(features_module, "MACRO_CLEAN", tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError):
        features_module.attach_macro_features([pd.Timestamp("2023-06-15")])


def test_macro_staleness_remains_missing_instead_of_zero_fill(monkeypatch, tmp_path):
    import src.research.features as features_module

    dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    macro = pd.DataFrame({
        "COMM_OIL_BRENT_GLB_D_BRENT": [70.0, 71.0],
        "FXRT_INDEX_DXY_USA_D_DXY": [100.0, 101.0],
        "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR": [10.0, 10.0],
        "FINC_BOND_YIELD2Y_USA_D_DGS2": [4.0, 4.0],
    }, index=dates)
    path = tmp_path / "macro.parquet"
    macro.to_parquet(path)
    monkeypatch.setattr(features_module, "MACRO_CLEAN", path)
    out = features_module.attach_macro_features([pd.Timestamp("2023-01-12")])
    assert out.isna().all(axis=None)


def test_dataset_close_cache_is_invalidated_when_input_changes():
    """La cache de cierres no puede servir datos de otro parquet en el mismo proceso."""
    import src.research.dataset as dataset_module

    first = synthetic_m5(n_sessions=1, seed=31)
    second = first.copy()
    second.loc[second.index[-1], "close"] += 17.0
    day = pd.Timestamp(first["time"].iloc[0]).date()
    assert dataset_module._closes(first, day)[-1] != dataset_module._closes(second, day)[-1]


def test_atr_uses_wilder_rma_with_explicit_seed():
    h = pd.Series([10.0, 12.0, 13.0, 15.0, 14.0])
    l = pd.Series([9.0, 10.0, 11.0, 12.0, 12.0])
    c = pd.Series([9.5, 11.0, 12.0, 13.0, 13.0])
    got = wilder_atr(h, l, c, period=3)
    tr = true_range(h, l, c).to_numpy()
    seed = tr[:3].mean()
    expected = (1 - 1 / 3) * seed + (1 / 3) * tr[3]
    assert got.iloc[2] == pytest.approx(seed)
    assert got.iloc[3] == pytest.approx(expected)
    assert got.iloc[3] != pytest.approx(tr[1:4].mean())


def test_intra_session_returns_exclude_overnight_gap():
    base = synthetic_m5(n_sessions=3, seed=19)
    dates = pd.to_datetime(base["time"]).dt.date.unique()
    first_next = dates[1]
    original = build_market_features(base)
    altered = base.copy()
    prior = pd.to_datetime(altered["time"]).dt.date == dates[0]
    altered.loc[prior, ["open", "high", "low", "close"]] *= 1.05
    changed = build_market_features(altered)
    b0 = changed.index.date == first_next
    assert changed.loc[b0, "logret_1"].iloc[0] == 0.0
    assert changed.loc[b0, "rv_12"].iloc[0] == pytest.approx(
        original.loc[b0, "rv_12"].iloc[0]
    )


def test_intraday_volatility_window_does_not_import_prior_session_returns():
    base = synthetic_m5(n_sessions=3, seed=23)
    dates = pd.to_datetime(base["time"]).dt.date.unique()
    original = build_market_features(base)
    altered = base.copy()
    prior = pd.to_datetime(altered["time"]).dt.date == dates[0]
    altered.loc[prior, ["open", "high", "low", "close"]] *= 3.0
    changed = build_market_features(altered)
    next_session = changed.index.date == dates[1]
    # All of the first 12 bars must be identical: no prior-session return can
    # enter an intraday volatility window.
    assert changed.loc[next_session, "rv_12"].iloc[:12].to_numpy() == pytest.approx(
        original.loc[next_session, "rv_12"].iloc[:12].to_numpy(), nan_ok=True
    )
