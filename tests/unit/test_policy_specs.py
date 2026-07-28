"""BL-47 — specs de política + equivalencia semántica con el código congelado.

Estos tests son el guardarraíl de la migración: si una política del motor
nuevo decide algo DISTINTO de lo que decide el productor congelado, la
migración dejó de ser migración y pasó a ser modelado (BL-47, 0 trials).

No requieren infraestructura: series sintéticas + los módulos congelados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.policy import PolicyContext
from src.strategies.policies.loader import (
    ALLOWED_MODULE_PREFIX,
    PolicySpecError,
    build_policy,
    canonical_policy_hash,
    load_all_policy_specs,
    load_policy_spec,
    policy_specs_dir,
    validate_policy_spec,
)

SPECS = {s["id"]: s for s in load_all_policy_specs()}


# --------------------------------------------------------------------------- §11
def test_todos_los_specs_cargan_y_validan():
    assert SPECS, "config/policies/ no puede estar vacío"
    for spec in SPECS.values():
        validate_policy_spec(spec)


def test_rule_based_implica_retrain_never_y_sin_pesos():
    for spec in SPECS.values():
        if spec["engine"]["type"] != "rule_based":
            continue
        assert spec["engine"]["retrain"] == "never"
        assert "train" not in (spec.get("capabilities") or [])
        assert not spec["engine"].get("model_snapshot_id")


def test_toda_policy_referencia_feature_set_y_resample():
    for spec in SPECS.values():
        assert spec["inputs"]["feature_set_id"]
        assert spec["inputs"]["resample_policy_id"]


def test_fallbacks_explicitos():
    for spec in SPECS.values():
        assert spec["policy"]["missing_input_policy"] in ("FAIL_CLOSED", "FLAT")
        assert spec["policy"]["stale_input_policy"] in ("FAIL_CLOSED", "FLAT", "HOLD")
        assert "default_target_exposure" in spec["policy"]["resolution"]


def test_cop_es_composite_no_ml():
    """Invariante 3: v11 es composite; el predictor es UN componente."""
    assert SPECS["smart_simple_v11"]["engine"]["type"] == "composite"


def test_spec_only_falla_cerrado():
    with pytest.raises(PolicySpecError):
        build_policy(SPECS["smart_simple_v11"])


def test_policy_hash_declarado_coincide_con_el_contenido():
    for spec in SPECS.values():
        declared = spec["governance"].get("policy_hash")
        if declared:
            assert declared == canonical_policy_hash(spec)


def test_cambiar_un_parametro_cambia_el_policy_hash():
    """Congelar la receta ES congelar la estrategia (invariante 2)."""
    spec = dict(SPECS["gold_trend_simple"])
    before = canonical_policy_hash(spec)
    mutated = {**spec, "policy": {**spec["policy"],
                                  "params": {**spec["policy"]["params"], "min_votes": 3}}}
    assert canonical_policy_hash(mutated) != before


def test_presentation_no_cambia_el_policy_hash():
    """§9.1: cambiar una etiqueta no crea una versión económica nueva."""
    spec = SPECS["btc_hodl_b1"]
    mutated = {**spec, "presentation": {"engine_label": "otra etiqueta"}}
    assert canonical_policy_hash(mutated) == canonical_policy_hash(spec)


def test_modulo_fuera_del_allowlist_es_rechazado():
    spec = {**SPECS["btc_hodl_b1"]}
    spec["engine"] = {**spec["engine"],
                      "implementation": {"mode": "coded_policy", "module": "os:system"}}
    spec["governance"] = {**spec["governance"], "policy_hash": None}
    with pytest.raises(PolicySpecError, match=ALLOWED_MODULE_PREFIX):
        validate_policy_spec(spec)


def test_dsl_rechaza_codigo_en_el_yaml(tmp_path):
    base = (policy_specs_dir() / "spx500_daily_ma200_v1.yaml").read_text(encoding="utf-8")
    hostile = base.replace(
        "      when:\n        operator: greater_than\n"
        "        left: feature.close\n        right: feature.ma_200",
        '      when: "eval(close > ma_200)"')
    path = tmp_path / "hostile.yaml"
    path.write_text(hostile, encoding="utf-8")
    with pytest.raises((PolicySpecError, ValueError)):
        load_policy_spec(path)


# --------------------------------------------------------------------------- R6
def _synthetic_close(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))))


def test_ma200_declarativa_equivale_a_la_regla_congelada():
    """(close > MA200) vectorizado == motor de políticas, barra a barra."""
    close = _synthetic_close()
    ma200 = close.rolling(200, min_periods=200).mean()
    legacy = (close > ma200).astype(float).to_numpy()

    policy = build_policy(SPECS["spx500_daily_ma200_v1"])
    engine = np.zeros(len(close))
    for i in range(len(close)):
        if not np.isfinite(ma200.iloc[i]):
            continue                     # warmup: sin snapshot => sin decisión => 0
        decision = policy.evaluate(
            {"close": float(close.iloc[i]), "ma_200": float(ma200.iloc[i])},
            PolicyContext(as_of="2026-07-28"))
        engine[i] = decision.target_exposure
    assert np.array_equal(legacy, engine)


def test_ma200_emite_trace_con_ganadora_o_fallback():
    policy = build_policy(SPECS["spx500_daily_ma200_v1"])
    on = policy.evaluate({"close": 10.0, "ma_200": 9.0}, PolicyContext(as_of="2026-07-28"))
    off = policy.evaluate({"close": 8.0, "ma_200": 9.0}, PolicyContext(as_of="2026-07-28"))
    assert on.direction == "LONG" and on.target_exposure == 1.0
    assert on.rule_trace.winning_rule_id == "close_above_ma200"
    assert off.direction == "FLAT" and off.target_exposure == 0.0
    assert off.rule_trace.fallback_applied is True


# --------------------------------------------------------------------------- R7
def test_gold_trend_simple_equivale_al_productor_congelado():
    """Camino A (scripts/analysis/gold_trend_simple.simulate), el del bundle."""
    from scripts.analysis.gold_trend_simple import simulate

    close = _synthetic_close(500, seed=11)
    feat = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(close), freq="D"),
        "close": close.to_numpy(float),
        "realized_vol_20": np.clip(
            close.pct_change().rolling(20).std().to_numpy(float) * np.sqrt(252), 0, None),
    })
    legacy_run = simulate(feat)
    legacy = np.roll(legacy_run["position"].to_numpy(float), -1)   # decisión de t

    policy = build_policy(SPECS["gold_trend_simple"])
    engine = np.zeros(len(feat))
    for i in range(len(feat)):
        snap = {
            "close": float(feat["close"].iloc[i]),
            "sma_63": float(close.rolling(63).mean().iloc[i]),
            "sma_126": float(close.rolling(126).mean().iloc[i]),
            "sma_252": float(close.rolling(252).mean().iloc[i]),
            "realized_vol_20": float(feat["realized_vol_20"].iloc[i]),
        }
        if any(not np.isfinite(v) for v in snap.values()):
            continue
        engine[i] = policy.evaluate(snap, PolicyContext(as_of="2026-07-28")).target_exposure

    # Paridad EXACTA fuera de la ventana de calentamiento declarada.
    warmup = int(SPECS["gold_trend_simple"]["inputs"]["warmup_bars"])
    assert np.array_equal(legacy[warmup:-1], engine[warmup:-1])

    # Y la divergencia de warmup está ACOTADA a esa ventana: si algún día
    # apareciera fuera de ella, este test cae y la migración se para.
    diverge = np.flatnonzero(legacy[:-1] != engine[:-1])
    assert diverge.size == 0 or diverge.max() < warmup


def test_btc_hodl_equivale_al_productor_congelado():
    from src.btc_strategy.strategies import build_positions, intent_hodl

    rng = np.random.default_rng(3)
    n = 300
    feat = pd.DataFrame({
        "realized_vol_20": rng.uniform(0.2, 1.5, n),
        "regime_risk_mult": rng.choice([1.0, 0.8, 0.5, 0.35], n),
    })
    legacy_run = build_positions(feat, intent_hodl)
    legacy = np.roll(legacy_run["position"].to_numpy(float), -1)

    policy = build_policy(SPECS["btc_hodl_b1"])
    engine = np.array([
        policy.evaluate(
            {"realized_vol_20": float(feat["realized_vol_20"].iloc[i]),
             "regime_risk_mult": float(feat["regime_risk_mult"].iloc[i])},
            PolicyContext(as_of="2026-07-28")).target_exposure
        for i in range(n)
    ])
    assert np.array_equal(legacy[:-1], engine[:-1])


def test_input_no_finito_falla_cerrado():
    """missing_input_policy: FAIL_CLOSED — nunca una posición inventada."""
    policy = build_policy(SPECS["btc_hodl_b1"])
    with pytest.raises(ValueError):
        policy.evaluate({"realized_vol_20": float("nan"), "regime_risk_mult": 1.0},
                        PolicyContext(as_of="2026-07-28"))


def test_determinismo_mismo_snapshot_misma_decision():
    for policy_id in ("spx500_daily_ma200_v1", "gold_trend_simple", "btc_hodl_b1"):
        policy = build_policy(SPECS[policy_id])
        snap = {name: 1.0 for name in policy.required_features()}
        a = policy.evaluate(snap, PolicyContext(as_of="2026-07-28"))
        b = policy.evaluate(dict(snap), PolicyContext(as_of="2026-07-28"))
        assert a.decision_fingerprint == b.decision_fingerprint
