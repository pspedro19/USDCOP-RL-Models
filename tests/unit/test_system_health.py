"""BL-25 — Monitoreo en tres relojes (control__system_health).

Verificación del BL (fail-first):
  1. Inyectar drift sintético  => promoción congelada (model clock, PSI > 0.25).
  2. TE live-vs-paper > 3σ     => evento `withdrawal_protocol_triggered` (PnL clock).

Además: tabla FABRIC §23.1 como contrato (12 filas), fail-closed del reloj de
datos, métrica ausente => N/A (jamás imputar en silencio) y JSON safety.

Contract: CTR-SYSTEM-HEALTH-001 (spec: .claude/specs/planes/04-CTR-QLAB-FABRIC-004.md §23)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.system_health import (
    PromotionsFrozenError,
    SystemHealthEngine,
    assert_promotions_not_frozen,
    check_promotions_frozen,
    compute_psi,
    write_snapshot,
)
from src.monitoring.system_health_contract import (
    FAILURE_TABLE,
    PSI_FREEZE_THRESHOLD,
    SHARPE_ROLLING_MIN_RATIO,
    SLIPPAGE_FACTOR_MAX,
    TRACKING_ERROR_SIGMA,
    Clock,
    DataProbe,
    HealthAction,
    HealthSignal,
)

RNG = np.random.default_rng(42)


@pytest.fixture()
def engine() -> SystemHealthEngine:
    return SystemHealthEngine()


# ---------------------------------------------------------------------------
# Reloj de MODELO — drift sintético congela promociones (Verificación BL-25 #1)
# ---------------------------------------------------------------------------

class TestModelClock:
    def test_psi_thresholds_are_exante_priors(self):
        # FABRIC §23: PSI > 0.25 = drift de features. NO tunear (0 trials).
        assert PSI_FREEZE_THRESHOLD == 0.25
        assert TRACKING_ERROR_SIGMA == 3.0
        assert SHARPE_ROLLING_MIN_RATIO == 0.5
        assert SLIPPAGE_FACTOR_MAX == 2.0

    def test_compute_psi_detects_shift(self):
        ref = RNG.normal(0.0, 1.0, 5000)
        shifted = RNG.normal(3.0, 1.0, 1000)
        same = RNG.normal(0.0, 1.0, 1000)
        assert compute_psi(ref, shifted) > 0.25
        assert compute_psi(ref, same) < 0.10

    def test_synthetic_drift_freezes_promotions(self, engine, tmp_path):
        ref = {"ret_1d": RNG.normal(0.0, 1.0, 5000)}
        cur = {"ret_1d": RNG.normal(3.0, 1.0, 500)}  # drift sintético inyectado

        status = engine.evaluate_model_clock(reference_features=ref, current_features=cur)

        assert status.clock == Clock.MODEL
        assert status.signal == HealthSignal.YELLOW
        assert HealthAction.FREEZE_PROMOTIONS in status.actions
        assert any(e.kind == "model_drift_psi" for e in status.events)

        snapshot = engine.build_snapshot(model=status)
        assert snapshot.promotions_frozen is True

        # El gate de promoción (Vote 1) debe BLOQUEAR sobre este snapshot.
        path = tmp_path / "system_health.json"
        write_snapshot(snapshot, path)
        frozen, reason = check_promotions_frozen(path)
        assert frozen is True
        with pytest.raises(PromotionsFrozenError):
            assert_promotions_not_frozen(path)

    def test_no_drift_keeps_promotions_open(self, engine, tmp_path):
        ref = {"ret_1d": RNG.normal(0.0, 1.0, 5000)}
        cur = {"ret_1d": RNG.normal(0.0, 1.0, 500)}
        status = engine.evaluate_model_clock(reference_features=ref, current_features=cur)
        assert status.signal == HealthSignal.GREEN
        assert HealthAction.FREEZE_PROMOTIONS not in status.actions

        snapshot = engine.build_snapshot(model=status)
        assert snapshot.promotions_frozen is False
        path = tmp_path / "system_health.json"
        write_snapshot(snapshot, path)
        frozen, _ = check_promotions_frozen(path)
        assert frozen is False
        assert_promotions_not_frozen(path)  # no levanta

    def test_prediction_drift_beyond_2_sigma(self, engine):
        ref = {"ret_1d": RNG.normal(0.0, 1.0, 2000)}
        cur = {"ret_1d": RNG.normal(0.0, 1.0, 300)}
        pred_ref = RNG.normal(0.0, 0.01, 500)
        pred_cur = np.full(20, 0.10)  # media fuera de ±2σ del histórico
        status = engine.evaluate_model_clock(
            reference_features=ref,
            current_features=cur,
            predictions_reference=pred_ref,
            predictions_current=pred_cur,
        )
        assert status.signal == HealthSignal.YELLOW
        assert any(e.kind == "prediction_drift" for e in status.events)

    def test_missing_snapshot_does_not_block_bootstrap(self, tmp_path):
        # Decisión documentada: sin snapshot (monitoring nunca corrió) la promoción
        # NO se bloquea — se advierte. El fail-closed duro es del reloj de DATOS.
        missing = tmp_path / "nope.json"
        frozen, reason = check_promotions_frozen(missing)
        assert frozen is False
        assert "snapshot" in reason.lower()
        assert_promotions_not_frozen(missing)  # warn, no exception


# ---------------------------------------------------------------------------
# Reloj de PnL — TE > 3σ dispara withdrawal (Verificación BL-25 #2)
# ---------------------------------------------------------------------------

class TestPnlClock:
    def test_tracking_error_above_3_sigma_triggers_withdrawal(self, engine):
        n = 52
        paper = RNG.normal(0.001, 0.01, n)
        live = paper - 0.02  # desviación acumulada sistemática vs paper

        status = engine.evaluate_pnl_clock(live_returns=live, paper_returns=paper)

        assert status.clock == Clock.PNL
        assert status.signal == HealthSignal.ORANGE
        assert HealthAction.TRIGGER_WITHDRAWAL in status.actions
        assert any(e.kind == "withdrawal_protocol_triggered" for e in status.events)

        snapshot = engine.build_snapshot(pnl=status)
        assert snapshot.withdrawal_triggered is True

    def test_matched_live_paper_is_green(self, engine):
        n = 52
        paper = RNG.normal(0.001, 0.01, n)
        live = paper + RNG.normal(0.0, 0.0005, n)
        status = engine.evaluate_pnl_clock(live_returns=live, paper_returns=paper)
        assert status.signal == HealthSignal.GREEN
        assert HealthAction.TRIGGER_WITHDRAWAL not in status.actions

    def test_sharpe_decay_flags_reduced(self, engine):
        n = 52
        paper = RNG.normal(0.0, 0.01, n)
        live = paper + RNG.normal(0.0, 0.0005, n)
        status = engine.evaluate_pnl_clock(
            live_returns=live,
            paper_returns=paper,
            rolling_sharpe=0.5,
            backtest_sharpe=3.0,  # rolling < 50% del backtest
        )
        assert status.signal == HealthSignal.ORANGE
        assert HealthAction.REDUCED in status.actions
        assert any(e.kind == "sharpe_decay" for e in status.events)

    def test_slippage_2x_modeled_flags(self, engine):
        n = 52
        paper = RNG.normal(0.0, 0.01, n)
        status = engine.evaluate_pnl_clock(
            live_returns=paper,
            paper_returns=paper,
            realized_slippage_bps=2.5,
            modeled_slippage_bps=1.0,
        )
        assert status.signal == HealthSignal.ORANGE
        assert any(e.kind == "slippage_excess" for e in status.events)

    def test_missing_series_is_na_never_imputed(self, engine):
        # §23.1: métrica ausente => N/A; jamás imputar en silencio.
        status = engine.evaluate_pnl_clock(live_returns=None, paper_returns=None)
        assert status.signal == HealthSignal.N_A
        assert any(e.kind == "metric_missing" for e in status.events)
        assert HealthAction.TRIGGER_WITHDRAWAL not in status.actions


# ---------------------------------------------------------------------------
# Reloj de PnL — el UMBRAL 3σ anclado (no solo la rama)
# ---------------------------------------------------------------------------
#
# `test_tracking_error_above_3_sigma_triggers_withdrawal` usa `live = paper - 0.02`,
# una diferencia CONSTANTE: sd(d) queda en ruido de coma flotante (~1e-18) y el
# z-score sale ~1e16. Eso demuestra que la rama EXISTE, no que el umbral sea 3 —
# con `TRACKING_ERROR_SIGMA * 1000` (=3000) el test sigue verde porque 1e16 > 3000.
#
# Los dos gemelos de abajo tienen dispersión REAL (ruido independiente en live, no
# `paper − constante`) y están calibrados a ≈3.5σ y ≈2.5σ: el 3 queda ENCERRADO
# entre ellos, así que mover el umbral en cualquier dirección rompe uno de los dos.

TE_SCENARIO_SEED = 20260728          # semilla fija: los dos gemelos son deterministas
TE_DRIFT_ABOVE = 0.0012435           # calibrado => te_z ≈ 3.50σ  (debe DISPARAR)
TE_DRIFT_BELOW = 0.0006830           # calibrado => te_z ≈ 2.47σ  (debe quedar VERDE)


def _te_scenario(drift: float, n: int = 52, noise: float = 0.004):
    """live = paper − drift + ruido INDEPENDIENTE.

    A diferencia de `paper − constante`, aquí sd(d) es una dispersión de verdad
    (~4e-3), así que `te_z = max_k |cumsum(d)_k| / (sd·√k)` es un z-score con
    sentido y no una división por el epsilon de la máquina.
    """
    rng = np.random.default_rng(TE_SCENARIO_SEED)
    paper = rng.normal(0.0015, 0.010, n)
    live = paper - drift + rng.normal(0.0, noise, n)
    return live, paper


class TestPnlClockThresholdIsAnchored:
    def test_te_scenarios_have_real_dispersion_not_floating_point_noise(self):
        # rojo si el escenario degenera a `paper − constante`: sd(d) ~1e-18 y el
        # z-score deja de medir nada (es el defecto que estos gemelos cierran).
        for drift in (TE_DRIFT_ABOVE, TE_DRIFT_BELOW):
            live, paper = _te_scenario(drift)
            sd = float(np.std(live - paper, ddof=1))
            assert sd > 1e-3, f"sd(d)={sd:.3g} — diferencia casi constante, z-score vacío"

    def test_tracking_error_at_3_5_sigma_triggers_withdrawal(self, engine):
        # rojo con: TRACKING_ERROR_SIGMA * 1000 en system_health.py:338 (3.5 < 3000)
        live, paper = _te_scenario(TE_DRIFT_ABOVE)
        status = engine.evaluate_pnl_clock(live_returns=live, paper_returns=paper)

        te_z = status.metrics["tracking_error_zscore"]
        assert 3.4 <= te_z <= 3.6, (
            f"escenario descalibrado: te_z={te_z:.4f} deberia estar ≈3.5σ — sin esa "
            "calibracion el test no ancla el umbral, solo la rama")
        assert te_z > TRACKING_ERROR_SIGMA
        assert status.signal == HealthSignal.ORANGE
        assert HealthAction.TRIGGER_WITHDRAWAL in status.actions
        assert any(e.kind == "withdrawal_protocol_triggered" for e in status.events)
        assert engine.build_snapshot(pnl=status).withdrawal_triggered is True

    def test_tracking_error_at_2_5_sigma_stays_green(self, engine):
        # rojo con: TRACKING_ERROR_SIGMA / 1000 (o cualquier umbral < 2.47) en system_health.py:338
        live, paper = _te_scenario(TE_DRIFT_BELOW)
        status = engine.evaluate_pnl_clock(live_returns=live, paper_returns=paper)

        te_z = status.metrics["tracking_error_zscore"]
        assert 2.4 <= te_z <= 2.6, (
            f"escenario descalibrado: te_z={te_z:.4f} deberia estar ≈2.5σ — el gemelo "
            "por debajo es lo que impide bajar el umbral sin romper nada")
        assert te_z < TRACKING_ERROR_SIGMA
        assert status.signal == HealthSignal.GREEN
        assert HealthAction.TRIGGER_WITHDRAWAL not in status.actions
        assert not any(e.kind == "withdrawal_protocol_triggered" for e in status.events)
        assert engine.build_snapshot(pnl=status).withdrawal_triggered is False


# ---------------------------------------------------------------------------
# Reloj de DATOS — fail-closed / QUARANTINE
# ---------------------------------------------------------------------------

class TestDataClock:
    def test_stale_active_component_is_red_fail_closed(self, engine):
        probes = [
            DataProbe(name="ohlcv_m5", status="stale", active_component=True),
            DataProbe(name="macro_daily", status="ok", active_component=True),
        ]
        status = engine.evaluate_data_clock(probes)
        assert status.clock == Clock.DATA
        assert status.signal == HealthSignal.RED
        assert HealthAction.FAIL_CLOSED in status.actions

    def test_parity_broken_quarantines(self, engine):
        probes = [DataProbe(name="replay_parity", status="parity_broken", active_component=True)]
        status = engine.evaluate_data_clock(probes)
        assert status.signal == HealthSignal.RED
        assert HealthAction.QUARANTINE in status.actions

    def test_diagnostic_only_failure_degrades_gracefully(self, engine):
        # §23.1: modelo diagnóstico falla => el resto continúa, sin efecto ACTION.
        probes = [
            DataProbe(name="forecast_zoo_csv", status="stale", active_component=False),
            DataProbe(name="ohlcv_m5", status="ok", active_component=True),
        ]
        status = engine.evaluate_data_clock(probes)
        assert status.signal == HealthSignal.YELLOW
        assert HealthAction.FAIL_CLOSED not in status.actions

    def test_probe_error_on_active_component_fails_closed(self, engine):
        # Fail-safe: si el probe revienta (DB caída) sobre componente activo => RED.
        probes = [DataProbe(name="ohlcv_m5", status="error", active_component=True)]
        status = engine.evaluate_data_clock(probes)
        assert status.signal == HealthSignal.RED
        assert HealthAction.FAIL_CLOSED in status.actions

    def test_all_green(self, engine):
        probes = [DataProbe(name="ohlcv_m5", status="ok", active_component=True)]
        status = engine.evaluate_data_clock(probes)
        assert status.signal == HealthSignal.GREEN
        assert status.actions == []


# ---------------------------------------------------------------------------
# Tabla §23.1 como contrato
# ---------------------------------------------------------------------------

class TestFailureTable:
    def test_twelve_failure_modes(self):
        assert len(FAILURE_TABLE) == 12

    def test_key_rows_match_fabric(self):
        by_id = {f.failure_id: f for f in FAILURE_TABLE}
        assert by_id["active_component_failure"].health_action == HealthAction.FAIL_CLOSED
        assert by_id["parity_broken"].health_action == HealthAction.QUARANTINE
        assert by_id["broker_mismatch"].health_action == HealthAction.QUARANTINE
        assert by_id["metric_missing"].health_action == HealthAction.GATE_NO_APPROVE
        assert by_id["png_missing"].health_action == HealthAction.NONE
        # La predicción sin lift NO invalida automáticamente PnL positivo.
        assert by_id["prediction_without_lift"].health_action == HealthAction.NONE

    def test_failure_ids_unique(self):
        ids = [f.failure_id for f in FAILURE_TABLE]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Snapshot — JSON safety + carry-forward de relojes no vencidos
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_json_is_safe(self, engine, tmp_path):
        ref = {"f": RNG.normal(0, 1, 1000)}
        cur = {"f": np.concatenate([RNG.normal(0, 1, 200), [np.nan]])}
        model = engine.evaluate_model_clock(reference_features=ref, current_features=cur)
        snapshot = engine.build_snapshot(model=model)
        path = tmp_path / "s.json"
        write_snapshot(snapshot, path)
        raw = path.read_text(encoding="utf-8")
        assert "Infinity" not in raw and "NaN" not in raw
        parsed = json.loads(raw)
        assert parsed["contract"] == "CTR-SYSTEM-HEALTH-001"
        assert set(parsed["clocks"]).issubset({"data", "model", "pnl"})

    def test_carry_forward_previous_clocks(self, engine, tmp_path):
        # Cadencias distintas: un run que solo evalúa DATA arrastra MODEL/PnL previos.
        ref = {"f": RNG.normal(0, 1, 1000)}
        cur = {"f": RNG.normal(3, 1, 300)}
        model = engine.evaluate_model_clock(reference_features=ref, current_features=cur)
        first = engine.build_snapshot(model=model)
        path = tmp_path / "s.json"
        write_snapshot(first, path)

        data = engine.evaluate_data_clock([DataProbe(name="x", status="ok", active_component=True)])
        second = engine.build_snapshot(data=data, previous_path=path)
        assert second.promotions_frozen is True  # el freeze del modelo persiste
        assert "model" in {k for k in second.clocks}

    def test_snapshot_never_contains_nan_zscore(self, engine):
        status = engine.evaluate_pnl_clock(
            live_returns=np.zeros(30), paper_returns=np.zeros(30)
        )
        z = status.metrics.get("tracking_error_zscore")
        assert z is None or math.isfinite(z)


# ---------------------------------------------------------------------------
# BL-18 — el docstring de HealthEvent prometía una inserción imposible
# (CXD-365/370 · brief DECISION-BL18 §4-5)
# ---------------------------------------------------------------------------

# Columnas NOT NULL de `control.metric_event` (database/migrations/070_fabric_control_plane.sql)
# que `HealthEvent` no tiene ni puede derivar. Si alguna llega a existir como campo,
# este candado cae y obliga a revisar el docstring en vez de dejarlo mentir.
_METRIC_EVENT_IDENTITY_NOT_NULL = (
    "catalog_version",
    "formula_version",
    "entity_type",
    "entity_id",
    "metric_namespace",
)


def _doc_normalizado(cls) -> str:
    """Docstring con espacios colapsados: las frases del contrato cruzan saltos de línea."""
    import re as _re

    return _re.sub(r"\s+", " ", (cls.__doc__ or "")).lower()


#: Las tres proposiciones POSITIVAS del contrato (brief DECISION-BL18 §5, CXD-365/373).
#: Se exigen afirmaciones, no se prohíben frases: un candado por frase prohibida se sortea
#: reescribiendo el texto con el significado invertido — pasó, y dio falso verde.
_PROPOSICIONES_CONTRATO = (
    (
        "restriccion",
        "declarados y gobernados como métricas en el catálogo",
        "el docstring debe decir que SOLO los kinds catalogados se normalizan a MetricEvent",
    ),
    (
        "negativa",
        "no se insertan ahí",
        "el docstring debe decir que ausencias, incidentes y acciones NO se insertan en metric_event",
    ),
    (
        "no-implicacion",
        "no convierte a un evento en una observación de catálogo",
        "el docstring debe decir que portar una medición NO convierte el evento en observación",
    ),
)


def test_health_event_docstring_states_the_three_contract_propositions() -> None:
    """El docstring debe AFIRMAR las tres proposiciones, no solo evitar una frase falsa.

    Este candado es un tripwire contra la deriva del texto, no una prueba de semántica:
    la garantía dura es ``test_health_event_does_not_carry_metric_event_identity``.
    """
    from src.monitoring.system_health_contract import HealthEvent

    doc = _doc_normalizado(HealthEvent)
    assert doc, "HealthEvent debe documentar su naturaleza"

    faltan = [
        (nombre, motivo)
        for nombre, frase, motivo in _PROPOSICIONES_CONTRATO
        if frase not in doc
    ]
    assert not faltan, "proposiciones ausentes del docstring: " + "; ".join(
        f"{nombre} ({motivo})" for nombre, motivo in faltan
    )

    # La restricción sólo vale si es exclusiva: "sólo ... declarados y gobernados".
    idx = doc.index("declarados y gobernados como métricas en el catálogo")
    assert "sólo" in doc[max(0, idx - 120):idx], (
        "la mención al catálogo debe ser una RESTRICCIÓN ('sólo los kind declarados...'), "
        "no una permisión ('todos los kind, estén o no declarados...')"
    )


def test_health_event_does_not_carry_metric_event_identity() -> None:
    """Tripwire: si HealthEvent gana identidad de métrica, hay que revisar el docstring."""
    from dataclasses import fields

    from src.monitoring.system_health_contract import HealthEvent

    presentes = {f.name for f in fields(HealthEvent)} & set(
        _METRIC_EVENT_IDENTITY_NOT_NULL
    )
    assert not presentes, (
        f"HealthEvent ganó campos de identidad de métrica ({sorted(presentes)}); "
        "revisar el docstring y el brief DECISION-BL18 antes de seguir"
    )
