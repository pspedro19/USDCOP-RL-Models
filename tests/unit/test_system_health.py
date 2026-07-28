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
