"""BL-25 — Motor único de monitoreo en tres relojes (control__system_health).

Un solo motor (``SystemHealthEngine``) evalúa los tres relojes de FABRIC §23:

- **DATA** (minutos): probes de freshness/paridad — rojo binario, fail-closed.
- **MODEL** (diario): PSI de features vs train + drift de la distribución de
  predicción — amarillo, congela promociones (PSI > 0.25).
- **PNL** (semanal): tracking error live-vs-paper (desviación acumulada, 3σ),
  decay de Sharpe, slippage realizado vs modelado — naranja, dispara
  withdrawal_protocol / REDUCED.

Los umbrales viven en ``system_health_contract`` (priors ex-ante de FABRIC
§23; 0 trials — cambiarlos requiere ADR). El Sharpe NO se recomputa aquí:
si se necesita, viene del SSOT ``services/common/metrics.py`` (BL-18).

Costuras declaradas hacia dependencias PLANNED:
- BL-18 (``control.metric_event``): los ``HealthEvent`` se persisten hoy en un
  JSONL append-only (``JsonlMetricEventSink``); el sink de DB lo aporta BL-18.
- BL-22 (``fact_position/fact_pnl``): el reloj PnL recibe series ya extraídas;
  cuando existan los facts, el DAG las leerá de ahí sin tocar el motor.

Contract: CTR-SYSTEM-HEALTH-001 · Version: 1.0.0 · Date: 2026-07-27
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from src.contracts.strategy_schema import safe_json_dumps
from src.monitoring.system_health_contract import (
    CONTRACT_ID,
    PREDICTION_DRIFT_SIGMA,
    PSI_BINS,
    PSI_FREEZE_THRESHOLD,
    SHARPE_ROLLING_MIN_RATIO,
    SLIPPAGE_FACTOR_MAX,
    TRACKING_ERROR_SIGMA,
    Clock,
    ClockStatus,
    DataProbe,
    HealthAction,
    HealthEvent,
    HealthSignal,
    SystemHealthSnapshot,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PromotionsFrozenError",
    "SystemHealthEngine",
    "JsonlMetricEventSink",
    "MetricEventSink",
    "assert_promotions_not_frozen",
    "check_promotions_frozen",
    "compute_psi",
    "load_snapshot",
    "write_snapshot",
]


class PromotionsFrozenError(RuntimeError):
    """El reloj de MODELO congeló las promociones (FABRIC §23, BL-25)."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# PSI — Population Stability Index (buckets por cuantiles de la referencia)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
    epsilon: float = 1e-4,
) -> float:
    """PSI estándar entre la distribución de referencia (train) y la actual.

    Buckets por cuantiles de la referencia (deciles por defecto). NaNs se
    descartan — jamás se imputan (§23.1 "métrica ausente").
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) == 0:
        raise ValueError(
            f"Insufficient data for PSI: ref={len(ref)}, cur={len(cur)} (bins={bins})"
        )

    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # Cuantiles duplicados (distribuciones degeneradas) => buckets únicos.
    edges = np.unique(edges)
    if len(edges) < 3:
        # Referencia constante: cualquier masa fuera del punto es drift total.
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / len(ref), epsilon, None)
    cur_pct = np.clip(cur_counts / len(cur), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------------------
# Sink de eventos — costura hacia control.metric_event (BL-18)
# ---------------------------------------------------------------------------

class MetricEventSink(Protocol):
    """Interfaz del destino de eventos. BL-18 aportará el sink de DB."""

    def emit(self, event: HealthEvent) -> None: ...


class JsonlMetricEventSink:
    """Sink append-only en JSONL (default hasta que exista control.metric_event)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def emit(self, event: HealthEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(safe_json_dumps(event.to_dict(), indent=None) + "\n")


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class SystemHealthEngine:
    """Motor único de los tres relojes (un solo lugar donde se decide salud)."""

    def __init__(self, event_sink: MetricEventSink | None = None):
        self.event_sink = event_sink

    # -- Reloj de DATOS (minutos, rojo binario, fail-closed) ----------------

    def evaluate_data_clock(self, probes: list[DataProbe]) -> ClockStatus:
        events: list[HealthEvent] = []
        actions: list[HealthAction] = []
        signal = HealthSignal.GREEN

        for probe in probes:
            if probe.status == "ok":
                continue
            if probe.status == "parity_broken":
                # §23.1: paridad rota => QUARANTINED, incidente crítico.
                signal = HealthSignal.RED
                self._add(actions, HealthAction.QUARANTINE)
                events.append(HealthEvent(
                    kind="parity_broken", clock=Clock.DATA, signal=HealthSignal.RED,
                    action=HealthAction.QUARANTINE,
                    message=f"Paridad rota en {probe.name} — QUARANTINED",
                    timestamp=_now_iso(),
                ))
            elif probe.active_component:
                # Componente activo rancio/ausente/error => fail-closed.
                # Jamás degradar a "usar el último valor" (§23).
                signal = HealthSignal.RED
                self._add(actions, HealthAction.FAIL_CLOSED)
                self._add(actions, HealthAction.BLOCK_SIGNAL)
                events.append(HealthEvent(
                    kind=f"data_{probe.status}", clock=Clock.DATA,
                    signal=HealthSignal.RED, action=HealthAction.FAIL_CLOSED,
                    message=(
                        f"{probe.name} ({probe.status}) es componente activo — "
                        "fail-closed: política degradada declarada, nunca el último valor"
                    ),
                    timestamp=_now_iso(),
                ))
            else:
                # Superficie diagnóstica: degradación elegante (§23.1).
                if signal == HealthSignal.GREEN:
                    signal = HealthSignal.YELLOW
                events.append(HealthEvent(
                    kind=f"data_{probe.status}", clock=Clock.DATA,
                    signal=HealthSignal.YELLOW, action=HealthAction.NONE,
                    message=f"{probe.name} ({probe.status}) es diagnóstico — degradación elegante",
                    timestamp=_now_iso(),
                ))

        status = ClockStatus(
            clock=Clock.DATA, signal=signal, actions=actions, events=events,
            metrics={
                "probes_total": len(probes),
                "probes_not_ok": sum(1 for p in probes if p.status != "ok"),
            },
            evaluated_at=_now_iso(),
        )
        self._emit(events)
        return status

    # -- Reloj de MODELO (diario, amarillo, congela promociones) ------------

    def evaluate_model_clock(
        self,
        reference_features: dict[str, np.ndarray],
        current_features: dict[str, np.ndarray],
        predictions_reference: np.ndarray | None = None,
        predictions_current: np.ndarray | None = None,
    ) -> ClockStatus:
        events: list[HealthEvent] = []
        actions: list[HealthAction] = []
        signal = HealthSignal.GREEN
        psi_by_feature: dict[str, float | None] = {}

        for name, ref in reference_features.items():
            cur = current_features.get(name)
            if cur is None:
                # §23.1: métrica ausente => N/A, jamás imputada.
                psi_by_feature[name] = None
                events.append(HealthEvent(
                    kind="metric_missing", clock=Clock.MODEL,
                    signal=HealthSignal.N_A, action=HealthAction.GATE_NO_APPROVE,
                    message=f"Feature {name} sin ventana actual — N/A",
                    timestamp=_now_iso(),
                ))
                continue
            try:
                psi = compute_psi(np.asarray(ref), np.asarray(cur))
            except ValueError as exc:
                psi_by_feature[name] = None
                events.append(HealthEvent(
                    kind="metric_missing", clock=Clock.MODEL,
                    signal=HealthSignal.N_A, action=HealthAction.GATE_NO_APPROVE,
                    message=f"PSI incomputable para {name}: {exc}",
                    timestamp=_now_iso(),
                ))
                continue
            psi_by_feature[name] = round(psi, 6)
            if psi > PSI_FREEZE_THRESHOLD:
                signal = HealthSignal.YELLOW
                self._add(actions, HealthAction.FREEZE_PROMOTIONS)
                events.append(HealthEvent(
                    kind="model_drift_psi", clock=Clock.MODEL,
                    signal=HealthSignal.YELLOW, action=HealthAction.FREEZE_PROMOTIONS,
                    message=f"PSI({name})={psi:.3f} > {PSI_FREEZE_THRESHOLD} — congelar promociones",
                    metric_value=float(psi), threshold=PSI_FREEZE_THRESHOLD,
                    timestamp=_now_iso(),
                ))

        pred_drift_z: float | None = None
        if predictions_reference is not None and predictions_current is not None:
            pref = np.asarray(predictions_reference, dtype=float)
            pcur = np.asarray(predictions_current, dtype=float)
            pref = pref[np.isfinite(pref)]
            pcur = pcur[np.isfinite(pcur)]
            if len(pref) >= 10 and len(pcur) >= 1:
                ref_std = float(np.std(pref, ddof=1))
                if ref_std > 0:
                    pred_drift_z = float(abs(np.mean(pcur) - np.mean(pref)) / ref_std)
                    if pred_drift_z > PREDICTION_DRIFT_SIGMA:
                        signal = HealthSignal.YELLOW
                        self._add(actions, HealthAction.FREEZE_PROMOTIONS)
                        events.append(HealthEvent(
                            kind="prediction_drift", clock=Clock.MODEL,
                            signal=HealthSignal.YELLOW,
                            action=HealthAction.FREEZE_PROMOTIONS,
                            message=(
                                f"Distribución de predicción fuera de ±{PREDICTION_DRIFT_SIGMA}σ "
                                f"histórico (z={pred_drift_z:.2f})"
                            ),
                            metric_value=pred_drift_z, threshold=PREDICTION_DRIFT_SIGMA,
                            timestamp=_now_iso(),
                        ))

        status = ClockStatus(
            clock=Clock.MODEL, signal=signal, actions=actions, events=events,
            metrics={
                "psi_by_feature": psi_by_feature,
                "psi_max": max((v for v in psi_by_feature.values() if v is not None), default=None),
                "prediction_drift_zscore": pred_drift_z,
                "psi_threshold": PSI_FREEZE_THRESHOLD,
            },
            evaluated_at=_now_iso(),
        )
        self._emit(events)
        return status

    # -- Reloj de PnL (semanal, naranja, withdrawal/REDUCED) ----------------

    def evaluate_pnl_clock(
        self,
        live_returns: np.ndarray | None,
        paper_returns: np.ndarray | None,
        rolling_sharpe: float | None = None,
        backtest_sharpe: float | None = None,
        realized_slippage_bps: float | None = None,
        modeled_slippage_bps: float | None = None,
    ) -> ClockStatus:
        events: list[HealthEvent] = []
        actions: list[HealthAction] = []
        signal = HealthSignal.GREEN
        te_z: float | None = None

        if live_returns is None or paper_returns is None:
            # §23.1: métrica ausente => N/A. Jamás imputar en silencio.
            events.append(HealthEvent(
                kind="metric_missing", clock=Clock.PNL,
                signal=HealthSignal.N_A, action=HealthAction.GATE_NO_APPROVE,
                message="Serie live o paper ausente — reloj PnL en N/A",
                timestamp=_now_iso(),
            ))
            status = ClockStatus(
                clock=Clock.PNL, signal=HealthSignal.N_A, actions=actions,
                events=events, metrics={"tracking_error_zscore": None},
                evaluated_at=_now_iso(),
            )
            self._emit(events)
            return status

        live = np.asarray(live_returns, dtype=float)
        paper = np.asarray(paper_returns, dtype=float)
        n = min(len(live), len(paper))
        live, paper = live[:n], paper[:n]

        # Test de desviación acumulada (FABRIC §23): |cumsum(d)_k| vs 3σ_d·√k.
        if n >= 8:
            d = live - paper
            sd = float(np.std(d, ddof=1))
            if sd > 0:
                cum = np.cumsum(d)
                k = np.arange(1, n + 1)
                te_z = float(np.max(np.abs(cum) / (sd * np.sqrt(k))))
                if te_z > TRACKING_ERROR_SIGMA:
                    signal = HealthSignal.ORANGE
                    self._add(actions, HealthAction.TRIGGER_WITHDRAWAL)
                    events.append(HealthEvent(
                        kind="withdrawal_protocol_triggered", clock=Clock.PNL,
                        signal=HealthSignal.ORANGE, action=HealthAction.TRIGGER_WITHDRAWAL,
                        message=(
                            f"Tracking error live-vs-paper {te_z:.2f}σ > "
                            f"{TRACKING_ERROR_SIGMA}σ — disparar withdrawal_protocol "
                            "(el retiro se dispara por protocolo, nunca por sentimiento)"
                        ),
                        metric_value=te_z, threshold=TRACKING_ERROR_SIGMA,
                        timestamp=_now_iso(),
                    ))
            else:
                te_z = 0.0

        # Decay: Sharpe rolling < 50% del backtest => REDUCED.
        sharpe_ratio_vs_backtest: float | None = None
        if rolling_sharpe is not None and backtest_sharpe is not None and backtest_sharpe > 0:
            sharpe_ratio_vs_backtest = float(rolling_sharpe / backtest_sharpe)
            if sharpe_ratio_vs_backtest < SHARPE_ROLLING_MIN_RATIO:
                signal = HealthSignal.ORANGE
                self._add(actions, HealthAction.REDUCED)
                events.append(HealthEvent(
                    kind="sharpe_decay", clock=Clock.PNL,
                    signal=HealthSignal.ORANGE, action=HealthAction.REDUCED,
                    message=(
                        f"Sharpe rolling {rolling_sharpe:.2f} < "
                        f"{SHARPE_ROLLING_MIN_RATIO:.0%} del backtest {backtest_sharpe:.2f}"
                    ),
                    metric_value=sharpe_ratio_vs_backtest,
                    threshold=SHARPE_ROLLING_MIN_RATIO,
                    timestamp=_now_iso(),
                ))

        # Slippage realizado > 2× modelado.
        slippage_factor: float | None = None
        if (
            realized_slippage_bps is not None
            and modeled_slippage_bps is not None
            and modeled_slippage_bps > 0
        ):
            slippage_factor = float(realized_slippage_bps / modeled_slippage_bps)
            if slippage_factor > SLIPPAGE_FACTOR_MAX:
                signal = HealthSignal.ORANGE
                self._add(actions, HealthAction.REDUCED)
                events.append(HealthEvent(
                    kind="slippage_excess", clock=Clock.PNL,
                    signal=HealthSignal.ORANGE, action=HealthAction.REDUCED,
                    message=(
                        f"Slippage realizado {realized_slippage_bps:.1f}bps > "
                        f"{SLIPPAGE_FACTOR_MAX}x modelado {modeled_slippage_bps:.1f}bps"
                    ),
                    metric_value=slippage_factor, threshold=SLIPPAGE_FACTOR_MAX,
                    timestamp=_now_iso(),
                ))

        status = ClockStatus(
            clock=Clock.PNL, signal=signal, actions=actions, events=events,
            metrics={
                "tracking_error_zscore": te_z,
                "tracking_error_threshold": TRACKING_ERROR_SIGMA,
                "n_periods": int(n),
                "sharpe_ratio_vs_backtest": sharpe_ratio_vs_backtest,
                "slippage_factor": slippage_factor,
            },
            evaluated_at=_now_iso(),
        )
        self._emit(events)
        return status

    # -- Snapshot -----------------------------------------------------------

    def build_snapshot(
        self,
        data: ClockStatus | None = None,
        model: ClockStatus | None = None,
        pnl: ClockStatus | None = None,
        previous_path: str | Path | None = None,
    ) -> SystemHealthSnapshot:
        """Consolida los relojes evaluados en este run.

        Relojes no vencidos (``None``) arrastran su último estado desde
        ``previous_path`` — cadencias distintas (min/día/semana) conviven en
        un solo snapshot sin re-evaluar lo que no toca.
        """
        clocks: dict[str, ClockStatus] = {}
        previous = load_snapshot(previous_path) if previous_path else None
        if previous is not None:
            clocks.update(previous.clocks)

        for status in (data, model, pnl):
            if status is not None:
                clocks[status.clock.value] = status

        model_status = clocks.get(Clock.MODEL.value)
        pnl_status = clocks.get(Clock.PNL.value)
        promotions_frozen = bool(
            model_status and HealthAction.FREEZE_PROMOTIONS in model_status.actions
        )
        withdrawal_triggered = bool(
            pnl_status and HealthAction.TRIGGER_WITHDRAWAL in pnl_status.actions
        )

        return SystemHealthSnapshot(
            generated_at=_now_iso(),
            clocks=clocks,
            promotions_frozen=promotions_frozen,
            withdrawal_triggered=withdrawal_triggered,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _add(actions: list[HealthAction], action: HealthAction) -> None:
        if action not in actions:
            actions.append(action)

    def _emit(self, events: list[HealthEvent]) -> None:
        if self.event_sink is None:
            return
        for event in events:
            try:
                self.event_sink.emit(event)
            except Exception as exc:  # noqa: BLE001 — el sink jamás tumba la evaluación
                logger.error("MetricEventSink failed: %s", exc)


# ---------------------------------------------------------------------------
# Persistencia del snapshot + gate de promociones
# ---------------------------------------------------------------------------

def write_snapshot(snapshot: SystemHealthSnapshot, path: str | Path) -> None:
    """Escribe el snapshot con JSON safety (nunca Infinity/NaN)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_json_dumps(snapshot.to_dict()), encoding="utf-8")


def load_snapshot(path: str | Path) -> SystemHealthSnapshot | None:
    """Lee un snapshot previo; None si no existe o es ilegible."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable health snapshot %s: %s", path, exc)
        return None
    clocks: dict[str, ClockStatus] = {}
    for key, c in (raw.get("clocks") or {}).items():
        try:
            clocks[key] = ClockStatus(
                clock=Clock(c["clock"]),
                signal=HealthSignal(c["signal"]),
                actions=[HealthAction(a) for a in c.get("actions", [])],
                events=[
                    HealthEvent(
                        kind=e["kind"], clock=Clock(e["clock"]),
                        signal=HealthSignal(e["signal"]), action=HealthAction(e["action"]),
                        message=e.get("message", ""),
                        metric_value=e.get("metric_value"), threshold=e.get("threshold"),
                        timestamp=e.get("timestamp"),
                    )
                    for e in c.get("events", [])
                ],
                metrics=c.get("metrics", {}),
                evaluated_at=c.get("evaluated_at"),
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed clock %s in %s: %s", key, path, exc)
    return SystemHealthSnapshot(
        generated_at=raw.get("generated_at", ""),
        clocks=clocks,
        promotions_frozen=bool(raw.get("promotions_frozen", False)),
        withdrawal_triggered=bool(raw.get("withdrawal_triggered", False)),
    )


def check_promotions_frozen(path: str | Path) -> tuple[bool, str]:
    """¿Está congelada la promoción según el último snapshot?

    Decisión documentada (BL-25): sin snapshot (el monitoreo nunca corrió) se
    devuelve ``(False, warning)`` — el freeze exige evidencia POSITIVA de
    drift; el fail-closed duro pertenece al reloj de DATOS. Esto evita el
    deadlock de bootstrap (sin health-DAG no habría jamás primera promoción).
    """
    snapshot = load_snapshot(path)
    if snapshot is None:
        return False, (
            f"health snapshot ausente/ilegible en {path} — monitoreo aún no corrió; "
            "promoción permitida con WARNING (BL-25)"
        )
    if snapshot.promotions_frozen:
        model = snapshot.clocks.get(Clock.MODEL.value)
        detail = "; ".join(e.message for e in model.events) if model else "sin detalle"
        return True, f"promociones CONGELADAS por reloj de modelo ({detail})"
    return False, "promociones abiertas (sin drift de modelo en el último snapshot)"


def assert_promotions_not_frozen(path: str | Path) -> None:
    """Gate para los DAGs de promoción (Vote 1 / deploy).

    Levanta ``PromotionsFrozenError`` si el reloj de MODELO congeló las
    promociones. Con snapshot ausente solo advierte (ver decisión arriba).
    """
    frozen, reason = check_promotions_frozen(path)
    if frozen:
        raise PromotionsFrozenError(reason)
    if "WARNING" in reason:
        warnings.warn(reason, stacklevel=2)
        logger.warning(reason)
