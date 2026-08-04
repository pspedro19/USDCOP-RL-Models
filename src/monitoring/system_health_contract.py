"""CTR-SYSTEM-HEALTH-001 — Contrato del monitoreo en tres relojes (BL-25).

Fuente normativa: FABRIC §23 (.claude/specs/planes/04-CTR-QLAB-FABRIC-004.md).
Un solo motor de evaluación, tres latencias, tres clases de acción automática:

| Reloj  | Frecuencia | Señal          | Acción automática                          |
|--------|-----------|-----------------|--------------------------------------------|
| Datos  | minutos   | rojo binario    | fail-closed / QUARANTINE                   |
| Modelo | diaria    | amarillo        | alerta + congelar promociones              |
| PnL    | semanal   | naranja         | disparar withdrawal_protocol / REDUCED     |

Los umbrales de este módulo son PRIORS ECONÓMICOS ex-ante copiados de FABRIC
§23 (quant-constitution §1: cero magia numérica). NO se tunean contra ningún
período — cambiarlos requiere ADR, no un commit.

Este módulo vive en src/monitoring/ (no en src/contracts/) porque los
contratos compartidos requieren C-NNN con Codex; si la raíz decide moverlo a
src/contracts/system_health_schema.py es un re-export puro.

Version: 1.0.0 · Date: 2026-07-27 · Trials: 0 (infra de monitoreo, no modelado)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

CONTRACT_ID = "CTR-SYSTEM-HEALTH-001"
CONTRACT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Umbrales ex-ante (FABRIC §23 — no tunear; ver quant-constitution §1)
# ---------------------------------------------------------------------------

#: PSI > 0.25 = drift de features => congelar promociones (reloj MODELO).
PSI_FREEZE_THRESHOLD = 0.25
#: Predicción media fuera de ±2σ del histórico => drift de predicción.
PREDICTION_DRIFT_SIGMA = 2.0
#: Brier rolling degradado > 20% en 6m (solo si hay calibración).
BRIER_DEGRADATION_MAX = 0.20
#: Test de desviación acumulada live-vs-paper: 3σ => withdrawal_protocol.
TRACKING_ERROR_SIGMA = 3.0
#: Sharpe rolling 12m < 50% del backtest => decay (REDUCED).
SHARPE_ROLLING_MIN_RATIO = 0.5
#: Slippage realizado > 2× modelado => alerta PnL.
SLIPPAGE_FACTOR_MAX = 2.0
#: Buckets del PSI (convención estándar de la industria, no un tuning).
PSI_BINS = 10


class Clock(str, Enum):
    """Los tres relojes del §23."""

    DATA = "data"
    MODEL = "model"
    PNL = "pnl"


class HealthSignal(str, Enum):
    """Semáforo por reloj. N_A = métrica ausente (jamás imputada)."""

    GREEN = "green"
    YELLOW = "yellow"    # reloj MODELO en drift
    ORANGE = "orange"    # reloj PnL degradado
    RED = "red"          # reloj DATOS (rojo binario)
    N_A = "n_a"          # §23.1: métrica ausente => N/A, el gate no aprueba


class HealthAction(str, Enum):
    """Acciones automáticas del §23/§23.1."""

    NONE = "none"
    FAIL_CLOSED = "fail_closed"                # target cero / política degradada declarada
    QUARANTINE = "quarantine"                  # paridad rota / broker discrepante
    BLOCK_SIGNAL = "block_signal"              # datos rancios: bloquea nueva señal
    FREEZE_PROMOTIONS = "freeze_promotions"    # drift de modelo
    TRIGGER_WITHDRAWAL = "trigger_withdrawal"  # TE 3σ => withdrawal_protocol
    REDUCED = "reduced"                        # decay de Sharpe
    GATE_NO_APPROVE = "gate_no_approve"        # métrica ausente
    KILL_SWITCH = "kill_switch"                # executor caído


# ---------------------------------------------------------------------------
# Tabla §23.1 — fallas diferenciales (DIAGNOSTIC vs ACTION) como contrato
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureMode:
    """Una fila de la tabla §23.1 de FABRIC."""

    failure_id: str
    failure: str
    diagnostic: str
    action: str
    response: str
    health_action: HealthAction


FAILURE_TABLE: tuple[FailureMode, ...] = (
    FailureMode(
        "stale_data", "Datos rancios", "Panel STALE", "Bloquea nueva señal",
        "incident + health snapshot", HealthAction.BLOCK_SIGNAL,
    ),
    FailureMode(
        "png_missing", "PNG ausente", "Oculta imagen", "Sin efecto",
        "degradación elegante", HealthAction.NONE,
    ),
    FailureMode(
        "diagnostic_model_failure", "Modelo diagnóstico falla",
        "Resto del zoo continúa", "Sin efecto", "error visible", HealthAction.NONE,
    ),
    FailureMode(
        "active_component_failure", "Componente activo falla", "No aplica",
        "Fail-closed", "target cero o política declarada; jamás reusar el último forecast",
        HealthAction.FAIL_CLOSED,
    ),
    FailureMode(
        "prediction_without_lift", "Predicción sin lift",
        "Veredicto negativo publicado", "No invalida automáticamente PnL positivo",
        "separar ciencia predictiva y decisión", HealthAction.NONE,
    ),
    FailureMode(
        "invalid_signal", "Señal inválida", "No aplica", "No se publica",
        "contract violation", HealthAction.BLOCK_SIGNAL,
    ),
    FailureMode(
        "parity_broken", "Paridad rota", "Advertencia", "QUARANTINED",
        "incidente crítico", HealthAction.QUARANTINE,
    ),
    FailureMode(
        "executor_down", "Executor caído", "Sin efecto", "No nuevas órdenes",
        "kill switch / recuperación", HealthAction.KILL_SWITCH,
    ),
    FailureMode(
        "broker_mismatch", "Broker discrepante", "Sin efecto", "QUARANTINED",
        "reconciliación", HealthAction.QUARANTINE,
    ),
    FailureMode(
        "metric_missing", "Métrica ausente", "N/A", "El gate no aprueba",
        "jamás imputar en silencio", HealthAction.GATE_NO_APPROVE,
    ),
    FailureMode(
        "bundle_incomplete", "Bundle incompleto", "UI parcial", "No hay promoción",
        "verify fail", HealthAction.FREEZE_PROMOTIONS,
    ),
    FailureMode(
        "public_forecast_down", "Forecast público caído", "Página degradada",
        "Posiciones sin cambio", "barrera contractual", HealthAction.NONE,
    ),
)


# ---------------------------------------------------------------------------
# Entradas / salidas del motor
# ---------------------------------------------------------------------------

ProbeStatus = Literal["ok", "stale", "missing", "parity_broken", "error"]


@dataclass
class DataProbe:
    """Resultado de un probe del reloj de DATOS (freshness/paridad por serie).

    ``active_component=True`` marca series de las que depende una estrategia
    ACTIVA (fail-closed); ``False`` = superficie diagnóstica (degradación
    elegante, §23.1).
    """

    name: str
    status: ProbeStatus
    active_component: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthEvent:
    """Evento emitido por un reloj — **envelope operativo mixto** (BL-18).

    Los ``kind`` que viajan aquí no son todos de la misma naturaleza: hay
    observaciones métricas (``model_drift_psi``, ``prediction_drift``,
    ``sharpe_decay``, ``slippage_excess``), la *ausencia* de una métrica
    (``metric_missing``), incidentes y diagnósticos (``parity_broken``,
    ``data_*``) y actos de protocolo (``withdrawal_protocol_triggered``).

    Sólo los ``kind`` **declarados y gobernados como métricas en el catálogo**
    pueden normalizarse después a un ``MetricEvent`` y persistirse en
    ``control.metric_event``. Los demás —ausencias, incidentes y acciones— no
    se insertan ahí, aunque adjunten la medición que los causó: llevar un
    número no convierte a un evento en una observación de catálogo.

    Este evento **no puede insertarse directamente** en ``control.metric_event``:
    la tabla exige ``catalog_version``, ``formula_version``, ``entity_type``,
    ``entity_id`` y ``metric_namespace`` como ``NOT NULL``, y ninguno existe ni
    se deriva de aquí. Rellenarlos con constantes sería fabricar linaje.

    El destino por defecto sigue siendo el JSONL append-only, que actúa como
    buffer durable; la frontera productor/consumidor es una decisión abierta en
    ``.claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md``.
    """

    kind: str
    clock: Clock
    signal: HealthSignal
    action: HealthAction
    message: str
    metric_value: float | None = None
    threshold: float | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "clock": self.clock.value,
            "signal": self.signal.value,
            "action": self.action.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class ClockStatus:
    """Estado consolidado de un reloj tras una evaluación."""

    clock: Clock
    signal: HealthSignal
    actions: list[HealthAction] = field(default_factory=list)
    events: list[HealthEvent] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock": self.clock.value,
            "signal": self.signal.value,
            "actions": [a.value for a in self.actions],
            "events": [e.to_dict() for e in self.events],
            "metrics": self.metrics,
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class SystemHealthSnapshot:
    """Snapshot publicado (semáforos de /production + gate de promociones)."""

    generated_at: str
    clocks: dict[str, ClockStatus]
    promotions_frozen: bool
    withdrawal_triggered: bool
    contract: str = CONTRACT_ID
    version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "version": self.version,
            "generated_at": self.generated_at,
            "clocks": {k: v.to_dict() for k, v in self.clocks.items()},
            "promotions_frozen": self.promotions_frozen,
            "withdrawal_triggered": self.withdrawal_triggered,
        }
