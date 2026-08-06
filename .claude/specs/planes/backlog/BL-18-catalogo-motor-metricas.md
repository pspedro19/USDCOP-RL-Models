---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-08-06
supersedes: []
code_anchors:
  - config/metrics/catalog.yaml
  - config/metrics/legacy_bypass_allowlist.yaml
  - src/metrics/engine.py
  - src/metrics/persistence.py
  - database/migrations/070_fabric_control_plane.sql
  - airflow/dags/forecast_h5_l6_weekly_monitor.py
  - airflow/dags/control_system_health.py
---

# BL-18 — Catálogo de métricas + motor único + metric_event

**Fuente**: FABRIC §19 + §28 E4 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: existen `config/metrics/catalog.yaml`, `src/metrics/annualization.py`, `src/metrics/engine.py` y el DDL de `control.metric_event`. El motor y la anualización gobernada no son todavía el único camino: permanecen más de 30 cálculos independientes de Sharpe/Calmar y no hay persistencia productiva general en `metric_event`.

## Qué falta exactamente
Migrar consumidores al motor, cablear la persistencia de `metric_event` y congelar un allowlist de implementaciones heredadas que sólo pueda decrecer. Añadir una implementación nueva o ampliar el allowlist sin retirar otra debe ser rojo; el objetivo final sigue siendo cero duplicados.

## Impacto frontend
Dashboard consume metric_event/API, no recalcula.

## Dependencias
BL-16.

## Verificación
Grep-CI: ningún sharpe/calmar fuera del motor; misma métrica idéntica en 5 entornos.

## Notas constitución
'annualization: from_asset_registry' resuelve mecánicamente la regla de relojes.

## Bloqueo de cableado medido (2026-08-03)

`src/metrics/persistence.py` depende de `control.metric_event`, definido por la migración 070.
En la base viva el esquema `control` no existe; por eso el módulo tiene tests pero cero llamadores
productivos. BL-18 no puede cerrar hasta aplicar 070 y demostrar al menos un productor y un
consumidor reales sobre el evento persistido.

## Estado PostgreSQL posterior a Fabric (2026-08-04)

La migración que crea `control.metric_event` ya fue aplicada. El sink
`src/metrics/persistence.py` fue ejecutado contra PostgreSQL real: insert, replay idempotente y
rechazo de colisión por UUID funcionan; las sondas de verificación se hicieron dentro de
transacciones revertidas. `f7f853e6` normaliza el string ISO contractual a `datetime` UTC-aware en
la frontera asyncpg y evita falsas colisiones cuando dos offsets representan el mismo instante.

El cableado productivo mínimo existe desde `55cda935`:

- `forecast_h5_l6_weekly_monitor.py::persist_governed_metric_events` calcula mediante
  `MetricEngine.from_asset_registry`, persiste el evento y está enlazado en el DAG semanal;
- `control_system_health.py` consume la métrica con `SELECT ... FROM control.metric_event` para el
  estado de salud del track paper.

El BL permanece **PARTIAL** porque esa costura no generaliza todavía el motor a todos los
consumidores, la cobertura sigue siendo parcial y el allowlist de cálculos heredados permanece por
encima de cero. Además, la tabla tiene
una identidad semántica única adicional: si un reintento conserva el payload pero regenera
`metric_event_id`, el `ON CONFLICT` actual no captura esa restricción y puede filtrar una excepción
del driver. Debe decidirse y probarse explícitamente si esa colisión se traduce a
`MetricContractError` o permanece fail-loud; no se declara resuelta por el cableado existente.
