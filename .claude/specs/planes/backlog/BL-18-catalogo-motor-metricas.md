---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - services/common/metrics.py
  - config/assets/pipelines.yaml
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
