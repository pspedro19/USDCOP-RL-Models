---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - services/common/metrics.py
  - config/assets/pipelines.yaml
---

# BL-18 — Catálogo de métricas + motor único + metric_event

**Fuente**: FABRIC §19 + §28 E4 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
`services/common/metrics.py` es el SSOT constitucional de DSR/bootstrap, pero Sharpe/Calmar se computan además en publishers, scripts de análisis y (prohibido pero posible) frontend.

## Qué falta exactamente
`config/metrics/catalog.yaml` (formula_version, annualization from_asset_registry, windows, thresholds) + metrics_engine.compute() único + tabla control.metric_event (thresholds copiados al evaluar).

## Impacto frontend
Dashboard consume metric_event/API, no recalcula.

## Dependencias
BL-16.

## Verificación
Grep-CI: ningún sharpe/calmar fuera del motor; misma métrica idéntica en 5 entornos.

## Notas constitución
'annualization: from_asset_registry' resuelve mecánicamente la regla de relojes.
