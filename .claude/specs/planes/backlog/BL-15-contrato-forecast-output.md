---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/contracts/signal_contract.py
  - scripts/pipeline/generate_weekly_forecasts.py
---

# BL-15 — Contrato forecast_output (Py+TS) + validación en el zoo

**Fuente**: plan 01 §1 / FABRIC §15.2 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El zoo emite CSV plano + JSONs ad-hoc; no existe contrato tipado de predicción.

## Qué falta exactamente
`src/contracts/forecast_output.py` + espejo TS (model_id, horizon, as_of, target_time, prediction point/lower/upper, direction_probability, model_fingerprint, diagnostic_only=true). El generador valida antes de publicar; el ledger/book rechaza el tipo.

## Impacto frontend
ForecastingView puede migrar a consumir el contrato (sin cambio visual obligatorio).

## Dependencias
BL-13.

## Verificación
Test: book_construction rechaza un forecast_output por tipo.

## Notas constitución
El allocator solo acepta strategy_output — rechazo físico, no convención.
