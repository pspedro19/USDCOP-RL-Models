---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
  - src/contracts/strategy_schema.py
---

# BL-46 — Políticas: backend (policy_version/signal) + frontend schema-driven (R4-R5)

**Fuente**: planes/05-rule-based-strategies.md §8-§9, §13 R4-R5 · **Ola**: 3-4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El frontend ya es motor-agnóstico para bundles (verificado: renderiza SPX-MA200 igual que COP-ML) pero NO hay panel explicativo por motor, ni rule_trace, ni policy_version en DB; los params por usuario (sb_trading_configs) se renderizan con campos fijos.

## Qué falta exactamente
R4: control.policy_version (engine_type, hashes, manifest_uri) + action.strategy_signal normalizada con decision_components JSONB + rule_trace_uri (converge con BL-42); endpoints GET /strategies*, POST /strategies/validate; librería de evaluación empaquetada (la MISMA de Airflow). R5: renderer único StrategyEngineExplanation con variantes RuleTracePanel/MLExplanationPanel/RLPolicyPanel/CompositeDecisionPanel; el frontend renderiza el trace, JAMÁS re-evalúa; sección presentation: con presentation_hash propio (cambiar etiqueta ≠ versión económica); configs por usuario renderizadas desde schema.

## Impacto frontend
Panel de explicación por motor en replay/production; tabla Condición|Observado|Umbral|Resultado. Conecta con BL-20 (la vista admin de interpretabilidad consume el mismo trace para rule-based).

## Dependencias
BL-45, BL-42 (misma tabla de señal — implementar juntas), BL-22.

## Verificación
RuleTracePanel muestra el trace real de MA200 sin recalcular (test: mock trace ⇒ render exacto); rbac de endpoints nuevos en la matriz.

## Notas constitución
El frontend presenta hechos; no decide ni recalcula (ley 12 FABRIC).
