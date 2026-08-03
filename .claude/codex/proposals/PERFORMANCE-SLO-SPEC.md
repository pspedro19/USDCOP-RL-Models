---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/proposals/QUALITY-HARNESS-SPEC.md
---

# SLO y performance engineering

| Journey | Métricas | Gate inicial |
|---|---|---|
| Login/registro | p50/p95/p99, error rate, locks | p95 API < 500 ms; error < 0.5% |
| Catálogo/carrito | LCP/INP/CLS, API p95, queries | LCP < 2.5 s; INP < 200 ms; CLS < 0.1 |
| Checkout/webhook | p95/p99, duplicados, grant lag | exactly-once; grant p95 < 10 s |
| Señal/inferencia | freshness, feature/inference latency | deadline; stale=0 para ejecución |
| Streaming | event lag, reconnect, drops, memoria | recuperación sin pérdida silenciosa |
| Backtest/training | duración, memoria, costo, reproducibilidad | budget; hashes iguales por seed |

Son budgets iniciales a calibrar con tráfico real. Cada SLO define ventana, población, exclusiones,
telemetría y error budget; el promedio nunca sustituye p95/p99.

## Harness obligatorio

1. Smoke por commit; ramp, spike, soak y stress en staging con datos sintéticos.
2. OpenTelemetry browser → API → DB → proveedor, sin PII.
3. Assertions funcionales junto a latencia: monto, entitlement, señal y auditoría coinciden.
4. Baseline versionada; regresión de percentiles bloquea release.
5. Chaos: timeout, 429/5xx, DB/Redis lentos, evento duplicado/desordenado y restart.
6. ML: freshness, drift, calibración, slippage, fills, PnL attribution, exposición, turnover,
   drawdown, constraints y delta contra champion.
