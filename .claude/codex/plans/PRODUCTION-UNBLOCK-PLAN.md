---
kind: roadmap
status: PAUSED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/harness/harness_engine.py
  - .claude/codex/evidence/harness-latest.json
  - .claude/codex/logs/HARNESS-ENGINEERING-STATUS.md
---

# Plan de desbloqueo hacia producción

La plataforma permanece `NO-GO` hasta que todos los gates obligatorios sean `PASS`. Un `WAIVER`
no equivale a producción: debe tener riesgo residual, compensación, dueño, expiración y aprobación.

## Orden de ejecución

| Fase | Bloqueo | Trabajo | Evidencia de salida | Gate |
|---|---|---|---|---|
| 0 | Gobernanza | congelar versión, hipótesis, datasets, trial ledger y owners | release manifest firmado + rollback tag | PASS |
| 1 | Datos PIT | conectar fuentes reales para USD/COP, XAU/USD, BTC/USDT y SP500; capturar `available_at`, revisiones, timezone, calendario, gaps, duplicados, lineage y checksums | cuatro data cards + hashes + profiling + freshness | PASS |
| 2 | Features | paridad offline/online, leakage tests, warm-up, imputación, clipping, PSI/KS, drift y contract hash por activo | feature manifests + parity report + drift baseline | PASS |
| 3 | Forecasting | train/validation/test temporal, baseline naive, MAE/RMSE/MASE, directional accuracy, calibración y cobertura por horizonte/régimen | OOS report por activo y modelo | PASS |
| 4 | Estrategias | net returns, Sharpe/Sortino/Calmar, drawdown/CVaR, turnover, slippage, capacidad, benchmark, costos 1x/2x/3x, seeds/folds | strategy evidence manifest + DSR/PBO + block bootstrap | PASS |
| 5 | Comercio | sandbox Wompi con tenant desechable: checkout, aprobado, rechazado, firma inválida, monto/moneda alterados, replay, concurrencia, renovación, refund y chargeback | E2E traces + webhook ledger + conciliación | PASS |
| 6 | RBAC | matriz completa por rol/tenant/recurso; BOLA/BFLA, expiración, revocación, spoofing headers, admin approval y audit trail | adversarial matrix + audit queries | PASS |
| 7 | Frontend | carrito mobile/desktop/keyboard, focus trap, loading/error/empty states, axe WCAG 2.2 AA, noticias XSS/URL safety, visual baselines | screenshots/traces/videos + axe report | PASS |
| 8 | Performance | ramp/spike/soak/stress; p95/p99 de login, catálogo, checkout, webhook, señal, dashboard y backtest; error budgets | load report + OTel traces + SLO dashboard | PASS |
| 9 | Promoción | champion/challenger, aprobación dual, manifest de artefactos, firma/hash, canary/paper, kill switch y rollback | promotion proposal + rollback rehearsal | PASS |
| 10 | Producción | drift, stale data, slippage/fills, PnL attribution, SLO, incident runbook, retraining mensual y auto-revoke | production readiness review + on-call signoff | PASS |

## Criterios económicos mínimos

- OOS return neto ≥ 0% y Sharpe ≥ 0.5 como piso técnico; el umbral final debe superar costos y baseline.
- PBO < 0.50, DSR > 0.95, intervalos por bootstrap temporal y trial count completo.
- Max drawdown ≤ 25%, capacidad y slippage documentados, sin dependencia de un solo régimen.
- Forecast supera baseline naive de forma estadísticamente y económicamente defendible.

## Criterios de release

El comando de release debe ejecutar, en orden:

1. `python .claude/codex/harness/harness_engine.py`
2. `python scripts/validation/commerce_rbac_harness.py --json`
3. `python scripts/validation/run_quant_harness.py`
4. `python scripts/validation/run_production_harness.py`
5. `npm run rbac:check && npm run rbac:test`
6. `npm run test:run && npm run test:e2e && npm run build`

El agregado solo puede cambiar a `GO` si no existen estados `FAIL`, `BLOCKED` ni artefactos sin hash.

## Proceso de memoria continua

Cada fase debe actualizar:

- `.claude/codex/evidence/` con manifest inmutable;
- el log específico del dominio;
- [HARNESS-ENGINEERING-STATUS.md](../logs/HARNESS-ENGINEERING-STATUS.md);
- [TRACEABILITY-MATRIX.md](../governance/TRACEABILITY-MATRIX.md);
- este plan con estado, fecha, owner y evidencia.

Nunca se deben borrar fallos históricos ni reemplazar un manifest anterior.
