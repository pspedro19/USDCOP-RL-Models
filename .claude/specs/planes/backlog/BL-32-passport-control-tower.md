---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/PassportView.tsx
  - usdcop-trading-dashboard/lib/passport/compose.ts
  - usdcop-trading-dashboard/lib/contracts/passport.contract.ts
  - src/contracts/passport.py
  - scripts/pipeline/export_control_tower.py
  - .claude/specs/platform/passport-control-tower.md
---

# BL-32 — Passport (vista live + MV) + Control Tower

**Fuente**: FABRIC §24.4-§24.5 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Hub/Production muestran piezas; no existe el SELECT único (identidad+gobierno+linaje+desempeño 5 entornos+riesgo) ni la portada de 10 segundos.

## Qué falta exactamente
v_strategy_passport_live (NO materializada) + mv_strategy_performance_daily (nocturna) + composición; Control Tower: LIBRO/SLEEVES/DATOS (§24.5) incl. N_global vs N_MAX, test pareado v11 vs v12/v14 con p/e-value, semáforo de retiro.

## Impacto frontend
Página/sección nueva (o evolución de /hub) — la cara del sistema.

## Dependencias
BL-18, BL-22, BL-24; BL-05 es el primer ladrillo.

## Verificación
El paseo Airflow→MLflow→JSONs→SQL se reemplaza por un SELECT (demo).

## Notas constitución
MV de Postgres no puede sostener estado live (se reemplaza en refresh) — por eso la división.

## Estado de implementación (2026-07-28, CLAUDE lane2)

**PARTIAL — la superficie completa existe; las fuentes de hechos no.**

Entregado: contrato `CTR-PASSPORT-001` espejado Py↔TS, compositor BFF sobre
artefactos publicados, rutas `/api/passport/tower` y `/api/passport/[strategyId]`,
página `/passport` (`research:read`, RBAC verde), proyección de gobernanza
`scripts/pipeline/export_control_tower.py` (trials FT/AT + N×3 + familias +
protocolos de retiro), y spec `.claude/specs/platform/passport-control-tower.md`
con la **DDL de referencia** de `v_strategy_passport_live` /
`mv_strategy_performance_daily` / `v_strategy_passport` (no aplicada: sus tablas de
hechos no existen).

Pendiente de CODEX (declarado como interfaz legible por máquina en
`PENDING_INTERFACES`, renderizada en la propia página): BL-18 (motor de métricas,
p/e-value del test pareado, DSR por entorno), BL-21 (estado live, canary),
BL-22 (PnL del libro, timing_ratio, turnover, atribución), BL-23 (held-out),
BL-24 (grafo de linaje, vintage), BL-25 (reloj exec + semáforo por estrategia),
BL-26/27 (riesgo, ρ, multiplicadores), BL-28 (paridad replay).

Verificación: `rbac:check` 97 API/32 páginas OK · `rbac:test` PASS ·
`tsc --noEmit` delta 0 · exportador corrido (ledger 239 líneas, N_global=239/989,
4 activos con `n_trials_total` == total del ledger).
