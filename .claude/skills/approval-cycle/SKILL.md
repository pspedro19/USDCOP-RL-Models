---
name: approval-cycle
description: Drive the 2-vote approval and production deploy for a strategy — run the backtest export, read the gates, prepare the human Vote 2, and monitor the deploy DAG. Use when promoting a strategy to production, when approval_state.json needs resetting, or when a deploy stalled.
---

# Approval cycle (2 votos)

## The boundary this skill must never cross

**Vote 2 es humano.** Esta skill prepara, explica y monitorea — **no aprueba**. No escribe
`approval_state.json`, no llama a `/api/production/approve`, no fuerza el deploy. Si el operador
pide "apruébalo tú", la respuesta es que el gate existe precisamente para que una persona mire
los números. Ver `.claude/rules/approval-gates.md`.

## Paso 1 — Vote 1 (automático)

```bash
python scripts/pipeline/train_and_export_smart_simple.py --phase backtest
```

Entrena 2020-2024, evalúa 2025 OOS, computa los 5 gates y escribe `summary_2025.json`,
`approval_state.json` (`PENDING_APPROVAL`) y `trades/{sid}_2025.json`.

## Paso 2 — Presentar los gates al operador

Lee `usdcop-trading-dashboard/public/data/production/approval_state.json` y reporta:

- Estado de los 5 gates (retorno > -15%, Sharpe > 0, maxDD < 20%, trades ≥ 10, p < 0.05)
- `backtest_recommendation` (PROMOTE / REVIEW / REJECT) y `backtest_confidence`
- Las métricas del **bundle**, nunca métricas recomputadas

**Advertencia obligatoria si aplica**: si el p-value viene de iterar sobre el mismo OOS, decláralo.
El DSR trial-aware de v11 es 0.50-0.92 < 0.95 — el backtest 2025 **no prueba edge tras selección**
(`.claude/rules/quant-constitution.md`). Con N < 20 trades, no reportes Sharpe ni p-value.

## Paso 3 — Vote 2 (humano, en `/dashboard`)

El operador revisa y hace click en Aprobar/Rechazar. **Solo `admin`.** Queda en `audit_log`.

## Paso 4 — Monitorear el deploy

Aprobar dispara `POST /api/production/deploy`, que prefiere la ruta Airflow:
`forecast_h5_l4b_production_deploy` → `guard_approved` (re-valida server-side) → `run_production`
→ `validate_output` → `register_bundle`.

```bash
airflow dags list-runs -d forecast_h5_l4b_production_deploy --limit 3
```

Deploy manual equivalente:
```bash
python scripts/pipeline/train_and_export_smart_simple.py --phase production --no-png --seed-db
```

## Paso 5 — Verificar

- `summary.json` fresco con métricas del año en curso
- `trades/{sid}.json` poblado
- Bundle registrado en `registry.json`
- Badge APPROVED visible en `/production`

## Rechazo / reintento

```bash
python scripts/pipeline/train_and_export_smart_simple.py --reset-approval
```

Vuelve a `PENDING_APPROVAL` con gates frescos. **Requiere Vote 2 humano otra vez.**

## Constraints

- Do NOT aprobar, ni editar `approval_state.json` a mano.
- Do NOT presentar métricas recomputadas por el frontend como base de decisión.
- El contenedor node **no tiene python3**: en contenedores la ruta Airflow es la única real.
- El seeding de DB requiere la migración 054.
