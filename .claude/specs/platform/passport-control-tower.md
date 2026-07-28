---
kind: as-built
status: PARTIAL
contract: CTR-PASSPORT-001
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/contracts/passport.py
  - usdcop-trading-dashboard/lib/contracts/passport.contract.ts
  - usdcop-trading-dashboard/lib/passport/compose.ts
  - usdcop-trading-dashboard/components/gm/views/PassportView.tsx
  - scripts/pipeline/export_control_tower.py
---

# Passport + Control Tower (CTR-PASSPORT-001)

> BL-32 · FABRIC §24.4-§24.5. **Superficie DIAGNÓSTICA**: muestra, no aprueba ni
> ejecuta. El Voto 2/2 y el deploy viven solo en `/dashboard`
> (`approval-gates.md` invariante 3).

## 1. Qué se implementó

| Pieza | Ruta |
|---|---|
| Contrato Py (SSOT) | `src/contracts/passport.py` |
| Contrato TS (espejo) | `usdcop-trading-dashboard/lib/contracts/passport.contract.ts` |
| Compositor BFF | `usdcop-trading-dashboard/lib/passport/compose.ts` |
| API torre | `app/api/passport/tower/route.ts` |
| API passport | `app/api/passport/[strategyId]/route.ts` |
| Vista | `components/gm/views/PassportView.tsx` + `app/passport/page.tsx` |
| Proyección de gobernanza | `scripts/pipeline/export_control_tower.py` → `public/data/control-tower/governance.json` |
| RBAC | `/passport` y `/api/passport` ⇒ `research:read` en `rbac.contract.ts` |

## 2. La primitiva: `Sourced<T>`

```ts
Sourced<T> = { value: T | null; source: { path, status, pending, note } }
```

`status` solo tiene dos valores: `published` | `unavailable`. **No existe
"estimated"** — estimar sería una decisión de modelado, no de ingeniería.

- `published` ⇒ `path` obligatorio (el artefacto del que salió la cifra).
- `unavailable` ⇒ `value = null` **y** `pending` obligatorio (quién la debe).

Esto es lo que separa *"no tenemos el dato"* de *"el valor es cero"*, que es
exactamente lo que una torre de control tiene que acertar. El validador de ambos
lenguajes rechaza cualquier violación.

## 3. División live / MV (§24.4)

FABRIC divide el Passport porque **una MV de Postgres se reemplaza en el refresh y
no puede sostener estado operativo en vivo**. El contrato conserva la división:

- `PassportLiveState` ← `v_strategy_passport_live` (no materializada).
- `EnvPerformance` × 5 ← `mv_strategy_performance_daily` (refresco nocturno).
- `StrategyPassport` = composición de ambas.

Hoy la composición la hace el BFF sobre artefactos publicados, con la MISMA forma
que tendrán las vistas. **Cuando existan las tablas de hechos, solo cambia el
compositor: ni el contrato ni la vista.**

## 4. DDL de referencia (NO aplicada — espera a BL-18/21/22/24)

> No vive en `database/migrations/**` a propósito: sus tablas de hechos no existen
> y una migración que no puede correr es deuda, no progreso.

```sql
-- v_strategy_passport_live — ligera, NO materializada (estado operativo).
CREATE OR REPLACE VIEW v_strategy_passport_live AS
SELECT s.strategy_id,
       COUNT(o.*) FILTER (WHERE o.state = 'OPEN')      AS open_orders,   -- BL-21 exec.order
       MAX(f.filled_at)                                 AS last_fill_at,  -- BL-21 exec.fill
       bool_or(q.active)                                AS quarantined,   -- BL-21
       bool_and(r.reconciled)                           AS reconciled,    -- BL-22
       bool_or(k.engaged)                               AS kill_switch_engaged -- BL-30
FROM   strategy s
LEFT JOIN exec_order o ON o.strategy_id = s.strategy_id
LEFT JOIN exec_fill  f ON f.strategy_id = s.strategy_id
LEFT JOIN exec_quarantine   q ON q.strategy_id = s.strategy_id
LEFT JOIN exec_reconciliation r ON r.strategy_id = s.strategy_id
LEFT JOIN control_kill_switch k ON k.strategy_id = s.strategy_id
GROUP BY s.strategy_id;

-- mv_strategy_performance_daily — materializada, refresco nocturno.
-- Una fila por (estrategia, entorno): la MISMA métrica del MISMO motor (BL-18).
CREATE MATERIALIZED VIEW mv_strategy_performance_daily AS
SELECT m.strategy_id, m.env, m.as_of,
       m.metric_value FILTER (WHERE m.metric_id = 'return_pct')  AS return_pct,
       m.metric_value FILTER (WHERE m.metric_id = 'sharpe')      AS sharpe,
       m.metric_value FILTER (WHERE m.metric_id = 'calmar')      AS calmar,
       m.metric_value FILTER (WHERE m.metric_id = 'max_dd_pct')  AS max_dd_pct,
       m.metric_value FILTER (WHERE m.metric_id = 'dsr_family')  AS dsr_family,
       m.n_trades
FROM   metric_event m                         -- BL-18
WHERE  m.metric_engine_version IS NOT NULL;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY mv_strategy_performance_daily;  -- nocturno

-- v_strategy_passport — la composición: el SELECT que reemplaza el paseo
-- Airflow → MLflow → JSONs → SQL.
CREATE OR REPLACE VIEW v_strategy_passport AS
SELECT i.*, g.*, l.*, p.*, live.*
FROM   strategy_identity i
JOIN   strategy_governance g USING (strategy_id)   -- trials FT/AT, N×3, votos, retiro
JOIN   lineage_node l        USING (strategy_id)   -- BL-24
LEFT JOIN mv_strategy_performance_daily p USING (strategy_id)
LEFT JOIN v_strategy_passport_live live   USING (strategy_id);
```

**Regla dura al aplicarla**: `mv_strategy_performance_daily` NUNCA puede alimentar
`PassportLiveState`. Esa confusión es el error que §24.4 documenta como cerrado.

## 5. Interfaces pendientes (el acople, CODEX)

La lista **legible por máquina** vive en `PENDING_INTERFACES`
(`lib/passport/compose.ts`) y se renderiza en la propia página. Resumen:

| BL | Campo | Productor esperado |
|---|---|---|
| BL-18 | `metric_engine`, `performance.*`, p/e-value del test pareado, DSR por entorno | catálogo de métricas + motor único + `metric_event` |
| BL-21 | `live.*`, `performance.canary` | event sourcing `exec.*` |
| BL-22 | `book.pnl_*`, `book.capital`, `timing_ratio`, `turnover`, atribución timing/beta/carry | `fact_position` / `fact_pnl` |
| BL-23 | `performance.held_out` | backfill anti-supervivencia |
| BL-24 | `lineage.lineage_graph`, `data.last_vintage_revision` | nodes/edges + camino dorado |
| BL-25 | `data.clocks.exec`, semáforo de retiro **por estrategia** | `control__system_health` |
| BL-26/27 | vol objetivo/prevista, gross/net, CVaR, ρ, `m_forward`/`m_dd` | `portfolio_snapshot` + allocator |
| BL-28 | `data.replay_parity` | diff semántico de bundles |

## 6. Decisiones de honestidad tomadas (y por qué)

1. **El semáforo de retiro por sleeve es `unknown`, no verde.**
   `system_health.json` publica `withdrawal_triggered`/`promotions_frozen`
   **globales**. Imputar una bandera del sistema a una sleeve concreta sería
   atribuirle un hecho que no es suyo. Las banderas globales se muestran en DATOS,
   donde corresponden; el semáforo por estrategia espera a BL-25.
2. **`CANARY` / `REDUCED` / `QUARANTINED` cuentan `null`, no `0`.**
   Ningún productor publica esos estados; `0` se leería como "lo comprobamos y no
   hay ninguna", que es falso.
3. **El test pareado v11 vs v12/v14 muestra los retornos publicados y `p`/`e` vacíos.**
   Calcular el pareado en el BFF sería (a) calcular en el frontend, prohibido por
   §24.1, y (b) **un trial nuevo**. Se publica cuando exista BL-18.
4. **`timing_ratio` sale `unavailable` en todas las filas.**
   BL-07 lo calculó *one-off y por diseño no lo persistió*; además v11 quedó
   `UNAVAILABLE` por falta del retorno subyacente en el adapter. No hay número que
   leer, y fabricar un proxy está explícitamente prohibido.
5. **El entorno `live` se etiqueta "forward publicado, NO reconciliado contra fills".**
   Es un forward del pipeline, no una posición reconciliada.
6. **Las cinco columnas NO vienen del mismo motor todavía** — se declara en la
   propia vista (`metric_engine` unavailable) en vez de insinuar comparabilidad.

## 7. Verificación

- `npm run rbac:check` → 97 rutas API / 32 páginas, todas cubiertas.
- `npm run rbac:test` → ALL RBAC CONTRACT TESTS PASS.
- `npx tsc --noEmit` → delta 0 contra el baseline (635 líneas).
- `tests/unit/test_passport_contract.py` + `tests/unit/contracts/passport-contract.test.ts`
  (espejo Py↔TS de vocabularios, guard §6, y prohibición de acciones).
- `python scripts/pipeline/export_control_tower.py --check` → artefacto al día.

## 8. Notas constitución

- **0 trials.** Nada aquí abre una celda de hipótesis: se copian cifras publicadas,
  se cuentan filas y se restan fechas.
- N_MAX=989 se muestra junto a N_global como **cota de gasto** (§9.7) y jamás entra
  en ninguna fórmula.
- Con N<20 la fila publica solo conteo y PnL (§6), impuesto por contrato en ambos
  lenguajes y no por criterio de la vista.
