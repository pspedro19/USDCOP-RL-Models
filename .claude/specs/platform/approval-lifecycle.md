---
kind: as-built
status: IMPLEMENTED
contract: CTR-APPROVAL-001
version: 2.0.0
last_verified: 2026-07-20
supersedes:
  - rules/approval-gates.md
code_anchors:
  - src/contracts/strategy_schema.py
  - scripts/pipeline/train_and_export_smart_simple.py
  - airflow/dags/forecast_h5_l4b_production_deploy.py
  - usdcop-trading-dashboard/lib/contracts/production-approval.contract.ts
---
# SDD Spec: Approval Lifecycle (reference)

> **Responsibility**: detalle completo del sistema de 2 votos — secuencia end-to-end, schema de
> `approval_state.json`, gates, flujo de deploy y componentes de dashboard.
>
> Las **invariantes** (quién vota qué, qué está prohibido) viven en `../../rules/approval-gates.md`,
> que se auto-carga. Este documento es la referencia on-demand; no dupliques invariantes aquí.

---

## 1. Sistema de dos votos

```
              BACKTEST COMPLETE
                     |
  Vote 1/2 (auto)    v
  Python export ──► approval_state.json (PENDING_APPROVAL, 5 gates)
                     |
  Vote 2/2 (humano)  v
  Operador en /dashboard ──► APPROVED ──► deploy ──► LIVE
                          └► REJECTED ──► --reset-approval
```

| Vote | Actor | Dónde | Disparo |
|------|-------|-------|---------|
| 1/2 | Script Python | CLI (`--phase backtest`) | Automático durante el export |
| 2/2 | Operador humano | `/dashboard` | Click Aprobar/Rechazar |

**Vote 2 ocurre en `/dashboard`, NO en `/production`.** `/production` muestra un
`ApprovalStatusCard` de solo lectura.

### Estados

| Status | Significado | Lo pone | Siguiente acción |
|--------|-------------|---------|------------------|
| `PENDING_APPROVAL` | Backtest listo, espera revisión humana | Script de export | Revisar en `/dashboard` |
| `APPROVED` | Humano aprobó | API del dashboard | `--phase production` |
| `REJECTED` | Humano rechazó | API del dashboard | Corregir + `--reset-approval` |
| `LIVE` | Desplegado a trading real | Promoción manual | Monitoreo Stage 7 |

---

## 2. Gates por defecto (5)

| Gate | Label | Comparador | Umbral |
|------|-------|------------|--------|
| `min_return_pct` | Retorno Minimo | `>` | -15% |
| `min_sharpe_ratio` | Sharpe Minimo | `>` | 0.0 |
| `max_drawdown_pct` | Max Drawdown | `<` | 20% |
| `min_trades` | Trades Minimos | `>=` | 10 |
| `statistical_significance` | Significancia (p<0.05) | `<` | 0.05 |

**Recomendación**: `PROMOTE` (5/5) · `REVIEW` (parcial, ninguno crítico) · `REJECT` (falla
retorno o drawdown). `backtest_confidence` = fracción de gates aprobados.

### Schema de un gate

```json
{"gate": "min_return_pct", "label": "Retorno Minimo",
 "passed": true, "value": 20.03, "threshold": -15.0}
```

---

## 3. Secuencia end-to-end

1. `python scripts/pipeline/train_and_export_smart_simple.py --phase backtest`
   → entrena 2020-2024, evalúa 2025 OOS, computa métricas, evalúa 5 gates (**Vote 1**),
   escribe `summary_2025.json`, `approval_state.json`, `trades/{sid}_2025.json`.
2. Operador entra a `/dashboard`: KPIs, p-value, candlestick, tabla de trades, panel de gates.
3. Click "Aprobar" → `POST /api/production/approve` → `status: APPROVED`, `approved_by`,
   `approved_at` (**Vote 2**).
4. El endpoint dispara `POST /api/production/deploy` (fire-and-forget).
5. Deploy prefiere la **ruta Airflow**: dispara `forecast_h5_l4b_production_deploy`, cuyo
   `guard_approved` **re-valida `status == APPROVED` server-side** (hard gate), corre el
   manifest, valida frescura de `summary.json` y termina en `register_bundle`.
   Sin entorno Airflow cae al `spawn('python3')` local — el contenedor node **no tiene python3**,
   así que en contenedores la ruta Airflow es la única real.
6. Deploy manual equivalente:
   `python scripts/pipeline/train_and_export_smart_simple.py --phase production --no-png --seed-db`
   (requiere migración 054 para el upsert de subtrades).

---

## 4. `approval_state.json`

```json
{
  "status": "PENDING_APPROVAL",
  "strategy": "smart_simple_v11",
  "backtest_year": 2025,
  "backtest_recommendation": "PROMOTE",
  "backtest_confidence": 1.0,
  "gates": [ /* ver §2 */ ],
  "backtest_metrics": {"return_pct": 20.03, "sharpe": 3.516, "max_dd_pct": 3.83,
                        "p_value": 0.0097, "trades": 24, "win_rate_pct": 70.8},
  "deploy_manifest": {
    "pipeline_type": "ml_forecasting",
    "script": "scripts/pipeline/train_and_export_smart_simple.py",
    "args": ["--phase", "production", "--no-png", "--seed-db"],
    "config_path": "config/execution/smart_simple_v1.yaml",
    "db_tables": ["forecast_h5_predictions", "forecast_h5_signals",
                  "forecast_h5_executions", "forecast_h5_subtrades",
                  "forecast_h5_paper_trading"]
  },
  "approved_by": null, "approved_at": null, "reviewer_notes": null
}
```

El deploy es **manifest-driven**: cada script de backtest embebe su `deploy_manifest`, así
cualquier estrategia (H1, H5, RL) auto-despliega sin detección hardcodeada en TypeScript.

---

## 5. Convención de archivos

Todo bajo `usdcop-trading-dashboard/public/data/production/`:

| Archivo | Propósito | Lo escribe | Lo lee |
|---------|-----------|------------|--------|
| `approval_state.json` | Gates + estado | Export Python / API | `/dashboard`, `/production` |
| `summary_{year}.json` | Métricas OOS backtest | `--phase backtest` | `/dashboard` |
| `summary.json` | Métricas producción | `--phase production` | `/production` |
| `trades/{sid}.json` | Trades producción | `--phase production` | `/production` |
| `trades/{sid}_{year}.json` | Trades backtest | `--phase backtest` | `/dashboard` |

Solo UNA estrategia activa a la vez (un `summary.json`, un `approval_state.json`).
Ver `../../rules/strategy-contract.md` para el `StrategySelector`.

---

## 6. Reset

```bash
python scripts/pipeline/train_and_export_smart_simple.py --reset-approval
```

Reescribe `approval_state.json` con gates frescos, `status: PENDING_APPROVAL` y
`approved_by`/`approved_at` en `null`. El operador debe re-aprobar (Vote 2 otra vez).

---

## 7. Componentes de dashboard

| Componente | Página | Rol |
|------------|--------|-----|
| `ApprovalPanel` | `/dashboard` | Interactivo: gates + botones Aprobar/Rechazar + notas |
| `ApprovalStatusCard` | `/production` | Solo lectura: badge APPROVED/REJECTED/PENDING |

### Integridad del Vote 2 (audit I-4)

El voto humano se emite sobre los números del **bundle publicado** (`summary_*.json` +
`approval_state.json.gates`), nunca sobre métricas recomputadas por el frontend. El replay de
`/dashboard` puede recomputar un preview, pero se etiqueta "PREVIEW DEL REPLAY" y
`GatesPanel`/`ApprovalPanel` siempre renderizan los gates del bundle
(`ForecastingBacktestSection.tsx::displayGates = approval.gates`). Ver
`../../rules/quant-constitution.md` §7.

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Invariantes de aprobación (auto-cargadas) | `../../rules/approval-gates.md` |
| Ciclo de vida completo (Stages 0-7) | `mlops-lifecycle.md` |
| Schemas de estrategia/trade | `../../rules/strategy-contract.md` |
| Contrato de datos del dashboard | `dashboard-integration.md` |
