---
kind: roadmap
status: SUPERSEDED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/contracts/signal_contract.py
  - src/contracts/strategy_manifest.py
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
  - scripts/pipeline/normalize_champions.py
---

# PLAN — Dos superficies en un Control Plane: ACTION (replay/acción) vs DIAGNOSTIC (forecasting)

> **SUPERSEDED 2026-07-27**: absorbido por `04-CTR-QLAB-FABRIC-004.md` (§3). Se
> conserva como detalle de referencia; las tareas pendientes viven en `backlog/`.

> Directiva del operador (2026-07-27). Los dos casos se manejan como **dos superficies
> independientes dentro del mismo Control Plane**:
>
> 1. **Replay/acción:** decide comprar, vender, mantener o quedar `FLAT`; produce PnL y
>    puede llegar a ejecución.
> 2. **Forecasting:** predice precio, retorno o dirección; produce métricas y
>    visualizaciones, pero **nunca órdenes**.
>
> Comparten los datos de mercado, pero tienen objetivos, validación, métricas,
> artefactos, permisos y vistas diferentes. La solución NO es aislarlos en dos sistemas
> desconectados: es compartir datos, linaje, métricas y observabilidad, con una
> **muralla contractual y de permisos** entre predicción y decisión.

## 1. Arquitectura conjunta

```text
                       DATOS COMPARTIDOS
Proveedor → Raw Snapshot → Canonical Snapshot → Features causales
                                                │
                         ┌──────────────────────┴─────────────────────┐
                         │                                            │
                  SUPERFICIE ACTION                         SUPERFICIE DIAGNOSTIC
                  Replay / estrategia                         Forecasting
                         │                                            │
                 Señal y exposición                           Predicción H1…H30
                         │                                            │
                 Backtest / paper                             Métricas predictivas
                         │                                            │
                  Book allocator                              Panel forecasting
                         │                                            │
                    Executor                                NO ejecución / NO capital
```

La bifurcación ocurre **después del dato canónico**. No se duplica ingesta, calendarios,
vintages ni validaciones básicas.

## 2. Superficie ACTION (replay/acción)

Registro:

```yaml
id: usdcop_smart_simple_v11
asset: usdcop
surface: action
capabilities: [backtest, gate, signal, paper, verify, execute]
lifecycle_state: CHAMPION
capital_tier: FULL
operational_state: NOMINAL
```

**Responsabilidad**: ¿qué posición debo tener, cuándo entro, cuándo salgo y cuánto
riesgo asumo?

**Flujo**: datos hasta t → features causales → regla/modelo CONGELADO → señal
LONG/SHORT/FLAT → target exposure → simulador determinista → órdenes/fills simulados →
posiciones y PnL → gates → paper → canary → champion → allocator → executor.

**Outputs**: `strategy_output`, `signal`, `order`, `fill`, `position`, `fact_pnl`,
bundles `summary/trades/signals`, resultados del juez.

**Métricas principales**: retorno neto · MaxDD · **Calmar** · Sharpe/Sortino · DSR y
PBO · turnover · exposición · slippage · implementation shortfall · fill rate ·
**paridad replay-paper-live** · descomposición PnL (beta, timing, carry, residual).

> La exactitud predictiva puede monitorearse, pero **no es la métrica que decide si la
> estrategia sirve**. La unidad de evaluación es la decisión económica completa.

**Vistas frontend**:

| Vista | Propósito |
|---|---|
| `/replay` | Reproducir señales, trades y equity histórica |
| `/dashboard` | Vote 2, gates y aprobación |
| `/production` | Monitorear paper, canary o live |
| `/execution` | Órdenes, fills, posiciones, riesgo y kill switch |
| Control Tower | Capital, PnL, drawdown y estado de cada sleeve |

## 3. Superficie DIAGNOSTIC (forecasting)

Registro separado:

```yaml
id: usdcop_forecast_zoo_v3
asset: usdcop
surface: diagnostic
capabilities: [fit, predict, evaluate, publish_panel]
lifecycle_state: ACTIVE
operational_state: NOMINAL
# prohibido declarar:
signal: false
allocate: false
execute: false
approve: false
```

**Responsabilidad**: ¿qué precio, retorno o dirección estima el modelo para un
horizonte determinado?

**Flujo**: datos hasta t → features → target supervisado → walk-forward/OOF →
modelo×horizonte → predicción → evaluación contra baseline → CSV/tablas/gráficos →
`/forecasting`.

**Outputs**: `forecast`, `forecast_interval`, `forecast_probability`,
`forecast_score`, `model_horizon_metrics`, métricas de calibración, CSV, PNG, panel JSON.

**Métricas principales**: DA · balanced-DA · **lift contra baseline** · RMSE/MAE ·
pinball · Brier (cuando aplique) · calibration error · cobertura de intervalos ·
drift de features · estabilidad por horizonte.

**Vistas frontend**:

| Vista | Propósito |
|---|---|
| `/forecasting` | Predicciones y evaluación |
| `/analysis` | Contexto, explicaciones y diagnóstico |
| Control Tower | Solo salud del modelo, **no capital** |

**Esta superficie NO muestra**: botones de aprobación · órdenes · capital asignado ·
recomendaciones de compra/venta · estado `CHAMPION` · enlaces al executor.

## 4. Comportamiento cuando algo falla

| Falla | Forecasting (diagnostic) | Estrategia (action) |
|---|---|---|
| Datos nuevos no llegan | Mostrar `STALE` | **Bloquear nueva señal** |
| PNG ausente | Ocultar gráfico | Sin efecto |
| Un modelo falla | Mostrar resto del zoo | Si es componente activo: **fail-closed** |
| Métrica no disponible | Marcar `N/A` | Gate no puede aprobar |
| Prediction output inválido | No publicar panel | Nunca reutilizar último forecast |
| Señal inválida | No aplica | `target_exposure = 0` o política degradada |
| Paridad rota | Advertencia diagnóstica | `QUARANTINED` |
| Executor caído | Sin efecto | No enviar órdenes |
| Forecast sin lift | Mostrar veredicto negativo | **No invalida una estrategia rentable** |
| Estrategia con mal PnL | Sin efecto sobre forecasting | Reducir capital o retirar |

> **Regla operativa**: la caída del forecasting público degrada una página. La caída de
> un componente de la estrategia **bloquea dinero**.

## 5. Frontend: barreras visibles

`/forecasting` muestra permanentemente:

```text
DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN
```

Campos: Modelo · Horizonte · Predicción · Intervalo · DA · Lift · RMSE · Última
actualización · Estado de datos. **Sin colores de compra/venta**: se escribe
"probabilidad estimada de subida: 58%", jamás "COMPRAR".

`/replay` y `/production` muestran: Estrategia · Versión · Estado · Señal · Exposición
objetivo · PnL · Drawdown · Slippage · Juez · Linaje. Aquí SÍ existen LONG/SHORT/FLAT —
son decisiones auditables de una política congelada.

## 6. Modelo final

```text
L0–L1 compartido
    ↓
┌─────────────────────────────────────────────────────────┐
│ ACTION: estrategia → señal → simulación → PnL → juez    │
│         → allocator → ejecución                         │
│ Métrica reina: desempeño económico y riesgo             │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ DIAGNOSTIC: modelo → forecast → evaluación → gráficos   │
│ Métrica reina: lift y error predictivo                  │
│ Sin asignación, sin órdenes, sin Vote 2                 │
└─────────────────────────────────────────────────────────┘
```

## 7. Estado actual vs objetivo (as-built 2026-07-27)

| Elemento del plan | Hoy | Gap |
|---|---|---|
| Datos compartidos, bifurcación post-canónico | ✅ (L0 único, seeds/DB) | — |
| Bundles ACTION inmutables + replay | ✅ (BundlePublisher, signals/trades/summary) | — |
| `/forecasting` sin botones de acción + caveat | ✅ (test de regresión del caveat) | falta el banner permanente exacto y quitar semántica de color |
| Registro con `surface:` explícito | ❌ | añadir campo a manifiestos/registry |
| Contratos separados strategy_output vs forecast_output | parcial (UniversalSignalRecord existe; forecast sin contrato tipado) | ver plan 01 |
| Esquemas DB action/forecast/execution/control | ❌ (todo en public) | ver plan 01 |
| CI de muralla (diagnostic no ejecuta, etc.) | parcial (rbac:check, manifests) | ver plan 01 |
| Doble linaje de trials (forecast_family/action_family) | parcial (un registry por activo) | ver plan 02 |
