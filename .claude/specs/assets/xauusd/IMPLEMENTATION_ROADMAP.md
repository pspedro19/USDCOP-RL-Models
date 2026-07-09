# Roadmap de Implementación

Construcción por fases. **No avanzar sin pasar el gate de aceptación** (tests verdes) de la fase previa. La Fase 6 es el gate honesto.

| Fase | Objetivo | Specs | Entregable | Gate de aceptación |
|---|---|---|---|---|
| **0** | Scaffold + entornos + **AssetProfile** | SPEC-00, **SPEC-12** | Repo, DVC, MLflow, CI, config cargable, **`config/assets/xauusd.yaml`** | CI verde; timestamps UTC verificados; **A1 (AssetProfile carga/valida)** |
| **1** | Datos listos | SPEC-01, 02, 03, (10: DAGs 1–2) | `data/features/` + labels crudos | Idempotencia, DST, point-in-time, auditoría de calidad — todos test verdes |
| **2** | Baselines | SPEC-07 | B1 y B2 con métricas OOS | B1 y B2 corren por el harness; comparativa B2 vs B1 documentada |
| **3** | Régimen | SPEC-04 | Labels Daily con histéresis | Estabilidad de labels; sentido económico; point-in-time |
| **4** | Entorno + riesgo | SPEC-05, 06 | `GoldTradingEnv` + `RiskLayer` | `check_env` pasa; riesgo determinista testeado; reward sobre PnL neto |
| **5** | Entrenamiento | SPEC-08, (10: DAG 3) | Agente multi-seed en MLflow | ≥5 seeds; swap en costos; VecNormalize sin leakage; reproducible |
| **6** | Validación + GATE + **publicación de bundle** | SPEC-09, **SPEC-12**, (10: DAG 4) | Reporte walk-forward + **bundle inmutable en el registro** | **Mediana de seeds gana a B1 y B2 OOS, con DSR e IC bootstrap** (= gates E1–E5). Además: `register_bundle` publica `backtests/<version>/…` inmutable + `registry.json` (**R3, R5, R9**). Si no pasa el gate → iterar o quedarse con baseline |
| **6.5** | **Visibilidad en el frontend + replay** | **SPEC-12**, SPEC-11 | Dropdown Activo→Estrategia→Versión→Año + replay dinámico | **R6/R7/R8**: front lee `/api/registry`; v1 y v2 coexisten y son replayables; **aditivo** (endpoints viejos siguen como fallback) |
| **7** | Ensemble (si aplica) | STRATEGY §5 | v2: especialistas + meta-selector por Sharpe | Solo si Fase 6 lo justifica; mismo gate. Cada versión = **bundle inmutable nuevo** |
| **8** | Paper + shadow | SPEC-11 Fase A, (10: DAG 5) | Demo con shadow-compare | Divergencia paper→backtest ~0 |
| **9** | Live | SPEC-11 Fase B | MT5 en VPS + monitoreo | Reconciliación de fills; monitores y kill switch activos |

## Notas de secuencia
- **Baselines (Fase 2) ANTES del RL.** Son la vara; sin ellos no hay con qué comparar. Si B2 no le gana a B1, ya aprendiste algo antes de tocar RL.
- DAGs de Airflow (SPEC-10) se construyen incrementalmente junto a las fases que orquestan, no todos al final.
- El punto de decisión de la Fase 6 es real: **si el RL no aporta sobre los baselines, el mejor baseline ES la estrategia.** Un buen científico hace que la complejidad se gane su lugar.
- **Escalabilidad (SPEC-12):** publica siempre vía `register_bundle` con **versionado inmutable** — nunca sobreescribas `summary_<year>.json`; así v1 y v2 coexisten y son replayables. Todo **aditivo**: no toques `lib/contracts/*.ts` ni las rutas `app/api/**` existentes. Cuando llegue un 2º activo, introduce la **fábrica de pipelines** en vez de copiar DAGs (`registry-lifecycle.md` §6.3) — objetivo "0 DAGs nuevos por activo".

## Estimación de esfuerzo (orden de magnitud, ajústalo)
Fases 0–1 son el grueso del trabajo de ingeniería de datos (donde se ganan o pierden los backtests). Fases 4–6 son el núcleo de investigación. No subestimes la Fase 1: la calidad de datos es el 80% del resultado honesto.
