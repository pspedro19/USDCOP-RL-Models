# XAU/USD RL Strategy — Paquete de Especificaciones (SDD)

Paquete de especificaciones para implementar la estrategia de trading algorítmico de XAU/USD descrita en [`STRATEGY.md`](./STRATEGY.md). Metodología **Spec-Driven Development + TDD**: cada spec define contrato (inputs/outputs), detalle de implementación concreto y **criterios de aceptación testeables** antes de escribir código.

> **Tesis (recordatorio):** en oro la dirección es casi ruido (~50–53% accuracy). El alpha vive en **gestión de riesgo + adaptación a régimen**, no en la predicción. El RL es el motor; la capa de riesgo es el cinturón; los dos baselines son la conciencia.

---

## Cómo usar este paquete

1. Lee `STRATEGY.md` (el norte) y `SPEC-00` (arquitectura + stack).
2. Sigue `IMPLEMENTATION_ROADMAP.md` — construye por fases, cada una con su gate de aceptación.
3. Cada `SPEC-XX` es autocontenida: **Propósito → Contrato → Implementación → Criterios de aceptación → Dependencias**.
4. Las decisiones de diseño no obvias están registradas en `adr/`.

**Regla de fase:** no avances de fase sin pasar los criterios de aceptación (tests verdes) de la anterior. La fase 6 es el gate honesto: si la mediana de seeds no le gana a **ambos** baselines OOS, el RL no aporta.

---

## Índice de especificaciones

| Spec | Título | Fase | Entregable |
|---|---|---|---|
| [SPEC-00](./specs/SPEC-00-overview-architecture.md) | Arquitectura y stack técnico | — | Scaffold + entornos |
| [SPEC-01](./specs/SPEC-01-data-ingestion.md) | Ingesta de datos (Dukascopy, FRED, calendario) | 1 | `data/raw/` + DVC |
| [SPEC-02](./specs/SPEC-02-data-processing.md) | Procesamiento, TZ/DST, auditoría de calidad | 1 | `data/processed/` |
| [SPEC-03](./specs/SPEC-03-feature-engineering.md) | Feature engineering (H1, Daily, macro rodante) | 1 | Feature store |
| [SPEC-04](./specs/SPEC-04-regime-classifier.md) | Clasificador de régimen (v1 reglas, v2 HMM) | 3 | Labels Daily |
| [SPEC-05](./specs/SPEC-05-rl-environment.md) | Entorno Gymnasium (riesgo integrado) | 4 | `GoldTradingEnv` |
| [SPEC-06](./specs/SPEC-06-risk-layer.md) | Capa de riesgo (vol-targeting, breakers) | 4 | Módulo determinista |
| [SPEC-07](./specs/SPEC-07-baselines.md) | Baselines B1 (long-only) y B2 (trend-follower) | 2 | Benchmarks |
| [SPEC-08](./specs/SPEC-08-model-training.md) | Entrenamiento LSTM→PPO, multi-seed, MLflow | 5 | Modelos + tracking |
| [SPEC-09](./specs/SPEC-09-validation-backtest.md) | Walk-forward, DSR, atribución por régimen | 6 | Reporte de validación |
| [SPEC-10](./specs/SPEC-10-airflow-orchestration.md) | Orquestación Airflow (DAGs) | 1–8 | DAGs productivos |
| [SPEC-11](./specs/SPEC-11-deployment-monitoring.md) | Paper/live, shadow, drift, kill switch | 8–9 | Despliegue |
| [SPEC-12](./specs/SPEC-12-scalable-registry-integration.md) | **Integración con el registro dinámico + fábrica de pipelines** | 0, 6, 8 | AssetProfile + bundle publicado + replay en el front |

Complementos: [`IMPLEMENTATION_ROADMAP.md`](./IMPLEMENTATION_ROADMAP.md) · [`adr/ADR-log.md`](./adr/ADR-log.md) · [`config/`](./config/) (plantillas YAML)

---

## Escalabilidad: el Oro se onboarda como un ACTIVO, no como un silo

> **Actualización 2026-07-03.** Este paquete describía un repo `gold-rl/` aparte con DAGs `xau_*` a mano. El sistema USD/COP ya tiene operativa una **columna vertebral multi-activo / multi-estrategia** — reutilízala en vez de copiarla. Detalle completo en [SPEC-12](./specs/SPEC-12-scalable-registry-integration.md).

El Oro entra por **config + datos**, no por código copiado:

1. **`AssetProfile`** (`config/assets/xauusd.yaml`) parametriza todo lo que hoy está pegado a COP (símbolo, sesión metales, drivers macro, thresholds de régimen). USD/COP mismo pasa a ser `config/assets/usdcop.yaml` — nada es caso especial.
2. El pipeline **publica un bundle** vía `register_bundle` (contrato de salida del DAG): escribe backtest **inmutable versionado** por `(strategy_id, version, year)` y hace upsert de `registry.json` + `manifest.json` — **de forma aditiva**, sin tocar los archivos que el front ya consume.
3. El **frontend lee `/api/registry`** y se arma solo: dropdown Activo→Estrategia→Versión→Año + **replay** dinámico. Cero `strategy_id`/`symbol`/año hardcodeado.
4. La **fábrica de pipelines** (config-driven) emite los DAGs por `(activo, estrategia)` → agregar Oro = una entrada de config, **0 DAGs nuevos**.

> **Estado de la columna vertebral (2026-07-03): IMPLEMENTADA y PROBADA sobre USD/COP.** El publisher está cableado en el export (aditivo), el versionado inmutable coexiste (3 versiones vivas bajo `smart_simple_v11` + rama `smart_simple_aggr`, probado como A/B), las rutas `/api/registry` · `/api/strategies/{id}/manifest` · `/api/registry/promote` existen, y el frontend arma solo el selector Estrategia→Versión + replay + promote (validado con Playwright). Blindado por 9 tests R (`tests/contracts/test_strategy_registry.py`). **El Oro no construye esta maquinaria — la reutiliza.** Lo pendiente es el *onboarding del activo* (crear `config/assets/xauusd.yaml`, aún inexistente), la **fábrica de pipelines** y el rebuild del dashboard baked. Detalle honesto en [SPEC-12 §Estado real hoy](./specs/SPEC-12-scalable-registry-integration.md#estado-real-hoy-honesto--qué-ya-funciona-y-qué-falta-cablear).

Reglas vigentes del sistema que este paquete debe respetar (en `.claude/rules/`):

| Regla | Qué gobierna |
|---|---|
| [`architecture-overview.md`](../rules/architecture-overview.md) | Mapa as-built (infra, contratos, drift TS↔Python §5.3) |
| [`_onboarding-playbook.md`](../rules/_onboarding-playbook.md) | Contrato `AssetProfile`, stages, tests **A1–F1** |
| [`registry-lifecycle.md`](../rules/registry-lifecycle.md) | `StrategyBundleManifest`, `registry.json`, contrato I/O de DAG, inmutabilidad, replay, fábrica, tests **R1–R9** |

---

## Stack técnico

| Capa | Herramienta | Notas |
|---|---|---|
| Orquestación | **Apache Airflow** ≥2.8 | Datasets para scheduling data-aware, TaskFlow API, dynamic task mapping |
| Versionado de datos | **DVC** | Datasets grandes fuera de git; remote en S3/GCS/Azure Blob |
| Tracking de experimentos | **MLflow** | Params, seeds, métricas, artefactos; **fuente del conteo de trials para el DSR** |
| RL | **Gymnasium** + **Stable-Baselines3** + **sb3-contrib** | `RecurrentPPO` para LSTM; PyTorch backend |
| Datos numéricos | pandas, numpy, polars (opcional para velocidad) | Parquet como formato canónico |
| Indicadores | pandas-ta o TA-Lib | Todos causales |
| Macro | fredapi | Series FRED (tasas reales, dólar) |
| Datos de oro | dukascopy-node (CLI) / dukascopy-python | XAUUSD tick→H1→Daily |
| Ejecución live | MetaTrader5 (Python API) | En VPS |
| Estadística | scipy, arch (bootstrap), statsmodels | Deflated Sharpe, block bootstrap |
| Calidad/tests | pytest, great-expectations (opcional) | TDD + data validation |

---

## Estructura del repositorio de código (propuesta)

```
gold-rl/
├── dags/                        # Airflow DAGs (SPEC-10)
│   ├── xau_data_ingestion.py
│   ├── xau_data_processing.py
│   ├── xau_regime.py
│   ├── xau_training.py
│   ├── xau_walkforward.py
│   └── xau_paper_shadow.py
├── src/gold_rl/
│   ├── data/
│   │   ├── ingest/              # SPEC-01: downloaders (gold, fred, calendar)
│   │   ├── process/             # SPEC-02: tz/dst, audit, resample, align
│   │   └── features/            # SPEC-03: feature builders
│   ├── regime/                  # SPEC-04: rules v1, hmm v2, hysteresis
│   ├── env/                     # SPEC-05: GoldTradingEnv + reward (DSR)
│   ├── risk/                    # SPEC-06: sizing, breakers, blackouts
│   ├── baselines/               # SPEC-07: B1, B2
│   ├── train/                   # SPEC-08: policy (LSTM), training loop, mlflow
│   ├── backtest/                # SPEC-09: walk-forward, metrics, DSR, attribution
│   ├── deploy/                  # SPEC-11: mt5 executor, shadow, monitors
│   └── config.py                # dataclasses de config (pydantic/attrs)
├── config/                      # YAMLs de configuración (versionados)
├── tests/                       # pytest (mirror de src/)
├── notebooks/                   # EDA, auditoría de datos, análisis
├── dvc.yaml / .dvc/             # pipeline DVC
├── pyproject.toml
└── README.md
```

> **Nota de integración (SPEC-12):** esta estructura `gold-rl/` es válida como *módulo de ciencia* (datos, régimen, entorno, riesgo, validación), pero **no se despliega como repo aislado**. En producción se enchufa al monorepo existente: el Oro vive como `config/assets/xauusd.yaml` (AssetProfile), sus artefactos de backtest se publican al **registro compartido** (`public/data/strategies/<sid>/…` + `registry.json`) vía `register_bundle`, y sus DAGs los emite la **fábrica de pipelines** (`airflow/dags/factories/…`). Así el Oro reutiliza aprobación (Vote 2), replay y monitoreo sin duplicar infraestructura.

---

## Prerrequisitos

- Python 3.11+, entorno aislado (venv/conda/uv).
- API key de FRED (gratis: https://fred.stlouisfed.org/docs/api/api_key.html).
- Node.js (si usas `dukascopy-node` CLI) o el equivalente Python.
- Cuenta de calendario económico (Trading Economics / Finnhub / scrape ForexFactory).
- Remote de DVC (S3/GCS/Azure) y servidor MLflow (local o remoto).
- Para live: VPS Windows con MT5 + broker que liste XAUUSD.

---

*Aviso: diseño metodológico, no asesoría financiera ni promesa de rentabilidad. El trading de oro implica riesgo sustancial. El backtest espectacular es la señal de alarma, no el premio.*
