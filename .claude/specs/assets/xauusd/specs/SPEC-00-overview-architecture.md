# SPEC-00 — Arquitectura y Stack Técnico

## Propósito
Fijar la arquitectura del sistema, el flujo de datos, los contratos entre componentes y las decisiones de stack. Es el mapa que las demás specs detallan.

## Arquitectura de componentes y flujo de datos

```
                    ┌──────────────────────────────────────────┐
   FUENTES          │  INGESTA (SPEC-01)  ── Airflow DAG 1      │
  Dukascopy ───────►│  gold H1/tick, DXY, TIPS, calendario      │──► data/raw/ (DVC)
  FRED      ───────►│  descarga incremental, idempotente        │
  Calendario ──────►└──────────────────────────────────────────┘
                                      │
                    ┌──────────────────────────────────────────┐
                    │  PROCESO (SPEC-02) ── Airflow DAG 2        │
                    │  TZ/DST → Sunday → audit → resample →      │──► data/processed/ (DVC)
                    │  align H1↔Daily (point-in-time)           │
                    └──────────────────────────────────────────┘
                                      │
                    ┌──────────────────────────────────────────┐
                    │  FEATURES (SPEC-03) + RÉGIMEN (SPEC-04)    │──► feature store + labels
                    │  H1 técnicas · Daily régimen · macro corr │    (Airflow DAG 2/3)
                    └──────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                         ▼
┌───────────────────────┐                         ┌───────────────────────────────┐
│ BASELINES (SPEC-07)    │                         │ ENTORNO RL (SPEC-05)          │
│ B1 long-only           │                         │ GoldTradingEnv                │
│ B2 trend-follower      │                         │ + RIESGO (SPEC-06) integrado  │
└───────────────────────┘                         │ ── TRAIN (SPEC-08) DAG 4      │
          │                                         └───────────────────────────────┘
          │                                                         │
          └───────────────────────────┬───────────────────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  VALIDACIÓN (SPEC-09) ── Airflow DAG 5    │
                    │  walk-forward · DSR · atribución régimen  │──► reporte + gate
                    └──────────────────────────────────────────┘
                                       │  (si pasa el gate)
                    ┌──────────────────────────────────────────┐
                    │  DESPLIEGUE (SPEC-11) ── DAG 6            │
                    │  paper+shadow → live MT5 → monitoreo      │
                    └──────────────────────────────────────────┘
```

## Contratos entre componentes (formatos canónicos)

Todo se persiste en **Parquet** particionado por año, con timestamps en **UTC** (columna `ts` tz-aware). Convención de nombres de columnas OHLCV: `open, high, low, close, volume`.

| Artefacto | Path | Índice | Clave de partición |
|---|---|---|---|
| Oro crudo H1 | `data/raw/gold/xauusd_h1/` | `ts` (UTC) | `year` |
| Macro crudo | `data/raw/fred/{series_id}/` | `date` | — |
| Calendario | `data/raw/calendar/` | `ts` (UTC) | `year` |
| OHLCV procesado H1 | `data/processed/gold_h1/` | `ts` (UTC) | `year` |
| OHLCV procesado Daily | `data/processed/gold_daily/` | `date` (NY-close) | `year` |
| Features H1 | `data/features/h1/` | `ts` (UTC) | `year` |
| Labels de régimen | `data/features/regime_daily/` | `date` | `year` |
| Splits walk-forward | `data/splits/{fold_id}.json` | — | — |

**Regla point-in-time (crítica):** el label/feature Daily del día `D` solo está disponible para las barras H1 del día `D+1` en adelante (el día D cierra al NY-close). Cualquier violación es look-ahead. Ver SPEC-03 §alineación.

## Decisiones de stack
Ver tabla en `README.md`. Decisiones no obvias registradas en `adr/ADR-log.md`:
- ADR-001: por qué H1+Daily y no intradía puro.
- ADR-002: por qué la capa de riesgo va dentro del entorno.
- ADR-003: por qué correlaciones rodantes y no supuestos macro fijos.
- ADR-004: por qué dos baselines.

## Configuración
Config centralizada en `config/*.yaml`, cargada a dataclasses tipadas (pydantic o attrs). Plantillas en `config/`. Nada de constantes mágicas dispersas en el código.

## Escalabilidad multi-activo (ver SPEC-12)
El Oro **no es un silo**: se onboarda como un activo más sobre la columna vertebral multi-activo existente. El literal `XAUUSD` NUNCA se hardcodea — vive en un **`AssetProfile`** (`config/assets/xauusd.yaml`) que parametriza símbolo, sesión (metales), drivers macro y thresholds de régimen. Los artefactos de backtest se publican al **registro dinámico** (`registry.json` + `manifest.json`) vía el contrato de salida `register_bundle`, con **versionado inmutable** por `(strategy_id, version, year)`, y el frontend los muestra solo (dropdown + replay) leyendo `/api/registry`. Todo **aditivo**: no se rompen los contratos `lib/contracts/*.ts` ni las rutas `app/api/**` existentes. Detalle completo y tests A1–F1 / R1–R9 en [SPEC-12](./SPEC-12-scalable-registry-integration.md) y en `.claude/specs/assets/_onboarding-playbook.md` y `.claude/specs/platform/registry-lifecycle.md`.

## Criterios de aceptación
- [ ] Repo scaffold creado según estructura del README; `pyproject.toml` con dependencias pineadas.
- [ ] `dvc init` + remote configurado; `mlflow` accesible.
- [ ] `config/` con plantillas cargables a dataclasses (test de carga verde).
- [ ] CI mínimo: `pytest` + `ruff`/`black` corriendo en cada push.
- [ ] Todos los timestamps del sistema en UTC tz-aware verificado por test.

## Dependencias
Ninguna (spec raíz).
