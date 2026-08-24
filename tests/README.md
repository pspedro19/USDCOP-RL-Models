# USD/COP Trading System — Test Suite

Índice de la suite. La documentación larga vive en [`docs/`](docs/); en esta raíz solo
quedan `conftest.py`, `__init__.py` y este índice.

## Estructura

| Directorio | Tests | Qué cubre | ¿Lo corre CI? |
|---|---:|---|---|
| `unit/` | 182 | Componentes individuales | **Sí** — `ci.yml` job `unit-tests` |
| `regression/` | 68 | Guards estructurales y de contrato (SSOT, layout, freshness, gates) | **Sí** — job dedicado |
| `integration/` | 29 | Feature parity, vistas de BD, pipeline entre capas | **Sí** |
| `scripts/` | 4 | CLIs de `scripts/` | No |
| `chaos/` | 3 | Degradación e inyección de fallos | No |
| `contracts/` | 3 | Paridad de contratos Python ↔ TypeScript | No |
| `load/` | 2 | Carga y rendimiento | No |
| `onboarding/` | 2 | Alta de un activo nuevo | No |
| `e2e/` | 1 | Pipeline completo | No |
| `data_quality/` | 1 | Validadores de calidad de datos | No |
| `fixtures/` · `support/` | — | Datos de prueba y helpers | — |
| `docs/` | — | Documentación de la suite (no es código) | — |

## Comandos

```bash
pytest tests/unit/ -v                                  # unitarios
pytest tests/regression/ -q                            # guards estructurales
pytest tests/integration/test_feature_parity.py -v     # paridad de features (crítico)
pytest tests/ --cov=src --cov-report=html              # con coverage (gate: 70%)
pytest tests/ -v -m "not slow"                         # excluir lentos
```

## Dos ficheros huérfanos en esta raíz

`test_observation_parity.py` y `test_trading_calendar.py` están **fuera de los tres
directorios que CI recorre**, así que hoy no los ejecuta nadie. En una corrida local
del 2026-08-24 dieron 12 fallos, todos en el calendario de festivos colombianos, con
la librería opcional `colombian-holidays` **sin instalar** — es decir, el fallo puede
ser del entorno y no del código, pero **nadie lo está comprobando**.

No se han movido a `unit/` a propósito: entrarían en el paso bloqueante de CI y lo
pondrían en rojo sin haber diagnosticado primero la causa. La decisión pendiente es
arreglarlos y moverlos, o retirarlos.

## Criterios de éxito

| Métrica | Umbral |
|---|---|
| Feature parity | diff < 1e-6 |
| Observation dimension | == 15 |
| Model compatibility | sin errores |
| `time_normalized` range | [0, 0.983] |
| Coverage | ≥ 70% (gate de CI) |

## Fixtures

**Modelo (PPO)**: `feature_config` (SSOT), `sample_observation` (15 dims),
`sample_ohlcv_df` (100 barras), `sample_macro_df` (10 días).
**Infraestructura**: `db_pool`, `redis_client`, `clean_db` / `clean_redis` (auto-cleanup).

**Datos** — `sample_ohlcv.csv`: 100 barras de 5 min desde 2024-01-02 08:00, base 4200 COP ·
`sample_macro.csv`: 10 días, variables `dxy, vix, embi, brent, treasuries, usdmxn`.
