# USD/COP Trading System - Test Suite

## Estructura de Tests

- unit/ - Tests unitarios de componentes individuales
- integration/ - Tests de integración (feature parity, DB views)
- e2e/ - Tests end-to-end del pipeline completo
- fixtures/ - Datos de prueba

## Comandos Básicos

### Ejecutar todos los tests
pytest tests/ -v

### Solo tests unitarios
pytest tests/unit/ -v

### Test de paridad (crítico)
pytest tests/integration/test_feature_parity.py -v

### Con coverage
pytest tests/ --cov=src --cov-report=html

### Tests rápidos (excluir lentos)
pytest tests/ -v -m "not slow"

## Criterios de Éxito

| Métrica | Umbral |
|---------|--------|
| Feature parity | diff < 1e-6 |
| Observation dimension | == 15 |
| Model compatibility | Sin errores |
| time_normalized range | [0, 0.983] |

## Fixtures Disponibles

### Model Fixtures (PPO Primary)
- feature_config - Configuración SSOT
- sample_observation - Vector de 15 dimensiones
- sample_ohlcv_df - 100 barras OHLCV
- sample_macro_df - 10 días macro

### Infrastructure Fixtures
- db_pool - PostgreSQL connection pool
- redis_client - Redis client
- clean_db / clean_redis - Auto cleanup

## Datos de Fixtures

### sample_ohlcv.csv
- 100 barras de 5 minutos
- Fecha: 2024-01-02 08:00
- Precio base: 4200 COP

### sample_macro.csv
- 10 días de datos macro
- Fecha: 2024-01-02
- Variables: dxy, vix, embi, brent, treasuries, usdmxn
