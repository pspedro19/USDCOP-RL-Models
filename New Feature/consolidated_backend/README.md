# Backend Consolidado - USD/COP Forecasting Pipeline

## Estructura del Proyecto (Mejores Prácticas)

```
consolidated_backend/
├── src/                          # Core ML Pipeline (Single Source of Truth)
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuración centralizada (Pydantic)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base classes (SOLID)
│   │   ├── config.py             # ML Config (Optuna, Pipeline, Models)
│   │   └── exceptions.py         # Custom exceptions
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # DataLoader - carga CSV, valida
│   │   ├── validator.py          # DataValidator - calidad datos
│   │   ├── reconciler.py         # DataReconciler - merge CSV + DB
│   │   └── alignment_validator.py # Validación alineación temporal
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── common.py             # prepare_features, create_targets
│   │   └── transformer.py        # FeatureTransformer (scaling)
│   │
│   ├── models/                   # 9 Modelos ML
│   │   ├── __init__.py
│   │   ├── factory.py            # ModelFactory (Factory Pattern)
│   │   ├── ridge.py              # Ridge Regression (MEJOR: 60.3% DA)
│   │   ├── bayesian_ridge.py     # Bayesian Ridge
│   │   ├── ard.py                # ARD (Auto Relevance Determination)
│   │   ├── xgboost_model.py      # XGBoost Pure
│   │   ├── lightgbm_model.py     # LightGBM Pure
│   │   ├── catboost_model.py     # CatBoost Pure
│   │   ├── hybrids.py            # Modelos Híbridos (Ridge + Classifier)
│   │   └── optuna_tuner.py       # Optimización Optuna
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # DA, RMSE, MAE, R², Sharpe
│   │   ├── purged_kfold.py       # PurgedKFold (evita data leakage)
│   │   ├── backtest.py           # Backtester base
│   │   ├── walk_forward_backtest.py # Walk-forward validation
│   │   └── visualization.py      # TrainingReporter
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── model_plots.py        # Ranking, Heatmaps
│   │   ├── forecast_plots.py     # Fan charts, predicciones
│   │   └── backtest_plots.py     # Equity curves, drawdown
│   │
│   ├── reports/
│   │   ├── __init__.py
│   │   └── generator.py          # ReportGenerator
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── data_quality.py       # DataQualityMonitor
│   │   └── quality_report.py     # QualityReport dataclasses
│   │
│   ├── mlops/
│   │   ├── __init__.py
│   │   ├── mlflow_client.py      # MLflow tracking
│   │   └── minio_client.py       # MinIO S3 storage
│   │
│   └── database.py               # DB connection pool
│
├── api/                          # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── config.py                 # API config
│   ├── database.py               # DB connections
│   │
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── jwt_handler.py        # JWT tokens
│   │   └── dependencies.py       # get_current_user
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py               # POST /auth/*
│   │   ├── forecasts.py          # GET /forecasts/*
│   │   ├── models.py             # GET /models/*
│   │   ├── images.py             # GET /images/*
│   │   └── health.py             # GET /health/*
│   │
│   ├── schemas/                  # Pydantic Schemas (CONTRATOS)
│   │   ├── __init__.py
│   │   ├── forecasts.py          # ForecastResponse, ForecastDashboard
│   │   ├── models.py             # ModelMetrics, ModelComparison
│   │   └── health.py             # HealthResponse
│   │
│   └── requirements.txt
│
├── pipelines/                    # Scripts de ejecución
│   ├── __init__.py
│   ├── run_hybrid_improved.py    # Pipeline principal
│   ├── run_hybrid_ensemble.py
│   ├── run_statistical_validation.py
│   └── results/                  # Resultados generados
│       └── hybrid_improved/
│           └── {timestamp}/
│               ├── metrics.csv
│               ├── summary.json
│               ├── forecasts.csv
│               ├── models/       # .pkl guardados
│               └── figures/      # PNG plots
│
├── scripts/                      # Utilidades
│   ├── __init__.py
│   ├── clear_minio.py
│   ├── generate_bi_csv.py
│   ├── run_forward_forecast.py
│   └── verify_env.py
│
├── data-engineering/             # ETL y Scrapers
│   ├── __init__.py
│   ├── dags/                     # Airflow DAGs
│   └── scrapers/                 # Data scrapers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── data/                         # Datos
│   ├── raw/
│   └── processed/
│
├── requirements.txt              # Dependencies principales
├── docker-compose.yml
├── .env.example
└── CONTRACTS.md                  # Contratos Backend ↔ Frontend
```

---

## Principios de Diseño Aplicados

### SOLID
- **S**ingle Responsibility: Cada módulo tiene una responsabilidad
- **O**pen/Closed: Extensible via Factory y Strategy patterns
- **L**iskov Substitution: BaseModel permite intercambiar modelos
- **I**nterface Segregation: Interfaces específicas por dominio
- **D**ependency Inversion: Depende de abstracciones (BaseModel)

### DRY (Don't Repeat Yourself)
- `common.py`: Funciones compartidas de features
- `factory.py`: Creación centralizada de modelos
- `config.py`: Configuración única (SSOT)

### SSOT (Single Source of Truth)
- `settings.py`: Todas las variables de entorno
- `config.py`: Configuración ML centralizada
- `schemas/`: Contratos Pydantic únicos

---

## Flujo de Datos

```
CSV → DataLoader → DataValidator → FeatureTransformer
                        ↓
              prepare_features()
              create_targets()
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   OptunaConfig   PurgedKFold   WalkForwardBacktest
        ↓               ↓               ↓
   OptunaTuner → ModelFactory → 9 Modelos
                        ↓
              ReportGenerator
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   metrics.csv    summary.json    figures/*.png
        ↓               ↓               ↓
        └───────────────┼───────────────┘
                        ↓
                   FastAPI
                        ↓
                   Frontend
```

---

## Archivos Generados (Output)

| Archivo | Descripción | Consumidor |
|---------|-------------|------------|
| `metrics.csv` | Matriz modelo × horizon con DA, RMSE | `/models/*` endpoints |
| `summary.json` | Mejor modelo, DA promedio, metadata | `/forecasts/dashboard` |
| `forecasts.csv` | Predicciones forward | `/forecasts/*` endpoints |
| `figures/*.png` | Visualizaciones | `/images/*` endpoints |
| `models/*.pkl` | Modelos serializados | Inference |

---

## Variables de Entorno

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=usdcop
DB_USER=postgres
DB_PASSWORD=postgres

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# MLflow
MLFLOW_TRACKING_URI=./mlruns

# API
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET_KEY=change-in-production

# ML Pipeline
ML_HORIZONS=1,5,10,15,20,25,30
ML_MODELS=ridge,bayesian_ridge,xgboost,lightgbm,catboost
OPTUNA_N_TRIALS=50
```

---

## Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python -m pipelines.run_hybrid_improved

# Iniciar API
uvicorn api.main:app --reload --port 8000

# Ejecutar tests
pytest tests/ -v
```
