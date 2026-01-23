# Plan: A/B Testing para Pipeline de Forecasting

## Resumen Ejecutivo

**Estado Actual**: Los DAGs de forecasting NO tienen capacidades de A/B testing.
**Recomendación**: Implementar A/B testing adaptado a la naturaleza semanal del forecasting.

---

## 1. Comparación: RL vs Forecasting A/B Testing

| Componente | RL Pipeline | Forecasting Pipeline | Gap |
|------------|-------------|---------------------|-----|
| **Experiment Runner DAG** | `l4_experiment_runner.py` | ❌ No existe | **CRÍTICO** |
| **Statistical Testing** | `src/inference/ab_statistics.py` | ❌ No existe | **ALTO** |
| **Shadow Mode** | `model_router.py` + `shadow_pnl.py` | ❌ No existe | MEDIO |
| **Canary Deployment** | `deployment_manager.py` | ❌ No existe | BAJO |
| **Experiment Registry** | `022_experiment_registry.sql` | ❌ No existe | **ALTO** |
| **Backtest Validation** | `l4_backtest_validation.py` | ✅ Existe (básico) | Mejorar |
| **Drift Monitor** | `mlops_drift_monitor.py` | ✅ Existe | OK |

---

## 2. ¿Por qué Forecasting necesita A/B Testing?

### 2.1 Escenarios de Uso

| Escenario | Ejemplo | Necesidad |
|-----------|---------|-----------|
| **Nuevo Modelo** | Agregar LSTM a los 9 modelos existentes | Comparar DA vs baseline |
| **Hyperparámetros** | Cambiar `n_estimators` de XGBoost | Validar mejora significativa |
| **Nuevas Features** | Agregar `DXY_weekly_momentum` | Verificar que no empeora |
| **Ensemble Strategy** | Nuevo método de consenso | A/B test por horizon |
| **Reentrenamiento** | Modelo mensual vs trimestral | Comparar walk-forward |

### 2.2 Diferencias Clave con RL

| Aspecto | RL Pipeline | Forecasting Pipeline |
|---------|-------------|---------------------|
| **Frecuencia de Inferencia** | Cada 5 minutos | Semanal |
| **Métrica Principal** | Sharpe Ratio | Direction Accuracy |
| **Tiempo de Validación** | Días | Semanas/Meses |
| **Granularidad de Datos** | 5-min bars | Daily prices |
| **Número de Modelos** | 1 activo + shadow | 9 modelos × 7 horizons |
| **Shadow Mode** | Real-time parallel | Post-hoc comparison |

---

## 3. Arquitectura Propuesta

### 3.1 Componentes Nuevos

```
┌─────────────────────────────────────────────────────────────────┐
│                    FORECASTING A/B TESTING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ forecast_l4_02_  │───▶│ src/forecasting/ │                   │
│  │ experiment_runner│    │ ab_statistics.py │                   │
│  └──────────────────┘    └──────────────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ bi.fact_forecast │    │ bi.forecast_     │                   │
│  │ _experiments     │    │ _comparisons     │                   │
│  └──────────────────┘    └──────────────────┘                   │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌──────────────────┐                                │
│              │ Promotion Gate   │                                │
│              │ (DA > baseline?) │                                │
│              └──────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Archivos a Crear

| Archivo | Propósito | Prioridad |
|---------|-----------|-----------|
| `src/forecasting/ab_statistics.py` | Tests estadísticos para DA, RMSE | P0 |
| `airflow/dags/forecast_l4_experiment_runner.py` | Orquestador de experimentos | P0 |
| `database/migrations/025_forecast_experiments.sql` | Tablas de experimentos | P0 |
| `src/forecasting/experiment_manager.py` | Gestión de experimentos | P1 |
| `config/forecast_experiments/` | Configs YAML | P1 |
| `scripts/compare_forecast_experiments.py` | CLI de comparación | P2 |

---

## 4. Diseño Detallado

### 4.1 Tests Estadísticos para Forecasting

```python
# src/forecasting/ab_statistics.py

class ForecastABStatistics:
    """
    Statistical tests adapted for forecasting models.

    Key differences from RL:
    - Direction Accuracy instead of Sharpe
    - McNemar test for paired predictions
    - RMSE comparison with F-test
    """

    def compare_direction_accuracy(
        self,
        baseline_predictions: pd.DataFrame,
        treatment_predictions: pd.DataFrame,
        actual_prices: pd.DataFrame,
    ) -> DAComparisonResult:
        """
        McNemar test for paired direction accuracy comparison.

        H0: baseline and treatment have same accuracy
        H1: treatment has different accuracy

        Returns p-value and effect size (Cohen's h)
        """
        pass

    def compare_rmse(
        self,
        baseline_errors: pd.Series,
        treatment_errors: pd.Series,
    ) -> RMSEComparisonResult:
        """
        F-test for variance comparison (RMSE).
        Paired t-test for mean error difference.
        """
        pass

    def compare_by_horizon(
        self,
        baseline_results: Dict[int, pd.DataFrame],
        treatment_results: Dict[int, pd.DataFrame],
    ) -> Dict[int, HorizonComparisonResult]:
        """
        Run comparison for each horizon (1, 5, 10, 15, 20, 25, 30 days).
        Apply Bonferroni correction for multiple comparisons.
        """
        pass

    def bootstrap_da_difference(
        self,
        baseline_correct: np.ndarray,
        treatment_correct: np.ndarray,
        n_bootstrap: int = 10000,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for DA difference.
        Returns: (mean_diff, ci_lower, ci_upper)
        """
        pass
```

### 4.2 Experiment Runner DAG

```python
# airflow/dags/forecast_l4_experiment_runner.py

"""
Forecast L4: Experiment Runner
==============================

Orchestrates A/B testing for forecasting models.

Workflow:
    1. Load experiment config (YAML)
    2. Train baseline models (if not cached)
    3. Train treatment models
    4. Run walk-forward backtest on both
    5. Compare using statistical tests
    6. Record results to database
    7. Generate recommendation

Schedule: Manual trigger
Config: experiment_name, compare_with, horizons

Author: Trading Team
Date: 2026-01-22
Contract: CTR-FORECAST-EXPERIMENT-001
"""

# Tasks:
# 1. validate_experiment_config
# 2. prepare_baseline_results (from cache or run)
# 3. train_treatment_models
# 4. run_treatment_backtest
# 5. statistical_comparison
# 6. record_experiment_results
# 7. generate_recommendation
# 8. send_notification
```

### 4.3 Database Schema

```sql
-- database/migrations/025_forecast_experiments.sql

-- Table 1: Experiment runs for forecasting
CREATE TABLE bi.forecast_experiment_runs (
    id SERIAL PRIMARY KEY,
    run_uuid UUID DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(200) NOT NULL,
    experiment_version VARCHAR(50),
    run_id VARCHAR(100) UNIQUE,

    -- Configuration
    config_hash VARCHAR(64),
    models_included TEXT[],  -- e.g., ['ridge', 'xgboost_pure']
    horizons_included INT[], -- e.g., [1, 5, 10, 15, 20, 25, 30]

    -- Status
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results (per model/horizon)
    training_metrics JSONB,
    backtest_metrics JSONB,

    -- MLflow integration
    mlflow_experiment_id VARCHAR(100),
    mlflow_run_ids JSONB,  -- {model_id: run_id}

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table 2: A/B comparisons for forecasting
CREATE TABLE bi.forecast_experiment_comparisons (
    id SERIAL PRIMARY KEY,
    comparison_uuid UUID DEFAULT gen_random_uuid(),

    -- Experiments being compared
    baseline_run_id VARCHAR(100) REFERENCES bi.forecast_experiment_runs(run_id),
    treatment_run_id VARCHAR(100) REFERENCES bi.forecast_experiment_runs(run_id),

    -- Results per horizon
    comparison_results JSONB,
    /*
    {
        "1": {"da_diff": 0.02, "p_value": 0.03, "significant": true},
        "5": {"da_diff": 0.01, "p_value": 0.15, "significant": false},
        ...
    }
    */

    -- Aggregate recommendation
    recommendation VARCHAR(50),  -- deploy_treatment | keep_baseline | inconclusive
    confidence_level DECIMAL(5,4),

    -- Statistical details
    statistical_tests JSONB,
    /*
    {
        "method": "mcnemar",
        "bonferroni_corrected": true,
        "n_comparisons": 7,
        "alpha": 0.05
    }
    */

    compared_at TIMESTAMPTZ DEFAULT NOW(),
    compared_by VARCHAR(100)
);

-- View: Latest experiment results
CREATE VIEW bi.v_forecast_experiment_summary AS
SELECT
    e.experiment_name,
    e.experiment_version,
    e.status,
    e.completed_at,
    jsonb_object_agg(h.horizon, h.da) as direction_accuracy_by_horizon,
    AVG((h.metrics->>'direction_accuracy')::float) as avg_direction_accuracy
FROM bi.forecast_experiment_runs e
CROSS JOIN LATERAL jsonb_each(e.backtest_metrics) as h(horizon, metrics)
WHERE e.status = 'success'
GROUP BY e.experiment_name, e.experiment_version, e.status, e.completed_at;
```

### 4.4 Experiment Config Format

```yaml
# config/forecast_experiments/new_features_v1.yaml

experiment:
  name: new_features_v1
  version: "1.0.0"
  description: "Add DXY momentum feature to forecasting"
  baseline_experiment: baseline_v2
  hypothesis: "DXY momentum improves DA for horizons >= 10d"

models:
  # Which models to include (null = all 9)
  include: null
  # Or specific subset:
  # include: [xgboost_pure, lightgbm_pure, catboost_pure]

horizons:
  # Which horizons to test (null = all 7)
  include: null
  # Or specific subset:
  # include: [10, 15, 20, 25, 30]

features:
  # New features to add
  additions:
    - dxy_momentum_5d
    - dxy_zscore_20d
  # Features to remove (if any)
  removals: []
  # Feature contract version
  contract_version: "2.0.0"

training:
  walk_forward_windows: 5
  min_train_pct: 0.4
  gap_days: 30

evaluation:
  primary_metric: direction_accuracy
  secondary_metrics:
    - rmse
    - sharpe_ratio
  significance_level: 0.05
  bonferroni_correction: true

comparison:
  baseline: baseline_v2
  statistical_tests:
    - mcnemar        # For direction accuracy
    - paired_ttest   # For return differences
    - bootstrap_ci   # For confidence intervals
```

---

## 5. Plan de Implementación

### Fase 1: Core Infrastructure (Prioridad P0)

| Tarea | Archivo | Esfuerzo |
|-------|---------|----------|
| 1.1 | Crear `src/forecasting/ab_statistics.py` | 4h |
| 1.2 | Crear `database/migrations/025_forecast_experiments.sql` | 2h |
| 1.3 | Crear `airflow/dags/forecast_l4_experiment_runner.py` | 6h |
| 1.4 | Actualizar `dag_registry.py` con nuevo DAG | 0.5h |

### Fase 2: Integration (Prioridad P1)

| Tarea | Archivo | Esfuerzo |
|-------|---------|----------|
| 2.1 | Crear `src/forecasting/experiment_manager.py` | 4h |
| 2.2 | Crear directorio `config/forecast_experiments/` | 1h |
| 2.3 | Integrar con MLflow | 2h |
| 2.4 | Actualizar `l3b_forecasting_training.py` para soportar experimentos | 2h |

### Fase 3: Tooling (Prioridad P2)

| Tarea | Archivo | Esfuerzo |
|-------|---------|----------|
| 3.1 | Crear `scripts/compare_forecast_experiments.py` | 3h |
| 3.2 | Crear `scripts/run_forecast_ab_test.sh` | 1h |
| 3.3 | Agregar visualización de resultados | 4h |
| 3.4 | Documentación en `docs/FORECAST_AB_TESTING_GUIDE.md` | 2h |

---

## 6. Decisiones de Diseño

### 6.1 ¿Shadow Mode para Forecasting?

**Recomendación: NO (por ahora)**

| Razón | Explicación |
|-------|-------------|
| Frecuencia baja | Inferencia semanal, no hay urgencia de validación real-time |
| Múltiples modelos | Ya ejecutamos 9 modelos, agregar shadow duplicaría complejidad |
| Walk-forward suficiente | Backtest histórico es más informativo que shadow mode |

**Alternativa**: Usar el campo `model_version` en `bi.fact_forecasts` para comparar versiones post-hoc.

### 6.2 ¿Canary Deployment?

**Recomendación: Simplificado**

| Stage | Descripción |
|-------|-------------|
| **Staging** | Experimento en walk-forward backtest |
| **Production** | Modelo aprobado reemplaza al anterior |

No hay necesidad de traffic splitting porque:
- No hay trading real (solo predicciones)
- Todas las predicciones se almacenan con `model_version`
- Comparación es retrospectiva

### 6.3 Métricas de Comparación

| Métrica | Test Estadístico | Threshold |
|---------|-----------------|-----------|
| Direction Accuracy | McNemar test | p < 0.05 |
| RMSE | Paired t-test | p < 0.05, d > 0.2 |
| Sharpe (if trading) | Welch's t-test | p < 0.05 |
| Win Rate | Chi-square | p < 0.05 |

---

## 7. Ejemplo de Uso

### 7.1 Via Airflow

```bash
# Trigger experiment
airflow dags trigger forecast_l4_02_experiment_runner \
  --conf '{
    "experiment_name": "new_features_v1",
    "compare_with": "baseline_v2",
    "horizons": [10, 15, 20, 25, 30],
    "notify_on_complete": true
  }'
```

### 7.2 Via Python

```python
from src.forecasting.experiment_manager import ForecastExperimentManager
from src.forecasting.ab_statistics import ForecastABStatistics

# Run experiment
exp = ForecastExperimentManager("new_features_v1")
exp.train()
exp.backtest()

# Compare with baseline
baseline = ForecastExperimentManager.load("baseline_v2")
ab = ForecastABStatistics()

result = ab.compare_by_horizon(
    baseline.backtest_results,
    exp.backtest_results,
)

# Recommendation
if result.overall_significant and result.treatment_wins > result.baseline_wins:
    print("RECOMMENDATION: Deploy treatment")
else:
    print("RECOMMENDATION: Keep baseline")
```

---

## 8. Cronograma Sugerido

| Semana | Fase | Entregables |
|--------|------|-------------|
| 1 | Fase 1.1-1.2 | `ab_statistics.py`, migrations |
| 2 | Fase 1.3-1.4 | `experiment_runner.py`, registry |
| 3 | Fase 2.1-2.4 | `experiment_manager.py`, integración |
| 4 | Fase 3.1-3.4 | Scripts, docs, testing |

**Total estimado**: 4 semanas (31.5 horas de desarrollo)

---

## 9. Preguntas para Decisión

1. **¿Priorizar DA o Sharpe?**
   - DA es más interpretable para forecasting
   - Sharpe requiere simular trading (más complejo)

2. **¿Comparar todos los 63 model/horizon combos?**
   - Opción A: Comparar agregado (promedio DA por horizon)
   - Opción B: Comparar cada combo (requiere Bonferroni)

3. **¿Integrar con MLflow experiments?**
   - Recomendado para lineage completo
   - Requiere modificar training DAG

4. **¿Automatizar promotion?**
   - Manual: Humano decide basado en resultados
   - Automático: Si p < 0.05 y DA_treatment > DA_baseline, auto-promote

---

## 10. Conclusión

Los DAGs de forecasting actualmente **NO tienen A/B testing**, lo cual es una brecha significativa comparado con el pipeline de RL.

**Recomendación**: Implementar Fase 1 (P0) inmediatamente para habilitar comparación estadística de experimentos. Las Fases 2 y 3 pueden implementarse incrementalmente según necesidad.

**Diferencia clave con RL**: El forecasting no necesita shadow mode real-time ni canary deployment, pero SÍ necesita:
- Tests estadísticos adaptados (McNemar para DA)
- Comparación por horizon con corrección múltiple
- Registro de experimentos en DB
- DAG orquestador de experimentos
