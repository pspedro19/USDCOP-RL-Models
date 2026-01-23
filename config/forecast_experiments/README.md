# Forecasting Experiment Configurations

This directory contains YAML configuration files for forecasting A/B experiments.

## Quick Start

1. **Run baseline experiment first** (establishes control):
   ```bash
   python scripts/run_forecast_experiment.py --config baseline_v1.yaml
   ```

2. **Run treatment experiment** (compared against baseline):
   ```bash
   python scripts/run_forecast_experiment.py --config feature_oil_vix_v1.yaml
   ```

3. **View results**:
   - Check Airflow logs for DAG `forecast_l4_02_experiment_runner`
   - Query `bi.forecast_experiment_comparisons` for A/B results
   - View MLflow for model artifacts

## Configuration Schema

```yaml
experiment:
  name: "experiment_name"           # Unique identifier
  version: "1.0.0"                  # Version for tracking
  description: "What this tests"    # Human-readable description
  hypothesis: "Expected outcome"    # What you expect to see
  baseline_experiment: "baseline_v1" # null if this IS the baseline

models:
  include: null  # null = all models, or list of model IDs

horizons:
  include: null  # null = all horizons, or list [1, 5, 10, ...]

features:
  contract_version: "1.0.0"
  additions: []   # Features to add
  removals: []    # Features to remove

training:
  walk_forward_windows: 5   # Number of CV folds
  min_train_pct: 0.4        # Minimum training data percentage
  gap_days: 30              # Days between train/test

evaluation:
  primary_metric: "direction_accuracy"
  secondary_metrics: ["rmse", "mae"]
  significance_level: 0.05
  bonferroni_correction: true

mlflow:
  enabled: true
  experiment_name: "custom_name"  # Optional, defaults to forecast_exp_{name}
```

## Available Models

| Model ID | Description |
|----------|-------------|
| `ridge` | Ridge Regression |
| `bayesian_ridge` | Bayesian Ridge |
| `ard` | Automatic Relevance Determination |
| `xgboost_pure` | XGBoost (ML only) |
| `lightgbm_pure` | LightGBM (ML only) |
| `catboost_pure` | CatBoost (ML only) |
| `hybrid_xgboost` | XGBoost + Linear blend |
| `hybrid_lightgbm` | LightGBM + Linear blend |
| `hybrid_catboost` | CatBoost + Linear blend |

## Available Horizons

Forecasting horizons in days: `1, 5, 10, 15, 20, 25, 30`

## Experiment Naming Convention

```
{type}_{variant}_v{version}.yaml

Types:
- baseline_   → Control experiments
- feature_    → Feature engineering experiments
- model_      → Model architecture/hyperparameter experiments
- horizon_    → Horizon-specific experiments
```

## Example Experiments

| File | Purpose |
|------|---------|
| `baseline_v1.yaml` | Control experiment with default settings |
| `feature_oil_vix_v1.yaml` | Tests adding oil and VIX features |
| `model_xgb_tuned_v1.yaml` | Tests XGBoost with higher regularization |

## Statistical Testing

A/B comparisons use:
- **McNemar test** for direction accuracy (paired binary outcomes)
- **Paired t-test** for RMSE (continuous paired data)
- **Fisher's combined test** for aggregate across horizons
- **Bonferroni correction** for multiple comparisons

## Database Tables

- `bi.forecast_experiment_runs` - Experiment run metadata
- `bi.forecast_experiment_comparisons` - A/B comparison results
- `bi.forecast_experiment_deployments` - Production deployment history

## Contract Reference

- `CTR-FORECAST-EXPERIMENT-CONFIG-001`
- `CTR-FORECAST-AB-001`
- `CTR-DAG-002`
