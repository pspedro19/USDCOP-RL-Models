# A/B Testing End-to-End Guide
## MinIO-First Architecture con Modular Reward System

Este documento explica cómo ejecutar A/B testing completo en el sistema USDCOP-RL.

---

## 1. Arquitectura del Sistema A/B Testing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         A/B TESTING FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L0: Macro Data Ingestion                                                   │
│       │ (FRED, TwelveData, BanRep, EMBI)                                   │
│       ↓                                                                      │
│  L1: Feature Computation                                                    │
│       │ (CanonicalFeatureBuilder SSOT)                                     │
│       ↓                                                                      │
│  L2: Dataset Preparation ─────────────────────────────────────────────────┐ │
│       │                                                                    │ │
│       │ Experiment A Config              Experiment B Config               │ │
│       │ (baseline_v1.yaml)               (new_reward_v1.yaml)              │ │
│       ↓                                  ↓                                 │ │
│  ┌────────────────────┐           ┌────────────────────┐                   │ │
│  │ L3: Train Model A  │           │ L3: Train Model B  │                   │ │
│  │ (RewardConfig A)   │           │ (RewardConfig B)   │                   │ │
│  └─────────┬──────────┘           └─────────┬──────────┘                   │ │
│            │                                │                              │ │
│            ↓                                ↓                              │ │
│  ┌────────────────────┐           ┌────────────────────┐                   │ │
│  │ MinIO:             │           │ MinIO:             │                   │ │
│  │ experiments/       │           │ experiments/       │                   │ │
│  │   baseline_v1/     │           │   new_reward_v1/   │                   │ │
│  │     models/        │           │     models/        │                   │ │
│  │     reward_config/ │           │     reward_config/ │                   │ │
│  └─────────┬──────────┘           └─────────┬──────────┘                   │ │
│            │                                │                              │ │
│            └────────────┬───────────────────┘                              │ │
│                         ↓                                                  │ │
│  ┌─────────────────────────────────────────────┐                           │ │
│  │ L4: Experiment Runner                       │                           │ │
│  │ - Backtest both models                      │                           │ │
│  │ - Compute Sharpe, Drawdown, Win Rate        │                           │ │
│  │ - Statistical comparison (ab_statistics)    │                           │ │
│  └─────────────────────┬───────────────────────┘                           │ │
│                        ↓                                                   │ │
│  ┌─────────────────────────────────────────────┐                           │ │
│  │ Promotion Gate                              │                           │ │
│  │ - min_sharpe >= 0.5                         │                           │ │
│  │ - max_drawdown <= 30%                       │                           │ │
│  │ - min_win_rate >= 40%                       │                           │ │
│  │ - min_trades >= 50                          │                           │ │
│  └─────────────────────┬───────────────────────┘                           │ │
│                        ↓                                                   │ │
│  ┌─────────────────────────────────────────────┐                           │ │
│  │ Winner Selection                            │                           │ │
│  │ - p-value < 0.05                            │                           │ │
│  │ - Effect size (Cohen's d)                   │                           │ │
│  │ - Bootstrap confidence intervals            │                           │ │
│  └─────────────────────┬───────────────────────┘                           │ │
│                        ↓                                                   │ │
│           ┌────────────────────────┐                                       │ │
│           │ Promote to Production  │                                       │ │
│           │ s3://production/models/│                                       │ │
│           │ + model_registry (PG)  │                                       │ │
│           └────────────────────────┘                                       │ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Preparación de Experimentos

### 2.1 Crear Configuración de Experimento A (Baseline)

```yaml
# config/experiments/baseline_v1.yaml
experiment:
  name: "baseline_v1"
  description: "Baseline model with default reward config"
  version: "v1.0.0"

# Dataset configuration
dataset:
  date_range_start: "2024-01-01"
  date_range_end: "2024-12-31"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# Training configuration
training:
  total_timesteps: 500000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64

# Reward configuration (DEFAULT SSOT)
reward:
  contract_id: "v1.0.0"
  enable_curriculum: true
  enable_normalization: true
  weights:
    pnl: 0.50
    dsr: 0.30
    sortino: 0.20
    regime_penalty: 1.0
    holding_decay: 1.0
    anti_gaming: 1.0
```

### 2.2 Crear Configuración de Experimento B (Treatment)

```yaml
# config/experiments/new_reward_v1.yaml
experiment:
  name: "new_reward_v1"
  description: "New model with enhanced macro reward weighting"
  version: "v1.1.0"

# Dataset configuration (same as baseline)
dataset:
  date_range_start: "2024-01-01"
  date_range_end: "2024-12-31"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# Training configuration (same)
training:
  total_timesteps: 500000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64

# Reward configuration (MODIFIED - higher macro sensitivity)
reward:
  contract_id: "v1.1.0"
  enable_curriculum: true
  enable_normalization: true
  weights:
    pnl: 0.40           # Reduced from 0.50
    dsr: 0.35           # Increased from 0.30
    sortino: 0.25       # Increased from 0.20
    regime_penalty: 1.5 # Increased from 1.0
    holding_decay: 0.8  # Reduced from 1.0
    anti_gaming: 1.2    # Increased from 1.0

  # Enhanced macro sensitivity
  oil_tracker:
    enabled: true
    breakdown_penalty: 0.15  # Increased from 0.10

  banrep_detector:
    enabled: true
    penalty_multiplier: 2.0  # Increased from 1.5
```

---

## 3. Métodos de Ejecución

### 3.1 Método 1: Script Automatizado (Recomendado)

```bash
# Ejecutar A/B test completo
./scripts/run_ab_experiment.sh all \
  --experiment-a baseline_v1 \
  --experiment-b new_reward_v1
```

Este script ejecuta:
1. Verificación de infraestructura (Postgres, Redis, MinIO, MLflow, Airflow)
2. Ingesta de datos (L0 + L1)
3. Preparación de datasets (L2)
4. Entrenamiento de ambos modelos (L3)
5. Backtest y comparación (L4)
6. Reporte de resultados

### 3.2 Método 2: DAGs de Airflow

#### Paso 1: Ejecutar L0 (Datos Macro)
```bash
airflow dags trigger v3.l0_macro_unified
```

#### Paso 2: Ejecutar L1 (Features)
```bash
airflow dags trigger v3.l1_feature_refresh
```

#### Paso 3: Ejecutar L2 para Experimento A
```bash
airflow dags trigger v3.l2_preprocessing_pipeline \
  --conf '{
    "experiment_name": "baseline_v1",
    "date_range_start": "2024-01-01",
    "date_range_end": "2024-12-31"
  }'
```

#### Paso 4: Ejecutar L3 para Experimento A
```bash
airflow dags trigger v3.l3_model_training \
  --conf '{
    "version": "v1.0.0",
    "experiment_name": "baseline_v1",
    "reward_contract_id": "v1.0.0",
    "enable_curriculum": true
  }'
```

#### Paso 5: Repetir L2 y L3 para Experimento B
```bash
# L2
airflow dags trigger v3.l2_preprocessing_pipeline \
  --conf '{
    "experiment_name": "new_reward_v1",
    "date_range_start": "2024-01-01",
    "date_range_end": "2024-12-31"
  }'

# L3 con reward config custom
airflow dags trigger v3.l3_model_training \
  --conf '{
    "version": "v1.1.0",
    "experiment_name": "new_reward_v1",
    "reward_contract_id": "v1.1.0",
    "enable_curriculum": true,
    "reward_weights": {
      "pnl": 0.40,
      "dsr": 0.35,
      "sortino": 0.25,
      "regime_penalty": 1.5,
      "holding_decay": 0.8,
      "anti_gaming": 1.2
    }
  }'
```

#### Paso 6: Ejecutar Comparación (L4)
```bash
airflow dags trigger l4_experiment_runner \
  --conf '{
    "experiment_name": "new_reward_v1",
    "compare_with": "baseline_v1",
    "notify_on_complete": true
  }'
```

### 3.3 Método 3: Python API Directo

```python
from src.ml_workflow.experiment_manager import ExperimentManager
from src.training.config import RewardConfig
from src.training.engine import TrainingEngine, TrainingRequest
from src.inference.ab_statistics import ABStatistics
from pathlib import Path

# ============================================================
# 1. Configurar Experimentos
# ============================================================

# Experimento A: Baseline
exp_a = ExperimentManager("baseline_v1")
reward_config_a = RewardConfig()  # Default SSOT

# Experimento B: Treatment
exp_b = ExperimentManager("new_reward_v1")
reward_config_b = RewardConfig(
    weight_pnl=0.40,
    weight_dsr=0.35,
    weight_sortino=0.25,
    weight_regime_penalty=1.5,
    weight_holding_decay=0.8,
    weight_anti_gaming=1.2,
)

# ============================================================
# 2. Entrenar Modelo A
# ============================================================

engine = TrainingEngine(project_root=Path("."))

request_a = TrainingRequest(
    version="v1.0.0",
    dataset_path=Path("data/train_baseline.csv"),
    experiment_name="baseline_v1",
    reward_config=reward_config_a,
    reward_contract_id="v1.0.0",
    enable_curriculum=True,
    total_timesteps=500000,
)

result_a = engine.run(request_a)

# Guardar reward config para lineage
exp_a.save_reward_config(reward_config_a, contract_id="v1.0.0")

# ============================================================
# 3. Entrenar Modelo B
# ============================================================

request_b = TrainingRequest(
    version="v1.1.0",
    dataset_path=Path("data/train_new_reward.csv"),
    experiment_name="new_reward_v1",
    reward_config=reward_config_b,
    reward_contract_id="v1.1.0",
    enable_curriculum=True,
    total_timesteps=500000,
)

result_b = engine.run(request_b)

# Guardar reward config para lineage
exp_b.save_reward_config(reward_config_b, contract_id="v1.1.0")

# ============================================================
# 4. Ejecutar Backtests
# ============================================================

from src.backtest.engine.unified_backtest_engine import UnifiedBacktestEngine

# Backtest A
backtest_a = UnifiedBacktestEngine()
result_a_bt = backtest_a.run(
    model_path=result_a.model_path,
    data_path=Path("data/test.csv"),
)

# Backtest B
backtest_b = UnifiedBacktestEngine()
result_b_bt = backtest_b.run(
    model_path=result_b.model_path,
    data_path=Path("data/test.csv"),
)

# ============================================================
# 5. Comparación Estadística
# ============================================================

ab_stats = ABStatistics()

# Comparar Sharpe Ratios
sharpe_result = ab_stats.compare_sharpe_ratios(
    control_sharpe=result_a_bt.sharpe_ratio,
    treatment_sharpe=result_b_bt.sharpe_ratio,
    control_returns=result_a_bt.returns,
    treatment_returns=result_b_bt.returns,
)

# Comparar Win Rates
winrate_result = ab_stats.compare_win_rates(
    control_wins=result_a_bt.winning_trades,
    control_total=result_a_bt.total_trades,
    treatment_wins=result_b_bt.winning_trades,
    treatment_total=result_b_bt.total_trades,
)

# Bayesian A/B Test
bayesian_result = ab_stats.bayesian_ab_test(
    control_successes=result_a_bt.winning_trades,
    control_total=result_a_bt.total_trades,
    treatment_successes=result_b_bt.winning_trades,
    treatment_total=result_b_bt.total_trades,
)

# ============================================================
# 6. Decisión de Promoción
# ============================================================

print("=" * 60)
print("A/B TEST RESULTS")
print("=" * 60)
print(f"Sharpe A: {result_a_bt.sharpe_ratio:.3f}")
print(f"Sharpe B: {result_b_bt.sharpe_ratio:.3f}")
print(f"Sharpe p-value: {sharpe_result.p_value:.4f}")
print(f"Win Rate A: {result_a_bt.win_rate:.2%}")
print(f"Win Rate B: {result_b_bt.win_rate:.2%}")
print(f"Win Rate p-value: {winrate_result.p_value:.4f}")
print(f"Bayesian P(B > A): {bayesian_result.posterior_probability:.2%}")
print("=" * 60)

# Determinar ganador
if sharpe_result.p_value < 0.05 and result_b_bt.sharpe_ratio > result_a_bt.sharpe_ratio:
    winner = "new_reward_v1"
    print(f"RECOMMENDATION: Promote {winner} (p < 0.05)")
else:
    winner = "baseline_v1"
    print(f"RECOMMENDATION: Keep {winner} (no significant improvement)")

# ============================================================
# 7. Promover Ganador (si aplica)
# ============================================================

if winner == "new_reward_v1":
    from src.ml_workflow.promotion_service import PromotionService

    promotion = PromotionService()
    model_id = promotion.promote(
        experiment_id="new_reward_v1",
        version="v1.1.0",
    )
    print(f"Model promoted to production: {model_id}")
```

---

## 4. Verificación de Resultados

### 4.1 Verificar en MinIO

```bash
# Listar experimentos
mc ls minio/experiments/

# Ver modelo A
mc ls minio/experiments/baseline_v1/models/

# Ver modelo B
mc ls minio/experiments/new_reward_v1/models/

# Ver reward configs
mc ls minio/experiments/baseline_v1/reward_configs/
mc ls minio/experiments/new_reward_v1/reward_configs/
```

### 4.2 Verificar en PostgreSQL

```sql
-- Ver modelos registrados
SELECT model_id, model_version, status,
       reward_contract_id, curriculum_final_phase,
       test_sharpe, test_max_drawdown, test_win_rate
FROM model_registry
ORDER BY created_at DESC
LIMIT 5;

-- Ver modelos desplegados
SELECT * FROM active_models;

-- Comparar reward configs
SELECT model_id, reward_contract_id, reward_weights
FROM model_registry
WHERE experiment_id IN ('baseline_v1', 'new_reward_v1');
```

### 4.3 Verificar en MLflow

```bash
# Abrir MLflow UI
mlflow ui --port 5000

# O via CLI
mlflow runs list --experiment-name usdcop-rl-training
```

---

## 5. Comparación de Experimentos (CLI)

```bash
# Comparar dos runs de MLflow
python scripts/compare_experiments.py \
  --run-id-a <mlflow_run_id_baseline> \
  --run-id-b <mlflow_run_id_new_reward> \
  --output-format markdown

# O por nombre de experimento
python scripts/compare_experiments.py \
  --experiment-name usdcop-rl-training \
  --baseline baseline_v1 \
  --treatment new_reward_v1 \
  --output-format json
```

### Output Ejemplo:

```markdown
## Experiment Comparison: baseline_v1 vs new_reward_v1

### Metrics Comparison
| Metric | Baseline | Treatment | Change | Significant |
|--------|----------|-----------|--------|-------------|
| Sharpe Ratio | 0.85 | 1.12 | +31.8% | ✓ (p=0.023) |
| Max Drawdown | 18.2% | 15.4% | -15.4% | ✓ (p=0.041) |
| Win Rate | 52.1% | 55.3% | +6.1% | ✗ (p=0.187) |
| Total Trades | 234 | 198 | -15.4% | - |

### Config Differences
| Parameter | Baseline | Treatment |
|-----------|----------|-----------|
| reward.weights.pnl | 0.50 | 0.40 |
| reward.weights.dsr | 0.30 | 0.35 |
| reward.weights.regime_penalty | 1.0 | 1.5 |

### Recommendation
**PROMOTE TREATMENT** - Statistically significant improvement in Sharpe Ratio
```

---

## 6. Flujo de Datos de Macro Features en Reward Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACRO FEATURES → REWARD COMPONENTS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Database: macro_indicators_daily                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ dxy, vix, embi, brent, wti, usdmxn, treasury_2y, treasury_10y       │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ↓                                                  │
│  15-Dimensional Observation Vector                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Index │ Feature           │ Source      │ Used By                    │   │
│  │───────│───────────────────│─────────────│────────────────────────────│   │
│  │   0   │ log_ret_5m        │ OHLCV       │ OilTracker (correlation)   │   │
│  │   4   │ atr_pct           │ OHLCV       │ BanrepDetector, RegimeDetect│   │
│  │   6   │ dxy_z             │ L0 Macro    │ (Available for agents)     │   │
│  │   7   │ dxy_change_1d     │ L0 Macro    │ (Available for agents)     │   │
│  │   8   │ vix_z             │ L0 Macro    │ (Available for agents)     │   │
│  │   9   │ embi_z            │ L0 Macro    │ (Available for agents)     │   │
│  │  10   │ brent_change_1d   │ L0 Macro    │ OilTracker (oil return)    │   │
│  │  11   │ rate_spread       │ L0 Macro    │ (Available for agents)     │   │
│  │  12   │ usdmxn_change_1d  │ L0 Macro    │ (Available for agents)     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ↓                                                  │
│  Reward Components (Use observation indices)                                │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐   │
│  │ BanrepDetector     │ │ OilCorrelation     │ │ RegimeDetector         │   │
│  │ Input: obs[4]      │ │ Input: obs[10],    │ │ Input: obs[4]          │   │
│  │ (atr_pct)          │ │        obs[0]      │ │ (atr_pct)              │   │
│  │                    │ │ (brent, log_ret)   │ │                        │   │
│  │ Output: penalty    │ │ Output: penalty    │ │ Output: adjustment     │   │
│  │ when volatility    │ │ when oil-COP       │ │ based on volatility    │   │
│  │ spike detected     │ │ correlation breaks │ │ regime (LOW/MED/HIGH)  │   │
│  └────────────────────┘ └────────────────────┘ └────────────────────────┘   │
│                          │                                                  │
│                          ↓                                                  │
│  Total Reward = w_pnl*PnL + w_dsr*DSR + w_sortino*Sortino                  │
│               - regime_penalty - oil_penalty - banrep_penalty               │
│               - holding_decay - anti_gaming                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Checklist Pre-Ejecución

```bash
# 1. Verificar infraestructura
docker-compose ps  # Debe mostrar: postgres, redis, minio, mlflow, airflow

# 2. Verificar datos macro
psql -c "SELECT COUNT(*) FROM macro_indicators_daily WHERE fecha >= '2024-01-01';"
# Debe retornar > 250 (días de trading)

# 3. Verificar conexiones
mc alias ls minio
psql -c "SELECT 1;"
redis-cli ping

# 4. Verificar DAGs activos
airflow dags list | grep -E "l0_macro|l1_feature|l2_preprocessing|l3_model|l4_experiment"

# 5. Verificar feature contract
python -c "from src.core.contracts.feature_contract import FEATURE_ORDER; print(len(FEATURE_ORDER))"
# Debe retornar: 15

# 6. Verificar reward config
python -c "from src.training.config import REWARD_CONFIG; print(REWARD_CONFIG)"
```

---

## 8. Troubleshooting

### Error: "Dataset hash mismatch"
```bash
# Regenerar dataset
airflow dags trigger v3.l2_preprocessing_pipeline --conf '{"force_rebuild": true}'
```

### Error: "MinIO connection refused"
```bash
# Verificar MinIO está corriendo
docker-compose up -d minio
mc admin info minio
```

### Error: "Reward config not found"
```python
# Crear reward config manualmente
from src.ml_workflow.experiment_manager import ExperimentManager
from src.training.config import RewardConfig

manager = ExperimentManager("experiment_name")
manager.save_reward_config(RewardConfig(), contract_id="v1.0.0")
```

### Error: "Promotion gate validation failed"
```bash
# Ver detalles del error
python scripts/promote_experiment.py --experiment-id <exp> --version <ver> --dry-run
```

---

## 9. Monitoreo de Shadow Mode

Después de promover un modelo, ejecutar en shadow mode para comparación en tiempo real:

```python
from src.inference.model_router import ModelRouter
from src.inference.shadow_pnl import ShadowPnLTracker

# Inicializar router con champion y shadow
router = ModelRouter(
    champion_model_id="baseline_v1",
    shadow_model_id="new_reward_v1",
)

# Trackear PnL virtual del shadow
shadow_tracker = ShadowPnLTracker()

# En cada tick:
result = router.predict(observation)
# Champion signal se usa para trading real
# Shadow signal se registra para comparación
shadow_tracker.track(
    signal=result.shadow_signal,
    price=current_price,
    timestamp=now,
)

# Ver métricas de shadow
print(f"Agreement rate: {router.agreement_rate:.1%}")
print(f"Shadow Sharpe: {shadow_tracker.virtual_sharpe:.3f}")
print(f"Shadow Win Rate: {shadow_tracker.win_rate:.1%}")
```

---

## 10. Resumen de Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `airflow/dags/l3_model_training.py` | Entrenamiento con RewardConfig |
| `airflow/dags/l4_experiment_runner.py` | Orquestación de A/B test |
| `src/training/config.py` | SSOT para RewardConfig |
| `src/inference/ab_statistics.py` | Tests estadísticos |
| `src/inference/shadow_pnl.py` | Tracking de PnL virtual |
| `src/ml_workflow/promotion_gate.py` | Validación pre-promoción |
| `src/ml_workflow/promotion_service.py` | Flujo de promoción |
| `scripts/compare_experiments.py` | CLI de comparación |
| `scripts/run_ab_experiment.sh` | Script automatizado |

---

**Autor:** Trading Team
**Versión:** 1.0.0
**Contract:** CTR-AB-TESTING-001
**Fecha:** 2026-01-19
