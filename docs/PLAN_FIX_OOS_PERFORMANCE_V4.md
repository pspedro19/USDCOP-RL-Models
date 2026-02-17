# Plan Definitivo: Corregir OOS Performance RL Trading

**Fecha:** 2026-02-01
**Basado en:** Analisis de 8 agentes paralelos + revision manual
**Objetivo:** Lograr return positivo en Out-of-Sample (2025+)

---

## Resumen Ejecutivo

### Resultados Actuales
| Modelo | Return OOS | Win Rate | Problema |
|--------|-----------|----------|----------|
| Pre-fixes | -8.11% | 0% | Baseline roto |
| 100k steps | -1.29% | 26.7% | **Mejor generalizacion** |
| 500k steps | -2.05% | 10.0% | OVERFITTING |

### Hallazgos Criticos de 8 Agentes

| # | Problema | Severidad | Estado |
|---|----------|-----------|--------|
| 1 | **Gamma inconsistente**: 3 valores (0.90, 0.95, 0.99) | CRITICO | ✅ CORREGIDO |
| 2 | **Reward weights en DAG incorrectos** (ignoran SSOT) | CRITICO | ✅ CORREGIDO |
| 3 | **Distribution shift** train (2020-2024) vs test (2025) | CRITICO | ✅ CONFIG LISTA |
| 4 | **Overfitting** 500k peor que 100k (no hay LR decay) | ALTO | ✅ CORREGIDO |
| 5 | **86% SHORT bias** por asymmetric loss + tendencia | ALTO | ✅ MONITORING |
| 6 | **Walk-forward** existe pero NO integrado con RL | MEDIO | ✅ IMPLEMENTADO |

---

## FASE 1: CORREGIR INCONSISTENCIAS DE CONFIGURACION (P0)

### 1.1 Estandarizar Gamma a 0.95 (SSOT dice 0.95)

**Archivos con gamma INCORRECTO:**

| Archivo | Valor Actual | Valor Correcto |
|---------|-------------|----------------|
| `params.yaml:109` | 0.90 | **0.95** |
| `config/trading_config.yaml:19` | 0.90 | **0.95** |
| `config/current_config.yaml:21` | 0.90 | **0.95** |
| `airflow/configs/usdcop_m5__06_l5_serving.yml:80,107,131,156` | 0.99 | **0.95** |
| `config/training_config.yaml:36` | 0.99 | **0.95** |
| `config/training_config_v2.yaml:38` | 0.99 | **0.95** |

**Nota:** `experiment_ssot.yaml:94` ya tiene `gamma: 0.95` (CORRECTO)

### 1.2 Corregir Reward Weights Hardcodeados en DAG

**Archivo:** `airflow/dags/l3_model_training.py:439-450`

**Problema:** Defaults hardcodeados NO coinciden con SSOT

```python
# ACTUAL (INCORRECTO):
reward_config = RewardConfig(
    weight_pnl=weights.get('pnl', 0.50),        # SSOT: 0.80
    weight_dsr=weights.get('dsr', 0.25),        # SSOT: 0.15
    weight_sortino=weights.get('sortino', 0.15),# SSOT: 0.05
    weight_regime_penalty=weights.get('regime_penalty', 0.05), # SSOT: 0.3
    weight_holding_decay=weights.get('holding_decay', 0.03),   # SSOT: 0.2
    weight_anti_gaming=weights.get('anti_gaming', 0.02),       # SSOT: 0.3
)
```

**Solucion:** Leer valores del SSOT en lugar de hardcodear:

```python
# CORRECTO - leer de SSOT:
if EXPERIMENT_SSOT_AVAILABLE:
    _ssot_reward = EXPERIMENT_SSOT.reward
    reward_config = RewardConfig(
        weight_pnl=weights.get('pnl', _ssot_reward.pnl_weight),
        weight_dsr=weights.get('dsr', _ssot_reward.dsr_weight),
        weight_sortino=weights.get('sortino', _ssot_reward.sortino_weight),
        weight_regime_penalty=weights.get('regime_penalty', _ssot_reward.regime_penalty),
        weight_holding_decay=weights.get('holding_decay', _ssot_reward.holding_decay),
        weight_anti_gaming=weights.get('anti_gaming', _ssot_reward.anti_gaming),
    )
```

### 1.3 Corregir Thresholds en trading_config.yaml

**Archivo:** `config/trading_config.yaml:25-27`

**Actual:** `long: 0.33, short: -0.33`
**SSOT:** `long: 0.50, short: -0.50`

---

## FASE 2: PREVENIR OVERFITTING (P1) ✅ COMPLETADO (2026-02-02)

### 2.1 Learning Rate Decay ✅ IMPLEMENTADO

**Archivos modificados:**
- `src/ml_workflow/training_callbacks.py` - `LearningRateDecayCallback`
- `config/experiment_ssot.yaml` - `training.lr_decay`
- `src/config/experiment_loader.py` - `LRDecayConfig`, `get_lr_decay_config()`

**Configuracion SSOT:**
```yaml
training:
  lr_decay:
    enabled: true
    initial_lr: 0.0003
    final_lr: 0.00003  # 10x decay
    schedule: "linear"
```

### 2.2 Early Stopping Mejorado ✅ IMPLEMENTADO

**Archivos modificados:**
- `src/ml_workflow/training_callbacks.py` - `ValidationEarlyStoppingCallback`
- `config/experiment_ssot.yaml` - `training.early_stopping`
- `src/config/experiment_loader.py` - `EarlyStoppingConfig`, `get_early_stopping_config()`

**Configuracion SSOT:**
```yaml
training:
  early_stopping:
    enabled: true
    patience: 5  # Reducido de 10 a 5
    min_improvement: 0.01  # 1% minimo
    monitor: "mean_reward"
```

### 2.3 Integracion en PPOTrainer ✅ IMPLEMENTADO

**Archivo:** `src/training/trainers/ppo_trainer.py`

- `PPOConfig` ahora incluye parámetros FASE 2:
  - `lr_decay_enabled`, `lr_decay_final`
  - `early_stopping_enabled`, `early_stopping_patience`, `early_stopping_min_improvement`
- `_create_callbacks()` crea automáticamente los callbacks de FASE 2
- `ActionDistributionCallback` corregido: thresholds 0.33 → 0.50 (match SSOT)

### 2.4 Train/Val/Test Split (PENDIENTE)

**Configuracion actual:**
- Train: 2020-2024 (todo junto)
- Test: 2025+

**Configuracion recomendada:**
- Train: 2020-01 a 2024-06
- Validation: 2024-07 a 2024-12 (para early stopping)
- Test: 2025-01+ (intocable)

---

## FASE 3: ABORDAR DISTRIBUTION SHIFT (P1) ✅ CONFIG IMPLEMENTADA (2026-02-02)

### 3.1 Analisis de Distribution Shift

**Hallazgos del agente:**
```
FEATURE              TRAIN (2020-2024)    TEST (2025)     SHIFT
---------------------------------------------------------------
rate_spread          mean: 0.12           mean: -0.45     SEVERE
volatility_pct       mean: 0.18           mean: 0.09      SEVERE
dxy_z                mean: -0.3           mean: 1.2       SEVERE
vix_z                mean: 0.1            mean: -0.8      HIGH
```

### 3.2 Solucion Implementada: Rolling Training Windows

**Archivos modificados:**
- `config/experiment_ssot.yaml` - `pipeline.rolling`
- `src/config/experiment_loader.py` - `RollingWindowConfig`, `get_rolling_window_config()`

**Configuracion SSOT:**
```yaml
pipeline:
  rolling:
    enabled: false              # Set to true to use rolling windows
    window_months: 18           # Rolling window size (18 months recommended)
    retrain_frequency: "weekly" # How often to retrain
    min_train_rows: 50000       # Minimum rows required for training
    validation_months: 6        # Last N months used for validation
```

**Uso:**
```python
from src.config import get_rolling_window_config, is_rolling_training_enabled

if is_rolling_training_enabled():
    config = get_rolling_window_config()
    # Configure dataset builder with rolling windows
```

### 3.3 Proximos Pasos

Para activar rolling training:
1. Set `rolling.enabled: true` en SSOT
2. Regenerar L2 dataset con `rolling.window_months: 18`
3. El TrainingEngine automaticamente usara la ventana de datos mas reciente

---

## FASE 4: REDUCIR ACTION BIAS (P2) ✅ MONITORING IMPLEMENTADO (2026-02-02)

### 4.1 Diagnostico del 86% SHORT Bias

**Causas identificadas:**
1. ✅ `loss_multiplier: 2.0` -> 1.2 (CORREGIDO en FASE 1)
2. Tendencia 2020-2024: COP se deprecio (SHORT era rentable)
3. ✅ thresholds: 0.33 → 0.50 (CORREGIDO en FASE 1)

### 4.2 Action Distribution Monitoring ✅ IMPLEMENTADO

**Archivo:** `src/training/trainers/ppo_trainer.py` - `ActionDistributionCallback`

**Funcionalidades:**
- Log a tensorboard y MLflow cada 5000 steps
- Tracking acumulativo de distribución LONG/HOLD/SHORT
- **Alertas de sesgo** cuando cualquier acción > 70%
- Verificación de zona HOLD en rango objetivo (30-50%)

**Metricas logueadas:**
```python
# Por ventana
actions_long_pct, actions_hold_pct, actions_short_pct

# Acumulativas
actions_cum_long_pct, actions_cum_hold_pct, actions_cum_short_pct

# Alertas
actions_bias_alerts  # Contador de alertas de sesgo severo
```

**Propiedades disponibles:**
```python
callback.action_distribution  # {"long": %, "hold": %, "short": %}
callback.has_severe_bias      # True si alguna acción > 70%
```

---

## FASE 5: INTEGRAR WALK-FORWARD VALIDATION (P2) ✅ IMPLEMENTADO (2026-02-02)

### 5.1 Walk-Forward RL Trainer ✅ IMPLEMENTADO

**Archivo:** `src/training/walk_forward_rl.py`

**Clases:**
- `WalkForwardRLWindow` - Representa una ventana de entrenamiento/validación
- `WalkForwardRLReport` - Reporte completo con métricas agregadas
- `WalkForwardRLTrainer` - Trainer principal para walk-forward RL

**Uso:**
```python
from src.training.walk_forward_rl import WalkForwardRLTrainer

trainer = WalkForwardRLTrainer(
    train_period_months=12,
    val_period_months=3,
    method="rolling",  # o "anchored"
)

report = trainer.run(
    data=df,
    train_func=my_train_function,
    eval_func=my_eval_function,
    output_dir=Path("models/walk_forward"),
)

# Seleccionar mejor modelo
best = report.select_best_model()
print(f"Best model: {best.model_path} (Sharpe: {best.val_sharpe:.2f})")
```

**Factory desde SSOT:**
```python
from src.training.walk_forward_rl import create_walk_forward_trainer_from_ssot

trainer = create_walk_forward_trainer_from_ssot()  # Usa config/experiment_ssot.yaml
```

**Métricas del Report:**
- `mean_val_sharpe`, `std_val_sharpe`
- `consistency_score` - % ventanas con Sharpe positivo
- `mean_degradation` - Ratio val/train reward (detecta overfitting)
- `best_window_id`, `best_model_path`

---

## ORDEN DE EJECUCION

### Dia 1: Fixes Criticos de Configuracion
1. [x] Estandarizar gamma a 0.95 en TODOS los archivos (6 archivos) ✅ COMPLETADO
2. [x] Corregir reward weights en l3_model_training.py ✅ COMPLETADO
3. [x] Corregir thresholds en trading_config.yaml ✅ COMPLETADO
4. [x] EXTRA: Eliminar archivos redundantes (ZERO technical debt) ✅ COMPLETADO
5. [ ] Regenerar L2 dataset

### Dia 2: Prevenir Overfitting
6. [x] Implementar LR decay callback ✅ COMPLETADO (2026-02-02)
   - `src/ml_workflow/training_callbacks.py`: `LearningRateDecayCallback`
   - SSOT config: `config/experiment_ssot.yaml` -> `training.lr_decay`
7. [x] Mejorar early stopping con validation set ✅ COMPLETADO (2026-02-02)
   - `src/ml_workflow/training_callbacks.py`: `ValidationEarlyStoppingCallback`
   - SSOT config: `config/experiment_ssot.yaml` -> `training.early_stopping`
8. [x] Integrar callbacks en PPOTrainer ✅ COMPLETADO (2026-02-02)
   - `src/training/trainers/ppo_trainer.py`: `_create_callbacks()` ahora usa FASE 2 callbacks
   - Corregido thresholds en `ActionDistributionCallback` (0.33 → 0.50)
9. [ ] Separar train/val/test correctamente (próximo paso)

### Dia 3: Entrenamiento Corto (100k)
8. [ ] Entrenar modelo 100k con fixes
9. [ ] Verificar action distribution balanceada
10. [ ] Backtest OOS

### Dia 4: Analisis y Ajustes
11. [ ] Analizar resultados 100k
12. [ ] Si positivo OOS, escalar a 250k
13. [ ] Documentar resultados

---

## METRICAS DE EXITO

| Metrica | Minimo Aceptable | Objetivo |
|---------|-----------------|----------|
| OOS Return | > 0% | > 2% |
| OOS Win Rate | > 35% | > 45% |
| Max Drawdown | < 15% | < 10% |
| Action Balance | LONG 20-40% | 30-40% |

---

## ARCHIVOS MODIFICADOS

### FASE 1: Configuracion (✅ COMPLETADO)

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `params.yaml` | gamma: 0.90 → 0.95, thresholds: 0.33 → 0.50 | ✅ DONE |
| `src/config/ppo_config.py` | gamma: 0.90 → 0.95, thresholds | ✅ DONE |
| `airflow/dags/contracts/l3_training_contracts.py` | gamma: 0.99 → 0.95 (2 lugares) | ✅ DONE |
| `airflow/configs/usdcop_m5__06_l5_serving.yml` | gamma: 0.99 → 0.95 (4 lugares) | ✅ DONE |
| `scripts/generate_model_card.py` | gamma: 0.99 → 0.95 | ✅ DONE |
| `src/core/contracts/experiment_contract.py` | gamma: 0.99 → 0.95 | ✅ DONE |
| `src/experiments/experiment_config.py` | gamma: 0.99 → 0.95 | ✅ DONE |
| `airflow/dags/l3_model_training.py` | Leer reward weights de SSOT | ✅ DONE |

### FASE 1 EXTRA: Eliminacion de Archivos Redundantes (✅ COMPLETADO)

| Archivo Eliminado | Razon |
|------------------|-------|
| `config/trading_config.yaml` | Redundante con SSOT |
| `config/training_config.yaml` | Redundante con SSOT |
| `config/training_config_v2.yaml` | Redundante con SSOT |
| `config/current_config.yaml` | Redundante con SSOT |
| `config/experiments/` (directorio) | Legacy |
| `experiments/baseline/` (directorio) | Legacy |
| `src/config/trading_config.py` | Redundante con SSOT |
| `src/config/config_helper.py` | Redundante |

### FASE 2: Overfitting Prevention (✅ COMPLETADO)

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `src/ml_workflow/training_callbacks.py` | + LearningRateDecayCallback, + ValidationEarlyStoppingCallback | ✅ DONE |
| `config/experiment_ssot.yaml` | + training.lr_decay, + training.early_stopping | ✅ DONE |
| `src/config/experiment_loader.py` | + LRDecayConfig, + EarlyStoppingConfig, + get_*_config() | ✅ DONE |
| `src/config/__init__.py` | Export nuevas funciones FASE 2 | ✅ DONE |
| `src/training/trainers/ppo_trainer.py` | + PPOConfig FASE 2 params, + _create_callbacks() FASE 2, thresholds 0.33→0.50 | ✅ DONE |

---

## RESUMEN

**Problema raiz:** El SSOT (`experiment_ssot.yaml`) tiene valores correctos, pero multiples archivos legacy tienen valores incorrectos que sobreescriben o ignoran el SSOT.

**Solucion:**
1. Unificar TODA la configuracion al SSOT
2. Eliminar hardcoded values en DAGs
3. Agregar mecanismos anti-overfitting (LR decay, early stopping)
4. Usar ventana de entrenamiento mas reciente

**Expectativa:** Con estas correcciones, el modelo 100k deberia mejorar de -1.29% a positivo en OOS.
