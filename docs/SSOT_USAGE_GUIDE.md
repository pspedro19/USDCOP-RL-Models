# SSOT Usage Guide - Single Source of Truth

## Arquitectura

```
config/experiment_ssot.yaml          ← ÚNICA fuente de verdad
         │
         ▼
src/config/experiment_loader.py      ← Módulo Python que lo lee
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
 L2 Builder  L3 PPO  L4 Backtest  Dashboard
```

## Uso Correcto

### Para obtener gamma:

```python
# ✅ CORRECTO - usa el SSOT
from src.config.experiment_loader import get_gamma

gamma = get_gamma()  # Siempre retorna el valor de experiment_ssot.yaml
```

```python
# ❌ INCORRECTO - hardcoded
gamma = 0.95  # No hagas esto
```

### Para obtener hyperparameters de PPO:

```python
# ✅ CORRECTO
from src.config.experiment_loader import get_ppo_hyperparameters

hyperparams = get_ppo_hyperparameters()
model = PPO("MlpPolicy", env, **hyperparams)
```

```python
# ❌ INCORRECTO - hardcoded values
model = PPO("MlpPolicy", env, gamma=0.95, learning_rate=0.0003, ...)
```

### Para obtener reward weights:

```python
# ✅ CORRECTO
from src.config.experiment_loader import get_reward_weights

weights = get_reward_weights()
# weights = {'pnl': 0.8, 'dsr': 0.15, 'sortino': 0.05, ...}
```

## Funciones Disponibles

| Función | Retorna | Descripción |
|---------|---------|-------------|
| `load_experiment_config()` | `ExperimentConfig` | Configuración completa |
| `get_gamma()` | `float` | Discount factor (0.95) |
| `get_learning_rate()` | `float` | Learning rate (0.0003) |
| `get_ent_coef()` | `float` | Entropy coefficient |
| `get_batch_size()` | `int` | Batch size |
| `get_ppo_hyperparameters()` | `Dict` | Todos los hyperparams de PPO |
| `get_reward_weights()` | `Dict` | Pesos del reward function |
| `get_environment_config()` | `Dict` | Config de TradingEnv |
| `get_feature_order()` | `Tuple[str]` | Orden de features |
| `get_training_config()` | `TrainingConfig` | Dataclass con training config |
| `get_reward_config()` | `RewardConfig` | Dataclass con reward config |

## Archivos Deprecados

Los siguientes archivos están **DEPRECADOS** y no deben usarse directamente:

- `config/training_config.yaml` - DEPRECATED
- `config/training_config_v2.yaml` - DEPRECATED
- `config/trading_config.yaml` - DEPRECATED
- `config/current_config.yaml` - DEPRECATED
- `params.yaml` (sección train) - DEPRECATED

Estos archivos se mantienen solo por compatibilidad con código legacy.

## El ÚNICO Archivo Autoritativo

**`config/experiment_ssot.yaml`** es la ÚNICA fuente de verdad.

Si necesitas cambiar:
- gamma
- learning_rate
- reward weights
- thresholds
- cualquier hyperparameter

**Cámbialo SOLO en `experiment_ssot.yaml`**

## Ejemplo de Migración

### Antes (malo):

```python
# En algún archivo random
config = yaml.load(open('config/trading_config.yaml'))
gamma = config['training']['gamma']
```

### Después (bueno):

```python
from src.config.experiment_loader import get_gamma

gamma = get_gamma()
```

## Beneficios

1. **DRY** - Un solo lugar para cada valor
2. **KISS** - Funciones simples y claras
3. **Consistencia** - Todos los componentes usan los mismos valores
4. **Auditabilidad** - Fácil de rastrear qué valores se usaron
5. **Testing** - Fácil de mockear en tests

## En Airflow DAGs

```python
# En L3 DAG
from src.config.experiment_loader import (
    get_ppo_hyperparameters,
    get_reward_weights,
    load_experiment_config,
)

# Los DAGs ya usan el loader, pero asegúrate de usar
# force_reload=True si necesitas valores frescos:
config = load_experiment_config(force_reload=True)
```
