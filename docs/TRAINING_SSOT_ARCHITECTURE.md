# Training Architecture - Clean Code SSOT
## USD/COP RL Trading System

**Fecha:** 2026-01-17
**Versión:** 2.0.0
**Principio:** DRY - Zero Duplicación

---

## Arquitectura Final

```
src/training/
├── config.py              # SSOT - Hyperparameters canónicos
├── engine.py              # Motor unificado - TODA la lógica de training
├── trainers/
│   └── ppo_trainer.py     # PPO wrapper (Stable Baselines 3)
├── environments/
│   ├── trading_env.py     # Trading environment
│   └── env_factory.py     # Environment factory
└── utils/
    └── reproducibility.py # Seeds, hashing

airflow/dags/
└── l3_model_training.py   # DAG thin wrapper (~270 líneas)

ELIMINADOS (redundantes):
├── scripts/train_with_mlflow.py     ❌ DELETED
├── src/training/train_ssot.py       ❌ DELETED
└── src/ml_workflow/training_pipeline.py  ❌ DELETED
```

---

## Flujo de Training

```
┌─────────────────────────────────────────────────────────────┐
│  Entry Points (thin wrappers)                               │
│  • Airflow DAG (l3_model_training.py)                       │
│  • CLI: python -c "from src.training import run_training"   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TrainingEngine (src/training/engine.py)                    │
│  ÚNICA CLASE QUE HACE TRAINING                              │
│                                                             │
│  1. _validate_dataset()     → Valida CSV, computa hash      │
│  2. _generate_norm_stats()  → Genera estadísticas           │
│  3. _create_contract()      → Crea feature contract         │
│  4. _init_mlflow()          → Inicializa tracking           │
│  5. _train_model()          → Ejecuta PPO via PPOTrainer    │
│  6. _register_model()       → Registra en DB                │
│  7. _finalize_mlflow()      → Cierra run                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Componentes Reutilizados                                   │
│  • EnvironmentFactory    → Crea train/eval environments     │
│  • PPOTrainer            → Wrapper SB3 PPO                  │
│  • PPO_HYPERPARAMETERS   → Valores canónicos                │
└─────────────────────────────────────────────────────────────┘
```

---

## Uso

### Desde Python

```python
from pathlib import Path
from src.training import run_training

result = run_training(
    project_root=Path("."),
    version="v1",
    dataset_path=Path("data/processed/RL_DS3_MACRO_CORE.csv"),
    total_timesteps=500_000,
)

if result.success:
    print(f"Model: {result.model_path}")
    print(f"MLflow run: {result.mlflow_run_id}")
```

### Desde Airflow

El DAG llama al engine automáticamente:

```bash
# Trigger manual
airflow dags trigger v3.l3_model_training

# Con configuración
airflow dags trigger v3.l3_model_training \
    --conf '{"version": "v5", "total_timesteps": 1000000}'
```

---

## Valores Canónicos (SSOT)

Definidos en `src/training/config.py`:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| learning_rate | 3e-4 | Standard PPO |
| batch_size | 64 | Eficiente para GPU |
| gamma | 0.90 | Horizonte corto (5min FX) |
| ent_coef | 0.05 | Más exploración |
| total_timesteps | 500,000 | Balance calidad/tiempo |

---

## Beneficios de la Consolidación

| Antes | Después |
|-------|---------|
| 4 archivos con lógica duplicada | 1 engine unificado |
| ~3000 líneas en training scripts | ~600 líneas (engine) |
| DAG de 1550 líneas | DAG de 270 líneas |
| Inconsistencias en hyperparameters | SSOT único |
| Difícil de mantener | Single point of change |

### Métricas

- **Reducción de código**: ~60%
- **Archivos eliminados**: 3
- **Single Responsibility**: ✅
- **DRY**: ✅ Zero duplicación
- **Testeable**: Engine se puede unit-testear sin Airflow

---

## Archivos Eliminados

| Archivo | Razón |
|---------|-------|
| `scripts/train_with_mlflow.py` | Redundante - DAG hace lo mismo |
| `src/training/train_ssot.py` | Reemplazado por engine.py |
| `src/ml_workflow/training_pipeline.py` | Reemplazado por engine.py |

---

## Testing

```python
# Test del engine sin training real
from src.training import TrainingEngine, TrainingRequest
from unittest.mock import patch, MagicMock

def test_engine_initialization():
    engine = TrainingEngine(project_root=Path("."))
    assert engine is not None

def test_request_validation():
    request = TrainingRequest(
        version="test",
        dataset_path=Path("data/test.csv"),
    )
    assert request.version == "test"
```

---

*Documento actualizado: 2026-01-17*
*Arquitectura: Clean Code DRY*
*Autor: Trading Team*
