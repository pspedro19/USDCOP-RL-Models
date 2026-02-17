# PLAN INTEGRAL: Corrección del Sistema RL USDCOP

**Fecha**: 2026-02-01
**Versión**: 1.0
**Autor**: Análisis de 10 agentes en paralelo

---

## RESUMEN EJECUTIVO

El sistema RL tiene **múltiples problemas interconectados** que causan el bajo rendimiento OOS (-8.11% return, 0% win rate). Los problemas principales son:

1. **Reward function desalineada**: PnL solo tiene 50% peso, penalidades dominan
2. **Costos duplicados**: Transaction costs se aplican 2 veces (en net_pnl Y market_impact)
3. **Circuit breaker rompe el aprendizaje**: Bloquea acciones sin feedback
4. **Configuración inconsistente**: Gamma 0.99 vs 0.90 en diferentes archivos
5. **Distribution shift severo**: 6 meses gap + macro features shifted significativamente

---

## FASE 1: CORRECCIONES CRÍTICAS (P0)

### 1.1 FIX: Eliminar Duplicación de Costos de Transacción

**Problema**: Los costos se restan del PnL en `trading_env.py:830` Y TAMBIÉN se aplica market_impact penalty en `reward_calculator.py:476`.

**Archivo**: `src/training/environments/trading_env.py`

```python
# LÍNEA 830 - ACTUAL (MALO):
net_pnl = gross_pnl - trade_cost  # Resta costos aquí

# Y LUEGO en reward_calculator.py:476:
impact_cost = self._market_impact.calculate(...)  # DUPLICA!

# CORRECCIÓN - Opción A: Mantener en net_pnl, eliminar de reward
# En reward_calculator.py línea 399-400, COMENTAR:
# impact_cost = self._market_impact.calculate(...)
# impact_cost = 0.0  # Costos ya incluidos en PnL

# CORRECCIÓN - Opción B: Eliminar de net_pnl, mantener en reward
# En trading_env.py línea 830:
# net_pnl = gross_pnl  # NO restar aquí, se aplica en reward
```

**Recomendación**: Usar Opción A (costos en PnL es más directo).

---

### 1.2 FIX: Aumentar Peso del PnL en Reward Function

**Problema**: PnL solo tiene 50% peso. Penalidades (regime, holding, anti_gaming) dominan.

**Archivo**: `config/experiment_ssot.yaml` (líneas 102-110)

```yaml
# ACTUAL (MALO):
reward:
  pnl_weight: 0.5
  dsr_weight: 0.3
  sortino_weight: 0.2
  regime_penalty: 1.0
  holding_decay: 1.0
  anti_gaming: 1.0

# CORRECCIÓN:
reward:
  pnl_weight: 0.8              # AUMENTAR de 0.5 a 0.8
  dsr_weight: 0.15             # REDUCIR de 0.3 a 0.15
  sortino_weight: 0.05         # REDUCIR de 0.2 a 0.05
  regime_penalty: 0.3          # REDUCIR de 1.0 a 0.3
  holding_decay: 0.2           # REDUCIR de 1.0 a 0.2
  anti_gaming: 0.3             # REDUCIR de 1.0 a 0.3
```

---

### 1.3 FIX: Desactivar Circuit Breaker Durante Entrenamiento

**Problema**: Circuit breaker bloquea acciones sin dar feedback al agente, creando action-reward mismatch.

**Archivo**: `src/training/environments/trading_env.py`

```python
# OPCIÓN A: Desactivar completamente para training
# En TradingEnvConfig, cambiar defaults:
max_consecutive_losses: int = 999  # Efectivamente desactivado

# OPCIÓN B: Hacer configurable
# En experiment_ssot.yaml:
environment:
  circuit_breaker_enabled: false  # Desactivar para training
```

**Archivo**: `config/experiment_ssot.yaml`

```yaml
environment:
  # ... existing config ...
  circuit_breaker:
    enabled: false              # NUEVO: Desactivar para training
    max_consecutive_losses: 999 # Backup si se activa
    cooldown_bars: 0
```

---

### 1.4 FIX: Estandarizar Gamma en Todos los Archivos

**Problema**: `experiment_ssot.yaml` usa 0.99, pero `config.py` y `ppo_trainer.py` usan 0.90.

**Archivos a modificar**:
1. `config/experiment_ssot.yaml` línea 82
2. `src/training/config.py` línea 103
3. `src/training/trainers/ppo_trainer.py` línea 67

```yaml
# EN TODOS LOS ARCHIVOS, usar:
gamma: 0.95  # Compromiso: 20-step horizon (~100 min para 5-min bars)
```

**Justificación**:
- 0.99 = horizon ~100 steps (demasiado largo para trading 5-min)
- 0.90 = horizon ~10 steps (demasiado corto)
- 0.95 = horizon ~20 steps (~1.5 horas, apropiado para day trading)

---

## FASE 2: CORRECCIONES DE ALTA PRIORIDAD (P1)

### 2.1 FIX: Aumentar Batch Size

**Problema**: batch_size=64 con n_steps=2048 = 32 updates/epoch (riesgo de overfitting).

**Archivo**: `config/experiment_ssot.yaml`

```yaml
# ACTUAL (MALO):
training:
  n_steps: 2048
  batch_size: 64    # 32 updates/epoch - OVERFITTING!
  n_epochs: 10

# CORRECCIÓN:
training:
  n_steps: 2048
  batch_size: 256   # 8 updates/epoch - ESTÁNDAR
  n_epochs: 10
```

---

### 2.2 FIX: Aumentar Zona de HOLD

**Problema**: Solo 1% HOLD (debería ser 30-40%). Thresholds actuales ±0.33.

**Archivo**: `config/experiment_ssot.yaml` o `config/trading_config.yaml`

```yaml
# ACTUAL:
thresholds:
  long: 0.33
  short: -0.33

# CORRECCIÓN:
thresholds:
  long: 0.50         # Requiere señal más fuerte para LONG
  short: -0.50       # Requiere señal más fuerte para SHORT
  # HOLD zone: [-0.50, +0.50] = reduce trading ~40%
```

---

### 2.3 FIX: Corregir Bounds de unrealized_pnl

**Problema**: unrealized_pnl clipped a [-0.1, 0.1] pero observation space es [-5, 5].

**Archivo**: `config/experiment_ssot.yaml` (líneas 404-414)

```yaml
# ACTUAL (MALO):
- name: unrealized_pnl
  normalization:
    method: clip
    clip: [-0.1, 0.1]  # Solo usa 2% del rango!

# CORRECCIÓN:
- name: unrealized_pnl
  normalization:
    method: clip
    clip: [-1.0, 1.0]  # Usa 20% del rango, más informativo
```

---

### 2.4 FIX: Rolling Z-Score Leakage en Macro Features

**Problema**: Rolling z-score usa dato actual T en lugar de T-1.

**Archivo**: `airflow/dags/l2_dataset_builder.py` (líneas 692-714)

```python
# ACTUAL (MALO - leakage):
dxy_mean = result['dxy'].rolling(252, min_periods=20).mean()
dxy_std = result['dxy'].rolling(252, min_periods=20).std()
result['dxy_z'] = (result['dxy'] - dxy_mean) / dxy_std.clip(lower=0.01)

# CORRECCIÓN (shift para usar solo datos pasados):
dxy_mean = result['dxy'].shift(1).rolling(252, min_periods=20).mean()
dxy_std = result['dxy'].shift(1).rolling(252, min_periods=20).std()
result['dxy_z'] = (result['dxy'].shift(1) - dxy_mean) / dxy_std.clip(lower=0.01)

# APLICAR MISMA CORRECCIÓN A:
# - vix_z (líneas 704-706)
# - embi_z (líneas 712-714)
```

---

## FASE 3: CORRECCIONES DE PRIORIDAD MEDIA (P2)

### 3.1 FIX: Implementar Learning Rate Schedule

**Archivo**: `config/experiment_ssot.yaml`

```yaml
training:
  learning_rate: 0.0003         # Initial LR (aumentar de 0.0001)
  learning_rate_schedule: linear
  learning_rate_final: 0.00001  # Final LR después de decay
```

**Archivo**: `src/training/trainers/ppo_trainer.py`

```python
from stable_baselines3.common.utils import linear_schedule

def _create_model(self):
    lr_schedule = linear_schedule(
        initial_value=self.config.learning_rate,
        final_value=self.config.learning_rate_final
    )

    return PPO(
        "MlpPolicy",
        self.env,
        learning_rate=lr_schedule,  # Usar schedule
        # ... rest of config
    )
```

---

### 3.2 FIX: Reducir Asymmetric Loss Multiplier

**Problema**: Loss multiplier de 2x amplifica pérdidas excesivamente.

**Archivo**: `src/training/rewards/reward_calculator.py` (línea 484)

```python
# ACTUAL:
loss_multiplier: float = 2.0  # Demasiado agresivo

# CORRECCIÓN:
loss_multiplier: float = 1.2  # Más balanceado
```

---

### 3.3 FIX: Aumentar Sortino Window

**Problema**: sortino_window=20 (100 min) es muy corto para estimación estable.

**Archivo**: `config/experiment_ssot.yaml`

```yaml
# ACTUAL:
reward:
  sortino_window: 20   # Muy corto

# CORRECCIÓN:
reward:
  sortino_window: 240  # 20 horas (más estable)
```

---

### 3.4 FIX: Corregir Win/Loss Trade Tracking

**Problema**: Incrementa winning_trades cada bar, no por trade.

**Archivo**: `src/training/environments/trading_env.py` (líneas 982-986)

```python
# ACTUAL (MALO):
if pnl > 0 and not self._portfolio.position.is_flat:
    self._portfolio.winning_trades += 1
elif pnl < 0 and not self._portfolio.position.is_flat:
    self._portfolio.losing_trades += 1

# CORRECCIÓN (solo contar en position close):
if position_changed and self._previous_position != TradingAction.HOLD:
    # Cerró una posición
    total_trade_pnl = self._accumulated_position_pnl
    if total_trade_pnl > 0:
        self._portfolio.winning_trades += 1
    elif total_trade_pnl < 0:
        self._portfolio.losing_trades += 1
    self._accumulated_position_pnl = 0.0
```

---

## FASE 4: VALIDACIÓN Y TESTING (P3)

### 4.1 Implementar Walk-Forward Validation

**Ya existe código en**: `src/backtesting/walk_forward.py`

**Usar en lugar de single split**:

```python
from src.backtesting.walk_forward import WalkForwardValidator, WalkForwardMethod

validator = WalkForwardValidator(
    train_period_days=90,
    test_period_days=30,
    step_days=30,
    method=WalkForwardMethod.ROLLING,
    min_train_samples=10000
)

# Generar 10-15 ventanas
windows = validator.generate_windows(
    start_date=datetime(2020, 3, 1),
    end_date=datetime(2026, 1, 15)
)

# Entrenar/evaluar en cada ventana
for window in windows:
    model = train_on_window(window.train_start, window.train_end)
    metrics = evaluate_on_window(model, window.test_start, window.test_end)
```

---

### 4.2 Añadir Monitoreo de Distribution Drift

**Crear alerta cuando features driftan >2σ**:

```python
# En L4 backtest o inference
def check_feature_drift(current_features, training_stats):
    alerts = []
    for feature, value in current_features.items():
        mean = training_stats[feature]['mean']
        std = training_stats[feature]['std']
        z_score = (value - mean) / std

        if abs(z_score) > 2.0:
            alerts.append(f"DRIFT: {feature} = {z_score:.2f}σ")

    return alerts
```

---

## CHECKLIST DE IMPLEMENTACIÓN

### Fase 1 (P0) - Críticas [Día 1-2]
- [ ] 1.1 Eliminar duplicación de costos de transacción
- [ ] 1.2 Aumentar peso PnL de 0.5 a 0.8
- [ ] 1.3 Desactivar circuit breaker para training
- [ ] 1.4 Estandarizar gamma a 0.95 en todos los archivos

### Fase 2 (P1) - Alta Prioridad [Día 3-4]
- [ ] 2.1 Aumentar batch_size de 64 a 256
- [ ] 2.2 Aumentar zona HOLD (thresholds ±0.50)
- [ ] 2.3 Corregir bounds unrealized_pnl
- [ ] 2.4 Fix rolling z-score leakage (shift T-1)

### Fase 3 (P2) - Media Prioridad [Día 5-6]
- [ ] 3.1 Implementar learning rate schedule
- [ ] 3.2 Reducir loss multiplier de 2.0 a 1.2
- [ ] 3.3 Aumentar sortino_window de 20 a 240
- [ ] 3.4 Corregir win/loss trade tracking

### Fase 4 (P3) - Validación [Día 7]
- [ ] 4.1 Implementar walk-forward validation
- [ ] 4.2 Añadir monitoreo de distribution drift
- [ ] 4.3 Regenerar datasets con fixes
- [ ] 4.4 Re-entrenar modelo
- [ ] 4.5 Validar en datos OOS

---

## MÉTRICAS DE ÉXITO

| Métrica | Actual | Objetivo |
|---------|--------|----------|
| Episode Return | -8.11% | > 0% |
| Win Rate | 0% | > 45% |
| HOLD Rate | 1% | 30-40% |
| Episode Length | 549 steps | 550+ steps |
| Sharpe Ratio (anualizado) | -13.65 | > 0.5 |
| Consistency (walk-forward) | N/A | >70% windows profitable |

---

## DIAGRAMA DE DEPENDENCIAS

```
FASE 1 (Críticas)
    │
    ├── 1.1 Fix duplicación costos ────┐
    ├── 1.2 Aumentar peso PnL ─────────┼── Afecta: Reward signal
    ├── 1.3 Desactivar circuit breaker ┘
    │
    └── 1.4 Estandarizar gamma ───────── Afecta: Value function

FASE 2 (Alta Prioridad)
    │
    ├── 2.1 Batch size 256 ─────────┐
    ├── 2.2 Zona HOLD ±0.50 ────────┼── Afecta: Policy learning
    │                               │
    ├── 2.3 Bounds unrealized_pnl ──┼── Afecta: Observation space
    │                               │
    └── 2.4 Fix z-score leakage ────┘── Afecta: Dataset integrity

FASE 3 (Media Prioridad)
    │
    └── Refinamientos de hiperparámetros

FASE 4 (Validación)
    │
    └── Walk-forward + Drift monitoring
```

---

## COMANDOS DE EJECUCIÓN

```bash
# 1. Aplicar fixes al código (manual)

# 2. Regenerar datasets con fixes
docker exec usdcop-airflow-scheduler python -c "
from airflow.dags.l2_dataset_builder import build_dataset
build_dataset()
"

# 3. Re-entrenar modelo
docker exec usdcop-airflow-scheduler python /opt/airflow/run_training_fixed.py

# 4. Validar en OOS
docker exec usdcop-airflow-scheduler python /opt/airflow/run_backtest_v2.py
```

---

**NOTA FINAL**: Este plan debe ejecutarse en orden. Los fixes de Fase 1 son prerrequisitos para que las mejoras de Fase 2-3 tengan efecto. Sin corregir la duplicación de costos y el peso del PnL, cualquier otro cambio tendrá impacto limitado.
