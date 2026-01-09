# DOCUMENTO MAESTRO COMPLETO
## USDCOP Trading System - Auditoría Integral + Plan de Acción
## Fecha: 2026-01-08
## Versión: 1.0

---

# ÍNDICE

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Diagnóstico del Sistema](#2-diagnóstico-del-sistema)
3. [Issues Críticos Identificados](#3-issues-críticos-identificados)
4. [Análisis del Panel de 10 Expertos](#4-análisis-del-panel-de-10-expertos)
5. [Plan de Acción - 3 Semanas](#5-plan-de-acción---3-semanas)
6. [Código de Fixes Completo](#6-código-de-fixes-completo)
7. [Checklist Ejecutable](#7-checklist-ejecutable)
8. [Criterios de Éxito](#8-criterios-de-éxito)
9. [Riesgos y Mitigaciones](#9-riesgos-y-mitigaciones)
10. [Comandos de Referencia](#10-comandos-de-referencia)

---

# 1. RESUMEN EJECUTIVO

## 1.1 Situación Actual

| Métrica | Valor | Estado |
|---------|-------|--------|
| **Equity** | $9,646.26 | -3.54% desde $10,000 |
| **Win Rate** | 22.8% | 13 wins / 44 losses |
| **Trades Totales** | 57 | ~6 trades/día |
| **Señales HOLD** | 0% | **CRÍTICO** |
| **Rango de Acciones** | -0.80 a +0.82 | Extremas |

## 1.2 Problema Principal

> **El modelo NUNCA hace HOLD. Siempre está en posición.**
>
> Esto no es un bug de producción - es cómo el modelo fue entrenado.
> El reward function de V19 solo medía P&L sin penalizar overtrading.

## 1.3 Decisión Estratégica

```
CAMINO B (PRAGMÁTICO) - SELECCIONADO
├── Fix bugs técnicos (Semana 1)
├── Reentrenar V20 en paralelo (Semana 1-2)
├── A/B Test V19 vs V20 (Semana 3)
└── Deploy ganador (Fin Semana 3)

Tiempo total: 3 semanas
```

## 1.4 Issues Identificados

| Severidad | Cantidad | Status |
|-----------|----------|--------|
| P0 Críticos | 6 | Pendientes |
| P1 Altos | 7 | Pendientes |
| P2 Medios | 4 | Pendientes |
| **Total** | **17** | **0% completado** |

---

# 2. DIAGNÓSTICO DEL SISTEMA

## 2.1 Conteo de Tablas Críticas

```sql
tabla                    | rows
-------------------------|--------
equity_snapshots         | 62
trades_history           | 57
trading_state            | 1
dw.fact_rl_inference     | 50
config.models            | 4
usdcop_m5_ohlcv          | 90,917
macro_indicators_daily   | 10,743
```

## 2.2 Distribución de Señales (últimos 7 días)

```
action_discretized | count | pct
-------------------|-------|------
SHORT              | 30    | 60.00%
LONG               | 20    | 40.00%
HOLD               | 0     | 0.00%   ← PROBLEMA CRÍTICO
```

## 2.3 Rango de Raw Actions

```
Métrica    | Valor
-----------|--------
min_action | -0.8073
max_action | 0.8234
avg_action | -0.0412
std_action | 0.5891
p10        | -0.7123
p90        | 0.6789
```

**Análisis**: Acciones muy extremas. El modelo tiene alta "confianza" pero esto indica sobreajuste o reward mal diseñado.

## 2.4 Estado de Trading Actual

```
model_id     | ppo_v1
position     | 0 (FLAT)
equity       | 9646.26
realized_pnl | -353.74
trade_count  | 57
wins         | 13 (22.8%)
losses       | 44 (77.2%)
```

## 2.5 Macro Data - NULLs Detectados

```sql
fecha      | dxy    | vix
-----------|--------|-------
2026-01-08 | NULL   | NULL   ← PROBLEMA
2026-01-07 | NULL   | NULL   ← PROBLEMA
2026-01-06 | 104.23 | 15.67
```

**Impacto**: El modelo está recibiendo observaciones incompletas. Los features `dxy_z` y `vix_z` usan fallback values.

---

# 3. ISSUES CRÍTICOS IDENTIFICADOS

## 3.1 Tabla Completa de Issues

| # | Issue | Severidad | Impacto | Documento |
|---|-------|-----------|---------|-----------|
| 1 | Threshold 0.30 vs 0.10 | P0 | Modelo opera diferente a training | 03, 06 |
| 2 | Look-Ahead Bias | P0 | Backtest inflado ~5-10% | 07 |
| 3 | Reward V20 Math Bug | P0 | Training no converge | 07 |
| 4 | StateTracker no persiste | P0 | Pérdida de estado en restart | 06, 07 |
| 5 | 0% HOLD signals | P0 | Overtrading destructivo | 05, 06 |
| 6 | Acciones extremas | P0 | Modelo sobreconfiado | 06 |
| 7 | Macro NULLs | P1 | Observaciones corruptas | 02, 07 |
| 8 | Timezone inconsistency | P1 | Features temporales incorrectos | 07 |
| 9 | US Holidays no excluidos | P1 | Tradea en baja liquidez | 07 |
| 10 | Dataset V20 no especificado | P1 | Training puede fallar | 07 |
| 11 | Entropy coefficient = 0 | P1 | Acciones extremas | 07 |
| 12 | Drift monitor legacy | P1 | No detecta drift real | 06 |
| 13 | Observation logging falta | P1 | Difícil debugging | 07 |
| 14 | Benchmarks no definidos | P2 | No hay comparación | 07 |
| 15 | Early stopping no impl. | P2 | Riesgo de overfit | 07 |
| 16 | Slippage subestimado | P2 | P&L optimista | 06 |
| 17 | Win rate 22.8% | P2 | Modelo no rentable | 02 |

## 3.2 El Issue Más Crítico: Look-Ahead Bias

```python
# CÓDIGO ACTUAL (INCORRECTO):
def execute_inference(current_bar):
    observation = build_observation(current_bar)
    action = model.predict(observation)
    execution_price = current_bar['close']  # ← LOOK-AHEAD BIAS
    execute_trade(action, execution_price)

# PROBLEMA:
# Cuando ves el close de una barra, esa barra YA TERMINÓ.
# En realidad no puedes ejecutar en un precio del pasado.
# Esto infla artificialmente los resultados del backtest.
```

## 3.3 El Issue del Reward Function

```python
# CÓDIGO V19 (PROBLEMÁTICO):
def reward_v19(pnl):
    return pnl  # Solo P&L, sin costos ni penalizaciones

# PROBLEMAS:
# 1. No hay costo por tradear → Overtrading
# 2. No hay reward por esperar → 0% HOLD
# 3. No penaliza pérdidas más que ganancias → Toma riesgos excesivos
```

---

# 4. ANÁLISIS DEL PANEL DE 10 EXPERTOS

## 4.1 Panel Convocado

| # | Experto | Especialidad | Veredicto |
|---|---------|--------------|-----------|
| 1 | Dr. Chen | Quant Finance PhD | "22.8% win rate necesita 3.5:1 R:R - insostenible" |
| 2 | Dr. Petrova | ML/RL PhD | "0% HOLD = reward mal diseñado" |
| 3 | Ing. Martinez | Software Architect | "StateTracker no persiste - métricas no confiables" |
| 4 | Rodriguez | Trading Ops | "57 trades no es estadísticamente significativo" |
| 5 | Dr. Kumar | Data Engineer | "Macro NULLs corrompen observaciones" |
| 6 | Schwartz | Risk Manager | "Sin límites de drawdown - riesgo excesivo" |
| 7 | Thompson | MLOps Engineer | "Pipeline de training incompleto" |
| 8 | Dr. Nakamura | Financial Engineer | "Costos de transacción subestimados 100x" |
| 9 | Dr. Williams | Behavioral Finance | "Overtrading consume 3%+ diario" |
| 10 | Garcia | Project Manager | "3 semanas es timeline realista" |

## 4.2 Veredicto Unánime

> **PAUSAR → FIX → VALIDAR → DECIDIR**
>
> No se puede confiar en las métricas actuales hasta corregir los bugs.
> El 22.8% win rate y 0% HOLD pueden ser resultado de:
> 1. Modelo mal entrenado (probable)
> 2. Bugs corrompiendo datos (posible)
> 3. Ambos (más probable)

## 4.3 Recomendaciones por Experto

### Dr. Chen (Quant Finance)
- Reducir frecuencia a 2-4 trades/día
- Target win rate: 40%+
- Implementar risk:reward mínimo de 2:1

### Dr. Petrova (ML/RL)
- Nuevo reward function con 5 componentes
- Entropy coefficient = 0.01 mínimo
- Action threshold = 0.15 (más amplio)

### Ing. Martinez (Software)
- StateTracker persistence es BLOQUEANTE
- Fix look-ahead bias antes de validar
- Logging de observations obligatorio

### Schwartz (Risk)
- Max daily loss: -2%
- Max drawdown: -10%
- Max consecutive losses: 8 → pause

---

# 5. PLAN DE ACCIÓN - 3 SEMANAS

## 5.1 Timeline Visual

```
SEMANA 1: ESTABILIZACIÓN + DISEÑO
══════════════════════════════════════════════════════════════════
│ Día │ Track A (Fixes)           │ Track B (V20 Design)         │
├─────┼───────────────────────────┼──────────────────────────────┤
│  1  │ Threshold + Look-ahead    │ Reward function V20          │
│  2  │ StateTracker persistence  │ Environment V20              │
│  3  │ Macro scraper fix         │ Dataset preparation          │
│  4  │ Timezone normalization    │ Training setup               │
│  5  │ Validación histórica      │ Hyperparameter config        │
══════════════════════════════════════════════════════════════════

SEMANA 2: TRAINING + VALIDACIÓN
══════════════════════════════════════════════════════════════════
│ Día │ Track A (V19 Monitor)     │ Track B (V20 Training)       │
├─────┼───────────────────────────┼──────────────────────────────┤
│  6  │ Paper trading limpio      │ PPO training (1M steps)      │
│  7  │ Colectar métricas         │ PPO training (2M steps)      │
│  8  │ Análisis distribución     │ PPO training (3M steps)      │
│  9  │ Comparar resultados       │ PPO training (4M steps)      │
│ 10  │ Documentar hallazgos      │ PPO training (5M steps)      │
│ 11  │ Preparar A/B test         │ Export ONNX + Validation     │
│ 12  │ Setup A/B environment     │ Backtest OOS                 │
══════════════════════════════════════════════════════════════════

SEMANA 3: A/B TESTING + DECISIÓN
══════════════════════════════════════════════════════════════════
│ Día │ Actividad                                                │
├─────┼──────────────────────────────────────────────────────────┤
│ 13  │ Deploy V20 en paper trading paralelo                     │
│ 14  │ A/B: V19 corregido vs V20 (día 1)                        │
│ 15  │ A/B: V19 corregido vs V20 (día 2)                        │
│ 16  │ A/B: V19 corregido vs V20 (día 3)                        │
│ 17  │ A/B: V19 corregido vs V20 (día 4)                        │
│ 18  │ A/B: V19 corregido vs V20 (día 5)                        │
│ 19  │ Análisis de resultados A/B                               │
│ 20  │ DECISIÓN: Deploy ganador                                 │
│ 21  │ Documentación + Retrospectiva                            │
══════════════════════════════════════════════════════════════════
```

## 5.2 Matriz de Decisión A/B

```
                        V20 Win Rate
                    <30%    30-40%   >40%
                  ┌───────┬────────┬────────┐
           <30%  │ STOP  │ V20    │ V20    │
V19        ─────┼───────┼────────┼────────┤
Win       30-40%│ V19   │ TEST+  │ V20    │
Rate       ─────┼───────┼────────┼────────┤
           >40% │ V19   │ V19    │ TEST+  │
                  └───────┴────────┴────────┘

STOP   = Pausar todo, revisar fundamentalmente
V19    = Continuar con V19 corregido
V20    = Deploy V20
TEST+  = Extender testing 2 semanas más
```

---

# 6. CÓDIGO DE FIXES COMPLETO

## 6.1 Fix #1: Threshold (SQL)

```sql
-- Ejecutar en PostgreSQL
UPDATE config.models
SET
    threshold_long = 0.10,
    threshold_short = -0.10,
    updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');

-- Verificar
SELECT model_id, threshold_long, threshold_short FROM config.models;
```

## 6.2 Fix #2: Look-Ahead Bias

```python
# airflow/dags/l5_multi_model_inference.py

class InferenceEngine:
    def __init__(self):
        self.pending_signal = None

    def on_bar_close(self, current_bar):
        """Called when a bar closes - generate signal for NEXT bar."""
        observation = self.observation_builder.build(current_bar)
        action = self.model.predict(observation)

        # Guardar señal para ejecutar en la siguiente barra
        self.pending_signal = self.discretize_action(action)

        logger.info(f"Signal generated: {self.pending_signal} at bar close {current_bar['time']}")

    def on_bar_open(self, new_bar):
        """Called when a new bar opens - execute pending signal."""
        if self.pending_signal is not None:
            # CORRECTO: Ejecutar al OPEN de la nueva barra
            execution_price = new_bar['open']

            logger.info(f"Executing {self.pending_signal} at {execution_price}")

            self.paper_trader.execute(self.pending_signal, execution_price)
            self.pending_signal = None
```

## 6.3 Fix #3: StateTracker Persistence

```python
# src/core/state/state_tracker.py

import redis
import json
import psycopg2
import os
from datetime import datetime

class StateTracker:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.redis = redis.from_url(
            os.getenv('REDIS_URL', 'redis://redis:6379'),
            decode_responses=True
        )
        self.state_key = f"trading:state:{model_id}"
        self.db_url = os.getenv('DATABASE_URL')

        # Load state on init
        self._load_state()

    def _load_state(self):
        """Load state from Redis on startup."""
        stored = self.redis.get(self.state_key)
        if stored:
            data = json.loads(stored)
            self.position = data.get('position', 0)
            self.entry_price = data.get('entry_price', 0.0)
            self.equity = data.get('equity', 10000.0)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            self.trade_count = data.get('trade_count', 0)
            self.wins = data.get('wins', 0)
            self.losses = data.get('losses', 0)
            logger.info(f"Loaded state from Redis: equity={self.equity}, trades={self.trade_count}")
        else:
            self._reset_to_defaults()
            logger.info("No stored state found, using defaults")

    def _persist_state(self):
        """Persist state to Redis + PostgreSQL."""
        state_dict = {
            'position': self.position,
            'entry_price': self.entry_price,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'trade_count': self.trade_count,
            'wins': self.wins,
            'losses': self.losses,
            'last_updated': datetime.utcnow().isoformat()
        }

        # Redis (primary - fast)
        self.redis.set(self.state_key, json.dumps(state_dict))

        # PostgreSQL (backup - durable)
        self._persist_to_postgres(state_dict)

    def _persist_to_postgres(self, state_dict):
        """Backup state to PostgreSQL."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute('''
                        INSERT INTO trading_state
                        (model_id, position, entry_price, equity, realized_pnl,
                         trade_count, wins, losses, last_updated)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (model_id) DO UPDATE SET
                            position = EXCLUDED.position,
                            entry_price = EXCLUDED.entry_price,
                            equity = EXCLUDED.equity,
                            realized_pnl = EXCLUDED.realized_pnl,
                            trade_count = EXCLUDED.trade_count,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            last_updated = NOW()
                    ''', (
                        self.model_id,
                        state_dict['position'],
                        state_dict['entry_price'],
                        state_dict['equity'],
                        state_dict['realized_pnl'],
                        state_dict['trade_count'],
                        state_dict['wins'],
                        state_dict['losses']
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist to PostgreSQL: {e}")

    def update_position(self, new_position, entry_price=None):
        """Update position and persist."""
        self.position = new_position
        if entry_price:
            self.entry_price = entry_price
        self._persist_state()

    def record_trade(self, pnl):
        """Record completed trade and persist."""
        self.realized_pnl += pnl
        self.equity += pnl
        self.trade_count += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self._persist_state()

        logger.info(f"Trade recorded: pnl={pnl:.2f}, equity={self.equity:.2f}, "
                   f"wins={self.wins}, losses={self.losses}")
```

## 6.4 Fix #4: Reward Calculator V20 (Corregido)

```python
# src/training/reward_calculator_v20.py

class RewardCalculatorV20:
    """
    Corrected reward calculator for V20.

    Order of operations (CRÍTICO):
    1. Base PnL
    2. Asymmetric penalty on PnL (NOT on costs)
    3. Transaction costs (additive, not multiplied)
    4. Hold bonus
    5. Consistency bonus
    6. Drawdown penalty
    """

    def __init__(self):
        self.transaction_cost = 0.002    # 0.2% per trade
        self.hold_bonus = 0.0001         # Per bar in hold
        self.loss_multiplier = 2.0       # Losses hurt 2x
        self.consistency_bonus = 0.0005  # Per consecutive win
        self.drawdown_threshold = 0.05   # 5% DD threshold
        self.drawdown_penalty = 0.001    # Penalty per % DD

    def calculate(
        self,
        pnl: float,
        action: int,
        prev_action: int,
        position_time: int,
        consecutive_wins: int,
        equity_peak: float,
        equity_current: float
    ) -> float:
        """Calculate reward with correct order of operations."""

        # 1. BASE REWARD = PnL
        base_reward = pnl

        # 2. ASYMMETRIC PENALTY (on PnL only, not costs)
        if pnl < 0:
            base_reward = pnl * self.loss_multiplier

        # 3. TRANSACTION COST (additive, separate)
        transaction_penalty = 0.0
        if action != prev_action:
            transaction_penalty = -self.transaction_cost

        # 4. HOLD BONUS (for patience)
        hold_bonus = 0.0
        if action == 0:  # HOLD
            hold_bonus = self.hold_bonus * min(position_time, 10)

        # 5. CONSISTENCY BONUS (for winning streaks)
        consistency_bonus = 0.0
        if pnl > 0:
            consistency_bonus = self.consistency_bonus * min(consecutive_wins, 5)

        # 6. DRAWDOWN PENALTY (for large drawdowns)
        drawdown_penalty = 0.0
        if equity_peak > 0:
            current_dd = (equity_peak - equity_current) / equity_peak
            if current_dd > self.drawdown_threshold:
                drawdown_penalty = -self.drawdown_penalty * (current_dd - self.drawdown_threshold) * 100

        # TOTAL REWARD
        total_reward = (
            base_reward
            + transaction_penalty
            + hold_bonus
            + consistency_bonus
            + drawdown_penalty
        )

        return total_reward


# Unit Tests
def test_reward_calculator():
    calc = RewardCalculatorV20()

    # Test 1: Winning trade, no position change
    reward = calc.calculate(
        pnl=0.01, action=1, prev_action=1,
        position_time=5, consecutive_wins=3,
        equity_peak=10000, equity_current=10100
    )
    expected = 0.01 + 0 + 0 + 0.0015 + 0
    assert abs(reward - expected) < 1e-6, f"Test 1 failed: {reward} != {expected}"

    # Test 2: Losing trade with position change
    reward = calc.calculate(
        pnl=-0.01, action=-1, prev_action=1,
        position_time=0, consecutive_wins=0,
        equity_peak=10000, equity_current=9900
    )
    expected = -0.01 * 2.0 - 0.002 + 0 + 0 + 0
    assert abs(reward - expected) < 1e-6, f"Test 2 failed: {reward} != {expected}"

    # Test 3: HOLD action
    reward = calc.calculate(
        pnl=0, action=0, prev_action=0,
        position_time=5, consecutive_wins=0,
        equity_peak=10000, equity_current=10000
    )
    expected = 0 + 0 + 0.0005 + 0 + 0
    assert abs(reward - expected) < 1e-6, f"Test 3 failed: {reward} != {expected}"

    print("All reward calculator tests passed!")


if __name__ == "__main__":
    test_reward_calculator()
```

## 6.5 Fix #5: Trading Calendar con US Holidays

```python
# airflow/dags/utils/trading_calendar.py

COLOMBIA_HOLIDAYS = [
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
    '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
    '2025-12-08', '2025-12-25',
    '2026-01-01', '2026-01-12', '2026-03-23', '2026-04-02', '2026-04-03',
    '2026-05-01', '2026-05-18', '2026-06-08', '2026-06-15', '2026-06-29',
    '2026-07-20', '2026-08-07', '2026-08-17', '2026-10-12', '2026-11-02',
    '2026-11-16', '2026-12-08', '2026-12-25',
]

US_HOLIDAYS = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', '2025-06-19',
    '2025-07-04', '2025-09-01', '2025-10-13', '2025-11-11', '2025-11-27',
    '2025-12-25',
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-05-25', '2026-06-19',
    '2026-07-03', '2026-09-07', '2026-10-12', '2026-11-11', '2026-11-26',
    '2026-12-25',
]

class TradingCalendarV2:
    """Trading calendar with US holidays for USD/COP."""

    def __init__(self):
        self.colombia_holidays = set(COLOMBIA_HOLIDAYS)
        self.us_holidays = set(US_HOLIDAYS)
        self.all_holidays = self.colombia_holidays | self.us_holidays

    def is_trading_day(self, date) -> bool:
        """Check if date is a valid trading day."""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)

        # Weekend
        if hasattr(date, 'weekday') and date.weekday() >= 5:
            return False

        # Holiday (Colombia or US)
        if date_str in self.all_holidays:
            return False

        return True

    def is_reduced_liquidity(self, date) -> bool:
        """Check if day has reduced liquidity (US holiday without COL holiday)."""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        return date_str in self.us_holidays and date_str not in self.colombia_holidays
```

## 6.6 Fix #6: Macro Scraper con Fallback

```python
# airflow/dags/utils/macro_scraper_robust.py

import time
import logging
import requests
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RobustMacroScraper:
    """Macro data scraper with retry logic and fallbacks."""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 60
        self.fred_api_key = os.getenv('FRED_API_KEY')

    def fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Execute fetch function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed")
        return None

    def fetch_dxy(self, date: datetime) -> Optional[float]:
        """Fetch DXY with fallback to FRED API."""
        # Primary source
        dxy = self.fetch_with_retry(self._fetch_dxy_primary, date)
        if dxy is not None:
            return dxy

        # Fallback: FRED API
        logger.info("Using FRED API fallback for DXY")
        return self._fetch_dxy_fred(date)

    def _fetch_dxy_fred(self, date: datetime) -> Optional[float]:
        """Fallback: FRED API for DXY."""
        if not self.fred_api_key:
            logger.warning("FRED API key not configured")
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "DTWEXBGS",
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": date.strftime("%Y-%m-%d"),
                "observation_end": date.strftime("%Y-%m-%d"),
            }
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get("observations"):
                value = data["observations"][0].get("value")
                if value and value != ".":
                    return float(value)
        except Exception as e:
            logger.error(f"FRED API error: {e}")

        return None

    def fill_missing_macro(self, conn, days_back: int = 7):
        """Fill missing macro data for recent days."""
        query = """
        SELECT fecha FROM macro_indicators_daily
        WHERE fecha > CURRENT_DATE - %s
          AND (fxrt_index_dxy_usa_d_dxy IS NULL OR volt_vix_usa_d_vix IS NULL)
        ORDER BY fecha
        """

        with conn.cursor() as cur:
            cur.execute(query, (days_back,))
            missing_dates = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(missing_dates)} dates with missing macro data")

        for date in missing_dates:
            dxy = self.fetch_dxy(date)
            vix = self.fetch_vix(date)

            if dxy is not None or vix is not None:
                update_query = """
                UPDATE macro_indicators_daily SET
                    fxrt_index_dxy_usa_d_dxy = COALESCE(%s, fxrt_index_dxy_usa_d_dxy),
                    volt_vix_usa_d_vix = COALESCE(%s, volt_vix_usa_d_vix),
                    updated_at = NOW()
                WHERE fecha = %s
                """
                with conn.cursor() as cur:
                    cur.execute(update_query, (dxy, vix, date))
                    conn.commit()

                logger.info(f"Updated macro for {date}: DXY={dxy}, VIX={vix}")
```

## 6.7 Fix #7: PPO Config V20

```python
# config/ppo_config_v20.py

PPO_CONFIG_V20 = {
    # Learning
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,

    # GAE
    "gamma": 0.99,
    "gae_lambda": 0.95,

    # Policy
    "clip_range": 0.2,
    "normalize_advantage": True,

    # CRÍTICO: Entropy para exploración
    "ent_coef": 0.01,  # Previene acciones extremas

    # Value function
    "vf_coef": 0.5,

    # Gradient clipping
    "max_grad_norm": 0.5,

    # KL divergence target
    "target_kl": 0.015,
}

POLICY_KWARGS_V20 = {
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256],
    },
}
```

## 6.8 Fix #8: Dataset Config V20

```python
# config/dataset_config_v20.py

DATASET_CONFIG_V20 = {
    "source_table": "usdcop_m5_ohlcv",

    "date_ranges": {
        "train": {
            "start": "2020-01-01",
            "end": "2024-12-31",
        },
        "validation": {
            "start": "2025-01-01",
            "end": "2025-06-30",
        },
        "test": {
            "start": "2025-07-01",
            "end": "2026-01-08",
        },
    },

    "features": [
        "log_ret_5m", "log_ret_1h",
        "rsi_9", "macd_hist", "bb_width",
        "vol_ratio", "atr_pct",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "dxy_z", "vix_z",
        "position", "time_normalized"
    ],

    "normalization": {
        "method": "z_score",
        "clip_range": (-5.0, 5.0),
        "use_train_stats": True,
    },

    "filters": {
        "market_hours_only": True,
        "market_open": "08:00",
        "market_close": "12:55",
        "exclude_holidays": True,
    },
}
```

## 6.9 Fix #9: Benchmarks

```python
# src/evaluation/benchmarks.py

import numpy as np
import pandas as pd

class BenchmarkStrategies:
    """Benchmark strategies for comparison."""

    @staticmethod
    def buy_and_hold(prices):
        """Simple buy and hold strategy."""
        returns = np.diff(prices) / prices[:-1]
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        return np.array(equity)

    @staticmethod
    def ma_crossover(prices, fast=20, slow=50):
        """Moving average crossover strategy."""
        df = pd.DataFrame({'close': prices})
        df['ma_fast'] = df['close'].rolling(fast).mean()
        df['ma_slow'] = df['close'].rolling(slow).mean()

        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1

        equity = [1.0]
        position = 0

        for i in range(1, len(df)):
            if position != 0:
                pnl = position * (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                equity.append(equity[-1] * (1 + pnl - 0.002))
            else:
                equity.append(equity[-1])

            if df['signal'].iloc[i] != position:
                position = df['signal'].iloc[i]

        return np.array(equity)


def compare_with_benchmarks(model_equity, prices, model_name="Model"):
    """Compare model performance with benchmarks."""

    benchmarks = {
        "Buy & Hold": BenchmarkStrategies.buy_and_hold(prices),
        "MA Crossover": BenchmarkStrategies.ma_crossover(prices),
        model_name: model_equity,
    }

    results = {}
    for name, equity in benchmarks.items():
        returns = np.diff(equity) / equity[:-1]
        results[name] = {
            "total_return": (equity[-1] / equity[0] - 1) * 100,
            "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 60),
            "max_drawdown": calculate_max_drawdown(equity) * 100,
        }

    return results
```

---

# 7. CHECKLIST EJECUTABLE

## 7.1 DÍA 1 - HOY (P0 CRÍTICOS)

### [ ] 1.1 Fix Threshold (SQL)
```bash
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
UPDATE config.models SET threshold_long = 0.10, threshold_short = -0.10, updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');
"
```

### [ ] 1.2 Fix Look-Ahead Bias
- Archivo: `airflow/dags/l5_multi_model_inference.py`
- Aplicar código de sección 6.2

### [ ] 1.3 Fix Reward V20 Math Bug
- Crear: `src/training/reward_calculator_v20.py`
- Aplicar código de sección 6.4
- Ejecutar tests

### [ ] 1.4 Timezone Audit
```bash
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT CASE WHEN DATE(time) < '2025-12-17' THEN 'OLD' ELSE 'NEW' END as period,
       EXTRACT(HOUR FROM time) as hour, COUNT(*) as bars
FROM usdcop_m5_ohlcv WHERE time > '2025-11-01'
GROUP BY 1, 2 ORDER BY 1, 2;
"
```

### [ ] 1.5 Macro Data Audit
```bash
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT fecha, fxrt_index_dxy_usa_d_dxy as dxy, volt_vix_usa_d_vix as vix
FROM macro_indicators_daily WHERE fecha > CURRENT_DATE - 7 ORDER BY fecha DESC;
"
```

### [ ] 1.6 StateTracker Audit
```bash
docker restart usdcop-airflow-worker && sleep 30
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT model_id, equity, trade_count, last_updated FROM trading_state;
"
```

## 7.2 DÍA 2 (P1 ALTOS)

### [ ] 2.1 Implementar StateTracker Persistence
- Aplicar código de sección 6.3

### [ ] 2.2 Fix US Holidays
- Aplicar código de sección 6.5

### [ ] 2.3 Macro Scraper con Fallback
- Crear: `airflow/dags/utils/macro_scraper_robust.py`
- Aplicar código de sección 6.6

### [ ] 2.4 Normalizar Timezones (si necesario)

### [ ] 2.5 Agregar Observation Logging

### [ ] 2.6 Dataset V20 Specification
- Crear: `config/dataset_config_v20.py`
- Aplicar código de sección 6.8

### [ ] 2.7 PPO Config con Entropy
- Crear: `config/ppo_config_v20.py`
- Aplicar código de sección 6.7

## 7.3 DÍA 3-5 (PREPARACIÓN V20)

### [ ] 3.1 Generar Dataset V20
### [ ] 3.2 Crear Environment V20
### [ ] 3.3 Implementar Benchmarks
### [ ] 3.4 Implementar Early Stopping

## 7.4 SEMANA 2 (TRAINING V20)

### [ ] 4.1 Iniciar Training V20
### [ ] 4.2 Validar Training Progress
### [ ] 4.3 Export ONNX
### [ ] 4.4 Backtest OOS

## 7.5 SEMANA 3 (A/B TESTING)

### [ ] 5.1 Deploy V20 en Paper Trading
### [ ] 5.2 A/B Test (5 días)
### [ ] 5.3 Decisión Final

---

# 8. CRITERIOS DE ÉXITO

## 8.1 Criterios por Nivel

| Nivel | Win Rate | HOLD % | Sharpe | MaxDD |
|-------|----------|--------|--------|-------|
| **Mínimo Viable** | ≥30% | ≥15% | ≥0.5 | ≤20% |
| **Target** | ≥40% | ≥25% | ≥1.0 | ≤15% |
| **Excelente** | ≥50% | ≥30% | ≥1.5 | ≤10% |

## 8.2 Criterios para Decisión Final

```python
if v20_win_rate > v19_win_rate + 5%:
    decision = "DEPLOY V20"
elif v19_win_rate >= 35%:
    decision = "KEEP V19 CORREGIDO"
else:
    decision = "PAUSE + REDESIGN"
```

---

# 9. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| V20 no mejora | Media | Alto | Tener V21 con más cambios listo |
| Bugs en nuevo código | Alta | Medio | Testing exhaustivo, rollback plan |
| Datos macro siguen NULL | Media | Medio | Implementar scraper alternativo |
| Overfit en training | Media | Alto | Usar validation split, early stopping |
| Timeline se extiende | Alta | Bajo | Buffer de 1 semana extra |

---

# 10. COMANDOS DE REFERENCIA

```bash
# Status de contenedores
docker ps --format "table {{.Names}}\t{{.Status}}"

# Logs de Airflow
docker logs usdcop-airflow-worker --tail 100

# Query rápida
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT * FROM trading_state"

# Reiniciar servicio
docker restart usdcop-airflow-worker

# Trigger DAG manual
docker exec usdcop-airflow-webserver airflow dags trigger l5_multi_model_inference

# Ver métricas de trading
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT model_id, equity, realized_pnl, trade_count, wins, losses,
       ROUND(100.0 * wins / NULLIF(trade_count, 0), 1) as win_rate_pct
FROM trading_state;
"

# Ver distribución de señales
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT action_discretized, COUNT(*), ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
FROM dw.fact_rl_inference WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY 1 ORDER BY 2 DESC;
"
```

---

# APÉNDICE: ARCHIVOS GENERADOS

```
diagnostica/
├── 01_queries_diagnostico.sql      # Queries SQL de diagnóstico
├── 02_DIAGNOSTIC_REPORT.md         # Hallazgos iniciales
├── 03_P0_FIXES.sql                 # SQL fixes originales
├── 04_FIX_CHECKLIST.md             # Checklist inicial
├── 05_EXPERT_PANEL_DECISION.md     # Panel 4 expertos
├── 06_PLAN_MAESTRO_10_EXPERTOS.md  # Plan 10 expertos
├── 07_ADDENDUM_FIXES_CRITICOS.md   # 9 fixes faltantes
├── 08_CHECKLIST_EJECUTABLE_COMPLETO.md
└── 09_Documento Maestro Completo.md ← ESTE ARCHIVO
```

---

*Documento Maestro Completo generado por auditoría integral*
*Claude Code - 2026-01-08*
*Versión 1.0 - Consolidación de documentos 01-08*
*Total items: 27 | Tiempo estimado: 3 semanas*
