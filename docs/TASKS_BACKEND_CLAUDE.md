# Tareas Backend - Claude (Python)

**Proyecto:** USDCOP RL Trading System V19
**Asignado a:** Claude
**Referencia Contratos:** `docs/API_CONTRACTS_SHARED.md`

---

## Contexto del Sistema

### Arquitectura de Datos
```
L0 (Ingestion) -> L1 (Features) -> L5 (Inference) -> API -> Frontend
```

### Tablas PostgreSQL Principales
| Tabla | Descripcion | Filas Actuales |
|-------|-------------|----------------|
| `usdcop_m5_ohlcv` | OHLCV 5-min (hypertable) | ~87,500 |
| `macro_indicators_daily` | 37 indicadores macro | ~10,700 |
| `inference_features_5m` | 13 core features calculadas | ~50 |
| `trading_metrics` | Metricas de trading | 0 (VACIO) |

### Configuracion V19
- **Observation Space:** 15 dimensiones (13 core + 2 state)
- **Core Features:** log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14, dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d
- **State Features:** position (-1/0/1), time_normalized (0-1)
- **Config Files:** `config/feature_config_v19.json`, `config/v19_norm_stats.json`

---

## Fase 1: Paquete Core (src/) - SEMANA 1

### BE-01: Crear ObservationBuilder V19 (15-dim)
**Archivo:** `src/core/builders/observation_builder.py`
**Complejidad:** Media
**Dependencias:** feature_config_v19.json

**Requisitos:**
```python
class ObservationBuilderV19:
    """
    Construye vector de observacion 15-dim para modelos RL.

    DEBE ser identico al usado en training para garantizar paridad.
    """

    FEATURE_ORDER = [
        # 13 core features (indices 0-12)
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        # 2 state features (indices 13-14)
        "position", "time_normalized"
    ]

    def __init__(self, config_path: str = "config/feature_config_v19.json"):
        self.config = self._load_config(config_path)
        self.norm_stats = self._load_norm_stats()

    def build(self, market_features: Dict[str, float],
              position: float,
              time_normalized: float) -> np.ndarray:
        """
        Construir observation vector.

        Args:
            market_features: Dict con 13 core features
            position: -1 (short), 0 (flat), 1 (long)
            time_normalized: 0.0 a 1.0 (progreso de sesion)

        Returns:
            np.ndarray shape (15,) con valores normalizados y clipped [-5, 5]
        """
        pass

    def normalize(self, feature_name: str, value: float) -> float:
        """Aplicar z-score normalization usando norm_stats"""
        pass
```

**Criterios de Aceptacion:**
- [ ] Carga feature_config_v19.json correctamente
- [ ] Carga v19_norm_stats.json correctamente
- [ ] Output shape es exactamente (15,)
- [ ] Features en orden correcto segun FEATURE_ORDER
- [ ] Valores clipped a [-5, 5]
- [ ] Maneja NaN/None con valor 0.0

---

### BE-02: Migrar norm_stats loader
**Archivo:** `src/core/normalizers/zscore_normalizer.py`
**Complejidad:** Baja
**Dependencias:** v19_norm_stats.json

**Requisitos:**
```python
class ZScoreNormalizer:
    """Normalizador z-score con stats pre-calculados de training"""

    def __init__(self, stats_path: str = "config/v19_norm_stats.json"):
        self.stats = self._load_stats(stats_path)

    def normalize(self, feature_name: str, value: float) -> float:
        """
        Normalizar valor usando z-score.

        Formula: (value - mean) / std

        Si feature no tiene stats, retornar valor sin cambio.
        """
        pass

    def denormalize(self, feature_name: str, z_value: float) -> float:
        """Revertir normalizacion (para debugging)"""
        pass
```

**Archivo norm_stats esperado:**
```json
{
  "log_ret_5m": {"mean": 0.0, "std": 0.0012},
  "rsi_9": {"mean": 50.0, "std": 15.0},
  "dxy_z": {"mean": 0.0, "std": 1.0},
  ...
}
```

---

### BE-03: Implementar StateTracker simplificado
**Archivo:** `src/core/state/state_tracker.py`
**Complejidad:** Media
**Dependencias:** Ninguna

**Requisitos:**
```python
@dataclass
class ModelState:
    model_id: str
    position: float = 0.0           # -1, 0, 1
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_equity: float = 10000.0
    peak_equity: float = 10000.0
    current_drawdown: float = 0.0
    trade_count_session: int = 0
    bars_in_position: int = 0

class StateTracker:
    """
    Tracker de estado por modelo para inferencia.

    Responsabilidades:
    - Mantener position actual por modelo
    - Calcular time_normalized
    - Persistir estado en PostgreSQL
    """

    def __init__(self, db_connection):
        self.states: Dict[str, ModelState] = {}
        self._load_from_db()

    def get_state_features(self, model_id: str,
                           current_bar: int,
                           total_bars: int = 60) -> Tuple[float, float]:
        """
        Obtener los 2 state features para observation.

        Returns:
            (position, time_normalized)
        """
        state = self.get_or_create(model_id)
        time_normalized = current_bar / total_bars
        return state.position, time_normalized

    def update_position(self, model_id: str,
                        new_position: float,
                        current_price: float):
        """Actualizar posicion y calcular PnL"""
        pass

    def _persist_to_db(self, state: ModelState):
        """Guardar estado en trading.model_states"""
        pass
```

---

### BE-04: Test Paridad (CRITICO)
**Archivo:** `src/tests/test_observation_parity.py`
**Complejidad:** Alta
**Dependencias:** BE-01, BE-02

**ESTE TEST DEBE PASAR ANTES DE ACTIVAR L5 EN PRODUCCION**

**Requisitos:**
```python
import pytest
import pandas as pd
import numpy as np
from src.core.builders.observation_builder import ObservationBuilderV19

# Path al dataset de training
TRAINING_DATA_PATH = "data/pipeline/07_output/RL_DS3_MACRO_CORE.csv"

class TestObservationParity:
    """
    Verificar que ObservationBuilder produce EXACTAMENTE
    los mismos valores que el dataset de training.
    """

    @pytest.fixture
    def builder(self):
        return ObservationBuilderV19()

    @pytest.fixture
    def training_data(self):
        return pd.read_csv(TRAINING_DATA_PATH, nrows=100)

    def test_feature_order_matches_training(self, builder, training_data):
        """Verificar orden de features"""
        expected_columns = builder.FEATURE_ORDER[:13]  # solo core
        actual_columns = [c for c in training_data.columns if c in expected_columns]
        assert actual_columns == expected_columns

    def test_observation_values_match(self, builder, training_data):
        """
        CRITICO: Valores deben coincidir con rtol=1e-5
        """
        for idx, row in training_data.iterrows():
            market_features = {
                col: row[col] for col in builder.FEATURE_ORDER[:13]
            }

            obs = builder.build(
                market_features=market_features,
                position=0.0,  # Asumiendo flat para test
                time_normalized=0.5
            )

            # Comparar core features (primeros 13)
            expected = np.array([row[col] for col in builder.FEATURE_ORDER[:13]])
            actual = obs[:13]

            np.testing.assert_allclose(
                actual, expected, rtol=1e-5,
                err_msg=f"Mismatch at row {idx}"
            )

    def test_observation_shape(self, builder):
        """Verificar shape correcto"""
        obs = builder.build(
            market_features={f: 0.0 for f in builder.FEATURE_ORDER[:13]},
            position=0.0,
            time_normalized=0.5
        )
        assert obs.shape == (15,)

    def test_observation_clipped(self, builder):
        """Verificar clipping a [-5, 5]"""
        extreme_features = {f: 100.0 for f in builder.FEATURE_ORDER[:13]}
        obs = builder.build(extreme_features, 0.0, 0.5)
        assert np.all(obs >= -5.0)
        assert np.all(obs <= 5.0)
```

**Comando de ejecucion:**
```bash
cd src && pytest tests/test_observation_parity.py -v --tb=short
```

---

### BE-05: Refactorizar L5 DAG para importar de src/
**Archivo:** `airflow/dags/l5_multi_model_inference.py`
**Complejidad:** Alta
**Dependencias:** BE-01, BE-03

**Cambios requeridos:**

1. **Eliminar FeatureBuilder local** (lineas ~563-700)
2. **Importar desde src/**:
```python
# Al inicio del archivo
import sys
sys.path.insert(0, '/opt/airflow/src')

from core.builders.observation_builder import ObservationBuilderV19
from core.state.state_tracker import StateTracker
```

3. **Usar ObservationBuilder en lugar de FeatureBuilder**:
```python
# En funcion build_observation()
def build_observation(**ctx) -> Dict[str, Any]:
    builder = ObservationBuilderV19()
    tracker = StateTracker(get_db_connection())

    # Obtener market features de inference_features_5m
    market_features = get_latest_features_from_db()

    # Obtener state features
    position, time_norm = tracker.get_state_features(
        model_id="ppo_v1",
        current_bar=ctx['bar_number'],
        total_bars=60
    )

    # Construir observation
    obs = builder.build(market_features, position, time_norm)

    return {"observation": obs.tolist(), "price": market_features['close']}
```

---

## Fase 2: Safety Layer - SEMANA 2

### BE-06: Implementar RiskManager
**Archivo:** `src/risk/risk_manager.py`
**Complejidad:** Media
**Dependencias:** Ninguna

**Requisitos:**
```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple
import logging

@dataclass
class RiskLimits:
    max_drawdown_pct: float = 15.0      # Kill switch
    max_daily_loss_pct: float = 5.0     # Stop trading today
    max_trades_per_day: int = 20        # Pause
    cooldown_after_losses: int = 3      # Consecutive losses
    cooldown_minutes: int = 30

class RiskManager:
    """
    Safety layer para validar senales antes de ejecucion.

    CRITICO: Esta clase previene perdidas catastroficas.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.kill_switch = False
        self.daily_pnl_pct = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None
        self._last_reset_date = datetime.now().date()

    def validate_signal(self, signal: str,
                        current_drawdown_pct: float) -> Tuple[bool, str]:
        """
        Validar si una senal puede ejecutarse.

        Returns:
            (allowed: bool, reason: str)
        """
        # Reset diario
        self._check_daily_reset()

        if self.kill_switch:
            return False, "KILL_SWITCH_ACTIVE"

        if current_drawdown_pct > self.limits.max_drawdown_pct:
            self.kill_switch = True
            logging.critical(f"KILL SWITCH ACTIVATED: Drawdown {current_drawdown_pct:.2f}%")
            return False, "MAX_DRAWDOWN_EXCEEDED"

        if self.daily_pnl_pct < -self.limits.max_daily_loss_pct:
            return False, "DAILY_LOSS_LIMIT"

        if self.trades_today >= self.limits.max_trades_per_day:
            return False, "MAX_TRADES_REACHED"

        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds
            return False, f"COOLDOWN_ACTIVE_{remaining}s"

        return True, "OK"

    def record_trade_result(self, pnl_pct: float):
        """Registrar resultado de trade para tracking"""
        self.trades_today += 1
        self.daily_pnl_pct += pnl_pct

        if pnl_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.limits.cooldown_after_losses:
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.limits.cooldown_minutes
                )
                logging.warning(f"Cooldown activated for {self.limits.cooldown_minutes}min")
        else:
            self.consecutive_losses = 0

    def get_status(self) -> dict:
        """Obtener estado actual para API endpoint"""
        return {
            "kill_switch_active": self.kill_switch,
            "daily_pnl_pct": self.daily_pnl_pct,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_active": self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "limits": {
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_trades_per_day": self.limits.max_trades_per_day
            }
        }

    def _check_daily_reset(self):
        """Reset contadores al cambio de dia"""
        today = datetime.now().date()
        if today > self._last_reset_date:
            self.daily_pnl_pct = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.cooldown_until = None
            self._last_reset_date = today
            logging.info("Daily risk counters reset")
```

---

### BE-07: Integrar RiskManager en L5 DAG
**Archivo:** `airflow/dags/l5_multi_model_inference.py`
**Complejidad:** Media
**Dependencias:** BE-05, BE-06

**Cambios:**
```python
# Import
from risk.risk_manager import RiskManager, RiskLimits

# Global instance
risk_manager = RiskManager(RiskLimits(
    max_drawdown_pct=15.0,
    max_daily_loss_pct=5.0,
    max_trades_per_day=20
))

# En funcion execute_inference()
def execute_inference(**ctx):
    obs = ctx['ti'].xcom_pull(key='observation')
    state = tracker.get_or_create(model_id)

    # Obtener accion del modelo
    action = model.predict(obs)
    signal = action_to_signal(action)

    # VALIDAR CON RISK MANAGER
    allowed, reason = risk_manager.validate_signal(
        signal=signal,
        current_drawdown_pct=state.current_drawdown * 100
    )

    if not allowed:
        logging.warning(f"Signal blocked by RiskManager: {reason}")
        signal = "HOLD"

    # Continuar con ejecucion/paper trading...
```

---

### BE-08: Crear endpoint /api/risk/status
**Archivo:** `services/multi_model_trading_api.py`
**Complejidad:** Baja
**Dependencias:** BE-06

**Agregar endpoint:**
```python
from pydantic import BaseModel
from typing import Optional

class RiskLimitsResponse(BaseModel):
    max_drawdown_pct: float
    max_daily_loss_pct: float
    max_trades_per_day: int

class RiskStatusResponse(BaseModel):
    kill_switch_active: bool
    current_drawdown_pct: float
    daily_pnl_pct: float
    trades_today: int
    consecutive_losses: int
    cooldown_active: bool
    cooldown_until: Optional[str]
    limits: RiskLimitsResponse

@app.get("/api/risk/status", response_model=RiskStatusResponse)
async def get_risk_status():
    """
    Obtener estado actual del RiskManager.

    IMPORTANTE: Este endpoint debe ser consultado por el dashboard
    para mostrar alertas visuales cuando hay restricciones activas.
    """
    # Cargar estado desde Redis o memoria compartida
    status = load_risk_status_from_redis()
    return RiskStatusResponse(**status)
```

---

### BE-09: Test unitarios RiskManager
**Archivo:** `src/tests/test_risk_manager.py`
**Complejidad:** Media
**Dependencias:** BE-06

```python
import pytest
from datetime import datetime, timedelta
from risk.risk_manager import RiskManager, RiskLimits

class TestRiskManager:

    @pytest.fixture
    def risk_manager(self):
        return RiskManager(RiskLimits(
            max_drawdown_pct=15.0,
            max_daily_loss_pct=5.0,
            max_trades_per_day=5,
            cooldown_after_losses=2,
            cooldown_minutes=10
        ))

    def test_allows_normal_trade(self, risk_manager):
        allowed, reason = risk_manager.validate_signal("LONG", 5.0)
        assert allowed is True
        assert reason == "OK"

    def test_kills_on_max_drawdown(self, risk_manager):
        allowed, reason = risk_manager.validate_signal("LONG", 16.0)
        assert allowed is False
        assert reason == "MAX_DRAWDOWN_EXCEEDED"
        assert risk_manager.kill_switch is True

    def test_blocks_after_daily_loss(self, risk_manager):
        risk_manager.daily_pnl_pct = -6.0
        allowed, reason = risk_manager.validate_signal("LONG", 5.0)
        assert allowed is False
        assert reason == "DAILY_LOSS_LIMIT"

    def test_activates_cooldown_after_losses(self, risk_manager):
        risk_manager.record_trade_result(-0.5)
        risk_manager.record_trade_result(-0.5)
        assert risk_manager.cooldown_until is not None

    def test_max_trades_per_day(self, risk_manager):
        for _ in range(5):
            risk_manager.record_trade_result(0.1)
        allowed, reason = risk_manager.validate_signal("LONG", 5.0)
        assert allowed is False
        assert reason == "MAX_TRADES_REACHED"
```

---

## Fase 3: Paper Trading - SEMANA 3

### BE-10: Implementar PaperTrader
**Archivo:** `src/trading/paper_trader.py`
**Complejidad:** Media

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import logging

@dataclass
class PaperTrade:
    trade_id: int
    model_id: str
    signal: str
    side: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"

class PaperTrader:
    """
    Simula ejecucion de trades sin ordenes reales.

    Usado para validar sistema antes de ir a produccion.
    """

    def __init__(self, initial_capital: float = 10000.0, db_connection=None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []
        self.trade_counter = 0
        self.db = db_connection

    def execute_signal(self, model_id: str, signal: str,
                       current_price: float) -> Optional[PaperTrade]:
        """
        Ejecutar senal en modo paper.

        Returns:
            PaperTrade si se abrio/cerro posicion, None si HOLD
        """
        current_position = self.positions.get(model_id)

        if signal == "HOLD":
            return None

        # Cerrar posicion existente si cambia direccion
        if current_position and current_position.status == "open":
            if (signal == "LONG" and current_position.side == "sell") or \
               (signal == "SHORT" and current_position.side == "buy") or \
               signal == "CLOSE":
                return self._close_position(model_id, current_price)

        # Abrir nueva posicion
        if signal in ["LONG", "SHORT"] and not current_position:
            return self._open_position(model_id, signal, current_price)

        return None

    def _open_position(self, model_id: str, signal: str,
                       price: float) -> PaperTrade:
        self.trade_counter += 1
        trade = PaperTrade(
            trade_id=self.trade_counter,
            model_id=model_id,
            signal=signal,
            side="buy" if signal == "LONG" else "sell",
            entry_price=price,
            entry_time=datetime.now()
        )
        self.positions[model_id] = trade
        logging.info(f"[PAPER] Opened {signal} at {price} for {model_id}")
        return trade

    def _close_position(self, model_id: str, price: float) -> PaperTrade:
        trade = self.positions.pop(model_id)
        trade.exit_price = price
        trade.exit_time = datetime.now()
        trade.status = "closed"

        # Calcular PnL
        if trade.side == "buy":
            trade.pnl_pct = (price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl_pct = (trade.entry_price - price) / trade.entry_price

        trade.pnl = self.current_capital * trade.pnl_pct
        self.current_capital += trade.pnl

        self.closed_trades.append(trade)
        self._persist_trade(trade)

        logging.info(f"[PAPER] Closed {trade.signal} at {price}, PnL: {trade.pnl_pct:.4%}")
        return trade

    def _persist_trade(self, trade: PaperTrade):
        """Guardar trade en trading_metrics"""
        if not self.db:
            return
        # INSERT INTO trading_metrics ...
```

---

### BE-11: Anadir PAPER_MODE flag a L5 DAG
**Archivo:** `airflow/dags/l5_multi_model_inference.py`
**Complejidad:** Baja
**Dependencias:** BE-10

```python
from airflow.models import Variable

# Al inicio
PAPER_MODE = Variable.get("PAPER_MODE", default_var="true").lower() == "true"

# En execute_inference()
if PAPER_MODE:
    paper_trader = PaperTrader(db_connection=get_db_connection())
    trade = paper_trader.execute_signal(model_id, signal, current_price)
    if trade:
        logging.info(f"[PAPER MODE] Trade: {trade}")
else:
    # Ejecucion real (futuro)
    logging.warning("REAL TRADING NOT IMPLEMENTED YET")
```

---

### BE-12: Persistir trades paper a trading_metrics
**Archivo:** `src/trading/paper_trader.py`
**Complejidad:** Baja
**Dependencias:** BE-10

```python
def _persist_trade(self, trade: PaperTrade):
    """Guardar trade en trading_metrics para dashboard"""
    if not self.db:
        return

    cur = self.db.cursor()
    cur.execute("""
        INSERT INTO trading_metrics
        (timestamp, metric_name, metric_value, metric_type, strategy_name, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, [
        trade.exit_time,
        'paper_trade_pnl',
        trade.pnl,
        'paper_trading',
        trade.model_id,
        json.dumps({
            'trade_id': trade.trade_id,
            'signal': trade.signal,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl_pct': trade.pnl_pct
        })
    ])
    self.db.commit()
```

---

## Fase 4: Monitoring - SEMANA 4

### BE-13: Implementar ModelMonitor
**Archivo:** `src/monitoring/model_monitor.py`
**Complejidad:** Media

```python
import numpy as np
from collections import deque
from scipy.stats import entropy

class ModelMonitor:
    """
    Detectar degradacion y drift en modelos RL.
    """

    def __init__(self, window_size: int = 100):
        self.action_history = deque(maxlen=window_size)
        self.pnl_history = deque(maxlen=window_size)
        self.baseline_action_dist = None

    def record_action(self, action: float):
        self.action_history.append(action)

    def record_pnl(self, pnl: float):
        self.pnl_history.append(pnl)

    def set_baseline(self, actions: List[float]):
        """Establecer distribucion baseline de acciones (de backtest)"""
        hist, _ = np.histogram(actions, bins=20, range=(-1, 1), density=True)
        self.baseline_action_dist = hist + 1e-10  # Evitar log(0)

    def check_action_drift(self) -> float:
        """
        Calcular KL divergence vs baseline.

        Returns:
            KL divergence (0 = igual, >0.5 = drift significativo)
        """
        if len(self.action_history) < 50 or self.baseline_action_dist is None:
            return 0.0

        current_hist, _ = np.histogram(
            list(self.action_history), bins=20, range=(-1, 1), density=True
        )
        current_hist = current_hist + 1e-10

        kl_div = entropy(current_hist, self.baseline_action_dist)
        return kl_div

    def check_stuck_behavior(self) -> bool:
        """Detectar si modelo esta generando misma accion repetidamente"""
        if len(self.action_history) < 20:
            return False

        recent = list(self.action_history)[-20:]
        unique_actions = len(set([round(a, 1) for a in recent]))
        return unique_actions <= 2

    def get_rolling_sharpe(self) -> float:
        """Sharpe ratio rolling de ultimos trades"""
        if len(self.pnl_history) < 10:
            return 0.0

        pnls = np.array(self.pnl_history)
        return pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(252)

    def get_health_status(self) -> dict:
        """Estado de salud del modelo para API"""
        return {
            "action_drift_kl": self.check_action_drift(),
            "stuck_behavior": self.check_stuck_behavior(),
            "rolling_sharpe": self.get_rolling_sharpe(),
            "actions_recorded": len(self.action_history),
            "trades_recorded": len(self.pnl_history)
        }
```

---

### BE-14: Crear endpoint /api/monitor/health
**Archivo:** `services/multi_model_trading_api.py`
**Complejidad:** Baja

```python
class ModelHealthResponse(BaseModel):
    model_id: str
    action_drift_kl: float
    stuck_behavior: bool
    rolling_sharpe: float
    status: str  # "healthy", "warning", "critical"

@app.get("/api/monitor/health")
async def get_model_health() -> List[ModelHealthResponse]:
    """Estado de salud de todos los modelos activos"""
    # Cargar desde Redis
    pass
```

---

### BE-15: Integrar alertas Telegram (Opcional)
**Archivo:** `src/monitoring/alerts.py`
**Complejidad:** Baja

```python
import requests

class TelegramAlerts:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_alert(self, message: str, level: str = "INFO"):
        emoji = {"INFO": "i", "WARNING": "!", "CRITICAL": "X"}
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        requests.post(url, json={
            "chat_id": self.chat_id,
            "text": f"[{emoji.get(level, 'i')}] USDCOP Trading\n{message}"
        })
```

---

## Verificacion Final

### Comandos de Test
```bash
# Test paridad (CRITICO)
cd src && pytest tests/test_observation_parity.py -v

# Test risk manager
cd src && pytest tests/test_risk_manager.py -v

# Verificar L5 genera metricas
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \
  "SELECT COUNT(*) FROM trading_metrics WHERE timestamp > NOW() - INTERVAL '1 hour';"

# Trigger manual L5 (paper mode)
docker exec usdcop-airflow-webserver airflow dags trigger v3.l5_multi_model_inference
```

### Checklist Pre-Produccion
- [ ] BE-04 Test paridad pasa con rtol=1e-5
- [ ] BE-09 Tests RiskManager pasan
- [ ] L5 DAG importa correctamente de src/
- [ ] trading_metrics tiene datos de paper trading
- [ ] API /api/risk/status responde correctamente
- [ ] Paper trading ejecutado 2+ semanas sin errores
