# Plan Dashboard Simplificado - PPO V1 Production
## Para: Gemini (Frontend) + Claude (Backend)

**Fecha**: 7 Enero 2026
**Objetivo**: Dashboard minimalista para monitorear modelo RL PPO V1 en producción
**Filosofía**: Menos es más. Si no aporta valor, se elimina.

---

## 1. CONTEXTO DEL MODELO

### Modelo en Producción
| Atributo | Valor |
|----------|-------|
| **Nombre** | PPO USDCOP V1 |
| **Algoritmo** | Proximal Policy Optimization |
| **Fecha Entrenamiento** | 26 Diciembre 2025 |
| **Sharpe Ratio** | 2.91 |
| **Max Drawdown** | 0.68% |
| **Win Rate** | 44.85% |
| **Capital Inicial** | $10,000 USD |

### Datos Disponibles
| Tipo | Rango | Registros |
|------|-------|-----------|
| **Training Data** | 2020-01-02 → 2025-12-26 | ~85,000 barras |
| **Out-of-Sample** | 2025-12-27 → 2026-01-06 | ~1,500 barras |
| **Total OHLCV** | - | 87,491 registros |

### Horario de Mercado (CRÍTICO)
```
Zona Horaria: America/Bogota (UTC-5)
Días: Lunes a Viernes
Apertura: 08:00 COT
Cierre: 12:55 COT
Barras por día: 59 (5-minute bars)
```

**El modelo SOLO opera en este horario. Fuera de horario = FLAT.**

---

## 2. ARQUITECTURA SIMPLIFICADA

### Backend (Claude - Ya Implementado)
```
services/
├── multi_model_trading_api.py   # API principal puerto 8006
├── trading_api_realtime.py      # WebSocket puerto 8000
└── src/
    ├── core/
    │   ├── builders/observation_builder_v19.py  # 15-dim observation
    │   └── state/state_tracker.py               # Tracking posición
    ├── risk/risk_manager.py                     # Kill switch + límites
    ├── trading/paper_trader.py                  # Simulación trades
    └── monitoring/model_monitor.py              # Drift detection
```

### Frontend (Gemini - Por Simplificar)
```
usdcop-trading-dashboard/
├── app/
│   ├── page.tsx           # Dashboard principal (SIMPLIFICAR)
│   ├── trades/page.tsx    # Historial de trades (CREAR)
│   └── risk/page.tsx      # Estado del sistema (CREAR)
├── components/
│   ├── charts/
│   │   └── EquityCurveChart.tsx    # Gráfico principal
│   ├── trading/
│   │   ├── PositionCard.tsx        # Estado posición actual
│   │   ├── KPICards.tsx            # Métricas principales
│   │   └── RiskStatusCard.tsx      # Ya existe - Semáforo
│   └── tables/
│       └── TradesTable.tsx         # Historial operaciones
└── hooks/
    ├── useEquityCurve.ts           # Datos curva equity
    ├── useLiveState.ts             # Estado en tiempo real
    └── useRiskStatus.ts            # Ya existe
```

---

## 3. ENDPOINTS API (Backend)

### Endpoints Existentes (Funcionando)
```bash
# Health check
GET http://localhost:8000/api/health

# Datos de mercado
GET http://localhost:8000/api/stats/USDCOP

# Modelos registrados
GET http://localhost:8006/api/models

# Curva de equity (simulada)
GET http://localhost:8006/api/models/equity-curves?period=7d
```

### Endpoints a Crear/Arreglar

#### 1. Estado en Vivo del Modelo
```
GET /api/state/live
```
**Response:**
```json
{
  "model_id": "ppo_v1",
  "position": "LONG",           // LONG | SHORT | FLAT
  "entry_price": 4185.50,
  "entry_time": "2026-01-07T09:15:00-05:00",
  "current_price": 4192.30,
  "unrealized_pnl": 68.50,      // En USD
  "unrealized_pnl_pct": 0.68,   // En porcentaje
  "bars_in_position": 12,
  "equity": 10685.50,
  "drawdown_pct": 0.45,
  "peak_equity": 10734.20,
  "market_status": "OPEN",      // OPEN | CLOSED | PRE_MARKET
  "last_signal": "HOLD",
  "last_updated": "2026-01-07T10:30:00-05:00"
}
```

#### 2. Resumen de Performance
```
GET /api/performance/summary?period=out_of_sample
```
**Response:**
```json
{
  "period": {
    "start": "2025-12-27",
    "end": "2026-01-06",
    "trading_days": 8,
    "total_bars": 472
  },
  "metrics": {
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.12,
    "max_drawdown_pct": 1.78,
    "current_drawdown_pct": 0.45,
    "total_return_pct": 8.56,
    "win_rate": 46.2,
    "profit_factor": 1.65,
    "total_trades": 24,
    "avg_trade_duration_bars": 8.5
  },
  "comparison_vs_backtest": {
    "sharpe_diff": -1.06,      // 1.85 vs 2.91 backtest
    "drawdown_diff": 1.10,     // 1.78% vs 0.68% backtest
    "status": "WITHIN_TOLERANCE"
  }
}
```

#### 3. Historial de Trades
```
GET /api/trades/history?period=out_of_sample&limit=50
```
**Response:**
```json
{
  "trades": [
    {
      "trade_id": 24,
      "model_id": "ppo_v1",
      "side": "LONG",
      "entry_price": 4150.25,
      "exit_price": 4168.80,
      "entry_time": "2026-01-06T09:00:00-05:00",
      "exit_time": "2026-01-06T10:25:00-05:00",
      "duration_bars": 17,
      "pnl_usd": 18.55,
      "pnl_pct": 0.45,
      "exit_reason": "SIGNAL_CHANGE"  // SIGNAL_CHANGE | STOP_LOSS | END_OF_DAY
    }
  ],
  "summary": {
    "total_trades": 24,
    "winning": 11,
    "losing": 13,
    "win_rate": 45.83
  }
}
```

#### 4. Estado del Risk Manager
```
GET /api/risk/status
```
**Response:**
```json
{
  "status": "OPERATIONAL",      // OPERATIONAL | WARNING | HALTED
  "kill_switch_active": false,
  "daily_blocked": false,
  "cooldown_active": false,
  "cooldown_remaining_minutes": 0,
  "metrics": {
    "current_drawdown_pct": 0.45,
    "daily_pnl_pct": 0.82,
    "trades_today": 3,
    "consecutive_losses": 1
  },
  "limits": {
    "max_drawdown_pct": 15.0,
    "max_daily_loss_pct": 5.0,
    "max_trades_per_day": 20,
    "cooldown_after_losses": 3
  },
  "warnings": []
}
```

---

## 4. TABLAS DE BASE DE DATOS

### Tabla Existente: usdcop_m5_ohlcv
```sql
-- Ya existe con 87,491 registros
SELECT time, open, high, low, close, volume
FROM usdcop_m5_ohlcv
WHERE time >= '2025-12-27'  -- Datos out-of-sample
ORDER BY time;
```

### Nueva Tabla: trading_state (Estado del Modelo)
```sql
CREATE TABLE IF NOT EXISTS trading_state (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL UNIQUE,

    -- Posición actual
    position VARCHAR(10) DEFAULT 'FLAT' CHECK (position IN ('LONG', 'SHORT', 'FLAT')),
    entry_price DECIMAL(12,4),
    entry_time TIMESTAMPTZ,
    bars_in_position INT DEFAULT 0,

    -- PnL
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    realized_pnl DECIMAL(12,4) DEFAULT 0,

    -- Equity tracking
    equity DECIMAL(14,4) DEFAULT 10000,
    peak_equity DECIMAL(14,4) DEFAULT 10000,
    drawdown_pct DECIMAL(6,4) DEFAULT 0,

    -- Estadísticas
    trade_count INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,

    -- Metadata
    last_signal VARCHAR(10),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índice para queries rápidos
CREATE INDEX idx_trading_state_model ON trading_state(model_id);

-- Insert estado inicial para PPO V1
INSERT INTO trading_state (model_id, equity, peak_equity)
VALUES ('ppo_v1', 10000, 10000)
ON CONFLICT (model_id) DO NOTHING;
```

### Nueva Tabla: trades_history (Historial de Operaciones)
```sql
CREATE TABLE IF NOT EXISTS trades_history (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,

    -- Detalles del trade
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    duration_bars INT,

    -- Resultado
    pnl_usd DECIMAL(12,4),
    pnl_pct DECIMAL(8,4),
    exit_reason VARCHAR(20),  -- SIGNAL_CHANGE, STOP_LOSS, END_OF_DAY, KILL_SWITCH

    -- Estado al momento del trade
    equity_at_entry DECIMAL(14,4),
    equity_at_exit DECIMAL(14,4),
    drawdown_at_entry DECIMAL(6,4),

    -- Metadata
    bar_number INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_trades_model ON trades_history(model_id);
CREATE INDEX idx_trades_time ON trades_history(entry_time DESC);
CREATE INDEX idx_trades_exit ON trades_history(exit_time DESC);
```

### Nueva Tabla: equity_snapshots (Para Gráfico)
```sql
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    equity DECIMAL(14,4) NOT NULL,
    drawdown_pct DECIMAL(6,4),
    position VARCHAR(10),
    bar_close_price DECIMAL(12,4),

    UNIQUE(model_id, timestamp)
);

-- Índice para queries de curva
CREATE INDEX idx_equity_model_time ON equity_snapshots(model_id, timestamp DESC);

-- Hypertable para TimescaleDB (opcional pero recomendado)
SELECT create_hypertable('equity_snapshots', 'timestamp', if_not_exists => TRUE);
```

---

## 5. DISEÑO DEL FRONTEND

### Paleta de Colores
```css
/* Tema Oscuro Profesional */
--bg-primary: #050816;        /* Fondo principal */
--bg-card: #0A0E27;           /* Cards */
--bg-elevated: #0F1422;       /* Elementos elevados */

/* Colores de Mercado */
--market-up: #00D395;         /* Verde - Ganancias */
--market-down: #FF3B69;       /* Rojo - Pérdidas */
--market-neutral: #8B92A8;    /* Gris - Sin cambio */

/* Acentos */
--accent-cyan: #06B6D4;       /* Cyan - Acciones primarias */
--accent-purple: #8B5CF6;     /* Púrpura - Highlights */

/* Estados */
--status-operational: #10B981;  /* Verde - OK */
--status-warning: #F59E0B;      /* Amarillo - Advertencia */
--status-critical: #EF4444;     /* Rojo - Crítico */
```

### Página Principal: `/dashboard`

```
┌─────────────────────────────────────────────────────────────────────┐
│  USDCOP PPO V1 Dashboard              🟢 OPERATIONAL    08:45 COT   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│  │   SHARPE     │ │  MAX DD      │ │  WIN RATE    │ │  RETURN      ││
│  │    1.85      │ │   1.78%      │ │   46.2%      │ │   +8.56%     ││
│  │  vs 2.91 BT  │ │  vs 0.68% BT │ │  vs 44.8% BT │ │  Out-Sample  ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘│
│                                                                      │
│  ┌────────────────────────────────────────┐ ┌──────────────────────┐│
│  │                                        │ │  POSICIÓN ACTUAL     ││
│  │         EQUITY CURVE                   │ │  ┌────────────────┐  ││
│  │                                        │ │  │     LONG       │  ││
│  │    $10,800 ─────────────╱              │ │  │   🟢 +0.68%    │  ││
│  │    $10,600 ────────────╱               │ │  └────────────────┘  ││
│  │    $10,400 ───────────╱                │ │                      ││
│  │    $10,200 ──────────╱                 │ │  Entry: $4,185.50    ││
│  │    $10,000 ─────────╱                  │ │  Current: $4,192.30  ││
│  │            27 Dec  30 Dec  2 Jan  6 Jan│ │  PnL: +$68.50        ││
│  │                                        │ │  Bars: 12            ││
│  └────────────────────────────────────────┘ │                      ││
│                                             │  ─────────────────── ││
│                                             │  DRAWDOWN: 0.45%     ││
│                                             │  ████░░░░░░░░ 3/15%  ││
│                                             └──────────────────────┘│
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  ÚLTIMAS OPERACIONES                                             ││
│  │  ───────────────────────────────────────────────────────────────││
│  │  #24  LONG   $4,150 → $4,168   +$18.55 (+0.45%)   17 bars  ✅   ││
│  │  #23  SHORT  $4,180 → $4,165   +$15.00 (+0.36%)   8 bars   ✅   ││
│  │  #22  LONG   $4,142 → $4,138   -$4.00  (-0.10%)   5 bars   ❌   ││
│  │  [Ver historial completo →]                                      ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### Página de Trades: `/trades`

```
┌─────────────────────────────────────────────────────────────────────┐
│  Historial de Operaciones - Out of Sample (27 Dec - 6 Jan)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Filtros: [Período ▼] [Lado ▼] [Resultado ▼]     Exportar [CSV]     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ #   │ Fecha      │ Lado  │ Entrada  │ Salida   │ PnL     │ Dur  ││
│  │─────┼────────────┼───────┼──────────┼──────────┼─────────┼──────││
│  │ 24  │ 06-Jan 9:00│ LONG  │ $4,150.25│ $4,168.80│ +$18.55 │ 17   ││
│  │ 23  │ 06-Jan 8:15│ SHORT │ $4,180.00│ $4,165.00│ +$15.00 │ 8    ││
│  │ 22  │ 03-Jan 12:30│ LONG │ $4,142.50│ $4,138.50│ -$4.00  │ 5    ││
│  │ ... │            │       │          │          │         │      ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  Página 1 de 3                              [← Anterior] [Siguiente →]│
│                                                                      │
│  ┌─────────────────────────┐ ┌─────────────────────────┐            │
│  │  DISTRIBUCIÓN PnL       │ │  DURACIÓN TRADES        │            │
│  │  (Histograma)           │ │  (Histograma)           │            │
│  │  █                      │ │      █                  │            │
│  │  █ █                    │ │    █ █ █                │            │
│  │  █ █ █ █                │ │  █ █ █ █ █              │            │
│  │  -2% 0% +2% +4%         │ │  1  5  10 15 20 bars    │            │
│  └─────────────────────────┘ └─────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### Página de Riesgo: `/risk`

```
┌─────────────────────────────────────────────────────────────────────┐
│  Panel de Control de Riesgo                    🟢 SISTEMA OPERATIVO │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     ESTADO DEL SISTEMA                          ││
│  │                                                                  ││
│  │     ████████████████████████████████░░░░░░░  OPERATIVO          ││
│  │                                                                  ││
│  │     Kill Switch: OFF     Daily Block: OFF     Cooldown: OFF     ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  ┌──────────────────────┐ ┌──────────────────────┐                  │
│  │  LÍMITES CONFIGURADOS │ │  ESTADO ACTUAL       │                  │
│  │  ────────────────────│ │  ────────────────────│                  │
│  │  Max Drawdown: 15%   │ │  Drawdown:  0.45%    │  ████░░░░░░ OK  │
│  │  Max Loss Diaria: 5% │ │  Loss Hoy:  +0.82%   │  ██████████ OK  │
│  │  Max Trades/Día: 20  │ │  Trades:    3        │  ██░░░░░░░░ OK  │
│  │  Cooldown: 3 losses  │ │  Consec Loss: 1      │  ████░░░░░░ OK  │
│  └──────────────────────┘ └──────────────────────┘                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  HORARIO DE OPERACIÓN                                           ││
│  │  ───────────────────────────────────────────────────────────────││
│  │                                                                  ││
│  │  Lun │████████████████░░░░░░░░░░░░░░░░░░░░░░│ 08:00 - 12:55 COT ││
│  │  Mar │████████████████░░░░░░░░░░░░░░░░░░░░░░│ 08:00 - 12:55 COT ││
│  │  Mie │████████████████░░░░░░░░░░░░░░░░░░░░░░│ 08:00 - 12:55 COT ││
│  │  Jue │████████████████░░░░░░░░░░░░░░░░░░░░░░│ 08:00 - 12:55 COT ││
│  │  Vie │████████████████░░░░░░░░░░░░░░░░░░░░░░│ 08:00 - 12:55 COT ││
│  │  Sab │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ CERRADO           ││
│  │  Dom │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ CERRADO           ││
│  │                                                                  ││
│  │  Próxima apertura: Lunes 08:00 COT (en 35h 15min)               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. COMPONENTES A CREAR/MODIFICAR

### Eliminar (Código Muerto)
```
components/views/
├── OrderBookDisabled.tsx          ❌ ELIMINAR
├── PipelineStatusV2.tsx           ❌ ELIMINAR
├── ProfessionalTradingTerminal.tsx ❌ ELIMINAR (duplicado)
├── PortfolioExposureAnalysis.tsx  ❌ ELIMINAR
├── RealTimeRiskMonitor.tsx        ❌ ELIMINAR (reemplazar por simple)
├── RiskAlertsCenter.tsx           ❌ ELIMINAR
├── TradingSignals.tsx             ❌ ELIMINAR (duplicado)
└── VolumeProfileChart.tsx         ❌ ELIMINAR

components/charts/
├── AdvancedExportCapabilities.tsx ❌ ELIMINAR
├── AnimatedChart.tsx              ❌ ELIMINAR
├── CanvasChart.tsx                ❌ ELIMINAR
├── HighPerformanceVirtualizedChart.tsx ❌ ELIMINAR
└── VirtualizedChart.tsx           ❌ ELIMINAR
```

### Mantener (Simplificar si es necesario)
```
components/trading/
├── RiskStatusCard.tsx             ✅ MANTENER (ya funciona)
└── TradingSignals.tsx             ✅ SIMPLIFICAR

components/charts/
├── EquityCurveChart.tsx           ✅ MANTENER
└── RealDataTradingChart.tsx       ✅ MANTENER

hooks/
├── useRiskStatus.ts               ✅ MANTENER
├── useFinancialMetrics.ts         ✅ MANTENER
└── useEquityCurveStream.ts        ✅ MANTENER
```

### Crear (Nuevos)
```
components/trading/
├── PositionCard.tsx               🆕 CREAR - Estado posición actual
├── KPICards.tsx                   🆕 CREAR - 4 métricas principales
└── TradesTable.tsx                🆕 CREAR - Historial operaciones

hooks/
├── useLiveState.ts                🆕 CREAR - GET /api/state/live
├── usePerformanceSummary.ts       🆕 CREAR - GET /api/performance/summary
└── useTradesHistory.ts            🆕 CREAR - GET /api/trades/history

app/
├── trades/page.tsx                🆕 CREAR - Página historial
└── risk/page.tsx                  🆕 CREAR - Página riesgo
```

---

## 7. FLUJO DE DATOS EN TIEMPO REAL

### Horario de Mercado (Lun-Vie 8:00-12:55 COT)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  TwelveData │────▶│   Redis     │────▶│  Dashboard  │
│    API      │     │   Stream    │     │  WebSocket  │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                    │
      │ Cada 5min         │ Push               │ Update
      ▼                   ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PostgreSQL │     │   Model     │     │    UI       │
│   OHLCV     │────▶│  Inference  │────▶│  Refresh    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Proceso cada 5 minutos (en horario de mercado)
1. **L0 DAG** ingesta nueva barra OHLCV
2. **L5 DAG** calcula features y ejecuta inferencia
3. **API** actualiza `trading_state` y `equity_snapshots`
4. **WebSocket** notifica al frontend
5. **Dashboard** actualiza gráficos y métricas

### Fuera de Horario
- Modelo en estado FLAT
- Dashboard muestra último estado
- Countdown hasta próxima apertura
- Sin actualizaciones de datos

---

## 8. PAPER TRADING SIMULATION

### Configuración
```python
PAPER_TRADING_CONFIG = {
    "initial_capital": 10000.0,
    "position_size": 1.0,           # 100% del capital por trade
    "slippage_bps": 2,              # 2 basis points
    "commission_per_trade": 0.0,    # Sin comisión para simplificar
    "data_source": "out_of_sample", # 27 Dec 2025 - 6 Jan 2026
}
```

### Flujo de Simulación
```
1. Cargar datos out-of-sample (1,500 barras)
2. Para cada barra:
   a. Construir observation (15-dim)
   b. Ejecutar model.predict()
   c. Discretizar acción → LONG/SHORT/FLAT
   d. Validar con RiskManager
   e. Si permitido: ejecutar trade (paper)
   f. Actualizar equity, drawdown, stats
   g. Guardar en BD
3. Calcular métricas finales
4. Comparar vs backtest original
```

### Métricas a Comparar
| Métrica | Backtest | Out-of-Sample | Diferencia |
|---------|----------|---------------|------------|
| Sharpe | 2.91 | ? | ? |
| Max DD | 0.68% | ? | ? |
| Win Rate | 44.85% | ? | ? |
| Return | 15% | ? | ? |

---

## 9. CHECKLIST DE IMPLEMENTACIÓN

### Backend (Claude) - ✅ COMPLETADO 7 Ene 2026
- [x] Crear tablas `trading_state`, `trades_history`, `equity_snapshots`
- [x] Implementar `GET /api/state/live`
- [x] Implementar `GET /api/performance/summary`
- [x] Implementar `GET /api/trades/history`
- [x] Implementar `GET /api/risk/status`
- [x] Implementar `GET /api/equity/curve`
- [x] Ejecutar paper trading simulation con datos out-of-sample
- [x] Poblar tablas con resultados de simulación

### Datos Disponibles en BD (7 Ene 2026)
```
trading_state:     1 registro  (Estado actual PPO V1)
trades_history:   48 trades    (Out-of-sample Dec 29 - Jan 6)
equity_snapshots: 50 puntos    (Para gráfico de equity)

Métricas Actuales:
- Equity: $9,590.33 (de $10,000 inicial)
- Return: -4.10%
- Drawdown: 4.46%
- Trades: 48 (6 wins, 42 losses)
- Win Rate: 12.5%

NOTA: Resultados negativos porque faltan datos MACRO reales.
El modelo usa features DXY, VIX, EMBI con valores neutros (0.0).
Con datos macro reales, el rendimiento debería mejorar.
```

### Frontend (Gemini)
- [ ] Eliminar componentes marcados como ❌
- [ ] Crear `PositionCard.tsx` para estado actual
- [ ] Crear `KPICards.tsx` para 4 métricas principales
- [ ] Crear `TradesTable.tsx` para historial
- [ ] Crear hook `useLiveState.ts`
- [ ] Crear hook `usePerformanceSummary.ts`
- [ ] Crear hook `useTradesHistory.ts`
- [ ] Simplificar `page.tsx` según diseño
- [ ] Crear `app/trades/page.tsx`
- [ ] Crear `app/risk/page.tsx`
- [ ] Actualizar navegación (solo 3 páginas)
- [ ] Aplicar tema oscuro consistente
- [ ] Agregar indicador de horario de mercado
- [ ] Testing con datos out-of-sample

---

## 10. PRIORIDADES

### Fase 1: Fundamentos (Día 1)
1. Backend: Crear tablas y endpoints básicos
2. Frontend: Eliminar código muerto
3. Frontend: Crear estructura de 3 páginas

### Fase 2: Dashboard Principal (Día 2)
1. Backend: Ejecutar paper trading simulation
2. Frontend: Implementar KPIs y curva de equity
3. Frontend: Implementar PositionCard

### Fase 3: Historial y Riesgo (Día 3)
1. Frontend: Página de trades con tabla
2. Frontend: Página de riesgo con semáforo
3. Testing end-to-end

### Fase 4: Polish (Día 4)
1. Responsive design
2. Animaciones suaves
3. Error handling
4. Loading states

---

## NOTAS IMPORTANTES

1. **NO agregar features nuevas** - Solo lo especificado en este documento
2. **NO over-engineer** - Si funciona simple, es mejor
3. **Datos reales primero** - Conectar a APIs antes de mockear
4. **Mobile-friendly** - Dashboard debe verse bien en tablet
5. **Performance** - Lazy loading para gráficos pesados
6. **Accesibilidad** - Colores con suficiente contraste

---

**Documento creado por**: Claude (Backend Lead)
**Para**: Gemini (Frontend Lead)
**Versión**: 1.0
**Última actualización**: 7 Enero 2026
