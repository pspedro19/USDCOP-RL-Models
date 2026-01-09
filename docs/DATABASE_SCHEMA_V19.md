# Database Schema Documentation V19

**Version:** 19.0
**Database:** PostgreSQL 15 + TimescaleDB
**Last Updated:** 2025-12-26

---

## Table of Contents

1. [Overview](#1-overview)
2. [Entity Relationship Diagram](#2-entity-relationship-diagram)
3. [Schema: public](#3-schema-public)
4. [Schema: dw (Data Warehouse)](#4-schema-dw-data-warehouse)
5. [Views and Materialized Views](#5-views-and-materialized-views)
6. [Functions](#6-functions)
7. [Indexes](#7-indexes)
8. [Sample Queries](#8-sample-queries)
9. [Maintenance](#9-maintenance)

---

## 1. Overview

### Database Configuration

| Property | Value |
|----------|-------|
| Database Name | `usdcop_trading` |
| Host | `usdcop-postgres-timescale` |
| Port | `5432` |
| Extensions | TimescaleDB |

### Schema Organization

| Schema | Purpose |
|--------|---------|
| `public` | Core trading data (OHLCV, users, metrics) |
| `dw` | Data warehouse for multi-model trading |

### Table Count Summary

| Schema | Tables | Hypertables | Views |
|--------|--------|-------------|-------|
| public | 4 | 2 | 4 |
| dw | 6 | 2 | 1 |

---

## 2. Entity Relationship Diagram

```
                         PUBLIC SCHEMA
    +------------------------------------------------------------------+
    |                                                                  |
    |  +------------------+     +--------------------+                 |
    |  |     users        |     |   trading_sessions |                 |
    |  +------------------+     +--------------------+                 |
    |  | PK id            |<----| FK user_id         |                 |
    |  | username         |     | session_start      |                 |
    |  | email            |     | session_end        |                 |
    |  | password_hash    |     | strategy_name      |                 |
    |  | is_active        |     | initial_balance    |                 |
    |  | is_admin         |     | final_balance      |                 |
    |  +------------------+     | status             |                 |
    |                           +--------------------+                 |
    |                                                                  |
    |  +------------------+     +--------------------+                 |
    |  | usdcop_m5_ohlcv  |     | macro_indicators   |                 |
    |  | (Hypertable)     |     | _daily             |                 |
    |  +------------------+     +--------------------+                 |
    |  | PK time, symbol  |     | PK fecha/date      |                 |
    |  | open             |     | dxy                |                 |
    |  | high             |     | vix                |                 |
    |  | low              |     | embi               |                 |
    |  | close            |     | brent              |                 |
    |  | volume           |     | ... (37 cols)      |                 |
    |  | source           |     +--------------------+                 |
    |  +------------------+                                            |
    |                                                                  |
    |  +------------------+                                            |
    |  | trading_metrics  |                                            |
    |  | (Hypertable)     |                                            |
    |  +------------------+                                            |
    |  | PK timestamp,    |                                            |
    |  |    metric_name,  |                                            |
    |  |    metric_type   |                                            |
    |  | metric_value     |                                            |
    |  | strategy_name    |                                            |
    |  | model_version    |                                            |
    |  +------------------+                                            |
    +------------------------------------------------------------------+

                          DW SCHEMA
    +------------------------------------------------------------------+
    |                                                                  |
    |  +------------------+                                            |
    |  |   dim_strategy   |                                            |
    |  +------------------+                                            |
    |  | PK strategy_id   |<---------+--------+--------+--------+     |
    |  | strategy_code    |          |        |        |        |     |
    |  | strategy_name    |          |        |        |        |     |
    |  | strategy_type    |          |        |        |        |     |
    |  | is_active        |          |        |        |        |     |
    |  +------------------+          |        |        |        |     |
    |                                |        |        |        |     |
    |  +------------------+  +-------+--+  +--+-------+  +------+---+  |
    |  |fact_strategy     |  |fact_equity|  |fact_strat|  |fact_strat|  |
    |  |_signals          |  |_curve     |  |_perform  |  |_positions|  |
    |  +------------------+  +----------+  +----------+  +----------+  |
    |  | PK signal_id     |  | PK eq_id  |  | PK pf_id  |  | PK pos_id |  |
    |  | FK strategy_id   |  | FK strat  |  | FK strat  |  | FK strat  |  |
    |  | timestamp_utc    |  | timestamp |  | date_cot  |  | side      |  |
    |  | signal           |  | equity    |  | sharpe    |  | quantity  |  |
    |  | confidence       |  | return_%  |  | win_rate  |  | entry_px  |  |
    |  | entry_price      |  | drawdown  |  | max_dd    |  | status    |  |
    |  +------------------+  +----------+  +----------+  +----------+  |
    |                                                                  |
    |  +--------------------+    +----------------------+              |
    |  | fact_rl_inference  |    | fact_agent_actions   |              |
    |  | (Hypertable)       |    +----------------------+              |
    |  +--------------------+    | PK id                |              |
    |  | PK inference_id    |    | timestamp            |              |
    |  | timestamp_utc      |    | bar_number           |              |
    |  | model_id           |    | action               |              |
    |  | action_raw         |    | position             |              |
    |  | observation (13)   |    | model_version        |              |
    |  | close_price        |    +----------------------+              |
    |  +--------------------+                                          |
    +------------------------------------------------------------------+
```

---

## 3. Schema: public

### 3.1 users

User authentication and profile management.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PK | Auto-increment ID |
| username | VARCHAR(50) | UNIQUE, NOT NULL | Login username |
| email | VARCHAR(100) | UNIQUE, NOT NULL | Email address |
| password_hash | VARCHAR(255) | NOT NULL | Bcrypt hash |
| first_name | VARCHAR(50) | | User first name |
| last_name | VARCHAR(50) | | User last name |
| is_active | BOOLEAN | DEFAULT true | Account status |
| is_admin | BOOLEAN | DEFAULT false | Admin privileges |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Creation timestamp |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | Last update |
| last_login | TIMESTAMPTZ | | Last login time |

---

### 3.2 usdcop_m5_ohlcv (TimescaleDB Hypertable)

Primary OHLCV market data - Single Source of Truth (SSOT).

```sql
CREATE TABLE usdcop_m5_ohlcv (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'USD/COP',
    open DECIMAL(12,6) NOT NULL,
    high DECIMAL(12,6) NOT NULL,
    low DECIMAL(12,6) NOT NULL,
    close DECIMAL(12,6) NOT NULL,
    volume BIGINT DEFAULT 0,
    source TEXT DEFAULT 'twelvedata',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (time, symbol),
    CONSTRAINT chk_prices_positive CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT chk_high_gte_low CHECK (high >= low),
    CONSTRAINT chk_high_gte_open CHECK (high >= open),
    CONSTRAINT chk_high_gte_close CHECK (high >= close),
    CONSTRAINT chk_low_lte_open CHECK (low <= open),
    CONSTRAINT chk_low_lte_close CHECK (low <= close),
    CONSTRAINT chk_volume_non_negative CHECK (volume >= 0)
);

-- Convert to hypertable
SELECT create_hypertable('usdcop_m5_ohlcv', 'time', if_not_exists => TRUE);
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| time | TIMESTAMPTZ | PK | Bar timestamp |
| symbol | TEXT | PK, DEFAULT 'USD/COP' | Trading pair |
| open | DECIMAL(12,6) | NOT NULL, > 0 | Open price |
| high | DECIMAL(12,6) | NOT NULL, >= open, close, low | High price |
| low | DECIMAL(12,6) | NOT NULL, <= open, close, high | Low price |
| close | DECIMAL(12,6) | NOT NULL, > 0 | Close price |
| volume | BIGINT | DEFAULT 0, >= 0 | Trading volume |
| source | TEXT | DEFAULT 'twelvedata' | Data source |
| created_at | TIMESTAMPTZ | DEFAULT NOW() | Insert timestamp |
| updated_at | TIMESTAMPTZ | DEFAULT NOW() | Update timestamp |

**Trading Hours:**
- Monday-Friday: 08:00-12:55 COT (13:00-17:55 UTC)
- Expected bars per day: 60

---

### 3.3 macro_indicators_daily

37-column macroeconomic indicators for model features.

```sql
CREATE TABLE macro_indicators_daily (
    fecha DATE PRIMARY KEY,
    -- OR: date DATE PRIMARY KEY,

    -- Commodities (4 columns)
    comm_agri_coffee_glb_d_coffee DECIMAL(10,2),
    comm_metal_gold_glb_d_gold DECIMAL(10,2),
    comm_oil_brent_glb_d_brent DECIMAL(10,2),
    comm_oil_wti_glb_d_wti DECIMAL(10,2),

    -- Country Risk (3 columns)
    crsk_sentiment_cci_col_m_cci DECIMAL(10,2),
    crsk_sentiment_ici_col_m_ici DECIMAL(10,2),
    crsk_spread_embi_col_d_embi DECIMAL(10,2),

    -- Equity (1 column)
    eqty_index_colcap_col_d_colcap DECIMAL(12,2),

    -- Fixed Income (4 columns)
    finc_bond_yield10y_col_d_col10y DECIMAL(6,3),
    finc_bond_yield10y_usa_d_ust10y DECIMAL(6,3),
    finc_bond_yield2y_usa_d_dgs2 DECIMAL(6,3),
    finc_bond_yield5y_col_d_col5y DECIMAL(6,3),

    -- Interest Rates (1 column)
    finc_rate_ibr_overnight_col_d_ibr DECIMAL(6,3),

    -- Foreign Trade (3 columns)
    ftrd_exports_total_col_m_expusd DECIMAL(12,2),
    ftrd_imports_total_col_m_impusd DECIMAL(12,2),
    ftrd_terms_trade_col_m_tot DECIMAL(10,2),

    -- FX Rates (4 columns)
    fxrt_index_dxy_usa_d_dxy DECIMAL(8,4),
    fxrt_reer_bilateral_col_m_itcr DECIMAL(10,4),
    fxrt_spot_usdclp_chl_d_usdclp DECIMAL(10,2),
    fxrt_spot_usdmxn_mex_d_usdmxn DECIMAL(10,4),

    -- GDP (1 column)
    gdpp_real_gdp_usa_q_gdp_q DECIMAL(12,2),

    -- Inflation (4 columns)
    infl_cpi_all_usa_m_cpiaucsl DECIMAL(10,3),
    infl_cpi_core_usa_m_cpilfesl DECIMAL(10,3),
    infl_cpi_total_col_m_ipccol DECIMAL(10,2),
    infl_pce_usa_m_pcepi DECIMAL(10,3),

    -- Labor (1 column)
    labr_unemployment_usa_m_unrate DECIMAL(6,2),

    -- Money Supply (1 column)
    mnys_m2_supply_usa_m_m2sl DECIMAL(14,2),

    -- Policy Rates (4 columns)
    polr_fed_funds_usa_m_fedfunds DECIMAL(6,3),
    polr_policy_rate_col_d_tpm DECIMAL(6,3),
    polr_policy_rate_col_m_tpm DECIMAL(6,3),
    polr_prime_rate_usa_d_prime DECIMAL(6,3),

    -- Production (1 column)
    prod_industrial_usa_m_indpro DECIMAL(10,3),

    -- Balance of Payments (4 columns)
    rsbp_current_account_col_q_cacct_q DECIMAL(12,2),
    rsbp_fdi_inflow_col_q_fdiin_q DECIMAL(12,2),
    rsbp_fdi_outflow_col_q_fdiout_q DECIMAL(12,2),
    rsbp_reserves_international_col_m_resint DECIMAL(14,2),

    -- Sentiment (1 column)
    sent_consumer_usa_m_umcsent DECIMAL(8,2),

    -- Volatility (1 column)
    volt_vix_usa_d_vix DECIMAL(8,2),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Key Features Used in RL Model (7 columns):**

| Column | Feature Name | Normalization |
|--------|--------------|---------------|
| fxrt_index_dxy_usa_d_dxy | dxy_z | z-score (mean=100.21, std=5.60) |
| volt_vix_usa_d_vix | vix_z | z-score (mean=21.16, std=7.89) |
| crsk_spread_embi_col_d_embi | embi_z | z-score (mean=322.01, std=62.68) |
| fxrt_spot_usdmxn_mex_d_usdmxn | usdmxn_change_1d | daily pct change |
| comm_oil_wti_glb_d_wti | brent_change_1d | daily pct change |
| finc_bond_yield10y_usa_d_ust10y | rate_spread | 10Y-2Y normalized |
| polr_fed_funds_usa_m_fedfunds | (part of rate spread) | |

---

### 3.4 trading_metrics (TimescaleDB Hypertable)

Performance metrics time series.

```sql
CREATE TABLE trading_metrics (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(15,6),
    metric_type TEXT NOT NULL,
    strategy_name TEXT,
    model_version TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, metric_name, metric_type)
);

SELECT create_hypertable('trading_metrics', 'timestamp', if_not_exists => TRUE);
```

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMPTZ | PK - Metric timestamp |
| metric_name | TEXT | PK - Metric identifier |
| metric_value | DECIMAL(15,6) | Metric value |
| metric_type | TEXT | PK - Category (performance, risk, model_accuracy) |
| strategy_name | TEXT | Strategy identifier |
| model_version | TEXT | Model version |
| metadata | JSONB | Additional data |

---

### 3.5 trading_sessions

User trading session tracking.

```sql
CREATE TABLE trading_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
    session_end TIMESTAMP WITH TIME ZONE,
    strategy_name TEXT,
    initial_balance DECIMAL(15,2),
    final_balance DECIMAL(15,2),
    total_trades INTEGER DEFAULT 0,
    profitable_trades INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Schema: dw (Data Warehouse)

### 4.1 dim_strategy

Strategy dimension table for multi-model trading.

```sql
CREATE TABLE dw.dim_strategy (
    strategy_id SERIAL PRIMARY KEY,
    strategy_code VARCHAR(50) UNIQUE NOT NULL,
    strategy_name VARCHAR(200) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL CHECK (strategy_type IN ('RL', 'ML', 'LLM', 'ENSEMBLE')),
    description TEXT,
    parameters JSONB,
    initial_equity DECIMAL(15,2) DEFAULT 10000.00,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Default Strategies:**

| strategy_code | strategy_name | strategy_type |
|---------------|---------------|---------------|
| RL_PPO | PPO Reinforcement Learning | RL |
| ML_XGB | XGBoost Classifier | ML |
| ML_LGBM | LightGBM Model | ML |
| LLM_CLAUDE | Claude LLM Signals | LLM |
| ENSEMBLE | Weighted Ensemble | ENSEMBLE |

---

### 4.2 fact_strategy_signals

Trading signals generated by each strategy.

```sql
CREATE TABLE dw.fact_strategy_signals (
    signal_id BIGSERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES dw.dim_strategy(strategy_id),
    timestamp_utc TIMESTAMPTZ NOT NULL,
    signal VARCHAR(20) NOT NULL CHECK (signal IN ('long', 'short', 'flat', 'close')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell', 'hold')),
    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    size DECIMAL(5,4) CHECK (size >= 0 AND size <= 1),
    entry_price DECIMAL(12,6),
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    risk_usd DECIMAL(12,2),
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fact_signals_strategy_time ON dw.fact_strategy_signals(strategy_id, timestamp_utc DESC);
```

---

### 4.3 fact_equity_curve

Equity curve time series per strategy.

```sql
CREATE TABLE dw.fact_equity_curve (
    equity_id BIGSERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES dw.dim_strategy(strategy_id),
    timestamp_utc TIMESTAMPTZ NOT NULL,
    equity_value DECIMAL(15,2) NOT NULL,
    return_since_start_pct DECIMAL(10,4),
    daily_return_pct DECIMAL(10,4),
    current_drawdown_pct DECIMAL(10,4),
    peak_equity DECIMAL(15,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fact_equity_strategy_time ON dw.fact_equity_curve(strategy_id, timestamp_utc DESC);
```

---

### 4.4 fact_strategy_performance

Daily performance metrics per strategy.

```sql
CREATE TABLE dw.fact_strategy_performance (
    perf_id BIGSERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES dw.dim_strategy(strategy_id),
    date_cot DATE NOT NULL,
    daily_return_pct DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    max_drawdown_pct DECIMAL(10,4),
    n_trades INTEGER DEFAULT 0,
    n_wins INTEGER DEFAULT 0,
    n_losses INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    gross_profit DECIMAL(15,2),
    gross_loss DECIMAL(15,2),
    net_profit DECIMAL(15,2),
    total_fees DECIMAL(15,2),
    avg_hold_time_minutes DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(strategy_id, date_cot)
);

CREATE INDEX idx_fact_perf_strategy_date ON dw.fact_strategy_performance(strategy_id, date_cot DESC);
```

---

### 4.5 fact_strategy_positions

Open and closed positions.

```sql
CREATE TABLE dw.fact_strategy_positions (
    position_id BIGSERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES dw.dim_strategy(strategy_id),
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    quantity DECIMAL(15,6) NOT NULL,
    entry_price DECIMAL(12,6) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DECIMAL(12,6),
    exit_time TIMESTAMPTZ,
    stop_loss DECIMAL(12,6),
    take_profit DECIMAL(12,6),
    leverage INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'cancelled')),
    realized_pnl DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    fees DECIMAL(12,4),
    close_reason VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fact_pos_strategy_status ON dw.fact_strategy_positions(strategy_id, status);
CREATE INDEX idx_fact_pos_entry_time ON dw.fact_strategy_positions(entry_time DESC);
```

---

### 4.6 fact_rl_inference (TimescaleDB Hypertable)

RL model inference log with full observation vector.

```sql
CREATE TABLE dw.fact_rl_inference (
    inference_id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    fold_id INT,

    -- 13 Feature Columns (Observation Vector)
    log_ret_5m FLOAT,
    log_ret_1h FLOAT,
    log_ret_4h FLOAT,
    rsi_9 FLOAT,
    atr_pct FLOAT,
    adx_14 FLOAT,
    dxy_z FLOAT,
    dxy_change_1d FLOAT,
    vix_z FLOAT,
    embi_z FLOAT,
    brent_change_1d FLOAT,
    rate_spread FLOAT,
    usdmxn_ret_1h FLOAT,

    -- State Features
    position FLOAT,
    time_normalized FLOAT,

    -- Model Output
    action_raw FLOAT NOT NULL,
    action_discretized VARCHAR(10) NOT NULL CHECK (action_discretized IN ('LONG', 'SHORT', 'HOLD')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    q_values FLOAT[],

    -- Market Context
    symbol VARCHAR(20) DEFAULT 'USD/COP',
    close_price DECIMAL(12,6) NOT NULL,
    raw_return_5m FLOAT,
    spread_bps DECIMAL(8,4),

    -- Portfolio State
    position_before FLOAT NOT NULL CHECK (position_before >= -1 AND position_before <= 1),
    portfolio_value_before DECIMAL(15,2),
    log_portfolio_before FLOAT,
    position_after FLOAT NOT NULL CHECK (position_after >= -1 AND position_after <= 1),
    portfolio_value_after DECIMAL(15,2),
    log_portfolio_after FLOAT,

    -- Transaction Costs
    position_change FLOAT,
    transaction_cost_bps DECIMAL(8,4),
    transaction_cost_usd DECIMAL(12,4),

    -- Performance
    reward FLOAT,
    cumulative_reward FLOAT,

    -- Metadata
    latency_ms INT,
    inference_source VARCHAR(50) DEFAULT 'airflow',
    dag_run_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('dw.fact_rl_inference', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX idx_rl_inference_model_time ON dw.fact_rl_inference(model_id, timestamp_utc DESC);
CREATE INDEX idx_rl_inference_action ON dw.fact_rl_inference(action_discretized);
```

---

### 4.7 fact_agent_actions

Simplified agent action log.

```sql
CREATE TABLE dw.fact_agent_actions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bar_number INTEGER,
    action DOUBLE PRECISION,
    position DOUBLE PRECISION DEFAULT 0.0,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_actions_time ON dw.fact_agent_actions(timestamp DESC);
```

---

## 5. Views and Materialized Views

### 5.1 latest_ohlcv

Most recent price for each symbol.

```sql
CREATE OR REPLACE VIEW latest_ohlcv AS
SELECT DISTINCT ON (symbol)
    symbol,
    time as timestamp,
    open, high, low, close, volume,
    source, created_at
FROM usdcop_m5_ohlcv
ORDER BY symbol, time DESC;
```

### 5.2 inference_features_5m (Materialized)

Pre-calculated inference features.

```sql
CREATE MATERIALIZED VIEW inference_features_5m AS
WITH ohlcv_with_returns AS (
    SELECT
        o.time,
        o.close,
        LEAST(GREATEST(LN(o.close / NULLIF(LAG(o.close, 1) OVER (ORDER BY o.time), 0)), -0.05), 0.05) AS log_ret_5m,
        LEAST(GREATEST(LN(o.close / NULLIF(LAG(o.close, 12) OVER (ORDER BY o.time), 0)), -0.05), 0.05) AS log_ret_1h,
        LEAST(GREATEST(LN(o.close / NULLIF(LAG(o.close, 48) OVER (ORDER BY o.time), 0)), -0.05), 0.05) AS log_ret_4h
    FROM usdcop_m5_ohlcv o
    WHERE o.symbol = 'USD/COP'
),
macro_with_changes AS (
    SELECT
        date,
        LEAST(GREATEST((dxy - 100.21) / 5.60, -4.0), 4.0) AS dxy_z,
        LEAST(GREATEST((dxy - LAG(dxy, 1) OVER (ORDER BY date)) / NULLIF(LAG(dxy, 1) OVER (ORDER BY date), 0), -0.03), 0.03) AS dxy_change_1d,
        LEAST(GREATEST((vix - 21.16) / 7.89, -4.0), 4.0) AS vix_z,
        LEAST(GREATEST((embi - 322.01) / 62.68, -4.0), 4.0) AS embi_z,
        LEAST(GREATEST((brent - LAG(brent, 1) OVER (ORDER BY date)) / NULLIF(LAG(brent, 1) OVER (ORDER BY date), 0), -0.10), 0.10) AS brent_change_1d,
        LEAST(GREATEST((10.0 - treasury_10y - 7.03) / 1.41, -4.0), 4.0) AS rate_spread
    FROM macro_indicators_daily
)
SELECT
    r.time,
    r.log_ret_5m,
    r.log_ret_1h,
    r.log_ret_4h,
    m.dxy_z,
    m.dxy_change_1d,
    m.vix_z,
    m.embi_z,
    m.brent_change_1d,
    m.rate_spread
FROM ohlcv_with_returns r
LEFT JOIN macro_with_changes m ON DATE(r.time) = m.date
WHERE r.time >= NOW() - INTERVAL '7 days'
ORDER BY r.time DESC;

CREATE UNIQUE INDEX idx_inference_features_time ON inference_features_5m (time);
```

### 5.3 inference_features_complete

Complete 13-feature view joining SQL and Python features.

```sql
CREATE OR REPLACE VIEW inference_features_complete AS
SELECT
    s.time,
    s.log_ret_5m,
    s.log_ret_1h,
    s.log_ret_4h,
    COALESCE(p.rsi_9, 50.0) AS rsi_9,
    COALESCE(p.atr_pct, 0.05) AS atr_pct,
    COALESCE(p.adx_14, 25.0) AS adx_14,
    s.dxy_z,
    s.dxy_change_1d,
    s.vix_z,
    s.embi_z,
    s.brent_change_1d,
    s.rate_spread,
    COALESCE(p.usdmxn_change_1d, 0.0) AS usdmxn_change_1d
FROM inference_features_5m s
LEFT JOIN python_features_5m p ON s.time = p.time;
```

---

## 6. Functions

### 6.1 is_market_open()

Check if market is currently open.

```sql
CREATE OR REPLACE FUNCTION is_market_open()
RETURNS BOOLEAN AS $$
DECLARE
    now_cot TIMESTAMP;
    hour_cot INTEGER;
    minute_cot INTEGER;
    dow_cot INTEGER;
BEGIN
    now_cot := NOW() AT TIME ZONE 'America/Bogota';
    hour_cot := EXTRACT(HOUR FROM now_cot);
    minute_cot := EXTRACT(MINUTE FROM now_cot);
    dow_cot := EXTRACT(DOW FROM now_cot);

    RETURN dow_cot BETWEEN 1 AND 5
           AND ((hour_cot >= 8 AND hour_cot < 12)
                OR (hour_cot = 12 AND minute_cot <= 55));
END;
$$ LANGUAGE plpgsql;
```

### 6.2 get_bar_number()

Calculate bar number (1-60) for a timestamp.

```sql
CREATE OR REPLACE FUNCTION get_bar_number(ts TIMESTAMPTZ DEFAULT NOW())
RETURNS INTEGER AS $$
DECLARE
    ts_cot TIMESTAMP;
    minutes_since_open INTEGER;
BEGIN
    ts_cot := ts AT TIME ZONE 'America/Bogota';
    minutes_since_open := (EXTRACT(HOUR FROM ts_cot) - 8) * 60 + EXTRACT(MINUTE FROM ts_cot);
    RETURN GREATEST(1, LEAST(60, (minutes_since_open / 5) + 1));
END;
$$ LANGUAGE plpgsql;
```

### 6.3 refresh_inference_features()

Refresh materialized view.

```sql
CREATE OR REPLACE FUNCTION refresh_inference_features()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. Indexes

### 7.1 OHLCV Indexes

```sql
CREATE INDEX idx_usdcop_m5_ohlcv_symbol_time ON usdcop_m5_ohlcv (symbol, time DESC);
CREATE INDEX idx_usdcop_m5_ohlcv_time ON usdcop_m5_ohlcv (time DESC);
CREATE INDEX idx_usdcop_m5_ohlcv_source ON usdcop_m5_ohlcv (source);
CREATE INDEX idx_usdcop_m5_ohlcv_close ON usdcop_m5_ohlcv (close);
```

### 7.2 Macro Indexes

```sql
CREATE INDEX idx_macro_daily_fecha ON macro_indicators_daily (fecha DESC);
CREATE INDEX idx_macro_daily_dxy ON macro_indicators_daily (fxrt_index_dxy_usa_d_dxy);
CREATE INDEX idx_macro_daily_vix ON macro_indicators_daily (volt_vix_usa_d_vix);
CREATE INDEX idx_macro_daily_embi ON macro_indicators_daily (crsk_spread_embi_col_d_embi);
```

### 7.3 DW Schema Indexes

```sql
-- Signals
CREATE INDEX idx_fact_signals_strategy_time ON dw.fact_strategy_signals(strategy_id, timestamp_utc DESC);

-- Equity
CREATE INDEX idx_fact_equity_strategy_time ON dw.fact_equity_curve(strategy_id, timestamp_utc DESC);

-- Performance
CREATE INDEX idx_fact_perf_strategy_date ON dw.fact_strategy_performance(strategy_id, date_cot DESC);

-- Positions
CREATE INDEX idx_fact_pos_strategy_status ON dw.fact_strategy_positions(strategy_id, status);
CREATE INDEX idx_fact_pos_entry_time ON dw.fact_strategy_positions(entry_time DESC);

-- RL Inference
CREATE INDEX idx_rl_inference_model_time ON dw.fact_rl_inference(model_id, timestamp_utc DESC);
CREATE INDEX idx_rl_inference_action ON dw.fact_rl_inference(action_discretized);
```

---

## 8. Sample Queries

### 8.1 Get Latest Price

```sql
SELECT * FROM latest_ohlcv WHERE symbol = 'USD/COP';
```

### 8.2 Get Today's OHLCV Bars

```sql
SELECT time, open, high, low, close, volume
FROM usdcop_m5_ohlcv
WHERE symbol = 'USD/COP'
  AND time >= CURRENT_DATE
ORDER BY time;
```

### 8.3 Get Latest Signals by Strategy

```sql
SELECT DISTINCT ON (ds.strategy_code)
    ds.strategy_code,
    ds.strategy_name,
    fs.signal,
    fs.side,
    fs.confidence,
    fs.timestamp_utc
FROM dw.fact_strategy_signals fs
JOIN dw.dim_strategy ds ON fs.strategy_id = ds.strategy_id
WHERE ds.is_active = TRUE
ORDER BY ds.strategy_code, fs.timestamp_utc DESC;
```

### 8.4 Get Strategy Performance Comparison

```sql
SELECT
    ds.strategy_code,
    ds.strategy_type,
    AVG(sp.sharpe_ratio) as avg_sharpe,
    AVG(sp.win_rate) as avg_win_rate,
    SUM(sp.net_profit) as total_profit,
    MAX(ABS(sp.max_drawdown_pct)) as worst_drawdown
FROM dw.fact_strategy_performance sp
JOIN dw.dim_strategy ds ON sp.strategy_id = ds.strategy_id
WHERE sp.date_cot >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ds.strategy_code, ds.strategy_type
ORDER BY avg_sharpe DESC;
```

### 8.5 Get Open Positions

```sql
SELECT
    ds.strategy_code,
    fp.side,
    fp.quantity,
    fp.entry_price,
    fp.unrealized_pnl,
    fp.entry_time,
    EXTRACT(EPOCH FROM (NOW() - fp.entry_time)) / 60 as holding_minutes
FROM dw.fact_strategy_positions fp
JOIN dw.dim_strategy ds ON fp.strategy_id = ds.strategy_id
WHERE fp.status = 'open'
ORDER BY fp.entry_time;
```

### 8.6 Get Inference Features for Current Bar

```sql
SELECT * FROM get_latest_features();
```

### 8.7 Calculate Daily PnL by Strategy

```sql
WITH daily_pnl AS (
    SELECT
        strategy_id,
        date_cot,
        net_profit,
        SUM(net_profit) OVER (PARTITION BY strategy_id ORDER BY date_cot) as cumulative_pnl
    FROM dw.fact_strategy_performance
    WHERE date_cot >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT
    ds.strategy_code,
    dp.date_cot,
    dp.net_profit,
    dp.cumulative_pnl
FROM daily_pnl dp
JOIN dw.dim_strategy ds ON dp.strategy_id = ds.strategy_id
ORDER BY ds.strategy_code, dp.date_cot;
```

---

## 9. Maintenance

### 9.1 Vacuum and Analyze

```sql
-- Run weekly
VACUUM ANALYZE usdcop_m5_ohlcv;
VACUUM ANALYZE macro_indicators_daily;
VACUUM ANALYZE dw.fact_strategy_signals;
VACUUM ANALYZE dw.fact_equity_curve;
VACUUM ANALYZE dw.fact_rl_inference;
```

### 9.2 Refresh Materialized Views

```sql
-- Run every 5 minutes during market hours
SELECT refresh_inference_features();
-- Or directly:
REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m;
```

### 9.3 Check Hypertable Status

```sql
SELECT
    hypertable_name,
    num_chunks,
    pg_size_pretty(hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass)) as size
FROM timescaledb_information.hypertables;
```

### 9.4 Data Retention Policy

```sql
-- Drop chunks older than 1 year
SELECT drop_chunks('usdcop_m5_ohlcv', older_than => INTERVAL '1 year');
SELECT drop_chunks('dw.fact_rl_inference', older_than => INTERVAL '1 year');
```

### 9.5 Check Index Usage

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname IN ('public', 'dw')
ORDER BY idx_scan DESC;
```

---

## Related Documentation

- [Multi-Model Backend](./MULTI_MODEL_BACKEND.md)
- [API Endpoints Reference](./API_ENDPOINTS_MULTIMODEL.md)
- [Architecture Overview V3](./ARQUITECTURA_INTEGRAL_V3.md)
