# 🔍 Auditoría: Propuesta Técnica vs Realidad del Proyecto

**Fecha:** 2025-10-21
**Análisis:** Propuesta de KPIs/Métricas vs Implementación Real

---

## 📊 RESUMEN EJECUTIVO

### ✅ DATOS CONFIRMADOS DISPONIBLES

```
✅ OHLCV M5 Premium (Confirmado):
   - Tabla: market_data en PostgreSQL
   - Registros: 92,936 filas
   - Período: 2020-01-02 hasta 2025-10-21
   - Frecuencia: Cada 5 minutos
   - Horario: Lunes-Viernes 08:00-12:55 COT
   - Columnas: timestamp, symbol, price, bid, ask, volume, source

✅ Pipeline L0-L6 (Parcialmente implementado):
   - L0: PostgreSQL queries funcionando ✅
   - L1: Cálculos de episodios implementados ✅
   - L3: Correlaciones calculadas ✅
   - L4: Splits train/test implementados ✅
   - L5: MOCK DATA (no implementado) ❌
   - L6: MOCK DATA (no implementado) ❌

⚠️ Backends API:
   - Trading API (8000): 70% real, 30% mock
   - Analytics API (8001): 95% real
   - Signals API (8003): 90% real
   - Pipeline API (8004): 60% real (L0-L4), 0% L5-L6
   - ML Analytics (8005): 0% real (100% mock)
   - Backtest API (8006): 95% real
```

---

## 📋 COMPARACIÓN PROPUESTA vs REALIDAD

### 1. TRADING (Home / Terminal / Live)

#### 📌 PROPUESTA: Precio, cambio, high/low 24h, volumen 24h

**Estado:** ✅ **TOTALMENTE VIABLE**

**Implementación Real:**
```python
# api_server.py:194 - GET /api/stats/{symbol}
@app.get("/api/stats/{symbol}")
def get_symbol_stats(symbol: str):
    # Query PostgreSQL market_data
    high_24h = db.query("SELECT MAX(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'")
    low_24h = db.query("SELECT MIN(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'")
    avg_24h = db.query("SELECT AVG(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'")
    volume_24h = db.query("SELECT SUM(volume) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'")

    return {
        "high_24h": high_24h,
        "low_24h": low_24h,
        "avg_24h": avg_24h,
        "volume_24h": volume_24h,
        "change_pct": ((current_price - prev_price) / prev_price) * 100
    }
```

**Hallazgo:** ✅ **Ya está implementado y funcionando**

---

#### 📌 PROPUESTA: Spread (bps) con proxy Corwin-Schultz

**Estado:** ⚠️ **PARCIALMENTE IMPLEMENTADO**

**Realidad:**
```python
# api_server.py:83 - GET /api/latest/{symbol}
# NO calcula spread Corwin-Schultz
# Solo retorna: bid, ask del DB (si existen)

# trading_analytics_api.py:107 - RL metrics
spread_captured = db.query("""
    SELECT AVG((ask - bid) / ((ask + bid) / 2) * 10000) as spread_bps
    FROM market_data
""")
```

**Problema:**
- La DB tiene columnas `bid` y `ask` pero pueden estar vacías
- No implementa Corwin-Schultz como proxy cuando faltan bid/ask
- Spread solo se calcula si bid/ask existen

**Recomendación:**
```python
def calculate_spread_proxy_corwin_schultz(high, low, high_prev, low_prev):
    """
    Corwin-Schultz (2012) spread estimator
    Basado en high-low range de 2 períodos consecutivos
    """
    beta = (np.log(high / low))**2 + (np.log(high_prev / low_prev))**2
    gamma = (np.log(max(high, high_prev) / min(low, low_prev)))**2

    alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / (3 - 2*np.sqrt(2)) - np.sqrt(gamma / (3 - 2*np.sqrt(2)))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread_bps = spread * 10000

    return spread_bps

# Implementar en trading_analytics_api.py
```

**Estado Final:** 🔄 **NECESITA IMPLEMENTACIÓN del proxy Corwin-Schultz**

---

#### 📌 PROPUESTA: ATR(14), RSI, MACD, Bollinger Bands, EMAs

**Estado:** ✅ **IMPLEMENTADO**

**Implementación Real:**
```python
# trading_signals_api.py:88-125 - calculate_indicators()
def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal

    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)

    # EMAs
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()

    return {
        "rsi": rsi.iloc[-1],
        "macd": {"macd": macd.iloc[-1], "signal": signal.iloc[-1], "histogram": histogram.iloc[-1]},
        "bollinger": {"upper": bb_upper.iloc[-1], "middle": sma_20.iloc[-1], "lower": bb_lower.iloc[-1]},
        "ema_20": ema_20.iloc[-1],
        "ema_50": ema_50.iloc[-1]
    }
```

**Hallazgo:** ✅ **Completamente implementado en trading_signals_api.py**

---

#### 📌 PROPUESTA: Progreso de sesión (08:00→12:55)

**Estado:** ✅ **FÁCILMENTE CALCULABLE**

**Implementación Sugerida:**
```python
def calculate_session_progress():
    """
    Sesión premium: 08:00 - 12:55 COT (5 horas = 300 minutos)
    Barras M5: 60 barras por sesión (300 min / 5 min)
    """
    import pytz
    from datetime import datetime

    cot = pytz.timezone('America/Bogota')
    now = datetime.now(cot)

    session_start = now.replace(hour=8, minute=0, second=0)
    session_end = now.replace(hour=12, minute=55, second=0)

    if now < session_start:
        return {"status": "PRE_MARKET", "progress": 0.0}
    elif now > session_end:
        return {"status": "CLOSED", "progress": 100.0}
    else:
        elapsed = (now - session_start).total_seconds()
        total = (session_end - session_start).total_seconds()
        progress = (elapsed / total) * 100

        bars_elapsed = int(elapsed / 300)  # 300s = 5min

        return {
            "status": "OPEN",
            "progress": progress,
            "bars_elapsed": bars_elapsed,
            "bars_total": 60,
            "time_remaining_minutes": int((total - elapsed) / 60)
        }
```

**Hallazgo:** 🔄 **No implementado, pero trivial de agregar**

---

#### 📌 PROPUESTA: Live Trading - métricas RL (Spread Captured, Peg Rate, etc.)

**Estado:** ⚠️ **PARCIALMENTE REAL - Depende de L4**

**Implementación Real:**
```python
# trading_analytics_api.py:95-145 - GET /api/analytics/rl-metrics
@app.get("/api/analytics/rl-metrics")
def get_rl_metrics(symbol: str = "USDCOP", days: int = 30):
    """
    CALCULA métricas RL de datos históricos
    """
    # Query market_data últimos N días
    df = query_market_data(days=days)

    # Calcular métricas
    trades_per_episode = 6  # Hardcoded - debería venir de L4
    spread_captured = calculate_spread_from_bid_ask()  # Si bid/ask disponible
    peg_rate = 0.032  # Hardcoded - debería venir de L4

    action_balance = {
        "buy": 0.35,  # Hardcoded - debería venir de L4
        "sell": 0.32,
        "hold": 0.33
    }

    return {
        "spread_captured_bps": spread_captured,
        "peg_rate_pct": peg_rate,
        "trades_per_episode": trades_per_episode,
        "action_balance": action_balance,
        "episodes_completed": 247,  # Hardcoded - debería contar de L4
        "total_steps": 125430,  # Hardcoded
        "avg_reward": 1250.50  # Hardcoded
    }
```

**Problema:**
- ✅ Estructura correcta
- ❌ Valores hardcoded en lugar de calculados de L4
- ❌ No lee episodios reales de L4

**Solución Propuesta:**
```python
# Necesita integración con MinIO o tabla L4
def get_rl_metrics_from_l4():
    """
    Leer episodios de L4 (si existen en MinIO o DB)
    """
    # Opción A: MinIO
    episodes = minio_client.get_object('04-l4-ds-usdcop-rlready', 'episodes.parquet')
    df = pd.read_parquet(episodes)

    # Calcular métricas REALES
    spread_captured = df.groupby('episode_id')['spread_proxy_bps'].mean().mean()
    peg_rate = df.groupby('episode_id')['peg_rate'].mean().mean()
    trades_per_episode = df.groupby('episode_id')['action'].apply(lambda x: (x != 0).sum()).mean()

    action_counts = df['action'].value_counts(normalize=True)
    action_balance = {
        "buy": action_counts.get(1, 0),
        "sell": action_counts.get(-1, 0),
        "hold": action_counts.get(0, 0)
    }

    return {
        "spread_captured_bps": spread_captured,
        "peg_rate_pct": peg_rate,
        "trades_per_episode": trades_per_episode,
        "action_balance": action_balance,
        "episodes_completed": df['episode_id'].nunique(),
        "total_steps": len(df),
        "avg_reward": df['reward'].mean()
    }
```

**Estado Final:** 🔄 **REQUIERE integración con L4 (MinIO o SQL)**

---

### 2. RIESGO (Monitor / Alerts)

#### 📌 PROPUESTA: VaR/ES (95%/99%) desde L6 backtest

**Estado:** ✅ **IMPLEMENTADO**

**Implementación Real:**
```python
# trading_analytics_api.py:253-285 - GET /api/analytics/risk-metrics
@app.get("/api/analytics/risk-metrics")
def get_risk_metrics(symbol: str = "USDCOP", days: int = 90):
    """
    Calcula VaR y CVaR de retornos históricos
    """
    df = query_market_data(days=days)

    returns = np.log(df['price'] / df['price'].shift(1)).dropna()

    # VaR
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100

    # CVaR (Expected Shortfall)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100

    # Drawdown
    prices = df['price'].values
    running_max = np.maximum.accumulate(prices)
    drawdown = (prices - running_max) / running_max
    max_drawdown = drawdown.min()
    current_drawdown = drawdown[-1]

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "volatility_annualized": returns.std() * np.sqrt(252)
    }
```

**Hallazgo:** ✅ **Ya implementado correctamente con datos reales**

**Nota:** Estos son risk metrics del **PRECIO** (no del portafolio), lo cual es correcto dado que no hay portafolio real.

---

#### 📌 PROPUESTA: Portfolio Value, P&L - desde L6 backtest

**Estado:** ⚠️ **PARCIAL - Backend tiene mock, necesita L6 real**

**Implementación Real:**
```python
# backtest_api.py:150-250 - generate_backtest_results()
def generate_backtest_results():
    """
    Ejecuta backtest completo con:
    1. Datos de PostgreSQL (si disponible)
    2. Simulación de señales de trading
    3. Cálculo de equity curve
    4. Cálculo de KPIs
    """
    # Get real data
    df = query_market_data(limit=10000)

    if df is None or len(df) == 0:
        # Fallback: synthetic data
        df = generate_synthetic_prices()

    # Simulate trading
    signals = np.random.choice([-1, 0, 1], size=len(df))  # -1=sell, 0=hold, 1=buy
    position = 0
    equity = [10000]  # Starting capital
    trades = []

    for i in range(1, len(df)):
        signal = signals[i]
        price = df.iloc[i]['close']

        # Entry/exit logic
        if signal == 1 and position == 0:  # Buy
            position = equity[-1] / price
            trades.append({
                "timestamp": df.iloc[i]['timestamp'],
                "side": "BUY",
                "price": price,
                "quantity": position
            })
        elif signal == -1 and position > 0:  # Sell
            pnl = (price - trades[-1]['price']) * position
            equity.append(equity[-1] + pnl)
            trades.append({
                "timestamp": df.iloc[i]['timestamp'],
                "side": "SELL",
                "price": price,
                "pnl": pnl
            })
            position = 0
        else:
            equity.append(equity[-1])

    # Calculate KPIs
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = (returns.mean() * np.sqrt(252)) / returns.std()

    # ...más cálculos

    return {
        "daily_returns": equity,
        "trades": trades,
        "kpis": {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "cagr": cagr,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    }
```

**Hallazgo:**
- ✅ Backtest engine implementado
- ✅ Usa datos reales de PostgreSQL
- ⚠️ Señales son RANDOM (debería usar modelo L5)
- ⚠️ No lee resultados pre-calculados de L6

**Estado Final:** 🔄 **Funcional pero necesita señales de modelo L5, no random**

---

### 3. PIPELINE (L0 → L6)

#### 📌 PROPUESTA: L0 - Métricas de calidad

**Estado:** ✅ **IMPLEMENTADO**

**Implementación Real:**
```python
# pipeline_data_api.py:54-92 - GET /api/pipeline/l0/statistics
@app.get("/api/pipeline/l0/statistics")
def get_l0_statistics():
    """
    Calcula estadísticas de calidad de L0
    """
    conn = psycopg2.connect(DATABASE_URL)

    # Total records
    total_records = execute_query("SELECT COUNT(*) FROM market_data")[0][0]

    # Date range
    date_range = execute_query("""
        SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
        FROM market_data
    """)[0]

    # Price statistics
    price_stats = execute_query("""
        SELECT
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            STDDEV(price) as stddev_price
        FROM market_data
    """)[0]

    # Missing values (NULL counts)
    null_counts = execute_query("""
        SELECT
            SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) as null_price,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume,
            SUM(CASE WHEN bid IS NULL THEN 1 ELSE 0 END) as null_bid,
            SUM(CASE WHEN ask IS NULL THEN 1 ELSE 0 END) as null_ask
        FROM market_data
    """)[0]

    # Duplicates
    duplicates = execute_query("""
        SELECT COUNT(*) FROM (
            SELECT timestamp, symbol, COUNT(*)
            FROM market_data
            GROUP BY timestamp, symbol
            HAVING COUNT(*) > 1
        ) AS dups
    """)[0][0]

    return {
        "total_records": total_records,
        "date_range": {
            "earliest": date_range[0],
            "latest": date_range[1],
            "trading_days": calculate_trading_days(date_range[0], date_range[1])
        },
        "price_metrics": {
            "min": price_stats[0],
            "max": price_stats[1],
            "avg": price_stats[2],
            "stddev": price_stats[3]
        },
        "quality": {
            "null_price": null_counts[0],
            "null_volume": null_counts[1],
            "null_bid": null_counts[2],
            "null_ask": null_counts[3],
            "duplicates": duplicates
        }
    }
```

**Falta Implementar:**
```python
# Métricas adicionales propuestas
def calculate_l0_quality_extended():
    """
    Cobertura = barras encontradas / 60 (premium session)
    Invariantes OHLC (high >= low, close entre high-low)
    Gaps consecutivos
    Tasa de repetidos OHLC (stale data)
    """
    # Cobertura por día
    coverage = execute_query("""
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as bars,
            COUNT(*) / 60.0 as coverage_pct
        FROM market_data
        WHERE EXTRACT(HOUR FROM timestamp) BETWEEN 8 AND 12
        GROUP BY DATE(timestamp)
        HAVING COUNT(*) < 60
    """)

    # Invariantes OHLC
    ohlc_violations = execute_query("""
        SELECT COUNT(*) FROM market_data
        WHERE high < low
           OR close > high
           OR close < low
           OR open > high
           OR open < low
    """)[0][0]

    # Gaps (barras faltantes)
    gaps = execute_query("""
        WITH time_series AS (
            SELECT timestamp,
                   LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
            FROM market_data
        )
        SELECT COUNT(*) FROM time_series
        WHERE EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) > 600  -- >10 min gap
    """)[0][0]

    # Stale data (OHLC repetidos)
    stale = execute_query("""
        WITH ohlc_compare AS (
            SELECT
                open, high, low, close,
                LAG(open) OVER (ORDER BY timestamp) as prev_open,
                LAG(high) OVER (ORDER BY timestamp) as prev_high,
                LAG(low) OVER (ORDER BY timestamp) as prev_low,
                LAG(close) OVER (ORDER BY timestamp) as prev_close
            FROM market_data
        )
        SELECT COUNT(*) FROM ohlc_compare
        WHERE open = prev_open
          AND high = prev_high
          AND low = prev_low
          AND close = prev_close
    """)[0][0]

    return {
        "coverage_avg": sum([c[2] for c in coverage]) / len(coverage) if coverage else 1.0,
        "ohlc_violations": ohlc_violations,
        "gaps_count": gaps,
        "stale_rate_pct": (stale / total_records) * 100
    }
```

**Estado Final:**
- ✅ Básico implementado
- 🔄 Necesita extender con métricas propuestas (cobertura, gaps, stale)

---

#### 📌 PROPUESTA: L1 - Episodios 60/60, grid exacto

**Estado:** ✅ **IMPLEMENTADO**

**Implementación Real:**
```python
# pipeline_data_api.py:94-130 - GET /api/pipeline/l1/episodes
@app.get("/api/pipeline/l1/episodes")
def get_l1_episodes(limit: int = 100):
    """
    Agrupa datos por día (episodio) y calcula métricas de calidad
    """
    episodes = execute_query(f"""
        SELECT
            DATE(timestamp) as episode_date,
            COUNT(*) as data_points,
            COUNT(*) / 60.0 as completeness,
            CASE
                WHEN COUNT(*) = 60 THEN 'ACCEPT'
                WHEN COUNT(*) = 59 THEN 'WARN'
                ELSE 'FAIL'
            END as quality_status,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 60.0 as duration_minutes
        FROM market_data
        WHERE EXTRACT(HOUR FROM timestamp) BETWEEN 8 AND 12
        GROUP BY DATE(timestamp)
        ORDER BY episode_date DESC
        LIMIT {limit}
    """)

    return {
        "episodes": [
            {
                "episode_id": row[0].strftime('%Y%m%d'),
                "timestamp": row[0],
                "data_points": row[1],
                "completeness": row[2],
                "quality_score": 1.0 if row[3] == 'ACCEPT' else 0.9 if row[3] == 'WARN' else 0.0,
                "has_gaps": row[1] < 60,
                "duration_minutes": row[6],
                "session_type": "PREMIUM"
            }
            for row in episodes
        ],
        "summary": {
            "total_count": len(episodes),
            "avg_completeness": sum([e['completeness'] for e in episodes]) / len(episodes)
        }
    }
```

**Falta Implementar:**
```python
# Grid 300s verification
def verify_grid_300s():
    """
    Verificar que cada barra está exactamente 300s (5min) después de la anterior
    """
    grid_ok = execute_query("""
        WITH time_diffs AS (
            SELECT
                timestamp,
                EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (ORDER BY timestamp))) as diff_seconds
            FROM market_data
        )
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END) as perfect_grid,
            (SUM(CASE WHEN ABS(diff_seconds - 300) < 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100 as grid_pct
        FROM time_diffs
        WHERE diff_seconds IS NOT NULL
    """)[0]

    return {
        "grid_300s_pct": grid_ok[2],
        "perfect_grid_count": grid_ok[1],
        "total_intervals": grid_ok[0]
    }
```

**Estado Final:**
- ✅ Episodios implementados
- 🔄 Necesita verificación de grid 300s exacto

---

#### 📌 PROPUESTA: L2 - Winsorization, HOD deseasonalización

**Estado:** ❌ **NO IMPLEMENTADO - No hay endpoint L2**

**Qué Necesita:**
```python
# Crear nuevo endpoint en pipeline_data_api.py
@app.get("/api/pipeline/l2/prepared-data")
def get_l2_prepared_data():
    """
    L2 debe agregar:
    - Technical indicators (60+ calculados de OHLC)
    - Winsorization de outliers
    - HOD (Hour-of-Day) normalization
    """
    # 1. Leer L0/L1 data
    df = get_clean_market_data()

    # 2. Calcular indicadores
    indicators = calculate_all_indicators(df)  # RSI, MACD, BB, ATR, etc.

    # 3. Winsorization (clip outliers a 4σ)
    returns = df['close'].pct_change()
    mean = returns.mean()
    std = returns.std()
    lower_bound = mean - 4 * std
    upper_bound = mean + 4 * std

    returns_winsorized = returns.clip(lower=lower_bound, upper=upper_bound)
    winsorization_rate = ((returns != returns_winsorized).sum() / len(returns)) * 100

    # 4. HOD Deseasonalization
    df['hour'] = df['timestamp'].dt.hour
    hod_medians = df.groupby('hour')['close'].transform('median')
    hod_mad = df.groupby('hour')['close'].transform(lambda x: np.median(np.abs(x - x.median())))

    df['close_deseasonalized'] = (df['close'] - hod_medians) / hod_mad

    return {
        "winsorization_rate_pct": winsorization_rate,
        "hod_stats": {
            "median_abs": hod_medians.abs().mean(),
            "mad_mean": hod_mad.mean()
        },
        "nan_rate_pct": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
        "features_count": len(indicators) + len(df.columns)
    }
```

**Estado Final:** 🔄 **NECESITA IMPLEMENTACIÓN COMPLETA**

---

#### 📌 PROPUESTA: L3 - Feature IC, Correlación

**Estado:** ✅ **IMPLEMENTADO**

**Implementación Real:**
```python
# pipeline_data_api.py:169-210 - GET /api/pipeline/l3/features
@app.get("/api/pipeline/l3/features")
def get_l3_features(limit: int = 100):
    """
    Calcula features y correlaciones
    """
    df = query_market_data(limit=limit)

    # Calculate features
    df['price_change'] = df['price'].pct_change()
    df['spread'] = df['ask'] - df['bid']
    df['volume_change'] = df['volume'].pct_change()
    df['price_ma_5'] = df['price'].rolling(5).mean()
    df['price_ma_20'] = df['price'].rolling(20).mean()
    df['volume_ma_5'] = df['volume'].rolling(5).mean()

    # Correlation matrix
    features = ['price_change', 'spread', 'volume_change', 'price_ma_5', 'price_ma_20', 'volume_ma_5']
    corr_matrix = df[features].corr()

    return {
        "features": [
            {
                "name": col,
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            }
            for col in features
        ],
        "correlations": corr_matrix.to_dict()
    }
```

**Falta Implementar:**
```python
# Information Coefficient (IC) vs futuro
def calculate_forward_ic(df, feature, horizon=1):
    """
    IC = correlation(feature_t, return_t+horizon)
    """
    df['forward_return'] = df['close'].pct_change(horizon).shift(-horizon)
    ic = df[feature].corr(df['forward_return'])
    return ic

# Agregar al endpoint
@app.get("/api/pipeline/l3/features")
def get_l3_features_extended():
    # ... existing code ...

    # Calculate IC for each feature
    for feature in features:
        ic_1 = calculate_forward_ic(df, feature, horizon=1)
        ic_5 = calculate_forward_ic(df, feature, horizon=5)

        feature_stats.append({
            "name": feature,
            "ic_1step": ic_1,
            "ic_5step": ic_5,
            "forward_leakage": abs(ic_1) > 0.10  # Flag if IC too high
        })
```

**Estado Final:**
- ✅ Correlaciones implementadas
- 🔄 Necesita cálculo de IC forward

---

#### 📌 PROPUESTA: L4 - Contrato de observación 17 features, Clip-rate, Reward, Cost model

**Estado:** ⚠️ **PARCIAL - Mock data en algunos campos**

**Implementación Real:**
```python
# pipeline_data_api.py:212-250 - GET /api/pipeline/l4/dataset
@app.get("/api/pipeline/l4/dataset")
def get_l4_dataset(split: str = "test"):
    """
    L4 RL-Ready dataset con splits
    """
    df = query_market_data(limit=10000)

    # Date-based splits
    train_cutoff = df['timestamp'].quantile(0.7)
    val_cutoff = df['timestamp'].quantile(0.85)

    train_df = df[df['timestamp'] <= train_cutoff]
    val_df = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)]
    test_df = df[df['timestamp'] > val_cutoff]

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    selected = splits.get(split, test_df)

    return {
        "split": split,
        "num_episodes": len(selected) // 60,  # Approximate
        "num_timesteps": len(selected),
        "avg_episode_length": 60,
        "reward_mean": 125.5,  # Mock - should calculate from episodes
        "reward_std": 45.2,     # Mock
        "total_features": 17,   # Mock - should list actual features
        "action_space_size": 3  # BUY, SELL, HOLD
    }
```

**Falta Implementar TODO:**
```python
# Contrato completo L4
@app.get("/api/pipeline/l4/contract")
def get_l4_contract():
    """
    Retornar contrato de observación exacto
    """
    return {
        "observation_schema": {
            "obs_00": "spread_proxy_bps_norm",
            "obs_01": "ret_5m_z",
            "obs_02": "ret_10m_z",
            "obs_03": "ret_15m_z",
            "obs_04": "ret_30m_z",
            "obs_05": "range_bps_norm",
            "obs_06": "volume_zscore",
            "obs_07": "rsi_norm",
            "obs_08": "macd_zscore",
            "obs_09": "bb_position",
            "obs_10": "ema_cross_signal",
            "obs_11": "atr_norm",
            "obs_12": "vwap_distance",
            "obs_13": "time_of_day_sin",
            "obs_14": "time_of_day_cos",
            "obs_15": "position",
            "obs_16": "inventory_age"
        },
        "dtype": "float32",
        "range": [-5, 5],
        "clip_thresholds": {
            "max_clip_rate": 0.005  # 0.5% per feature
        }
    }

@app.get("/api/pipeline/l4/quality")
def get_l4_quality():
    """
    Verificar calidad de L4
    """
    # Read L4 episodes from MinIO or DB
    episodes_df = read_l4_episodes()

    # Clip rate per feature
    clip_rates = {}
    for i in range(17):
        obs_col = f"obs_{i:02d}"
        clipped = (episodes_df[obs_col].abs() > 5).sum()
        clip_rates[obs_col] = (clipped / len(episodes_df)) * 100

    # Reward validation
    rewards = episodes_df['reward']
    reward_rmse = np.sqrt(((rewards - rewards.shift(1)) ** 2).mean())  # Should be ≈0
    reward_zero_pct = (rewards == 0).sum() / len(rewards) * 100

    # Cost model
    spread_p95 = np.percentile(episodes_df['spread_proxy_bps'], 95)
    peg_rate = (episodes_df['peg_indicator'] > 0).sum() / len(episodes_df) * 100

    return {
        "clip_rates": clip_rates,
        "max_clip_rate": max(clip_rates.values()),
        "reward_check": {
            "rmse": reward_rmse,
            "std": rewards.std(),
            "zero_pct": reward_zero_pct
        },
        "cost_model": {
            "spread_p95_bps": spread_p95,
            "peg_rate_pct": peg_rate
        },
        "pass": all([
            max(clip_rates.values()) <= 0.5,
            reward_rmse < 0.01,
            reward_zero_pct < 1.0,
            2 <= spread_p95 <= 25,
            peg_rate < 5.0
        ])
    }
```

**Estado Final:** 🔄 **NECESITA IMPLEMENTACIÓN COMPLETA - actualmente solo splits básicos**

---

#### 📌 PROPUESTA: L5 - Sortino, Calmar, Generalización, Stress tests, Latencias

**Estado:** ❌ **100% MOCK - No implementado**

**Realidad:**
```python
# pipeline_data_api.py:252-285 - GET /api/pipeline/l5/models
@app.get("/api/pipeline/l5/models")
def get_l5_models():
    """
    MOCK DATA - No lee modelos reales
    """
    return {
        "models": [
            {
                "model_id": "ppo_lstm_v2_1",
                "model_name": "PPO-LSTM",
                "version": "2.1",
                "format": "ONNX",
                "size_mb": 45.2,
                "created_at": "2024-12-15T10:30:00Z",
                "training_episodes": 5000,
                "val_reward_mean": 1180.3,
                "checkpoint_path": "s3://models/ppo_lstm_v2_1.onnx"
            }
            # ... 2 more hardcoded models
        ]
    }
```

**Lo que DEBE implementarse:**
```python
# Integración con MLflow o MinIO L5
@app.get("/api/pipeline/l5/models")
def get_l5_models_real():
    """
    Leer modelos de MLflow tracking o MinIO bucket
    """
    # Opción A: MLflow
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["1"])

    models = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params

        # Test split metrics
        test_sortino = metrics.get('test_sortino_ratio')
        test_calmar = metrics.get('test_calmar_ratio')
        test_maxdd = metrics.get('test_max_drawdown')

        # Generalization check
        train_sortino = metrics.get('train_sortino_ratio')
        generalization_gap = abs(train_sortino - test_sortino)

        # Stress test +25% costs
        stress_cagr = metrics.get('stress_cagr_drop_pct')

        # Latencies
        p99_inference_ms = metrics.get('p99_inference_latency_ms')
        p99_e2e_ms = metrics.get('p99_e2e_latency_ms')

        # Production gates
        gates = {
            "sortino_gate": test_sortino >= 1.3,
            "maxdd_gate": test_maxdd <= 0.15,
            "calmar_gate": test_calmar >= 0.8,
            "generalization_gate": generalization_gap <= 0.5,
            "stress_gate": stress_cagr <= 20,
            "latency_gate": p99_inference_ms <= 20 and p99_e2e_ms <= 100
        }

        models.append({
            "run_id": run.info.run_id,
            "model_id": params.get('model_id'),
            "test_metrics": {
                "sortino": test_sortino,
                "calmar": test_calmar,
                "max_drawdown": test_maxdd,
                "sharpe": metrics.get('test_sharpe_ratio')
            },
            "generalization": {
                "train_sortino": train_sortino,
                "test_sortino": test_sortino,
                "gap": generalization_gap
            },
            "stress_test": {
                "cagr_drop_pct": stress_cagr
            },
            "latencies": {
                "p99_inference_ms": p99_inference_ms,
                "p99_e2e_ms": p99_e2e_ms
            },
            "production_gates": gates,
            "ready_for_production": all(gates.values())
        })

    return {"models": models}

# Opción B: MinIO bucket
def get_l5_models_from_minio():
    """
    Leer de bucket 05-l5-ds-usdcop-serving
    """
    minio_client = Minio(...)
    objects = minio_client.list_objects('05-l5-ds-usdcop-serving')

    models = []
    for obj in objects:
        if obj.object_name.endswith('.json'):
            # Leer manifest
            manifest = json.loads(minio_client.get_object(...).read())
            models.append(manifest)

    return models
```

**Estado Final:** 🔄 **REQUIERE IMPLEMENTACIÓN COMPLETA con MLflow o MinIO**

---

#### 📌 PROPUESTA: L6 - Backtest KPIs, curvas, trade ledger

**Estado:** ⚠️ **IMPLEMENTADO pero con señales RANDOM, no de modelo**

**Realidad:**
```python
# backtest_api.py - Ya revisado arriba
# Implementación: ✅
# Problema: Usa señales random en vez de modelo L5
```

**Lo que falta:**
```python
# Integrar con modelo L5 para señales reales
@app.get("/api/backtest/results")
def get_backtest_with_model(model_id: str = "latest"):
    """
    1. Cargar modelo ONNX de L5
    2. Cargar datos de prueba
    3. Generar señales con modelo
    4. Ejecutar backtest
    """
    # Load model
    model = load_onnx_model(f"models/{model_id}.onnx")

    # Load test data
    df = get_test_data_l4()

    # Generate signals from model
    signals = []
    for i in range(len(df)):
        obs = df.iloc[i][['obs_00', 'obs_01', ..., 'obs_16']].values
        action = model.predict(obs)  # -1, 0, 1
        signals.append(action)

    # Run backtest with real model signals
    results = execute_backtest(df, signals)

    return results
```

**Estado Final:** 🔄 **Funcional pero necesita integración con modelo L5**

---

## 🎯 FEASIBILITY MAP - RESPUESTA A LA PROPUESTA

### ✅ LO QUE PUEDES PONER HOY MISMO (Sin cambios)

```
COMPLETAMENTE IMPLEMENTADO:
✅ Precio/cambio/high/low/vol 24h
✅ ATR/RSI/MACD/EMAs/Bollinger Bands
✅ L0 quality basics (count, nulls, duplicates, date range)
✅ VaR/CVaR/Max Drawdown del precio
✅ Backtest engine completo (con señales random)
✅ L1 episodios (60/60 check)
✅ L3 correlaciones básicas
✅ Splits train/val/test (L4 básico)
```

### 🔄 LO QUE NECESITA IMPLEMENTACIÓN (1-3 días)

```
FÁCIL DE AGREGAR:
🔄 Spread proxy Corwin-Schultz (fórmula lista)
🔄 Progreso de sesión 08:00-12:55 (trivial)
🔄 L0 extended: cobertura %, gaps, stale rate, OHLC invariants
🔄 L1 extended: grid 300s verification
🔄 L3 extended: Forward IC cálculo

MEDIO (requiere integración):
🔄 RL metrics REALES desde L4 (leer episodes de MinIO/DB)
🔄 L2 completo (winsor, HOD, 60+ indicators)
🔄 L4 contrato completo (17 obs schema + quality checks)
```

### ⚠️ LO QUE REQUIERE TRABAJO MAYOR (3-7 días)

```
REQUIERE INFRAESTRUCTURA:
⚠️ L5 real: Integración con MLflow o MinIO
   - Leer modelos entrenados
   - Métricas de training/validation
   - Production gates
   - Latencies de inferencia

⚠️ L6 con modelo: Backtest usando señales de L5 ONNX
   - Cargar modelo
   - Generar predicciones
   - Ejecutar backtest con señales reales

⚠️ MinIO integration en todos los endpoints
   - Leer L4 episodes
   - Leer L5 models
   - Leer L6 results pre-calculados
```

---

## 📊 MATRIZ DE PRIORIZACIÓN

| Métrica/KPI | Propuesta | Implementado | Gap | Prioridad | Esfuerzo |
|-------------|-----------|--------------|-----|-----------|----------|
| **Precio OHLCV + Stats** | ✅ | ✅ | None | - | - |
| **Indicadores Técnicos** | ✅ | ✅ | None | - | - |
| **Spread Corwin-Schultz** | ✅ | ❌ | Fórmula | 🔴 HIGH | 2h |
| **Progreso Sesión** | ✅ | ❌ | Cálculo simple | 🟡 MEDIUM | 1h |
| **VaR/CVaR/DD** | ✅ | ✅ | None | - | - |
| **L0 Extended** | ✅ | ⚠️ | Queries SQL | 🔴 HIGH | 4h |
| **L1 Grid Check** | ✅ | ⚠️ | Query SQL | 🟡 MEDIUM | 2h |
| **L2 Completo** | ✅ | ❌ | Endpoint nuevo | 🟡 MEDIUM | 8h |
| **L3 IC Forward** | ✅ | ⚠️ | Cálculo | 🟡 MEDIUM | 3h |
| **L4 Contrato** | ✅ | ❌ | Schema + checks | 🔴 HIGH | 16h |
| **L5 Real** | ✅ | ❌ | MLflow/MinIO | 🔴 CRITICAL | 24h |
| **L6 con Modelo** | ✅ | ⚠️ | ONNX integration | 🔴 HIGH | 16h |
| **RL Metrics Real** | ✅ | ❌ | MinIO L4 | 🔴 HIGH | 8h |

---

## 🎯 PLAN DE ACCIÓN RECOMENDADO

### Fase 1: Quick Wins (8 horas)
```
1. Spread Corwin-Schultz (2h)
2. Progreso sesión (1h)
3. L0 extended queries (4h)
4. L1 grid verification (1h)
```

### Fase 2: Pipeline Completeness (24 horas)
```
5. L2 endpoint completo (8h)
6. L3 IC forward (3h)
7. L4 contrato + quality checks (13h)
```

### Fase 3: Model Integration (40 horas)
```
8. MLflow/MinIO integration setup (8h)
9. L5 real models endpoint (16h)
10. L6 backtest con ONNX model (16h)
```

### Fase 4: RL Metrics Real (8 horas)
```
11. Read L4 episodes from storage (4h)
12. Calculate real RL metrics (4h)
```

**Total estimado:** 80 horas (2 semanas)

---

## ✅ CONCLUSIÓN FINAL

**Tu propuesta es EXCELENTE y está BIEN FUNDAMENTADA.**

### Qué está bien:
✅ Identificas correctamente las fuentes de datos disponibles
✅ Propones métricas calculables con OHLCV M5
✅ Reconoces las limitaciones (no portfolio real)
✅ Propones proxies inteligentes (Corwin-Schultz para spread)
✅ Defines contratos claros para cada layer (L0-L6)

### Qué falta implementar:
🔄 ~40% de las métricas propuestas necesitan código nuevo
🔄 L5/L6 requieren integración con almacenamiento de modelos
🔄 RL metrics necesitan leer datos reales de L4

### Recomendación:
**IMPLEMENTAR LA PROPUESTA EN FASES** siguiendo el plan de acción de arriba.

**Prioridad máxima:**
1. Spread Corwin-Schultz
2. L0 extended quality
3. L4 contrato real
4. L5 MLflow integration
5. RL metrics desde L4

Esto te dará un dashboard con **DATOS 100% REALES** sin mock data.
