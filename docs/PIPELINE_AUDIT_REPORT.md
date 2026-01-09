# INFORME DE AUDITORÍA DE PIPELINES USD/COP

**Fecha**: 2025-12-15
**Autor**: Auditoría Automatizada
**Objetivo**: Identificar redundancias y optimizar arquitectura de DAGs

---

## ARQUITECTURA GENERAL

El sistema sigue una arquitectura de capas L0-L6:

```
L0 (Adquisición) → L1 (Estandarización) → L2 (Preparación) → L3 (Features) → L4 (RL Ready) → L5 (Entrenamiento) → L6 (Backtest)
```

### Buckets MinIO

| Capa | Bucket |
|------|--------|
| L0 | `00-raw-usdcop-marketdata`, `00-raw-macro-marketdata` |
| L1 | `01-l1-ds-usdcop-standardize` |
| L2 | `02-l2-ds-usdcop-prepare` |
| L3 | `03-l3-ds-usdcop-feature` |
| L4 | `04-l4-ds-usdcop-rlready` |
| L5 | `05-l5-ds-usdcop-serving` |
| L6 | `06-emerald-backtest` |

---

## PIPELINES L0 - ADQUISICIÓN DE DATOS

### 1. usdcop_m5__00b_l0_macro_scraping (+ variantes)

**Archivo**: `airflow/dags/usdcop_m5__00b_l0_macro_scraping.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Scraping de datos macroeconómicos |
| **Fuentes** | TwelveData API, FRED API, BCRP (EMBI Colombia) |
| **Schedule** | 3 veces/día: 7:55, 10:30, 12:00 COT (L-V) |
| **Output Tables** | `dw.fact_macro_realtime`, `macro_ohlcv` |
| **Owner** | data_engineering |

**Datos Obtenidos**:
- DXY (US Dollar Index)
- VIX (Volatility Index)
- EMBI Colombia (Riesgo país)
- Brent, WTI (Petróleo)
- GOLD (Oro)
- USD/MXN, USD/CLP (FX pairs)
- Fed Funds, Treasury 2Y, Treasury 10Y (Tasas)

**DAGs Generados**:
- `usdcop_m5__00b_l0_macro_scraping` (principal, 7:55 COT)
- `usdcop_m5__00b_l0_macro_scraping_0755` (7:55 COT)
- `usdcop_m5__00b_l0_macro_scraping_1030` (10:30 COT)
- `usdcop_m5__00b_l0_macro_scraping_1200` (12:00 COT)

**Métricas Calculadas**:
- Z-scores (DXY, VIX, EMBI)
- Cambios 1D (%)
- VIX Regime (0=low, 1=normal, 2=elevated, 3=high)
- Rate spread, Term spread

---

### 2. usdcop_m5__01_l0_intelligent_acquire

**Archivo**: `airflow/dags/usdcop_m5__01_l0_intelligent_acquire.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Adquisición inteligente de OHLCV USD/COP con detección de gaps |
| **Fuente** | TwelveData API (20+ API keys en rotación) |
| **Schedule** | Manual (trigger) |
| **Output** | PostgreSQL `usdcop_m5_ohlcv`, MinIO `00-raw-usdcop-marketdata` |
| **Owner** | trading-team |

**Características**:
- Auto-detección de gaps históricos desde 2020
- Modo incremental vs histórico completo
- Ventana de trading: 8:00-12:55 COT (60 barras/día)
- Inserción con ON CONFLICT (GREATEST/LEAST para high/low)
- Integración con DWH Kimball (`fact_bar_5m`, `fact_l0_acquisition`)

**Modos de Operación**:
- `complete_historical`: Descarga todo desde 2020
- `fill_historical_gaps`: Rellena días faltantes intermedios
- `recent_incremental`: Actualización diaria
- `up_to_date`: Sin acción necesaria

---

### 3. usdcop_m5__01b_l0_macro_acquire

**Archivo**: `airflow/dags/usdcop_m5__01b_l0_macro_acquire.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Adquirir datos macro (WTI, DXY) desde TwelveData |
| **Schedule** | @daily con catchup desde 2002 |
| **Output** | PostgreSQL `macro_ohlcv`, MinIO `00-raw-macro-marketdata` |
| **Owner** | pipeline_engineer |

**Datos Obtenidos** (limitado):
- CL (WTI Crude Oil) - intervalo 1h
- DXY (US Dollar Index) - intervalo 1h

---

## PIPELINES L1-L2 - TRANSFORMACIÓN

### 4. usdcop_m5__02_l1_standardize

**Archivo**: `airflow/dags/usdcop_m5__02_l1_standardize.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Estandarización de datos L0 con validación de calidad |
| **Input** | MinIO `00-raw-usdcop-marketdata` |
| **Output** | MinIO `01-l1-ds-usdcop-standardize` |
| **Schedule** | Manual |
| **Owner** | data-team |

**Validaciones**:
- 60 barras por episodio (día)
- Calendario de feriados (US + Colombia + custom)
- Repeated OHLC detection (máx 5%)
- Grid exacto 300s entre barras
- Ventana premium 08:00-12:55 COT
- OHLC invariants (high >= max(open,close), etc.)

**Outputs**:
- `standardized_data_all.parquet/csv`
- `standardized_data_accepted.parquet/csv`
- `_reports/daily_quality_60.csv`
- `_statistics/hod_baseline.parquet`
- `_metadata.json`

**Quality Flags**:
- OK: 60/60 barras, sin problemas
- WARN: 59/60 barras con 1 gap
- FAIL: Feriado, gaps múltiples, repeated OHLC >5%, etc.

---

### 5. usdcop_m5__03_l2_prepare

**Archivo**: `airflow/dags/usdcop_m5__03_l2_prepare.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Preparación ML: deseasonalización, winsorización, splits |
| **Input** | MinIO `01-l1-ds-usdcop-standardize` |
| **Output** | MinIO `02-l2-ds-usdcop-prepare` |
| **Schedule** | Manual |
| **Owner** | data-team |

**Transformaciones**:
- Deseasonalización usando HOD baselines (mediana por hora)
- Winsorización de retornos (4σ)
- Split STRICT (60/60 solo) vs FLEX (59/60 con padding)

**Outputs**:
- `prepared_premium.parquet` (STRICT)
- `prepared_premium_flex.parquet` (FLEX)
- `metadata.json`

---

## PIPELINES L3 - FEATURE ENGINEERING

### 6. usdcop_m5__04_l3_feature

**Archivo**: `airflow/dags/usdcop_m5__04_l3_feature.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Features normalizados para RL con validación anti-leakage |
| **Input** | MinIO `02-l2-ds-usdcop-prepare` |
| **Output** | MinIO `03-l3-ds-usdcop-feature` |
| **Schedule** | Manual |
| **Owner** | data-team |

**Features Tier 1** (8):
- hl_range_surprise, atr_surprise
- body_ratio_abs, wick_asym_abs
- macd_strength_abs, compression_ratio
- band_cross_abs_k, entropy_absret_k

**Features Tier 2** (6):
- momentum_abs_norm, doji_freq_k
- gap_prev_open_abs, rsi_dist_50
- stoch_dist_mid, bb_squeeze_ratio

**Características**:
- Lag shift(5) para anti-leakage
- HOD residuals (volatility surprise vs baseline)
- Entropy-based orthogonal features
- IC < 0.10 threshold validation
- Causal rolling (solo información pasada)

---

### 7. usdcop_m5__04b_l3_llm_features

**Archivo**: `airflow/dags/usdcop_m5__04b_l3_llm_features.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Features interpretables para LLM (sin normalización) |
| **Input** | MinIO `02-l2-ds-usdcop-prepare` |
| **Output** | MinIO `03-l3-ds-usdcop-feature/llm-features` |
| **Schedule** | Manual |
| **Owner** | llm-team |

**Features LLM** (valores absolutos):
- price, ema_20, ema_50, sma_20, sma_50
- macd, macd_signal, macd_hist
- rsi_7, rsi_14
- stoch_k, stoch_d
- bb_upper, bb_middle, bb_lower, bb_width
- atr_14, atr_20
- volume, volume_sma_20, volume_ratio
- trend (bullish/bearish/neutral), trend_strength
- volatility_5, volatility_20, volatility_regime

**Diferencia clave**: NO normalización - valores legibles para LLMs.

---

## PIPELINE L4 - RL READY

### 8. usdcop_m5__05_l4_rlready

**Archivo**: `airflow/dags/usdcop_m5__05_l4_rlready.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Dataset final para entrenamiento RL |
| **Input** | MinIO `03-l3-ds-usdcop-feature` |
| **Output** | MinIO `04-l4-ds-usdcop-rlready` |
| **Schedule** | Manual |
| **Owner** | data-team |

**Observation Space** (32 features):
- obs_00 a obs_13: L3 Tier 1+2 (14)
- obs_14, obs_15: hour_sin, hour_cos (2)
- obs_16: spread_proxy_bps_norm (1)
- obs_17 a obs_23: Macro features (7)
- obs_24 a obs_31: MTF features 15m/1h (8)

**Normalización**:
- Median/MAD por hora
- Clip target |z| = 4.5
- Anti-leakage lag: 7 barras

---

## PIPELINES L5 - ENTRENAMIENTO

### 9. usdcop_m5__05a_l5_rl_training

**Archivo**: `airflow/dags/usdcop_m5__05a_l5_rl_training.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Entrenar modelo PPO-LSTM con Stable Baselines 3 |
| **Input** | MinIO `04-l4-ds-usdcop-rlready` |
| **Output** | MinIO `05-l5-ds-usdcop-serving`, MLflow |
| **Schedule** | Manual |
| **Owner** | ml-team |

**GO/NO-GO Gates**:
- Sortino ≥ 1.3
- MaxDD ≤ 15%
- Calmar ≥ 0.8
- Inference latency p99 ≤ 20ms
- E2E latency p99 ≤ 100ms

**Artefactos**:
- policy.onnx
- model.zip
- norm_stats.json
- training_metrics.json

---

### 10. usdcop_m5__05b_l5_ml_training

**Archivo**: `airflow/dags/usdcop_m5__05b_l5_ml_training.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Entrenar modelo LightGBM clasificación multiclase |
| **Input** | MinIO `04-l4-ds-usdcop-rlready` |
| **Output** | MinIO `05-l5-ds-usdcop-serving` |
| **Schedule** | Manual |
| **Owner** | ml-team |

**Configuración**:
- Objetivo: multiclass {0: short, 1: flat, 2: long}
- n_estimators: 1000, max_depth: 6
- learning_rate: 0.05
- early_stopping: 50 rounds

**Gates**:
- Accuracy > 55% (vs random 33%)
- Precision > 52%
- ECE (calibration error) < 10%
- Cada clase ≥ 15% representación

---

### 11. usdcop_m5__05c_l5_llm_setup_corrected

**Archivo**: `airflow/dags/usdcop_m5__05c_l5_llm_setup.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Configurar prompts y esquemas para LLM trading |
| **Output** | MinIO `05-l5-ds-usdcop-serving` |
| **Schedule** | Manual |
| **Owner** | llm-team |

**LLM Config**:
- Primary: DeepSeek V3
- Fallback: Claude Sonnet 4.5
- Temperature: 0.1 (bajo para consistencia)

**Output Schema (Pydantic)**:
```python
class AlphaArenaSignal:
    signal: Literal["buy_to_enter", "sell_to_enter", "hold", "close"]
    coin: Literal["USDCOP"]
    quantity: float  # 0.0-1.0
    leverage: int    # 1-20
    profit_target: float
    stop_loss: float
    risk_usd: float
    confidence: float
    invalidation_condition: str
    justification: str
```

---

## PIPELINES L5-L6 - PRODUCCIÓN

### 12. usdcop_m5__06_l5_realtime_inference

**Archivo**: `airflow/dags/usdcop_m5__06_l5_realtime_inference.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Ejecutar modelo PPO en tiempo real |
| **Schedule** | `*/5 8-12 * * 1-5` (cada 5 min, 8AM-12PM COT, L-V) |
| **Output Tables** | `dw.fact_rl_inference`, `dw.fact_agent_actions`, `dw.fact_equity_curve_realtime` |
| **Owner** | trading_rl |

**Flujo**:
1. Verificar horario de mercado
2. Obtener OHLCV + features recientes
3. Normalizar observación (20 features)
4. Ejecutar modelo PPO
5. Almacenar inferencia y acción
6. Actualizar equity curve
7. Generar alertas si necesario

**Configuración**:
- Model: ppo_usdcop_v14_fold0
- Initial equity: $10,000
- Cost per trade: 3 bps
- Position threshold: 0.1

---

### 13. usdcop_m5__07_l6_backtest_multi_strategy

**Archivo**: `airflow/dags/usdcop_m5__07_l6_backtest_multi_strategy.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Backtest comparativo de 3 estrategias |
| **Input** | L4 test data + modelos L5 |
| **Output** | MinIO `06-emerald-backtest` |
| **Schedule** | Manual |
| **Owner** | alpha-arena-team |

**Estrategias Evaluadas**:
1. RL_PPO (policy.onnx)
2. ML_LGBM (lightgbm_model.pkl)
3. LLM_DEEPSEEK (rule-based fallback)

**Outputs**:
- metrics/{strategy}/kpis_test.json
- trades/{strategy}/trade_ledger.parquet
- returns/{strategy}/daily_returns.parquet
- comparison/alpha_arena_leaderboard.json

---

### 14. usdcop_m5__99_alert_monitor

**Archivo**: `airflow/dags/usdcop_m5__99_alert_monitor.py`

| Campo | Valor |
|-------|-------|
| **Propósito** | Monitoreo y alertas del sistema |
| **Output** | `dw.fact_inference_alerts`, webhook opcional |
| **Owner** | monitoring |

**Umbrales de Alerta**:

| Métrica | Warning | Error | Critical |
|---------|---------|-------|----------|
| Latencia (ms) | 500 | 1000 | 5000 |
| Drawdown (%) | 2 | 5 | 10 |
| Pérdida diaria (%) | 1 | 3 | 5 |
| Sin inferencia (min) | 10 | 20 | - |

---

## DIAGRAMA DE DEPENDENCIAS

```
                    ┌─────────────────────────────────┐
                    │  TwelveData / FRED / BCRP APIs  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ 00b_macro_scraping│           │ 01_intelligent    │
        │ (DXY,VIX,EMBI,etc)│           │    _acquire       │
        └─────────┬─────────┘           │  (USD/COP OHLCV)  │
                  │                     └─────────┬─────────┘
                  │                               │
                  ▼                               ▼
        ┌─────────────────────────────────────────────────┐
        │              PostgreSQL (L0 Tables)              │
        │  usdcop_m5_ohlcv | macro_ohlcv | fact_macro_*   │
        └─────────────────────────┬───────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   02_l1_standardize     │
                    │  (Quality, Dedup, Grid) │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    03_l2_prepare        │
                    │ (Deseason, Winsorize)   │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │  04_l3_feature    │           │ 04b_l3_llm_features│
    │ (RL normalized)   │           │ (LLM interpretable)│
    └─────────┬─────────┘           └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │  05_l4_rlready    │
    │ (Obs space 32dim) │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┬───────────────────┐
    ▼                   ▼                   ▼
┌─────────┐       ┌─────────┐       ┌─────────────┐
│05a_rl   │       │05b_ml   │       │05c_llm_setup│
│training │       │training │       │  (prompts)  │
│ (PPO)   │       │(LightGBM│       └─────────────┘
└────┬────┘       └────┬────┘
     │                 │
     └────────┬────────┘
              ▼
    ┌───────────────────┐
    │06_realtime_infer  │──────────▶ Production
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │07_backtest_multi  │──────────▶ Evaluation
    └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │  99_alert_monitor │──────────▶ Monitoring
    └───────────────────┘
```

---

## ANÁLISIS DE REDUNDANCIAS

Ver documento separado: [PIPELINE_REDUNDANCY_SUMMARY.md](./PIPELINE_REDUNDANCY_SUMMARY.md)
