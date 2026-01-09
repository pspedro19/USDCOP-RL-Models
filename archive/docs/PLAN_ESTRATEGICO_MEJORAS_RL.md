# 🎯 PLAN ESTRATÉGICO: MEJORAS USD/COP RL SYSTEM
## Roadmap de Implementación 6 Semanas

**Versión:** 1.0
**Fecha:** 2025-11-05
**Objetivo:** Mejorar sistema RL de Sharpe -0.42 a 0.8-1.5 sin desmejorar componentes actuales

---

## 📋 ÍNDICE

1. [Arquitectura del Sistema y Responsabilidades](#arquitectura)
2. [Fase 1: Validación y Diagnóstico (Semana 1)](#fase-1)
3. [Fase 2: Pipeline Enhancement - L3/L4 (Semanas 2-3)](#fase-2)
4. [Fase 3: Modelo SAC (Semana 4)](#fase-3)
5. [Fase 4: Optimización (Semana 5)](#fase-4)
6. [Fase 5: Validación Final (Semana 6)](#fase-5)
7. [Matriz de Compatibilidad](#compatibilidad)
8. [Criterios de Rollback](#rollback)

---

## 🏗️ ARQUITECTURA DEL SISTEMA Y RESPONSABILIDADES {#arquitectura}

### **Estructura de Archivos y Ownership**

```
PROYECTO: USDCOP-RL-Models
│
├── 📁 airflow/dags/                    [AGENTE: Pipeline Engineer]
│   ├── usdcop_m5__03_l2_prepare.py    ❌ NO TOCAR (funciona)
│   ├── usdcop_m5__04_l3_feature.py    ✏️ MODIFICAR (añadir features)
│   ├── usdcop_m5__05_l4_rlready.py    ✏️ MODIFICAR (expandir obs_XX)
│   └── utils/pipeline_config.py        ✏️ MODIFICAR (nuevos params)
│
├── 📁 notebooks/                       [AGENTE: ML Engineer]
│   ├── usdcop_rl_notebook.ipynb       ✏️ MODIFICAR (añadir celdas SAC)
│   └── utils/
│       ├── environments.py             ✏️ MODIFICAR (actualizar obs_space)
│       ├── sb3_helpers.py              ✏️ MODIFICAR (añadir SAC)
│       ├── config.py                   ✏️ MODIFICAR (nuevos hyperparams)
│       ├── agents.py                   ❌ NO TOCAR
│       ├── backtesting.py              ✏️ MODIFICAR (walk-forward)
│       ├── metrics.py                  ✏️ MODIFICAR (WFE metric)
│       └── validation.py               ✏️ MODIFICAR (multi-seed)
│
├── 📁 config/                          [AGENTE: Config Manager]
│   ├── quality_thresholds.yaml         ❌ NO TOCAR
│   └── feature_config.yaml             ➕ CREAR (nueva config)
│
└── 📁 MinIO Buckets/                   [AGENTE: Data Engineer]
    ├── 02-l2-ds-usdcop-prepare/       ❌ NO TOCAR
    ├── 03-l3-ds-usdcop-feature/       📝 DATOS NUEVOS (features macro/MTF)
    └── 04-l4-ds-usdcop-rlready/       📝 DATOS NUEVOS (obs_17 a obs_44)
```

### **Agentes y Responsabilidades**

| Agente | Responsabilidad | Archivos | Dependencias |
|--------|----------------|----------|--------------|
| **Pipeline Engineer** | L3/L4 enhancement | `usdcop_m5__04_l3_feature.py`, `usdcop_m5__05_l4_rlready.py` | Ninguna (puede empezar inmediatamente) |
| **ML Engineer** | Modelos y entrenamiento | `notebooks/utils/*.py`, `usdcop_rl_notebook.ipynb` | Espera a que Pipeline Engineer termine L4 |
| **Data Engineer** | Gestión de buckets MinIO | Verificación de datos L3/L4 | Validación post-pipeline |
| **Config Manager** | Configuraciones centralizadas | `config/*.yaml`, `notebooks/utils/config.py` | Coordina con ambos engineers |
| **QA Engineer** | Validación y testing | Scripts de validación | Checkpoint al final de cada fase |

---

## 🔴 FASE 1: VALIDACIÓN Y DIAGNÓSTICO (SEMANA 1) {#fase-1}

**Objetivo:** Confirmar hipótesis de problema raíz antes de invertir en soluciones

**Responsable:** ML Engineer
**Duración:** 3-5 días
**Prerequisitos:** Ninguno

### **1.1 Archivo: `notebooks/utils/validation.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Añadir función: `validate_model_robust()`**

**Descripción:**
- Evaluar modelo con 10 seeds diferentes
- Capturar métricas: Sharpe, win rate, return, trades
- Calcular mean ± std para cada métrica
- Generar reporte comparativo

**Parámetros de configuración:**
```
n_seeds: 10
deterministic: True
n_eval_episodes_per_seed: 5
```

**Output esperado:**
```
DataFrame con columnas:
- seed (int)
- sharpe_ratio (float)
- win_rate (float)
- total_return_pct (float)
- trades_total (int)
- max_drawdown_pct (float)
```

**Criterio de éxito:**
- Si Sharpe_mean < 0.3 → Problema estructural confirmado
- Si Sharpe_mean > 0.5 → Problema es hyperparameters
- Si Sharpe_std > 0.4 → Alta variabilidad, necesita más timesteps

---

#### **Añadir función: `feature_importance_analysis()`**

**Descripción:**
- Usar RandomForestRegressor para medir poder predictivo de features
- Target: forward_return (5 steps ahead)
- Generar ranking de importancia

**Parámetros:**
```
n_estimators: 200
max_depth: 8
min_samples_leaf: 100
test_size: 0.2
```

**Output esperado:**
```
DataFrame con columnas:
- feature_name (str): obs_00, obs_01, ...
- importance (float): 0.0 to 1.0
- rank (int): 1 to 17

Métricas agregadas:
- r2_score (float)
- max_importance (float)
- top_5_features (list)
```

**Criterio de decisión:**
- Si max_importance < 0.10 → Features insuficientes, PROCEDER A FASE 2
- Si max_importance > 0.20 → Features tienen señal, problema es arquitectura

---

#### **Añadir función: `baseline_comparison()`**

**Descripción:**
- Implementar 3 estrategias simples:
  1. Buy-and-hold (benchmark pasivo)
  2. RSI mean reversion (RSI < 30 buy, > 70 sell)
  3. MA crossover (SMA 5 vs SMA 20)

**Output esperado:**
```
DataFrame comparativo:
Strategy     | Sharpe | Return% | Win Rate | Max DD%
-------------|--------|---------|----------|--------
Buy-Hold     |   X.XX |   ±X.XX |    XX.X% |  -X.XX
RSI          |   X.XX |   ±X.XX |    XX.X% |  -X.XX
MA Cross     |   X.XX |   ±X.XX |    XX.X% |  -X.XX
RL (Current) |   X.XX |   ±X.XX |    XX.X% |  -X.XX
```

**Criterio de decisión:**
- Si RL no supera ningún baseline → Problema severo
- Si RL supera al menos 1 baseline → Hay señal, necesita mejora

---

### **1.2 Archivo: `notebooks/usdcop_rl_notebook.ipynb`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Nueva Celda 6.5: "Validación Robusta (10 Seeds)"**

**Ubicación:** Después de celda de entrenamiento, antes de backtest

**Descripción:**
- Llamar a `validation.validate_model_robust()`
- Mostrar tabla de resultados con formato
- Generar gráfico de distribución (boxplot de Sharpe por seed)
- Imprimir criterio de decisión

**Output visual:**
```
📊 VALIDACIÓN CON 10 SEEDS
========================
Sharpe:     X.XX ± X.XX
Win Rate:   XX.X% ± X.X%
Return:     ±X.XX% ± X.XX%
Trades:     X.X promedio

[BOXPLOT: Distribución de Sharpe]

✅ DECISIÓN: [Problema estructural / Hyperparams / Continuar]
```

---

#### **Nueva Celda 6.6: "Feature Importance Analysis"**

**Descripción:**
- Llamar a `validation.feature_importance_analysis()`
- Generar bar chart de top 10 features
- Mostrar R² score
- Interpretar resultados

**Output visual:**
```
📊 FEATURE IMPORTANCE (RandomForest)
====================================
R² Score: X.XXXX

Top 5 Features:
1. obs_XX: X.XX (importance)
2. obs_XX: X.XX
3. obs_XX: X.XX
4. obs_XX: X.XX
5. obs_XX: X.XX

[BAR CHART: Top 10 features]

⚠️ DIAGNÓSTICO: [Features suficientes / Insuficientes]
```

---

#### **Nueva Celda 6.7: "Baseline Comparison"**

**Descripción:**
- Llamar a `validation.baseline_comparison()`
- Mostrar tabla comparativa
- Generar radar chart comparando estrategias

**Output visual:**
```
📊 COMPARACIÓN VS BASELINES
===========================
[TABLA COMPARATIVA]

[RADAR CHART: Sharpe, Return, Win Rate, Max DD]

✅ RESULTADO: RL [supera / no supera] baselines
```

---

### **1.3 Entregable Semana 1**

**Documentos a generar:**

1. **`reports/semana1_diagnostico.md`** (crear nuevo)
   - Resumen de validación con 10 seeds
   - Feature importance análisis
   - Comparación vs baselines
   - **DECISIÓN GO/NO-GO para Fase 2**

2. **`outputs/validation/`** (carpeta nueva)
   - `validation_10seeds.csv`
   - `feature_importance.csv`
   - `baseline_comparison.csv`
   - Plots: `sharpe_distribution.png`, `feature_importance.png`, `baseline_radar.png`

**Criterios de Decisión:**

| Métrica | Umbral Verde | Umbral Amarillo | Umbral Rojo |
|---------|-------------|----------------|-------------|
| Sharpe (10 seeds mean) | > 0.5 | 0.2 - 0.5 | < 0.2 |
| Max Feature Importance | > 0.20 | 0.10 - 0.20 | < 0.10 |
| RL supera baselines | 3/3 | 1-2/3 | 0/3 |

**Decisión:**
- **Verde (todos verdes):** Problema es hyperparams → Saltar a Fase 4
- **Amarillo (mixto):** Proceder con Fase 2 (features) + Fase 3 (SAC)
- **Rojo (mayoría rojos):** Problema estructural SEVERO → Proceder con Fase 2 URGENTE

---

## 🟡 FASE 2: PIPELINE ENHANCEMENT - L3/L4 (SEMANAS 2-3) {#fase-2}

**Objetivo:** Expandir features de 17 a ~45 (macro + multi-timeframe + technical)

**Responsable:** Pipeline Engineer (L3/L4) + Config Manager
**Duración:** 10-12 días
**Prerequisitos:** Decisión de Fase 1 = Proceder

---

### **2.1 Archivo: `config/feature_config.yaml`**

**Estado:** ➕ CREAR NUEVO

**Descripción:**
Archivo de configuración centralizada para nuevas features

**Contenido completo:**

```yaml
# Feature Enhancement Configuration
# Versión: 1.0 - Fase 2

feature_groups:

  # Grupo 1: Features originales L3 (MANTENER)
  original_l3:
    enabled: true
    count: 17
    obs_range: [0, 16]
    description: "Features originales con IC < 0.10"

  # Grupo 2: Macro Features (NUEVO)
  macro:
    enabled: true
    sources:
      wti_oil:
        enabled: true
        ticker: "CL=F"
        provider: "yfinance"
        resample: "5T"
        features:
          - wti_return_5
          - wti_return_20
          - wti_zscore_60

      dxy_dollar:
        enabled: true
        ticker: "DX-Y.NYB"
        provider: "yfinance"
        resample: "5T"
        features:
          - dxy_return_5
          - dxy_return_20
          - dxy_zscore_60

      correlation:
        enabled: true
        features:
          - cop_wti_corr_60

    obs_range: [17, 23]
    count: 7

  # Grupo 3: Multi-Timeframe (NUEVO)
  multi_timeframe:
    enabled: true
    timeframes:

      tf_15min:
        resample: "15T"
        indicators:
          - sma_20
          - rsi_14
          - macd
          - macd_hist
          - trend_direction  # Direccional: -1/0/+1

      tf_1hour:
        resample: "1H"
        indicators:
          - sma_50
          - vol_regime  # ATR / ATR_MA
          - adx_14

    obs_range: [24, 31]
    count: 8

  # Grupo 4: Technical Indicators Adicionales (NUEVO)
  technical:
    enabled: true
    indicators:
      momentum:
        - cci_20
        - williams_r_14
        - roc_10
        - stoch_k_14

      volume:
        - obv
        - mfi_14

      volatility:
        - keltner_upper_20
        - keltner_lower_20

      trend:
        - adx_14_5m
        - ema_6
        - ema_21

    obs_range: [32, 44]
    count: 13

# Normalización por grupo
normalization:
  original_l3:
    method: "z_score"  # Mantener actual
    clip: [-5, 5]

  macro:
    method: "robust_scaler"  # Median + IQR
    clip: [-5, 5]

  multi_timeframe:
    method: "robust_scaler"
    clip: [-5, 5]

  technical:
    method: "robust_scaler"
    clip: [-5, 5]

# Quality Gates (mantener estrictos para originales, relajar para nuevos)
quality_gates:
  original_features:
    ic_threshold: 0.10  # Mantener
    causality_strict: true

  new_features:
    ic_threshold: 0.30  # Relajado (queremos señal predictiva)
    causality_strict: false
    min_importance: 0.05  # Filtrar features inútiles

# Gestión de datos externos
external_data:
  cache_enabled: true
  cache_path: "./data/external_cache/"
  update_frequency: "daily"
  fallback_on_error: true  # Si WTI falla, usar valores previos

  data_sources:
    yfinance:
      retry_attempts: 3
      timeout_seconds: 30

# Validación post-pipeline
validation:
  check_nan_percentage: 0.05  # Max 5% NaN permitido
  check_feature_correlation: 0.95  # Eliminar si corr > 0.95
  check_variance_threshold: 0.001  # Eliminar features constantes
```

**Uso:**
- Pipeline L3/L4 leerán este config
- Permite habilitar/deshabilitar grupos de features
- Facilita rollback si algún grupo desmejora

---

### **2.2 Archivo: `airflow/dags/usdcop_m5__04_l3_feature.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 2.2.1: Importar configuración de features**

**Ubicación:** Inicio del archivo, después de imports existentes

**Descripción:**
- Cargar `feature_config.yaml`
- Crear funciones helper para verificar qué grupos están enabled

**Pseudo-lógica:**
```
Cargar YAML en variable FEATURE_CONFIG
Crear función: is_group_enabled(group_name) → bool
Crear función: get_obs_range(group_name) → tuple(start, end)
```

---

#### **Cambio 2.2.2: Nueva función `fetch_external_data()`**

**Ubicación:** Después de funciones helper existentes

**Descripción:**
- Descargar datos de WTI y DXY desde yfinance
- Implementar cache local (guardar en MinIO bucket auxiliar)
- Resample a 5min
- Manejar errores con fallback

**Pseudo-lógica:**
```
SI cache existe Y es reciente (< 24h):
    Cargar desde cache
SINO:
    INTENTAR descargar yfinance (con retry)
    SI falla:
        SI cache antiguo existe:
            Cargar cache antiguo + Warning
        SINO:
            ERROR crítico
    SI éxito:
        Guardar en cache
        Continuar

Resample a 5min con forward-fill
Return: DataFrame con columnas [timestamp, wti, dxy]
```

**Configuración:**
```
Cache bucket: "00-external-data-cache"
Cache key: "macro_data_{YYYY-MM-DD}.parquet"
Update frequency: Daily
```

---

#### **Cambio 2.2.3: Nueva función `calculate_macro_features()`**

**Ubicación:** Después de `calculate_tier2_features()`

**Descripción:**
- Calcular las 7 features macro según config
- Merge con datos USD/COP por timestamp
- Forward-fill missing values

**Pseudo-lógica:**
```
Cargar macro data (WTI, DXY)
Merge con df principal por timestamp

Calcular:
1. wti_return_5 = pct_change(5) de wti
2. wti_return_20 = pct_change(20) de wti
3. wti_zscore_60 = (wti - rolling_mean(60)) / rolling_std(60)
4. dxy_return_5 = pct_change(5) de dxy
5. dxy_return_20 = pct_change(20) de dxy
6. dxy_zscore_60 = (dxy - rolling_mean(60)) / rolling_std(60)
7. cop_wti_corr_60 = rolling_corr(close, wti, window=60)

Aplicar shift(5) para causality (consistente con Tier 1/2)

Return: df con nuevas columnas macro_*
```

---

#### **Cambio 2.2.4: Nueva función `calculate_mtf_features()`**

**Ubicación:** Después de `calculate_macro_features()`

**Descripción:**
- Resample a 15min y 1h
- Calcular indicadores específicos por timeframe
- Merge back a 5min con forward-fill

**Pseudo-lógica:**
```
# 15min timeframe
df_15m = resample('15T', OHLCV aggregation)

Calcular:
1. sma_20_15m = SMA(close, 20)
2. rsi_14_15m = RSI(close, 14)
3. macd_15m, macd_hist_15m = MACD(close, 12, 26, 9)
4. trend_15m = IF close > sma_20 THEN 1 ELSE IF close < sma_20 THEN -1 ELSE 0

# 1hour timeframe
df_1h = resample('1H', OHLCV aggregation)

Calcular:
5. sma_50_1h = SMA(close, 50)
6. atr_14_1h = ATR(14)
7. vol_regime_1h = atr_14_1h / rolling_mean(atr_14_1h, 50)
8. adx_14_1h = ADX(14)

Merge ambos back a df principal (5min)
Forward-fill para propagar valores

Aplicar shift(5) para causality

Return: df con columnas mtf_*
```

---

#### **Cambio 2.2.5: Nueva función `calculate_additional_technical()`**

**Ubicación:** Después de `calculate_mtf_features()`

**Descripción:**
- Calcular 13 indicadores técnicos adicionales
- Normalizar cada uno (RobustScaler)

**Pseudo-lógica:**
```
Calcular indicadores raw:
Momentum:
1. cci_20 = CCI(20)
2. williams_r_14 = Williams %R(14)
3. roc_10 = Rate of Change(10)
4. stoch_k_14 = Stochastic %K(14)

Volume:
5. obv = On-Balance Volume
6. mfi_14 = Money Flow Index(14)

Volatility:
7. keltner_upper_20 = Keltner Upper Band(20)
8. keltner_lower_20 = Keltner Lower Band(20)

Trend:
9. adx_14_5m = ADX(14) en 5min
10. ema_6 = EMA(6)
11. ema_21 = EMA(21)

Para cada indicador:
    Normalizar: (value - rolling_median(60)) / rolling_IQR(60)
    Clip a [-5, 5]

Aplicar shift(5) para causality

Return: df con columnas tech_*
```

---

#### **Cambio 2.2.6: Actualizar `main_l3_task()`**

**Ubicación:** Función principal que orquesta el pipeline L3

**Descripción:**
- Agregar llamadas a nuevas funciones
- Implementar feature groups condicionales (según config)

**Pseudo-lógica:**
```
# Features originales (SIEMPRE)
df = calculate_tier1_features(df)
df = calculate_tier2_features(df)

# Nuevas features (CONDICIONAL)
SI FEATURE_CONFIG['macro']['enabled']:
    macro_data = fetch_external_data()
    df = calculate_macro_features(df, macro_data)

SI FEATURE_CONFIG['multi_timeframe']['enabled']:
    df = calculate_mtf_features(df)

SI FEATURE_CONFIG['technical']['enabled']:
    df = calculate_additional_technical(df)

# Quality gates actualizado
df = apply_quality_gates(df, FEATURE_CONFIG)

# Guardar en MinIO
Guardar df en bucket 03-l3-ds-usdcop-feature/enhanced/
```

**Output:**
- Bucket L3 contendrá 2 versiones:
  - `rl-features/` (original, 17 features)
  - `rl-features-enhanced/` (nuevo, ~45 features)

---

### **2.3 Archivo: `airflow/dags/usdcop_m5__05_l4_rlready.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 2.3.1: Actualizar `OBS_MAPPING`**

**Ubicación:** Diccionario que mapea obs_XX a features

**Descripción:**
- Expandir de 17 a 45 features
- Mantener obs_00 a obs_16 EXACTAMENTE igual (compatibilidad)
- Añadir obs_17 a obs_44 para nuevas features

**Nuevo mapping:**
```
OBS_MAPPING = {
    # ORIGINALES (NO CAMBIAR)
    'obs_00': 'hl_range_surprise',
    'obs_01': 'atr_surprise',
    'obs_02': 'body_ratio_abs',
    'obs_03': 'wick_asym_abs',
    'obs_04': 'macd_strength_abs',
    'obs_05': 'compression_ratio',
    'obs_06': 'band_cross_abs_k',
    'obs_07': 'entropy_absret_k',
    'obs_08': 'momentum_abs_norm',
    'obs_09': 'doji_freq_k',
    'obs_10': 'gap_prev_open_abs',
    'obs_11': 'rsi_dist_50',
    'obs_12': 'stoch_dist_mid',
    'obs_13': 'bb_squeeze_ratio',
    'obs_14': 'hour_sin',
    'obs_15': 'hour_cos',
    'obs_16': 'spread_proxy_bps_norm',

    # MACRO (NUEVO)
    'obs_17': 'wti_return_5',
    'obs_18': 'wti_return_20',
    'obs_19': 'wti_zscore_60',
    'obs_20': 'dxy_return_5',
    'obs_21': 'dxy_return_20',
    'obs_22': 'dxy_zscore_60',
    'obs_23': 'cop_wti_corr_60',

    # MULTI-TIMEFRAME (NUEVO)
    'obs_24': 'sma_20_15m',
    'obs_25': 'rsi_14_15m',
    'obs_26': 'macd_15m',
    'obs_27': 'macd_hist_15m',
    'obs_28': 'trend_15m',
    'obs_29': 'sma_50_1h',
    'obs_30': 'vol_regime_1h',
    'obs_31': 'adx_14_1h',

    # TECHNICAL (NUEVO)
    'obs_32': 'cci_20_norm',
    'obs_33': 'williams_r_14_norm',
    'obs_34': 'roc_10_norm',
    'obs_35': 'stoch_k_14_norm',
    'obs_36': 'obv_norm',
    'obs_37': 'mfi_14_norm',
    'obs_38': 'keltner_upper_20_norm',
    'obs_39': 'keltner_lower_20_norm',
    'obs_40': 'adx_14_5m_norm',
    'obs_41': 'ema_6_norm',
    'obs_42': 'ema_21_norm',
    # obs_43, obs_44 reservados para futuro
}
```

---

#### **Cambio 2.3.2: Actualizar normalización**

**Ubicación:** Función de normalización de obs_XX

**Descripción:**
- Mantener Z-score para obs_00 a obs_16 (original)
- Usar RobustScaler para obs_17 a obs_44 (nuevo)

**Pseudo-lógica:**
```
PARA cada obs_XX en obs_cols:
    SI obs_XX in [obs_00 a obs_16]:
        # Normalización original (mantener)
        z_raw = (value - rolling_mean) / rolling_std
        obs_XX = clip(z_raw, -5, 5)

    SINO SI obs_XX in [obs_17 a obs_44]:
        # RobustScaler (nuevo)
        median = rolling_median(value, 60)
        q75, q25 = rolling_quantile(value, 60, [0.75, 0.25])
        iqr = q75 - q25

        SI iqr > 1e-8:
            obs_XX = (value - median) / iqr
        SINO:
            obs_XX = value - median

        obs_XX = clip(obs_XX, -5, 5)
```

---

#### **Cambio 2.3.3: Actualizar lógica de lectura de L3**

**Ubicación:** Task que lee datos de bucket L3

**Descripción:**
- Intentar leer de `rl-features-enhanced/` primero
- Si no existe, fallback a `rl-features/` (compatibilidad)

**Pseudo-lógica:**
```
bucket_l3 = "03-l3-ds-usdcop-feature"
prefixes_to_try = [
    "rl-features-enhanced/",  # Intentar primero
    "rl-features/"            # Fallback
]

PARA cada prefix en prefixes_to_try:
    files = list_keys(bucket_l3, prefix)
    SI files existe y no vacío:
        df = read_parquet(files)

        # Verificar features esperadas
        expected_cols = [val for val in OBS_MAPPING.values()]
        missing_cols = [col for col in expected_cols if col not in df.columns]

        SI missing_cols:
            Warning(f"Faltan {len(missing_cols)} features, usando prefix anterior")
            Continuar con siguiente prefix
        SINO:
            Success(f"Cargado desde {prefix}")
            Break

SI no se cargó nada:
    ERROR crítico
```

**Output:**
- Bucket L4 contendrá archivos con 17 o 45 obs_XX dependiendo de qué L3 usó

---

### **2.4 Archivo: `notebooks/utils/config.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 2.4.1: Actualizar `CONFIG` dict**

**Ubicación:** Variable global CONFIG

**Descripción:**
- Añadir nuevas configuraciones para feature groups

**Nuevo contenido a agregar:**
```python
CONFIG = {
    # ... configuraciones existentes ...

    # ========== DATASET L4 ENHANCED ==========
    'obs_dim': 45,  # ACTUALIZAR de 17 → 45
    'obs_dim_original': 17,  # Mantener referencia
    'episode_length': 60,  # MANTENER

    # ========== FEATURE GROUPS ==========
    'use_original_features': True,   # obs_00 a obs_16 (SIEMPRE True)
    'use_macro_features': True,      # obs_17 a obs_23
    'use_mtf_features': True,        # obs_24 a obs_31
    'use_technical_features': True,  # obs_32 a obs_44

    # ========== NORMALIZATION ==========
    'normalization_method_original': 'z_score',  # Para obs_00-16
    'normalization_method_new': 'robust_scaler', # Para obs_17-44

    # ... resto sin cambios ...
}
```

**Uso:**
- Permite deshabilitar grupos de features para A/B testing
- Environment leerá esto para determinar observation_space shape

---

### **2.5 Archivo: `notebooks/utils/environments.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 2.5.1: Actualizar `TradingEnvironmentL4.__init__()`**

**Ubicación:** Constructor de la clase

**Descripción:**
- Detectar automáticamente cuántas features tiene el dataset
- Ajustar observation_space dinámicamente

**Pseudo-lógica:**
```
INIT(data, episode_length, lags):
    # Detectar feature columns
    self.obs_cols = [col for col in data.columns
                     if col.startswith('obs_')
                     and not col.endswith('_z_raw')]

    self.obs_cols.sort()  # obs_00, obs_01, ...
    self.n_features = len(self.obs_cols)

    SI self.n_features == 17:
        Logger.info("Usando feature set ORIGINAL (17 features)")
    SINO SI self.n_features == 45:
        Logger.info("Usando feature set ENHANCED (45 features)")
    SINO:
        Warning(f"Feature count inesperado: {self.n_features}")

    # Actualizar observation shape
    self.observation_shape = (lags, self.n_features)

    # ... resto del init ...
```

**Compatibilidad:**
- Funciona con datasets de 17 features (legacy)
- Funciona con datasets de 45 features (enhanced)
- No requiere cambios en código que llame al environment

---

#### **Cambio 2.5.2: Actualizar `TradingEnvL4Gym.observation_space`**

**Ubicación:** Constructor de wrapper Gym

**Descripción:**
- Ajustar observation_space según n_features detectadas

**Pseudo-lógica:**
```
TradingEnvL4Gym.__init__(df, ...):
    self.custom_env = TradingEnvironmentL4(df, ...)

    n_features = self.custom_env.n_features
    lags = self.custom_env.lags

    # Observation space flattened
    flat_size = n_features * lags  # 17*10=170 o 45*10=450

    self.observation_space = spaces.Box(
        low=-5.0,
        high=5.0,
        shape=(flat_size,),  # DINÁMICO
        dtype=np.float32
    )

    Logger.info(f"Observation space: {self.observation_space.shape}")
```

**Output esperado:**
- Con 17 features: `Box(170,)`
- Con 45 features: `Box(450,)`

---

### **2.6 Entregable Semana 2-3**

**Pipeline Outputs:**

1. **MinIO Buckets actualizados:**
   ```
   03-l3-ds-usdcop-feature/
   ├── rl-features/                    [MANTENER - 17 features]
   └── rl-features-enhanced/           [NUEVO - 45 features]
       └── usdcop_m5_l3_enhanced_{date}.parquet

   04-l4-ds-usdcop-rlready/
   ├── legacy/                          [MANTENER - 17 obs]
   └── enhanced/                        [NUEVO - 45 obs]
       └── usdcop_m5_l4_enhanced_{date}.parquet
   ```

2. **Logs de validación:**
   ```
   logs/l3_enhanced_quality_report.json
   logs/l4_enhanced_validation.json
   ```

3. **Feature importance NUEVO:**
   - Ejecutar `validation.feature_importance_analysis()` con 45 features
   - Comparar con baseline de 17 features

**Criterios de Éxito Fase 2:**

| Métrica | Baseline (17 feat) | Target (45 feat) | Status |
|---------|-------------------|------------------|--------|
| Max feature importance | < 0.10 | > 0.15 | ✅/❌ |
| R² score (RF) | < 0.05 | > 0.10 | ✅/❌ |
| Top 5 features nuevas | N/A | ≥ 2 de macro/MTF | ✅/❌ |
| NaN% en dataset final | 0% | < 5% | ✅/❌ |

**Rollback Plan:**

SI algún criterio falla:
1. Deshabilitar grupo problemático en `feature_config.yaml`
2. Re-ejecutar pipeline L3/L4
3. Validar que con features restantes mejora vs baseline

---

## 🟢 FASE 3: MODELO SAC (SEMANA 4) {#fase-3}

**Objetivo:** Implementar SAC como alternativa a PPO, mantener ambos disponibles

**Responsable:** ML Engineer
**Duración:** 5-7 días
**Prerequisitos:** Fase 2 completada (L4 con 45 features disponible)

---

### **3.1 Archivo: `notebooks/utils/sb3_helpers.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 3.1.1: Nueva función `train_with_sac()`**

**Ubicación:** Después de `train_with_sb3()`

**Descripción:**
- Implementar training con SAC
- Configuración optimizada para FX exóticos
- Callbacks para early stopping y checkpointing

**Pseudo-lógica:**
```
FUNCIÓN train_with_sac(df_train, df_val, total_timesteps=300000):
    IMPORTAR:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    Crear environment:
        train_env = TradingEnvL4Gym(df_train, continuous_actions=True)
        eval_env = TradingEnvL4Gym(df_val, continuous_actions=True)

    Configuración SAC:
        learning_rate = 1e-4  # 50% menor que PPO
        buffer_size = 1_500_000  # Large replay buffer
        batch_size = 256
        gamma = 0.99
        tau = 0.005
        ent_coef = 'auto'  # Auto-tuning CRÍTICO
        target_entropy = 'auto'
        learning_starts = 10_000  # Warm-up
        net_arch = [256, 256]

    Crear modelo SAC

    Crear callbacks:
        EvalCallback(cada 10k steps, guardar best)
        CheckpointCallback(cada 50k steps)

    model.learn(total_timesteps, callbacks)

    Guardar modelo final

    RETURN model
```

**Parámetros configurables:**
```python
SAC_CONFIG = {
    'learning_rate': 1e-4,
    'buffer_size': 1_500_000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'ent_coef': 'auto',
    'learning_starts': 10_000,
    'train_freq': 1,
    'gradient_steps': 1,
    'net_arch': [256, 256],
    'device': 'cuda' if available else 'cpu'
}
```

---

#### **Cambio 3.1.2: Actualizar `train_with_sb3()` (PPO)**

**Ubicación:** Función existente

**Descripción:**
- NO CAMBIAR la lógica
- Solo añadir logging para comparación

**Pseudo-lógica:**
```
AL FINAL de la función, AÑADIR:
    Logger.info(f"PPO training completado")
    Logger.info(f"  Total timesteps: {total_timesteps}")
    Logger.info(f"  Observation dim: {train_env.observation_space.shape}")
    Logger.info(f"  Action space: {train_env.action_space}")
```

---

### **3.2 Archivo: `notebooks/utils/config.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 3.2.1: Añadir SAC configuration**

**Ubicación:** Después de configuraciones de PPO

**Nuevo contenido:**
```python
CONFIG = {
    # ... existente ...

    # ========== ALGORITHMS (actualizar) ==========
    'available_algorithms': ['SB3_PPO', 'SB3_SAC'],  # AÑADIR SAC

    # PPO config (MANTENER existente)
    'ppo_learning_rate': 0.0003,
    # ...

    # SAC config (NUEVO)
    'sac_learning_rate': 1e-4,
    'sac_buffer_size': 1_500_000,
    'sac_batch_size': 256,
    'sac_gamma': 0.99,
    'sac_tau': 0.005,
    'sac_ent_coef': 'auto',
    'sac_learning_starts': 10_000,
    'sac_net_arch': [256, 256],

    # ... resto sin cambios ...
}
```

---

### **3.3 Archivo: `notebooks/usdcop_rl_notebook.ipynb`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 3.3.1: Actualizar Celda 5 (Selección de algoritmo)**

**Ubicación:** Celda donde se configura ALGORITHM

**Descripción:**
- Actualizar lista de algoritmos disponibles

**Cambio:**
```python
# ANTES:
ALGORITHM = 'SB3_PPO'  # ⭐ UNICO QUE FUNCIONA EN WINDOWS

# DESPUÉS:
ALGORITHM = 'SB3_SAC'  # ⭐ RECOMENDADO para FX (mejor que PPO)
# Opciones disponibles:
# - 'SB3_PPO'  # RecurrentPPO con LSTM
# - 'SB3_SAC'  # SAC con auto-entropy tuning (NUEVO)
```

---

#### **Cambio 3.3.2: Nueva Celda 6.1: "Training SAC"**

**Ubicación:** Alternativa a la celda de training PPO

**Descripción:**
- Entrenar con SAC si está seleccionado
- Logging detallado

**Pseudo-código:**
```python
SI ALGORITHM == 'SB3_SAC':
    logger.step("Entrenando SAC...")

    agent_sac = train_with_sac(
        df_train,
        df_val,
        total_timesteps=300_000
    )

    SI agent_sac is not None:
        logger.success("SAC entrenado exitosamente")
        Guardar metadata:
            - Timesteps totales
            - Observation dim
            - Tiempo de entrenamiento
    SINO:
        logger.error("SAC falló")

SINO SI ALGORITHM == 'SB3_PPO':
    # Lógica existente de PPO
    ...
```

---

#### **Cambio 3.3.3: Nueva Celda 6.8: "Comparación SAC vs PPO"**

**Ubicación:** Después de entrenar ambos modelos

**Descripción:**
- Evaluar ambos modelos en mismo test set
- Comparar métricas lado a lado

**Pseudo-código:**
```python
# Cargar modelos
model_sac = SAC.load('./models_sac/best_model.zip')
model_ppo = RecurrentPPO.load('./models_sb3/best_model.zip')

# Evaluar ambos con validation.validate_model_robust()
results_sac = validate_model_robust(model_sac, test_env, n_seeds=10)
results_ppo = validate_model_robust(model_ppo, test_env, n_seeds=10)

# Crear tabla comparativa
comparison_df = pd.DataFrame({
    'SAC': results_sac,
    'PPO': results_ppo,
    'Difference': results_sac - results_ppo,
    'Improvement %': ((results_sac / results_ppo) - 1) * 100
})

Mostrar tabla
Generar radar chart comparando ambos

# DECISIÓN
SI results_sac['sharpe_mean'] > results_ppo['sharpe_mean'] + 0.2:
    Imprimir("✅ SAC CLARAMENTE SUPERIOR → Usar SAC")
SINO SI results_sac['sharpe_mean'] > results_ppo['sharpe_mean']:
    Imprimir("⚠️ SAC ligeramente mejor → Considerar ensemble")
SINO:
    Imprimir("❌ PPO sigue siendo mejor → Mantener PPO")
```

---

### **3.4 Entregable Semana 4**

**Modelos generados:**

1. **`models_sac/`** (carpeta nueva)
   ```
   best_model.zip           (Best checkpoint basado en eval)
   sac_checkpoint_50k.zip
   sac_checkpoint_100k.zip
   ...
   sac_final.zip           (Modelo al finalizar 300k timesteps)
   ```

2. **`models_sb3/`** (existente, mantener)
   ```
   best_model.zip          (PPO)
   ppo_final.zip
   ```

**Reportes:**

3. **`reports/semana4_sac_vs_ppo.md`** (crear)
   - Comparación detallada SAC vs PPO
   - Tabla de métricas (10 seeds cada uno)
   - Gráficos comparativos
   - Recomendación de qué modelo usar

**Criterios de Éxito Fase 3:**

| Métrica | PPO (Baseline) | SAC (Target) | Mejora Esperada |
|---------|---------------|--------------|-----------------|
| Sharpe (mean) | X.XX | > Baseline + 0.15 | +15-30% |
| Win Rate | XX% | > Baseline + 3% | +5-10% |
| Training stability | Variable | Más estable | Menos oscilación |
| Convergence speed | ~120k steps | ~200k steps | Más lento pero mejor |

**Decisión:**

- **SAC > PPO + 0.2 Sharpe:** Usar SAC exclusivamente
- **SAC > PPO + 0.1:** Mantener ambos, considerar ensemble (Fase 6)
- **SAC ≤ PPO:** Mantener PPO, investigar por qué SAC no mejora

---

## 🔵 FASE 4: OPTIMIZACIÓN (SEMANA 5) {#fase-4}

**Objetivo:** Hyperparameter tuning sistemático con Optuna

**Responsable:** ML Engineer
**Duración:** 5-7 días
**Prerequisitos:** Fase 3 completada, mejor modelo identificado (SAC o PPO)

---

### **4.1 Archivo: `notebooks/utils/optimization.py`**

**Estado:** ➕ CREAR NUEVO

**Descripción:**
Módulo dedicado a hyperparameter optimization con Optuna

**Contenido completo:**

```
# Pseudo-código del archivo completo

IMPORTAR:
    optuna
    optuna.pruners.MedianPruner
    stable_baselines3 (SAC/PPO según mejor modelo)

CLASE OptunaOptimizer:

    INIT(env_train, env_val, model_class, base_params):
        self.env_train = env_train
        self.env_val = env_val
        self.model_class = model_class  # SAC o PPO
        self.base_params = base_params
        self.study = None

    FUNCIÓN objective(trial):
        """
        Función objetivo para Optuna
        Optimiza hacia Sharpe ratio máximo
        """

        # Hyperparameters a optimizar (ajustar según modelo)
        SI model_class == SAC:
            params = {
                'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-3),
                'gamma': trial.suggest_uniform('gamma', 0.95, 0.9999),
                'batch_size': trial.suggest_categorical('batch', [64, 128, 256]),
                'buffer_size': trial.suggest_categorical('buffer', [500k, 1M, 1.5M]),
                'tau': trial.suggest_uniform('tau', 0.001, 0.01),
                'n_neurons_1': trial.suggest_categorical('n1', [128, 256, 512]),
                'n_neurons_2': trial.suggest_categorical('n2', [128, 256, 512])
            }

        SI model_class == PPO:
            params = {
                'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-3),
                'gamma': trial.suggest_uniform('gamma', 0.95, 0.9999),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch', [32, 64, 128]),
                'n_epochs': trial.suggest_int('epochs', 5, 20),
                'ent_coef': trial.suggest_uniform('ent', 0.001, 0.05),
                'clip_range': trial.suggest_uniform('clip', 0.1, 0.3)
            }

        # Crear modelo con params
        model = self.model_class(
            policy="MlpPolicy",
            env=self.env_train,
            **params,
            **self.base_params,
            verbose=0
        )

        # Training (reduced timesteps para speed)
        model.learn(total_timesteps=100_000)

        # Evaluar en validation (NO test!)
        val_results = []
        PARA i in range(5):  # 5 episodios para balance speed/robustez
            obs = self.env_val.reset()
            done = False
            rewards = []
            MIENTRAS not done:
                action = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env_val.step(action)
                rewards.append(reward)

            sharpe = mean(rewards) / (std(rewards) + 1e-8) * sqrt(252)
            val_results.append(sharpe)

        mean_sharpe = mean(val_results)

        # Pruning para eficiencia
        trial.report(mean_sharpe, step=100_000)
        SI trial.should_prune():
            raise optuna.TrialPruned()

        RETURN mean_sharpe

    FUNCIÓN optimize(n_trials=40):
        """
        Ejecutar optimization
        """

        # Crear study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50_000)
        )

        Logger.info(f"Iniciando Optuna con {n_trials} trials...")

        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True
        )

        Logger.success("Optimization completada")
        Logger.info(f"Best trial: {self.study.best_trial.number}")
        Logger.info(f"Best Sharpe: {self.study.best_value:.2f}")
        Logger.info(f"Best params: {self.study.best_params}")

        RETURN self.study.best_params

    FUNCIÓN save_results(output_dir):
        """
        Guardar resultados del study
        """

        # DataFrame con todos los trials
        df = self.study.trials_dataframe()
        df.to_csv(f"{output_dir}/optuna_trials.csv")

        # Visualizaciones
        import optuna.visualization as vis

        fig_history = vis.plot_optimization_history(self.study)
        fig_history.write_html(f"{output_dir}/optuna_history.html")

        fig_importance = vis.plot_param_importances(self.study)
        fig_importance.write_html(f"{output_dir}/optuna_importance.html")

        fig_parallel = vis.plot_parallel_coordinate(self.study)
        fig_parallel.write_html(f"{output_dir}/optuna_parallel.html")

        Logger.success(f"Resultados guardados en {output_dir}")

FUNCIÓN retrain_with_best_params(best_params, env_train, model_class, timesteps=500_000):
    """
    Re-entrenar modelo con mejores hyperparameters
    """

    Logger.info("Re-entrenando con hyperparameters óptimos...")

    model = model_class(
        policy="MlpPolicy",
        env=env_train,
        **best_params,
        verbose=1,
        tensorboard_log='./logs_optimized/'
    )

    # Callbacks
    eval_callback = EvalCallback(...)
    checkpoint_callback = CheckpointCallback(...)

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save('./models_optimized/final_model.zip')

    RETURN model
```

---

### **4.2 Archivo: `notebooks/usdcop_rl_notebook.ipynb`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 4.2.1: Nueva Celda 6.9: "Hyperparameter Optimization"**

**Ubicación:** Después de comparación SAC vs PPO

**Descripción:**
- Ejecutar Optuna optimization
- Mostrar progreso y resultados

**Pseudo-código:**
```python
from utils.optimization import OptunaOptimizer

# Definir cuál modelo optimizar (basado en decisión de Fase 3)
model_to_optimize = SAC  # o PPO
logger.header("Hyperparameter Optimization - SAC")

# Split train en train/val para optimization
split_idx = int(len(df_train) * 0.8)
df_train_opt = df_train.iloc[:split_idx].copy()
df_val_opt = df_train.iloc[split_idx:].copy()

train_env_opt = TradingEnvL4Gym(df_train_opt)
val_env_opt = TradingEnvL4Gym(df_val_opt)

# Parámetros base (no optimizables)
base_params = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tensorboard_log': None  # Deshabilitar durante optimization
}

# Crear optimizer
optimizer = OptunaOptimizer(
    env_train=train_env_opt,
    env_val=val_env_opt,
    model_class=model_to_optimize,
    base_params=base_params
)

# Ejecutar (40 trials, ~6-8 horas)
logger.step("Ejecutando Optuna (40 trials, ~6-8 horas)...")
best_params = optimizer.optimize(n_trials=40)

# Guardar resultados
optimizer.save_results('./outputs/optuna/')

# Mostrar resultados
logger.success("Optimization completada")
Mostrar tabla con best_params
Mostrar gráfico de optimization history
```

---

#### **Cambio 4.2.2: Nueva Celda 6.10: "Re-entrenar con Best Params"**

**Ubicación:** Después de optimization

**Descripción:**
- Re-entrenar modelo con hyperparameters óptimos
- Usar full training set (no split)

**Pseudo-código:**
```python
from utils.optimization import retrain_with_best_params

logger.header("Re-entrenando con Hyperparameters Óptimos")

# Full training set
full_train_env = TradingEnvL4Gym(df_train)

# Re-entrenar (500k timesteps para convergencia completa)
model_optimized = retrain_with_best_params(
    best_params=best_params,
    env_train=full_train_env,
    model_class=model_to_optimize,
    timesteps=500_000
)

logger.success("Modelo optimizado guardado")

# Evaluar inmediatamente
logger.step("Evaluando modelo optimizado...")
results_optimized = validate_model_robust(model_optimized, test_env, n_seeds=10)

Mostrar resultados:
    Sharpe: X.XX ± X.XX
    Win Rate: XX.X% ± X.X%
    Return: ±X.XX% ± X.XX%
```

---

### **4.3 Entregable Semana 5**

**Archivos generados:**

1. **`outputs/optuna/`** (carpeta nueva)
   ```
   optuna_trials.csv              (Todos los trials con params y resultados)
   optuna_history.html            (Gráfico de optimization history)
   optuna_importance.html         (Importancia de cada hyperparameter)
   optuna_parallel.html           (Parallel coordinate plot)
   ```

2. **`models_optimized/`** (carpeta nueva)
   ```
   final_model.zip                (Modelo con best hyperparams, 500k timesteps)
   checkpoint_100k.zip
   checkpoint_200k.zip
   ...
   ```

3. **`reports/semana5_optimization.md`** (crear)
   - Best hyperparameters encontrados
   - Comparación antes/después de optimization
   - Análisis de importancia de cada hyperparameter

**Criterios de Éxito Fase 4:**

| Métrica | Pre-Optimization | Post-Optimization | Mejora Esperada |
|---------|-----------------|-------------------|-----------------|
| Sharpe (mean) | X.XX | > Pre + 0.1 | +10-20% |
| Sharpe (std) | X.XX | < Pre | Más consistente |
| Win Rate | XX% | > Pre + 2% | +3-5% |

**Análisis de sensitivity:**

- Identificar top 3 hyperparameters más importantes
- Documentar rangos óptimos para cada uno
- Recomendar valores default para futuros entrenamientos

---

## 🟣 FASE 5: VALIDACIÓN FINAL (SEMANA 6) {#fase-5}

**Objetivo:** Walk-forward validation para confirmar robustez

**Responsable:** QA Engineer + ML Engineer
**Duración:** 5-7 días
**Prerequisitos:** Fase 4 completada, modelo optimizado disponible

---

### **5.1 Archivo: `notebooks/utils/backtesting.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 5.1.1: Nueva función `walk_forward_validation()`**

**Ubicación:** Al final del archivo

**Descripción:**
- Implementar walk-forward analysis
- Entrenar en ventana de 1 año, testear en siguiente quarter
- Calcular WFE (Walk-Forward Efficiency)

**Pseudo-lógica:**
```
FUNCIÓN walk_forward_validation(df, model_class, model_params,
                                train_days=252, test_days=63):
    """
    Walk-Forward Optimization

    train_days: 252 (1 año trading)
    test_days: 63 (3 meses / 1 quarter)
    """

    # Extraer fechas únicas
    df['date'] = to_date(df['timestamp'])
    unique_dates = sorted(df['date'].unique())

    results = []
    fold = 1
    train_start = 0
    test_start = train_days

    MIENTRAS test_start + test_days <= len(unique_dates):
        Logger.info(f"=== FOLD {fold} ===")

        # Definir ventanas
        train_dates = unique_dates[train_start:test_start]
        test_dates = unique_dates[test_start:test_start+test_days]

        df_fold_train = df[df['date'] in train_dates]
        df_fold_test = df[df['date'] in test_dates]

        Logger.info(f"Train: {train_dates[0]} to {train_dates[-1]}")
        Logger.info(f"Test:  {test_dates[0]} to {test_dates[-1]}")

        # Entrenar
        env_train = TradingEnvL4Gym(df_fold_train)
        model = model_class("MlpPolicy", env_train, **model_params, verbose=0)
        model.learn(total_timesteps=200_000)

        # Testear con múltiples seeds
        env_test = TradingEnvL4Gym(df_fold_test)

        fold_sharpes = []
        fold_returns = []
        fold_winrates = []
        fold_maxdds = []

        PARA seed in range(5):
            backtest_df, info = detailed_backtest_sb3(model, env_test)

            fold_sharpes.append(info['sharpe_ratio'])
            fold_returns.append(info['total_return_pct'])
            fold_winrates.append(info['win_rate'])
            fold_maxdds.append(info['max_drawdown_pct'])

        # Guardar resultados
        results.append({
            'fold': fold,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'sharpe_mean': mean(fold_sharpes),
            'sharpe_std': std(fold_sharpes),
            'return_mean': mean(fold_returns),
            'winrate_mean': mean(fold_winrates),
            'maxdd_mean': mean(fold_maxdds)
        })

        # Avanzar ventana
        train_start += test_days
        test_start += test_days
        fold += 1

    results_df = DataFrame(results)

    # Calcular WFE
    # Asumiendo 10% in-sample return promedio (ajustar según tus datos)
    in_sample_return_assumption = 0.10
    out_sample_return = results_df['return_mean'].mean() / 100
    wfe = (out_sample_return / in_sample_return_assumption) * 100

    Logger.header("WALK-FORWARD RESULTS")
    Mostrar tabla results_df

    Logger.metric("Avg Sharpe", results_df['sharpe_mean'].mean())
    Logger.metric("Avg Return", results_df['return_mean'].mean())
    Logger.metric("Avg Win Rate", results_df['winrate_mean'].mean())
    Logger.metric("WFE", f"{wfe:.1f}%")

    # Criterio
    SI wfe > 60:
        status = 'ROBUSTO'
    SINO SI wfe > 30:
        status = 'MARGINAL'
    SINO:
        status = 'OVERFITTING'

    Logger.info(f"Status: {status}")

    RETURN results_df, wfe, status
```

---

### **5.2 Archivo: `notebooks/utils/metrics.py`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 5.2.1: Nueva función `calculate_wfe()`**

**Ubicación:** Al final del archivo

**Descripción:**
- Calcular Walk-Forward Efficiency correctamente
- Documentar interpretación

**Pseudo-lógica:**
```
FUNCIÓN calculate_wfe(in_sample_returns, out_sample_returns):
    """
    Walk-Forward Efficiency

    WFE = (Avg OOS Return / Avg IS Return) × 100

    Interpretación:
    - WFE > 100%: OOS mejor que IS (raro, posible suerte)
    - WFE 60-100%: Excelente (robusto)
    - WFE 30-60%: Aceptable (algo de overfitting)
    - WFE < 30%: Pobre (overfitting severo)
    """

    avg_is = mean(in_sample_returns)
    avg_oos = mean(out_sample_returns)

    SI avg_is == 0:
        RETURN None  # No se puede calcular

    wfe = (avg_oos / avg_is) * 100

    RETURN wfe
```

---

### **5.3 Archivo: `notebooks/usdcop_rl_notebook.ipynb`**

**Estado:** ✏️ MODIFICAR

**Cambios a implementar:**

#### **Cambio 5.3.1: Nueva Celda 7.1: "Walk-Forward Validation"**

**Ubicación:** Nueva sección después de optimization

**Descripción:**
- Ejecutar walk-forward con modelo optimizado

**Pseudo-código:**
```python
from utils.backtesting import walk_forward_validation

logger.header("Walk-Forward Validation")

# Cargar TODO el dataset (2002-2025)
# Asumiendo que tienes df_all con todos los datos históricos
logger.step("Cargando dataset completo 2002-2025...")

# Ejecutar walk-forward
logger.step("Ejecutando Walk-Forward (8-10 folds, ~2-3 horas)...")

wf_results, wfe, status = walk_forward_validation(
    df=df_all,
    model_class=SAC,  # O PPO, según mejor modelo
    model_params=best_params,  # De Fase 4
    train_days=252,
    test_days=63
)

# Guardar resultados
wf_results.to_csv('./outputs/walk_forward_results.csv')

# Visualización
Crear figura con 4 subplots:
    1. Sharpe por fold (line plot con error bands)
    2. Returns por fold (bar chart)
    3. Win rate por fold (line plot)
    4. Max drawdown por fold (bar chart rojo)

Guardar como './outputs/walk_forward_analysis.png'

# Mostrar interpretación
logger.header("INTERPRETACIÓN")
SI status == 'ROBUSTO':
    logger.success(f"✅ WFE = {wfe:.1f}% → Modelo ROBUSTO para producción")
SINO SI status == 'MARGINAL':
    logger.warning(f"⚠️ WFE = {wfe:.1f}% → Modelo MARGINAL, usar con precaución")
SINO:
    logger.error(f"❌ WFE = {wfe:.1f}% → OVERFITTING, NO usar en producción")
```

---

#### **Cambio 5.3.2: Nueva Celda 7.2: "Out-of-Sample Final Test"**

**Ubicación:** Después de walk-forward

**Descripción:**
- Test final en datos 2024-2025 (nunca vistos)

**Pseudo-código:**
```python
logger.header("Out-of-Sample Test (2024-2025)")

# Filtrar datos más recientes
df_oos = df_all[df_all['timestamp'] >= '2024-01-01'].copy()

SI len(df_oos) > 0:
    logger.info(f"Testing en {len(df_oos)} rows (2024-2025)")

    env_oos = TradingEnvL4Gym(df_oos)

    # Cargar modelo optimizado
    model_final = SAC.load('./models_optimized/final_model.zip')

    # Evaluar con 10 seeds
    oos_results = validate_model_robust(model_final, env_oos, n_seeds=10)

    # Mostrar resultados
    logger.header("OUT-OF-SAMPLE RESULTS")
    Mostrar tabla:
        Sharpe:     X.XX ± X.XX
        Return:     ±X.XX% ± X.XX%
        Win Rate:   XX.X% ± X.X%
        Max DD:     -X.XX%

    # Criterio de producción
    SI oos_results['sharpe_mean'] > 0.5 AND oos_results['winrate_mean'] > 0.48:
        logger.success("✅ MODELO APROBADO PARA PRODUCCIÓN")
    SINO:
        logger.warning("⚠️ Resultados OOS por debajo de mínimos")

    # Guardar
    oos_results.to_csv('./outputs/oos_final_results.csv')

SINO:
    logger.warning("No hay datos 2024-2025 para OOS test")
```

---

### **5.4 Entregable Semana 6**

**Reportes finales:**

1. **`reports/FINAL_VALIDATION_REPORT.md`** (crear)
   - Resumen walk-forward (8-10 folds)
   - WFE y status (ROBUSTO/MARGINAL/OVERFITTING)
   - Out-of-sample test results
   - **DECISIÓN GO/NO-GO PARA PRODUCCIÓN**

2. **`outputs/walk_forward/`** (carpeta nueva)
   ```
   walk_forward_results.csv       (Resultados por fold)
   walk_forward_analysis.png      (Visualización 4 subplots)
   fold_models/                   (Modelos de cada fold)
       fold_1_model.zip
       fold_2_model.zip
       ...
   ```

3. **`outputs/oos_test/`** (carpeta nueva)
   ```
   oos_final_results.csv          (10 seeds en 2024-2025)
   oos_equity_curve.png
   oos_trade_distribution.png
   ```

**Criterios de Decisión Final:**

| Criterio | Mínimo Aceptable | Target | Status |
|----------|-----------------|--------|--------|
| WFE | > 40% | > 60% | ✅/❌ |
| Avg Sharpe (WF folds) | > 0.5 | > 0.8 | ✅/❌ |
| OOS Sharpe (2024-2025) | > 0.3 | > 0.6 | ✅/❌ |
| OOS Win Rate | > 48% | > 52% | ✅/❌ |
| Max DD (any fold) | < -30% | < -20% | ✅/❌ |
| Consistency (σ Sharpe entre folds) | < 0.5 | < 0.3 | ✅/❌ |

**Decisión GO/NO-GO:**

```
SI TODOS los criterios "Mínimo Aceptable" son ✅:
    → ✅ GO TO PRODUCTION (paper trading primero)

SI ≥ 4/6 criterios mínimos ✅:
    → ⚠️ MARGINAL - Proceder con precaución extrema

SI < 4/6 criterios mínimos ✅:
    → ❌ NO-GO - Requiere más investigación/cambios estructurales
```

---

## 🔄 MATRIZ DE COMPATIBILIDAD {#compatibilidad}

**Compatibilidad entre versiones de features y modelos:**

| Dataset L4 | Features | Modelos Compatibles | Observaciones |
|------------|---------|---------------------|---------------|
| `legacy/` | 17 (obs_00-16) | PPO (actual), SAC (nuevo) | ✅ Mantener para baseline |
| `enhanced/` | 45 (obs_00-44) | PPO, SAC, Optimized | ✅ Usar para mejores resultados |

**Migración sin breaking changes:**

```python
# Environment detecta automáticamente n_features
env = TradingEnvL4Gym(df)  # Funciona con 17 o 45 features

# Config permite deshabilitar grupos
CONFIG['use_macro_features'] = False  # Volver a 17 features
```

**Rollback rápido:**

1. **Si features macro fallan:**
   ```yaml
   # feature_config.yaml
   macro:
     enabled: false  # Deshabilitar
   ```

2. **Si SAC no mejora:**
   ```python
   # notebook
   ALGORITHM = 'SB3_PPO'  # Volver a PPO
   ```

3. **Si optimización empeora:**
   ```bash
   # Usar modelo pre-optimization
   cp models_sac/best_model.zip models_final/
   ```

---

## ⚠️ CRITERIOS DE ROLLBACK {#rollback}

**Rollback triggers por fase:**

### **Fase 2 (Features) - Rollback si:**

| Trigger | Acción |
|---------|--------|
| Max feature importance < 0.10 (igual que baseline) | Deshabilitar grupo que no aportó |
| NaN% > 10% en dataset final | Revisar merge de datos externos |
| Pipeline L3/L4 toma >2x tiempo normal | Optimizar cálculo de features |
| Feature correlation >0.95 entre >5 pares | Eliminar features redundantes |

### **Fase 3 (SAC) - Rollback si:**

| Trigger | Acción |
|---------|--------|
| SAC Sharpe < PPO Sharpe - 0.1 | Mantener solo PPO |
| SAC no converge en 300k timesteps | Ajustar learning rate o buffer |
| SAC usa >2x memoria que PPO | Reducir buffer_size |

### **Fase 4 (Optimization) - Rollback si:**

| Trigger | Acción |
|---------|--------|
| Optimized Sharpe < Pre-optimized | Usar modelo pre-optimization |
| Top 3 hyperparams tienen importance <5% | No hay ganancia, skip optimization |
| 40 trials completan en <2 horas | Trials muy cortos, aumentar timesteps |

### **Fase 5 (Validation) - Rollback si:**

| Trigger | Acción |
|---------|--------|
| WFE < 30% | NO PRODUCCIÓN, volver a investigación |
| OOS Sharpe < 0 | Modelo no generaliza, revisar todo |
| >50% de folds con Sharpe < 0 | Overfitting severo |

---

## 📝 CHECKLIST DE EJECUCIÓN

**Pre-inicio:**
- [ ] Backup completo del proyecto actual
- [ ] Verificar acceso a MinIO buckets
- [ ] Confirmar GPU disponible para entrenamiento
- [ ] Instalar dependencias nuevas: `optuna`, `yfinance`

**Fase 1 (Semana 1):**
- [ ] Crear `notebooks/utils/validation.py` con 3 funciones
- [ ] Añadir 3 celdas nuevas al notebook (6.5, 6.6, 6.7)
- [ ] Ejecutar validación con 10 seeds
- [ ] Ejecutar feature importance analysis
- [ ] Ejecutar baseline comparison
- [ ] Generar `reports/semana1_diagnostico.md`
- [ ] **DECISIÓN: ¿Proceder a Fase 2?**

**Fase 2 (Semanas 2-3):**
- [ ] Crear `config/feature_config.yaml`
- [ ] Modificar `usdcop_m5__04_l3_feature.py` (6 cambios)
- [ ] Modificar `usdcop_m5__05_l4_rlready.py` (3 cambios)
- [ ] Ejecutar pipeline L3 con features enhanced
- [ ] Ejecutar pipeline L4 con 45 obs
- [ ] Verificar datos en bucket `03-l3-ds-usdcop-feature/enhanced/`
- [ ] Verificar datos en bucket `04-l4-ds-usdcop-rlready/enhanced/`
- [ ] Re-ejecutar feature importance con 45 features
- [ ] Comparar con baseline de 17 features
- [ ] **DECISIÓN: ¿Features mejoran?**

**Fase 3 (Semana 4):**
- [ ] Añadir SAC config a `config.py`
- [ ] Crear `train_with_sac()` en `sb3_helpers.py`
- [ ] Añadir celdas 6.1 y 6.8 al notebook
- [ ] Entrenar SAC (300k timesteps)
- [ ] Entrenar PPO (para comparación)
- [ ] Comparar SAC vs PPO (10 seeds cada uno)
- [ ] Generar `reports/semana4_sac_vs_ppo.md`
- [ ] **DECISIÓN: ¿Qué modelo usar?**

**Fase 4 (Semana 5):**
- [ ] Crear `notebooks/utils/optimization.py`
- [ ] Añadir celdas 6.9 y 6.10 al notebook
- [ ] Ejecutar Optuna (40 trials, ~6-8 horas)
- [ ] Guardar resultados en `outputs/optuna/`
- [ ] Re-entrenar con best params (500k timesteps)
- [ ] Evaluar modelo optimizado
- [ ] Generar `reports/semana5_optimization.md`
- [ ] **DECISIÓN: ¿Optimization mejora?**

**Fase 5 (Semana 6):**
- [ ] Añadir `walk_forward_validation()` a `backtesting.py`
- [ ] Añadir `calculate_wfe()` a `metrics.py`
- [ ] Añadir celdas 7.1 y 7.2 al notebook
- [ ] Ejecutar walk-forward (8-10 folds)
- [ ] Calcular WFE
- [ ] Ejecutar OOS test (2024-2025)
- [ ] Generar `reports/FINAL_VALIDATION_REPORT.md`
- [ ] Generar todas las visualizaciones
- [ ] **DECISIÓN FINAL: GO/NO-GO PRODUCCIÓN**

---

## 🎯 MÉTRICAS DE ÉXITO GLOBAL

**Baseline actual (Semana 0):**
```
Sharpe:     -0.42
Win Rate:   27%
Trades/ep:  5.2
Return:     -0.60%
```

**Target Final (Semana 6):**

| Categoría | Mínimo Viable | Target Óptimo | World-Class |
|-----------|--------------|---------------|-------------|
| Sharpe | 0.5 | 1.0 | 1.5+ |
| Win Rate | 48% | 53% | 58%+ |
| WFE | 40% | 60% | 70%+ |
| Return/mes | 2% | 5% | 8%+ |
| Max DD | -25% | -18% | -12% |
| Trades/ep | 6-10 | 8-12 | 10-15 |

**Progreso esperado por semana:**

```
Semana 1 (Validación):         Sharpe ~  -0.40 (confirmar baseline)
Semana 3 (Features):           Sharpe ~   0.30 (features direccionales)
Semana 4 (SAC):                Sharpe ~   0.60 (mejor algoritmo)
Semana 5 (Optimization):       Sharpe ~   0.75 (hyperparams óptimos)
Semana 6 (Validation OOS):     Sharpe ~   0.50 (más conservador en OOS)
```

---

## 📞 CONTACTO Y SOPORTE

**Para cada fase, documentar:**
1. Fecha inicio/fin real
2. Problemas encontrados
3. Soluciones aplicadas
4. Decisiones tomadas
5. Métricas logradas vs target

**En caso de bloqueos:**
- Revisar sección de Rollback correspondiente
- Verificar logs de pipeline/entrenamiento
- Consultar documentación de SB3/Optuna
- Comparar con baseline para confirmar no hay regresión

---

**FIN DEL PLAN ESTRATÉGICO**

*Versión 1.0 - Generado el 2025-11-05*
