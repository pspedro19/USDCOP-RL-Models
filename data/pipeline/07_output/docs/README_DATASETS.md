# Documentacion de Datasets RL para USD/COP

## Informacion General

| Atributo | Valor |
|----------|-------|
| **Activo** | USD/COP (Dolar estadounidense / Peso colombiano) |
| **Frecuencia** | 5 minutos |
| **Horario** | 8:00 - 12:55 COT (13:00 - 17:55 UTC), Lunes a Viernes |
| **Rango de fechas** | 2020-03-02 a 2025-10-29 |
| **Total filas** | 83,886 por dataset |
| **Total dias** | 1,472 dias de trading |
| **Barras por dia** | ~57 barras (60 intervalos menos gaps) |

---

## Pipeline de Generacion

```
FUENTES DE DATOS
================
INPUT/HISTORICAL/          INPUT/RECENT/           backups/database/
  - DATASET_MACRO_DAILY      - datos_diarios_hpc     - usdcop_m5_ohlcv_*.csv.gz
  - DATASET_MACRO_MONTHLY    - datos_mensuales_hpc
  - DATASET_MACRO_QUARTERLY  - datos_trimestrales_hpc

                    |
                    v
            01_clean_macro_data.py
            - Combina historico + reciente
            - Reciente sobrescribe en overlapping
            - Elimina fines de semana (diario)
            - Normaliza a inicio de mes (mensual)
            - Desde 2019-Q4 (trimestral)
                    |
                    v
              OUTPUT/
              - MACRO_DAILY_CLEAN.csv
              - MACRO_MONTHLY_CLEAN.csv
              - MACRO_QUARTERLY_CLEAN.csv
                    |
                    v
            02_resample_consolidated.py
            - merge_asof con direction='backward' (sin look-ahead bias)
            - ffill controlado (5d diario, 35d mensual, 95d trimestral)
            - Expansion a grid 5min (60 intervalos/dia)
                    |
                    v
              OUTPUT_V2/
              - MACRO_DAILY_CONSOLIDATED.csv
              - MACRO_5MIN_CONSOLIDATED.csv
                    |
                    v
            03_create_rl_datasets_v3.py
            - Merge con OHLCV de USD/COP
            - Calculo de indicadores tecnicos
            - Features macro derivados
            - Normalizacion por tipo
            - Generacion de 5 datasets
                    |
                    v
              OUTPUT_RL/
              - RL_DS1_MINIMAL.csv
              - RL_DS2_TECHNICAL_MTF.csv
              - RL_DS3_MACRO_CORE.csv      <-- RECOMENDADO
              - RL_DS4_COST_AWARE.csv
              - RL_DS5_REGIME.csv
```

---

## Estrategias de Preparacion de Datos

### 1. Manejo de Missing Values

| Tipo de Variable | Estrategia | Razon |
|------------------|------------|-------|
| OHLC | Sin gaps | Fuente primaria completa |
| Indicadores tecnicos | Warmup natural | Fecha corte 2020-03-01 elimina NaN |
| Macro diarias | `ffill()` sin limite | Son indicadores lentos, mantener ultimo valor |
| Cross-pairs (FX) | `ffill()` sin limite | Mismo razonamiento |
| Derivados (z-score, etc) | `ffill()` despues de calcular | Propagar ultimo valor conocido |

### 2. Normalizacion por Tipo de Feature

| Tipo | Metodo | Rango Final |
|------|--------|-------------|
| **Retornos** | `log(close/close.shift(n))` + clip | [-0.05, 0.05] |
| **Z-scores** | `(x - rolling_mean) / rolling_std` + clip | [-4, +4] |
| **RSI** | Sin normalizar | [0, 100] |
| **ADX** | Sin normalizar | [0, 100] |
| **BB position** | `(close - lower) / (upper - lower)` | [0, 1] |
| **Categoricas** | Label encoding | [0, 1, 2, 3] |
| **Ciclicas** | `sin(2*pi*x/periodo)`, `cos(2*pi*x/periodo)` | [-1, +1] |

### 3. Prevencion de Look-Ahead Bias

- `merge_asof(direction='backward')`: Solo usa datos disponibles en el pasado
- ffill controlado con limites por frecuencia
- Calculo de indicadores usa solo datos historicos

### 4. Manejo de Outliers

- Retornos: clip a ±5% (movimientos mayores son raros pero reales)
- Z-scores: clip a ±4 desviaciones estandar
- VIX regime: categorizado en bins, no valores extremos

### 5. Columnas Blacklisted (Nunca incluidas)

```python
BLACKLIST = [
    'volume',           # 100% ceros - USD/COP es OTC, no hay volume
    'obv_change_z',     # Depende de volume (ceros)
    'macd_hist_z_old',  # Bug en calculo (signal=1 causaba hist=0)
    'macd_1h_z_old',    # Mismo bug
    'dxy_cop_corr_z',   # Division por cero generaba infinitos
    't_in_session',     # Sin varianza (bug en calculo UTC/COT)
    'risk_score_z',     # Outliers extremos, bajo valor predictivo
]
```

---

## Descripcion de Datasets

### DS1_MINIMAL (15 columnas, 10 features)

**Filosofia**: Baseline minimo para validar pipeline y establecer rendimiento base.

**Cuando usar**:
- Primera iteracion del modelo
- Validar que el ambiente RL funciona
- Comparar contra modelos mas complejos

**Columnas**:

| Columna | Tipo | Descripcion | Rango |
|---------|------|-------------|-------|
| `timestamp` | datetime | Fecha y hora UTC | - |
| `open` | float | Precio apertura | ~3000-5000 |
| `high` | float | Precio maximo | ~3000-5000 |
| `low` | float | Precio minimo | ~3000-5000 |
| `close` | float | Precio cierre | ~3000-5000 |
| `log_ret_5m` | float | Log return 5 minutos | [-0.05, 0.05] |
| `log_ret_1h` | float | Log return 1 hora (12 barras) | [-0.05, 0.05] |
| `log_ret_4h` | float | Log return 4 horas (48 barras) | [-0.05, 0.05] |
| `rsi_9` | float | RSI periodo 9 | [0, 100] |
| `atr_pct` | float | ATR como % del precio | [0, ~2] |
| `bb_position` | float | Posicion en Bollinger Bands | [0, 1] |
| `dxy_z` | float | DXY normalizado (z-score 50 bars) | [-4, 4] |
| `vix_level` | float | VIX sin normalizar (interpretable) | [10, 80+] |
| `hour_sin` | float | Codificacion ciclica hora (sin) | [-1, 1] |
| `hour_cos` | float | Codificacion ciclica hora (cos) | [-1, 1] |

---

### DS2_TECHNICAL_MTF (19 columnas, 14 features)

**Filosofia**: Multi-timeframe tecnico puro, sin variables macro.

**Cuando usar**:
- Estrategias trend-following
- Si macro variables no mejoran resultados
- Comparar tecnico vs macro

**Columnas adicionales vs DS1**:

| Columna | Tipo | Descripcion | Rango |
|---------|------|-------------|-------|
| `log_ret_15m` | float | Log return 15 minutos (3 barras) | [-0.05, 0.05] |
| `rsi_9_15m` | float | RSI calculado en timeframe 15min | [0, 100] |
| `rsi_9_1h` | float | RSI calculado en timeframe 1H | [0, 100] |
| `atr_pct_1h` | float | ATR% en timeframe 1H | [0, ~2] |
| `adx_14` | float | ADX periodo 14 (fuerza de tendencia) | [0, 100] |
| `sma_ratio` | float | close / SMA(20) - 1 | [-0.1, 0.1] |

---

### DS3_MACRO_CORE (24 columnas, 19 features) - RECOMENDADO

**Filosofia**: Balance optimo tecnico + macro para USD/COP como moneda emergente commodity-linked.

**Cuando usar**:
- **PRODUCCION** - Dataset principal recomendado
- Captura regimenes risk-on/risk-off
- Entender drivers fundamentales de COP

**Por que es el mejor para USD/COP**:
1. **DXY**: Dolar fuerte = COP debil (correlacion ~0.6)
2. **VIX**: Crisis = fuga de emergentes = COP debil
3. **Brent**: Colombia exporta 40-50% petroleo
4. **EMBI**: Riesgo pais Colombia afecta flujos
5. **USD/MXN**: Proxy de EM, a veces lidera movimientos

**Columnas adicionales vs DS1**:

| Columna | Tipo | Descripcion | Rango |
|---------|------|-------------|-------|
| `adx_14` | float | Fuerza de tendencia | [0, 100] |
| `dxy_z` | float | DXY normalizado | [-4, 4] |
| `dxy_change_1d` | float | Cambio % DXY 1 dia | [-0.1, 0.1] |
| `dxy_mom_5d` | float | Momentum DXY 5 dias | [-0.1, 0.1] |
| `vix_z` | float | VIX normalizado | [-4, 4] |
| `vix_regime` | int | 0=calm(<20), 1=elevated(20-25), 2=stress(25-30), 3=crisis(>30) | [0, 3] |
| `embi_z` | float | EMBI Colombia normalizado | [-4, 4] |
| `brent_change_1d` | float | Cambio % Brent 1 dia | [-0.1, 0.1] |
| `brent_vol_5d` | float | Volatilidad Brent 5 dias | [0, 0.05] |
| `rate_spread` | float | Spread tasas Colombia-USA normalizado | [-4, 4] |
| `usdmxn_ret_1h` | float | Retorno USD/MXN 1 hora | [-0.1, 0.1] |

---

### DS4_COST_AWARE (21 columnas, 16 features)

**Filosofia**: Filtros para reducir overtrading y costos de transaccion.

**Cuando usar**:
- Si DS3 genera demasiados trades (>3/dia)
- Si profit factor es bajo (<1.3)
- Mercados con spreads altos

**Columnas especiales**:

| Columna | Tipo | Descripcion | Rango |
|---------|------|-------------|-------|
| `ret_lag_1` | float | Retorno hace 1 barra | [-0.05, 0.05] |
| `ret_lag_3` | float | Retorno hace 3 barras | [-0.05, 0.05] |
| `ret_atr_adj` | float | Retorno / ATR (normalizado por vol) | [-3, 3] |
| `momentum_6` | float | Suma retornos ultimas 6 barras | [-0.1, 0.1] |
| `rsi_extreme` | int | 1 si RSI < 25 o > 75, else 0 | [0, 1] |
| `adx_strong` | int | 1 si ADX > 25, else 0 | [0, 1] |
| `atr_percentile` | float | Percentil ATR vs ultimos 50 bars | [0, 1] |
| `vol_regime` | int | 0=low, 1=normal, 2=high | [0, 2] |
| `session_liquid` | int | 1 si 8am-1pm COT (mas liquido), else 0 | [0, 1] |

---

### DS5_REGIME (30 columnas, 25 features)

**Filosofia**: Maximo contexto para detectar cambios de regimen de mercado.

**Cuando usar**:
- Arquitecturas con attention (Transformer, BiLSTM-Attention)
- Si DS3 muestra dependencia fuerte de regimen
- Investigacion de features

**Columnas adicionales vs DS3**:

| Columna | Tipo | Descripcion | Rango |
|---------|------|-------------|-------|
| `log_ret_15m` | float | Log return 15 minutos | [-0.05, 0.05] |
| `dxy_vol_5d` | float | Volatilidad DXY 5 dias | [0, 0.02] |
| `embi_change_5d` | float | Cambio % EMBI 5 dias | [-0.1, 0.1] |
| `curve_slope` | float | UST10Y - UST2Y (yield curve) | [-1, 3] |
| `usdclp_ret_1h` | float | Retorno USD/CLP 1 hora (Chile proxy EM) | [-0.1, 0.1] |
| `dow_sin` | float | Dia de semana codificacion ciclica | [-1, 1] |

---

## Diccionario de Variables Macro

### Fuentes Diarias

| Variable | Codigo Interno | Fuente | Impacto USD/COP |
|----------|----------------|--------|-----------------|
| DXY Index | `FXRT_INDEX_DXY_USA_D_DXY` | Bloomberg/Reuters | MUY ALTO - Correlacion ~0.6 |
| VIX | `VOLT_VIX_USA_D_VIX` | CBOE | MUY ALTO - Flight to safety |
| Brent Oil | `COMM_OIL_BRENT_GLB_D_BRENT` | ICE | MUY ALTO - 40% exportaciones |
| UST 10Y | `FINC_BOND_YIELD10Y_USA_D_UST10Y` | Treasury | ALTO - Spread tasas |
| UST 2Y | `FINC_BOND_YIELD2Y_USA_D_DGS2` | Treasury | MEDIO - Yield curve |
| EMBI Colombia | `CRSK_SPREAD_EMBI_COL_D_EMBI` | JP Morgan | MUY ALTO - Riesgo pais |
| USD/MXN | `FXRT_SPOT_USDMXN_MEX_D_USDMXN` | Reuters | ALTO - Proxy EM, lidera COP |
| USD/CLP | `FXRT_SPOT_USDCLP_CHL_D_USDCLP` | Reuters | MEDIO - Commodity currency |

### Interpretacion de VIX Regime

| Valor | Nombre | VIX Range | Significado |
|-------|--------|-----------|-------------|
| 0 | Calm | < 20 | Mercado tranquilo, risk-on |
| 1 | Elevated | 20-25 | Cautela moderada |
| 2 | Stress | 25-30 | Estres de mercado |
| 3 | Crisis | > 30 | Panico, flight to safety |

---

## Parametros Tecnicos

```python
# Indicadores
RSI_PERIOD = 9
ATR_PERIOD = 10
ADX_PERIOD = 14
SMA_PERIOD = 20
BB_PERIOD = 10
BB_STD = 2.0
MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 5

# Normalizacion
ZSCORE_WINDOW = 50  # Ventana rolling para z-score
CLIP_STD = 4.0      # Limite para z-scores

# VIX thresholds
VIX_THRESHOLDS = [20, 25, 30]

# Horario liquido Colombia
LIQUID_HOURS = (8, 13)  # 8am - 1pm COT
```

---

## Orden de Experimentacion Recomendado

```
1. DS1_MINIMAL
   Objetivo: Sharpe > 0.3
   Si funciona: Pipeline validado, continuar
   Si no funciona: Revisar ambiente RL

2. DS3_MACRO_CORE *** PRIORIDAD ***
   Objetivo: Sharpe > 0.5, MDD < 20%
   Si funciona: USAR EN PRODUCCION
   Si no funciona: Probar DS4

3. DS4_COST_AWARE
   Objetivo: Reducir trades, mejorar profit factor
   Usar si: DS3 tiene overtrading

4. DS2_TECHNICAL_MTF
   Usar si: Variables macro no aportan valor

5. DS5_REGIME
   Usar si: Arquitectura con attention
   Cuidado: Mayor riesgo de overfitting
```

---

## Metricas de Exito Minimas

| Dataset | Sharpe | MDD | Win Rate | Profit Factor |
|---------|--------|-----|----------|---------------|
| DS1 | > 0.3 | < 25% | > 48% | > 1.2 |
| DS2 | > 0.4 | < 22% | > 50% | > 1.3 |
| DS3 | > 0.5 | < 20% | > 52% | > 1.5 |
| DS4 | > 0.4 | < 18% | > 55% | > 1.6 |
| DS5 | > 0.5 | < 20% | > 52% | > 1.5 |

Si no alcanzas estos minimos, hay problema en el pipeline, no en el dataset.

---

## Checklist Pre-Entrenamiento

- [ ] Verificar que usa archivos de OUTPUT_RL/
- [ ] Verificar 0 nulls, 0 infinitos
- [ ] Verificar rango de fechas (desde 2020-03-02)
- [ ] Verificar split temporal (no random)
- [ ] Configurar walk-forward: 12 meses train, 3 meses test
- [ ] Usar algoritmo PPO (mejor que SAC para FX)

---

*Generado: 2025-11-27*
*Script: 03_create_rl_datasets.py*
