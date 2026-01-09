# ✅ FASE 2: L3/L4 FEATURE ENGINEERING - COMPLETADA

**Fecha:** 2025-11-05
**Status:** Archivos modificados, listo para testing
**Duración:** ~45 min

---

## 📦 ARCHIVOS MODIFICADOS (4 archivos)

### **1. airflow/dags/usdcop_m5__04_l3_feature.py**
- **Cambios:**
  - ✅ Añadido import `psycopg2` para conexión a PostgreSQL
  - ✅ Creado `TIER_3_FEATURES` (7 macro features)
  - ✅ Creado `TIER_4_FEATURES` (8 multi-timeframe features)
  - ✅ Actualizado `ALL_FEATURES` de 16 a 31 features
  - ✅ Añadida función `calculate_tier3_macro_features()`
  - ✅ Añadida función `calculate_tier4_mtf_features()`
  - ✅ Añadidas tareas `task_tier3` y `task_tier4` al DAG
  - ✅ Actualizado flujo del DAG: `load >> tier1 >> tier2 >> tier3 >> tier4 >> causality >> forward_ic >> quality >> save`
  - ✅ Actualizado todas las funciones downstream para leer de `calculate_tier4_mtf_features`

**Líneas clave:**
- Línea 80: `import psycopg2`
- Línea 637-645: `TIER_3_FEATURES` (macro)
- Línea 648-660: `TIER_4_FEATURES` (MTF)
- Línea 664: `ALL_FEATURES` expandido a 31 features
- Línea 1423-1623: Función `calculate_tier3_macro_features()`
- Línea 1626-1783: Función `calculate_tier4_mtf_features()`
- Línea 3383-3394: Nuevas tareas del DAG
- Línea 3423: Dependencias actualizadas

---

### **2. airflow/dags/usdcop_m5__05_l4_rlready.py**
- **Cambios:**
  - ✅ Expandido `FEATURE_MAP` de 17 a 32 observations (obs_00 a obs_31)
  - ✅ Añadido obs_17 a obs_23 para macro features
  - ✅ Añadido obs_24 a obs_31 para MTF features

**Líneas clave:**
- Línea 68-116: `FEATURE_MAP` completo con 32 features

**Mapping:**
```python
# TIER 1+2 (obs_00 a obs_13): 14 features L3
# Adicionales (obs_14 a obs_16): hour_sin, hour_cos, spread_proxy
# TIER 3 Macro (obs_17 a obs_23):
#   - wti_return_5, wti_return_20, wti_zscore_60
#   - dxy_return_5, dxy_return_20, dxy_zscore_60
#   - cop_wti_corr_60
# TIER 4 MTF (obs_24 a obs_31):
#   15min: sma_20_15m, ema_12_15m, rsi_14_15m, macd_15m, trend_15m
#   1h: sma_50_1h, vol_regime_1h, adx_14_1h
```

---

### **3. notebooks/utils/config.py**
- **Cambios:**
  - ✅ Actualizado `obs_dim` de 17 a 32

**Líneas clave:**
- Línea 11: `'obs_dim': 32,  # Features del L4 (obs_00 a obs_31) - FASE 2: Expandido de 17 a 32`

---

### **4. notebooks/utils/environments.py**
- **Cambios:**
  - ✅ Observation space ahora se calcula dinámicamente
  - ✅ Detecta automáticamente `n_features` del environment custom
  - ✅ Cambiado hardcoded `(10 * 17)` a `(lags * n_features)`

**Líneas clave:**
- Línea 279-288: Cálculo dinámico del observation space
- Línea 297: Comentario actualizado

**Código nuevo:**
```python
# Detectar automáticamente n_features
lags = self.custom_env.lags
n_features = self.custom_env.n_features
obs_size = lags * n_features
logger.info(f"Observation space: (lags={lags}, n_features={n_features}) → flat={obs_size}")

self.observation_space = spaces.Box(
    low=-5.0, high=5.0, shape=(obs_size,), dtype=np.float32
)
```

---

## 🎯 QUÉ HACE FASE 2

**Objetivo:** Expandir features de 17 a 32 para mejorar poder predictivo del modelo RL

### **Tier 3: Macro Features (7 features)**

**Qué hace:**
- Conecta a PostgreSQL tabla `macro_ohlcv` (creada en Fase 0)
- Descarga datos de WTI y DXY (horario 1h)
- Resample de 1h a 5min con forward-fill
- Merge con dataset principal USD/COP

**Features calculadas:**
1. `wti_return_5`: Retorno 5 bars de WTI (25 min)
2. `wti_return_20`: Retorno 20 bars de WTI (100 min)
3. `wti_zscore_60`: Z-score WTI vs rolling 60-bar mean
4. `dxy_return_5`: Retorno 5 bars de DXY
5. `dxy_return_20`: Retorno 20 bars de DXY
6. `dxy_zscore_60`: Z-score DXY vs rolling 60-bar mean
7. `cop_wti_corr_60`: Correlación rolling 60-bar entre USD/COP y WTI

**Por qué es importante:**
- USD/COP es altamente correlacionado con WTI (commodity currency)
- DXY captura fortaleza del dólar globalmente
- Añade contexto macroeconómico que faltaba

**Handling de datos faltantes:**
- Si no hay datos macro en PostgreSQL → Features se llenan con NaN
- Quality gates de L3 evaluarán si las features son útiles
- No bloquea el pipeline si macro data no está disponible

---

### **Tier 4: Multi-Timeframe Features (8 features)**

**Qué hace:**
- Resample datos 5min a 15min y 1h
- Calcula indicadores técnicos en timeframes superiores
- Merge de vuelta a 5min con forward-fill

**Features 15min (5 features):**
1. `sma_20_15m`: SMA(20) en 15min - trend context
2. `ema_12_15m`: EMA(12) en 15min - momentum
3. `rsi_14_15m`: RSI(14) en 15min - overbought/oversold
4. `macd_15m`: MACD en 15min - trend strength
5. `trend_15m`: **DIRECTIONAL** {-1, 0, +1} - Bullish/Bearish/Neutral
   - +1 si close > sma_20_15m (alcista)
   - -1 si close < sma_20_15m (bajista)
   - 0 si close == sma_20_15m (neutral, raro)

**Features 1h (3 features):**
6. `sma_50_1h`: SMA(50) en 1h - long-term trend
7. `vol_regime_1h`: Volatility regime (ATR ratio to median)
8. `adx_14_1h`: ADX(14) en 1h - trend strength indicator

**Por qué es importante:**
- Triple Screen Trading: 5min (execution) + 15min (trend) + 1h (context)
- Reduce ruido del 5min al incorporar señales de timeframes superiores
- `trend_15m` es **DIRECCIONAL** - única feature con signo direccional
- ADX y vol_regime ayudan a identificar mercados trending vs ranging

---

## 📊 CAMBIOS EN EL PIPELINE

### **Flujo DAG Anterior (Fase 1):**
```
load_data >> tier1 >> tier2 >> causality >> forward_ic >> quality >> save
```

### **Flujo DAG Nuevo (Fase 2):**
```
load_data >> tier1 >> tier2 >> tier3 >> tier4 >> causality >> forward_ic >> quality >> save
```

**Nuevas tareas:**
- `calculate_tier3_macro_features`: Fetch macro data + calculate 7 features
- `calculate_tier4_mtf_features`: Resample + calculate 8 MTF features

**Tiempo estimado:**
- Tier 3 (macro): ~5-10 seg (depende de PostgreSQL query)
- Tier 4 (MTF): ~10-15 seg (resampling + cálculos)
- **Total nuevo:** +15-25 seg al pipeline L3

---

## 🔍 FEATURE IMPORTANCE ESPERADA

Basado en literatura financiera y correlaciones históricas:

### **Macro Features (Tier 3):**
| Feature | Importancia Esperada | Razón |
|---------|---------------------|-------|
| `cop_wti_corr_60` | **ALTA** | COP es commodity currency correlacionada con oil |
| `wti_return_20` | **ALTA** | Tendencia WTI de mediano plazo |
| `dxy_zscore_60` | **MEDIA** | DXY captura fortaleza USD global |
| `wti_return_5` | **MEDIA** | WTI de corto plazo |
| `dxy_return_20` | **MEDIA** | DXY de mediano plazo |
| `wti_zscore_60` | **BAJA** | Z-score menos útil que returns |
| `dxy_return_5` | **BAJA** | DXY de corto plazo menos relevante |

### **MTF Features (Tier 4):**
| Feature | Importancia Esperada | Razón |
|---------|---------------------|-------|
| `trend_15m` | **ALTA** | Direccional - única feature con signo |
| `sma_20_15m` | **ALTA** | Trend context crítico |
| `adx_14_1h` | **ALTA** | Identifica trending markets |
| `vol_regime_1h` | **MEDIA** | Volatility regime importante |
| `rsi_14_15m` | **MEDIA** | Overbought/oversold signals |
| `macd_15m` | **MEDIA** | Momentum confirmation |
| `sma_50_1h` | **BAJA** | Long-term trend menos útil en 5min |
| `ema_12_15m` | **BAJA** | Redundante con sma_20_15m |

**Nota:** Forward IC en L3 validará estas expectativas.

---

## 🚀 PRÓXIMOS PASOS

### **PASO 1: Verificar Datos Macro**

Antes de ejecutar el DAG, verificar que hay datos macro en PostgreSQL:

```bash
# Conectar a PostgreSQL
psql -h localhost -U trading_user -d trading

# Query para verificar macro data
SELECT symbol, COUNT(*), MIN(time), MAX(time)
FROM macro_ohlcv
GROUP BY symbol;

# Expected output:
#  symbol | count |       min        |       max
# --------+-------+------------------+------------------
#  WTI    | 8760  | 2024-01-01 00:00 | 2025-01-01 00:00
#  DXY    | 8760  | 2024-01-01 00:00 | 2025-01-01 00:00
```

**Si no hay datos:**
→ Ejecutar primero el DAG `usdcop_m5__01b_l0_macro_acquire` (Fase 0)
→ O usar script manual: `scripts/upload_macro_manual.py`

---

### **PASO 2: Ejecutar DAG L3**

```bash
# Trigger DAG desde Airflow UI
# O desde CLI:
airflow dags trigger usdcop_m5__04_l3_feature
```

**Monitorear:**
- Task `calculate_tier3_macro_features`: Verificar logs de PostgreSQL connection
- Task `calculate_tier4_mtf_features`: Verificar resampling correcto
- Task `run_causality_tests`: Verificar que 31 features pasan causality tests
- Task `calculate_forward_ic`: Verificar IC de nuevas features < 0.10

**Outputs esperados:**
```
outputs/
  ├── all_features_with_tier3.parquet  (después de Tier 3)
  ├── all_features_final.parquet       (después de Tier 4)
  └── features.parquet                  (final L3 output)
```

---

### **PASO 3: Verificar Feature Statistics**

Después de ejecutar L3, verificar logs:

```
TIER 3 FEATURE STATISTICS (after shift):
  wti_return_5: NaN rate = 5.23%
  wti_return_20: NaN rate = 8.45%
  wti_zscore_60: NaN rate = 12.10%
  ...

TIER 4 FEATURE STATISTICS (after shift):
  sma_20_15m: NaN rate = 3.45%
  trend_15m: NaN rate = 3.45%
  adx_14_1h: NaN rate = 15.20%
  ...
```

**Criterios:**
- ✅ NaN rate < 20% → Feature pasa quality gate
- ⚠️ NaN rate 20-30% → Revisar pero puede pasar
- ❌ NaN rate > 30% → Feature será dropeada por quality gate

---

### **PASO 4: Analizar Forward IC**

Verificar archivo `forward_ic_report.json`:

```json
{
  "median_ic": 0.08,
  "p95_ic": 0.35,
  "features_passed": 28,
  "features_dropped": 3,
  "dropped_features": ["ema_12_15m", "wti_zscore_60", "dxy_return_5"],
  "trainable_features": [
    "hl_range_surprise",
    "atr_surprise",
    ...
    "cop_wti_corr_60",
    "trend_15m",
    "adx_14_1h"
  ]
}
```

**Criterios:**
- ✅ Median IC < 0.15 → Sistema causal
- ✅ P95 IC < 0.40 → Sin leakage severo
- ✅ Features dropped < 5 → Mayoría de features útiles

---

### **PASO 5: Ejecutar DAG L4**

Una vez L3 complete exitosamente:

```bash
# Trigger L4
airflow dags trigger usdcop_m5__05_l4_rlready
```

**Verificar:**
- Observation space expandido correctamente
- Normalización aplicada a todas las 32 features
- CSV y Parquet generados con 32 columns (obs_00 a obs_31)

**Outputs esperados:**
```
outputs/
  ├── features.parquet          (32 obs columns)
  ├── features.csv              (human-readable)
  └── feature_spec.json         (metadata)
```

---

### **PASO 6: Actualizar Notebook RL**

Modificar `notebooks/usdcop_rl_notebook.ipynb`:

```python
# ANTES (Fase 1):
from utils.config import CONFIG
obs_dim = CONFIG['obs_dim']  # 17

# DESPUÉS (Fase 2):
from utils.config import CONFIG
obs_dim = CONFIG['obs_dim']  # 32 (auto-updated)

# Environment ahora detecta automáticamente 32 features
env = TradingEnvironmentL4Wrapper(df=df_train)
print(f"Observation space: {env.observation_space.shape}")
# Output: (320,)  → (10 lags * 32 features)
```

**No requiere cambios adicionales** - el código es compatible con cualquier número de features.

---

### **PASO 7: Re-entrenar Modelo**

```python
# En el notebook
from stable_baselines3 import PPO

# PPO detectará automáticamente el nuevo observation space
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# Entrenar con 32 features
model.learn(total_timesteps=50_000)
```

**Expectativa:**
- Sharpe actual: -0.42
- Sharpe esperado (Fase 2): **+0.3 a +0.6**
- Mejora esperada: **+0.7 a +1.0 puntos Sharpe**

---

## ⚠️ TROUBLESHOOTING

### **Error: "No macro data found in PostgreSQL"**

**Síntoma:**
```
TIER 3: No macro data found in PostgreSQL. Skipping Tier 3 features.
```

**Solución:**
1. Verificar que el DAG `usdcop_m5__01b_l0_macro_acquire` se ejecutó
2. O ejecutar script manual:
   ```bash
   python scripts/upload_macro_manual.py --csv investing_wti.csv --symbol WTI
   python scripts/upload_macro_manual.py --csv investing_dxy.csv --symbol DXY
   ```

**Impacto:**
- Tier 3 features se llenan con NaN
- Quality gates las droparán
- No bloquea el pipeline, pero reduce features de 32 a 25

---

### **Error: "psycopg2.OperationalError: could not connect to server"**

**Síntoma:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Solución:**
```bash
# Verificar que PostgreSQL está corriendo
docker ps | grep postgres

# Si no está corriendo, iniciar:
docker-compose up -d timescaledb
```

---

### **Error: "Not enough data for 15min calculations"**

**Síntoma:**
```
TIER 4: Not enough data for 15min calculations
```

**Causa:** Episodios muy cortos (< 60 steps)

**Solución:**
- Episodios con < 60 steps → MTF features = NaN (esperado)
- Quality gates manejarán esto
- Si todos episodios son cortos → Aumentar `episode_length` en L2

---

### **Error: "observation_space shape mismatch"**

**Síntoma:**
```
ValueError: observation_space shape (170,) does not match observation (320,)
```

**Causa:** Config no actualizado o environment cacheado

**Solución:**
```python
# Restart kernel en Jupyter
# Reimport config
from utils.config import CONFIG
print(CONFIG['obs_dim'])  # Debe ser 32

# Re-crear environment
env = TradingEnvironmentL4Wrapper(df=df_train)
print(env.observation_space.shape)  # Debe ser (320,)
```

---

## 📈 PROGRESO TOTAL DEL PROYECTO

```
✅ Fase 0: Pipeline L0 Macro Data       [COMPLETADA]
✅ Fase 1: Validación y Diagnóstico     [COMPLETADA]
✅ Fase 2: L3/L4 Feature Engineering    [COMPLETADA - HOY]
⬜ Fase 3: Reward Shaping + SAC         [Siguiente]
⬜ Fase 4: Optuna Optimization          [Siguiente]
⬜ Fase 5: Walk-Forward Validation      [Final]
```

**Mejora acumulada esperada:**
- Fase 0: Infraestructura (no cambio en Sharpe)
- Fase 1: Diagnóstico (no cambio en Sharpe)
- **Fase 2: +0.7 a +1.0 Sharpe** (de -0.42 a +0.3/+0.6)
- Fase 3: +0.2 a +0.4 Sharpe (reward shaping + SAC)
- Fase 4: +0.1 a +0.3 Sharpe (hyperparameter tuning)
- Fase 5: Validación final

**Meta final:** Sharpe de +0.8 a +1.5

---

## 🔗 ARCHIVOS RELACIONADOS

### **Fase 2 (este archivo):**
```
1. FASE_2_COMPLETADA.md               [ESTE ARCHIVO - resumen]
2. airflow/dags/usdcop_m5__04_l3_feature.py  [Modificado - Tier 3+4]
3. airflow/dags/usdcop_m5__05_l4_rlready.py  [Modificado - FEATURE_MAP]
4. notebooks/utils/config.py                 [Modificado - obs_dim]
5. notebooks/utils/environments.py           [Modificado - dynamic obs space]
```

### **Fases anteriores:**
```
6. FASE_0_COMPLETADA.md                [Fase 0 - Macro pipeline]
7. FASE_0_INSTRUCCIONES.md             [Fase 0 - Instrucciones detalladas]
8. FASE_1_COMPLETADA.md                [Fase 1 - Validación]
9. FASE_1_INSTRUCCIONES.md             [Fase 1 - Instrucciones detalladas]
```

### **Documentación técnica:**
```
10. ADDENDUM_MACRO_FEATURES.md         [Especificación macro features]
11. ADDENDUM_MTF_SPECIFICATION.md      [Especificación MTF features]
12. PLAN_ESTRATEGICO_v2_UPDATES.md     [Plan completo Fases 2-5]
```

---

## ✅ CHECKLIST COMPLETO

**Archivos modificados:**
- [x] `airflow/dags/usdcop_m5__04_l3_feature.py` (Tier 3+4 functions)
- [x] `airflow/dags/usdcop_m5__05_l4_rlready.py` (FEATURE_MAP expanded)
- [x] `notebooks/utils/config.py` (obs_dim: 17 → 32)
- [x] `notebooks/utils/environments.py` (dynamic observation space)
- [x] `FASE_2_COMPLETADA.md` (este documento)

**Para ejecutar:**
- [ ] Verificar datos macro en PostgreSQL
- [ ] Ejecutar DAG `usdcop_m5__04_l3_feature`
- [ ] Verificar logs de Tier 3 y Tier 4
- [ ] Analizar forward IC report
- [ ] Ejecutar DAG `usdcop_m5__05_l4_rlready`
- [ ] Verificar L4 output (32 features)
- [ ] Actualizar notebook RL (si necesario)
- [ ] Re-entrenar modelo con 32 features
- [ ] Evaluar mejora en Sharpe ratio

**Decisión siguiente:**
- [ ] Si Sharpe > 0.3 → Continuar con Fase 3 (Reward Shaping)
- [ ] Si Sharpe < 0.3 → Revisar feature importance y ajustar
- [ ] Si mejora < 0.5 → Considerar añadir más features técnicas

---

## 🎉 RESUMEN EJECUTIVO

**Fase 2 COMPLETADA:**
- ✅ Expandido features de 17 a 32 (+88% features)
- ✅ Añadido 7 macro features (WTI, DXY)
- ✅ Añadido 8 MTF features (15min, 1h)
- ✅ Pipeline L3+L4 actualizado
- ✅ Environment detection automática

**Cambios key:**
- `ALL_FEATURES`: 16 → 31 (en L3)
- `FEATURE_MAP`: 17 → 32 (en L4)
- `obs_dim`: 17 → 32 (en config)
- Observation space: (170,) → (320,)

**Próximo paso:**
- **Ejecutar pipeline completo L3+L4**
- **Verificar que 32 features se generan correctamente**
- **Re-entrenar modelo y medir mejora en Sharpe**

---

**FIN DEL DOCUMENTO**

*Fase 2 completada - 2025-11-05*
*Próximo: Ejecutar DAGs y medir impacto, luego Fase 3 (Reward Shaping)*
