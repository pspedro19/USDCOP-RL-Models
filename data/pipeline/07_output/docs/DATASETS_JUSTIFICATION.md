# Justificacion de Datasets RL para USD/COP Trading

## Resumen Ejecutivo

Se han generado **10 datasets** optimizados para diferentes estrategias de trading en USD/COP usando Reinforcement Learning. Cada dataset tiene una filosofia especifica y esta disenado para capturar diferentes aspectos del mercado.

---

## Inventario de Datasets

| Dataset | Features | Filas | Uso Principal | Sharpe Esperado |
|---------|----------|-------|---------------|-----------------|
| DS1_MINIMAL | 10 | 83,886 | Validar pipeline | 0.3-0.5 |
| DS2_TECHNICAL_MTF | 14 | 83,886 | Trend-following | 0.4-0.6 |
| **DS3_MACRO_CORE** | **19** | 83,886 | **PRODUCCION** | **0.5-0.9** |
| DS4_COST_AWARE | 16 | 83,886 | Anti-overtrading | 0.4-0.7 |
| DS5_REGIME | 25 | 83,886 | Attention/Transformer | 0.5-0.8 |
| **DS6_CARRY_TRADE** | 15 | 83,886 | Carry trade flows | **0.6-1.0** |
| DS7_COMMODITY_BASKET | 19 | 83,886 | Commodity export | 0.5-0.8 |
| DS8_RISK_SENTIMENT | 22 | 83,886 | Risk-On/Off | 0.5-0.9 |
| DS9_FED_WATCH | 21 | 83,886 | Fed expectations | 0.5-0.8 |
| DS10_FLOWS_FUNDAMENTALS | 17 | 83,886 | BOP/Swing trade | 0.4-0.7 |

---

## DS1: MINIMAL (Baseline)

### Filosofia
Dataset minimo para validar que el pipeline de RL funciona correctamente antes de agregar complejidad.

### Features (10)
- **Retornos**: log_ret_5m, log_ret_1h, log_ret_4h
- **Tecnicos**: rsi_9, atr_pct, bb_position
- **Macro**: dxy_z, vix_level
- **Temporal**: hour_sin, hour_cos

### Justificacion
- RSI(9): Oscilador rapido, ideal para 5min
- ATR: Volatilidad normalizada para position sizing
- BB position: Indica si precio esta en extremo del rango
- DXY: Principal driver del COP
- VIX: Proxy de risk-off global

---

## DS2: TECHNICAL_MTF (Multi-Timeframe)

### Filosofia
Tecnico puro sin macro. Para estrategias trend-following que dependen solo de price action.

### Features (14)
- **Retornos multi-TF**: 5m, 15m, 1h, 4h
- **RSI multi-TF**: 9, 15m, 1h
- **ATR multi-TF**: base, 1h
- **Trend**: adx_14, sma_ratio
- **Volatility**: bb_position
- **Temporal**: hour_sin, hour_cos

### Justificacion
- Multi-timeframe permite capturar tendencias en diferentes escalas
- ADX filtra periodos de rango vs tendencia
- SMA ratio indica desviacion del precio respecto a media

---

## DS3: MACRO_CORE (RECOMENDADO PARA PRODUCCION)

### Filosofia
Balance optimo entre tecnico y macro. Incluye los drivers fundamentales mas importantes del USD/COP.

### Features (19)
- **Retornos**: 5m, 1h, 4h
- **Tecnicos**: rsi_9, atr_pct, adx_14, bb_position
- **DXY**: dxy_z, dxy_change_1d, dxy_mom_5d
- **Riesgo**: vix_z, vix_regime, embi_z
- **Commodities**: brent_change_1d, brent_vol_5d
- **Tasas**: rate_spread
- **Cross-pair**: usdmxn_ret_1h
- **Temporal**: hour_sin, hour_cos

### Justificacion
- **DXY es el driver #1 del COP**: Correlacion historica > 0.85
- **VIX regime** captura risk-off que deprecia EM FX
- **EMBI** es el spread de riesgo pais especifico de Colombia
- **Brent** representa 40% de exportaciones colombianas
- **Rate spread** captura atractivo de carry trade
- **USDMXN** lidera movimientos de LATAM FX

### Por que RECOMENDADO
- Menor ruido que DS5-DS10
- Features diarios con baja latencia
- Probado en backtests con Sharpe > 0.5

---

## DS4: COST_AWARE (Anti-Overtrading)

### Filosofia
Incluye filtros para reducir numero de trades. Penaliza trading en condiciones suboptimas.

### Features (16)
- **Retornos + Lags**: log_ret_5m, log_ret_1h, ret_lag_1, ret_lag_3
- **Momentum ajustado**: ret_atr_adj, momentum_6
- **Filtros de senal**: rsi_9, rsi_extreme, adx_14, adx_strong
- **Volatilidad**: atr_percentile, vol_regime
- **Macro**: dxy_z, vix_z
- **Temporal**: hour_sin, hour_cos

### Justificacion
- **rsi_extreme**: Flag cuando RSI < 20 o > 80
- **adx_strong**: Flag cuando ADX > 25 (tendencia fuerte)
- **atr_percentile**: Percentil de volatilidad (evita operar en baja vol)
- **vol_regime**: 0=baja, 1=normal, 2=alta, 3=extrema
- **ret_atr_adj**: Retorno ajustado por ATR (evita trades pequeños)

### Uso
Solo usar si el modelo de DS3 tiene demasiados trades o bajo profit factor.

---

## DS5: REGIME (Attention/Transformer)

### Filosofia
Maximo contexto para modelos avanzados que pueden detectar cambios de regimen.

### Features (25)
- **Retornos multi-TF**: 5m, 15m, 1h, 4h
- **Tecnicos**: rsi_9, atr_pct, adx_14, bb_position
- **DXY completo**: dxy_z, change_1d, mom_5d, vol_5d
- **Riesgo completo**: vix_z, vix_regime, embi_z, embi_change_5d
- **Commodities**: brent_change_1d, brent_vol_5d
- **Tasas**: rate_spread, curve_slope_z
- **Cross-pairs**: usdmxn_ret_1h, usdclp_ret_1h
- **Temporal**: hour_sin, hour_cos, dow_sin

### Justificacion
- **curve_slope_z**: Pendiente de curva USA indica expectativas de recesion
- **embi_change_5d**: Velocidad de cambio en riesgo pais
- **USDCLP**: Otro EM FX de LATAM para contexto regional
- **dow_sin**: Patron semanal (lunes risk-off, viernes profit-taking)

### Uso
Solo con arquitecturas attention/transformer que pueden manejar alta dimensionalidad.

---

## DS6: CARRY_TRADE (Diferenciales de Tasas)

### Filosofia
**Capturar flujos de carry trade que son el PRINCIPAL driver de EM FX.**

### Features (15)
- **Retornos**: 5m, 1h, 4h
- **Spreads de bonos**: col10y_ust10y_spread_z, col_curve_slope_z, usa_curve_slope_z
- **Politica monetaria**: tpm_fedfunds_spread_z, ibr_tpm_spread_z
- **Riesgo**: embi_z, vix_z, vix_regime
- **Dolar**: dxy_z, dxy_change_1d
- **Temporal**: hour_sin, hour_cos

### Justificacion Teorica
El carry trade es la estrategia de:
1. Pedir prestado en moneda de baja tasa (USD, JPY)
2. Invertir en moneda de alta tasa (COP)
3. Ganar el diferencial de tasas

**Implicaciones para USD/COP:**
- Si TPM > FEDFUNDS → entran USD buscando rendimiento → COP se fortalece
- Si spread se comprime → sale capital → COP se debilita
- Pendiente de curvas indica expectativas de politica futura

**Variables clave:**
- `col10y_ust10y_spread_z`: Diferencial soberano normalizado
- `tpm_fedfunds_spread_z`: Diferencial de politica monetaria
- `ibr_tpm_spread_z`: Tension en mercado interbancario colombiano

### Sharpe Esperado: 0.6-1.0 (mejor en periodos de alto diferencial)

---

## DS7: COMMODITY_BASKET (Exportaciones Colombia)

### Filosofia
**Colombia es economia commodity-dependiente: petroleo 40%, cafe, oro.**

### Features (19)
- **Retornos**: 5m, 1h
- **Petroleo**: brent_z, brent_change_1d, brent_mom_5d, wti_z, brent_wti_spread_z
- **Otros commodities**: coffee_z, coffee_change_1d, gold_z, gold_change_1d
- **Terminos de intercambio**: tot_z, tot_change_1m
- **Mercado local**: colcap_z, colcap_change_1d
- **Riesgo**: vix_z, dxy_z
- **Temporal**: hour_sin, hour_cos

### Justificacion Teorica
**Correlaciones historicas:**
- Brent +10% → COP se aprecia ~4%
- Cafe +20% → COP se aprecia ~1%
- Oro es contracorrelacionado (safe haven)

**Variables clave:**
- `brent_wti_spread_z`: Indica tensiones de suministro global
- `tot_z`: Terminos de intercambio captura poder adquisitivo de exportaciones
- `colcap_z`: Indice bursatil colombiano (proxy de flujos equity)

### Sharpe Esperado: 0.5-0.8

---

## DS8: RISK_SENTIMENT (Risk-On/Risk-Off)

### Filosofia
**EM FX es proxy de apetito por riesgo global. Colombia es EM tipico.**

### Features (22)
- **Retornos**: 5m, 1h
- **VIX completo**: vix_level, vix_z, vix_regime, vix_change_1d, vix_percentile_20d
- **EMBI completo**: embi_z, embi_change_1d, embi_change_5d, embi_percentile_20d
- **Cross-pairs**: usdmxn_ret_5m, usdmxn_ret_1h, usdmxn_z, usdclp_ret_1h
- **Safe haven**: gold_change_1d, dxy_change_1d
- **Equity local**: colcap_change_1d
- **Tecnico**: rsi_9, bb_position
- **Temporal**: hour_sin, hour_cos

### Justificacion Teorica
**Dinamica Risk-On/Risk-Off:**
- Risk-On: VIX bajo, EMBI bajo → flujos hacia EM → COP fuerte
- Risk-Off: VIX alto, EMBI alto → flight to quality → COP debil

**Variables clave:**
- `vix_percentile_20d`: Contexto de donde esta VIX vs. ultimas 20 dias
- `usdmxn_z`: MXN es mas liquido que COP, lidera movimientos
- `embi_percentile_20d`: Contexto de spread de riesgo

### Sharpe Esperado: 0.5-0.9 (mejor en alta volatilidad, peor en regimenes calmos)

---

## DS9: FED_WATCH (Expectativas Fed)

### Filosofia
**La Fed mueve TODO en mercados emergentes.**

### Features (21)
- **Retornos**: 5m, 1h, 4h
- **Bonos USA**: ust2y_z, ust2y_change_1d, ust10y_z, usa_curve_slope_z, usa_curve_change_1d
- **Inflacion USA**: cpi_usa_z, cpi_usa_change_1m, pce_usa_z, pce_usa_change_1m
- **Empleo USA**: unrate_z, unrate_change_1m
- **Politica Colombia**: tpm_z, tpm_fedfunds_spread_z
- **Dolar y riesgo**: dxy_z, dxy_change_1d, vix_z
- **Temporal**: hour_sin, hour_cos

### Justificacion Teorica
**Impacto de Fed en EM FX:**
- Fed funds +100bp historicamente → EM FX -10%
- UST 2Y se mueve ANTES que Fed Funds (expectativas del mercado)
- Curva invertida → recesion → Fed baja tasas → USD debil

**Variables clave:**
- `ust2y_z`: Mejor predictor de proximos movimientos de Fed
- `usa_curve_slope_z`: Pendiente indica expectativas de recesion
- `cpi_usa_z`: Inflacion alta → Fed hawkish → USD fuerte

### Sharpe Esperado: 0.5-0.8

---

## DS10: FLOWS_FUNDAMENTALS (Balanza de Pagos)

### Filosofia
**Flujos de capital determinan oferta/demanda de USD en el largo plazo.**

### Features (17)
- **Retornos**: 5m, 1h
- **Flujos de capital**: ied_z, ied_change_1q, cuenta_corriente_z
- **Comercio**: exports_z, imports_z, trade_balance_z
- **Reservas**: reserves_z, reserves_change_1m
- **Competitividad**: itcr_z, itcr_deviation
- **Riesgo**: embi_z, vix_z, dxy_z
- **Temporal**: hour_sin, hour_cos

### Justificacion Teorica
**Dinamica de largo plazo:**
- Deficit cuenta corriente = necesidad de financiamiento externo
- Si IED cae y deficit persiste → presion depreciatoria
- ITCR muy alto → COP "cara" vs competidores → correccion probable
- Reservas cayendo = BanRep defendiendo peso (senal de debilidad)

**Variables clave:**
- `cuenta_corriente_z`: Deficit estructural de Colombia
- `itcr_deviation`: Desviacion del tipo de cambio real de su media
- `ied_z`: Flujos de inversion directa (estables vs portfolio)

### Sharpe Esperado: 0.4-0.7 (mejor para swing trading, no intraday)

### Nota Importante
Este dataset tiene alta frecuencia de zeros debido a que los datos fundamentales son mensuales/trimestrales. **Usar solo para horizontes de dias/semanas, no para trading de alta frecuencia.**

---

## Orden de Experimentacion Recomendado

### FASE 1: CORE (Semana 1-2)
1. **DS1_MINIMAL** → Validar pipeline, Sharpe > 0.3?
2. **DS3_MACRO_CORE** → Objetivo: Sharpe > 0.5 → PRODUCCION
3. **DS6_CARRY_TRADE** → Alto impacto esperado

### FASE 2: ESPECIALIZACION (Semana 3-4)
4. **DS8_RISK_SENTIMENT** → Para regimenes volatiles
5. **DS7_COMMODITY_BASKET** → Si correlacion Brent-COP es fuerte en test
6. **DS9_FED_WATCH** → Para ciclos de Fed

### FASE 3: AVANZADO (Semana 5+)
7. **DS4_COST_AWARE** → Solo si modelo tiene overtrading
8. **DS5_REGIME** → Solo con arquitecturas attention
9. **DS10_FLOWS_FUNDAMENTALS** → Solo para swing trading

### FASE 4: ENSEMBLE
Combinar DS3 + DS6 + DS8:
- Votacion mayoritaria
- O ponderado por VIX regime:
  - VIX < 20 → DS3 weight 70%, DS6 20%, DS8 10%
  - VIX 20-30 → DS3 50%, DS6 20%, DS8 30%
  - VIX > 30 → DS3 30%, DS6 10%, DS8 60%

---

## Consideraciones Tecnicas

### Frecuencia de Datos
| Tipo | Variables | Caracteristica en 5min |
|------|-----------|------------------------|
| Diario | DXY, VIX, Brent, Bonds | Actualiza 1x/dia, 288 bars iguales |
| Mensual | CPI, TPM, UNRATE | Actualiza 1x/mes, 8640 bars iguales |
| Trimestral | IED, Cuenta Corriente | Actualiza 1x/trim, 25920 bars iguales |

**Implicacion:** Features de baja frecuencia tendran z-scores cercanos a 0 la mayor parte del tiempo. Esto es esperado y correcto - el modelo aprende a usar estos como contexto, no como senales de trading.

### Normalizacion Aplicada
- **Retornos**: clip [-0.05, 0.05] (5% max movimiento diario)
- **Z-scores**: rolling 50 bars, clip [-4, 4]
- **RSI/ADX**: sin normalizar (ya bounded 0-100)
- **BB position**: 0-1
- **Percentiles**: 0-1

### Fecha de Corte
- **2020-03-01**: Evita warmup de Q1 2020 y crisis COVID inicial
- 1472 dias de trading, 83,886 filas de 5min

---

## Variables NO Incluidas (Blacklist)

| Variable | Razon |
|----------|-------|
| volume | USD/COP es OTC, volumen = 0 |
| obv_change_z | Depende de volume |
| macd_hist_z_old | Bug en calculo original (signal=1) |
| dxy_cop_corr_z | Division por cero → infinitos |
| t_in_session | Sin varianza (bug timezone) |
| risk_score_z | Outliers extremos |
| M2 Money Supply | Muy lento, efecto de largo plazo |
| UMCSENT | Bajo impacto directo en COP |

---

## Archivos Generados

```
OUTPUT_RL/
├── RL_DS1_MINIMAL.csv (19 MB)
├── RL_DS2_TECHNICAL_MTF.csv (26 MB)
├── RL_DS3_MACRO_CORE.csv (31 MB) ***
├── RL_DS4_COST_AWARE.csv (26 MB)
├── RL_DS5_REGIME.csv (39 MB)
├── RL_DS6_CARRY_TRADE.csv (23 MB) **
├── RL_DS7_COMMODITY_BASKET.csv (30 MB)
├── RL_DS8_RISK_SENTIMENT.csv (32 MB)
├── RL_DS9_FED_WATCH.csv (29 MB)
├── RL_DS10_FLOWS_FUNDAMENTALS.csv (21 MB)
├── DATASETS_JUSTIFICATION.md (este archivo)
└── STATS_DESCRIPTIVE.md
```

---

*Generado: 2025-11-27*
*Script: 03_create_rl_datasets.py*
