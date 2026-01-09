# Diccionario de Variables - Datasets RL USD/COP
**Generado:** 2025-11-27

Este documento describe todas las variables disponibles en los 10 datasets de Reinforcement Learning para el par USD/COP, organizadas por categoria y mostrando en cuales datasets aparecen.

---

## Resumen de Datasets

| ID | Nombre | Features | Uso Principal |
|----|--------|----------|---------------|
| DS1 | MINIMAL | 10 | Baseline - Validar pipeline |
| DS2 | TECHNICAL_MTF | 14 | Analisis tecnico multi-timeframe |
| DS3 | MACRO_CORE | 19 | **PRODUCCION** - Balance optimo |
| DS4 | COST_AWARE | 16 | Anti-overtrading |
| DS5 | REGIME | 25 | Deteccion de cambios de regimen |
| DS6 | CARRY_TRADE | 18 | Flujos de carry trade |
| DS7 | COMMODITY_BASKET | 17 | Commodities colombianos |
| DS8 | RISK_SENTIMENT | 21 | Risk-On/Risk-Off |
| DS9 | FED_WATCH | 17 | Expectativas Fed |
| DS10 | FLOWS_FUNDAMENTALS | 14 | Flujos de capital / Swing |

---

## Variables Base (en todos los datasets)

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `datetime` | Timestamp UTC | - | Todos |
| `open` | Precio apertura 5min | ~3000-5000 | Todos |
| `high` | Precio maximo 5min | ~3000-5000 | Todos |
| `low` | Precio minimo 5min | ~3000-5000 | Todos |
| `close` | Precio cierre 5min | ~3000-5000 | Todos |

---

## 1. RETORNOS LOGARITMICOS

Retornos log del precio de cierre en diferentes ventanas temporales.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `log_ret_5m` | Retorno 5 minutos | [-0.05, 0.05] | DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10 |
| `log_ret_15m` | Retorno 15 minutos | [-0.05, 0.05] | DS2, DS5 |
| `log_ret_1h` | Retorno 1 hora | [-0.05, 0.05] | DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10 |
| `log_ret_4h` | Retorno 4 horas | [-0.05, 0.05] | DS1, DS2, DS5, DS6, DS9 |
| `ret_lag_1` | Retorno 5m rezagado 1 periodo | [-0.05, 0.05] | DS4 |
| `ret_lag_3` | Retorno 5m rezagado 3 periodos | [-0.05, 0.05] | DS4 |
| `ret_atr_adj` | Retorno ajustado por ATR | [-3, 3] | DS4 |
| `momentum_6` | Momentum 6 periodos | [-0.05, 0.05] | DS4 |

---

## 2. INDICADORES TECNICOS

Indicadores de analisis tecnico clasico.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `rsi_9` | RSI periodo 9 | [0, 100] | DS1, DS2, DS3, DS4, DS5, DS8 |
| `rsi_9_15m` | RSI 9 en timeframe 15min | [0, 100] | DS2 |
| `rsi_9_1h` | RSI 9 en timeframe 1h | [0, 100] | DS2 |
| `rsi_extreme` | RSI en zona extrema (>70 o <30) | {0, 1} | DS4 |
| `atr_pct` | ATR como % del precio | [0, 1] | DS1, DS2, DS3, DS5 |
| `atr_pct_1h` | ATR 1h como % del precio | [0, 1] | DS2 |
| `atr_percentile` | Percentil de ATR (20 dias) | [0, 1] | DS4 |
| `adx_14` | ADX periodo 14 (fuerza tendencia) | [0, 100] | DS2, DS3, DS4, DS5 |
| `adx_strong` | ADX > 25 (tendencia fuerte) | {0, 1} | DS4 |
| `bb_position` | Posicion en Bollinger Bands | [0, 1] | DS1, DS2, DS3, DS5, DS8 |
| `sma_ratio` | Ratio precio vs SMA | [-0.05, 0.05] | DS2 |
| `vol_regime` | Regimen de volatilidad | {0, 1, 2} | DS4 |

---

## 3. DOLAR INDEX (DXY)

Indice del dolar estadounidense - principal driver global.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `dxy_z` | Z-score del DXY | [-4, 4] | DS1, DS3, DS4, DS5, DS7, DS8 |
| `dxy_change_1d` | Cambio % DXY 1 dia | [-0.05, 0.05] | DS3, DS5, DS8 |
| `dxy_mom_5d` | Momentum DXY 5 dias | [-0.07, 0.07] | DS3, DS5 |
| `dxy_vol_5d` | Volatilidad DXY 5 dias | [0, 0.02] | DS5 |

---

## 4. VOLATILIDAD GLOBAL (VIX)

Indice de volatilidad - indicador de miedo del mercado.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `vix_level` | Nivel absoluto del VIX | [10, 85] | DS1, DS8 |
| `vix_z` | Z-score del VIX | [-4, 4] | DS3, DS4, DS5, DS7, DS8 |
| `vix_regime` | Regimen VIX: 0=bajo, 1=normal, 2=alto, 3=crisis | {0, 1, 2, 3} | DS3, DS5, DS6, DS8 |
| `vix_change_1d` | Cambio % VIX 1 dia | [-0.3, 0.3] | DS8 |
| `vix_percentile_20d` | Percentil VIX 20 dias | [0, 1] | DS8 |

---

## 5. RIESGO PAIS (EMBI)

Spread de bonos emergentes - riesgo percibido de Colombia.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `embi_z` | Z-score del EMBI Colombia | [-4, 4] | DS3, DS5, DS6, DS8, DS10 |
| `embi_change_1d` | Cambio % EMBI 1 dia | [-0.1, 0.1] | DS8 |
| `embi_change_5d` | Cambio % EMBI 5 dias | [-0.1, 0.1] | DS5, DS8 |
| `embi_percentile_20d` | Percentil EMBI 20 dias | [0, 1] | DS8 |

---

## 6. PETROLEO Y COMMODITIES

Principales exportaciones de Colombia.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `brent_z` | Z-score Brent | [-4, 4] | DS7 |
| `brent_change_1d` | Cambio % Brent 1 dia | [-0.1, 0.1] | DS3, DS5, DS7 |
| `brent_mom_5d` | Momentum Brent 5 dias | [-0.15, 0.15] | DS7 |
| `brent_vol_5d` | Volatilidad Brent 5 dias | [0, 0.02] | DS3, DS5 |
| `wti_z` | Z-score WTI | [-4, 4] | DS7 |
| `brent_wti_spread_z` | Z-score spread Brent-WTI | [-4, 4] | DS7 |
| `coffee_z` | Z-score precio cafe | [-4, 4] | DS7 |
| `coffee_change_1d` | Cambio % cafe 1 dia | [-0.1, 0.1] | DS7 |
| `gold_z` | Z-score oro | [-4, 4] | DS7 |
| `gold_change_1d` | Cambio % oro 1 dia | [-0.05, 0.05] | DS7, DS8 |
| `colcap_z` | Z-score indice COLCAP | [-4, 4] | DS7 |
| `colcap_change_1d` | Cambio % COLCAP 1 dia | [-0.05, 0.05] | DS7, DS8 |

---

## 7. TASAS DE INTERES Y CARRY TRADE

Diferenciales de tasas entre Colombia y USA.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `rate_spread` | Spread tasas COL-USA normalizado | [-4, 4] | DS3, DS5 |
| `curve_slope_z` | Z-score pendiente curva USA | [-4, 4] | DS5 |
| `spread_normalized` | Spread bonos 10Y COL-USA / 10 | [-0.5, 0.5] | DS6 |
| `spread_change_1d` | Cambio spread 1 dia | [-0.02, 0.02] | DS6 |
| `spread_z_20d` | Z-score spread (ventana 20 dias) | [-4, 4] | DS6 |
| `col_curve_normalized` | Pendiente curva COL / 2 | [-1, 1] | DS6 |
| `usa_curve_normalized` | Pendiente curva USA / 2 | [-1, 1] | DS6, DS9 |
| `usa_curve_inverted` | Curva USA invertida | {0, 1} | DS6, DS9 |
| `policy_spread_normalized` | Spread TPM-FedFunds / 10 | [-0.5, 1] | DS6 |
| `ibr_tpm_normalized` | Spread IBR-TPM / 2 | [-1, 1] | DS6 |
| `carry_favorable` | Spread politica > 2pp | {0, 1} | DS6 |
| `col_hiking` | BanRep subiendo tasas | {0, 1} | DS6 |
| `fed_hiking` | Fed subiendo tasas | {0, 1} | DS6 |

---

## 8. INDICADORES FED (Regimen Monetario USA)

Indicadores binarios del regimen de politica monetaria de la Fed.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `fed_hawkish` | Fed agresiva (inflacion + empleo apretado) | {0, 1} | DS9 |
| `fed_dovish` | Fed acomodaticia | {0, 1} | DS9 |
| `inflation_hot` | Inflacion MoM > 0.3% | {0, 1} | DS9 |
| `inflation_crisis` | Inflacion MoM > 0.5% | {0, 1} | DS9 |
| `cpi_accelerating` | Inflacion acelerando | {0, 1} | DS9 |
| `labor_tight` | Desempleo < 4% | {0, 1} | DS9 |
| `labor_weak` | Desempleo > 5% | {0, 1} | DS9 |
| `unemployment_rising` | Desempleo subiendo | {0, 1} | DS9 |
| `rates_restrictive` | UST 2Y > 4% | {0, 1} | DS9 |
| `rates_accommodative` | UST 2Y < 2% | {0, 1} | DS9 |

---

## 9. CROSS-PAIRS (Pares Correlacionados)

Movimientos de otras monedas latinoamericanas.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `usdmxn_ret_1h` | Retorno USD/MXN 1 hora | [-0.05, 0.05] | DS3, DS5, DS8 |
| `usdmxn_z` | Z-score USD/MXN | [-4, 4] | DS8 |
| `usdclp_ret_1h` | Retorno USD/CLP 1 hora | [-0.05, 0.05] | DS5, DS8 |

---

## 10. FLUJOS DE CAPITAL Y FUNDAMENTALES

Indicadores de balanza de pagos de Colombia.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `ied_normalized` | Inversion Extranjera Directa / 10B | [0, 1] | DS10 |
| `ied_growing` | IED creciendo vs trimestre anterior | {0, 1} | DS10 |
| `ca_normalized` | Cuenta Corriente / 10B | [-1, 0] | DS10 |
| `ca_improving` | Deficit reduciendose | {0, 1} | DS10 |
| `exports_growing` | Exportaciones creciendo | {0, 1} | DS10 |
| `trade_improving` | Balanza comercial mejorando | {0, 1} | DS10 |
| `itcr_deviation` | Desviacion ITCR de media 1Y | [-0.3, 0.3] | DS10 |
| `itcr_change_1m` | Cambio % ITCR 1 mes | [-0.1, 0.1] | DS10 |
| `reserves_falling` | Reservas BanRep cayendo | {0, 1} | DS10 |

---

## 11. VARIABLES TEMPORALES

Encoding ciclico de hora y dia.

| Variable | Descripcion | Rango | Datasets |
|----------|-------------|-------|----------|
| `hour_sin` | Seno de hora UTC | [0, 0.87] | Todos |
| `hour_cos` | Coseno de hora UTC | [-1, -0.5] | Todos |
| `dow_sin` | Seno de dia de semana | [-0.5, 1] | DS5 |

---

## Matriz de Variables por Dataset

| Variable | DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 | DS9 | DS10 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|
| log_ret_5m | X | X | X | X | X | X | X | X | X | X |
| log_ret_15m | | X | | | X | | | | | |
| log_ret_1h | X | X | X | X | X | X | X | X | X | X |
| log_ret_4h | X | X | | | X | X | | | X | |
| rsi_9 | X | X | X | X | X | | | X | | |
| atr_pct | X | X | X | | X | | | | | |
| adx_14 | | X | X | X | X | | | | | |
| bb_position | X | X | X | | X | | | X | | |
| dxy_z | X | | X | X | X | | X | | | |
| dxy_change_1d | | | X | | X | | | X | | |
| vix_level | X | | | | | | | X | | |
| vix_z | | | X | X | X | | X | X | | |
| vix_regime | | | X | | X | X | | X | | |
| embi_z | | | X | | X | X | | X | | X |
| brent_change_1d | | | X | | X | | X | | | |
| rate_spread | | | X | | X | | | | | |
| usdmxn_ret_1h | | | X | | X | | | X | | |
| spread_normalized | | | | | | X | | | | |
| policy_spread_normalized | | | | | | X | | | | |
| usa_curve_inverted | | | | | | X | | | X | |
| brent_z | | | | | | | X | | | |
| coffee_z | | | | | | | X | | | |
| gold_z | | | | | | | X | | | |
| fed_hawkish | | | | | | | | | X | |
| inflation_hot | | | | | | | | | X | |
| labor_tight | | | | | | | | | X | |
| ied_normalized | | | | | | | | | | X |
| ca_normalized | | | | | | | | | | X |
| itcr_deviation | | | | | | | | | | X |
| hour_sin | X | X | X | X | X | X | X | X | X | X |
| hour_cos | X | X | X | X | X | X | X | X | X | X |

---

## Notas Importantes

### Normalizacion Aplicada
- **Retornos:** Clipping a [-0.05, 0.05] (limita movimientos extremos)
- **Z-scores:** Ventana rolling 50 bars, clipping a [-4, 4]
- **RSI/ADX:** Sin normalizar (ya estan en rango 0-100)
- **Bollinger Position:** Normalizado a [0, 1]
- **Percentiles:** Normalizados a [0, 1]

### Variables Binarias
Las variables con rango `{0, 1}` son indicadores binarios que capturan regimenes o estados del mercado. Son especialmente utiles para:
- Deteccion de cambios de regimen
- Filtrado de senales
- Estrategias condicionales

### Periodo de Datos
- **Inicio:** 2020-03-02 (evita warmup Q1 2020)
- **Fin:** 2025-10-29
- **Filas:** 83,886 observaciones (5 minutos)
- **Horario:** L-V, 13:00-17:55 UTC (8:00-12:55 hora Colombia)

---

## Recomendacion de Uso

1. **Comenzar con DS1 o DS3** para validar el pipeline
2. **DS3 MACRO_CORE** es el dataset recomendado para produccion
3. **DS6 CARRY_TRADE** para estrategias de diferencial de tasas
4. **DS8 RISK_SENTIMENT** para regimenes de alta volatilidad
5. **Ensemble DS3 + DS6 + DS8** para maxima robustez
