# Guia de Variables - Datasets USD/COP
**Para cualquier publico | Actualizado: Noviembre 2025**

---

## Que son estos datasets?

Son 10 conjuntos de datos disenados para entrenar modelos de inteligencia artificial que predicen movimientos del dolar frente al peso colombiano (USD/COP).

Cada dataset tiene un enfoque diferente:

| Dataset | Enfoque | Para que sirve |
|---------|---------|----------------|
| **DS1 - Minimo** | Lo basico | Probar que todo funciona |
| **DS2 - Tecnico** | Graficos y patrones | Trading tecnico clasico |
| **DS3 - Macro Core** | Economia global | **RECOMENDADO para produccion** |
| **DS4 - Anti-Sobreoperar** | Filtros de calidad | Evitar operar de mas |
| **DS5 - Regimenes** | Cambios de mercado | Detectar cuando cambia el mercado |
| **DS6 - Carry Trade** | Tasas de interes | Flujos por diferencial de tasas |
| **DS7 - Commodities** | Materias primas | Petroleo, cafe, oro |
| **DS8 - Riesgo** | Miedo del mercado | Risk-On vs Risk-Off |
| **DS9 - Fed Watch** | Banco Central USA | Expectativas de la Fed |
| **DS10 - Flujos** | Balanza de pagos | Inversion extranjera |

---

## Variables por Categoria

### PRECIO DEL DOLAR
*Datos basicos del USD/COP cada 5 minutos*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Precio de apertura | `open` | Todos |
| Precio maximo | `high` | Todos |
| Precio minimo | `low` | Todos |
| Precio de cierre | `close` | Todos |
| Fecha y hora | `datetime` | Todos |

---

### CAMBIOS DE PRECIO (RETORNOS)
*Cuanto subio o bajo el dolar en diferentes periodos*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Cambio en 5 minutos | `log_ret_5m` | Todos |
| Cambio en 15 minutos | `log_ret_15m` | DS2, DS5 |
| Cambio en 1 hora | `log_ret_1h` | Todos |
| Cambio en 4 horas | `log_ret_4h` | DS1, DS2, DS5, DS6, DS9 |
| Impulso (momentum) 6 periodos | `momentum_6` | DS4 |

---

### INDICADORES TECNICOS
*Herramientas clasicas de analisis de graficos*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| RSI (sobrecompra/sobreventa) | `rsi_9` | DS1, DS2, DS3, DS4, DS5, DS8 |
| RSI en marco de 15 minutos | `rsi_9_15m` | DS2 |
| RSI en marco de 1 hora | `rsi_9_1h` | DS2 |
| RSI en zona extrema? | `rsi_extreme` | DS4 |
| Volatilidad (ATR) | `atr_pct` | DS1, DS2, DS3, DS5 |
| Volatilidad 1 hora | `atr_pct_1h` | DS2 |
| Fuerza de tendencia (ADX) | `adx_14` | DS2, DS3, DS4, DS5 |
| Tendencia fuerte? | `adx_strong` | DS4 |
| Posicion en Bandas Bollinger | `bb_position` | DS1, DS2, DS3, DS5, DS8 |
| Precio vs Promedio Movil | `sma_ratio` | DS2 |
| Regimen de volatilidad | `vol_regime` | DS4 |

---

### INDICE DEL DOLAR (DXY)
*Fortaleza global del dolar vs otras monedas*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Nivel del dolar global | `dxy_z` | DS1, DS3, DS4, DS5, DS7, DS8 |
| Cambio del dolar en 1 dia | `dxy_change_1d` | DS3, DS5, DS8 |
| Tendencia dolar 5 dias | `dxy_mom_5d` | DS3, DS5 |
| Volatilidad dolar 5 dias | `dxy_vol_5d` | DS5 |

---

### MIEDO DEL MERCADO (VIX)
*Indice de volatilidad - cuando sube, hay panico*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Nivel de miedo absoluto | `vix_level` | DS1, DS8 |
| Nivel de miedo normalizado | `vix_z` | DS3, DS4, DS5, DS7, DS8 |
| Regimen: bajo/normal/alto/crisis | `vix_regime` | DS3, DS5, DS6, DS8 |
| Cambio del miedo en 1 dia | `vix_change_1d` | DS8 |
| Miedo vs ultimos 20 dias | `vix_percentile_20d` | DS8 |

---

### RIESGO COLOMBIA (EMBI)
*Cuanto extra piden por prestarle a Colombia*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Riesgo pais nivel | `embi_z` | DS3, DS5, DS6, DS8, DS10 |
| Cambio riesgo 1 dia | `embi_change_1d` | DS8 |
| Cambio riesgo 5 dias | `embi_change_5d` | DS5, DS8 |
| Riesgo vs ultimos 20 dias | `embi_percentile_20d` | DS8 |

---

### PETROLEO
*Principal exportacion de Colombia*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Precio Brent nivel | `brent_z` | DS7 |
| Cambio Brent 1 dia | `brent_change_1d` | DS3, DS5, DS7 |
| Tendencia Brent 5 dias | `brent_mom_5d` | DS7 |
| Volatilidad Brent 5 dias | `brent_vol_5d` | DS3, DS5 |
| Precio WTI nivel | `wti_z` | DS7 |
| Diferencia Brent vs WTI | `brent_wti_spread_z` | DS7 |

---

### CAFE
*Segunda exportacion de Colombia*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Precio cafe nivel | `coffee_z` | DS7 |
| Cambio cafe 1 dia | `coffee_change_1d` | DS7 |

---

### ORO
*Refugio en tiempos de crisis*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Precio oro nivel | `gold_z` | DS7 |
| Cambio oro 1 dia | `gold_change_1d` | DS7, DS8 |

---

### BOLSA COLOMBIA (COLCAP)
*Indice de la bolsa de valores colombiana*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Nivel bolsa Colombia | `colcap_z` | DS7 |
| Cambio bolsa 1 dia | `colcap_change_1d` | DS7, DS8 |

---

### TASAS DE INTERES
*Diferencial entre Colombia y USA*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Diferencial tasas COL vs USA | `rate_spread` | DS3, DS5 |
| Pendiente curva USA | `curve_slope_z` | DS5 |
| Spread bonos 10 anos COL-USA | `spread_normalized` | DS6 |
| Cambio spread 1 dia | `spread_change_1d` | DS6 |
| Pendiente curva Colombia | `col_curve_normalized` | DS6 |
| Pendiente curva USA | `usa_curve_normalized` | DS6, DS9 |
| Curva USA invertida? | `usa_curve_inverted` | DS6, DS9 |
| Diferencial TPM vs Fed Funds | `policy_spread_normalized` | DS6 |
| Diferencial IBR vs TPM | `ibr_tpm_normalized` | DS6 |
| Carry trade favorable? | `carry_favorable` | DS6 |
| BanRep subiendo tasas? | `col_hiking` | DS6 |
| Fed subiendo tasas? | `fed_hiking` | DS6 |

---

### POLITICA DE LA FED
*Indicadores de que hara el banco central de USA*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Fed agresiva contra inflacion? | `fed_hawkish` | DS9 |
| Fed relajada/estimulando? | `fed_dovish` | DS9 |
| Inflacion alta (>0.3% mensual)? | `inflation_hot` | DS9 |
| Inflacion en crisis (>0.5%)? | `inflation_crisis` | DS9 |
| Inflacion acelerando? | `cpi_accelerating` | DS9 |
| Mercado laboral apretado? | `labor_tight` | DS9 |
| Desempleo alto? | `labor_weak` | DS9 |
| Desempleo subiendo? | `unemployment_rising` | DS9 |
| Tasas restrictivas (>4%)? | `rates_restrictive` | DS9 |
| Tasas bajas (<2%)? | `rates_accommodative` | DS9 |

---

### OTRAS MONEDAS LATAM
*Peso mexicano y peso chileno - se mueven similar al colombiano*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Cambio dolar/peso mexicano 1h | `usdmxn_ret_1h` | DS3, DS5, DS8 |
| Nivel dolar/peso mexicano | `usdmxn_z` | DS8 |
| Cambio dolar/peso chileno 1h | `usdclp_ret_1h` | DS5, DS8 |

---

### FLUJOS DE CAPITAL
*Dinero entrando y saliendo de Colombia*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Inversion Extranjera nivel | `ied_normalized` | DS10 |
| Inversion creciendo? | `ied_growing` | DS10 |
| Cuenta Corriente nivel | `ca_normalized` | DS10 |
| Deficit reduciendose? | `ca_improving` | DS10 |
| Exportaciones creciendo? | `exports_growing` | DS10 |
| Balanza comercial mejorando? | `trade_improving` | DS10 |
| Peso sobre/subvaluado | `itcr_deviation` | DS10 |
| Cambio tipo de cambio real 1 mes | `itcr_change_1m` | DS10 |
| Reservas BanRep cayendo? | `reserves_falling` | DS10 |

---

### HORA Y DIA
*Patrones por horario de trading*

| Que es | Nombre Tecnico | En que datasets |
|--------|----------------|-----------------|
| Hora del dia (ciclico) | `hour_sin`, `hour_cos` | Todos |
| Dia de la semana | `dow_sin` | DS5 |

---

## Resumen Visual

```
DS1 MINIMO        = Precio + RSI + VIX + Hora
DS2 TECNICO       = Precio + RSI + ADX + ATR + Bollinger
DS3 MACRO CORE    = Tecnico + Dolar + VIX + Petroleo + Tasas   <-- RECOMENDADO
DS4 ANTI-SOBREOPERAR = Tecnico + Filtros de calidad
DS5 REGIMENES     = Todo lo anterior + Mas contexto
DS6 CARRY TRADE   = Tasas Colombia vs USA
DS7 COMMODITIES   = Petroleo + Cafe + Oro + COLCAP
DS8 RIESGO        = VIX + EMBI + Otras monedas LATAM
DS9 FED WATCH     = Indicadores de politica Fed
DS10 FLUJOS       = Inversion extranjera + Balanza pagos
```

---

## Para Empezar

1. **Usa DS3 (Macro Core)** - Es el mas balanceado y probado
2. Si te interesa el carry trade, agrega **DS6**
3. En mercados volatiles, agrega **DS8**
4. Combina varios para mejor prediccion

---

## Datos Tecnicos

- **Periodo:** Marzo 2020 - Octubre 2025
- **Frecuencia:** Cada 5 minutos
- **Horario:** 8:00 AM - 12:55 PM hora Colombia (sesion principal)
- **Total filas:** 83,886 observaciones por dataset
