# Explicacion Completa de los 10 Datasets USD/COP
**Guia detallada para cualquier persona | Noviembre 2025**

---

# DS1 - MINIMO (10 variables)

## Para que sirve?
Es el dataset mas simple. Sirve para probar que el sistema funciona antes de usar datos mas complejos. Como un "Hola Mundo" del trading algoritmico.

## Variables incluidas:

### Retornos (cambios de precio)

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cuanto subio o bajo el dolar en los ultimos 5 minutos. Si es positivo, el dolar subio (peso se devaluo). Si es negativo, el dolar bajo. |
| **log_ret_1h** | Lo mismo pero mirando la ultima hora completa. Da una vision menos ruidosa. |
| **log_ret_4h** | Cambio en las ultimas 4 horas. Muestra la tendencia de medio dia. |

### Indicadores tecnicos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | El "termometro" de sobrecompra/sobreventa. Va de 0 a 100. Arriba de 70 = "muy caro, puede bajar". Abajo de 30 = "muy barato, puede subir". |
| **atr_pct** | Que tan volatil esta el mercado ahora mismo. Numero alto = mercado movido. Numero bajo = mercado tranquilo. |
| **bb_position** | Donde esta el precio dentro de las Bandas de Bollinger. 0 = en el piso (barato). 1 = en el techo (caro). 0.5 = en el medio. |

### Contexto global

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **dxy_z** | Que tan fuerte esta el dolar contra TODAS las monedas del mundo (euro, yen, libra, etc). Positivo = dolar fuerte globalmente. Negativo = dolar debil. |
| **vix_level** | El "indice del miedo" de Wall Street. Normal = 15-20. Arriba de 30 = nerviosismo. Arriba de 40 = panico. |

### Hora del dia

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Codificacion matematica de la hora. Ayuda al modelo a aprender que a ciertas horas el mercado se comporta diferente. |
| **hour_cos** | Complemento del anterior. Juntos capturan el patron ciclico del dia. |

---

# DS2 - TECNICO MULTI-TIMEFRAME (14 variables)

## Para que sirve?
Para estrategias de analisis tecnico puro. Mira el mismo indicador en diferentes marcos de tiempo (5 min, 15 min, 1 hora) para confirmar senales.

## Variables incluidas:

### Retornos en multiples marcos de tiempo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio en 5 minutos - movimientos rapidos, mucho ruido. |
| **log_ret_15m** | Cambio en 15 minutos - menos ruido, tendencia corta. |
| **log_ret_1h** | Cambio en 1 hora - tendencia mas clara. |
| **log_ret_4h** | Cambio en 4 horas - tendencia de medio dia. |

### RSI en multiples marcos de tiempo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | RSI calculado en velas de 5 minutos. Muy sensible, cambia rapido. |
| **rsi_9_15m** | RSI en velas de 15 minutos. Menos sensible, mas confiable. |
| **rsi_9_1h** | RSI en velas de 1 hora. Senales mas fuertes y duraderas. |

*Cuando los tres RSI coinciden (ej: todos arriba de 70), la senal es mas confiable.*

### Volatilidad

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **atr_pct** | Volatilidad en 5 minutos. Que tanto se mueve el precio ahora. |
| **atr_pct_1h** | Volatilidad en 1 hora. Vision mas amplia de que tan movido esta el mercado. |

### Tendencia

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **adx_14** | Fuerza de la tendencia (no direccion). 0-20 = sin tendencia, mercado lateral. 20-40 = tendencia moderada. 40+ = tendencia fuerte. |
| **sma_ratio** | Esta el precio arriba o abajo de su promedio? Positivo = arriba del promedio. Negativo = abajo. |
| **bb_position** | Posicion en Bandas Bollinger (0=piso, 1=techo). |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Encoding de hora (parte 1). |
| **hour_cos** | Encoding de hora (parte 2). |

---

# DS3 - MACRO CORE (19 variables) ⭐ RECOMENDADO

## Para que sirve?
El dataset estrella. Combina analisis tecnico con factores macroeconomicos. Es el mejor balance entre complejidad y utilidad. **Usar este para produccion.**

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio de precio en 5 minutos. |
| **log_ret_1h** | Cambio de precio en 1 hora. |
| **log_ret_4h** | Cambio de precio en 4 horas. |

### Tecnicos basicos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | Sobrecompra/sobreventa (0-100). |
| **atr_pct** | Volatilidad actual. |
| **adx_14** | Fuerza de la tendencia. |
| **bb_position** | Posicion en Bandas Bollinger. |

### Dolar global (DXY)

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **dxy_z** | Fuerza del dolar vs mundo. Si sube, generalmente USD/COP sube. |
| **dxy_change_1d** | Cuanto cambio el dolar global hoy. Movimiento positivo grande = presion alcista en USD/COP. |
| **dxy_mom_5d** | Tendencia del dolar en la ultima semana. Positivo = dolar fortaleciendose. |

### Riesgo global

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **vix_z** | Nivel de miedo normalizado. Cuando sube, inversionistas huyen de mercados emergentes como Colombia → peso se devalua. |
| **vix_regime** | Categoria de miedo: 0=tranquilo, 1=normal, 2=nervioso, 3=panico. En regimen 3, el peso casi siempre cae. |
| **embi_z** | Riesgo especifico de Colombia. Cuanto extra piden por prestarle a Colombia vs USA. Alto = desconfianza = peso debil. |

### Petroleo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **brent_change_1d** | Cambio del petroleo hoy. Colombia exporta petroleo, si sube el precio entran dolares → peso se fortalece. |
| **brent_vol_5d** | Volatilidad del petroleo. Alta volatilidad = incertidumbre. |

### Tasas de interes

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rate_spread** | Diferencia entre tasas de Colombia y USA. Si Colombia paga mas, atrae dolares → peso fuerte. |

### Peso mexicano

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **usdmxn_ret_1h** | Como se movio el peso mexicano en la ultima hora. Mexico y Colombia se mueven parecido. Si el mexicano cae, el colombiano probablemente tambien. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Encoding de hora (parte 1). |
| **hour_cos** | Encoding de hora (parte 2). |

---

# DS4 - ANTI-SOBREOPERAR (16 variables)

## Para que sirve?
Evitar operar cuando no hay buenas oportunidades. Incluye filtros de calidad para reducir el numero de operaciones y mejorar el ratio ganancia/perdida.

## Variables incluidas:

### Retornos y rezagos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio actual. |
| **log_ret_1h** | Cambio en 1 hora. |
| **ret_lag_1** | Cual fue el cambio hace 5 minutos? Sirve para ver si el movimiento continua o se revierte. |
| **ret_lag_3** | Cual fue el cambio hace 15 minutos? Mas contexto historico. |

### Momentum ajustado

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **ret_atr_adj** | Cambio de precio AJUSTADO por volatilidad. Un movimiento de 0.1% significa diferente cosas en mercado tranquilo vs volatil. Este lo normaliza. |
| **momentum_6** | Suma de los ultimos 6 cambios. Positivo = tendencia alcista reciente. Negativo = tendencia bajista. |

### Filtros de senal

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | RSI para detectar extremos. |
| **rsi_extreme** | Esta el RSI en zona extrema? 1 = Si (>70 o <30). 0 = No. Solo operar cuando es 1 puede dar mejores senales. |
| **adx_14** | Fuerza de tendencia. |
| **adx_strong** | Hay tendencia fuerte? 1 = Si (ADX>25). 0 = No. Evitar operar cuando es 0. |

### Volatilidad

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **atr_percentile** | La volatilidad actual comparada con los ultimos 20 dias. 0.9 = mas volatil que el 90% de los dias recientes. |
| **vol_regime** | Regimen de volatilidad: 0=baja, 1=normal, 2=alta. Cada regimen requiere estrategia diferente. |

### Contexto macro

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **dxy_z** | Fuerza del dolar global. |
| **vix_z** | Nivel de miedo. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Encoding de hora (parte 1). |
| **hour_cos** | Encoding de hora (parte 2). |

---

# DS5 - REGIMENES (25 variables)

## Para que sirve?
Detectar cuando el mercado cambia de "modo". Por ejemplo: de tranquilo a volatil, de tendencia a lateral, de risk-on a risk-off. Ideal para modelos avanzados (redes neuronales con atencion).

## Variables incluidas:

### Retornos completos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_15m** | Cambio 15 min. |
| **log_ret_1h** | Cambio 1 hora. |
| **log_ret_4h** | Cambio 4 horas. |

### Tecnicos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | RSI. |
| **atr_pct** | Volatilidad. |
| **adx_14** | Fuerza tendencia. |
| **bb_position** | Bollinger. |

### Dolar global completo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **dxy_z** | Nivel del dolar. |
| **dxy_change_1d** | Cambio diario. |
| **dxy_mom_5d** | Momentum semanal. |
| **dxy_vol_5d** | Volatilidad del dolar. Si esta muy volatil, hay incertidumbre global. |

### Riesgo completo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **vix_z** | Miedo nivel. |
| **vix_regime** | Categoria miedo. |
| **embi_z** | Riesgo Colombia. |
| **embi_change_5d** | Esta aumentando o disminuyendo el riesgo pais? Positivo = empeorando. |

### Petroleo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **brent_change_1d** | Cambio petroleo. |
| **brent_vol_5d** | Volatilidad petroleo. |

### Tasas

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rate_spread** | Diferencial Colombia-USA. |
| **curve_slope_z** | Pendiente de la curva de tasas USA. Negativa = posible recesion proxima. |

### Monedas latinas

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **usdmxn_ret_1h** | Movimiento peso mexicano. |
| **usdclp_ret_1h** | Movimiento peso chileno. Si ambos caen, es tema regional, no solo Colombia. |

### Tiempo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |
| **dow_sin** | Dia de la semana. Lunes se comporta diferente a viernes. |

---

# DS6 - CARRY TRADE (18 variables)

## Para que sirve?
Capturar flujos de "carry trade" - cuando inversionistas piden prestado donde las tasas son bajas (USA) para invertir donde son altas (Colombia). Estos flujos mueven el peso.

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_1h** | Cambio 1 hora. |
| **log_ret_4h** | Cambio 4 horas. |

### Spread de bonos soberanos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **spread_normalized** | Diferencia entre lo que paga un bono colombiano a 10 anos vs uno americano. Ej: Colombia 10%, USA 4% = spread de 6%. Spread alto atrae dolares a Colombia. |
| **spread_change_1d** | El spread subio o bajo hoy? Subiendo = mas atractivo invertir en Colombia. |
| **spread_z_20d** | El spread actual vs los ultimos 20 dias. Muy alto = inusualmente atractivo. |

### Curvas de rendimiento

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **col_curve_normalized** | Pendiente de tasas Colombia. Positiva = economia "normal". Negativa = expectativa de problemas. |
| **usa_curve_normalized** | Pendiente de tasas USA. |
| **usa_curve_inverted** | Esta la curva USA invertida? 1=Si, 0=No. Curva invertida predice recesion en USA → dolar puede debilitarse. |

### Politica monetaria

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **policy_spread_normalized** | Diferencia entre tasa de BanRep (Colombia) y Fed (USA). Ej: BanRep 10%, Fed 5% = spread 5%. Mayor spread = mas carry trade = peso fuerte. |
| **ibr_tpm_normalized** | Diferencia entre tasa interbancaria (IBR) y tasa politica (TPM). Si IBR >> TPM = estres en el sistema bancario. |
| **carry_favorable** | Es buen momento para carry trade? 1=Si (spread > 2%). 0=No. |

### Direccion de tasas

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **col_hiking** | BanRep esta subiendo tasas? 1=Si. Subiendo tasas = peso tiende a fortalecerse. |
| **fed_hiking** | Fed esta subiendo tasas? 1=Si. Fed subiendo = dolar global fuerte = presion en peso. |

### Riesgo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **embi_z** | Riesgo Colombia. Si sube mucho, inversionistas huyen aunque el spread sea atractivo. |
| **vix_regime** | Nivel de panico. En panico, el carry trade se "deshace" = peso cae fuerte. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |

---

# DS7 - COMMODITIES (17 variables)

## Para que sirve?
Colombia depende de exportar petroleo, cafe, y algo de oro. Este dataset captura como los precios de estas materias primas afectan el peso.

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_1h** | Cambio 1 hora. |

### Petroleo Brent

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **brent_z** | Precio del Brent (petroleo europeo, referencia para Colombia) comparado con su historia reciente. Alto = petroleo caro = bueno para Colombia. |
| **brent_change_1d** | Cambio hoy. Petroleo +3% hoy = entran dolares a Colombia = peso fuerte. |
| **brent_mom_5d** | Tendencia de la semana. Petroleo subiendo toda la semana = muy positivo para peso. |

### Petroleo WTI

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **wti_z** | Precio del WTI (petroleo americano). Generalmente se mueve similar al Brent. |
| **brent_wti_spread_z** | Diferencia Brent - WTI. Cuando Brent >> WTI, generalmente mejor para Colombia (exportamos referencia Brent). |

### Cafe

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **coffee_z** | Precio del cafe. Colombia es el 3er exportador mundial. Cafe caro = dolares entrando. |
| **coffee_change_1d** | Cambio hoy. |

### Oro

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **gold_z** | Precio del oro. Colombia exporta algo de oro, pero mas importante: oro alto = miedo global = pesos emergentes debiles. Relacion mixta. |
| **gold_change_1d** | Cambio hoy. |

### Bolsa Colombia

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **colcap_z** | Nivel del indice COLCAP (bolsa colombiana). Bolsa subiendo = optimismo = peso tiende a fortalecerse. |
| **colcap_change_1d** | Cambio hoy de la bolsa. |

### Contexto

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **vix_z** | Miedo global. |
| **dxy_z** | Dolar global. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |

---

# DS8 - RIESGO / SENTIMIENTO (21 variables)

## Para que sirve?
Detectar si el mercado esta en modo "Risk-On" (inversionistas buscan retorno, van a emergentes) o "Risk-Off" (huyen a activos seguros como dolar). El peso colombiano sufre mucho en Risk-Off.

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_1h** | Cambio 1 hora. |

### VIX (miedo) - analisis completo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **vix_level** | Nivel absoluto del VIX. |
| **vix_z** | VIX normalizado. |
| **vix_regime** | Categoria: tranquilo/normal/nervioso/panico. |
| **vix_change_1d** | Cambio hoy. VIX subiendo rapido = ALERTA. |
| **vix_percentile_20d** | VIX actual vs ultimos 20 dias. 0.95 = miedo inusualmente alto. |

### EMBI (riesgo Colombia) - analisis completo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **embi_z** | Riesgo Colombia nivel. |
| **embi_change_1d** | Cambio hoy. |
| **embi_change_5d** | Cambio semanal. EMBI subiendo varios dias = situacion deteriorandose. |
| **embi_percentile_20d** | Riesgo vs ultimos 20 dias. |

### Pares latinos (indicadores lideres)

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **usdmxn_ret_1h** | Peso mexicano cambio 1h. Mexico es mas liquido, a veces se mueve primero. |
| **usdmxn_z** | Nivel peso mexicano. |
| **usdclp_ret_1h** | Peso chileno cambio 1h. Chile es vecino, economia similar. |

### Refugios

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **gold_change_1d** | Oro subiendo = miedo = malo para peso. |
| **dxy_change_1d** | Dolar subiendo = flight to safety = malo para peso. |

### Mercado local

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **colcap_change_1d** | Bolsa Colombia. Bolsa cayendo + peso cayendo = risk-off fuerte. |

### Tecnicos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rsi_9** | RSI del USD/COP. |
| **bb_position** | Bollinger del USD/COP. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |

---

# DS9 - FED WATCH (17 variables)

## Para que sirve?
Anticipar movimientos de la Reserva Federal (banco central USA). La Fed mueve TODO. Si sube tasas, dolar sube, emergentes sufren. Este dataset captura las senales de hacia donde va la Fed.

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_1h** | Cambio 1 hora. |
| **log_ret_4h** | Cambio 4 horas. |

### Regimen de la Fed

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **fed_hawkish** | La Fed esta en modo agresivo? 1=Si. Cuando hay inflacion alta + empleo fuerte, la Fed sube tasas agresivamente = dolar fuerte = peso debil. |
| **fed_dovish** | La Fed esta en modo relajado? 1=Si. Cuando no hay inflacion y hay desempleo, la Fed baja tasas = dolar debil = peso puede fortalecerse. |

### Inflacion USA

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **inflation_hot** | Inflacion caliente? 1=Si (>3.6% anualizada). La Fed TIENE que subir tasas = presion en peso. |
| **inflation_crisis** | Inflacion en crisis? 1=Si (>6% anualizada). Fed en modo emergencia. |
| **cpi_accelerating** | Inflacion acelerando? 1=Si. Peor que inflacion alta es inflacion que SUBE. |

### Empleo USA

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **labor_tight** | Mercado laboral apretado? 1=Si (desempleo <4%). Empresas compiten por trabajadores = presion inflacionaria = Fed sube. |
| **labor_weak** | Desempleo alto? 1=Si (>5%). Fed puede bajar tasas. |
| **unemployment_rising** | Desempleo subiendo? 1=Si. Senal de recesion = Fed baja tasas. |

### Tasas USA

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **rates_restrictive** | Tasas restrictivas? 1=Si (bono 2Y >4%). Dinero caro, economia se frena. |
| **rates_accommodative** | Tasas bajas? 1=Si (bono 2Y <2%). Dinero barato, economia se estimula. |

### Curva USA

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **usa_curve_normalized** | Pendiente de la curva. Normal es positiva (tasas largas > cortas). |
| **usa_curve_inverted** | Curva invertida? 1=Si. ALERTA de recesion en 6-18 meses. Historicamente muy confiable. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |

---

# DS10 - FLUJOS Y FUNDAMENTALES (14 variables)

## Para que sirve?
Mirar los flujos reales de dolares entrando y saliendo de Colombia: inversion extranjera, exportaciones, importaciones, reservas del banco central. Util para horizontes mas largos (swing trading).

## Variables incluidas:

### Retornos

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **log_ret_5m** | Cambio 5 min. |
| **log_ret_1h** | Cambio 1 hora. |

### Inversion Extranjera Directa (IED)

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **ied_normalized** | Cuanta inversion extranjera llega a Colombia (fabricas, empresas, etc). Escalado de 0 a 1. Mas = mas dolares entrando = peso fuerte a largo plazo. |
| **ied_growing** | La IED esta creciendo vs trimestre anterior? 1=Si. Tendencia positiva. |

### Cuenta Corriente

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **ca_normalized** | Balance de cuenta corriente (exportaciones - importaciones + otros). Colombia siempre tiene deficit (importamos mas de lo que exportamos). Numero cerca de 0 = deficit pequeno = mejor. |
| **ca_improving** | El deficit se esta reduciendo? 1=Si. Buena senal para el peso. |

### Comercio

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **exports_growing** | Exportaciones creciendo? 1=Si. Mas exportaciones = mas dolares entrando. |
| **trade_improving** | Balanza comercial mejorando? 1=Si. |

### Tipo de Cambio Real (ITCR)

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **itcr_deviation** | El peso esta caro o barato vs su historia? Positivo = peso "caro" (puede devaluarse). Negativo = peso "barato" (puede apreciarse). |
| **itcr_change_1m** | Cambio en el ultimo mes. |

### Reservas Internacionales

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **reserves_falling** | Reservas del Banco de la Republica cayendo? 1=Si. Significa que BanRep esta vendiendo dolares para defender el peso. No puede hacerlo indefinidamente. |

### Riesgo

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **embi_z** | Riesgo pais Colombia. |

### Hora

| Variable | Que significa en lenguaje simple |
|----------|----------------------------------|
| **hour_sin** | Hora (parte 1). |
| **hour_cos** | Hora (parte 2). |

---

# Resumen: Cual Dataset Usar?

| Situacion | Dataset Recomendado |
|-----------|---------------------|
| Primera vez, probar sistema | DS1 Minimo |
| Trading tecnico puro | DS2 Tecnico |
| **Uso general / Produccion** | **DS3 Macro Core** ⭐ |
| Reducir operaciones, solo las mejores | DS4 Anti-Sobreoperar |
| Detectar cambios de mercado | DS5 Regimenes |
| Estrategia de tasas de interes | DS6 Carry Trade |
| Seguir petroleo, cafe, oro | DS7 Commodities |
| Mercados volatiles, crisis | DS8 Riesgo |
| Anticipar movimientos de la Fed | DS9 Fed Watch |
| Horizontes mas largos | DS10 Flujos |

---

# Glosario de Terminos

| Termino | Significado |
|---------|-------------|
| **RSI** | Relative Strength Index - mide sobrecompra/sobreventa |
| **ATR** | Average True Range - mide volatilidad |
| **ADX** | Average Directional Index - mide fuerza de tendencia |
| **Bollinger** | Bandas que muestran precio "normal" vs extremo |
| **DXY** | Dollar Index - dolar vs canasta de monedas |
| **VIX** | Indice de volatilidad/miedo de Wall Street |
| **EMBI** | Emerging Markets Bond Index - riesgo paises emergentes |
| **Brent/WTI** | Tipos de petroleo (europeo/americano) |
| **Carry Trade** | Pedir prestado en moneda con tasa baja, invertir en tasa alta |
| **COLCAP** | Indice de la bolsa de Colombia |
| **BanRep** | Banco de la Republica (banco central Colombia) |
| **Fed** | Federal Reserve (banco central USA) |
| **TPM** | Tasa de Politica Monetaria (tasa BanRep) |
| **IBR** | Indicador Bancario de Referencia (tasa interbancaria) |
| **IED** | Inversion Extranjera Directa |
| **ITCR** | Indice de Tasa de Cambio Real |
| **Z-score** | Valor normalizado (cuantas desviaciones de la media) |
| **Risk-On** | Inversionistas buscan riesgo/retorno |
| **Risk-Off** | Inversionistas huyen a activos seguros |
| **Hawkish** | Politica monetaria agresiva (subir tasas) |
| **Dovish** | Politica monetaria relajada (bajar tasas) |
