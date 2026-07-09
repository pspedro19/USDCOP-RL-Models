# Historia de Estrategias, Lineage de Datos y Veredictos — Oro · BTC · (COP referencia)

> **SSOT del ciclo de ciencia de datos por activo**: qué datos entran, con qué frecuencia,
> qué transformaciones, qué estrategias se probaron, sus resultados COMPLETOS y el veredicto
> honesto. Actualizado 2026-07-07 tras la poda del operador ("solo quedan las ganadoras").
> Metodología fija (los 3 activos): **diseño/entrenamiento ≤ dic-2024 · OOS = TODO 2025 ·
> 2026 = producción/forward**. Disciplina: `.claude/rules/quant-constitution.md`.

## 1. Ingeniería de datos por activo

| | XAU/USD (Oro) | BTC/USDT | USD/COP (ref) |
|---|---|---|---|
| Frecuencia base | **Diaria**, cierre 17:00 ET (NY close) | **Diaria**, cierre 00:00 UTC | 5m sesión + diaria |
| Fuente | histórico 2004→ (seed parquet) | Binance spot klines público, 3,245 barras 2017→ | TwelveData |
| Calendario | Lun–Vie (validador CTR-DQ-OHLCV-001: ancla fecha en UTC — bug "Sunday pile-up" corregido) | **24/7**, anualización **√365** | Lun–Vie 8:00–12:55 COT |
| Anti-leakage | features `shift(1)`; fit congelado ≤2024 | igual + funding `shift(1)` merge_asof backward | macro T-1, norm train-only |
| Transformaciones | ATR/ADX Wilder, Hurst suavizado, SMA 63/126/252, z_sma50, rvol_20, dist_max_52w | SMA/breakout, rvol, Hurst, régimen ciclo, z_funding (derivados mig-052: 2,493 días funding/OI) | 21 features H5 (Ridge/BR/XGB) |
| Sizing | vol-target 10% ann, floor 6%, cap 1.5× | vol-target 30%, **spot-only exposición ∈ [0,1]** | vol-target + regime gate + effective HS |
| Régimen | classify_regime paramétrico por AssetProfile (A10-01: pivote 0.5 explícito hasta fit D1) | ciclo (trend/chop) dwell 5 | Hurst gate (0.52/0.42) |

## 2. Horarios de ejecución (si operaran en producción)

- **Oro**: decisión diaria al cierre NY 17:00 ET (≈16–17:00 COT); ciclo DS Dom 01:45 COT. Sin venue de ejecución cableado (web/backtest only).
- **BTC**: rebalanceo diario de exposición al cierre 00:00 UTC (19:00 COT), fill simulado al open +1bps; 24/7 sin cierre forzado; ciclo DS Dom 02:00 COT. Ejecutable spot vía MEXC/SignalBridge (paper).
- **COP**: cadena H5 semanal (señal Lun 08:15 COT, entrada Lun 09–12:55, TP/HS cada 30min, cierre Vie 12:50).

## 3. Historia completa de estrategias y veredictos (poda 2026-07-07)

### BTC/USDT — ganadora: `btc_trend_b2` → **producción-PAPER** (protocolo WITHDRAWAL-PROTOCOL-BTC.md)

| Estrategia | Full 2018→26 (Calmar/Sharpe) | DSR | B1′ | Costos ×2/×3 | OOS-2025 | Veredicto |
|---|---|---|---|---|---|---|
| **btc_trend_b2** (trend diario) | **1.833 / 1.40** | **0.9987 ✓** | ✓ (vs 0.36) | 1.62/1.36 ✓ | −1.4% plano | **GANADORA → paper 26 sem** |
| btc_hodl_b1 (HODL vol-target) | ~0.8 Sharpe | — | (es baseline) | — | — | baseline, se mantiene visible |
| btc_exposure_s3 (regime-gated) | 0.62 Sharpe | — | no bate B2 | — | — | **ARCHIVADA** (necesita on-chain no integrado) |
| btc_trend_funding_s4 (funding gate) | 1.825 / 1.33 | 0.9976 | ✓ | 1.41/1.11 | −1.42% | **ARCHIVADA — H-POS-01 refutada**: no añade sobre B2 |

### XAU/USD — veredicto honesto: **el timing no bate a la exposición constante**

| Estrategia | Full 2004→26 (Calmar/Sharpe) | DSR | B1′ | OOS-2025 | Veredicto |
|---|---|---|---|---|---|
| gold_trend_ens (SMA 3/6/12m) | **0.171 / 0.587** | **0.989 ✓** | **✗ (vs 0.223)** | +34.8%, Calmar 8.1 | mejor variante de trend (H-XAU-TREND-01 ✓ bate B2) pero **falla B1′** → experimental |
| gold_trend_b2 (trend simple) | 0.128 / 0.362 | 0.892 ✗ | ✗ | +23.8% | superada por ENS; queda como baseline trend |
| **gold_long_only_b1** (HODL vol-target) | B1′ Calmar 0.223 | — | (es la B1′) | — | **GANADORA honesta de Oro**: la exposición constante vol-targeted rinde más que cualquier timing probado |
| gold_regime_gated_v1 | peor que ambas | — | ✗ | — | **ARCHIVADA** |

> **Conclusión constitución §3.3**: en Oro *"el baseline ES la estrategia"* — lo cosechable es
> beta vol-targeted, no timing. Toda hipótesis nueva de Oro debe batir Calmar 0.223 de B1′.

### Hipótesis: estado del registro (fuertes que quedan vivas)
- ✅ corridas: H-XAU-TREND-01 (ENS>B2 sí; ENS>B1′ no), H-POS-01 BTC (refutada), H-COP-V11-01 (IC incluye 0), H-LATAM-02 (refutada), H-PORT-01 (no concluyente N=52).
- 🔜 pendientes de mayor valor: **H-COP-CARRY-00** (medición swap broker — 0 compute, gate de toda la tesis carry), H-BTC-CYCLE-02 + on-chain (necesita extractores mig-052 restantes), D1 Gold (fit de umbrales Hurst por activo).

## 4. Producción-paper BTC (cableado 2026-07-07)
`run_btc_pipeline.py --phase production` exporta `summary_btc_trend_b2.json` + `approval_state_btc_trend_b2.json` (PENDING + gates + DSR + deploy_manifest paper) + `trades/btc_trend_b2.json` → selector multi-estrategia en `/production` → Vote-2 en `/dashboard` (por-sid) → deploy L4b (`conf.strategy_id`) → PAPER (PreTradeGate simula). Graduación a dinero real: **solo** vía WITHDRAWAL-PROTOCOL-BTC.md firmado.

## 5. Ronda de hipótesis 2026-07-07 (directiva del operador) — lineage-verificado

**Lineage backtesteable confirmado**: BTC OI/long-short = forward-only (31 días, NO backtesteables);
funding 2019→ (refutado); DXY/VIX daily 2020→ (1,622 obs); IBR 2008→; EMBI/prime 2020→.

| ID | Activo | Prior ex-ante | Resultado | Trials |
|---|---|---|---|---|
| H-BTC-VOLBRK-01 | BTC | rvol z>2 (252d, causal shift-1) ⇒ corta exposición ×0.5 — solo frena | **CORRIDA: NO AÑADE** — S5 idéntica a B2 (Calmar 1.832=1.832; ADX/SMA ya evita los spikes). No se adopta | +1 (N=5) |
| **H-XAU-DXY-01** | Oro | Tilt de EXPOSICIÓN sobre B1 (no timing): mult = 1.0 si ret20d(DXY)<0; 0.6 si >+1σ — el denominador es el ancla macro diaria disponible. **Bar: Calmar > 0.223 (B1′)** | PENDIENTE — diseño 2020-24, OOS=2025; sensibilidad pre-reg. {0.5,0.7} se reporta completa | +3 al correr |
| **H-COP-CORE-01** | COP | Carry condicionado (OLA4 C4): short USDCOP si IBR−prime>2pp Y no risk-off (zVIX>1.5 ∨ zΔEMBI20>1.5); mecánica TP/HS v11 congelada. Juez vs NULL-A (Calmar) | PENDIENTE — datos completos en macro_indicators_daily; **H-COP-CARRY-00 (swap broker) sigue siendo el gate de ejecución** | +1 (+6 sens.) |

> Ningún resultado de esta ronda se eligió mirando el OOS: priors declarados arriba ANTES de
> correr; S5 se corrió con su prior único (z=2.0) y se descarta por no añadir, no por celda.

### Resultados de la ronda (corridos 2026-07-07, todas las celdas reportadas)

**H-XAU-DXY-01 — NO RECHAZA H0 (dirección correcta, magnitud insuficiente).**
Las 3 celdas mejoran a B1 de forma consistente (señal robusta de signo) pero ninguna alcanza el bar:
| Celda | Calmar full | Sharpe | OOS-2025 |
|---|---|---|---|
| B1 (bar 0.223 en ventana B1′) | 0.107 | 0.503 | +38.0% |
| tilt 0.5 | 0.115 | 0.548 | — |
| **tilt 0.6 (prior)** | 0.113 | 0.539 | +34.9% |
| tilt 0.7 | 0.112 | 0.531 | — |
El MaxDD no cambia (−45.5%, es pre-2020 donde no hay DXY). El freno por dólar-fuerte suma ~+0.04
Sharpe pero cuesta retorno en el bull 2025. **No se adopta.** +3 trials.

**H-COP-CORE-01 — NO RECHAZA H0 (la tesis carry-condicionado está muerta en el régimen 2025).**
NULL-A 2025: ann +3.77%, Calmar 1.52 (bar). Grid 3×3 (carry pp × risk z):
- carry>2.0 y >2.5: **0 semanas activas en TODO 2025** — el diferencial IBR−prime nunca superó 2pp.
- carry>1.5: 7–9 semanas activas, Calmar 0.32 / −0.21 / −0.01 — las semanas CON carry fueron las malas.
**Caveat de proxy declarado**: se usó `prime_US` (≈FFR+3pp) por ser la única tasa US diaria en
macro_daily; el prior original era vs FFR ⇒ el umbral efectivo es ~3pp más duro. Refinamiento
registrado (no un re-pick): re-correr con FEDFUNDS mensual (macro_indicators_monthly) si se desea
la lectura exacta del prior — pero la celda más laxa disponible ya pierde contra NULL-A, y esto
además refuerza H-COP-CARRY-00: con diferencial teórico <2pp, el swap real del broker (≤ teórico)
hace la cosecha aún menor. +1 trial prior, +8 celdas.

**Balance de la ronda**: 3 hipótesis corridas, 3 no-adopciones honestas (+7 trials nuevos al
presupuesto DSR). Las campeonas siguen: btc_trend_b2 (paper), gold_long_only_b1, smart_simple_v11
(forward). Próxima frontera con datos NUEVOS (no con los de hoy): on-chain BTC (mig-052 extractores),
DFII10 real-yield diario para Oro, y la medición del swap del broker para COP.

## 6. Ronda MR-dip 2026-07-07 (directiva: "estrategias con mejor WR/alfa con la info que tenemos")

> **PRE-REGISTRO ex-ante — escrito ANTES de correr cualquier backtest.** Familia sin probar con
> datos existentes: mean-reversion de corto plazo (compra del dip post-venta-forzada). Historia
> económica: proveer liquidez al vendedor forzado (liquidaciones apalancadas en BTC; stop-outs y
> rebalanceos en Oro) cosecha el rebote. Estructura de WR alto por construcción (holds cortos,
> muchos trades). Confirmado que macro daily NO tiene DFII10/T10YIE (sin real-yield diario).

**Priors congelados (idénticos ambos activos, cero optimización):**
- Trigger: `ret_3d < −1.5·σ3` con `σ3 = rvol_20_diaria·√3`, evaluado con info a t−1 (causal).
- Filtro de tendencia: `close > SMA200` a t−1 (no comprar cuchillos en bear).
- Hold: 5 días o hasta nueva señal (re-arma). Sizing: capa vol-target existente del activo.
- Sensibilidades pre-registradas (se reportan TODAS, elegir la mejor está prohibido):
  z∈{1.0, 2.0} y hold∈{3, 10}, one-at-a-time alrededor del prior (4 celdas extra por variante).

| ID | Variante | Bar de adopción (ex-ante) |
|---|---|---|
| H-BTC-MR-01a | MR standalone BTC (intent=dip window) | informativa (WR/PF); no compite sola |
| H-BTC-MR-01b | **Combinada**: intent = max(B2, MR) — MR cosecha el chop donde B2 está flat | Calmar Y Sharpe > B2 en diseño 2018–24, confirma OOS-2025, costos ×2, DSR>0.95 trial-aware |
| H-XAU-MR-01a | MR standalone Oro | informativa |
| H-XAU-MR-01b | **Overlay sobre B1**: posición = B1 +0.5× en ventana dip, cap 1.5 | Calmar Y Sharpe > B1 misma ventana, confirma OOS-2025, costos ×2, DSR |

Presupuesto: 2 priors + 8 sens + 2 standalone = **+12 trials** al correr. Diseño ≤2024, juez OOS=2025.

### Resultados (corridos 2026-07-07, `scripts/analysis/mr_dip_hypothesis.py`, todas las celdas)

**H-BTC-MR-01 — NO RECHAZA H0 (ninguna celda combinada bate a B2 en diseño).**
| Celda (diseño 2018–24) | Sharpe | Calmar | vs B2 (1.581 / 2.839) |
|---|---|---|---|
| prior combinada | 1.445 | 1.838 | ✗ ambos |
| z2.0 combinada (mejor celda) | 1.556 | 2.295 | ✗ ambos |
| z1.0 / h3 / h10 combinadas | 1.107–1.437 | 0.95–2.20 | ✗ |
| standalone prior (informativa) | 0.199 | 0.107 | WR 44.8%, PF 1.44 — el premium de dip en BTC es débil |
Prior combinada: costos×2 Calmar 1.673, DSR(49)=0.81 < 0.95. **No se adopta.** El trigger −1.5σ
compra ANTES de que termine la cascada de liquidaciones (WR<50% standalone). +6 trials.

**H-XAU-MR-01 — combinada NO pasa el bar (DSR 0.0009, muere a costos ×2: Calmar 0.097→0.045).**
Pasa Calmar+Sharpe vs B1 en diseño (0.465/0.097 vs 0.415/0.085) pero el swap 2× sobre exposición
1.5 la mata y el DSR con 49 trials es ~0. **No se adopta como campeona.** +6 trials.

**HALLAZGO honesto (standalone Oro, pre-registrada solo como informativa):** `gold_dip_mr`
standalone en diseño 2004–24: **WR 65.5%, PF 2.23, MaxDD −9.4%, Sharpe 0.533, Calmar 0.153**
(mejor Sharpe/Calmar/DD que B1, pero en mercado solo ~8% del tiempo → retorno total 36% en 21a).
OOS-2025: +7.5%, PF 10.98 (4 trades — sin solemnidad estadística, N<20). **Adoptarla ahora por
estos números sería selección post-hoc** (su bar era informativo y el 2025 ya fue mirado).
Camino honesto registrado: **H-XAU-MRSLEEVE-01** — sleeve MR standalone en paper, juez = forward
2026-H2→2027 (período siguiente), sin re-tunear parámetros (prior 1.5σ/5d congelado). Pendiente
de Vote humano para publicarla como experimental.

**Balance ronda MR**: +12 trials (total ≈49). Campeonas intactas. La frontera con datos actuales
queda formalmente agotada: trend, régimen, DXY, funding, vol-breaker y MR ya probados. Los
desbloqueos reales siguen siendo datos nuevos: on-chain BTC (mig-052), DFII10/COT/GLD para Oro,
swap broker para COP.

## 7. Veredicto OOS-2025 ÚNICO (directiva operador 2026-07-07: "métricas solo del 2025, como COP")

> `scripts/analysis/oos2025_report.py` — diseño/entrenamiento ≤2024, evaluación = TODO 2025,
> motor de costos real por activo, p-value = block-bootstrap sobre retornos DIARIOS (los conteos
> de trades son bajos: sin solemnidad estadística por-trade, constitución §6).

| Estrategia | Ret 2025% | Sharpe | p | WR% | PF | MaxDD% | Trades | $10K→ |
|---|---|---|---|---|---|---|---|---|
| **Oro · gold_long_only_b1** | **+38.4** | 3.08 | 0.000 | — | — | −4.1 | 1 | $13,836 |
| Oro · gold_trend_ens | +35.2 | 2.90 | 0.000 | — | — | −4.1 | 1 | $13,518 |
| Oro · gold_trend_b2 | +24.1 | 2.44 | 0.001 | — | — | −4.1 | 2 | $12,411 |
| Oro · gold_dip_mr (prior) | +7.5 | 1.87 | 0.043 | 75 | 10.98 | −1.1 | 4 | $10,752 |
| Oro · buy&hold | +65.6 | — | — | — | — | ~−8 | — | $16,560 |
| BTC · btc_hodl_b1 | +4.7 | 0.35 | 0.398 | — | — | −15.5 | 1 | $10,470 |
| **BTC · btc_trend_b2** | **−1.4** | −0.05 | 0.619 | 25 | 1.03 | −8.6 | 8 | $9,863 |
| BTC · dip_mr / combinada | −3.5 / −5.4 | <0 | >0.7 | 33/31 | <1 | −8/−13 | 9/16 | <$9,700 |
| BTC · buy&hold | −7.3 | — | — | — | — | — | — | $9,270 |

**Lectura honesta:**
- **Oro 2025 = año de beta** (+65.6% B&H). TODO el retorno de las estrategias es captura parcial
  de beta vol-targeted (exposición media ~0.55 ⇒ +38.4%). Ningún timing añadió sobre estar largo.
- **BTC: NINGUNA estrategia tiene alfa validada en 2025** (todas p>0.35 salvo nada). B2 batió al
  B&H (−1.4% vs −7.3%, +5.9pp con menos DD) pero en absoluto es plana. Su promoción a paper
  descansa en la ventana larga (DSR 0.9987) — **bajo el bar solo-2025 no habría PROMOTE**; el
  juez limpio es el forward 2026 en paper, igual que COP v11.
- COP referencia: smart_simple_v11 2025 +25.63% (p=0.006 contaminado por selección, DSR<0.95)
  → congelada, forward 2026 es el juez. La misma vara aplica a los 3 activos.

## 8. Veredicto BTC + frecuencia de datasets + ronda "estrategia ACTIVA" (operador 2026-07-07)

### 8.1 VEREDICTO BTC (cerrado por directiva del operador)
**BTC NO mejora con más estrategias sobre los datos actuales.** 7 intentos (B1/B2/S3/S4-funding/
S5-volbrk/MR-dip/MR-comb) — ninguno bate a B2, y B2 mismo es plano en OOS-2025 (−1.4%, p=0.62).
**Moratoria de trials BTC hasta tener features de calidad nuevos**: on-chain MVRV-Z/NUPL/Puell +
ETF flows + stablecoin supply (extractores mig-052 pendientes). Cada trial extra sin datos nuevos
solo infla la deflación DSR sin posibilidad real de alfa.

### 8.2 Frecuencia y profundidad REAL de cada dataset (¿backtesteable?)
| Dataset | Frecuencia | Historia | ¿Backtesteable? |
|---|---|---|---|
| COP OHLCV 5m / diario | 5min sesión / diario | 2019→ / 2015→ | ✓ (motor v11) |
| COP inferencias semanales (9 modelos) | semanal | walk-forward 2025→ | ✓ pero DA~50% (ruido honesto) |
| Macro daily (DXY/VIX/UST/EMBI/Brent/IBR…) | diaria | 1954→ (por serie) | ✓ (T−1) |
| Oro OHLCV diario | diaria (NY close) | 2004→ (~5.6K barras) | ✓ |
| BTC OHLCV diario | diaria (00:00 UTC) | 2017→ (3.2K) | ✓ |
| BTC funding/OI | diaria / forward-only | 2019→ / 31 días | ✓ refutado / ✗ |
| Noticias (Investing/Portafolio) + LLM semanal | 3×/día + semanal | **2026→ (~350 art.)** | **✗ sin historia — solo overlay forward** |
| Inferencia semanal Oro/BTC (regla) | semanal | derivada del diario | = señal trend diaria re-muestreada |
> Noticias NO pueden entrar a un backtest honesto (no hay historia). Pueden usarse como overlay
> cualitativo forward (ya visible en /analysis), nunca como feature retro-etiquetada (look-ahead §4).

### 8.3 PRE-REGISTRO H-XAU-WKTPHS-01 — Oro ACTIVO con mecánica semanal COP-v11 (ex-ante)
El "pocos trades" de Oro no es falta de señal sino de MECÁNICA: B1 nunca cierra. COP opera 34×/año
porque su cadencia es semanal con TP/HS y re-entrada. Se trasplanta la mecánica (no los parámetros
fit) a Oro — priors tomados de la familia v11 congelada, cero fit sobre Oro:
- Cadencia: decisión al cierre del viernes (causal), entrada lunes open, salida por TP/HS intra-semana
  (high/low diario, HS primero si ambos el mismo día — conservador) o cierre viernes (week_end).
- Señal: ensemble votos {SMA63, SMA126, SMA252} ≥ 2/3 ⇒ LONG; si no, semana flat.
- TP = +1.0×ATR_w · HS = −2.0×ATR_w (ATR_w = ATR14_diario×√5; HS 2.0 = piso duro v11, no se baja).
- Sizing: vol-target 10% ann, cap 1.5 (capa existente). Costos: motor Oro (2bps/lado + swap 2.5%).
- Sensibilidades pre-registradas TP∈{0.75, 1.5} — TODAS las celdas se reportan, elegir está prohibido.
- Bar de adopción: OOS-2025 Sharpe>0 con p<0.05 (bootstrap diario) Y ≥20 trades/año Y no
  degradar Calmar vs B1 en diseño ≤2024 más de lo que compensa el DD. +3 trials.

### 8.4 Resultados H-XAU-WKTPHS-01 (`scripts/analysis/gold_weekly_tphs.py`, todas las celdas)

| Celda | Ventana | Ret% | Sharpe | Calmar | MaxDD% | p | Trades | WR% | PF |
|---|---|---|---|---|---|---|---|---|---|
| **prior TP=1.0** | diseño≤24 | +89.8 | 0.394 | 0.110 | −27.2 | 0.036 | 697 | 56.8 | 1.33 |
| **prior TP=1.0** | **OOS2025** | **+39.6** | **3.008** | **8.51** | **−4.5** | **0.000** | **52** | **76.9** | **3.95** |
| TP=0.75 | diseño≤24 / OOS | +50.2 / +43.4 | 0.27 / 3.63 | 0.06 / 10.4 | −29.8 / −4.0 | 0.103 / 0.000 | 697 / 52 | 57.5 / 80.8 | 1.26 / 5.05 |
| TP=1.5 | diseño≤24 / OOS | +78.8 / +39.3 | 0.35 / 2.83 | 0.08 / 8.4 | −32.9 / −4.5 | 0.053 / 0.001 | 697 / 52 | 55.8 / 75.0 | 1.31 / 3.93 |

**Honest-gate del prior**: bar pre-registrado **PASA** (p=0.000 OOS, 52 trades/año, Calmar diseño
0.110 > B1 0.085 con MaxDD −27 vs −45). PERO gates transversales de PROMOTE **fallan en diseño**:
B1′ (exposición constante emparejada) Calmar 0.192 > 0.110 — la mecánica activa CUESTA ~0.19 de
Sharpe en 21 años (churn + swap + gaps por encima del TP); costos ×2 diseño Calmar 0.025 (casi
muere), ×3 negativo. En OOS-2025 sí bate B1′ (8.51 vs 5.46), sobrevive ×2 (7.55) y DSR-año 0.9565.

**Veredicto**: `gold_weekly_tphs` = **candidata EXPERIMENTAL-paper** (NO PROMOTE — constitución
§3 exige B1′ y costos×2 en diseño). Es la versión ACTIVA honesta del oro con mecánica v11: 52
decisiones/año, WR 77% en 2025, comparable 1:1 con COP (34 trades, WR 70.8%, +25.6%). Juez limpio
= forward 2026. Publicación al registry condicionada a Vote humano. +3 trials (total ≈52).

### 8.5 PRE-REGISTRO H-BTC-WKTPHS-01 — mismo trasplante mecánico a BTC (orden del operador)

> La moratoria BTC (§8.1) aplica a FAMILIAS DE SEÑAL nuevas sobre los mismos datos; el operador
> ordenó el trasplante mecánico semanal (misma familia WKTPHS ya declarada). Priors idénticos,
> cero fit sobre BTC: señal viernes = votos SMA{63,126,252} ≥ 2/3; entrada lunes open; TP=+1.0×ATR_w,
> HS=−2.0×ATR_w; capa de riesgo BTC (vol-target 30%, floor 30%, **spot-only exposición ≤1.0**);
> costos motor BTC (13 bps/lado, sin swap); semanas ISO sobre calendario 24/7 (lunes = primer día
> de la semana ISO). Sens TP∈{0.75,1.5}. Bar: idéntico a §8.3. +3 trials.

**Resultados (`scripts/analysis/btc_weekly_tphs.py`, todas las celdas) — REFUTADA ROTUNDAMENTE:**
| Celda | Diseño 18–24 (Sharpe/Calmar) | OOS-2025 | vs B2 (1.58/2.84) |
|---|---|---|---|
| prior TP=1.0 | 0.556 / 0.238 | **−28.7%**, WR 45.7, PF 0.70 | ✗ lejos |
| TP=0.75 / TP=1.5 | 0.34/0.11 · 0.79/0.42 | −24.2% / −23.7% | ✗ |
Prior falla B1′ en diseño (0.402 > 0.238) y muere a costos ×2 (0.118). En 2025 TODAS las celdas
pierden −24…−29% (35 re-entradas semanales × 26 bps ida-vuelta + whipsaw en año lateral).
**La mecánica semanal TP/HS NO se trasplanta a BTC**: la vol de BTC es 3–5× la del oro/FX y el
week_end forzado corta las tendencias largas que son TODO el edge de B2 (§8.1 confirmada
empíricamente). Moratoria BTC ratificada. +3 trials (total ≈55).

### 8.6 PRE-REGISTRO H-WKTPHS-HORIZON-01 — horizonte flexible (directiva operador 2026-07-07)

> Operador: "quizás no cortar semana a semana sino cada 2–3 semanas, mensual o bimestral, usando
> las inferencias de nuestros horizontes". Ataca la causa raíz del fallo BTC (week_end corta
> tendencias). Grilla de HORIZONTE declarada ex-ante: N ∈ {1, 2, 3, 4, 8} semanas ISO por bloque.
> Mecánica idéntica (§8.3/§8.5): señal al cierre del bloque anterior (votos SMA ≥2/3), entrada
> primer día del bloque, TP=+1.0×ATR_N / HS=−2.0×ATR_N con ATR_N = ATR14d×√(días del bloque:
> 7N BTC, 5N Oro), salida TP/HS o fin de bloque. TP/HS multiplicadores FIJOS al prior (no se
> re-abren). **Protocolo de selección honesto: la celda se elige en DISEÑO ≤2024 (Calmar), el
> OOS-2025 solo CONFIRMA la elegida; todas las celdas se reportan y cuentan.** +10 trials (5×2
> activos). Bar BTC: batir a B2 en diseño (Calmar Y Sharpe) — si ninguna lo bate, la moratoria
> queda ratificada también en horizonte.

**Resultados (`scripts/analysis/wktphs_horizon_grid.py`, todas las celdas — total trials ≈65):**

*BTC — REFUTADA EN TODO EL RANGO 1sem–2meses.* Mejor celda en diseño: N=3 (Sharpe 0.692 /
Calmar 0.478, p=0.041) — sigue a **6× de distancia** de B2 (1.581/2.839) y pierde contra B1′
(0.40). El perfil por-N es no-monótono (0.24→0.11→0.48→0.14→0.13) = superficie de ruido, no
estructura. El OOS de N=3 (+18.5%) es seductor pero elegirla por eso está prohibido; por diseño
ninguna celda pasa el bar. **La moratoria BTC queda ratificada también en horizonte: ni semanal
ni mensual ni bimestral — B2 pasiva sigue siendo lo único defendible hasta datos on-chain.**

*Oro — la selección por diseño (Calmar) elige N=8 (bimestral):* diseño Sharpe 0.628 / Calmar
0.112 / p=0.002 (el mejor perfil de la grilla); OOS-2025 confirma: +26.1%, Sharpe 3.19, 7 trades,
WR 85.7%, PF 20.7, MaxDD −3.1%. N=1 (semanal, 52 trades) queda segunda (diseño 0.110). N=2 tiene
el mejor OOS (+50.2%) pero elegirla por el OOS está prohibido (diseño 0.087, 4ª de 5). TODAS las
celdas siguen por debajo de B1′ en diseño (0.19) ⇒ ninguna es PROMOTE; N=8 y N=1 son candidatas
**experimental-paper** equivalentes en honestidad — elegir entre ellas (actividad 52/año vs 7/año)
es preferencia OPERATIVA del dueño, no estadística. Juez = forward 2026.

### 8.7 PRE-REGISTRO H-XAU-DYNEXIT-01 — salidas DINÁMICAS decididas por la estrategia (operador)

> Operador: "que la estrategia misma decida entrar o salir — una semana, un mes, dos meses —
> dinámicamente". Principio de ingeniería: **el calendario no tiene información; las salidas se
> disparan por el mercado**. Esto es lo que btc_trend_b2 ya hace (por eso batió a todos los
> bloques §8.6); se construye la versión Oro con trade discreto + trailing. Priors ex-ante:
> - Entrada: votos SMA{63,126,252} ≥ 2/3 al cierre (causal, entra al open siguiente).
> - Salida (el primero que dispare): (a) señal muere (votos < 2/3 al cierre → sale open
>   siguiente); (b) **Chandelier trailing = max(close desde entrada) − 3.0×ATR14** (intra-día,
>   clásico ex-ante 3.0; sens {2.0, 4.0}).
> - Sizing vol-target al momento de entrada (10%/floor 6%/cap 1.5), fijo durante el trade.
> - Costos motor Oro. Hold resultante = variable (días→meses), decidido por el precio.
> - Selección en diseño ≤2024, OOS-2025 confirma; se reporta distribución de duración de holds.
> - Bar: mejor Calmar de diseño que la mejor celda de calendario (N=8, 0.112) — si el exit
>   dinámico no bate al calendario, el calendario gana por parsimonia. +3 trials.

**Resultados (`scripts/analysis/gold_dynamic_exit.py`, todas las celdas — total trials ≈68):**

| Celda | Ventana | Ret% | Sharpe | Calmar | MaxDD% | p | Trades | WR% | PF | hold med/max |
|---|---|---|---|---|---|---|---|---|---|---|
| **prior M=3.0** | diseño≤24 | +165.4 | 0.576 | **0.204** | **−22.6** | 0.003 | 212 | 29.7 | 2.10 | 9d / 97d |
| **prior M=3.0** | OOS2025 | +38.0 | 2.75 | 5.62 | −6.5 | 0.000 | 8 | 87.5 | 86 | 33d / 58d |
| M=2.0 | diseño / OOS | +161.9 / +41.0 | 0.60 / 2.97 | 0.200 / 7.07 | −22.7 / −5.6 | 0.003 / 0.000 | 307 / 12 | 35.5 / 66.7 | 1.8 / 9.9 | 6/60 · 22/50 |
| M=4.0 | diseño / OOS | +112.4 / +23.1 | 0.46 / 2.35 | 0.166 / 4.86 | −21.3 / −4.6 | 0.011 / 0.001 | 172 / 3 | 26.2 / 100 | 2.1 / — | 9/114 · 100/114 |

**El PRIOR (M=3.0) gana por Calmar sin elegir nada** — mejor resultado de timing en Oro de todo
el programa: bate a TODA la grilla de calendario (0.204 vs 0.112), a B1 (0.085) por 2.4×, y es
**la primera estrategia de Oro que bate a B1′ en la métrica primaria** (Calmar 0.204 vs 0.192;
Sharpe 0.576 vs 0.586 — empate). Sobrevive costos ×2 (0.109 > 0). MaxDD −22.6% vs −45.5% de B1.
El hold es genuinamente dinámico (mediana 9 días, máx 97 en diseño; 33/58 en 2025). **DSR(68
trials) = 0.002 ⇒ NO hay claim de alfa estadística** — lo que hay es INGENIERÍA DE RIESGO
superior sobre la misma beta (perfil trend: WR diseño 29.7% con PF 2.1; el WR 87.5% de 2025 es
el año excepcional, no la expectativa). **Veredicto: `gold_dynamic_exit` reemplaza a las
variantes de calendario como LA candidata experimental-paper de Oro** (§8.4/§8.6 quedan
superadas por diseño). Juez = forward 2026. Publicación condicionada a Vote humano.

### 8.8 PRE-REGISTRO H-DYNEXIT-XASSET-01 — misma familia dinámica en USDCOP y BTC (operador)

> Trasplante de §8.7 sin re-tunear (M=3.0 prior, sens {2,4}, señal votos SMA{63,126,252} ≥2/3,
> salida = señal-muere ∨ Chandelier). Diferencias declaradas ex-ante por estructura del activo:
> - **BTC**: long-flat spot-only (sin shorts), capa de riesgo BTC (vol-target 30%, floor 30%,
>   cap 1.0), costos 13 bps/lado. **Bar: batir a B2 en diseño 2018–24 (Sharpe 1.581/Calmar 2.839).**
> - **USDCOP**: SIMÉTRICA long/short (el edge histórico del COP es el short): votos_up≥2 ⇒ LONG,
>   votos_up≤1 ⇒ SHORT; trailing espejo para shorts (min close + M×ATR). Vol-target 10%, cap 1.5.
>   Costos 5 bps/lado (conservador vs v11: 0 maker + 1bp slip) y **sin crédito de carry** (short
>   USDCOP cobra IBR−FFR>0 — omitirlo es conservador). Seed diario 2020→ ⇒ diseño efectivo
>   2021–2024 (burn-in SMA252). Bar: contexto vs smart_simple_v11 2025 (+25.6%, 34 tr) y NULL-A
>   2025 (Calmar 1.52) — v11 NO se toca (congelada); esto solo mide si la familia viaja.
> +6 trials (total ≈74).

**Resultados (`scripts/analysis/cop_btc_dynamic_exit.py`, todas las celdas):**

*USDCOP — LA FAMILIA NO VIAJA (refutada rotundamente).* Prior diseño 2021–24: +2.0%, Sharpe
0.097, p=0.37; OOS-2025: **−8.2%, WR 20%** (todas las celdas OOS negativas −6.5…−8.7%). El
trend-following diario NO funciona en COP — consistente con todo lo sabido (R²<0, DA~50%): el
edge del COP vive en la mecánica semanal TP/HS + regime gate + sesgo short/carry, no en la
tendencia diaria. **v11 (congelada) queda intocada y muy por encima** (+25.6% 2025 vs −8.2%).

*BTC — el mejor retador hasta ahora, pero B2 retiene el trono (3ª ratificación).* Prior diseño:
Sharpe 1.029 / Calmar 1.171; sens 2.0/4.0: 1.22/1.38 y 1.21/1.38 — TODAS debajo de B2
(1.581/2.839). OOS-2025 plano (−2.8…+0.5%) como B2. La señal de votos 63/126/252 es PEOR para
BTC que la nativa de B2 (SMA100+ADX25). Con esto B2 ha derrotado 10+ retadores en igualdad de
condiciones honestas.

**Cierre de la campaña dinámica (total ≈74 trials): cada activo tiene SU mecánica** — COP =
semanal TP/HS+gate (v11), Oro = dynexit Chandelier (§8.7, única que bate B1′), BTC = trend
pasivo B2 (sus salidas YA son dinámicas por señal). La familia ganadora de un activo no viaja
a los otros: la microestructura (vol, sesión, persistencia de tendencia) manda.

## 9. PUBLICACIÓN "lo mejor de lo mejor" (orden del operador, ejecutada 2026-07-07)

Registry podado a **exactamente 1 campeona por activo** (`scripts/pipeline/publish_gold_dynexit.py`;
todo lo demás `archived`, bundles inmutables intactos):

| Activo | Campeona visible | Status | Full window | OOS-2025 | Honest-gate |
|---|---|---|---|---|---|
| USD/COP | smart_simple_v11 | **production** | +25.6% (2025) | +25.6%, 34 tr, WR 70.8 | congelada; juez = forward 2026 |
| XAU/USD | **gold_dynamic_exit v1.0.0** | experimental (REVIEW) | +253.9%, Sharpe 0.69, Calmar 0.259, DD −22.6, 227 tr | **+38.0%, Sharpe 2.75, p=0.0002, 9 tr** | **bate B1′ (0.259>0.227)** ✓ costos ×2 (0.151) ✓ DSR 0.0077 ✗ → sin claim de alfa (riesgo-ingeniería sobre beta) |
| BTC/USDT | btc_trend_b2 | experimental (paper pendiente Vote-2) | +351%, Sharpe 1.40, DSR 0.9987 | −1.4% (plano) | juez = forward 2026 |

`gold_long_only_b1` → archived (superada por dynexit en Calmar y DD; sigue siendo el B1 de
referencia en los honest-gates). Verificado en navegador: selector con 3 estrategias, replay
dinámico de dynexit funcional (serie diaria, ventana 2025 = +37.4%/9 trades). El bundle publica
`verdict_notes` + `honest_gate` + DSR para que el Vote-2 humano vea la letra chica.
