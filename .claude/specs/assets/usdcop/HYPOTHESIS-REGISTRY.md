---
kind: as-built
status: PARTIAL
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - scripts/analysis/cop_trials_dsr.py
  - scripts/analysis/cop_null_suite.py
  - scripts/analysis/portfolio_layer.py
# Conteo de trials LEGIBLE POR MÁQUINA. `scripts/analysis/profitability_evidence.py` lo lee de
# aquí y lanza excepción si falta: el DSR jamás debe depender de un número hardcodeado en el
# código (era el caso en cop_trials_dsr.py:TRIALS_SCENARIOS y publish_gold_dynexit.py:48).
n_trials_total: 59
n_trials_scenarios: [46, 58, 72]   # conservador / central / amplio — se publican los tres
n_trials_sources:
  - "EXPERIMENT_LOG.md: FC-H5-SIMPLE-001 + FC-SIZE-001 (reconstrucción retroactiva v1.0→v11)"
  - ".claude/specs/assets/usdcop/EXP-DIR-001-directional-trials.md (27 trials direccionales)"
  - "public/data/strategies/{smart_simple_v11,smart_simple_aggr}/backtests/* (5 bundles = suelo)"
sigma_trials: null                 # nunca se persistió la dispersión de Sharpe entre trials
sigma_trials_grid: [0.05, 0.10, 0.15]   # titular = el DSR MÍNIMO de la rejilla
---
# HYPOTHESIS-REGISTRY — USD/COP (retroactivo + prospectivo)

> **Creado 2026-07-06 (G2 del plan maestro, hallazgo I-1 de la auditoría).** El track COP no
> tenía registro de trials — la disciplina vivía solo en BTC. Este archivo reconstruye
> **retroactivamente** el conteo de trials v1.0→v11 y registra prospectivamente todo lo
> nuevo (propuesta COP del plan). Regla heredada (`.claude/rules/quant-constitution.md`):
> cada versión / celda de grid / gate mirado = **1 trial**; ningún claim de edge sin DSR
> recomputado con el conteo actualizado.
>
> Contract: CTR-QUANT-CONSTITUTION-001 · Script: `scripts/analysis/cop_trials_dsr.py`

---

## 1. Reconstrucción retroactiva de trials (v1.0 → smart_simple_v11)

| Fuente (EXPERIMENT_LOG.md) | Trials | Evidencia |
|---|---|---|
| FC-H5-SIMPLE-001: eval v1.0 | 1 | "v1.0 Results (OOS 2025)" |
| FC-H5-SIMPLE-001: **grid 2D hs/tp sobre el OOS 2025** | **42** | "Current config ranks **#8 of 42** in 2D grid search" |
| FC-H5-SIMPLE-001: re-eval v1.1 (mismo OOS) | 1 | "v1.1 Results (OOS 2025 + 2026 YTD)" |
| FC-SIZE-001: grid sizing (baseline + celdas tv/ml, mismo OOS) | ~6 | tabla "Baseline (1x), tv=12%/ml=1.5x, …" |
| v1.1 → v2.0 (regime gate on/off, effective HS, +XGB, dyn leverage, fix retrain semanal) | ≥5 | CLAUDE.md "v1.1→v2.0 (2026-03-18)" |
| Variante A/B `smart_simple_aggr` | ≥1 | registry |
| **Cota inferior documentada** | **N ≈ 56** | (holgado: N=70 con versiones no registradas) |

## 2. DSR honesto de smart_simple_v11 (OOS 2025) — computado 2026-07-06

Serie semanal desde los trades publicados (`trades/smart_simple_v11_2025.json`, 34 trades,
52 semanas, **semanas flat = 0% real, no missing**): Sharpe semanal **0.3585**
(anualizado ≈ **2.59** por este método), skew −1.27, kurt 5.13, **PSR(1 trial) = 0.979**.

La dispersión entre trials (σ de Sharpe semanal entre celdas del grid) no se persistió →
se reporta **rango declarado**, nunca la celda amable:

| Escenario | σ_trials | DSR | ¿>0.95? |
|---|---|---|---|
| N=44, σ=0.05 (el MÁS caritativo) | 0.05 | **0.919** | **NO** |
| N=56, σ=0.10 (base) | 0.10 | **0.763** | **NO** |
| N=70, σ=0.15 (conservador) | 0.15 | **0.496** | **NO** |

### Lectura honesta (la conclusión de G2)

- **El p=0.0097 / p=0.006 celebrado es el p-value de la celda ganadora**, no un estimador
  insesgado: v1.1 salió de un grid sobre el OOS 2025 y se re-evaluó en el mismo OOS.
- **Bajo el mismo bar que Gold/BTC (DSR>0.95), v11 NO pasa en ningún escenario** (0.50–0.92).
  Nótese que `gold_trend_b2` fue cuestionado por DSR 0.921 — v11 en su escenario MÁS
  caritativo da 0.919.
- **Esto NO dice que v11 no tenga edge** — dice que el backtest 2025 no puede probarlo tras
  la selección. **El único juez limpio es el forward 2026** (reglas congeladas desde
  2026-03-18), gobernado por `WITHDRAWAL-PROTOCOL.md`.
- Recomputo exacto (grid re-corrido walk-forward, σ real entre celdas): pertenece a la suite
  COP-NULL (OLA 4).

## 3. Registro prospectivo (propuesta COP — se activan al comprometerse a correr)

| ID | H0 | Estado | Trials |
|---|---|---|---|
| H-COP-CARRY-00 | El swap real del broker (short) no traspasa ≥50% del diferencial teórico | PENDIENTE (medición, 0 compute) | 0 |
| H-COP-V11-01 | v11 no supera a NULL-A en Calmar | **CORRIDA 2026-07-06**: ΔCalmar=+5.39, **IC95=[−1.47, +33.4] INCLUYE 0** ⇒ H0 NO rechazada — no se puede afirmar que v11 supere a siempre-short; el forward 2026 decide | +1 |
| H-COP-CARRY-01 | El PnL short-USDCOP no se explica por carry (DECOMP) | PENDIENTE (OLA 4) | +1 |
| H-COP-CARRY-02 | Risk-off gate no mejora Calmar vs NULL-A | PENDIENTE (COP-CORE) | +1 (+6 sensibilidad pre-registrada) |
| H-COP-TREND-01 | TSMOM 4/8/13w no supera B1′ | PENDIENTE | +1 |
| H-COP-XLEAD-01 | MXN/CLP/Brent t−1 no mejoran DA del intent | PENDIENTE — **decisión 2026-07-06: CLP entra como FEATURE (daily close TwelveData `USD/CLP`), NO como activo propio**; MXN/BRL ya seedeados permiten correr la mitad del test sin esperar CLP. ⚠️ v11 CONGELADA: si el feature pasa, entra a una versión NUEVA evaluada en período posterior, jamás a v11 | +1 |

> Sensibilidades pre-registradas COP-CORE (cada celda = 1 trial, NO se elige la mejor):
> carry {1.5, 2.0, 2.5} pp × risk-off {1.0, 1.5, 2.0}σ.

## 4. Regla operativa desde hoy

1. Todo experimento COP nuevo se registra AQUÍ antes de correr (fila = ≥1 trial).
2. `smart_simple_v11` está **CONGELADA** (ver `WITHDRAWAL-PROTOCOL.md`): ningún cambio de
   parámetro se evalúa en 2025/2026-pasado; un diagnóstico sobre OOS solo genera hipótesis
   para el período siguiente.
3. Cualquier claim de edge del track recomputa el DSR con N actualizado
   (`scripts/analysis/cop_trials_dsr.py`).


## 5. Resultados COP-NULL (2026-07-06, `scripts/analysis/cop_null_suite.py`)

| Serie 2025 (semanal) | Ann | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|
| v11 (publicado) | +25.6% | −6.1% | 4.19 | 2.59 |
| NULL-A (siempre-short 1× + TP/HS v11) | +3.8% | −2.5% | 1.52 | 0.78 |
| **NULL-B (constante ~0.68× short, sin salidas)** | +16.5% | −2.2% | **7.67** | — |

**Lecturas honestas:**
1. **H-COP-V11-01:** ΔCalmar(v11,NULL-A)=+5.39 pero IC95 incluye 0 (N=52 semanas) → estadísticamente
   NO se puede afirmar que la capa ML supere a estar-corto; criterio pre-firmado: NULL-A sigue viva
   como "la estrategia" hasta el veredicto forward (WITHDRAWAL-PROTOCOL corte A/B).
2. **NULL-B tiene MEJOR Calmar que v11 (7.67 vs 4.19)** — evidencia directa del hallazgo II-1:
   buena parte del retorno 2025 es beta corta con menos exposición, no timing.
3. **DECOMP:** el spot hold-to-Friday siempre-short dio +24.9% ann; la mecánica TP/HS le RESTÓ
   −18.9pp acumulados al short incondicional (los stops cortan ganadores en año tendencial) —
   el valor de v11, si existe, vive en la SELECCIÓN de semanas, no en las salidas.
4. **STRESS-2122:** siempre-short + TP/HS por la depreciación 2021-22 = **−3.1% acumulado, MaxDD −8.0%**
   — la mecánica protegió el corto mucho mejor que el escenario temido (−15/30%). El riesgo
   estructural existe pero está acotado por los stops.
5. Carry: no medible sin swaps del broker → **H-COP-CARRY-00 sigue siendo el experimento #1**.

Trials añadidos al conteo: +4 (NULL-A, NULL-B, DECOMP, STRESS-2122).


## 6. Resultados OLA 7 (2026-07-06, `scripts/analysis/portfolio_layer.py`)

**P2 portafolio equal-risk** (cop_v11 + gold_ens + btc_b2, |ρ|máx=0.08, breaker DD 12/18%):
mix ann=+15.4%, MaxDD −4.3%, Calmar 3.61 vs mejor sleeve (cop_v11) 4.19.
**H-PORT-01:** ΔCalmar=−0.94, IC95=[−17.0, +11.9] incluye 0 ⇒ la diversificación NO se puede
probar con N=52 semanas (tampoco refutar). Se re-evalúa cuando existan ≥2 sleeves promovidos
limpios con historia forward. +1 trial.

**H-LATAM-02 (TSMOM 4/8/13w {COP,MXN,BRL}; CLP excluido — sin seed, exclusión declarada):**
COP −2.1% · MXN −3.5% · BRL −1.6% · basket −2.1% ann vs B1′ +1.3%. ΔCalmar=−0.15,
IC95=[−0.75, +0.27]. **La prima de tendencia NO existe en LATAM FX semanal** (consistente con
el régimen mean-reverting Hurst 0.28-0.49) ⇒ NO se construye LATAM-XS-TSMOM. El almuerzo
gratis de amplitud, si existe, está en el CARRY — que sigue gated en **H-COP-CARRY-00**
(medición del broker, 0 compute, acción del operador). +1 trial.

---

## Pooling LATAM (H-LATAM-02) — BLOQUEADO POR DATOS, no refutado

**Medido 2026-07-21.** El pooling era el único remedio que la librería de skills avala para el
problema de muestra de COP (31 trades, 52 observaciones semanales:
`performance-metrics/SKILL.md:169` — "los ratios con menos de 36 observaciones mensuales son
estadísticamente poco fiables").

Resultado de `portfolio_layer::latam_tsmom()` (TSMOM 4/8/13w):

| Serie | ann% | MaxDD% | Calmar |
|---|---|---|---|
| COP | −1.36 | −27.33 | −0.05 |
| MXN | −7.01 | −2.76 | −2.541 |
| BRL | −5.69 | −2.47 | −2.303 |
| basket | −2.01 | −27.33 | −0.074 |
| basket B1′ | +1.11 | −22.95 | +0.048 |

La cesta pierde contra su propio B1′. Pero **este test no concluye nada**, y la razón es la
cobertura de los seeds:

| Seed | Rango | Días |
|---|---|---|
| `usdcop_m5` | 2020-01-02 → 2026-07-17 | 2.388 |
| `usdmxn_m5` | **2026-03-16** → 2026-07-17 | **122** |
| `usdbrl_m5` | **2026-03-16** → 2026-07-17 | **123** |

MXN y BRL tienen **~17 semanas**. Una cesta de 6,5 años de COP con 4 meses de MXN/BRL no
triplica la muestra efectiva: la contamina. Los Calmar de −2.5 son el artefacto esperado de
anualizar 17 observaciones, no una medición.

**Estado**: `BLOCKED_DATA`. La hipótesis no está refutada — no ha podido probarse. Desbloquearla
requiere backfill histórico de MXN y BRL (el extractor existe; el seed no tiene la historia).
CLP queda fuera por falta de seed, exclusión ya declarada.

**No cuenta como trial**: no se evaluó ninguna hipótesis, se descubrió que no era evaluable.


---

## H-VOLF-01 — ¿HAR-RV bate a la persistencia prediciendo volatilidad a 5d?

**Registrada 2026-07-21, ANTES de implementar el harness.** Contexto: la dirección está
cerrada (mejor celda modelo×horizonte p_adj=1.0 sobre 63 celdas); el re-propósito sancionado
del forecasting es predecir VOLATILIDAD, que alimenta el sizing — la única palanca con edge
demostrado (2026: las estrategias pierden menos que sus subyacentes).

El bar honesto NO es "¿predice algo?" — la vol es fuertemente autocorrelada y cualquier cosa
"predice" — sino "¿bate a la persistencia?" (sigma_hat_{t+5} = sigma_t). Si no la bate, la
persistencia ES el estimador y esta vía se cierra (resultado aceptable).

- **H0**: QLIKE_OOS(HAR-RV) >= QLIKE_OOS(persistencia EWMA lambda=0.94) en h=5d.
- **H1**: HAR-RV (Corsi: regresión lineal sobre RV diaria/semanal/mensual, 3+1 coeficientes,
  walk-forward expansivo con refit mensual, entrenado <=2024) mejora el QLIKE OOS-2025.
- **Estadístico**: QLIKE medio sobre OOS-2025 + block bootstrap pareado (bloque 20) del
  diferencial de pérdidas; IC95 debe excluir cero.
- **Baselines en la misma tabla, siempre**: persistencia RV-20d y persistencia EWMA(0.94).
- **Presupuesto cerrado: 1 modelo × 1 horizonte (5d) × este activo = 1 trial.** No se corre
  el zoo de 9 modelos sobre vol: serían 63 trials y es el mismo error direccional con otro
  target. No se barre lambda ni las ventanas HAR (1/5/22 son el estándar de Corsi, prior).
- **Expectativa honesta ex-ante**: H-VOL-01 (EWMA en el sizer) ya falló NO_RECHAZA. La
  persistencia puede ganar aquí también.
- **Sin test económico salvo que H0 se rechace** — sin QLIKE ganado no hay trial de sizing
  que pagar.

**Coste en trials: +1.**

### Resultado H-VOLF-01 (2026-07-21) — **NO_RECHAZA H0**

QLIKE OOS-2025: HAR 0.3104 vs EWMA 0.3783 (mejor en media), pero IC95 del diferencial
[−0.196, +0.014] **incluye cero**. La mejora no es distinguible de ruido con un año de datos.
La persistencia sigue siendo el estimador. Sin test económico (criterio pre-firmado).

### Resultado H-COP-XLEAD-01 (2026-07-21) — **NO_RECHAZA H0**

Diseño pre-firmado: Ridge+BR del track, walk-forward semanal expansivo, OOS-2025 (51 lunes),
UNA variable = `include_xlead` (features `usdmxn_ret_1d_lag`, `usdclp_ret_1d_lag`, t−1 causal,
historia 2020→ del MACRO_DAILY_CLEAN).

| | DA OOS-2025 |
|---|---|
| sin líderes | 0.5294 |
| con líderes MXN+CLP | 0.5294 |

Δ = **0.0000** — cero semanas discordantes de 51 (McNemar p=1.0): añadir los líderes no
cambió NI UN signo semanal. Los pares EM líquidos no imprimen información a horizonte 5d que
el set de 25 features no tenga ya.

**Hallazgo colateral que importa más que el veredicto**: `MACRO_DAILY_CLEAN` tiene las series
MXN y CLP **corruptas ×10.000 desde 2026-01-27** (cotización sin separador decimal: MXN 17.36
→ 171.335). El veredicto OOS-2025 NO está contaminado (ventanas de train < 2026), verificado:
std pre-2026 = 0.008 (sano). Pero cualquier consumidor 2026 de esas columnas (página de
análisis, features futuras) está leyendo basura. Guard de regresión añadido
(`test_macro_clean_fx_scale.py`); la corrección de la limpieza es tarea del pipeline macro.

Trial pagado (N=58). Con esto y EXP-DIR-001: la dirección COP queda cerrada también con
líderes cross-asset. Solo quedan vintages PIT (Codex) como fuente de información nueva.


---

## H-CARRY-01 (PLANNED 2026-07-22, sin correr — 0 trials hasta ejecutar)

**La única familia de señal NUEVA que la librería de skills respalda y que este registro
nunca ha probado: carry.** No es dirección-desde-precio (la puerta cerrada con 27+ trials);
es retorno-si-nada-se-mueve (`xasset-alpha-engine`: evidencia cross-asset en 58 mercados de
futuros; `currencies-and-fx`: IRP/forward points).

**Datos que lo desbloquean (verificados hoy, calidad 9.7/10)**: IBR overnight (1.800 obs),
TPM, fed funds/prime diarios en `macro_indicators_daily` CLEAN; funding BTC (ya probado como
freno, jamás como tilt de cartera); vistas wide con `_t1` como única forma legal.

**Diseño pre-firmado (una variable: el tilt; todo lo demás congelado)**:
- Señal: carry_col = (IBR − fedfunds) diario, forma _t1, z-score 252d SIN mirar retornos.
- Aplicación: tilt multiplicativo acotado [0.5×, 1.5×] sobre los pesos ERC del libro
  (nunca una estrategia nueva; los campeones no se tocan; v11 sigue FROZEN).
- Ventanas: DISEÑO ≤2024 (con la historia profunda nueva: IBR/prime hasta donde alcancen),
  UN disparo OOS-2025. Sin grid: el clip [0.5,1.5] y la ventana 252 son priors declarados
  aquí, antes de mirar dato alguno.
- Veredicto: Calmar_libro_con_tilt > Calmar_libro (1.271 sin cash / 1.511 con cash, medidos)
  en OOS-2025 Y DSR trial-aware > 0.95 con N actualizado. Baselines obligatorios: libro sin
  tilt (B1′ natural), tilt aleatorio mismo clip (tonto), costos ×2.
- Consecuencia si pasa: candidata a versión nueva del LIBRO → secuencia completa de
  promoción. Si no pasa: se escribe NO_RECHAZA y carry queda cerrado con 1 trial.

**Rechazado explícitamente en la misma decisión** (0 trials, que quede escrito): re-entrenar
el zoo de forecasting a horizonte 7 semanas. Razones: (1) dirección cerrada en todos los
niveles probados (27 trials, mejor celda p_adj=1.0; XLEAD delta 0.0000; FUND −2.2pp);
(2) 9 modelos × H nuevo = +63 trials contra esa puerta, deflactando todo el programa;
(3) el zoo es superficie de transparencia (DA≈0.46-0.5 publicado como caveat), no fuente de
edge — quant-constitution §1: mirar el OOS para elegir horizonte es el grid que COP ya pagó.
El "multi-horizonte" honesto disponible es de ROBUSTEZ, no de señal: replay de la mecánica
campeona congelada sobre las décadas nuevas de diseño (Oro 1979-2019, COP post-1991-2019)
con pass/fail pre-firmado — registrado como H-ROBUST-DECADES-01 (PLANNED) en cada activo
cuando se decida pagarlo.

---

## ENMIENDA a H-CARRY-01 (2026-07-21, ANTES de cualquier corrida — 0 miradas, 0 trials)

La revisión cruzada Codex↔Claude del plan consolidado encontró dos defectos de diseño en la
entrada PLANNED original, y se corrigen ex-ante (enmendar un diseño no corrido es legítimo;
lo ilegítimo sería enmendar después de mirar):

1. **Scope**: "tilt sobre los pesos ERC del libro" era una categoría imposible — el libro
   diario (portfolio_daily.py) EXCLUYE a COP por reloj (52 vs 252). El carry COP solo puede
   modular (a) el sleeve COP semanal o (b) el overlay semanal libro+COP (que debe existir
   primero — U2 del plan). Jamás los pesos diarios XAU/BTC/SPX.
2. **Juez**: el OOS-2025 ya fue observado repetidamente por este programa; no puede volver a
   llamarse OOS para esta hipótesis. **El juez es el forward posterior al freeze de la
   candidata** (constitución §1: el juez es el período siguiente).
3. **Gate previo endurecido** (U1): ≥20 accruals válidos del swap real del broker y
   pass-through mediano neto ≥50% sin que el IC95 (block bootstrap) quede por debajo de 50%.
   Si falla, H-CARRY-01 se cancela SIN gastar trial.

Sin cambios: una variable (el tilt), clip [0.5,1.5] y ventana z-252 como priors ex-ante,
candidata = `strategy_id` nuevo, v11 intacta, baselines (overlay sin tilt, exposición
emparejada, tilt aleatorio, costos ×1/×2/×3 + swap real). PASS económico: ΔCalmar vs mejor
baseline con IC95>0 y DD dentro de protocolo; claim de alfa además exige DSR>0.95.

---

## EVENTO DE REVISIÓN DE DATOS 2026-07-21 — el backtest 2025 cambia de +26.05% a +13.05%

**No es un trial ni un cambio de estrategia: es la MISMA mecánica congelada medida sobre
datos reparados.** La remediación de calidad (gap-fill dirigido M5 + backfill máximo,
CTR-MKT-CANON-001) revisó el seed diario USD/COP: **+57 días que antes no existían y 31
cierres corregidos (15 de ellos en 2025, deltas hasta $38.8 ≈ 1%)**, verificado por diff
contra el seed en HEAD (LFS).

Re-run del ciclo completo (`train_and_export_smart_simple --phase both`, 2026-07-21):

| Ventana | Antes (datos con huecos) | Ahora (datos reparados) |
|---|---|---|
| Backtest OOS-2025 | +26.05%, Sharpe 3.84, p=0.004, 31 trades | **+13.05%, Sharpe 1.506, p=0.1151, 34 trades (2L/32S), WR 73.5%, PF 1.562** |
| Producción 2026 YTD (entrenado 2020-2025) | −0.30%, 5 trades | **+2.45%, 11 trades (10L/1S)** — N<20 ⇒ solo conteo y PnL |

**Lecturas obligadas:**
1. p=0.1151 ⇒ el 2025 reparado es **NO estadísticamente significativo** — coherente con el
   veredicto DSR<0.95 que ya congeló a v11; ahora ni el titular sobrevive a los datos limpios.
   La mitad del "+26%" era artefacto de 57 días ausentes y 31 cierres malos.
2. El forward 2026 (+2.45% YTD con el gate bloqueando la mayoría de semanas mean-reverting)
   sigue siendo el único juez; Corte A 2026-09-16 intacto.
3. Los bundles publicados con los números viejos quedan como historia generada sobre el
   dato viejo; los nuevos exports (2026-07-21) son la referencia. Cualquier cita futura del
   "+25.6%/p=0.006" debe llevar este asterisco.
4. Fix de código en el mismo run: `MIN_TRADES_FOR_STATS` promovida a módulo (NameError que
   abortaba el export de trades — el ciclo backtest→production ahora ejecuta completo).


---

## DIAGNÓSTICO DE INGENIERÍA INVERSA DEL P&L 2026-07-21 (+1 trial: la celda cap-1.5)

Descomposición del OOS-2025 REPARADO (+13.05%, 34 trades) sobre trades publicados —
mediciones descriptivas (0 trials) salvo donde se indica:

| Componente | Números | Lectura |
|---|---|---|
| take_profit | 19 trades, **+30.26pp** | TODO el retorno vive aquí |
| hard_stop | 5 trades, **−17.50pp** (todos −3.5% exacto) | Se come el 58% de los TP. **Los 5 HS tenían leverage = 2.0 (el máximo), vs 1.70 en TP y 1.24 en week_end**: el vol-targeter estaba en máxima agresión exactamente en las semanas que gapearon — confirmación literal de em-fx §2 ("la vol realizada se mide en la calma que precede al gap; el peso vol-targeted es un TECHO") |
| week_end | 10 trades (29%), **+0.085pp ≈ 0** | Opcionalidad que expiró sin valor; con carry cobrado serían positivos |
| CARRY no cobrado | IBR 8.80% − FFR≈4.37% = 4.43pp; 103 días-posición cortos × lev 1.67 → **+2.01pp/año teórico (+1.0 al 50% pass-through)** | Retorno sin riesgo de señal nuevo; gate = H-COP-CARRY-00 |
| Timing intra-sesión (M5) | media +0.7bps vs open, agregado ±1pp ruido | **NO es palanca** — descartado con números, coherente con RL p=0.272 |

**Trial pagado (N=59): sensibilidad ÚNICA pre-declarada cap-leverage-1.5** (prior ex-ante:
em-fx "ceiling" + el MAX_LEV=1.5 que ya usa el manifest SPX; NO grid, una celda):
retorno +12.85→+11.51pp, maxDD −7.74%→−5.21%, **Calmar-proxy 1.66→2.21 (+33%)**.
Combinado con carry al 50%: retorno ≈ igual (+12.5pp) con un tercio menos de drawdown.

**Candidata v12 registrada (PLANNED, sin correr nada más):** `smart_simple_v12_lev_cap`
= v11 con MAX_LEVERAGE 2.0→1.5, UNA variable, juez = **forward desde su freeze** (2025 ya
está doblemente contaminado). v11 NO se toca (Corte A 2026-09-16 sigue siendo su juez).
La decisión de abrir v12 es del operador; correrla en paper paralelo a v11 no consume
más trials hasta abrir resultados forward.

---

## VEREDICTO FUSIONADO tras revisión adversarial de Codex (2026-07-21, 0 trials nuevos)

Codex ejecutó R1-R5 (`codex exec -p audit`, informe en
`.claude/codex/PNL-ADVERSARIAL-REVIEW-USDCOP-2026-07-21.md`); Claude verificó la
hipergeométrica de forma independiente (coincide al 4º decimal). Ajustes aceptados:

1. **R1 — El patrón HS se REBAJA de "confirmación" a "consistente/sugestivo"**:
   19/34 trades tenían lev 2.0; P(5/5 HS en lev-max | azar) = 4.1789% (Fisher bilateral
   5.26%). La candidata v12 se sostiene por el prior em-fx (techo de cola), no por esta
   estadística de N=5.
2. **R2 — El veredicto del cap SOBREVIVE a la composición**: +13.045→+11.810%,
   maxDD −7.74→−5.21%, ret/|DD| 1.686→2.268. Caveat aceptado: es replay con salidas
   fijas; el motor real recalcula TP/HS con el leverage → v12 exige corrida de motor,
   prospectiva.
3. **R3 — Banda honesta del carry**: bruto +1.95 a +2.83 pp/año; al 50% de pass-through
   +0.97 a +1.42 pp; **banda de estrés −0.62 a +2.83 pp con signo neto NO identificado**
   — el swap real depende de tom-next/forward implícito + basis + fee del broker, no de
   IBR−FFR. Refuerza que H-COP-CARRY-00 (statements reales) es EL gate; el diferencial
   de tasas era techo, no estimado.
4. **R4 — week_end confirmado como ruido**: 6 ganadores +2.73pp / 4 perdedores −2.65pp;
   menor lev, menor Hurst, régimen indeterminado. Ninguna regla nueva.
5. **R5 — Dos bloqueadores de ingeniería para correr v12 en paralelo** (nuevos, reales):
   (a) las tablas H5 no tienen `strategy_id` (unicidad por fecha — dos estrategias
   colisionarían), (b) el pipeline sobreescribe artefactos raíz de v11. Arranque
   prospectivo tentativo de v12: 2026-07-27, tras resolver ambos.

N sigue en 59; 2025 siguió tratado como contaminado; v11 FROZEN.

---

## FASES A+B DEL PLAN CONSOLIDADO — ejecutadas 2026-07-21 (0 trials nuevos; N=59)

**Fase A (ingeniería)**: A1 migración 064 (`strategy_id` en signals/executions/paper,
unicidad `(fecha, strategy_id)`, colisión v11/v12 probada resuelta) · A2 artefactos raíz
solo-campeona (`_is_champion`, manifest re-freeze v3 hash c2ddb5eb54d88400) · A3 test PIT
fila-a-fila de features macro (`test_cop_features_pit.py`, T-1 verificado, 2/2 verde) ·
A4 ledger ya cubría COP semanal con supresión N<16.

**Fase B (matemáticas, artefactos en `.claude/evidence/cop_{plan_b,v12_design}/2026-07-21/`)**:

- **B1 — Motor real pareado (misma celda 1.5 pagada; 25 features, camino exacto de main)**:

| Año | v11 (cap 2.0) | v12 (cap 1.5) | Delta |
|---|---|---|---|
| 2022 | −3.78% / DD 5.88 | −3.74% / DD 5.87 | ≈0 (cap no muerde) |
| 2023 | −1.14% / DD 7.68 | **+1.91% / DD 4.96** | v12 mejor en TODO |
| 2024 | +7.68% / DD 6.88 | **+14.23% / DD 3.50** | v12 casi 2× retorno, ½ DD |
| 2025* | +13.05% / DD 7.74 | +9.22% / DD 5.19 | cede retorno, Calmar ≈ igual |

  *2025 = re-medición de la misma celda con mejor instrumento, no trial nuevo. Compuesto
  2022-2024 (años que la elección del cap JAMÁS miró): v11 +2.4% vs **v12 +12.1%**.
  Mecanismo (el que Codex predijo en R2): menos leverage → el effective-HS salta menos →
  más trades sobreviven hasta el TP. El motor confirma que el replay SUBESTIMABA a v12.
- **B2 — Potencia del forward (bootstrap PAREADO por trade)**: P(Calmar v12>v11) = 97.6%
  a 26 semanas, 99.8% a 52. (La v1 del cálculo escalaba uniforme y daba 50% — corregida.)
- **B3 — Gap de lunes (M5 reparado, 335 lunes)**: |gap| p50 16.7 / p90 73 / p99 157 bps;
  peor −163/+247. A lev 2.0 el p99 ≈ 3.1% de equity — el riesgo que el HS no limita;
  a cap 1.5 ≈ 2.3%.
- **B4 — Sharpe honesto (Lo-Mertens, skew −1.29)**: 2025 = 1.48, IC95 [−1.27, +4.24]
  — **INCLUYE CERO**. Ningún titular de Sharpe COP debe publicarse sin este IC.

**Spec v12 respaldado por diseño**: la evidencia de v12 ya no es solo la celda 2025
contaminada — es el delta pareado en 3 años de diseño que la elección del cap nunca vio.
Arranque paper 2026-07-27 (A1/A2 listos). Juez: forward.

---

## EVENTO DE RE-MEDICIÓN #2 (2026-07-21): FUGA DE PURGA corregida — el 2025 honesto es +7.66%

**BUG A-1 de la auditoría ML (verificado en código antes de aceptar)**: las últimas ~5 filas
de cada ventana de train llevaban el label de la semana que se iba a operar (target 5d
construido sobre el frame completo + filtro solo-NaN global), y la fila de predicción
estaba DENTRO del fit con su propio desenlace como label. Producción nunca vio esos labels
(NaN el domingo) → el backtest usaba una metodología que el modelo servido no tiene.

**Fix**: purga de 7 días calendario (~5 hábiles) en el fit; features de predicción tomadas
aparte del frame sin purga (exactamente lo que ve el DAG). Re-medición (no trial):

| Ventana | con fuga | **purgado (honesto)** |
|---|---|---|
| OOS-2025 | +13.05%, p=0.115, 34 tr | **+7.66%, p=0.2154, 32 tr, Sharpe 0.987** |
| 2026 YTD | +2.45% | **+3.36%, 11 tr** (el año limpio SUBE) |

Cascada completa de honestidad del 2025: +26.05% (datos rotos + fuga) → +13.05%
(reparados + fuga) → **+7.66% (reparados + purgados)**. Sigue batiendo al B&H (−14.5%)
por 22pp, pero cada capa de rigor recortó el backtest a la mitad. El 2026 — el único
juez limpio — MEJORÓ con el fix. BUG A-2 (circuit breaker sin salida) corregido con
cooldown de 4 semanas + re-anclaje (no muerde en 2025/2026: números idénticos).

## ENMIENDA EX-ANTE al juez de v12 (auditoría estadística, ANTES del arranque 2026-07-27)

El diseño original ("ΔCalmar IC95 excluye 0 a 26 semanas") estaba mal especificado en tres
puntos, medidos por el auditor y verificables en `.claude/evidence/`:

1. **Potencia ≈ 3.4%** del criterio literal (el maxDD con N=26 es ruido) — el Corte A
   habría salido INCONCLUSO casi seguro aunque v12 fuera genuinamente mejor.
2. **Reloj en unidad equivocada**: la información del cap solo llega en trades con
   lev>1.5 — y en 2026 YTD el cap NO ha mordido ni una vez (lev máx 1.27, 0/29 semanas).
   26 semanas calendario en régimen 2026-like = 0 observaciones informativas.
3. **El titular "P=97.6% a 26 sem" era condicional al régimen 2025**: banda honesta por
   mezcla de regímenes = [0.50 – 0.98] (replicado con block bootstrap b=4/b=8: robusto a
   autocorrelación, frágil a régimen).

**Juez enmendado (pre-firmado, 0 miradas)**:
- **Reloj**: N_bind ≥ 12 trades donde el cap muerde (lev_v11 > 1.5), no semanas.
  Si a las 52 semanas N_bind < 12 → INCONCLUSO → extensión automática (mismo mecanismo
  del WITHDRAWAL-PROTOCOL §4). Mínimo calendario: 26 semanas se mantiene.
- **Estadístico**: block bootstrap circular (b=4) de la serie pareada semanal sobre
  Δ(ret_ann − λ·|maxDD|) con λ=1 fijado aquí. NO sign test (los deltas son negativos en
  semanas TP por construcción; castigaría a v12 aunque su Calmar fuera mejor).
- **Monitoreo semanal sin gastar alfa**: e-process (betting martingale) sobre los deltas
  clipeados a ±10%; PASS anticipado si e-value ≥ 20 (α=0.05 anytime-valid). El operador
  puede mirar TODAS las semanas — el control de error no depende de cuándo mire.
- La hipergeométrica 4.18% del patrón HS queda re-clasificada como DESCRIPTIVA (el null
  "lev ⊥ HS" es mecánicamente falso: el buffer de precio es inverso al leverage). La
  evidencia real de v12 es el delta pareado 2022-2024 en años que la elección jamás miró.

**Deudas registradas por la misma auditoría**: la tabla DSR §2 está stale (usa el Sharpe
pre-reparación 0.3585 semanal; con 0.209 el DSR base cae <0.50) → recomputar con σ_trials
REAL re-corriendo las 42 celdas pagadas sobre datos reparados (0 trials, COP-NULL OLA 4);
añadir `circular_block_bootstrap()` y `e_process()` a `services/common/metrics.py`;
persistir script generador junto a cada JSON de evidencia.

---

## AUDITORÍA MATEMÁTICA FINANCIERA (2026-07-21) — la aritmética replica; el REPORTE tenía 3 sesgos optimistas

Recálculos independientes: composición 2025/2022-24/2026, maxDD, PF, p-value — **todos
replican al 4º decimal** ✓. Los hallazgos son de matemática financiera, no de aritmética:

1. **Sharpe 1.506 está inflado por construcción**: es solo-trades (34 semanas), sin
   risk-free. Con las 18 semanas flat = 1.21; en exceso de rf USD 4.37% = **0.80**.
   El skill performance-metrics exige (Rp−Rf)/σ. Todo titular futuro publica el de exceso.
2. **El fill del HS es una orden imposible**: el yaml declara `hard_stop: "limit"` — un
   buy-limit al nivel del stop NO se ejecuta en gap-through. Medido 2025: trade 7 abrió
   14bps más allá del stop → corrección −0.28pp; con slippage intradía realista
   −0.3/−1.0pp/año y DD mayor. Favorece estructuralmente a v12 (menos HS). El yaml debe
   decir stop-market y el motor llenar open-aware (re-medición, 0 trials).
3. **El forward 2026 está inflado por carry omitido**: 10L/1S pagan diferencial —
   recalculado −0.24pp (el +3.36% purgado sería ≈+3.12% con pass-through completo).
   El único juez limpio tiene sesgo pro-LONG hasta que el motor devengue carry
   bidireccional (gate: H-COP-CARRY-00).
4. Carry teórico corregido: **+1.83/+1.91pp** (no +2.01: IBR es E.A. vs FFR nominal
   act/360 → diferencial 4.27pp; 98 noches, no 103 días).
5. **La palanca más grande del track no es de señal: colateral ocioso = +4.3-4.5pp/año**
   (18 semanas 100% flat + margen no usado). Es elección de venue/broker declarable
   ex-ante (0 trials). En venue sin interés (MEXC/USDT) ≈ 0 → entonces el Sharpe DEBE
   reportarse en exceso de cash (≈0.8).
6. Kelly guard: riesgo actual = 0.23×Kelly (v12 ≈ 0.17×) — no hay upside legal en subir
   tamaño; el debate 2.0 vs 1.5 vive dentro de la zona fraccional segura.
7. Calmar a cierres diarios = 1.69 ✓ defendible; el multi-año honesto (v12 2022-25:
   CAGR 5.2%, Calmar ≤0.88) es el número comparable con CTAs. vt_min=0.5 anula los
   multiplicadores de de-risking (candidato a spec v12 o +1 trial).

---

## PANEL CODEX (4º auditor, 2026-07-21) — integración final; el juez de v12 queda SELLADO

Codex confirmó independientemente la fuga de purga y el estado NO-PROMOTE, y corrigió el
diseño del juez donde aún estaba flojo. **Especificación FINAL del juez de v12 (sella y
reemplaza las versiones anteriores; 0 miradas):**

1. **Reloj propio desde SU freeze**: arranque paper 2026-07-27 → corte-26 ≈ **2027-01-22**
   y corte-52 ≈ **2027-07-23**. El 2026-09-16 es EXCLUSIVAMENTE el Corte A de v11.
2. **Semana a semana = solo integridad** (costos, exposición, missingness, safety).
   Ninguna decisión de superioridad semanal (ilustración del panel: 38 miradas al 5% =
   FWER 85.8%).
3. **Corte-26 = safety/futility únicamente** (no promoción). **Corte-52 = ÚNICO test
   confirmatorio**: ΔCalmar neto compuesto v12−v11, unilateral α=0.05, bootstrap circular
   pareado b=4 pre-firmado. Si se exigiera promoción temprana: exactamente dos miradas,
   α26=0.01 / α52=0.04 — ninguna más.
4. **Gates económicos conjuntos**: retorno v12>v11 Y DD_v12≤DD_v11 Y DD absoluto<12%.
5. **Intention-to-treat**: flat deliberado = retorno 0; dato faltante = NA (jamás 0).
   Ledger append-only con strategy_id, semana ISO, config/code/data hash.
6. **N_bind ≥ 12 se mantiene como criterio de INFORMACIÓN**: si el cap casi no muerde,
   INCONCLUSO — ni derrota ni victoria. N<20 trades ⇒ solo conteo y PnL.
7. **El e-process queda DEGRADADO a monitoreo descriptivo** hasta que exista una
   implementación congelada (null condicional, transformación, betting schedule,
   dependencia). No habilita PASS. (Corrección justa de Codex a la enmienda previa.)

**Violación operativa encontrada por el panel y corregida en el mismo commit**: el monitor
L6 publicaba `running_sharpe` desde n=4 y una "GATE DECISION" semanal desde la semana 15
(promote/discard/switch por DA) — solo informativa (la promoción real es doble-voto), pero
presionaba decisiones fuera de los cortes. Fix: Sharpe suprimido hasta N≥20 y el log
re-etiquetado como descriptivo.

**Veredicto consolidado de los 4 auditores**: NO PROMOTE; no abrir variantes nuevas;
v11 congelada rumbo a su Corte A con los números honestos (+7.66% OOS-2025 purgado,
+3.36% YTD-2026); v12 al forward con este juez sellado; las palancas de rentabilidad
inmediata son de infraestructura (colateral ocioso +4.3pp, carry si el swap pasa el gate).
