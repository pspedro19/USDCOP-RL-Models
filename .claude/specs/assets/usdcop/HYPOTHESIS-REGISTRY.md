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
n_trials_total: 73
n_trials_scenarios: [46, 58, 72]   # conservador / central / amplio — se publican los tres
n_trials_sources:
  - "EXPERIMENT_LOG.md: FC-H5-SIMPLE-001 + FC-SIZE-001 (reconstrucción retroactiva v1.0→v11)"
  - ".claude/specs/assets/usdcop/EXP-DIR-001-directional-trials.md (27 trials direccionales)"
  - "public/data/strategies/{smart_simple_v11,smart_simple_aggr}/backtests/* (5 bundles = suelo)"
sigma_trials: 0.0473   # MEDIDA 2026-07-21 (42 celdas re-sim, motor purgado; N_eff=10 clusters)                 # nunca se persistió la dispersión de Sharpe entre trials
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

---

## VEREDICTO DE PANEL 4/4 (2026-07-21): ¿más features macro/técnicas? — NO direccional, SÍ a riesgo/carry/N

Cuatro auditores independientes (PhD fin-math Claude, ML Claude, estadística Claude,
Codex exec-audit) convergieron con cálculos separados. Números triangulados:

- **Techo de detectabilidad**: MDE a N=300 semanas = DA≥57.2% / IC≥0.143 (con maxT
  familia-6: IC≥0.148-0.20) vs IC plausible de macro semanal 0.02-0.05. Potencia real
  8-22%. Verificación de IC=0.05: **47.6-49.5 años** (dos cálculos independientes).
- **El ruido gana**: E[max IC espurio] con 6 candidatas ≈ 0.11 — 2.5× el efecto buscado.
  "Si una feature sobrevive el screen, la explicación más probable es leakage."
- **Fundamental Law**: con BR efectivo 34 y TC~0.6, mover el Calmar exige IC≥0.086 —
  fuera del rango institucional para un solo par EM de sesión 5h.
- **Valor de +1pt DA ≈ +1.9pp/año** — y es incertificable (297 años para confirmar +1pt).

**Ranking unánime del dinero**: (1) colateral remunerado +4.3-4.5pp [0 trials, venue];
(2) carry ejecutable +1.0-1.9pp [gate H-COP-CARRY-00, statements]; (3) ejecución honesta
(stop-market open-aware, carry bidireccional, costos ×2/×3 con dientes); (4) forward
congelado v11/v12 sin reutilizar 2025; (5) **comprar N, no columnas** (historia mensual
larga PIT — forwards BanRep desde 1997, remesas desde 2000 — y eventual panel LatAm);
(6) al final, UN experimento pre-registrado de carry/flujo como tilt/gate.

**Tabla de candidatas (ML, con acuerdo Codex)**: VIVAS = forwards BanRep + TES-extranjeros
(cantidad/flujo, información nueva; condición dura: vintage PIT; Codex precisa que los
forwards BanRep son MENSUALES desde 1997 → el estudio legal es mensual de carry/curva,
no fingir 300 semanas). MUERTAS para dirección = Ecopetrol (≈WTI×COLCAP, canales
refutados K3/K4), carbón (3er intento del canal ToT), remesas (~60 datos únicos, contenido
de sorpresa ≈0 a 5d), carry real (canal ya cubierto por H-CARRY-01; infexp-EOF mejora el
calificador suspect/genuine del gate, 0 trials), stress LatAm (muerta para signo, VIVA
para magnitud/riesgo).

**La puerta ABIERTA con prior a favor — familia de RIESGO (~5 trials, screening ≤2024,
bar = batir persistencia en pinball q90, el listón de H-VOLF-01)**: calendario
BanRep/DANE/FOMC (la persistencia es ciega al calendario POR CONSTRUCCIÓN; 0 datos
nuevos), vol-of-vol, cola de gap-lunes condicionada (B3), EMBI re-propósito a cola,
RESINT (riesgo de intervención). Vol se predice con IC 0.3-0.5 → potencia ~100% al N
actual. Contabilidad por celda (precedente H-CARRY-02 "+1 (+6 sensibilidad)").

GBM/no-lineal y multi-task: muertos a este N (DIR-001 H3 α=−0.096; pooling BLOCKED_DATA).
N=59 sin cambio — el panel entero fue análisis de diseño, cero variantes miradas.

---

## H-RISK-FAM-01 (PRE-REGISTRO 2026-07-21 — familia COMPLETA, 0 miradas hasta hoy)

**La única familia con prior estadístico a favor según el panel 4/4** (vol/colas se
predicen con IC 0.3-0.5 → potencia ~100% al N actual; mecanismo documentado: 5/5 HS a
lev-max; la persistencia es ciega al calendario por construcción).

**Familia CERRADA (estas 5 y ninguna más; añadir después = nueva familia):**
| Celda | Feature (forma exacta, todo t−1) | Prior económico |
|---|---|---|
| F1 | `event_week` = dummy {decisión BanRep, IPC DANE, FOMC} en la semana operada (calendarios ex-ante del SSOT `schedule.publication`) | eventos programados ⇒ P(gap)↑ sin decir signo |
| F2 | `vol_of_vol_20` = std20 de Δvol20 diaria | fragilidad que el nivel de vol no ve ("la calma que precede al gap") |
| F3 | `gap_tail_cond` = P̂(|gap lunes|>p90) condicionada a nivel VIX/EMBI (kernel simple, ventana 252) | B3 midió la incondicional (p99=157bps); condicionar es el paso |
| F4 | `embi_accel` = Δ5d del EMBI z-scoreado 252 | precio del riesgo soberano acelerando ⇒ cola |
| F5 | `resint_z` = reservas internacionales z-12m (mensual, lag publicación real) | riesgo de intervención BanRep |

**Protocolo (sellado):** target = {gap_week: |gap lunes|>p90} y {q90 del rango semanal}.
Screening SOLO ≤2024-12-31, purged K-fold K=5 (purga 5bd + embargo = lookback máx de la
celda). Métrica = Brier (gap_week) y pinball q90 (rango) **vs el baseline de persistencia
EWMA** — el listón de H-VOLF-01. Se publica la tabla ENTERA (las 5 celdas, ganen o
pierdan). Contabilidad: **+5 trials al abrir resultados** (por celda, precedente
H-CARRY-02); hoy N sigue en 59 — registrar no cuesta.
**Consecuencia pre-firmada**: si ≥1 celda bate a la persistencia en su scoring con IC95
block-bootstrap (b=4) que excluye 0 → UN trial económico adicional: sizing modulado
P(gap) (candidata v13, techo probabilístico), juez = forward desde su freeze. Si ninguna
gana → familia CERRADA por escrito, el cap bruto de v12 queda como la última palabra en
sizing. Nada de esto toca v11 ni v12.

---

## RE-MEDICIÓN #3 (2026-07-21, Puerta 1.4): fills de HS open-aware — 2025 = +7.35%

El yaml pedía `hard_stop: "limit"` (orden que no se ejecuta en gap-through); el motor
llenaba al nivel exacto — un precio que en gap no existió. Fix: stop-market open-aware
(gap-through → fill al open). Re-medición: **2025: +7.66→+7.35% (−0.31pp; el auditor
fin-math había medido −0.28pp a mano — validación cruzada), p=0.2277, Sharpe 0.942**;
2026 sin cambio (+3.36%). Serie oficial de honestidad del OOS-2025:
+26.05 → +13.05 → +7.66 → **+7.35** (datos, purga, fills). Cada número anterior queda
como historia de su capa de rigor. Manifest v5. Herramientas añadidas al SSOT de
métricas: `circular_block_bootstrap()` y `e_process()` (descriptivo, no habilita PASS).


---

## RESULTADO H-RISK-FAM-01 (2026-07-21) — 4/5 celdas GANAN en pinball q90 · trials 59→64

Screening ejecutado por el protocolo pre-registrado (210 semanas de diseño 2020-12→2024-12,
purged K-fold, generador persistido en `.claude/evidence/cop_risk_family/2026-07-21/`).
**Corrección contra nosotros mismos durante la corrida**: el baseline EWMA pre-firmado
(1.645σ√5) resultó mal especificado para el target RANGO (es un cuantil de retorno →
subestima → todo gana trivial). Se sustituyó por el null que aísla la feature:
**intercepto-solo en el mismo CV**. Con el null justo:

| Celda | Brier vs base | Pinball q90 vs null 0.314 | WIN |
|---|---|---|---|
| f1_event (calendario aprox.) | 0.0672 / 0.0674 | 0.317 | NO (la aproximación de fechas lo debilitó, como se declaró) |
| f2_volofvol | 0.0667 | **0.280** | **SÍ** |
| f3_gaptail (cond. VIX) | 0.0667 | **0.278** | **SÍ** |
| f4_embiacc | 0.0689 | **0.293** | **SÍ** |
| f5_resintz | 0.0668 | **0.261** | **SÍ** |

Target binario (gap_week, tasa base 7.1%): NINGUNA celda bate la frecuencia base — el
evento es demasiado raro para 210 obs. La información viva está en el CUANTIL del rango,
exactamente donde el panel predijo (vol IC 0.3-0.5).

**Consecuencia pre-firmada ACTIVADA**: 1 trial económico disponible — sizing modulado por
q90 condicional (candidata v13, techo de leverage probabilístico), juez = forward desde su
freeze. NO se corre hoy: v12 primero (su forward arranca el lunes), v13 después si v12
gradúa o en paralelo si el operador lo decide.

## σ_TRIALS MEDIDA (COP-NULL OLA 4, deuda saldada)

42 celdas re-simuladas con el motor purgado sobre datos reparados: **σ = 0.0473 semanal**
(el grid asumido [0.05,0.10,0.15] era razonable; el "conservador" 0.15 era 3× la realidad).
N_eff por clusters de correlación >0.95: **10 de 42**. **DSR de v11 con σ medida: 0.72 en
los TRES escenarios (N=59/10/27)** — robusto e insensible al conteo. Veredicto sin cambio:
v11 no pasa 0.95, freeze intacto — pero la tabla ya no es una asunción, es una medición.
Artefactos + generadores: `.claude/evidence/cop_sigma_trials/2026-07-21/`.


---

## RESULTADO H-LATAM-02 con historia profunda (2026-07-21) — NO_RECHAZA · trials 64→65

La hipótesis desbloqueada por el backfill se corrió con la mecánica EXACTA pre-registrada
(votos TSMOM 4/8/13w /3, shift(1), {COP,MXN,BRL}, sin vol-target/carry/CLP) sobre 35 años
(COP 1989→, MXN 1990→, BRL 1994→). Generador persistido en
`.claude/evidence/cop_latam_deep/2026-07-21/`.

**DISEÑO ≤2024** (1.833 semanas): cesta ann 5.9% / Calmar 0.277 vs B1′ 0.173. Pero la
tabla por décadas es el hallazgo:

| Década | Calmar cesta |
|---|---|
| 1990s | 0.671 |
| 2000s | **1.119** |
| 2010s | 0.184 |
| 2020s | **0.037** |

**El edge TSMOM en FX latino lleva DOS DÉCADAS muriendo** — decaimiento de alfa clásico y
documentado, no ruido (n=521/década).

**OOS-2025 (un disparo)**: cesta −1.89% (Calmar −0.49) vs B1′ −9.69% (−0.97).
ΔCalmar = +0.47 pero **IC95 block-4 = [−0.58, +3.29] INCLUYE CERO → NO_RECHAZA**.
La cesta amortigua (pierde menos que la exposición pasiva emparejada) pero no se
distingue estadísticamente de ella.

**Consecuencia**: la vía "breadth por TSMOM LATAM ingenuo" queda CERRADA con 1 trial —
conocimiento que ahorra el esfuerzo futuro. Lo que sigue vivo de la familia LATAM:
H-LATAM-01 (la pata de CARRY de la cesta) sigue gated en tasas locales medidas
(TIIE/Selic/TPM-CLP no ingestadas) + swaps reales — la teoría dice que en FX latino el
carry es el retorno y el momentum el ruido, y esta corrida es consistente con eso.
Las rutas activas hacia la meta 20-30% quedan: v12 (lunes), v13 (materia prima validada),
transformer-vol (+1 trial disponible), carry/colateral (operador).

---

## REGLA DE VENTANAS CANÓNICAS DEL PROGRAMA COP (operador, 2026-07-21) + re-reporte alineado de H-LATAM-02

**Regla (vinculante para toda hipótesis futura del programa)**: ventana primaria =
**DISEÑO 2020-01→2024-12 · OOS 2025 (un disparo) · LIVE 2026+**, anclada al inicio del
M5 de USD/COP (2019-12-18). Historia más profunda solo como contexto de robustez,
etiquetada, jamás como juez. Estrategias con macro: universo = intersección con la
cobertura CLEAN real de cada serie. Nada se rellena hacia atrás.

**H-LATAM-02 re-reportada en la ventana alineada** (mismo trial N=65, 0 miradas nuevas —
re-slice del artefacto; `h_latam_02_ALIGNED_window.json`):

| Ventana alineada | Cesta TSMOM | B1′ (pasivo emparejado) |
|---|---|---|
| DISEÑO 2020-2024 (261 sem) | ann +0.79% · maxDD −21.3% · **Calmar 0.037** | ann +4.31% · maxDD −12.0% · **Calmar 0.358** |
| Por par (diseño) | COP −0.09 · MXN 0.047 · BRL 0.093 | — |
| OOS-2025 (52 sem) | −1.89% · Calmar −0.49 | −9.20% · Calmar −0.97 |
| ΔCalmar OOS | +0.476, IC95 block-4 [−0.57, +3.29] | **NO_RECHAZA** |

**Lectura alineada — MÁS dura que la de 35 años**: en la ventana del programa la cesta
ni siquiera bate a su propio baseline pasivo en diseño (0.037 vs 0.358) y las tres patas
están muertas individualmente. El decaimiento por décadas queda como contexto; el
veredicto en la ventana canónica es inequívoco: vía cerrada.

---

## H-VOLT-01 (PRE-REGISTRO 2026-07-21, 0 miradas) — transformer de volatilidad intradía → q90 COP

**Hipótesis**: la FORMA intradía (información que la persistencia rv20 y HAR jamás vieron
— solo consumen cierres diarios) mejora el pronóstico del cuantil q90 del rango semanal
de USD/COP. Prior: el mejor del sistema (vol IC 0.3-0.5, potencia ~100%; panel 4/4).

**Diseño SELLADO (una configuración, cero búsqueda de arquitectura):**
- Tokens: por día de sesión, vector de 8 features intradía del M5 (vol realizada, rango,
  up-vol, down-vol, vol primera hora, vol última hora, |gap apertura|, |ret sesión|).
- Secuencia: 60 días de tokens → encoder transformer 2 capas · 4 cabezas · d=64 ·
  dropout 0.1 (~200k params) → cabeza dual: vol realizada 5d + q90 rango semanal (pinball).
- Pool de entrenamiento multi-símbolo {COP, MXN, BRL, XAU, BTC} con embedding de activo
  (truco de entrenamiento; el DESPLIEGUE y la evaluación son SOLO COP — regla de ventanas
  del operador respetada: todo ≥ 2019-12/2020 según el M5 de cada par).
- Splits temporales pre-declarados: train ≤2022-12 · val 2023 (early-stop únicamente) ·
  **TEST = 2024, un disparo** (dentro del diseño; 2025/2026 NO se tocan — quedan para el
  paso económico futuro si esto gana). Seeds: [42, 123, 456] (3 — DL supervisado).
- **Bar pre-firmado**: batir en TEST-2024 (COP solamente) a TODOS: persistencia rv20,
  EWMA λ=0.94, y la mejor celda simple de H-RISK-FAM-01 (resint/gap-cola) en
  **pinball q90** con IC95 block-bootstrap (b=4) que excluya 0 en ≥2/3 seeds.
- **Consecuencia**: WIN → su salida se convierte en el insumo del techo de v13
  (reemplaza/compone con las 4 features simples; el paso económico es el mismo trial v13
  ya activado, no uno nuevo). LOSE → las features simples de H-RISK-FAM-01 son el
  estimador operativo y el transformer queda cerrado.
- Costo: **+1 trial al abrir TEST-2024** (N 65→66). Generador persistido junto al artefacto.


---

## RESULTADO H-VOLT-01 (2026-07-21) — NO_RECHAZA (0/3 seeds) · trials 65→66

Ejecutado EXACTAMENTE el diseño sellado (9.084 secuencias multi-símbolo, 3.997 train /
1.428 val / 260 TEST-COP-2024; 3 seeds; generador persistido en
`.claude/evidence/cop_vol_transformer/2026-07-21/`).

| Pinball q90 (TEST-2024 COP) | seed 42 | seed 123 | seed 456 |
|---|---|---|---|
| Transformer | 0.1222 | 0.1264 | 0.1361 |
| Persistencia (q90 rolling-252) | **0.1233** | 0.1233 | 0.1233 |
| EWMA 1.645σ | 0.1524 | 0.1524 | 0.1524 |

**Lectura**: el transformer APLASTA a EWMA (−20%) pero NO bate a la persistencia empírica
(mejor seed: empate estadístico, IC95 [−0.009, +0.010] incluye 0; 0/3 seeds ganan).
**La persistencia queda imbatida por TERCERA vez** (HAR en H-VOLF-01, EWMA-sizer en
H-VOLE-01, transformer aquí) — a horizonte semanal, el cuantil empírico rodante del rango
de COP es un estimador que ni 200k parámetros con forma intradía mejoran. Nota de
implementación: el tercer baseline pre-firmado (mejor celda simple de H-RISK-FAM-01) no
alcanzó a evaluarse — irrelevante para el veredicto: fallar un baseline obligatorio ya
es FAIL.

**Consecuencia**: la vía transformer queda CERRADA para vol COP. El insumo operativo del
techo de v13 son las 4 features simples validadas de H-RISK-FAM-01 COMPUESTAS con la
persistencia rolling-q90 (el listón imbatido), no un modelo profundo. El presupuesto de
lo aprendido hoy: 2 hipótesis grandes probadas y cerradas con evidencia (LATAM-TSMOM,
transformer-vol) + 4 predictores simples validados — el sistema sabe más y gasta menos.

---

## H-V13-QRISK-01 (PRE-REGISTRO 2026-07-21, 0 miradas) — techo de leverage probabilístico

**Fórmula SELLADA (cero grid; cada constante con su prior declarado):**
- `q90_hat_t` = 0.5·persistencia(q90 rolling-252 del rango 5d) + 0.5·QR(q90) sobre las
  4 features ganadoras de H-RISK-FAM-01 {vol-of-vol, gap-cola, EMBI-acel, RESINT},
  ajustada walk-forward SOLO con datos < t (composición 50/50: el listón imbatido ancla,
  las features validadas modulan — sin pesos optimizados).
- `techo_t = 1.5 × clip(mediana_diseño(q90_hat) / q90_hat_t, 0.5, 1.0)` — riesgo predicho
  sobre la mediana ⇒ el techo baja proporcionalmente; JAMÁS sube de 1.5 (v13 ⊂ v12).
- Todo lo demás idéntico a v12. Ventanas canónicas del operador (diseño 2020/22-24).
- **Design-run pareado** (v12 vs v13, mismas señales): abre +1 trial (N 66→67).
  Aprobación por diseño = Calmar_v13 ≥ Calmar_v12 en 2022-2024 (replay de salidas con
  effective-HS recalculado por leverage). **Juez real = forward** desde su freeze,
  mismo protocolo sellado de v12 (reloj propio, corte-52).


---

## H-V13-QRISK-01 design-run: INSTRUMENTO INVÁLIDO (2026-07-21) · trials 66→67

El replay de salidas construido para el design-run pareado NO reproduce al motor: dio
v12 = −12.8% en 2022-24 donde el motor real (autoridad, `cop_v12_design/`) dio +12.1%
compuesto — la aproximación effective-HS/producción-wrapper es incorrecta. **Regla
aplicada: un instrumento que no reproduce al motor no puede juzgar nada.** La mirada se
contabiliza igual (+1, conservador), el resultado se marca INVALID_INSTRUMENT y NO se
interpreta ni a favor ni en contra de v13.

**Pendiente**: el design-run válido de v13 exige implementar el techo dinámico DENTRO
del motor (flag de candidata, mismo camino que validó a v12). El spec de v13 sigue
sellado sin cambios. v12 queda formalmente congelada para el lunes:
`config/execution/smart_simple_v12_lev_cap.yaml` + `config/strategy_manifests/usdcop_v12.yaml`
(hash 3bc28b36448892a1).

---

## RE-MEDICIÓN #4 (2026-07-21) — el diseño 2021-2024 con el motor HONESTO es NEGATIVO

Corridas pareadas v11/v12 con el motor actual (purga + fills open-aware) sobre las
ventanas canónicas — los "+12.1% vs +2.4%" del diseño eran del motor PRE-purga y quedan
retractados como evidencia (la fuga inflaba precisamente los años de diseño):

| Año | v11 (cap 2.0) | v12 (cap 1.5) |
|---|---|---|
| 2021 | −2.53% / DD 4.2 | −2.14% / DD 3.8 |
| 2022 | −9.05% / DD 9.4 | −9.22% / DD 9.4 |
| 2023 | −2.54% / DD 6.7 | −2.53% / DD 6.9 |
| 2024 | −4.19% / DD 8.0 | **+4.60% / DD 3.5** |
| **Compuesto 21-24** | **−17.22%** | **−9.43%** |
| 2025 (celda pagada) | +7.35% / DD 7.8 | +7.69% / DD 5.7 |

**Lecturas obligadas:**
1. **La familia smart_simple NO tiene rentabilidad demostrada en NINGÚN período limpio de
   diseño** — toda la rentabilidad histórica vive en 2025, el año sobre el que el grid de
   42 celdas seleccionó. Es la confirmación más fuerte hasta ahora de que el backtest
   2025 es artefacto de selección (coherente con DSR 0.72<0.95 y NULL-B).
2. v12 domina a v11 en riesgo en todos los años (DD menor o igual; 2024 positivo; 2025
   Calmar ~1.36 vs ~0.94) — el cap sigue siendo mejor mecánica — pero "mejor que v11"
   ya no significa "rentable en diseño".
3. **El único período positivo limpio es el forward 2026 (+3.36% YTD)** — el juez real.
   El Corte A (16-sep) y W6 (si NULL-A ≥ v11, la mecánica sin modelo ES la estrategia)
   pasan a ser el evento central del track.
4. v12 va a paper el lunes igual (costo cero, menos riesgo que v11); su caso de diseño
   se re-basa en dominancia de riesgo, no en retorno. Expectativas re-fijadas por escrito.
0 trials nuevos (re-medición de celdas pagadas con mejor instrumento).

---

## ORÁCULO 2025/2026 (descriptivo, 0 trials) + H-TP-LADDER-01 (PLANNED, 0 miradas)

**Techo de la clase de mecánica** (M5 real, entrada solo lunes-cierre como v11, salida
perfecta, lev 1.5, costos 1bp/lado — `.claude/evidence/cop_oracle/2026-07-21/`):

| Año | Oráculo libre | Oráculo clase-v11 | Realizado v11 | Captura |
|---|---|---|---|---|
| 2025 | 179pp | **120pp** | +7.35% | **6.1%** |
| 2026 YTD | 103pp | 50pp | +3.36% | 6.8% |

**Lecturas:**
1. **El 20-30% SÍ existe dentro de la clase de mecánica** (bastaría capturar 17-25% del
   techo vs el 6% actual) — la meta es geométricamente alcanzable sin cambiar la clase
   de entrada. La restricción no es la señal: es CUÁNTO se deja correr al ganador.
2. Los bolsillos grandes (top semanas 3.7-6.0pp disponibles) mueren hoy en el TP fijo
   ~1.6%: el TP captura la cola corta de semanas que ofrecían 3× más.
3. **PROHIBICIÓN explícita** (constitución §1 + evento de hoy): entrenar CUALQUIER modelo
   por prueba-y-error contra estos puntos de entrada/salida 2025 = imitación del oráculo
   = la forma pura del overfitting que hoy vimos colapsar el diseño a −17%. El oráculo
   es mapa de techo, jamás target de entrenamiento.

**H-TP-LADDER-01 (PLANNED, la hipótesis mecánica que el diagnóstico habilita)**: salida
escalonada pre-declarada — 50% del tamaño al TP actual (vol-escalado ~1.6%), 50% restante
al 2×TP o week-end, mismo HS. UNA variable (estructura de salida), prior ex-ante (el
oráculo muestra bolsillos 2-3× el TP; el skill em-fx: dejar correr en EM con stop de
tamaño), diseño en motor 2021-2024 pareado vs v12 + re-medición 2025 de la misma celda,
juez = forward. Costo: +1 trial al abrir. Nota: revisa el DO-NOT histórico "no trailing"
— esto NO es trailing (es ladder fijo); si el operador la aprueba, es la siguiente en cola.

---

## CONTRAFACTUAL DE DECISIONES 30% (2026-07-21, descriptivo 0 trials)

10 políticas simuladas sobre M5 real + señales reales del motor purgado
(`.claude/evidence/cop_counterfactual/2026-07-21/`). Meta: 30% en 2025 y su equivalente
2026-YTD (+16.0%). NINGUNA política de la clase alcanza ambas:

| Política (salida × leverage) | 2025 | DD | 2026 YTD | DD |
|---|---|---|---|---|
| S1 v11 tal cual | +6.75% | −8.0 | +3.24% | −1.5 |
| S3 ladder 2×TP (lev igual) | +8.25% | −7.3 | +3.25% | −1.5 |
| S7 ladder 2×TP, lev ×1.5 (cap 3) | **+20.11%** | −7.0 | +4.88% | −2.3 |
| S10 ladder 2×TP, lev ×2 (cap 4) | **+21.74%** | −11.4 | +6.51% | −3.0 |

**Las decisiones que el 30% de 2025 exigía**: (1) salida escalonada dejando correr la
mitad al 2×TP (+1.5pp por sí sola — confirma el diagnóstico del oráculo, base de
H-TP-LADDER-01); (2) **leverage ~3-4×** — y ahí está el precio: el 30% pleno requería
lev ~4.5 con DD proyectado 15-18%, y por simetría el 2022 (−9% a lev 2) habría sido
−18/−25%. El 30% de 2025 era una decisión de APALANCAMIENTO, no de señal.

**2026 es el veredicto estructural**: ni la MEJOR política de la clase (S10) pasa de
+6.5% YTD vs los +16% del ritmo-30% — con el gate permitiendo 11 trades en un año
mean-reverting, el ritmo 30% NO EXISTÍA en un solo par. Confirmación empírica de la
Fundamental Law del panel: el 30% sostenido exige breadth (más sleeves), no más presión
sobre COP. Prohibido desplegar la celda ganadora por estos números; H-TP-LADDER-01 sigue
su cauce (motor + forward) y el apalancamiento del LIBRO (no de la pata) es la vía al
objetivo. 0 trials (mapa de decisiones, sin selección de despliegue).


---

## RESULTADO H-TP-LADDER-01 / v14 (2026-07-21) — PASA generalización de diseño · trials 67→68

Motor real pareado (v12 vs v14=v12+ladder 50/50 a TP y 2×TP; flag en motor, default off):

| Año | v12 (ret/DD/Calmar~) | v14 (ret/DD/Calmar~) | v14 gana |
|---|---|---|---|
| 2021 | −2.14 / 3.8 / −0.56 | −2.71 / 3.8 / −0.71 | no |
| 2022 | −9.22 / 9.4 / −0.98 | **−7.98 / 9.5 / −0.84** | sí |
| 2023 | −2.53 / 6.9 / −0.37 | **−0.93 / 6.5 / −0.14** | sí |
| 2024 | +4.60 / 3.5 / 1.31 | +4.61 / 3.5 / 1.32 | sí |
| 2025* | +7.69 / 5.7 / 1.36 | +6.59 / 5.1 / 1.28 | no (*re-medición) |

**Criterio pre-declarado (≥3/4 años de diseño): PASA (3/4).** El perfil es el de una
mejora estructural genuina: mejora los años MALOS y cede en el año seleccionado —
lo inverso del overfitting. Magnitudes modestas (0.2-1.6pp/año): se dice claro.
Congelada `smart_simple_v14_ladder` (SSOT + manifest); juez = forward desde freeze,
mismo protocolo sellado de v12. Entra a paper junto a v11/v12 (strategy_id propio, 064).


---

## CELDA S7-MOTOR (2026-07-21): el "escenario 20%" NO replica en ejecución honesta · trials 68→69

La config del +20.11% del contrafactual (ladder + cap 3.0) corrida en el MOTOR real:

| Año | S7-motor | (v14 cap 1.5) | HS |
|---|---|---|---|
| 2021 | −3.10% | −2.71% | 0 |
| 2022 | −7.64% | −7.98% | 5 |
| 2023 | −1.26% | −0.93% | 5 |
| 2024 | **−2.66%** (la sim decía +4.6 a lev bajo) | +4.61% | **4** |
| **Compuesto 21-24** | **−14.0%** | −6.3% | |
| 2025 | **+12.77%** (la sim prometía +20.11%) | +6.59% | 5 |

**Dos razones por las que el 20% era espejismo:**
1. **El simulador crudo ignoraba la dinámica del HS efectivo**: a lev 3 el buffer de
   precio se comprime a 3.5%/3 ≈ 1.17% — semanas que a lev 1.5 sobrevivían hasta el TP
   mueren en −3.5% de equity. 2024 VOLTEA de +4.6% a −2.66% (4 hard stops nuevos). El
   +20.11% de la sim vale +12.77% en ejecución honesta.
2. **El diseño paga −14% compuesto** — peor que v14 en la ventana limpia. El leverage
   sobre ESTA estrategia amplifica la parte que no tiene edge junto con la que sí.

**Conclusión sellada**: el apalancamiento no es una variable de diseño de la pata — es
una decisión de CAPITAL sobre lo graduado (a nivel libro, post-forward, con guard
fraccional-Kelly). La escalera (v14) se queda; el lev alto en la pata queda cerrado con
evidencia de motor. Celda reportada completa, no desplegable.


---

## MONITOREO 2026-07-21: v11/v12/v14 en 2025 + replay Ene-Jul 2026 · trials 69→71

Pedido del operador ("¿cómo va 2025 con cada una y cómo va el paper 2026?"). 2025 =
re-lectura de celdas pagadas (0 trials). **Replay 2026 de v12/v14 = 2 celdas nuevas
miradas (+2 trials)**: ese tramo queda CONTAMINADO para v12/v14 y su juez sigue siendo
el forward post-freeze (paper 2026-07-27). Prohibido seleccionar entre configs con esto.

| Config | 2025 ret/DD/HS | 2026 YTD ret/DD/HS | Nota |
|---|---|---|---|
| v11 producción | +7.35% / −7.8 / 5 | +3.36% / −1.5 / 1 | 2026 = forward REAL |
| v12 cap 1.5 | +7.69% / −5.7 / 4 | +3.12% / −1.5 / 1 | 2026 = replay descriptivo |
| v14 cap+ladder | +6.59% / −5.1 / 4 | +3.09% / −1.5 / 1 | 2026 = replay descriptivo |

Lectura: en 2026 el cap 1.5 casi no muerde (lev máx ~1.27) y la escalera apenas cambia
salidas → las tres son casi idénticas YTD (diferencia −0.24/−0.27pp por semanas de lev
recortado). La separación real entre ellas solo la dará el forward post-freeze.
Artefacto: `.claude/evidence/cop_monitor_2025_2026/2026-07-21/` (mensual + trades).


---

## H-CHRONOS-01 (PRE-REGISTRO 2026-07-22, 0 miradas) — fundacional zero-shot → q90 rango semanal COP

Diseño consolidado Claude+Codex (revisión adversarial R1-R6, `codex exec -p audit`,
artefacto del review en `.claude/evidence/cop_chronos/`). Aprobado por el operador.

**Hipótesis**: `amazon/chronos-bolt-base` zero-shot (SIN fine-tuning) predice el q90 del
rango de las próximas 5 sesiones mejor que la persistencia rolling-252. Mejora candidata
al SIZING, nunca a la dirección.

**Diseño SELLADO (15 puntos Codex, todos los grados de libertad clavados ex-ante):**
1. Modelo único: `amazon/chronos-bolt-base`, revisión = último commit HF con fecha
   < 2025-01-01 (publicación ≠ cutoff del corpus; el checkpoint precede al test).
2. Sin fine-tuning, sin calibración de hiperparámetros, sin modelos alternativos.
   **TimesFM prohibido incluso si Chronos pierde.**
3. Serie base: rango de sesión % = (max(high)−min(low))/close_última ×100 desde M5,
   sesiones con ≥30 barras (constructor idéntico a H-VOLT-01). Escala definida UNA vez.
4. Target: y_t = max(rng_{t+1..t+5}) sobre las próximas 5 sesiones ELEGIBLES (no días
   calendario).
5. Anti off-by-one: en el origen t solo son observables etiquetas hasta y_{t−5}.
6. Input univariado: historial causal de y (hasta y_{t−5}); contexto = máximo soportado
   por el modelo, truncado por la izquierda.
7. Horizon=5; se lee el canal q90 nativo del paso 5 (= y_t). SIN sample paths (los
   cuantiles de Bolt son marginales; máximos desde marginales = inválido — Codex R2).
8. TEST primario: 2025, UN disparo (orígenes con fecha en 2025; enero-2026 solo para
   realizar etiquetas de diciembre, jamás como input). 2026 YTD solo descriptivo.
   Confirmación real = forward post-freeze.
9. Baseline único del gate: persistencia q90 rolling-252 sobre y observable en t (mismo
   cutoff y_{t−5} para ambos). EWMA/HAR/celdas H-RISK: solo descriptivo.
10. Métrica: Δ = pinball_modelo − pinball_persistencia, pareada, AGREGADA POR SEMANA ISO
    del origen (targets solapados ⇒ 250 pérdidas diarias ≠ N=250).
11. IC95 bootstrap circular b=4 semanas, 2000 resamples, seed 42 — fijados aquí.
12. **WIN solo si el borde SUPERIOR del IC95(Δ) < 0.** Todo lo demás = NO_RECHAZA.
13. Smoke test funcional en ≤2023 (correr sin errores); CERO métricas comparativas.
14. Política de fallo: error de inferencia en un origen ⇒ se excluye y loguea; >5% de
    orígenes fallidos ⇒ corrida INVÁLIDA. Versiones de librerías al artefacto.
    Cambiar config tras ver problemas en TEST ⇒ corrida INVÁLIDA.
15. Contabilidad: abrir 2025 = **+1 trial (N 71→72)**. Pierde ⇒ puerta
    transformer/fundacional COP CERRADA por escrito. Gana ⇒ habilita UN trial económico
    posterior (techo de sizing tipo v13), juez = forward post-freeze.


---

## RESULTADO H-CHRONOS-01 (2026-07-22) — NO_RECHAZA por 0.0004 · trials 71→72 · PUERTA CERRADA

Disparo único TEST-2025 (246 orígenes, 0 fallos, 53 semanas ISO; revisión del modelo
6f8ced46 del 2024-11-28, pre-2025 como exigía el punto 1):

| | Pinball q90 semanal |
|---|---|
| chronos-bolt-base (zero-shot) | **0.10495** |
| persistencia rolling-252 | 0.12720 |
| Δ medio (modelo − base) | **−0.02225** (−17.5%) |
| IC95 block-bootstrap b=4 | **[−0.04584, +0.00040]** |

**El borde superior cruza cero por +0.0004 ⇒ NO_RECHAZA** (regla 12 sellada: WIN solo si
sup < 0). Es, por lejos, lo más cerca que algo ha estado de batir a la persistencia
(el transformer local ni siquiera ganó en media). La media favorece al fundacional con
claridad; la evidencia no alcanza el bar pre-firmado, y el bar no se mueve ex-post.

**Consecuencia del pre-registro (punto 15): la puerta transformer/fundacional COP queda
CERRADA por escrito** — local (H-VOLT-01) y zero-shot (H-CHRONOS-01), ambos NO_RECHAZA.
Prohibido re-correr con otros resamples/bloques/modelos (TimesFM sigue prohibido).
Única vía constitucional de revisita: NUEVO pre-registro sobre un período que hoy no
existe (p.ej. 2026 completo juzgado en 2027), con ADR y aprobación del operador — el
diagnóstico de hoy solo genera esa hipótesis futura, jamás un re-test sobre 2025.

Artefacto: `.claude/evidence/cop_chronos/2026-07-21/` (JSON + orígenes CSV + generator
+ review adversarial de Codex `codex_chronos_review.txt`).


---

## ENJAMBRE 2026-07-22 (9 agentes Claude + verificación Codex por agente) · trials 72→73

Programa aprobado por el operador ("procede a implementar toda, usa 10 agentes y un agente
de codex que verifique"). Cada implementación pasó `codex exec -p audit` independiente;
5 rechazos iniciales, todos corregidos y re-verificados. QA: cero regresiones funcionales,
motor bit-idéntico con flags OFF (2025 +7.35%/32 tr, 2026 +3.36%/11 tr, diff 0.0000pp).
Manifest v11 re-congelado v7 (hash d7a477d51d454e36, cambio clase feature-gated).

### H-V13-QRISK-01 design-run EN EL MOTOR: APRUEBA DISEÑO (+1 trial, 72→73)

Instrumento corregido (el replay externo fue INVALID_INSTRUMENT): techo probabilístico
DENTRO del motor (flag `v13_ceiling_enabled`, default False = bit-idéntico, verificación
automática `bit_identity_v12` MATCH 2022-24). Fórmula sellada sin cambios: q90_hat =
0.5·persistencia(q90 roll-252 rango 5d) + 0.5·QuantReg(τ=.90, {vol-of-vol, gap-cola,
EMBI-acel, RESINT-z}, expanding <t); techo = min(1.5·clip(mediana_diseño/q90_hat, .5, 1),
vt_max) — invariante en el motor; constructor acotado a cutoff 2024-12-31.
DESVIACIÓN DECLARADA (fail-safe de ingeniería, una variante, no optimizado): QR <52 sem
o feature NaN ⇒ persistencia sola; mediana <26 valores ⇒ techo inactivo. Instrumentado:
en el design-run 2022-24 NO actuó nunca — 157/157 semanas con QR real (fallbacks solo en
2020-21 pre-diseño). Techo vinculante 77/157 semanas (mín 1.03).

| Año | v12 | v13 |
|---|---|---|
| 2022 | −9.22% / DD 9.38 / HS 5 | −7.46% / DD 9.38 / HS 4 |
| 2023 | −2.53% / DD 6.92 / HS 5 | −1.24% / DD 6.19 / HS 4 |
| 2024 | +4.60% / DD 3.50 / HS 2 | +4.58% / DD 3.50 / HS 2 |
| Compuesto | −7.44% / DD 12.89 / Calmar −0.1975 | **−4.42% / DD 10.04 / Calmar −0.1489** |

**Calmar_v13 ≥ Calmar_v12 ⇒ APRUEBA por diseño** (robusto en ret/DD). Ambos negativos en
diseño — dominancia de riesgo, no rentabilidad; juez real = forward desde su freeze
(protocolo sellado de v12). Codex re-verify: APROBADO sin issues.
Evidencia: `.claude/evidence/cop_v13_engine/2026-07-22/`.

### H-ENTRY-01 — PRE-REGISTRO SELLADO (0 trials hasta ejecutar)

Microestructura de entrada del lunes. Prior económico (perfil descriptivo SOLO diseño
2020-24, `.claude/evidence/cop_entry_profile/2026-07-22/`): la primera media hora es la
ventana más cara del día — rango de closes/30min en lunes 29.2 pb (8:00-8:30) vs 13.3 pb
(9:30-10:30), ratio 2.20x, ≥2.0x en CADA año de diseño; vol 5m 2.54x. Regla as-is del
motor documentada: entrada al CLOSE DIARIO del lunes (~12:55 COT), pese a señal desde
08:15 (el timestamp 09:00 de los trades es cosmético). Hallazgo de calidad: 94.5% de
barras M5 flat (2020-22 = snapshots) — proxies con closes, válidos.
**Hipótesis**: TWAP 9:30-10:30 (12 closes M5) mejora el precio efectivo de entrada
(improvement_bp = dir·(entry_engine−entry_twap)/entry_engine·1e4). **Bar**: IC95 bootstrap
por bloque semanal (10k, seed 42) pooled 2020-24 excluye 0 a favor de la TWAP y supera el
costo incremental declarado. **Instrumento**: `scripts/analysis/cop_entry_compare.py`
CONGELADO SIN EJECUTAR (raise duro >2024, exige --confirm-trial; hardening post-Codex:
truncado ≤2024 tras carga). **Primera ejecución real = +1 trial**; juez de confirmación =
forward shadow. Si el IC95 de diseño incluye 0 ⇒ REJECT sin variantes.

### Eventos 0-trials (ingeniería de riesgo + infraestructura + datos; N NO cambia)

**BOOK-ERC-01 (libro multi-activo)**: pesos ERC solo-covarianza (Ledoit-Wolf, semanas
ISO-2025) sobre 3 campeonas: COP v12 0.4205 / XAU gold_trend_simple 0.3833 / BTC hodl_b1
0.1961, RC 33.33% c/u; vol libro 7.27% → escala 1.376 a objetivo 10%. Correlaciones 2025
≈ 0 (−0.02/+0.09/+0.05). Pesos = riesgo, no retorno; juez = forward, sin claims.
`config/book/book_v1.yaml` · `.claude/evidence/book_construction/2026-07-22/`.
Codex: OBSERVACIONES (3 menores) → corregidas (FAIL duro en ERC, cobertura 52w XAU/BTC).

**GOBERNADOR KELLY (BOOK-LEVERAGE-GOVERNOR.md, DRAFT_AWAITING_OPERATOR_SIGNATURE)**:
f* Kelly serie honesta (43 trade-semanas): continuo 9.39 / discreto 7.99, **IC95 block-b4
[−6.14, +48.71] INCLUYE CERO** (P(f*≤0)=14.5%) — la serie honesta no prueba ni que el
Kelly óptimo sea positivo ⇒ 1.0x pre-graduación es necesidad estadística, no prudencia.
Regla post-graduación: clip(0.25·f*_forward, 0, 1.5) con piso cero (Kelly fwd ≤0 ⇒ flat +
revisión); haircut ×½ = POLÍTICA declarada (no estimador Lo-Mertens — corrección Codex);
escalera DD 7%→1.0x / 10%→0.5x / 12%→flat, histéresis 4w−2pp, anti-relajación §5.
`.claude/evidence/book_kelly/2026-07-22/`. Codex: RECHAZADO → 3 issues corregidos.

**STRESS MC DEL LIBRO (v2 post-Codex)**: 10k paths 52w seed 42, block b=4. Codex rechazó
v1 (4 issues: high-water inicial, hostil no causal, runs no contiguos, etiquetado
trigger/applied) — todos corregidos. Base (hereda régimen 2025): MaxDD p95 5.01% (1.0x) /
7.46% (1.5x). **Escalera desde 1.5x: p95 7.40% (base) y 9.89%, p99 11.12% (hostil causal
por runs) — contiene el p95 < 12% en ambos regímenes** (P(DD>12%) hostil 0.3% vs 3.1%
estático). Advertencias: ES97.5 base positivo = artefacto deriva 2025; hostil causal más
benigno que look-ahead (las semanas post-vol-alta de 2025 se recuperaron) — no es techo
del riesgo real. Presupuesto 7/10/12 = PROPUESTA, opera el operador.
`.claude/evidence/book_mc_stress/2026-07-22/`.

**E1-XASSET-INFRA**: señales trend (TSMOM 3/6/12m) + value (reversión 5y) z-scoreadas XS,
causales shift(1), SOLO ≤2024, universo {COP, MXN, BRL, XAU, BTC, SPX} (45y XAU … 7.4y
BTC; USD/CLP no existe como OHLC). Carry = NULL con la lista de qué falta por asset
(funding BTC ya en DB, gateado por S4/H-POS-01). Harness purged-CV+embargo wrappea el
SSOT de métricas; smoke solo sintético. Ningún OOS abierto, cero métricas de performance.
Codex: APROBADO sin issues.

**DATA-EVENT C1 (forwards BanRep)**: `macro_banrep_forwards_monthly` — 2.308 filas,
2005-01→2026-05, 9 tenores/mes, sin huecos; devaluación implícita ponderada por monto
(outright no se publica). PIT conservador published_at = fin de mes + 60d (CHECK exacto
en tabla). Cobertura 1997-2004 NO existe machine-readable (retirada con OBIEE) — la
premisa "1997→" del plan queda corregida. Codex: RECHAZADO → 5 issues corregidos
(constraint exacto, span esperado completo, FAIL duro, cross-check TOTAL bloqueante,
validación de fila fuente) + re-ingesta limpia.

**DATA-EVENT C2 (remesas)**: `macro_remesas_monthly` — 317 filas, 2000-01→2026-05, sin
huecos, estacionalidad dic verificada (525 vs 458 USD mn). Endpoint JSON SUAMECA serie
15363; PIT = fin de M+1 (cargue real día 24-26 de M+1). Codex: RECHAZADO → 2 issues
corregidos (FAIL duro, span esperado) + re-ingesta limpia.
Uso de ambos: join `published_at <= as_of`, nunca por month; ningún estudio abierto.


---

## DIRECTIVA OPERADOR 2026-07-22: paper anclado a ENERO 2026 + refresh automático (0 trials)

"El paper debe ser siempre calculado desde enero del 2026 y dejarlo habilitado para que
corra el resto del año." Implementación:
- `scripts/pipeline/candidates_paper_ledger.py` → `public/data/production/paper/
  candidates_ledger_2026.json` (dashboard-served, tracked): series 2026 completas de
  v11 (+3.36%, forward real), v12 (+3.12%) y v14 (+3.09%) — para v12/v14 el tramo
  Ene→2026-07-21 es el replay YA PAGADO (trials 69→71) y el campo `judge_window`
  separa las semanas post-freeze que son las ÚNICAS que consume el juez sellado
  (hoy 0 trades: el paper judicial arranca el lunes 2026-07-27).
- Refresh: tarea `paper_ledger_2026` en `forecast_h5_l6_weekly_monitor` (Vie 14:30 COT)
  — corre el resto del año sin intervención. Regenerar la misma celda semanalmente =
  monitoreo, 0 trials; los cortes decisorios siguen siendo los de FASE 3 del plan.
- v13 EXCLUIDA del ledger hasta su freeze (abrir su 2026 = +1 trial, decisión operador).
- Libro: pesos ERC congelados en el JSON; componente COP poblado, XAU/BTC = NA hasta
  integrar sus bundles (ITT, nunca 0 inventado).
- Gate semana-15 NEUTRALIZADO el mismo día (3 YAML + monitor `sealed_judge_only`):
  el monitor semanal reporta integridad; ninguna decisión fuera de los cortes sellados.
