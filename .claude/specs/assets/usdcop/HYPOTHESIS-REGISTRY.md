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
n_trials_total: 58
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
