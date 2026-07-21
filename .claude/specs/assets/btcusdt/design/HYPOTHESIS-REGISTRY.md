---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/btc_strategy/strategies.py
# Conteo de trials LEGIBLE POR MÁQUINA (ver §"Conteo para el DSR" abajo, que da la fórmula
# pero nunca su resultado). `scripts/analysis/profitability_evidence.py` lo lee de aquí y
# lanza excepción si falta — el DSR no puede depender de un número hardcodeado en el código.
n_trials_total: 34
n_trials_scenarios: [23, 34, 48]   # solo registro / registro+sensibilidades / +descartados
n_trials_sources:
  - "Registro principal de este archivo: 21 filas (H-xxx)"
  - "Sensibilidades pre-registradas: σ_objetivo 3 + banda 3 + pesos R 3 + re-versión LLM ≥1"
  - "public/data/strategies/btc_*/backtests/* = 16 bundles publicados (suelo de verificación)"
  - "H-VOL-01 EWMA sizing (registrada 2026-07-21, prospectiva): +1"
sigma_trials: null
sigma_trials_grid: [0.05, 0.10, 0.15]   # titular = el DSR MÍNIMO de la rejilla
---
# HYPOTHESIS REGISTRY — Registro de Hipótesis y Trials

> **Insumo directo del Deflated Sharpe Ratio.** El DSR necesita saber **cuántas hipótesis se
> probaron** (incluidas las descartadas) para corregir el Sharpe por selección. Este
> documento es la **cuenta oficial de trials**. Toda hipótesis probada —cada componente en
> aislamiento, cada variante de peso, cada análisis de sensibilidad, cada re-fit del
> clasificador— se registra aquí ANTES de correr su test, con su H0/H1 y su criterio de
> decisión pre-registrado.
>
> Se sincroniza con MLflow (cada fila = un `run` etiquetado). El conteo `N_trials` alimenta
> la fórmula del DSR en SPEC-11 §9.

## Convenciones

- **Estado:** `PENDIENTE` (registrada, no corrida) · `RECHAZA_H0` (aporta) · `NO_RECHAZA`
  (descartada) · `INCONCLUSA`.
- **α global** = 0.05, con corrección de comparaciones múltiples (Benjamini-Hochberg FDR
  sobre la familia de tests de aporte de componentes; ver SPEC-11 §9).
- Cada fila cuenta como **≥ 1 trial** en el DSR aunque termine `NO_RECHAZA`. Las variantes de
  sensibilidad cuentan como trials adicionales.

## Registro

| ID | Spec | Hipótesis nula (H0) | Estadístico / test | Criterio de decisión | N_trials | Estado |
|---|---|---|---|---|---|---|
| H-REG-01 | SPEC-01 | Los regímenes del HMM no se relacionan con el retorno forward ajustado por downside | Kruskal-Wallis sobre Sortino forward por régimen + permutación | p < α tras BH | 1 | PENDIENTE |
| H-REG-02 | SPEC-01 | El re-fit del HMM produce labels inestables (≠ del histórico congelado) | % de labels que cambian entre re-fits consecutivos en solape | **> 20 %** ⇒ inestable, NO entra | 1 | PENDIENTE |
| H-POS-01 | SPEC-02 | El funding extremo no precede a caídas de vol-adjusted return | Diferencia de CVaR forward tras funding p>95 vs. resto (bootstrap) | IC 95 % del delta no cruza 0 | 1 | PENDIENTE |
| H-CMB-01 | SPEC-03 | La combinación en riesgo (R) **no** mejora Calmar OOS vs. multiplicación ingenua ciclo×funding | Bootstrap pareado sobre ΔCalmar OOS | IC 95 % de ΔCalmar > 0 | 1 | PENDIENTE |
| H-CMB-02 | SPEC-03 | z_ciclo y z_funding son ortogonales (no requieren combinación) | \|ρ rodante 90d\| sostenido | **> 0.4** ⇒ combinar (rechaza ortogonalidad) | 1 | PENDIENTE |
| H-LIQ-01 | SPEC-04 | El gate de liquidez no aporta Calmar sobre el núcleo | Bootstrap sobre ΔCalmar (núcleo+M_liq vs núcleo) | IC 95 % > 0 + DSR | 1 | PENDIENTE |
| H-LIQ-02 | SPEC-04 | El aporte de ETF flows/stablecoins es momentum disfrazado | ΔCalmar controlando por momentum de precio (regresión parcial) | Aporte sobrevive control ⇒ se conserva | 1 | PENDIENTE |
| H-EVT-01 | SPEC-05 | El clasificador de eventos no tiene poder de detección (recall = base) | Recall sobre corpus etiquetado (cota superior, §7.8) + FP/año | Recall≈100 % clases catastróficas ∧ FP/año aceptable | 1 | PENDIENTE |
| H-EVT-02 | SPEC-05 | El gate de eventos no mejora Calmar/CVaR (las aplanadas cuestan más de lo que salvan) | PnL diferencial por aplanada (bootstrap) | Suma de PnL diferencial > 0 | 1 | PENDIENTE |
| H-ENG-01 | SPEC-06 | S3 (motor completo de reglas) no supera a B2 en Calmar OOS | Bootstrap pareado ΔCalmar(S3, B2) | IC 95 % > 0, dentro de presupuesto de turnover | 1 | PENDIENTE |
| H-ENG-02 | SPEC-06 | S3 no supera a B1 (HODL vol-targeted) en Calmar OOS | Bootstrap pareado ΔCalmar(S3, B1) | IC 95 % > 0 | 1 | PENDIENTE |
| H-MET-01 | SPEC-08 | El meta-modelo no discrimina señales buenas de malas (AUC = 0.5) | Test de DeLong sobre AUC OOS | AUC > 0.5 con p < α | 1 | PENDIENTE |
| H-MET-02 | SPEC-08 | S4 (con meta-labeling) no supera a S3 en Calmar OOS | Bootstrap pareado ΔCalmar(S4, S3) | IC 95 % > 0 | 1 | PENDIENTE |
| H-RL-01 | SPEC-09 | S5 (con RL táctico) no supera a S4 en Calmar OOS (mediana de seeds) | Bootstrap sobre mediana de ΔCalmar(S5, S4) por seed | IC 95 % > 0 sobre mediana ≥5 seeds | 1 | PENDIENTE |
| H-SYS-01 | SPEC-11 | El sistema final tiene Sharpe ≤ 0 tras deflación | **Deflated Sharpe Ratio** con N_trials de este registro | DSR > 0 con confianza ≥ 95 % | 1 | PENDIENTE |

## Sensibilidades (trials adicionales, pre-registradas)

Cada celda es un trial extra en el DSR. **No se elige la mejor**: se reportan todas (SPEC-11).

| Familia | Celdas | Trials |
|---|---|---|
| σ_objetivo | {25 %, 30 %, 35 %} | 3 |
| Banda de rebalanceo | {10 %, 12.5 %, 15 %} | 3 |
| Pesos R (ciclo/funding) | {60/40, 70/30, 80/20} | 3 |
| Re-versión del clasificador LLM | por versión de modelo | ≥ 1 c/u |

## Conteo para el DSR

```
N_trials_total = Σ(filas del registro principal) + Σ(celdas de sensibilidad) + Σ(re-versiones LLM) + componentes descartados
```

> **Regla dura:** ningún claim de edge (Sharpe/Calmar positivo) es válido sin recomputar el
> DSR con el `N_trials_total` actualizado. Añadir un experimento y no actualizar el conteo es
> una violación de la constitución (§2.2.3).


## Resultado H-POS-01 (2026-07-06, OLA 5 — btc_trend_funding_s4)

S4 = B2 trend × funding brake (k=0.25, floor=0.4, priors ex-ante, `src/btc_strategy/strategies.py`).
Data: 2492 días de funding Binance (`crypto_derivatives_daily`), features causales shift(1).

| Métrica | B2 | S4 |
|---|---|---|
| DSR (N=4 trials) | 0.9987 | 0.9976 |
| OOS-2025 return | −1.37% | **−1.42%** |
| OOS-2025 MaxDD | −8.6% | **−7.3%** |

**Veredicto honesto:** S4 pasa el DSR pero NO cumple el gate pre-registrado (OOS-2025 positivo ∧
bate a B2 en Sharpe+Calmar). El freno por funding reduce levemente el DD pero no aporta retorno
OOS ⇒ **NO se promueve**; se documenta como "funding-brake solo no rescata el año lateral".
Siguiente palanca: z_ciclo on-chain (H-REG-01/H-BTC-CYCLE-02) — requiere extractor on-chain (B3).
Trial añadido: +1 (N=4).

---

## H-VOL-01 — EWMA/RiskMetrics como estimador de volatilidad para el sizing

**Registrada 2026-07-21, ANTES de implementarla.** Skill aplicada:
`vendor/quant-skills/04-backtesting-validation/volatility-modeling`.

**Motivación (mecánica, no de resultados)**: el sizer usa `realized_vol_20` — una media móvil
simple de 20 días con pesos iguales. La skill señala el defecto: una ventana con pesos iguales
trata un shock de hace 20 días igual que el de ayer, y luego lo **descarta de golpe** cuando
sale de la ventana. EWMA de RiskMetrics (`σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1}`, λ=0.94 diario,
ventana efectiva ≈ 17d) decae de forma suave.

Esto NO es una búsqueda de señal: no toca `intent`, solo el estimador que alimenta
`vol_target_size`. Es exactamente la palanca que la librería sí sanciona — riesgo y exposición,
nunca dirección.

- **H0**: `Calmar(BTC con σ_EWMA) ≤ Calmar(BTC con realized_vol_20)`.
- **H1**: EWMA mejora el Calmar al reaccionar antes a los cambios de régimen de vol.
- **Estadístico**: ΔCalmar por block bootstrap pareado (bloque 20d, 365/año, 5000 muestras).
- **Criterio**: IC95 excluye cero **en el forward**.
- **λ = 0.94 fijo**, el estándar de RiskMetrics. **No se barre λ**: barrer sería grid search
  sobre el test, y cada celda costaría un trial.

**Un solo cambio de variable** (`experiment-protocol` regla 1): el estimador de volatilidad.
Nada más se toca.

**Advertencia honesta esperada**: EWMA es IGARCH (α+β=1) — los shocks de vol **persisten
indefinidamente**, sin reversión a la media. En un activo con vol tan mean-reverting como BTC
eso puede dejar el tamaño demasiado bajo demasiado tiempo tras un shock. Es una razón real por
la que esto puede EMPEORAR el Calmar, y se reporta pase lo que pase.

**Coste en trials**: +1. `n_trials_total` 31 → 32.

### Resultado H-VOL-01 (2026-07-21) — **NO_RECHAZA H0**

| Ventana | realized_vol_20 | EWMA λ=0.94 |
|---|---|---|
| Historia completa · ann | 20.56% | 21.12% |
| Historia completa · MaxDD | −54.61% | −52.57% |
| Historia completa · **Calmar** | 0.376 | **0.402** |
| **OOS-2025 · ann** | −5.09% | **−7.56%** |
| **OOS-2025 · Calmar** | −0.198 | **−0.286** |

`ΔCalmar(EWMA, base)` = **+0.026 observado**, bootstrap +0.030, **IC95 [−0.096, +0.174] —
incluye cero**.

**Veredicto: H0 NO se rechaza.** EWMA mejora marginalmente el Calmar sobre la historia completa,
pero (a) la mejora no es distinguible de cero y (b) **en el año held-out es peor** (−0.286 vs
−0.198).

**La advertencia pre-registrada se cumplió.** Escribí antes de correrlo que EWMA es IGARCH
(α+β=1), así que los shocks de volatilidad persisten sin reversión a la media, y que en un
activo con vol tan mean-reverting como BTC eso podía dejar el tamaño suprimido demasiado tiempo.
Es exactamente lo que pasó en el 2025 lateral: EWMA mantuvo la exposición baja tras los shocks
y se perdió el rebote.

**Acción**: `realized_vol_20` se mantiene. El estimador EWMA queda en
`services/common/metrics.py::ewma_volatility` como herramienta disponible, **no cableado al
sizer**. Trial contabilizado (N=32) — el coste de mirar se paga aunque el resultado sea negativo.


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

### Resultado H-VOLF-01 (2026-07-21) — **NO_RECHAZA H0, con hallazgo invertido**

QLIKE OOS-2025: HAR 0.7117 vs EWMA 0.4836 — **HAR es significativamente PEOR**
(diferencial +0.228, IC95 [+0.013, +0.412] excluye cero por el lado positivo). La estructura
lineal de HAR no captura los saltos de vol de BTC; la persistencia EWMA gana con claridad.
Coherente con H-VOL-01. **En BTC, la persistencia ES el estimador — dos veces confirmado.**

---

## H-BASIS-01 — basis extremo como freno (PENDIENTE, deliberadamente sin correr)

**Registrada 2026-07-21 como PENDIENTE. NO se corre todavía, y la razón queda escrita:**

`basis_annualized` tiene 994 días de historia en el seed y `merge_funding_features` no lo usa
(solo funding). Es la única feature de derivados profunda sin explotar. PERO: S4
(funding-brake) ya falló su gate pre-registrado, y un segundo freno del mismo tipo sobre la
misma señal-base tiene prior bajo. Criterio de activación: motor de bandas de SPEC-06
construido, O ≥180 días de OI acumulado (hoy: 44 y creciendo semanalmente vía
`l0c_ingest_derivatives`, verificado success). Registrar sin correr no cuesta trial;
correrla costará +1.

## Nota de datos: `liquidations_usd` está MUERTA

0/2506 no-nulos en el seed. La REST API de liquidaciones fue deprecada; la columna espera la
Fase-3 WS del extractor. Que nadie la "descubra" y construya features sobre NULLs.

## Estado de la vía vol-forecasting en BTC (cierre 2026-07-21)

Dos hipótesis corridas, dos NO_RECHAZA con la misma conclusión desde ángulos distintos:
H-VOL-01 (EWMA en el sizer) y H-VOLF-01 (HAR **significativamente peor** que EWMA, IC95
[+0.013, +0.412]). **En BTC la persistencia es el estimador, dos veces confirmado.** La vía
queda cerrada salvo dato nuevo (OI con historia, on-chain de migración 052).

---

## H-DIR-FUND-01 — funding como predictor DIRECCIONAL (registrada ANTES de implementar)

**Registrada 2026-07-21.** `z_funding` (2.506 días) solo se probó como FRENO (S4, no
promovida). Nunca como predictor de dirección: funding extremo positivo = crowding largo =
probabilidad de reversión. Mecanismo distinto al freno; información que el zoo price-only no
contiene. Es el "unblock cripto-nativo" del STRATEGIC-ASSESSMENT.

- **H0**: DA_OOS(Ridge base+z_funding) ≤ DA_OOS(Ridge base) — el funding no añade dirección.
- **Diseño pre-firmado (UNA variable)**: modelo Ridge lineal (el del track, sin barrer),
  horizonte 5d, features base {ret_1d, ret_5d, ret_20d, rv_20} vs base+{z_funding(t−1)};
  walk-forward expansivo refit 21d; veredicto = DA sobre OOS-2025 con **McNemar pareado**
  (mismas semanas, mismos aciertos/fallos discordantes).
- **Criterio**: p<0.05 del McNemar Y DA_con > DA_sin. Cualquier otro resultado ⇒ NO_RECHAZA
  y la vía direccional-funding se cierra hasta que OI/basis acumulen historia.
- **Si pasa**: NO se opera — entra a la secuencia de promoción completa (manifest, PIT, forward,
  DSR/PBO) como candidata; la dirección sigue diagnóstica hasta superar todo (regla Codex).

**Coste en trials: +1 (33 → 34).**


### Resultado H-DIR-FUND-01 (2026-07-21) — **NO_RECHAZA H0**

| | DA OOS-2025 (diaria, h=5d) |
|---|---|
| Ridge base {ret_1d, ret_5d, ret_20d, rv_20} | **0.4986** |
| Ridge base + z_funding(t−1) | 0.4767 |

Δ = **−2.19pp** (el funding EMPEORA la dirección), McNemar p = 0.2153.

**Veredicto**: el funding no contiene dirección a 5 días — coherente con S4 (como freno
tampoco aportó retorno). Con esto, **las dos vías de funding están cerradas** (freno y
dirección). La vía cripto-nativa direccional queda a la espera de datos con historia que hoy
no existe: OI (44 días acumulando), basis con más profundidad, on-chain (migración 052).
Trial pagado (N=34). Cierre honesto: dos experimentos, dos noes, cero operaciones basadas
en funding.
