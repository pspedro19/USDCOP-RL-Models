---
kind: as-built
status: PARTIAL
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - src/strategies/spx500_regime_gated_v1/run_strategy.py
  - src/strategies/spx500_regime_gated_v1/load_real.py
  - src/validation/sp500_oos_gate.py
n_trials_total: 17
n_trials_scenarios: [13, 17, 23]
n_trials_sources:
  - "_config_family(): 4 vol targets x 3 MA windows = 12 celdas, todas evaluadas"
  - "B1/B2/S3: 3 variantes de política declaradas en el patrón"
  - "H-SIMP-SPX-01 + H-SIMP-SPX-02 (registradas 2026-07-21, prospectivas): +2"
  - "sin bundles publicados todavía (piso = 0): la estrategia no ha pasado a `available`"
sigma_trials: null
sigma_trials_grid: [0.05, 0.10, 0.15]
---

# HYPOTHESIS-REGISTRY — SPX500

> **Creado 2026-07-21.** La estrategia `spx500_regime_gated_v1` se integró con su motor, sus
> gates y sus tests, pero **corriendo sobre `datagen.generate()`** — una serie sintética cuyo
> propio docstring advierte `NO evidencia de alfa`. Cualquier métrica publicada hasta ahora
> mide el cableado, no el mercado.
>
> Este registro existe porque el DSR necesita un conteo de trials auditable, y porque el
> harness `scripts/analysis/profitability_evidence.py` **lanza excepción** si no lo encuentra.

---

## 1. Trials declarados

| Fuente | Celdas | Nota |
|---|---|---|
| `_config_family()`: vol target ∈ {0.08, 0.10, 0.12, 0.15} × MA ∈ {150, 200, 250} | 12 | Todas se evalúan para alimentar el PBO |
| Variantes de política B1 / B2 / S3 | 3 | Declaradas en el patrón; solapan parcialmente con lo anterior |
| **Total conservador** | **14** | |

El grid de 12 celdas **existe para medir el proceso de selección**, no para elegir su ganadora.
El adaptador de evidencia está clavado al prior central (target 0.10, MA 200); tomar la mejor
celda del barrido sería precisamente el sesgo que el PBO cuantifica.

## 2. Estado de los datos

| Aspecto | Estado |
|---|---|
| Precio | **Real** — `data/snapshots/public_daily/spx500_daily.parquet`, SPY `adj_close` (total-return, requerido por SDD-000 §4; usar `^GSPC` inflaría el alfa ~1.8 %/año) |
| Cobertura | 1.643 filas, 2020-01-02 → 2026-07-17 |
| VIX | **Proxy** — vol realizada 21d anualizada. `FRED:VIXCLS` no está en el snapshot |
| Macro stress | **Proxy** — z-score 252d del proxy de vol. `FRED:NFCI` / HY-OAS ausentes |
| `available_at` | Presente pero **reconstruido** (cierre + 1 día), no un vintage del proveedor |

**Consecuencia**: el clasificador de régimen recibe volatilidad *realizada* en lugar de
*implícita*. Es un modelo distinto y más débil que el diseñado, no el mismo modelo con otros
datos. Se reporta como proxy en cada artefacto en vez de sustituirlo en silencio.

Sin vintages reales, el estado máximo alcanzable es `research_validated` — nunca `production`
(`quant-constitution.md` §4).

## 3. Prospectivo

Toda hipótesis nueva se registra aquí **antes** de correr su test e incrementa
`n_trials_total`. El harness compara `params_hash` entre corridas y exige el incremento cuando
cambia.

---

## 4. Hipótesis prospectivas (registradas ANTES de correr)

### H-SIMP-SPX-01 — ¿el gate de régimen aporta o solo reduce exposición?

**Registrada**: 2026-07-21. **CORREGIDA el mismo día** (ver nota de retractación abajo).

**Medición con la política real** (`spx_regime_gated_v1`, datos SPY reales 2020-2026):

| Variante | ann | MaxDD | Calmar | up cap | down cap | ratio |
|---|---|---|---|---|---|---|
| `spx500_regime_gated_v1` | 5.70% | **−8.85%** | **0.644** | 32.1% | 31.0% | 1.038 |
| `spx500_trend_simple` (sin gate) | 7.01% | −12.14% | 0.577 | 43.3% | 42.4% | 1.021 |
| baseline tonto MA200 siempre-on | 148.92% | — | **1.645** | — | — | — |

**El gate SÍ aporta frente a la versión sin gate**: sube el Calmar de 0.577 a 0.644 recortando
el drawdown de 12.1% a 8.9%. Cuesta 1.3 puntos de retorno anual a cambio de un tercio menos de
caída — un intercambio defendible.

**Pero ninguna de las dos bate al baseline tonto** (Calmar 1.645). Ése sigue siendo el problema
real, y no lo causa el gate.

- **H0**: `Calmar(spx500_trend_simple) ≥ Calmar(spx500_regime_gated_v1)`.
- **H1**: el gate aporta ⇒ ya **no se rechaza** con los datos actuales.
- **La hipótesis viva pasa a ser H-SIMP-SPX-02**: ¿alguna de las dos bate al MA200 pelado
  siempre-encendido? Registrada abajo.
- **Juez**: forward. La observación se hizo sobre 2020-2026.

> ### ⚠️ Retractación (2026-07-21)
>
> La primera versión de esta hipótesis afirmaba que "el gate de régimen es una perilla de
> apalancamiento, no un filtro", citando capture 43.3%/42.4% y ratio 1.02.
>
> **Ese número no era del gate.** El adaptador de evidencia construía
> `trend_on * vol_target_weights` a mano — que es exactamente la variante SIMPLE — y **nunca
> invocaba `regime.py`**. Se destapó porque la variante "simplificada" puntuó idéntica a la
> "gated" (Calmar 0.577 vs 0.578): dos cosas distintas no dan el mismo número.
>
> Medir una estrategia que no es la estrategia es peor que no medirla: toda conclusión sobre
> "el gate" habría sido sobre código que jamás se ejecutó. Corregido en
> `profitability_adapters.py::spx500()`, que ahora llama a `spx_regime_gated_v1(df)`.

### H-SIMP-SPX-02 — ¿algo bate al MA200 pelado?

- **H0**: `Calmar(MA200 siempre-on) ≥ Calmar(cualquier variante con vol target o gate)`.
- **Motivación**: 1.645 contra 0.644 y 0.577. Toda la maquinaria de sizing cuesta más de lo
  que aporta **en este periodo**.
- **Advertencia de régimen**: 2020-2026 es un mercado alcista de renta variable con una única
  caída profunda (marzo 2020). Que un largo-siempre gane ahí es en buena parte **beta**, no
  habilidad. Un forward con un bajista prolongado es la prueba real, y es justo donde un gate
  de régimen debería justificarse.
- **Juez**: forward.

**Coste en trials**: +2 (H-SIMP-SPX-01 y -02). `n_trials_total` 14 → 16.


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

QLIKE OOS-2025: HAR 0.6307 vs EWMA 0.8202 (mejor en media, la mayor diferencia de los 4),
pero IC95 [−0.511, +0.039] **incluye cero** — un año de datos de índice no basta para
distinguirlo. Candidato natural a re-evaluación cuando el forward acumule; sin re-corridas
sobre el mismo 2025.

---

## ERRATUM 2026-07-21 — el Calmar 1.641 de `ma200_always_on` queda RETRACTADO y recomputado

**Defecto (encontrado por Codex en su plan v1 SPX, verificado en código por Claude):**
`profitability_evidence.py::evaluate` puntuaba el baseline tonto con (a) la señal SIN rezagar
contra el retorno open-to-open de la misma barra — un día de look-ahead — y (b) el stream de
costos de la estrategia gated en lugar del turnover propio del baseline. Afectaba también al
tonto de Oro (`sma_vote_always_on`).

**Corrección aplicada** (mismo convenio de ejecución que toda estrategia: lag 1 barra +
costo |ΔW|×tarifa-unitaria): commit en `profitability_evidence.py`; artefactos anteriores
(`.claude/evidence/profitability/2026-07-20/`) se conservan como INVÁLIDOS para este gate,
los corregidos viven en `2026-07-21/`.

**Números corregidos:**
- `ma200_always_on`: Calmar 1.641 → **0.5393**. El campeón gated (0.6482) **SÍ bate al
  baseline tonto**; el gate `calmar_gt_dumb_baseline` pasa ahora.
- `sma_vote_always_on` (Oro): → **−0.032**.
- Nada más cambia: DSR 0.8813 < 0.95, `net_return_gt_b1` sigue FAIL → SPX sigue
  `research_only`. Esta corrección NO es una hipótesis nueva ni consume trial
  (medición defectuosa de un gate ya registrado; conteo se conserva). H-SIMP-SPX-02 sigue
  viva con el forward como juez, ahora contra el baseline BIEN medido.
