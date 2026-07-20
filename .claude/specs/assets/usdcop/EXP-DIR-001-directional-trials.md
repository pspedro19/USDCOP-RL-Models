---
kind: audit
status: HISTORICAL
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/forecasting/enhance_v2.py
  - scripts/pipeline/train_and_export_smart_simple.py
  - services/common/metrics.py
---
# EXP-DIR-001/002 — ¿existe edge direccional en USD/COP semanal?

**Fecha**: 2026-07-20 · **Trials registrados**: 9 (7 direccionales + 2 de estrategia)
**Motivo**: el operador pidió "un muy buen forecasting direccional". Antes de intentar
mejorarlo había que medir si el edge existe.

## Contexto: un bug encontrado primero

`enhance_v2.py` leía `macro["fecha"]` (es el DatetimeIndex, no una columna) y buscaba
nombres de columna en minúscula que `MACRO_DAILY_CLEAN` no usa. El `KeyError` se capturaba
y el bloque de relleno dejaba `carry_diff` y `term_spread` en **0.0 constante** — dos
features muertas en `feature_cols` desde que v2 se desplegó.

Corregido. `carry_diff` pasó a llamarse `rate_diff_ibr_ust2y` porque es un diferencial
COP-overnight vs US-2Y, no un carry FX puro (FedFunds es mensual y no está en el CLEAN).

**Consecuencia constitucional**: cambiar el conjunto de features es un **trial nuevo** (§1).
El 2025 dejó de ser evidencia limpia para esta versión.

## EXP-DIR-001 — precisión direccional (walk-forward estricto)

Hipótesis declaradas ex-ante, **todas reportadas** (§1 prohíbe elegir la mejor celda).
`alpha = DA − DA(always_short)`.

| Hipótesis | 2024 α | 2025 α | 2026 α |
|---|---|---|---|
| H1 baseline Ridge+BR | −0.057 | −0.077 | 0.000 |
| H2 macro revivido | +0.019 | −0.135 | +0.053 |
| H3 clasificador (predice signo) | +0.038 | −0.096 | +0.105 |
| H4 demeaned (corrige sesgo) | 0.000 | −0.192 | 0.000 |
| H5 ventana corta 2 años | +0.076 | −0.212 | +0.105 |
| H6 **always_short** (baseline tonto) | 0 | 0 | 0 |
| H7 momentum (control sin modelo) | −0.151 | −0.173 | +0.105 |

N por año: 2024=53, 2025=52, **2026=19** (bajo el umbral de 20 → no concluir de 2026).

**Ninguna hipótesis bate al baseline tonto en 2025.** Ese año tuvo 67% de semanas bajistas;
"siempre SHORT" acierta 67.3% y ningún modelo se acerca. Con 9 trials, cualquier ganador
aparente exigiría deflación por DSR antes de reclamar nada.

**Veredicto: no hay edge direccional.** El modelo no predice mejor que una constante.

## EXP-DIR-002 — el baseline tonto a nivel de ESTRATEGIA

§3.3 exige comparar contra "siempre-short 1× con la misma mecánica de salidas". Mismo
regime gate, mismos TP/HS, mismo sizing; solo cambia la fuente de dirección.

| Año | Variante | Retorno | maxDD | **Calmar** | PF | Trades | WR |
|---|---|---|---|---|---|---|---|
| 2025 | modelo | +26.05% | 3.50% | **7.44** | 3.30 | 31 | 77.4% |
| 2025 | always-short | **+28.20%** | 5.25% | 5.37 | 2.13 | 43 | 76.7% |
| 2026 | modelo | −0.30% | 1.16% | **−0.26** | 0.74 | 5 | 40.0% |
| 2026 | always-short | −2.28% | 4.55% | −0.50 | 0.64 | 8 | 50.0% |

**El baseline tonto gana en retorno bruto 2025 (+28.20% vs +26.05%).**
**El modelo gana en Calmar — la métrica primaria de graduación (§2) — en ambos años.**

## Conclusión

El modelo **no es un predictor de dirección; es un filtro de operaciones**. Opera 31 veces
en vez de 43 y evita las peores semanas: sacrifica 2.15 pp de retorno bruto a cambio de
bajar el drawdown de 5.25% a 3.50% y subir el profit factor de 2.13 a 3.30.

Por la métrica que la constitución define como primaria, **el modelo se justifica**. Por
precisión direccional, **no**. Ambas cosas son ciertas y el sistema debe describirse así:
el alpha vive en la selectividad y en la mecánica de salidas, no en adivinar el signo.

## EXP-DIR-004 — la historia macro que el modelo NO usaba

**EXP-DIR-003 se equivocó.** Concluyó "falta el dato" tras mirar los seeds de 5 minutos.
La tabla macro diaria tiene **11 series con historia 2020-2026 completa** que el modelo H5
nunca consume — incluidos los pares LatAm cuya ausencia yo había culpado del callejón sin
salida. Corregido y re-testeado.

Features construidas (todas `shift(1)`, merge backward): `mxn_ret5`, `clp_ret5`,
`peer_basket`, `cop_vs_peers`, `coffee_ret5`, `brent_ret5`, `colcap_ret5`, `col10y_chg`,
`col_curve`.

`alpha = DA − DA(always_short)`. 6 conjuntos × 2 familias de modelo × 3 años = **12 celdas**.

| Conjunto | Modelo | 2024 α | 2025 α | 2026 α |
|---|---|---|---|---|
| K1 peers (MXN+CLP) | reg | +0.019 | −0.154 | +0.105 |
| K1 peers | **clf** | +0.019 | **−0.039** | +0.053 |
| K2 peer_spread | reg | +0.038 | −0.135 | +0.053 |
| K2 peer_spread | clf | +0.019 | −0.058 | +0.053 |
| K3 terms_trade (café+brent) | reg | 0.000 | −0.115 | +0.053 |
| K3 terms_trade | clf | −0.019 | −0.058 | +0.105 |
| K4 local_risk (COLCAP+curva) | reg | +0.019 | −0.154 | **+0.158** |
| K4 local_risk | clf | 0.000 | −0.077 | +0.105 |
| K5 all_new | reg | +0.019 | −0.154 | +0.053 |
| K5 all_new | **clf** | −0.019 | **−0.039** | +0.053 |
| K6 baseline (25 feats) | reg | +0.019 | −0.135 | +0.053 |
| K6 baseline | clf | +0.038 | −0.096 | +0.105 |

**Ninguna de las 12 celdas bate al baseline tonto en 2025.** N: 2024=53, 2025=52, 2026=**19**.

### Juicio trial-aware (§2)

21 trials acumulados (9 de DIR-001/002 + 12 de DIR-004). Pasando la mejor celda por
`services/common/metrics.py::deflated_sharpe_ratio`:

| | DA 2025 | SR/periodo | **DSR** | ¿Supera 0.95? |
|---|---|---|---|---|
| Mejor candidato (K1/K5-clf) | 0.6346 | +0.2795 | **0.9006** | **No** |
| Baseline always_short | 0.6731 | +0.3690 | **0.9703** | **Sí** |

**El baseline tonto pasa el bar constitucional; ningún modelo lo hace.**

### Veredicto de DIR-004

La dirección semanal de USD/COP **no es predecible** con este conjunto de información —
que ahora incluye pares LatAm, términos de intercambio, riesgo local y curva soberana, no
solo precio y técnicos. Con 21 trials, seguir buscando es minería de datos, no investigación.

## EXP-DIR-005 — ¿hay skill DONDE el modelo se compromete?

Todo lo anterior midió precisión **incondicional**: se califica al modelo en las 52 semanas.
Pero el sistema no opera 52 semanas — opera cuando el gate abre y la confianza alcanza
(31 de 52 en 2025). La pregunta operativamente relevante es si acierta **en las semanas que
elige**, no en todas.

DA por convicción (|predicción|), walk-forward estricto, 124 semanas pooled:

| Corte | N | DA modelo | DA always-short | alpha |
|---|---|---|---|---|
| top 15% | 18 | 0.6111 | 0.5000 | **+0.111** |
| top 25% | 31 | 0.5806 | 0.5484 | +0.032 |
| top 40% | 49 | 0.5918 | 0.5510 | +0.041 |
| top 60% | 74 | 0.6081 | 0.5811 | +0.027 |

**El alpha es positivo en los cuatro cortes.** Es el primer resultado favorable de toda la
serie: el modelo sí discrimina donde se compromete, y eso explica *mecánicamente* por qué la
selectividad mejora el Calmar (EXP-DIR-002).

Pero no alcanza para reclamar edge:

- **Por quintiles no es monótono** — Q5 da +0.12 pero Q2 da **−0.24**. Si la convicción
  midiera señal, la relación sería creciente; aquí es ruidosa.
- **Por año, 2025 sigue perdiendo**: top-25% da 0.6154 contra 0.6731 del baseline. El
  resultado pooled lo sostiene 2024 (+0.087), no el año objetivo.
- **N=18-31 por celda**, en el límite del umbral de 20 (§6).

### Veredicto de DIR-005

Hay un efecto de convicción **débil y no monótono**, suficiente para explicar por qué el
filtro funciona, insuficiente para llamarlo forecasting direccional. La descripción honesta
del sistema no cambia: **filtro de operaciones con un sesgo direccional marginal**.

## EXP-DIR-006 — sensibilidad al horizonte (y por qué se detuvo la búsqueda)

Último eje genuinamente distinto: todo lo anterior predijo 5 días. Si la dirección semanal
es ruido pero otra ventana no, eso es una pregunta sobre el mercado, no otro hiperparámetro.
Mismas 25 features, misma disciplina walk-forward; solo cambia el target.

| Horizonte | 2024 α | 2025 α | 2026 α |
|---|---|---|---|
| **1 día** | −0.094 | **+0.212** | −0.100 |
| 2 días | 0.000 | −0.019 | +0.050 |
| 5 días (producción) | +0.019 | −0.135 | +0.053 |
| 10 días | +0.094 | −0.038 | −0.222 |
| 20 días | +0.057 | −0.077 | −0.250 |

**El horizonte de 1 día da +0.212 en 2025** — el mejor número de toda la serie. Y es
precisamente por eso que **no se reporta como hallazgo**: es −0.094 en 2024 y −0.100 en 2026.
Positivo en exactamente un año, negativo en los dos adyacentes. Esa es la firma del
sobreajuste, no del edge.

### Por qué se detiene aquí

27 trials acumulados. La celda h1/2025 demuestra el punto de §1 mejor que cualquier
argumento: **si sigues probando combinaciones, encontrarás una ganadora espuria**. Cada
intento adicional además deteriora el DSR de cualquier claim futuro sobre v11.

La búsqueda queda cerrada por decisión metodológica, no por agotamiento de ideas. Reabrirla
exige **información nueva** (corpus de noticias con historia, datos de flujo/posicionamiento),
no más combinaciones de lo que ya hay.

## EXP-DIR-003 — inventario de fuentes (parcialmente equivocado, ver DIR-004)

Re-tunear el mismo conjunto de features estaba agotado. La pregunta correcta pasó a ser si
existe **información nueva** que el modelo H5 no consume. Inventario medido:

| Fuente | Disponible | ¿Viable para backtest multi-año? |
|---|---|---|
| News Engine (~60 features/día por diseño) | **2 snapshots, 28 artículos** | **No** |
| Pares LatAm (MXN, BRL) — `latam_basket_z`, `cop_vs_peers_z` existen en el registry | **2.3K filas, 2026-03 → 2026-07** | **No** (4 meses) |
| Macro (DXY, VIX, EMBI, WTI, UST) | 1954 → 2026 | Ya está en las 25 features |

`CLAUDE.md` afirmaba que los seeds de MXN/BRL tenían "95K filas, 2020-01 → 2026-01".
La realidad medida son **2.279 y 2.310 filas desde 2026-03-16**. Corregido en `CLAUDE.md`.

**Conclusión de DIR-003**: no es que falte esfuerzo de modelado — **falta el dato**. Las dos
fuentes que podrían aportar señal direccional (contexto de noticias y valor relativo contra
pares LatAm) no tienen historia suficiente para entrenar ni para validar.

## La frontera real

Para que exista un forecasting direccional evaluable hay que **adquirir datos primero**:

1. **Backfill de MXN/BRL/CLP a 2020** — el código de features cruzadas ya existe
   (`calculate_cross_pair_lead`, `latam_basket_z`, `cop_vs_peers_z`); solo le falta historia.
   Es el desbloqueo más barato y el de hipótesis económica más sólida (valor relativo EM).
2. **Historia de news con `published_at`** — el motor produce features, pero 28 artículos no
   entrenan nada. Requiere backfill de corpus, no más modelado.

Hasta entonces, cualquier intento de "mejorar la dirección" es re-tunear 25 features sobre
1.637 filas, que es exactamente el sobreajuste que §1 prohíbe.

## Implicaciones

- **No perseguir DA con más tuning sobre 2025** — sería el grid-search sobre OOS que §1
  prohíbe, y EXP-DIR-001 ya muestra que el techo está en el ruido.
- **H3 (clasificador) queda registrado como hipótesis viva**: único enfoque con modelo
  positivo en 2024 y 2026. Su juez es el forward, no un re-test sobre 2025.
- **Reportar Calmar como métrica principal en el dashboard**, no el retorno bruto: es lo
  que separa al modelo del baseline y es lo que la constitución manda.
- Estos 9 trials entran al conteo de DSR de cualquier claim futuro sobre v11.
