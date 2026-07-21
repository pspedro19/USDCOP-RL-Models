---
kind: as-built
status: PARTIAL
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - scripts/analysis/gold_dynamic_exit.py
  - scripts/pipeline/publish_gold_dynexit.py
  - src/gold_rl/backtest.py
# Conteo de trials LEGIBLE POR MÁQUINA.
n_trials_total: 75
n_trials_scenarios: [21, 75, 91]   # suelo publicado / programa declarado / +descartados
n_trials_sources:
  - "scripts/pipeline/publish_gold_dynexit.py:48 TRIALS_PROGRAM = 74 (número heredado)"
  - "public/data/strategies/gold_*/backtests/* = 21 bundles publicados (suelo verificable)"
  - "H-SIMP-GOLD-01 (registrada 2026-07-21, prospectiva): +1"
sigma_trials: null
sigma_trials_grid: [0.05, 0.10, 0.15]   # titular = el DSR MÍNIMO de la rejilla
---

# HYPOTHESIS-REGISTRY — XAU/USD (reconstruido)

> **Creado 2026-07-20.** Gold era el único track con estrategia publicada y **sin registro de
> trials**: su conteo vivía hardcodeado en `publish_gold_dynexit.py:48` (`TRIALS_PROGRAM = 74`)
> sin ninguna fuente que lo justificara. Un DSR calculado contra un número que nadie puede
> auditar no es un DSR — es una cifra decorativa. Este archivo lo ancla.
>
> Regla que aplica (`.claude/rules/quant-constitution.md`): cada versión, cada celda de grid y
> cada gate mirado = **1 trial**; ningún claim de edge sin DSR recomputado con el conteo
> actualizado; **elegir la mejor celda está prohibido**.

---

## 1. Suelo verificable — bundles realmente publicados

Cada versión publicada implica que su resultado se miró ⇒ ≥ 1 trial. Este es el **piso**, no
el total: los trials descartados antes de publicar no dejan bundle.

| Estrategia | Versiones publicadas |
|---|---|
| `gold_dxy_tilt` | 2 |
| `gold_dxy_tilt_s05` | 2 |
| `gold_dxy_tilt_s07` | 2 |
| `gold_dynamic_exit` | 1 |
| `gold_long_only_b1` | 4 |
| `gold_regime_gated_v1` | 4 |
| `gold_trend_b2` | 4 |
| `gold_trend_ens` | 2 |
| **Suelo** | **21** |

Los sufijos `_s05` / `_s07` son, por su propio nombre, **celdas de sensibilidad publicadas como
estrategias separadas**. Eso es exactamente lo que la constitución cuenta como trials
adicionales, y explica por qué el total declarado (74) está muy por encima del suelo.

## 2. El número heredado: 74

`TRIALS_PROGRAM = 74` no tiene procedencia documentada. Se conserva porque **es el más
conservador de los disponibles** (74 > 21 ⇒ deflacta más ⇒ es el más exigente con la
estrategia). Bajarlo requeriría evidencia; subirlo, no.

**Consecuencia ya medida**: con N=74, `gold_dynamic_exit` da **DSR = 0.0077** contra un bar de
0.95. No es un fallo marginal — está dos órdenes de magnitud por debajo. El +55% de retorno se
explica por selección sobre 74 intentos, no por señal.

## 3. σ entre trials: nunca se persistió

La dispersión de Sharpe entre trials es un insumo del DSR y **no se guardó en ninguna corrida**.
Mientras no exista, se evalúa sobre la rejilla `sigma_trials_grid` y **el titular es el DSR
mínimo**, nunca la celda amable. Es el mismo patrón ya usado en COP (`cop_trials_dsr.py`).

## 4. Prospectivo

Toda hipótesis nueva sobre XAU/USD se registra **aquí y antes** de correr su test, con H0/H1 y
criterio de decisión pre-registrado, e incrementa `n_trials_total` en el front-matter. El
harness `scripts/analysis/profitability_evidence.py` compara el `params_hash` de cada corrida
contra el anterior y **exige el incremento** cuando cambia.

---

## 5. Hipótesis prospectivas (registradas ANTES de correr)

### H-SIMP-GOLD-01 — la maquinaria de salida destruye la tendencia

**Registrada**: 2026-07-21, antes de implementar `gold_trend_simple`.

**Motivación**: la medición de evidencia del 2026-07-20 mostró que el baseline tonto —el mismo
voto SMA 63/126/252, siempre encendido, **sin trailing exit**— rindió 2819.75% con Calmar 0.908
frente al 5.80% y Calmar 0.0045 de `gold_dynamic_exit`. El capture ratio lo confirma desde otro
ángulo: la estrategia captura **más bajada (45.6%) que subida (41.3%)**, ratio 0.906. El exit
no es inútil: está invertido.

- **H0**: `Calmar(gold_trend_simple) ≤ Calmar(gold_dynamic_exit)` — el trailing exit aporta.
- **H1**: `Calmar(gold_trend_simple) > Calmar(gold_dynamic_exit)` — el exit resta.
- **Estadístico**: ΔCalmar por block bootstrap pareado (bloque 20d, 252/año, 5000 muestras).
- **Criterio de decisión**: IC95 excluye cero **en el forward**, no en 2004-2026.
- **Costos**: idénticos a `gold_dynamic_exit` (2 bps + swap), y stress ×2/×3.

> **El juez es el forward.** La observación que motiva esta hipótesis se hizo sobre 2004-2026,
> así que ese periodo **ya no puede evaluarla** (`quant-constitution.md` §1). El backtest
> histórico se publica como contexto y queda explícitamente marcado como NO-evidencia para
> H-SIMP-GOLD-01.

**Coste en trials**: +1. `n_trials_total` 74 → 75.
