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
n_trials_total: 14
n_trials_scenarios: [12, 14, 20]
n_trials_sources:
  - "_config_family(): 4 vol targets x 3 MA windows = 12 celdas, todas evaluadas"
  - "B1/B2/S3: 3 variantes de política declaradas en el patrón"
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
