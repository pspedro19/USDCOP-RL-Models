---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/pipeline/run_spx500_pipeline.py
  - scripts/pipeline/run_gold_pipeline.py
  - scripts/pipeline/run_btc_pipeline.py
  - scripts/pipeline/train_and_export_smart_simple.py
---

# BL-47 — Migración de estrategias al motor de políticas (R6-R8)

**Fuente**: planes/05-rule-based-strategies.md §12-§13 R6-R8 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Avance 2026-07-28 (R6 + R7 con paridad medida; R8 SPEC_ONLY)

Specs de política en `config/policies/*.yaml` (loader/factory `src/strategies/policies/`,
validador §11 `scripts/validation/validate_policy_specs.py`, arnés de paridad
`scripts/validation/check_policy_parity.py`, tests `tests/unit/test_policy_specs.py`).

| Política | Motor · modo | Paridad vs productor congelado (dato real) |
|---|---|---|
| `spx500_daily_ma200_v1` | rule_based · declarative (DSL) | EXACTA, toda la ventana |
| `gold_trend_simple` | rule_based · coded_policy | EXACTA fuera del calentamiento; divergencia acotada a `bars<252` (declarada) |
| `btc_hodl_b1` | rule_based · coded_policy | EXACTA, toda la ventana |
| `smart_simple_v11` | **composite** · SPEC_ONLY | no migrada — `build_policy()` falla cerrado |

Ningún camino legacy se apagó ni se modificó: el criterio de corte (≥2 semanas verdes,
calendario BL-28/31) sigue pendiente. 0 trials.

**Divergencias declaradas (decisión del operador, NO resueltas aquí)**: dos productores
distintos para `gold_trend_simple` (con y sin multiplicador de régimen); semántica de
calentamiento (NaN como voto negativo vs fallo cerrado); `regime_risk_mult` nunca se
aplica en el pipeline BTC publicado. Detalle en cada spec.

## Estado actual (as-built verificado 2026-07-27)
Publishers actuales por activo (run_spx500/gold/btc_pipeline + publish_gold_dynexit stateful) producen bundles válidos; v11 corre en su cadena artesanal como composite de facto (Ridge/BR + Hurst + sizing + TP/HS).

## Qué falta exactamente
R6: migrar spx500_daily_ma200_v1 primero (la más simple) — paridad semantic_hash de señales/trades/PnL/bundles legacy vs motor nuevo ANTES de apagar el camino viejo. R7: xauusd_trend_simple_v1 y btcusdt_hodl_b1 (dynexit exige el estado de §15.2 resuelto en BL-45). R8: USD/COP al FINAL como engine.type=composite (conserva L3 del predictor; L7 queda para la última etapa del strangler). Clasificación §12 aplicada en todos los specs (PPO = rl).

## Impacto frontend
Cero cambio visual si la paridad es verde (ese es el criterio).

## Dependencias
BL-45, BL-46; se ejecuta DENTRO del calendario de BL-28/31 (mismo patrón strangler: paridad ≥2 semanas por estrategia, rollback declarado).

## Verificación
Diff semántico verde por estrategia; el A/B vivo v11/v12/v14 intocable durante la migración (sus ledgers son el patrón de paridad).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_policy_specs.py -q
verde:   26 passed

muta:    src/strategies/policies/gold.py:62
         `in_market = votes >= min_votes`  ->  `votes > min_votes`
espera:  rojo del arnés de paridad contra el PRODUCTOR CONGELADO REAL:
         "865/5610 barras divergen, primera idx=9, legacy=0.6449 motor=0.0"
```

**Historial honesto**: BL-47 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) — no hubo defecto que cerrar el 2026-07-28. Y muerde por la razón correcta: el
arnés **no compara contra un fixture escrito a mano sino contra el productor congelado real**,
así que el mensaje del fallo no es "un assert falló" sino el conteo exacto de barras
divergentes, la primera posición y los dos valores enfrentados. Un candado de paridad que se
compare contra un fixture propio es el mismo defecto circular que se encontró en BL-13.

**Aviso de CI (CLD-216) — RESUELTO el 2026-08-03, nota corregida el 2026-08-06.** Decía que
`check_policy_parity.py` «no está en ningún workflow» y que la garantía dependía de que alguien lo
ejecutara a mano. **Ya no es cierto**: `fabric-contracts.yml` lo ejecuta con `--ci-eligible` desde
`041cb287`, y bajo esa bandera un `DataUnavailable` es **`[FAIL]`, no `[SKIP]`** — la red ya no
depende de nadie.

Lo que el aviso sí acertó y sigue vigente: la mutación de Gold quedaba verde para
`validate_policy_specs.py`, porque ese validador comprueba la **forma** del spec, no la paridad
numérica. Son dos gates distintos y el segundo es el que faltaba.

**Estado actual del gate, medido**: hoy devuelve `EXIT=0` con «0 specs elegibles — nada
verificado», porque las tres policies construibles están en `PARITY_PENDING` tras las demociones de
identidad. Ese cero es **por gobierno**, y `c97e70f3` lo separó del cero **por rotura** —registro de
policies o de arneses vacío ⇒ `EXIT=1`— para que una lista vacía no pueda pasar por vacuidad.

## Por qué BL-47 es TIME_GATED y no «pendiente de trabajo»

**Hoy no hay ningún slice de código DESBLOQUEADO**, y ése es el punto — no que no quede
implementación. R6 y R7 exigen **≥2 semanas de paridad por estrategia** dentro del calendario de
BL-28/31 antes de apagar el camino viejo, y las tres policies están en `PARITY_PENDING` esperando
una **re-promoción que es acto exclusivo del operador**. Abrir código antes de eso sería saltarse
el calendario, que es justo lo que el patrón strangler existe para impedir.

Los slices de SPX, BTC y Gold (2026-08-06) dejaron las policies **listas para** ese paralelo
—identidad congelada, productores declarados, cadena atravesable de punta a punta—: eso es la
**precondición**, no el trabajo restante.

**Y sí queda trabajo de código, después.** El orden es: (1) tiempo de observación, (2) decisión del
operador, y **sólo entonces** (3) se habilitan los slices que hoy están bloqueados —el **corte** de
los caminos legacy que R6/R7 declaran, y **R8**, que sigue `SPEC_ONLY` y exige migrar USD/COP como
`engine.type=composite`—.

> **Corrección de la redacción anterior (2026-08-06, R2 tras CXD-644).** Esta sección decía «no
> falta implementación» y cerraba con «lo que falta es tiempo […], no líneas». Es **falso como
> absoluto**, y la propia ficha lo desmiente dos secciones más arriba: R8 sigue `SPEC_ONLY` y el
> apagado del legacy es trabajo real. Lo que quise decir —y lo único que se sostiene— es que **hoy
> nada de eso está desbloqueado**. Convertir «no hay slice abierto ahora» en «no falta
> implementación» es exactamente el tipo de salto de una afirmación medida a una absoluta que este
> repo lleva corrigiendo.

## Notas constitución
v11 FROZEN: migrar su cáscara a composite NO toca fórmula ni señal (re-freeze consciente de manifiesto, 0 trials, bit-check obligatorio).
