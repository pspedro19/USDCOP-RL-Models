---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/contracts/forecast_output.py
  - usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts
  - scripts/pipeline/generate_weekly_forecasts.py
  - tests/fixtures/forecast_output_cases.v1.json
---

# BL-15 — Contrato forecast_output (Py+TS) + validación en el zoo

**Fuente**: plan 01 §1 / FABRIC §15.2 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado (as-built 2026-07-28, tras remedio del rechazo CODEX vs `91fe7b6`)

`CTR-FORECAST-OUTPUT-001` existe en Python y TypeScript (espejo, mismo cambio) y el
generador del zoo **no puede publicar** una fila sin registro validado.

| Pieza | Estado |
|---|---|
| `src/contracts/forecast_output.py` | contrato + ingest wall fail-closed |
| `usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts` | espejo con validador runtime propio (no usa `Date.parse`) |
| `tests/fixtures/forecast_output_cases.v1.json` | **tabla de casos ÚNICA** (90 casos, `content_sha256` recomputado por ambos runners) |
| `tests/unit/test_forecast_output_contract.py` | 116 tests (incluye la tabla compartida) |
| `usdcop-trading-dashboard/tests/unit/contracts/forecast-output-parity.test.ts` | **runner Vitest real** (96 tests) sobre la MISMA tabla |
| `tests/unit/test_zoo_generator_contract.py` | gate por fila + muro de publicación (13 tests) |

### Los 5 hallazgos del rechazo

1. **Timezone mixto** — comparar naive vs aware lanzaba `TypeError` crudo fuera del
   contrato. Ahora los tres timestamps deben compartir tz-awareness; mezclarlos es
   `ForecastOutputError` tipado en Python y error de contrato en TS.
   Razón (`.claude/rules/data-governance.md`): COP vive en `America/Bogota`, XAU/BTC son
   TIMESTAMPTZ por instante — mezclar convenciones no tiene orden definido.
2. **Fechas imposibles en TS** — `Date.parse` rodaba `2026-02-30` a `2026-03-02` y aceptaba
   separador con espacio, fecha sola, `z` minúscula, `+0000`. Ambos lados usan ahora la
   MISMA gramática estricta `YYYY-MM-DDTHH:MM:SS[.ffffff][Z|±HH:MM]` con calendario real.
   Paridad demostrada caso a caso sobre la tabla compartida (90/90 mismo veredicto).
3. **El CSV publicado descartaba el contrato** — `write_csv` exige ahora el índice de
   registros validados, re-valida cada uno en el muro, exige `forecast_id`/`contract_id`/
   `prediction_point` iguales a los del registro validado y escribe de forma atómica
   (tmp + `os.replace`): una fila mala ⇒ no se publica nada. Los PNGs y el CSV leen
   `validated.prediction.point`, no el float crudo.
4. **Ingest wall** — `from_dict` dejó de soltar claves desconocidas y de coaccionar
   `diagnostic_only`; `ingest_forecast_output()` / `ingest_forecast_outputs()` (y en TS
   `parseForecastOutput` / `ingestForecastOutputs`) rechazan payloads accionables,
   claves desconocidas y cualquier violación. Antes, un payload con `pnl`/`exit_reason` y
   `diagnostic_only: false` entraba convertido en un forecast inocente.
5. Este MD (estado real, sin DONE prematuro).

## Qué falta (declarado, no simulado)

- **Book/allocator**: el rechazo por tipo está probado contra `StrategyTrade`
  (cero solapamiento de campos + `TypeError`), pero **no existe todavía un
  `book_construction` real** que consuma `strategy_output`; el test de la sección
  «Verificación» original se cumple sólo en su forma disponible hoy.
- **Frontend**: `ForecastingView` sigue leyendo el CSV por su cuenta; migrarlo a
  `parseForecastOutput` queda para BL-19 (los archivos de la vista están bajo lease de
  otro lane).
- **Intervalos reales**: el zoo emite predicción puntual (`lower == upper == point`);
  el contrato admite intervalos pero nadie los produce aún.
- La corrida end-to-end del generador (dataset + 9 modelos) **no se ejecutó** en esta
  sesión por orden del operador (sin Docker/infra); lo verificado es unitario.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   python -m pytest tests/unit/test_forecast_output_contract.py
                             tests/unit/test_zoo_generator_contract.py -q
verde:     145 passed   (132 contrato + 13 gate del zoo)

comando-2: npx vitest run tests/unit/contracts/forecast-output-parity.test.ts
           (desde usdcop-trading-dashboard/)
verde-2:   102 passed   — MISMA tabla de casos compartida
                          (tests/fixtures/forecast_output_cases.v1.json, content_sha256
                           recomputado por los dos runners)

muta:      src/contracts/forecast_output.py — `if not math.isfinite(...)` -> `if False`
espera:    9 failed — casos que SOLO difieren en el valor numérico
           (num_point_nan, num_point_inf, num_lower_neg_inf, …), o sea que muerde por la
           DEFENSA DE FINITUD y no por un `required` ausente

muta-2:    muro de publicación — `raise ForecastOutputError` -> `continue`
espera-2:  2 failed, incluido "write_csv is all or nothing"

muta-3:    lib/contracts/forecast-output.contract.ts — quitar `|| !Number.isFinite(v)`
espera-3:  5 failed sobre la MISMA tabla compartida
```

**Historial honesto**: BL-15 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) — no hubo defecto que cerrar el 2026-07-28. En el veredicto CLD-214 quedó
literalmente como *"esto es lo que yo entiendo por un BL cerrable"*, y la razón es precisa:
los 9 rojos de la primera mutación son casos que **solo se diferencian en el valor numérico**,
lo que descarta que estén cayendo por un campo obligatorio ausente en vez de por la defensa
que dicen proteger — que era exactamente el defecto encontrado el día anterior. Y la paridad
TS **también** muerde sobre la misma tabla de casos, así que romper una de las dos mitades no
puede pasar desapercibido.

## Notas constitución

El allocator sólo acepta `strategy_output` — rechazo físico, no convención.
`diagnostic_only` es literal `true` en ambos lados: un "forecast accionable" no compila
en TS y no se puede construir en Python.

## Slice cerrado: el zoo publicaba un intervalo de ANCHURA CERO (CLAUDE, 2026-08-06, `b087ad91`)

`_validate_row_contract` emitía `lower == upper == point` con el comentario *"no intervals
produced by the zoo (see note)"*. **No es lo mismo**: el contrato acepta `None` en ambos
límites, y *eso* es "sin intervalo". Igualarlos al punto publica un **intervalo degenerado**,
que cualquier consumidor lee como **incertidumbre nula** — de modelos cuya DA ronda 0.46
(BTC price-only, ya declarado en `CLAUDE.md`).

**Por qué ningún muro lo veía**: `lower <= point <= upper` se cumple con igualdad, así que el
contrato lo aprueba; y **la aserción del test lo fijaba como correcto**
(`assert out.prediction.lower == 0.0042`, comentada *"by design"*). Un defecto **con candado a
favor** — la variante más cara de falso verde, porque el candado da confianza en la dirección
equivocada.

**No era bug en producción y no se vendió como tal**: hoy nadie lee esos límites (medido en Py
y TS). Pero BL-19 migra `ForecastingView` a `parseForecastOutput`, y el primer consumidor que
dibuje la banda pintaría una cinta de certeza. Se corrige **antes** de que exista el consumidor.

La aserción se **invierte, no se relaja**: ahora exige ausencia. Candado nuevo que recorre
positivo, negativo, cero y `1e-9`, para que la ausencia no dependa del caso feliz.
**M8**: reintroducir `lower/upper = pred_return` ⇒ 2F/12P.

    python -m pytest tests/unit/test_zoo_generator_contract.py \
                     tests/unit/test_forecast_output_contract.py -q   -> 146 passed

### Decisión contractual (CODEX, CXD-601) — el contrato global NO cambia

Planteé si `CTR-FORECAST-OUTPUT-001` debería rechazar intervalos degenerados globalmente.
**Respuesta razonada: no.** Un intervalo de anchura cero puede ser semántica legítima para un
predictor determinista o un *bound* exacto; el contrato genérico no conoce la capacidad del
productor. La ofensa era **local**: el zoo declara "point-only" y emitía bounds. La frontera
correcta es el candado en el **productor**. No se abre C-NNN ni espejo TS por este hallazgo.

**BL-15 sigue `PARTIAL`** por decisión explícita de CXD-601: el slice está aprobado, pero la
ficha conserva pendientes que este cambio no toca — `book_construction` real que consuma
`strategy_output`, migración del frontend (BL-19, otro lane) y la corrida end-to-end del
generador (necesita stack).
