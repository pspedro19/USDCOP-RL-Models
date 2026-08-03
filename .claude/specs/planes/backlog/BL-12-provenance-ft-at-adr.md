---
kind: roadmap
status: IMPLEMENTED
version: 1.1.0
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - .claude/rules/quant-constitution.md
---

# BL-12 — Provenance FT→AT + enmienda constitución §2 (ADR)

**Fuente**: plan 02 §3-§4 / FABRIC §10.1-§10.2 · **Ola**: 2 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La práctica ya se siguió (H-META-01 cobró AT al convertir consenso en sizing) pero la regla no está escrita como dura, ni existe el campo provenance.

## Qué falta exactamente
ADR menor + regla en quant-constitution §2: 'convertir un forecast en señal económica = +1 AT; los FT del predictor viajan en provenance y entran al N_cluster'. Campo `provenance:` en pre-registros nuevos.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
Regla publicada; próximo pre-registro la usa.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_bl09_bl11_bl12_governance.py -q
verde:   34 passed

muta:    scripts/validation/check_trial_ledger.py — check_provenance_wall neutralizada
espera:  4 failed — la muralla FT->AT cae por CUATRO aristas distintas, cada una en su
         propio test:
           familia sin bloque provenance
           heredar forecasts mientras se afirma NO cruzar la muralla
           provenance tomada de otro cluster
           citar un trial de acción (AT-) como si fuera un forecast (FT-)

muta-2:  borrar forecast_trial_ids de una familia (mutación de DATOS, no de código)
espera-2: 1 failed — test_ft_to_at_wall_has_a_real_provenance_case
```

**Historial honesto**: BL-12 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) y, en palabras del propio veredicto CLD-212, **"el único de los cinco de
gobernanza que no tiene fisuras"** — no hubo defecto que cerrar el 2026-07-28. Lo que lo hace
fiable es que aguantó las dos mutaciones **por vías independientes**: la de CÓDIGO
(neutralizar el validador) y la de DATOS (vaciar el campo en una familia). Un candado que
solo cae por una de las dos protege el validador o protege el dato, pero no la garantía. El
mismo muro se aplica en `tests/regression/test_strategy_manifests.py:439`, trasladado del
directorio de familias al manifiesto.

## Notas constitución
Cambiar la constitución requiere ADR (su propia cabecera lo exige).


## Cierre (2026-08-03, cross-review CXD-191 + CXD-197)

**PARTIAL -> IMPLEMENTED.** Promovido por el dueño (CLAUDE) tras aprobación explícita de CODEX,
con **los dos ejes de mutación que la propia ficha exige, ejecutados por agentes distintos**.

- **Verde en árbol limpio**: `python -m pytest tests/regression/test_bl09_bl11_bl12_governance.py -q`
  = **34 passed**, exactamente el número que la ficha declara.
- **Eje CÓDIGO — mutación ejecutada por CODEX** (no por el dueño): neutralizar
  `check_provenance_wall` en `scripts/validation/check_trial_ledger.py` produjo **4F/30P**, una
  por cada arista independiente de la muralla FT→AT. Restauración byte-exacta
  `sha256 = EFA0984A...FE2AAC`. Veredicto `CXD-191`: **APROBADA para PARTIAL->IMPLEMENTED**.
- **Eje DATOS — mutación ejecutada por CLAUDE**: vaciar `forecast_trial_ids` en
  `registries/families/vol_sizing.yaml` (`[FT-0050, FT-0051]` -> `[]`) produjo **1F/33P**, y el
  rojo fue el test nominal que la ficha predice: `test_ft_to_at_wall_has_a_real_provenance_case`.
  Restauración byte-exacta verificada por hash: `sha256[:16] = 77D854575D50D766` **idéntico antes
  y después**, `git status --porcelain` de la ruta = `[]`, y la suite vuelve a **34 passed**.
- **Por qué importa que sean dos ejes y dos agentes**: la ficha ya advertía que un candado que
  solo cae por una vía protege el validador **o** protege el dato, no la garantía. Hasta hoy solo
  estaba ejecutado el eje de código, y por el mismo agente que juzgaba. Ahora cada eje tiene un
  ejecutor distinto, que es la forma en que este protocolo convierte una afirmación en evidencia.
- **Nota de entorno (no afecta al veredicto, sí a la reproducibilidad)**: este gate no colectaba
  en la máquina del operador — `services/common/__init__` importa `psycopg2`, ausente. Se instaló
  `psycopg2-binary==2.9.9` (driver, no numérico): **numpy 2.4.6 / pandas 3.0.3 idénticos antes y
  después**. Sin eso, las 34 pruebas no eran ejecutables por nadie.
