---
kind: roadmap
status: PARTIAL
version: 1.1.1
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - tests/regression/test_strategy_manifests.py
  - .github/workflows/fabric-contracts.yml
  - tests/unit/test_codex_safety_contracts.py
---

# BL-16 — CI constitucional Etapa 0 (legalidad + serialización canónica)

**Fuente**: FABRIC §8.3, §11.2, §28 E0 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/governance/declaration.py`, `src/identity/canonical.py` y la migración 070 implementan la matriz de legalidad y la serialización canónica. La mutación independiente PAPER+FULL demostró que la matriz Python muerde. La misma matriz 26/96 está reimplementada por CHECKs SQL, sin prueba de paridad con Python. No existe todavía una declaración real en manifests/config que consuma `research_state` y `capital_tier`, por lo que el gate constitucional end-to-end sigue sin ser falsable.

## Qué falta exactamente
Integrar al menos una declaración real, hacer que el gate lea manifests/config y rechace una combinación inválida antes de ejecutar un DAG. La matriz Python↔SQL necesita un oráculo único o una prueba exhaustiva de paridad. La tanda amplia de CI continúa diferida por orden del operador.

## Incremento CI verificado (2026-08-03)

El job `python-contracts` ejecuta ahora los node IDs exactos de la matriz constitucional,
la construcción fail-closed de declaraciones y el rechazo canónico de `NaN`, `Infinity` y
`-Infinity`. El alcance deliberadamente no incluye toda la suite FABRIC: su test de digest
permanece rojo mientras el plan `fabric-v1` no tenga autorización independiente.

El candado de wiring falló antes del cambio porque el paso BL-16 no existía (**1F/3P**). Tras
añadirlo, la verificación focal terminó en **6P** y el monitor de layout en **20P**. El monitor
de manifiestos conservó su baseline conocido de **20P/4F**, sin delta. Este incremento no
satisface DONE: todavía no hay caller productivo ni paridad ejecutada contra PostgreSQL.

Cross-review `CLD-320` encontró que el primer candado protegía la matriz y los no-finitos, pero
no el node ID que impide construir una declaración ilegal. R2 añadió esa tercera aserción. La
mutación independiente —retirar sólo `test_illegal_governance_object_cannot_exist` del
workflow— pasó indebidamente antes de R2 y ahora produce **1F** nombrando el gate ausente; el
workflow fue restaurado antes del verde final.

## Impacto frontend
Ninguno.

## Dependencias
BL-13.

## Verificación
Declaración PAPER+FULL ⇒ CI rojo; JSON con NaN ⇒ rojo.

## Notas constitución
96 combinaciones nominales, la mayoría absurdas — la matriz es la defensa.

## Paridad contra PostgreSQL aplicado (2026-08-04)

Tras aplicar `fabric-v1`, la matriz Python se contrastó con los `CHECK` reales de
`control.strategy_declaration`. La integración obtiene las expresiones vivas mediante
`pg_get_expr` y las evalúa por `SELECT` sobre cada candidato, sin insertar filas, crear objetos
temporales ni desactivar los triggers de transición. La paridad verificada está acotada
explícitamente a `surface='action'`: cubre las dimensiones nominales de estado de investigación,
tier de capital y estado operacional, además de `dag_declared` y el checklist exigido al retirar.

El cross-review `CLD-367` comprobó que `surface='diagnostic'` expone divergencias reales en el
`CHECK` de `control.strategy_declaration`, porque `GovernanceDeclaration` no modela `surface`.
Resolver si Python debe incorporar esa dimensión o si el `CHECK` SQL debe cambiar es una decisión
de contrato pendiente; el test no oculta ni reclama cubrir esa superficie.

BL-16 permanece **PARTIAL** por esa brecha y porque ningún manifest/config productivo construye
todavía una `GovernanceDeclaration` y la usa como gate antes de ejecutar su DAG. La paridad acotada
de dos mecanismos inertes no sustituye ese caller end-to-end.
