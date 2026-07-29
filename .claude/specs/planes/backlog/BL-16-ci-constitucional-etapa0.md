---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - tests/regression/test_strategy_manifests.py
---

# BL-16 — CI constitucional Etapa 0 (legalidad + serialización canónica)

**Fuente**: FABRIC §8.3, §11.2, §28 E0 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/governance/declaration.py`, `src/identity/canonical.py` y la migración 070 implementan la matriz de legalidad y la serialización canónica. La mutación independiente PAPER+FULL demostró que la matriz muerde. No existe todavía una declaración real en manifests/config que consuma `research_state` y `capital_tier`, por lo que el gate constitucional end-to-end sigue sin ser falsable.

## Qué falta exactamente
Integrar al menos una declaración real, hacer que el gate lea manifests/config y rechace una combinación inválida antes de ejecutar un DAG. Falta también un candado propio para NaN/Inf dentro del lote constitucional y su invocación en CI; la tanda amplia de CI está diferida por orden del operador.

## Impacto frontend
Ninguno.

## Dependencias
BL-13.

## Verificación
Declaración PAPER+FULL ⇒ CI rojo; JSON con NaN ⇒ rojo.

## Notas constitución
96 combinaciones nominales, la mayoría absurdas — la matriz es la defensa.
