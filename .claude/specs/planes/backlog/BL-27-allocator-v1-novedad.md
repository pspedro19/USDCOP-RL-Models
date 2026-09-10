---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-06
supersedes: []
code_anchors:
  - src/portfolio/allocator.py
  - tests/unit/test_allocator_config.py
  - config/book/allocator_v1.yaml
  - scripts/analysis/book_construction.py
  - scripts/analysis/book_kelly_governor.py
  - config/book/book_v1.yaml
---

# BL-27 — Allocator v1 (inverse-vol+caps) + multiplicadores + gate de novedad

**Fuente**: FABRIC §20 + §12 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0/+1 por celda si se abre book_allocation

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/portfolio/allocator.py` contiene inverse-vol, multiplicadores, optimización restringida, cuatro niveles de fallback y la función `novelty_gate`. `AllocatorV1.from_config` carga de forma fail-closed `config/book/allocator_v1.yaml`: solver, gross cap, target vol, turnover, relajación, rangos de multiplicadores y umbrales de novedad vienen del SSOT; `sleeve_caps` y `asset_caps` siguen siendo argumentos obligatorios porque ningún config los declara. Una instancia configurada rechaza overrides divergentes, mientras el constructor directo conserva el camino explícito anterior. El target-zero final sigue siendo deliberadamente el kill-path: devuelve todo cero y emite `ALLOCATOR_FALLBACK_4_TARGET_ZERO/CRITICAL`; pasarlo por `_validated_solution` podría lanzar cuando más se necesita aplanar.

Este enlace SSOT **no tiene consumidor productivo**: añadir `from_config` cierra la duplicación de autoridad y queda fijado causalmente en `tests/unit/test_allocator_config.py`, pero no demuestra que ningún pipeline lo recorra. El allocator continúa sin productor/consumidor real y el gate de novedad no participa en una promoción real.

## Qué falta exactamente
Cablear el allocator configurado y `novelty_gate` al flujo productivo de promoción, preservando el target-zero CRITICAL fuera de la validación que puede lanzar. Falta la corrida shadow de al menos 26 periodos y el candado que prohíbe `normalize()`.

## Impacto frontend
Control Tower: pesos propuestos vs realizados (shadow).

## Dependencias
BL-26, BL-22.

## Verificación
Shadow ≥26 periodos vs baseline neto de costos ANTES de mover capital; normalize() prohibido por grep-CI.

## Notas constitución
'Probé 12 esquemas y elegí el mejor' es el mismo pecado un nivel arriba — el allocator tiene familia y juez propios.

## Bloqueo de cableado medido (2026-08-03)

El allocator debe persistir en `portfolio.allocation` y publicar `portfolio.target`, objetos de
la migración 077. No existen en la base viva; `src/portfolio/allocator.py` tiene tests pero cero
llamadores productivos. Además de la sombra temporal ya declarada, DONE exige un productor y un
consumidor reales después de aplicar 077 con autorización.
