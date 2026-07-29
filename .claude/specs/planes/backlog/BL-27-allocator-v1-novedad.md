---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - scripts/analysis/book_construction.py
  - scripts/analysis/book_kelly_governor.py
  - config/book/book_v1.yaml
---

# BL-27 — Allocator v1 (inverse-vol+caps) + multiplicadores + gate de novedad

**Fuente**: FABRIC §20 + §12 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0/+1 por celda si se abre book_allocation

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/portfolio/allocator.py` contiene inverse-vol, multiplicadores, optimización restringida, cuatro niveles de fallback y la función `novelty_gate`. El allocator no carga todavía el SSOT por un constructor `from_config`, no tiene consumidor productivo y el gate de novedad no participa en una promoción real.

## Qué falta exactamente
Atar el código a `config/book/allocator_v1.yaml`, validar también el target-zero final contra las restricciones/turnover o registrar la excepción crítica de forma explícita, y cablear `novelty_gate` al flujo de promoción. Falta la corrida shadow de al menos 26 periodos y el candado que prohíbe `normalize()`.

## Impacto frontend
Control Tower: pesos propuestos vs realizados (shadow).

## Dependencias
BL-26, BL-22.

## Verificación
Shadow ≥26 periodos vs baseline neto de costos ANTES de mover capital; normalize() prohibido por grep-CI.

## Notas constitución
'Probé 12 esquemas y elegí el mejor' es el mismo pecado un nivel arriba — el allocator tiene familia y juez propios.
