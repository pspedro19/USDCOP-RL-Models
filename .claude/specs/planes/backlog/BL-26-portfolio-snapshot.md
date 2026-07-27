---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/analysis/book_construction.py
  - config/book/book_v1.yaml
---

# BL-26 — portfolio_snapshot: barrera temporal del libro

**Fuente**: FABRIC §14.1 · **Ola**: 5 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
book_construction usa pesos ERC estáticos sobre trades históricos; no existe snapshot con cutoff ni políticas de faltante.

## Qué falta exactamente
Contrato + builder: cutoff explícito, accepted/stale/missing, max_age por sleeve, política de faltante declarada (USE_LAST_VALID sin max_age PROHIBIDO).

## Impacto frontend
Control Tower muestra el snapshot vigente.

## Dependencias
BL-15 (tipos), BL-17.

## Verificación
Libro con señal COP de hoy + Oro de ayer ⇒ rechazado sin políticas declaradas.

## Notas constitución
'La señal de hoy de SPX + la de ayer de Oro no es un libro, es una foto movida'.
