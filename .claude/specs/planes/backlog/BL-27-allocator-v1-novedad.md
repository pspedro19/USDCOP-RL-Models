---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/analysis/book_construction.py
  - scripts/analysis/book_kelly_governor.py
  - config/book/book_v1.yaml
---

# BL-27 — Allocator v1 (inverse-vol+caps) + multiplicadores + gate de novedad

**Fuente**: FABRIC §20 + §12 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0/+1 por celda si se abre book_allocation

## Estado actual (as-built verificado 2026-07-27)
ERC + Kelly governor DRAFT existen (evidencia 2026-07-22, firma del operador pendiente). Sin multiplicadores, sin gate de novedad, sin optimización restringida.

## Qué falta exactamente
config/book/allocator_v1.yaml (baseline inverse-vol caps; HRP SOLO shadow con juez propio); b_prov=b_base×m_forward(solo-reduce)×m_liq×m_div×m_ops×m_dd(histéresis); cvxpy §20.3 SIN normalize(); fallback 4 peldaños; novelty_gate (ρ<0.60 ∨ ΔIR>0.15) en promociones y como m_div.

## Impacto frontend
Control Tower: pesos propuestos vs realizados (shadow).

## Dependencias
BL-26, BL-22.

## Verificación
Shadow ≥26 periodos vs baseline neto de costos ANTES de mover capital; normalize() prohibido por grep-CI.

## Notas constitución
'Probé 12 esquemas y elegí el mejor' es el mismo pecado un nivel arriba — el allocator tiene familia y juez propios.
