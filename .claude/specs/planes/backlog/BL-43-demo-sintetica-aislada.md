---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/public/data/registry.json
  - scripts/pipeline/normalize_champions.py
---

# BL-43 — Aislar el modelo sintético demo (CI que lo bloquee fuera de demo)

**Fuente**: Plan Consolidado §7 (final) · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Entrega parcial: `src/governance/synthetic_isolation.py` y la migración 081 implementan la frontera demo/sintética y ya viajan en checkout limpio. El test existente de relación real sí mata la eliminación del guard `algorithm`; añadir ese eje al bucle de la relación demo no aportaría esa prueba. El DDL sólo tiene validación estática y la base viva no tiene `demo.*`; `investor_demo` sigue en el camino de compatibilidad.

## Qué falta exactamente
Ejecutar 081 contra PostgreSQL y mover la fila a `demo.*` sin dejarla en superficies de performance reales. La capa visible debe conservar una etiqueta inequívoca mientras exista compatibilidad; CI/Playwright se validarán al final por orden del operador.

## Impacto frontend
Si alguna vista lo muestra, gana badge DEMO inequívoco o desaparece.

## Dependencias
BL-13 (campo surface).

## Verificación
Test rojo con SYNTHETIC+active fuera de demo.

## Notas constitución
Desconfianza de la magia: un equity sintético presentado como real es el peor bug de honestidad posible.
