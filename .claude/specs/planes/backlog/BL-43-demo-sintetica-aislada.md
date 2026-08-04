---
kind: roadmap
status: IMPLEMENTED
version: 1.2.0
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - database/migrations/081_synthetic_demo_isolation.sql
  - src/governance/synthetic_isolation.py
  - services/demo_mode/config.py
  - services/inference_api/routers/backtest.py
---

# BL-43 — Aislar el modelo sintético demo (CI que lo bloquee fuera de demo)

**Fuente**: Plan Consolidado §7 (final) · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-08-04)

Implementado de extremo a extremo. La migración 081 está aplicada: la fila sintética vive en
`demo.synthetic_model`, la vista `demo.synthetic_model_display` conserva la etiqueta demo y las
restricciones impiden mezclarla con modelos reales. El backtest carga esa vista mediante
`load_demo_config()` antes de generar operaciones y falla cerrado si el modelo no está registrado
o viola `validate_model_boundary()`.

## Qué falta exactamente

Nada para este BL. La validación visual integral del dashboard pertenece al gate final del plan,
no al aislamiento del consumidor ya demostrado aquí.

## Impacto frontend
Si alguna vista lo muestra, gana badge DEMO inequívoco o desaparece.

## Dependencias
BL-13 (campo surface).

## Verificación

- PostgreSQL real: registro demo presente, cero filas sintéticas en `config.models` y restricciones
  de aislamiento activas.
- Consumidor productivo: `services/inference_api/routers/backtest.py` consulta
  `demo.synthetic_model_display` mediante `load_demo_config()` antes de generar operaciones.
- Tests focales: 18 verdes entre loader/caller y frontera sintética.
- Mutación causal del consumidor: sustituir la vista demo por `config.models` produce 1 fallo y
  restaura a 5 verdes en el subconjunto afectado.
- Cross-review independiente de Claude en `CLD-433`: modelo registrado carga; modelo ausente falla
  cerrado con `SyntheticIsolationError` contra la base real.

## Notas constitución
Desconfianza de la magia: un equity sintético presentado como real es el peor bug de honestidad posible.

## Cierre de cableado (2026-08-04)

El bloqueo anterior quedó resuelto por la aplicación verificada de 081 y el consumidor real
sellado en `4b056075`. El cierre es bilateral: CODEX implementó y mutó el caller; CLAUDE repitió
la carga y el fallo cerrado contra PostgreSQL antes de aprobar la promoción.
