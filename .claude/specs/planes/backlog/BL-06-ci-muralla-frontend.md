---
kind: roadmap
status: IMPLEMENTED
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
  - usdcop-trading-dashboard/app/api/data/[...path]/route.ts
  - tests/regression/test_forecasting_caveat_present.py
---

# BL-06 — CI muralla frontend: forecasting sin aprobar/ejecutar

**Fuente**: plan 01 §4 / FABRIC §25 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Verificado limpio HOY: ForecastingView sin botones aprobar ni endpoints de ejecución (único link: /pricing). Sin test que lo candadee.

## Qué falta exactamente
Test estático: en ForecastingView (y components/forecasting/*) prohibidos `\/api\/production\/approve`, `\/api\/execution`, `onApprove`, verbos COMPRAR/VENDER.

## Impacto frontend
Ninguno — candado.

## Dependencias
—

## Verificación
Test rojo al inyectar un fetch de approve en la vista.

## Implementación (cierre 2026-07-28)

**Estado: IMPLEMENTED.** Cross-review de CODEX: **APROBADO** contra `edba615`
(pack inmutable en `.claude/coordination/reviews/BL-06.md`, `7 passed` en el
momento de la aprobación).

El candado vive en `tests/regression/test_forecasting_caveat_present.py`:

| Test | Qué candadea |
|---|---|
| `test_forecasting_has_no_action_capabilities` | prohíbe `api/production/approve`, `/api/execution`, `onApprove` y los verbos de orden `COMPRAR`/`VENDER` en las superficies GM de forecasting |
| `test_legacy_forecasting_same_rules` | aplica las mismas prohibiciones al `ForecastingDashboard` legacy, para que la superficie vieja no sea la puerta de atrás |

Constantes del candado: la lista de endpoints prohibidos y `_ORDER_VERBS`
(`\b(COMPRAR|VENDER)\b`) están declaradas en el propio módulo de test.

Corrida de cierre (2026-07-28T11:1x COT):

```
python -m pytest tests/regression/test_forecasting_caveat_present.py -q
20 passed in 0.89s
```

**Nota honesta sobre el conteo**: el archivo pasó de 7 a 20 tests porque es
COMPARTIDO con BL-01/BL-02/BL-03/BL-04, que siguen en remedio. Los dos tests que
constituyen el candado de BL-06 son los de la tabla y están verdes; la evolución
del resto del archivo pertenece a esos otros BLs y no altera este cierre.

**Retracción registrada**: el duplicado `tests/regression/test_forecasting_muralla.py`
fue retirado en `d7cfd67` (self-red-team detectó redundancia con el candado ya
aprobado). El archivo no existe hoy y el candado aprobado es el único vigente.

## Notas constitución
FABRIC §25 bloque Muralla: 'frontend forecasting sin verbos de orden…'.
La superficie DIAGNOSTIC no gana capacidades de acción por accidente: el día que
forecasting pueda aprobar o ejecutar, este test se pone rojo.
