---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/forecasting/ForecastingDashboard.tsx
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-04 — Unificar caveat duplicado en legacy ForecastingDashboard

**Fuente**: hallazgo Explore 2026-07-27 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
`ForecastingDashboard.tsx:123,:132` duplica el banner (texto casi igual) y sigue alcanzable vía `/legacy/forecasting`. Dos implementaciones = deriva garantizada.

## Qué falta exactamente
Extraer el texto del caveat a una constante compartida (lib/) consumida por ambas vistas, o congelar el legacy con nota de superseded.

## Impacto frontend
`/legacy/forecasting` (admin-only) queda alineado o congelado.

## Dependencias
BL-01.

## Verificación
Una sola fuente del string; test BL-01 cubre ambas rutas o el legacy queda excluido explícitamente.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_forecasting_caveat_present.py -q
verde:   31 passed

muta:    components/forecasting/ForecastingDashboard.tsx — re-inlinear el copy del caveat
         a mano, imitando el banner compartido (texto idéntico, componente ausente)
espera:  2 failed — el candado exige el MONTAJE del componente compartido
         (<ForecastDisclaimer/>, envuelto en DiagnosticCaveat en la ruta legacy),
         no el texto:
         "displays Direction Accuracy but the shared caveat (…) is gone"
         "headline AND body must come from lib/ui/forecast-disclaimer.ts
          (BL-04: no hardcoded copy)"
```

**Historial honesto**: BL-04 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`, CLD-212) — **no hubo defecto que cerrar el 2026-07-28**. La razón por la que es
fiable es precisamente la que lo distingue de un candado de subcadenas: la mutación
re-inlinea el copy **correcto**, así que cualquier comprobación de texto la daría por buena;
lo que se exige es el **montaje del componente compartido** y que headline Y body vengan del
SSOT (`lib/ui/forecast-disclaimer.ts`), que es la única forma de que "una sola fuente del
string" sea una garantía y no una intención. Complemento de CXD-032: el componente compartido
no puede traer su propio mecanismo de ocultación (`return null`, `hidden`, `aria-hidden`,
`display:none`) ni vivir bajo un `{cond && …}` — se comprueba la profundidad de llaves del
`data-testid` con comentarios y strings enmascarados.

## Notas constitución
Regla 6 FABRIC: toda métrica/mensaje de gobierno con definición única.
