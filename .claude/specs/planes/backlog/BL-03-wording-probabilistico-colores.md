---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-03 — Wording probabilístico y neutralización de verde/rojo en predicciones

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
`DIRECTION_TONE` mapea UP→pos(verde)/DOWN→neg(rojo) (:61-63) y se aplica a PREDICCIONES en :524,:536,:584-589,:709,:781,:879. No hay verbos imperativos (verificado), pero la semántica de color compra/vende está presente en la superficie diagnóstica.

## Qué falta exactamente
En superficies DIAGNOSTIC: mostrar 'probabilidad estimada de subida: NN%' y tonos neutros (o un único acento) en badges de predicción; los tonos pos/neg quedan reservados a superficies ACTION (replay/production), donde LONG/SHORT son decisiones auditables.

## Impacto frontend
Cambia la paleta de los badges/cards de predicción en `/forecasting` (todas las assets). Sin cambio en /dashboard ni /production.

## Dependencias
BL-01/BL-02 (tests actualizados con el nuevo wording).

## Verificación
Grep: `DIRECTION_TONE` sin usos en ramas de forecast; snapshot visual.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   python -m pytest tests/regression/test_forecasting_caveat_present.py -q
verde:     31 passed   (baseline 28 antes del cierre)

comando-2: npx vitest run tests/unit/components/forecasting-caveat-surfaces.test.tsx
                        tests/unit/components/forecasting-weekly-branch.test.tsx
           (desde usdcop-trading-dashboard/)
verde-2:   47 passed

muta:      el CUERPO de directionLabel devuelve el token crudo (LONG/SHORT) en vez de
           mapear a las constantes neutras del SSOT
espera:    1 failed, 30 passed — "el CUERPO de directionLabel no referencia
           FORECAST_DIRECTION_LABEL_UP (importarlo no basta: el import sobrevive a un
           cuerpo vaciado)"
```

**Historial honesto**: hasta el 2026-07-28 el candado Python **no mordía**, y no por técnica
sino por **ALCANCE**: miraba EL FICHERO —donde el bloque de `import` satisface la
comprobación de subcadenas— en vez del CUERPO de la función. Se arregló de raíz en vez de
delegar: se extrae el cuerpo del arrow con el mismo primitivo enmascarado del módulo (llaves
dentro de string o comentario no cierran el cuerpo) y se exige que las constantes SSOT estén
EN EL CUERPO y que cada `return` referencie una constante, no el parámetro.

**Cobertura delegada, declarada**: el check estático **no puede probar el DOM** — un cuerpo
que devolviera SIEMPRE la misma constante pasaría el check y solo cae en Vitest, que compara
fila a fila. Por eso va acompañado de un guard de delegación que fija fichero, helper, los
dos títulos de test y el aserto concreto, para que la delegación no se evapore en silencio.

**Defecto de producto — RESUELTO el 2026-08-05 (`802b0267`), con decisión del operador.** El
texto original está debajo. Era real y era peor de lo que decía: no eran dos afirmaciones
contradictorias sino **cuatro declaraciones, tres incompatibles** — `analysis-assets.ts` (los
cuatro activos `model_zoo`), el comentario de `ForecastingView.tsx:1031` («Gold = weekly
inference», contradicho por la línea inmediatamente debajo), `ForecastingLegacy.tsx:57`
(`isUsdcop`) y `CLAUDE.md:181`.

Se resolvió midiendo **lo servido**, no lo declarado: `xauusd` y `btcusdt` publican **459
ficheros cada uno** (`bi_dashboard_unified.csv` + PNGs por modelo) y además
`weekly_inference_*.json`; `spx500` estaba declarado `model_zoo` con **cero** artefactos y su
`csvPath` apuntaba a un fichero inexistente. Decisión del operador: manda el disco — Oro y BTC
**son** model zoo, sus JSON weekly quedan como superficie secundaria, la puerta es
`forecast_mode` en ambas vistas, y `forecast_mode` gana el valor `'none'` para declarar la
ausencia de superficie en vez de ocultarla (`spx500` sale del selector pero sigue en
`ANALYSIS_ASSETS`, que es SSOT compartida con `/analysis`).

Texto original: las dos puertas de `/forecasting`
discriminan por criterios DISTINTOS —la vista GM por `forecast_mode`, y
`components/legacy/ForecastingLegacy.tsx:121` por `isUsdcop`—, así que
`/forecasting?asset=btcusdt` afirma "9 modelos de Machine Learning" y `/legacy/forecasting`
con BTC monta el banner weekly que afirma "política basada en REGLAS: no hay conjunto de
modelos ML". Dos afirmaciones de hecho contradictorias sobre el mismo producto; qué es
Oro/BTC es decisión de producto, no de ingeniería.

**Re-verificación 2026-08-05 (repetir el ataque, no citar la ficha):**

    pytest test_forecasting_caveat_present.py                31 passed
    vitest forecasting-caveat-surfaces + weekly-branch       47 passed
    MUTANTE declarado: el CUERPO de directionLabel devuelve
      el token crudo (`return d`) en vez de la constante SSOT
      -> 1 failed, 30 passed  (test_direction_label_maps_tokens_to_the_ssot_constants)
    restaurado con `git checkout --` -> byte-idéntico

El candado sigue vivo y sigue apuntando al **cuerpo** de la función, que es donde falló de
origen: mirar el fichero entero dejaba que el bloque de `import` satisficiera la comprobación.

## Notas constitución
FABRIC §24.3: 'sin colores ni etiquetas imperativas'. No es cosmética: es la barrera visible.
