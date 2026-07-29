---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-02 — Banner fuerte en Gold weekly-inference

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El caveat solo renderiza si `isModelZoo` (~:1057). Gold en modo weekly_inference muestra badges direccionales con colores (~:781,:879) SIN el banner fuerte — solo la nota suave de metodología (~:811-816).

## Qué falta exactamente
Extender el banner 'DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN' a TODA superficie de forecasting, incluido weekly_inference.

## Impacto frontend
`/forecasting?asset=xauusd` gana el banner ámbar permanente.

## Dependencias
BL-01 (el test debe cubrir ambos modos).

## Verificación
Banner visible con asset=xauusd; test de BL-01 lo exige por modo.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   npx vitest run tests/unit/components/forecasting-weekly-branch.test.tsx
           (desde usdcop-trading-dashboard/)
verde:     6 passed

comando-2: python -m pytest tests/regression/test_forecasting_caveat_present.py -q
verde-2:   31 passed

muta:      components/forecasting/ForecastDisclaimer.tsx — quitar 'weekly' de la variante
espera:    rojo en el render de punta a punta de la rama weekly_inference
           (banner + AssetWeeklyBody + tabla + etiquetas neutras)
           <conteo exacto de failed sin registrar>

muta-2:    envolver el disclaimer en {isModelZoo && ...}
espera-2:  rojo — el banner compartido debe renderizar INCONDICIONALMENTE en toda
           superficie de forecasting ('hidden por rama' es el bug rechazado en CXD-032)
           <conteo exacto de failed sin registrar>

muta-3:    borrar la rama weekly dejando el candado, hacer divergir el tipo, borrar el test
           de render, o AÑADIR un activo weekly_inference sin actualizar la declaración
espera-3:  rojo en test_weekly_branch_and_its_lock_stay_coherent_with_the_data
```

**Historial honesto**: hasta el 2026-07-28 el candado estático **mordía sobre CÓDIGO
MUERTO**. Los tres activos de `ANALYSIS_ASSETS` están en `forecast_mode: 'model_zoo'`, luego
`isModelZoo` es siempre `true`, `AssetWeeklyBody` es inalcanzable, y el banner de la
superficie weekly **no estaba verificado por nada ejecutable**: se congelaba una rama que
nadie podía ver. El cierre son dos piezas: un test de render que **inyecta la SSOT**
(`vi.mock` de `analysis-assets`, el único punto por el que `forecast_mode` entra a la vista)
y ejercita la rama con el resto del árbol REAL; y una invariante de coherencia que no
legisla el roadmap —no exige "la rama debe tener consumidor", eso obligaría hoy a borrar
código que Oro/BTC recuperarán— sino que **las tres capas se muevan juntas** y el estado real
esté DECLARADO: dato/tipo ↔ código ↔ candados. Hoy pasa describiendo la realidad (cero
consumidores declarados) y la deuda deja de ser silenciosa.

**Límite declarado**: el candado de coherencia se comprueba leyendo su propio módulo, así que
**borrar el fichero entero de regresión no lo dispara**. Eso necesita un gate de existencia
en CI.

## Notas constitución
La muralla es por superficie, no por asset ni por modo de render.
