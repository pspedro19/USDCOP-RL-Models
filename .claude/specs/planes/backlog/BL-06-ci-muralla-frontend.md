---
kind: roadmap
status: PARTIAL
version: 1.2.0
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

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_forecasting_caveat_present.py -q
verde:   31 passed   (28 en el momento del cierre; el fichero es COMPARTIDO con
                      BL-01/02/03/04 y creció con ellos)

muta:    widget rogue montado desde un directorio inventado
         (usdcop-trading-dashboard/lib/telemetry/RogueProbe.tsx) con la evasión completa:
           fetch(`${P}/appro` + 've')
           fetch('/api/exec' + 'ution/orders')
           <button>Comprar ahora</button>
espera:  1 failed, 27 passed — 3 hits:
         "RogueProbe.tsx:5: 'api/production/approve' :: fetch(`${P}/appro` + 've')"
         "RogueProbe.tsx:6: 'api/execution'          :: fetch('/api/exec' + 'ution/orders')"
         "RogueProbe.tsx:8: verbo de orden 'comprar' :: <button>Comprar ahora</button>"
         restaurado => 36 passed en los dos candados (caveat + /replay read-only)
```

**Historial honesto**: hasta el 2026-07-28 esa mutación —**misma capacidad de acción, cero
rojo**— daba **28 passed, 0 failed**. El perímetro derivado por cierre de imports SÍ atrapaba
un widget de aprobación puesto en cualquier carpeta; lo que fallaba era el **MATCHER**:
literales exactos en mayúsculas, evadibles con una concatenación y una minúscula. Por eso
BL-06 fue **retirado de DONE** (`e144ede`) y por eso `PROGRESS.md` estuvo declarando 2/47
cuando eran 1/47.

Y hubo **un agujero más que no se vio en el primer arreglo**: la evasión de referencia
``fetch(`${P}/appro` + 've')`` **seguía verde**, porque el plegado unía las piezas pero
`${P}` era opaco (`/appro` + `ve` = `/approve`, nunca `api/production/approve`). Una variable
extra derrotaba toda la cinta. Se cerró con **sustitución conservadora de constantes**: un
`${IDENT}` ligado EXACTAMENTE UNA VEZ a una cadena literal se sustituye; ligado dos veces a
valores distintos se descarta en vez de adivinar, porque adivinar fabrica falsos positivos.
El primitivo vivía duplicado dentro del candado de `/replay`, así que se extrajo a
`tests/support/js_source_scan.py` y lo importan los dos (−177 líneas netas).

**Límites declarados en el propio test**, porque un candado que se vende como total es peor
que uno honesto: cadenas ensambladas en runtime (`atob`, `fromCharCode`, `join`),
indirecciones que el paso de constantes no ve (identificador importado, sustitución
recursiva, miembro de objeto), homoglifos, y el diccionario de verbos. `buy`/`sell` en inglés
**no** son verbos a propósito: "Buy & Hold" es la etiqueta honesta del baseline en estas
mismas superficies (6 ocurrencias reales), y un blacklist que dispara sobre código honesto es
un blacklist que alguien borra.

## Notas constitución
FABRIC §25 bloque Muralla: 'frontend forecasting sin verbos de orden…'.
La superficie DIAGNOSTIC no gana capacidades de acción por accidente: el día que
forecasting pueda aprobar o ejecutar, este test se pone rojo.
