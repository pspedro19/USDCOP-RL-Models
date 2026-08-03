---
kind: roadmap
status: IMPLEMENTED
version: 1.1.0
last_verified: 2026-07-31
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
  - tests/regression/test_knowledge_frontmatter.py
---

# BL-01 — Test de regresión del caveat `da-caveat`

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El banner existe (`ForecastingView.tsx` ~:1057-1076, texto 'Superficie de diagnóstico, no de señales…') pero NO hay ningún test que lo proteja (grep en tests/: cero asserts sobre `da-caveat`).

## Qué falta exactamente
Test (grep estático o e2e) que falle si el banner o su `data-testid="da-caveat"` desaparece o cambia a texto sin la frase 'no es una señal/no de señales'.

## Impacto frontend
Ninguno visible — es un candado. Congela el contrato honesto de la vista.

## Dependencias
—

## Verificación
`pytest` nuevo test rojo al comentar el banner, verde con él.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   python -m pytest tests/regression/test_forecasting_caveat_present.py -q
verde:     31 passed

comando-2: npx vitest run tests/unit/components/forecasting-caveat-surfaces.test.tsx
           (desde usdcop-trading-dashboard/)
verde-2:   41 passed

muta:      usdcop-trading-dashboard/lib/ui/forecast-disclaimer.ts — copy decorativo que
           CONSERVA el data-testid y solo vacia el sentido:
           'Superficie de diagnóstico: señal validada, opere con confianza.'
espera:    rojo — el candado exige las CLÁUSULAS COMPLETAS normalizadas, no el prefijo
           'Superficie de diagn'
           "Disclaimer copy contains promotional/action language — a diagnostic caveat
            that recommends acting is worse than no caveat"
           (conteo exacto de failed <sin registrar — el veredicto CLD-212 se anotó como
            "rojo", no como número)
```

**Historial honesto**: BL-01 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`, CLD-212) — **no hubo defecto que cerrar el 2026-07-28**. Lo que sí conviene
saber es que la PRIMERA versión del candado tenía justo ese hueco: fijaba únicamente el
prefijo `'Superficie de diagn'`, así que la mutación engañosa de arriba pasaba. Se endureció
tras la revisión de CODEX, **antes** del tablero de mutación: hoy (a) parsea las constantes
del SSOT, (b) exige las cláusulas no-señal completas y (c) rechaza lenguaje promocional o
imperativo. Y recibió un segundo endurecimiento el 2026-07-28 por rebote de BL-06
(`5ec84a19`): el escaneo de marketing tenía el hueco de concatenación —`'opere con ' +
'confianza'` eran dos líneas inofensivas— y pasó a usar la vista normalizada compartida de
`tests/support/js_source_scan.py`. Lo que lo hace fiable es que la mutación **conserva el
`data-testid`**: no se comprueba la presencia del ancla, se comprueba el contenido.

## Notas constitución
La honestidad publicada es parte del producto (FABRIC §3.2); sin test es disciplina humana.

## Cierre (2026-07-31, cross-review CXD-189)

**PARTIAL -> IMPLEMENTED.** Verificado por CRITERIO, no por ancla viva.

- **Verde en arbol limpio** (K-050: `git status --porcelain` de las rutas medidas = `[]` antes y
  despues, `sha256` publicado): `python -m pytest tests/regression/test_forecasting_caveat_present.py -q`
  = **31 passed**; `npx vitest run forecasting-caveat-surfaces + forecasting-weekly-branch`
  = **47 passed**.
- **Mutacion ejecutada por CODEX** (no por el dueño): copy enganoso que **conserva** el ancla
  `da-caveat` => Python **4F/27P**, Vitest **2F/45P**. Restauracion byte-exacta independiente,
  `forecast-disclaimer.ts = 506BB286...F01A8A`.
- **Requisito exacto del BL cumplido**: el candado no cae cuando desaparece el testid, sino cuando
  el SENTIDO deja de ser honesto conservando el testid — que era la evasion que lo motivaba.
- Veredicto CODEX: **APROBADO para promocion** (`CXD-189`).
