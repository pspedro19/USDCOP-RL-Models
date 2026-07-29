---
kind: roadmap
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/dashboard/page.tsx
  - usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx
---

# BL-34 — Ruta /replay (alias de la sección de /dashboard)

**Fuente**: plan 00 §2 / FABRIC §24.2 · **Ola**: T · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El replay completo vive como ForecastingBacktestSection dentro de /dashboard; no existe app/replay/. Funcional hoy; el gap es de navegación/nomenclatura vs la constitución.

## Qué falta exactamente
Ruta /replay que monte la misma sección (o redirect), RBAC en la matriz, nav actualizada. Vote-2 se QUEDA en /dashboard.

## Impacto frontend
Nueva entrada de navegación; cero lógica nueva.

## Dependencias
Al final (cosmético); tras BL-05.

## Verificación

**Cross-review APROBADO por CODEX el 2026-07-28 (CXD-087) contra el snapshot inmutable `531c9eb4`.**
El otro ingeniero **ejecutó la mutación**, no se limitó a leer el diff.

```
comando-1: python -m pytest tests/regression/test_replay_is_read_only.py -q
           => 8 passed
comando-2: npx vitest run tests/unit/components/replay-read-only-render.test.tsx
           (desde usdcop-trading-dashboard/)
           => 5 passed

muta:      components/production/ForecastingBacktestSection.tsx:1619
           const canPromote = userRole === 'admin' && !readOnly;  ->  const canPromote = true;
espera:    pytest  1 failed / 7 passed  (cae la conjunción pura con !readOnly)
           vitest  2 failed / 3 passed
           "expected [ 'Aprobar (Voto 2/2)', 'Rechazar' ] to deeply equal []"

muta-2:    un componente NUEVO con botones "Aprobar (Voto 2/2)" / "Rechazar" /
           "Desplegar a producción" y POST a /api/production/approve, montado en
           app/replay/page.tsx — probado también desde components/ y desde lib/ui/
espera:    el perímetro derivado del cierre de imports lo caza esté donde esté el fichero
```

**Evidencia del cross-review**: restaurado con SHA256
`72B33DE60C4E136F462973013CBEA4BCA264179AD38D8393D8C8EF02AC8F1C51`, idéntico al inicial.
Monitores en verde: RBAC coverage 95 API / 32 páginas, contrato RBAC ALL PASS,
manifests + scripts-layout 41/41. Advertencia no bloqueante: jsdom emite warnings de SVG
(`linearGradient`/`defs`) que no afectan a la garantía.

**Historial honesto**: hasta el 2026-07-28, **ninguna de las dos mutaciones movía un test**.
`canPromote = true` daba **578 passed, `rbac:check` OK, `rbac:test` PASS: cero delta**, y un
widget de aprobación nuevo montado en `/replay` también pasaba limpio. El único test que las
cazaba era la spec de Playwright, y `grep playwright .github/workflows` daba **0**: ningún
workflow la ejecutaba. El comportamiento era correcto en runtime — lo que no existía era la
protección.

**Asimetría de diseño**: `/replay` comparte implementación con `/dashboard`
(`BacktestTerminalPage` + `ForecastingBacktestSection readOnly`), así que la maquinaria de
aprobación está **legítimamente** dentro del cierre de imports y una lista negra plana sería
roja en árbol prístino. Por eso el candado se parte: los otros 28 ficheros del cierre llevan
lista negra, y el único fichero dual queda exento pero fijado por tests estructurales
(`canPromote` debe ser conjunción pura con `!readOnly`; los paneles solo montan dentro de
`{canPromote && …}`) más el render en jsdom. **Cada exención se auto-verifica**: si deja de
aplicar, rojo.

## Evidencia de runtime

Capturas selladas contra el `BUILD_ID` del artefacto realmente servido (K-044), en
`.claude/coordination/integration/evidence/build-1785286031174-no-orderbook__*.png`:
anónimo redirigido a `/login`, `/api/registry` respondiendo 401 JSON, y la vista admin con
la cabecera "INVESTIGACIÓN · SOLO LECTURA", el chip **SOLO LECTURA** y la nota de que el
Voto 2 vive únicamente en `/dashboard`.

## Notas constitución
No mover los botones de aprobación: /replay es lectura.
