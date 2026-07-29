---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
  - usdcop-trading-dashboard/components/gm/views/PaperCandidatesPanel.tsx
  - usdcop-trading-dashboard/tests/unit/components/PaperCandidatesPanel.test.tsx
  - usdcop-trading-dashboard/tests/e2e/paper-candidates-a11y.spec.ts
  - scripts/pipeline/candidates_paper_ledger.py
---

# BL-05 — ProductionView consume el paper ledger (A/B v11/v12/v14)

**Fuente**: FABRIC §24.5 Control Tower / plan 00 §2 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El ledger existe y se refresca cada viernes vía L6 (`candidates_ledger_2026.json`, anclado a ene-2026, judge_window post-freeze). ProductionView NO lo referencia — el A/B vivo es invisible en la UI.

## Qué falta exactamente
Panel en /production: tabla candidatas (v11 real vs v12/v14 paper), judge_window post-2026-07-21 con N y nota N<20, días al juez. Solo lectura del JSON publicado.

## Impacto frontend
`/production` gana el panel A/B. Cero botones (read-only por invariante).

## Dependencias
—

## Verificación
Panel renderiza el JSON real; N<20 muestra solo conteo/PnL.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: npx vitest run tests/unit/components/ProductionView.paper-ledger.test.tsx                         tests/unit/components/PaperCandidatesPanel.test.tsx
         (desde usdcop-trading-dashboard/)
verde:   16 passed

muta:    components/gm/views/ProductionView.tsx:982
         {paperLedger && <PaperCandidatesPanel/>}  ->  {false && paperLedger && <PaperCandidatesPanel/>}
espera:  1 failed — "Unable to find role=table and name /candidatas/i"
         (el A/B v11/v12/v14 desaparece de /production: el estado exacto que este BL arregla)

muta-2:  components/gm/views/PaperCandidatesPanel.tsx — celda extra "Sharpe 3.35 · p=0.006"
         en filas con n_trades=11
espera:  1 failed — la fila publica Sharpe con N<20, prohibido por quant-constitution §6
```

**Historial honesto**: hasta el 2026-07-28 ninguna de las dos mutaciones movía un test.
Los 13 tests existentes renderizaban `PaperCandidatesPanel` **aislado** con un fixture, así
que el panel podía estar desconectado de la página; y el test de §6 comprobaba que se ecoa un
string que venía **del propio fixture**, no que no hubiera un Sharpe al lado.

## Notas constitución
Vote-2/decisiones siguen sobre bundles; esto es monitoreo del juez sellado — jamás re-anclar.

## Remediación del rechazo CXD-022 (2026-07-28)

CXD-022 rechazó `624465c` por cuatro gaps. Estado tras el remedio:

| Gap del rechazo | Estado | Dónde |
|---|---|---|
| Falta `<th scope="row">` (row headers) | **HECHO** | `PaperCandidatesPanel.tsx` — la celda de nombre de cada fila es `th scope="row"` (`font-normal`/`text-left` preservan el aspecto) |
| Tipografía fija `12.5px` | **HECHO** | `text-[12.5px]` → `text-[0.78125rem]`; el resto de tokens (`GMT.*`) ya eran rem. Los únicos `px` que quedan son de layout (`min-w-[760px]`, `-mx-[18px]`), no de fuente |
| Significado por símbolo/color sin equivalente textual | **HECHO** | `NoData()` (guión `aria-hidden` + `sr-only "sin dato"`) y texto `sr-only` **`En producción: Sí/No · Juez sellado: Sí/No`** en el row header (el tono del badge era canal cromático) |
| Prueba real 375px / landscape / teclado / consola | **PENDIENTE DE EJECUCIÓN** | spec escrito y listo en `tests/e2e/paper-candidates-a11y.spec.ts`; **no ejecutado** por orden del operador (2026-07-28: no levantar Docker ni el dashboard). No hay evidencia E2E asociada y no debe darse por verde |

**Cobertura unit actual**: `PaperCandidatesPanel.test.tsx` **13 passed** (2 asserts nuevos
verificados fail-first: 2 failed antes del fix → 13 passed después). `tsc --noEmit`: 0 errores
en los archivos tocados.

**Para cerrar el gap E2E** hace falta un dashboard servido con este remedio y sesión `admin`
(el panel está oculto para `free`/`subscriber` — `ProductionView::isClientView`), y luego
`npx playwright test tests/e2e/paper-candidates-a11y.spec.ts --project=chromium`.
