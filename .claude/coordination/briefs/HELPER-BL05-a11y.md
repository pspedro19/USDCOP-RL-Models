# ENCARGO CLD-HLP → claude-helper-417962fe — BL-05 remediación del rechazo CXD-022

Asignado por: claude-root-a060f9b7 · 2026-07-27T23:31:00-05:00 · reporte AQUÍ MISMO (append
"## REPORTE" al final) o por tu salida; la raíz integra y commitea. NO commitees.

## Contexto
CXD-022 RECHAZÓ BL-05@624465c61b964da15a1f7abe2390a8b5d1f844ff con gaps concretos:
(1) falta `<th scope="row">` (row headers) en la tabla de candidatas;
(2) tipografía fija `12.5px` (debe ser responsive/relativa);
(3) la evidencia E2E (60d0af8) ANTECEDE al remedio — falta prueba real post-hash:
    viewport 375px, landscape, navegación por teclado, consola limpia;
(4) heredaba manifest FROZEN 1 fail — YA RESUELTO por ecbfca5 (re-freeze v5), solo
    verifica que tests/regression/test_strategy_manifests.py está 18 passed.

## Scope (SOLO estos paths)
- usdcop-trading-dashboard/components/gm/views/PaperCandidatesPanel.tsx
- usdcop-trading-dashboard/tests/unit/components/PaperCandidatesPanel.test.tsx
- 1 spec Playwright NUEVO (p.ej. usdcop-trading-dashboard/tests/e2e/paper-candidates-a11y.spec.ts
  o donde viva el patrón e2e del repo — mira cómo corre `qa:visual`/playwright existente)
PROHIBIDO: todo lo demás; en especial ProductionView.tsx (solo si el cambio de
PaperCandidatesPanel lo exige, y decláralo), .claude/coordination/**, WIP del operador
(types.ts, NewsClusterCard, ReferencesSection, HubView, LandingView, generate_weekly_forecasts).

## Tareas (fail-first: rojo pegado antes de cada fix)
1. Primera celda de cada fila (nombre candidata) → `<th scope="row">` con estilo intacto.
2. `12.5px` → tamaño relativo (rem/clamp o clase del design system GM) sin romper layout.
3. Playwright post-hash contra el dev server (:3001 si está vivo; si no, decláralo):
   375px viewport + landscape + tab-navigation llega a la región scrolleable (tabIndex) +
   cero errores de consola; screenshots como evidencia (guárdalas junto al spec o en
   .claude/coordination/reviews/e2e/ SOLO si la raíz te lo confirma — por defecto en tests/e2e/__screenshots__).
4. Suite: npx vitest run tests/unit/components/PaperCandidatesPanel.test.tsx (verde, extiende
   asserts para scope=row y tipografía relativa) + pytest tests/regression/test_strategy_manifests.py -q (18 passed).

## Reglas
Fail-first · 0 trials (decisión de modelado ⇒ PARA y reporta) · K-023 (WIP ajeno no se toca)
· tu reporte incluye: mapeo gap→fix, salidas rojo→verde, paths tocados, evidencia Playwright.
