---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app
  - usdcop-trading-dashboard/components
  - usdcop-trading-dashboard/lib/ui/gm-tokens.ts
  - usdcop-trading-dashboard/playwright.config.ts
---

# Auditoría UX, accesibilidad y mobile

## Dirección adecuada

El producto es un terminal institucional de trading: alta densidad, dark-first, movimiento mínimo y claridad
operativa. La firma visual debe ser la trazabilidad de cada número —LIVE/PAPER/BACKTEST, timestamp, fuente y
calidad—, no decoración. La recomendación generada por `ui-ux-pro-max` favorece un dashboard denso y dark;
se rechazan como genéricas sus sugerencias Inter + glow/purple cuando contradigan la identidad ya establecida.

## Gates críticos

| Área | Requisito | Prueba |
|---|---|---|
| Contraste | texto normal ≥4.5:1, componentes/datos ≥3:1 | axe + cálculo de tokens |
| Teclado | orden lógico, focus visible, skip-link, Escape | Playwright keyboard-only |
| Touch | 44×44 CSS px mínimo y separación de 8px | medición DOM en 390px |
| Zoom/texto | 200% sin pérdida; body móvil ≥16px | screenshot + overflow assertion |
| Responsive | sin scroll horizontal; contenido prioritario primero | 390/768/1440/1920 |
| Motion | `prefers-reduced-motion` | emulación Playwright |
| Charts | tabla/resumen alternativo, tooltip por teclado, no color-only | axe + assertions |
| Forms | labels, autocomplete, inputmode, error junto al campo | DOM contract |
| Async | feedback >300ms, botón bloqueado contra doble submit | network delay test |
| Mobile chrome | `dvh`, safe-area, sticky offsets | portrait + landscape |

## Flujos visuales obligatorios

- Registro: vacío, validación, pendiente, duplicado y responsive keyboard.
- Login/reset: error, lockout, password manager/autocomplete y foco.
- Catálogo/carrito: vacío, cargando, error, owned, coming-soon, checkout.
- Billing: plan activo, expiración, fallo de proveedor, transacción pendiente/aprobada/declinada.
- Terminal: chart cargando/vacío/error, replay, tooltip teclado/touch y tabla alternativa.
- Admin: queue vacía/cargada, drawer de usuario, roles, overrides e impersonación visible.
- Execution: paper/live inequívocos, kill switch protegido contra toque accidental.

## Hallazgos de QA visual

El repositorio posee numerosos tests que toman screenshots en rutas y directorios distintos. Esto dificulta
retención, comparación y trazabilidad. Deben migrarse a `testInfo.outputPath()` y adjuntos Playwright, con un
solo reporter y manifest. Las screenshots no deben contener tokens, emails reales, API keys, balances reales
ni referencias de pago.

## Criterio mobile

La aplicación web debe ser operable en móvil para consulta y control de emergencia, pero la habilitación de
live trading, cambio de límites o llaves requiere confirmación reforzada. No se diseñará una app nativa hasta
que PWA responsive, accesibilidad y flujos touch estén verdes.

