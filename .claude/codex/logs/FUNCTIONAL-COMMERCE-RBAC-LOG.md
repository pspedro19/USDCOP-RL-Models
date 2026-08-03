---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/api/cart/checkout/route.ts
  - usdcop-trading-dashboard/app/api/billing/webhook/route.ts
  - usdcop-trading-dashboard/lib/contracts/rbac.contract.ts
  - usdcop-trading-dashboard/app/api/analysis/news-feed/route.ts
---

# Validación funcional: commerce, RBAC y noticias

## Auditoría inicial

- El checkout lee cart y entitlements en servidor; el cliente no decide add-ons ni precios.
- El webhook verifica firma antes de modificar entitlements, pero inicialmente no verificaba
  monto/moneda contra una orden persistida ni protegía replay con `provider_event_id`.
- `user_cart` es por usuario y con PK `(user_id, asset_id)`; la compra no debe conceder acceso.
- News feed agrega titulares desde análisis; debe conservar URL/fecha/source y marcar el texto
  como contenido no confiable para UI/LLM, sin convertirlo en instrucciones.
- RBAC tiene matriz estática + resolución DB dinámica y deny-by-default; faltan pruebas completas
  de cada ruta sensible, tenant isolation y transición webhook→entitlement.

## Gate de cierre

No marcar compra/RBAC como verde hasta cubrir: firma, monto, moneda, orden, replay, duplicado,
concurrencia, refund/chargeback, BOLA por usuario/tenant, expiración, revocación, add-on no
entitled, accesibilidad del carrito, estados de loading/error y noticias con URLs maliciosas.

## Implementado y verificado

- Migration `058_billing_webhook_idempotency.sql`: unique `(reference,event_type)`.
- Migration `059_checkout_order_ledger.sql`: `checkout_orders` inmutable y `billing_events` con
  `provider_event_id` único.
- Webhook: firma → referencia → monto exacto plan+add-ons → idempotencia → entitlement/audit.
- Cart checkout: lee filas server-side, filtra activos ya poseídos y nunca acepta add-ons del body.
- RBAC: roles/IDs inválidos rechazados; `93 API routes, 30 pages` cubiertas; matriz contractual PASS.
- Noticias: URLs externas limitadas a `http/https`, `noopener noreferrer`; esquemas peligrosos quedan texto.
- Tests: billing/cart Vitest `2 passed`; assurance `8 passed` (6 fallos históricos pendientes);
  RBAC contract + coverage PASS.

## Pendientes explícitos

Refund/chargeback aún requiere máquina de estados y pruebas de reloj; checkout_orders debe ser
escrito por el endpoint checkout y reconciliado transaccionalmente con `billing_events` en producción.
También faltan E2E reales con sandbox Wompi, axe/focus-trap del drawer y corpus adversarial LLM.
