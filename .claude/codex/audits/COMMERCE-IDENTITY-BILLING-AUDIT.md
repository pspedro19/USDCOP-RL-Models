---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - database/migrations/055_rbac_monetization.sql
  - database/migrations/056_rbac_dynamic_roles.sql
  - database/migrations/057_catalog_watchlist_cart.sql
  - usdcop-trading-dashboard/lib/billing/provider.ts
  - usdcop-trading-dashboard/lib/billing/wompi.ts
  - usdcop-trading-dashboard/app/api/billing/webhook/route.ts
  - usdcop-trading-dashboard/app/api/cart/checkout/route.ts
---

# Auditoría — identidad, catálogo, carrito y pagos

## Estado funcional

| Dominio | Estado observado | Evidencia | Veredicto |
|---|---|---|---|
| Registro y aprobación | Flujo PENDING → APPROVED → reset obligatorio documentado y probado | `rbac-monetization.md`, QA de registro | IMPLEMENTED |
| RBAC | Matriz estática + permisos dinámicos + overrides + auditoría | migraciones 055/056, `rbac.contract.ts` | IMPLEMENTED, requiere pruebas adversariales continuas |
| Catálogo | Catálogo y precios expuestos mediante BFF | rutas `/api/catalog`, `/api/billing/prices` | IMPLEMENTED |
| Watchlist | Persistencia por usuario con PK compuesta | migración 057 + rutas | IMPLEMENTED |
| Carrito | Persistencia por usuario y checkout hosted | migración 057 + rutas | PARTIAL |
| Pagos | Adaptador Wompi y webhook firmado | `lib/billing/wompi.ts` | UNSAFE_FOR_PRODUCTION hasta cerrar P0 |
| Suscripciones | Expiración lazy a 30 días | webhook + entitlements | PARTIAL; no hay ledger completo de suscripción |
| Refund/chargeback | No se observa flujo completo | rutas/esquema actuales | DESIGNED/ABSENT |

## Hallazgos

### PAY-P0-001 — add-ons concedidos sin ser cobrados

`WompiProvider.createCheckout()` calcula `amountInCents` únicamente con el precio del plan. Sin embargo,
`encodeReference()` incluye los add-ons del carrito y el webhook incorpora esos add-ons en `assets`.
Resultado: un usuario puede recibir activos adicionales pagando solo el plan base.

**Cierre obligatorio:** cálculo server-side `plan + Σ(add-ons)`, catálogo de precios inmutable, monto esperado
persistido antes de redirigir, y comparación exacta de `amount_in_cents` + moneda en el webhook.

### PAY-P0-002 — falta idempotencia y protección contra replay

No existe un ledger con `provider_event_id`/`transaction_id` único. El mismo webhook aprobado puede procesarse
más de una vez y extender repetidamente `expires_at` desde `now`.

**Cierre obligatorio:** tabla `billing_events` append-only, constraint única por proveedor+evento/transacción,
transacción DB y respuesta 200 idempotente para duplicados conocidos.

### PAY-P0-003 — la referencia es una autorización insuficiente

La referencia codifica plan, UUID, add-ons y timestamp. Aunque el webhook tiene firma, los derechos concedidos
se reconstruyen desde una cadena y no desde una orden server-side congelada.

**Cierre obligatorio:** referencia opaca aleatoria; tabla `checkout_orders` con usuario, items, precios, moneda,
total, estado y expiración. El webhook solo puede aplicar la orden almacenada.

### PAY-P0-004 — conciliación incompleta

No se valida monto, moneda, merchant/account, transaction id ni correspondencia exacta con una orden pendiente.
La comparación del checksum debería ser constante en tiempo. Eventos no reconocidos caen semánticamente en
`subscription.cancelled`.

### PAY-P1-001 — carrito no se limpia al aprobar

La ruta declara que la limpieza queda diferida, pero el webhook no la realiza. Debe limpiarse en la misma
transacción que concede el entitlement o conservarse como snapshot histórico de la orden.

### PAY-P1-002 — modelo de suscripción incompleto

`expires_at = now + 30 días` no representa renovación, prorrateo, cancelación al final del periodo, refund,
chargeback, gracia, invoice, impuesto, estado de cobranza ni historial de precios.

### ID-P1-001 — autorización de objeto debe verificarse por usuario

Toda ruta con `[userId]`, `[id]` o `[assetId]` requiere pruebas BOLA/IDOR: usuario A nunca puede leer o mutar
recursos de B. Los checks de UI no cuentan; la validación debe ser server-side.

### ID-P1-002 — JWT con permisos horneados tiene ventana de revocación

Cambios dinámicos aplican al siguiente login. Operaciones sensibles —pagos, llaves, live enable, kill switch,
impersonación y administración— deben revalidar estado/rol/permisos directamente contra la autoridad vigente.

## Modelo mínimo recomendado

```text
checkout_orders(id, user_id, provider, currency, subtotal, tax, total, status, expires_at)
checkout_order_items(order_id, sku, asset_id, quantity, unit_amount, price_version)
billing_events(id, provider, provider_event_id UNIQUE, transaction_id, payload_hash, received_at, processed_at)
subscriptions(id, user_id, plan, status, period_start, period_end, cancel_at_period_end)
invoices(id, subscription_id, provider_invoice_id UNIQUE, amount, currency, status)
entitlement_ledger(id, user_id, source_event_id, before, after, created_at)
```

## Pruebas obligatorias

1. Monto correcto para base, un add-on y varios add-ons.
2. Monto alterado, moneda distinta o referencia desconocida → no concede acceso.
3. Replay del mismo evento → una sola transición.
4. Eventos fuera de orden: approved → declined/cancelled/chargeback.
5. Dos webhooks concurrentes → serialización/idempotencia.
6. Usuario A intenta operar carrito/watchlist/orden de B → 403/404 sin fuga.
7. Add-on ya poseído → no se cobra dos veces.
8. Fallo de auditoría → política explícita; para grants financieros se recomienda fail-closed.
9. Expiración, renovación y cancelación verificadas con reloj controlado.
10. Redacción de PII y secretos en logs, traces, screenshots y videos.
