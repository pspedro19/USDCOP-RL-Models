---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/lib/contracts/catalog.contract.ts
  - usdcop-trading-dashboard/app/api/cart/checkout/route.ts
  - database/migrations/057_catalog_watchlist_cart.sql
---

# Marketplace de modelos: alcance y contrato requerido

El catálogo actual vende acceso por activo/add-on. No existe aún un SKU de modelo o estrategia;
por tanto la interfaz no debe comunicar “compra de modelo” hasta implementar este contrato.

## Entidad vendible

`ModelSKU` debe incluir `sku`, strategy/model ID, versión inmutable, asset, horizonte, modalidad
(señal/API/artefacto), licencia, vigencia, precio/moneda/impuestos, compatibilidad, estado de aprobación,
hash del artefacto y snapshot de evidencia. Una compra referencia exactamente esa versión; una actualización
requiere política explícita y nunca reemplaza evidencia histórica.

## Disclosure antes de pagar

- propósito, uso prohibido, requerimientos de capital/datos/ejecución;
- OOS y periodo, baseline, costos asumidos, DSR/trials, drawdown, capacidad y limitaciones;
- `as_of`, caducidad, paper/production, riesgo de pérdida y ausencia de garantía;
- licencia, soporte, actualizaciones, revocación, refund y tratamiento de datos.

## Flujo transaccional

Cotización server-side → orden inmutable → pago → webhook firmado e idempotente → reconciliación de
monto/moneda/SKU → entitlement versionado → recibo/audit log. El carrito solo transporta IDs; nunca precios,
permisos ni evidencia confiados al cliente.

## Harness de aceptación

- contract/property tests para precio, moneda, expiración y combinaciones de SKUs;
- integración con webhook duplicado, tardío, fuera de orden, monto alterado, refund y chargeback;
- concurrencia de checkout y grants exactamente una vez;
- BOLA: un usuario no descarga artefactos/licencias de otro;
- E2E desktop/mobile con teclado, lector, fallos de red y evidencia visual;
- revocación y rollback sin borrar ledger, recibos ni disclosure adquirido.

Hasta que estos gates existan, el producto permitido es “suscripción/acceso a activos”, no marketplace de modelos.
