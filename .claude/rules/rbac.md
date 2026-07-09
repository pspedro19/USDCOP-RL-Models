# Rule: RBAC + Monetización (CTR-RBAC-001)

> Reglas duras de control de acceso y producto. SSOT de matrices/roles/planes:
> `usdcop-trading-dashboard/lib/contracts/rbac.contract.ts`. Spec profunda:
> `../specs/platform/rbac-monetization.md`. **El rol dice quién eres; el plan
> (entitlements) dice qué pagaste; ambos se validan server-side por request.**

## Reglas duras

1. **Deny-by-default.** Toda página (≠ landing/pricing/login) y `/api/**` exigen sesión +
   permiso server-side (`middleware.ts` desde el contrato). Ocultar UI no es control de acceso.
   El **test de cobertura** (`npm run rbac:check`) falla si una ruta real no está en la matriz.
2. **Nada monetizado anónimo.** `/data/**` y `/forecasting/**` exigen sesión en el edge; el
   delay por plan (free ⇒ análisis T+7) lo aplica `/api/data` (`lib/auth/entitlements.ts`).
3. **Rol ≠ plan.** El JWT es cache; `sb_users.entitlements` (JSONB, migración 055) es la
   verdad; vencido ⇒ se sirve como `free` (degradación automática en `effectiveEntitlements`).
4. **Vote 2 / promover / kill global: solo `admin`,** siempre al `audit_log` (append-only,
   trigger que bloquea UPDATE/DELETE).
5. **SignalBridge multi-tenant estricto:** llaves por usuario cifradas (`user_exchange_keys`),
   scoped a su `user_id`; **rechazar llaves con permiso withdraw**. El sistema jamás ejecuta
   por un usuario con llaves del sistema ni al revés.
6. **Paper-first:** `PreTradeGate` (services/signalbridge_api/app/services/pretrade.py) es el
   último gate antes del exchange — modo global (`trading_mode`, default PAPER simula y NO
   envía), kill-switch por usuario (`sb_trading_configs.trading_enabled`, default False),
   caps de notional y trades/día. **Fail-safe: error ⇒ BLOCK.**
7. **Billing por webhook:** el proveedor es la verdad (`lib/billing/`, DIP: interface +
   Wompi); webhook firma-verificada actualiza `entitlements`; nunca confiar en el cliente.
8. **Subscribers ven OUTPUTS, no INTERNALS** (sin gates/configs/experimentos/registry crudo).
9. **Gate legal antes de `auto` con dinero real de terceros** (SFC Colombia); hasta entonces
   paper-only. Disclaimer persistente en toda superficie con señales.

## DO NOT
- Do NOT añadir una ruta sin entrada en `rbac.contract.ts` (el CI la detecta).
- Do NOT poner artefactos monetizables nuevos en `public/` sin gate.
- Do NOT saltarte `PreTradeGate` en ningún path de orden nuevo.
- Do NOT actualizar entitlements desde el cliente — solo el webhook o `/admin`.
- Do NOT editar/borrar filas de `audit_log` (trigger lo impide; no lo quites).
