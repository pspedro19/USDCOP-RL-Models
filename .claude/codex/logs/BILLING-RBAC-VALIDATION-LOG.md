---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Billing / cart validation (2026-07-20)

- Checkout amount is now computed server-side from plan and published add-on prices; client-supplied totals are ignored.
- Wompi checkout includes plan + add-on COP cents in the integrity signature.
- Payment-approved webhooks require an exact amount match against the same price SSOT before entitlements are granted.
- Migration `058_billing_webhook_idempotency.sql` adds a unique `(reference,event_type)` ledger; provider retries return `duplicate:true` without extending access or duplicating audit entries.
- Remaining operational requirement: apply migration 058 in each environment and add integration tests with a provider fixture (approved, underpaid, replay).
