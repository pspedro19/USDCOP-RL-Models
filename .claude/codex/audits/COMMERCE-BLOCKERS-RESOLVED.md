---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Commerce blockers resolution

Implemented locally without provider credentials:

- Checkout now persists an immutable server-side quote in `checkout_orders` before redirect.
- `billing_events` stores provider-correlated, idempotent events.
- Database trigger enforces legal order lifecycle transitions, including refund and chargeback.
- Webhook handles approved, failed/cancelled, refunded and charged-back events; refund/chargeback revokes asset entitlements.
- Existing integration contract remains green (`2 passed`).

Production evidence still requiring sandbox credentials: signed provider E2E, settlement reconciliation, and real refund/chargeback callbacks. These are external gates, not safely fabricatable in CI.
