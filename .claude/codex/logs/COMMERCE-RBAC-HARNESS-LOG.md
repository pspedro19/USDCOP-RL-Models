---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Commerce/RBAC Harness

`python scripts/validation/commerce_rbac_harness.py --json` runs deterministic checks for server-side cart totals, signed/idempotent webhooks, amount tampering, legal payment state transitions, entitlement grant/revocation on refund/chargeback, and owner-scoped RBAC/BOLA.

The harness is provider/DB independent by design; run it in CI before integration tests. A failing check is a promotion blocker. Extend `CommerceHarness.run()` when new SKUs, subscription renewal states, or provider contracts are added.
