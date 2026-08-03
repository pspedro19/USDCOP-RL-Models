---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# SP500 OOS gate

`src/validation/sp500_oos_gate.py` is the final promotion check for
`spx500_regime_gated_v1`. It rejects the current synthetic scaffold and any
manifest without point-in-time lineage (`point_in_time` and
`available_at_required`). A promotable report must include purged/CPCV,
leakage, PBO and Deflated Sharpe evidence, explicit commission/slippage costs,
four benchmarks, and meet Sharpe >= 0.5, return >= 0%, max drawdown <= 25%,
PBO < 0.50 and DSR > 0.95. The result contains deterministic reason codes for
frontend/backend display and audit logs.

Run: `python -m pytest tests/unit/test_sp500_oos_gate.py -q`.
