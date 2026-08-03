---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Quantitative unblock log

2026-07-20 — Added `config/quant_evidence/assets.json` with explicit evidence
contracts for USD/COP, XAU/USD, BTC/USDT and S&P 500. The quantitative harness
now validates manifest-provided trial count, DSR, PBO, costs and benchmark
outperformance. All four remain externally blocked pending provider PIT data,
frozen OOS manifests and reproducible DSR/PBO computation; no synthetic result
is treated as production evidence.

Verification: `python -m pytest tests/unit/test_quant_harness.py -q` → 2 passed.
