---
kind: adr
status: PARTIAL
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/forecasting/directional_replay.py
  - .claude/specs/assets/usdcop/EXP-DIR-001-directional-trials.md
  - .claude/rules/quant-constitution.md
  - services/common/metrics.py
---

# ADR-0021 — Volatility-first forecasting and data unlocks

**Decision status:** ACCEPTED (research gate) — 2026-07-21.
Front-matter `status: PARTIAL` tracks implementation, not the decision: prohibition 1
is enforced in code (`src/forecasting/directional_replay.py` refuses to run unless
`signal_authorized is False`), while the volatility track and the PIT data unlocks
below are not built.

## Decision

The existing directional model zoo remains a transparency and diagnostic surface.
Directional accuracy (DA) must not be used as an executable trading signal or as
evidence of profitability. The forecasting track is repurposed to predict realized
volatility/magnitude for risk sizing, subject to out-of-sample and forward gates.

No additional model search is authorized on the current price-only data after the
27 directional trials and the completed sweep. Further progress requires new,
point-in-time data:

- BTC/USDT: funding, open interest, basis and other crypto-native flows.
- USDCOP: point-in-time macro vintages with release timestamps and revision history.

## Prohibitions

1. `signal_authorized` must remain `false` for diagnostic model-zoo forecasts.
2. DA, directional hit rate, or a single backtest cannot justify promotion.
3. No look-ahead, revised macro series, or post-period feature may enter training.
4. Production promotion requires clean OOS/forward results, data-lineage checks,
   and DSR/PBO or equivalent multiple-testing controls.

## Acceptance gates

The volatility track is not promotable until each asset has: PIT data manifest,
time-split OOS evaluation, calibration/error metrics, turnover/cost stress,
drawdown and sizing simulation, reproducible seed, and an approval checkpoint.

Until then the system is a research/risk-control product, not a proven profitable
strategy. The reported BTC risk-control observation is retained as evidence to be
revalidated forward, not as a guarantee.
