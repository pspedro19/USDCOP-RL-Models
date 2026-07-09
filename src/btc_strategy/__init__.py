"""BTC/USDT spot-exposure strategy — science stack (materializes design/SPEC-01..10, honest baselines first).

Crypto-adapted mirror of the Gold science stack, with the structural differences that BTC is
**24/7 and spot-only**:
  - annualization uses **√365** (not √252) — see backtest.ANNUALIZE
  - exposure is a **spot fraction ∈ [0, 1]** (never negative, never > 1; no liquidation by design)
  - no overnight swap; cost is taker-fee-on-turnover (rebalancing the spot book)

This is a NEW, parallel package — it does not import or touch src/gold_rl.
"""
from . import backtest, indicators, strategies  # noqa: F401

__all__ = ["indicators", "strategies", "backtest"]
