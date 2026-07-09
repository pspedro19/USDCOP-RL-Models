"""Gold (XAU/USD) RL strategy — science modules (SPECSGOLD).

Implements the roadmap on top of the ingested, tz-aligned Gold data:
  indicators.py  — SPEC-03 features + SPEC-04 regime classifier (rules v1 + hysteresis)
  strategies.py  — SPEC-06 risk (vol-targeting) + SPEC-07 baselines + regime-gated strategy
  backtest.py    — SPEC-09 walk-forward backtest, metrics, gates, regime attribution, bootstrap
Orchestrated by scripts/run_gold_pipeline.py which publishes bundles to the dynamic registry
(SPEC-12) so the Gold asset is visible + replayable on the dashboard.
"""
