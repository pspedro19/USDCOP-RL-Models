---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Quantitative multi-asset harness

`src.validation.quant_harness` is the deterministic contract layer for USDCOP,
XAUUSD, BTCUSDT and SPX500. It validates schema, monotonic timestamps,
point-in-time `available_at`, finite data, sample size, forecast MAE/RMSE/
directional accuracy versus a baseline, net strategy returns, Sharpe,
drawdown, turnover and declared transaction costs. DSR/PBO is intentionally a
blocking evidence check: the experiment manifest must provide trial count and
the computed deflated statistics before promotion.

Run with `python scripts/validation/run_quant_harness.py --input input.json --output evidence.json`.
The JSON output is immutable evidence for the checkpoint ledger; a failed check
must remain NO-GO and trigger the relevant agent (data, statistics, strategy or
production) rather than being overridden by the CLI.
