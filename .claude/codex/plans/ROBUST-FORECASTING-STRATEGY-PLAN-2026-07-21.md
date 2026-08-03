---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Robust forecasting and per-asset strategy plan

## Integrity baseline

The alleged active Ridge/BR/XGBoost divergence was a false alarm: the YAML already
used Ridge + BayesianRidge and XGBoost was offline-only. The verification still
justified two controls now required everywhere:

1. Missing ensemble configuration fails closed to the promoted set; no code default
   may silently enable XGBoost.
2. Repository identity, SSOT, approval bundle and serving config must agree in one
   structural CI test.

Agent reports are hypotheses until checked against source, artifacts and execution
evidence.

## Promotion sequence

1. Freeze an immutable strategy manifest per asset (features, timezone, frequency,
   models/rules, costs, execution lag, seed and code hash).
2. Validate data lineage and PIT availability before any directional claim.
3. Run purged/embargoed walk-forward OOS; report DA only diagnostically and report
   calibration, volatility error, drawdown, turnover and costs.
4. Run forward paper execution with realistic next-open/next-bar fills and a weekly
   ledger.
5. Require DSR/PBO or equivalent multiple-testing controls and an explicit approval
   checkpoint. Fail closed on missing or contradictory metadata.

## Recommended strategy surface by asset

- **USDCOP:** weekly directional forecast for regime/context only; M5 execution
  during 08:00–12:55 America/Bogota. Prefer volatility forecast plus conservative
  sizing; require point-in-time macro vintages before promotion.
- **XAUUSD:** causal daily trend/momentum (UTC-anchored), Parkinson/range volatility
  for sizing, next-session execution. Keep rules fixed and judge them on forward
  results rather than fitting a new model.
- **BTCUSDT:** daily UTC regime and volatility forecast; use funding/open-interest/
  basis as crypto-native features. Treat √365 annualization explicitly and avoid
  mixing it with a 252-day portfolio clock without a declared conversion.
- **SP500:** daily signal on completed session, next-open execution, 252 trading-day
  clock, volatility targeting and drawdown breaker. Do not use intraday assumptions
  until a separate session model is validated.

## Definition of done

Directional weekly forecasts become executable strategies only after each asset has
an immutable manifest, PIT/OOS evidence, paper-forward ledger, cost stress, risk
limits, monitoring and rollback. Until then they remain transparent research or
context inputs to sizing.
