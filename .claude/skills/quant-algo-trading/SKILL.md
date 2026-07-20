---
name: quant-algo-trading
description: Router for the quantitative trading skill library — regime detection, position sizing, risk, backtest validation, performance metrics and cross-asset instrument knowledge for USD/COP (production), XAU/USD and BTC/USDT (experimental). Use when asked where to start, which skill applies, how to size a position, whether a backtest is real, how to compute Deflated Sharpe, carry or roll yield, vol targeting, or drawdown limits. Do NOT use for US single-name equity screening — that library was deliberately not imported.
---

# Quant algo trading — router

## Read this before using anything below

The imported skills were written for a **US discretionary equity trader**. They know nothing
about this repo's rules. `.claude/rules/quant-constitution.md` **overrides every one of them**:

1. **No grid search over the OOS period.** Parameters are ex-ante priors. Report full
   sensitivities; picking the best cell is prohibited.
2. **No edge claim without trial-aware Deflated Sharpe > 0.95**, computed with
   `services/common/metrics.py::deflated_sharpe_ratio`. Every version, grid and gate looked at
   counts as one trial.
3. **Mandatory baselines before PROMOTE**: B1 (buy & hold), B1′ (exposure-matched), the track's
   dumb baseline, and cost stress ×2.
4. **No Sharpe or p-value with N < 20 trades** — report count and PnL only.
5. **Calmar is the primary graduation metric.** Sharpe is secondary.

**Anything a library skill outputs is a hypothesis, never a decision.**

## Where to go

| Intent | Skill |
|---|---|
| Is this backtest real? Purged CV, PBO, DSR | `xasset-alpha-engine` |
| Correlation, shrinkage, bootstrap CI, regression | `statistics-fundamentals` |
| EWMA / GARCH / realized vs implied vol | `volatility-modeling` |
| Realized risk: Parkinson, Yang-Zhang, drawdown duration | `historical-risk` |
| VaR / ES / Monte Carlo / stress | `forward-risk-var` |
| How big should this position be? Kelly, vol-scaled | `bet-sizing` |
| Sharpe, Sortino, **Calmar** | `performance-metrics` |
| TWR vs MWR, log returns, period conversion | `return-calculations` |
| Carry, IRP, forward points, cross rates | `currencies-and-fx` |
| Contango, backwardation, roll yield | `commodities` |
| Funding, on-chain, NVT | `digital-assets` |

Repo-native workflows live in their own skills — use them, not a library substitute:
`run-experiment` (5 seeds, frozen config), `onboard-asset`, `approval-cycle`, `data-recovery`,
`contract-change`, `dag-change`.

## By asset

- **USD/COP** — `currencies-and-fx` (carry/IRP) + `xasset-alpha-engine`.
  ⚠️ `xasset-alpha-engine/scripts/xasset/instruments.py` registers `USDMXN`, `USDBRL`, `USDZAR`,
  `USDTRY` but **not `USDCOP`**. Adding it is the first concrete task.
- **XAU/USD** — `commodities` (roll yield) + `xasset-alpha-engine` (`XAUUSD`, `MGC`).
- **BTC/USDT** — `digital-assets` + `xasset-alpha-engine` (`BTCUSDT.P` perp funding).

## The rest of the library

106 more skills are vendored, **not discovered**, at `vendor/quant-skills/`. They are
**read-only reference — do not run them**: most need paid US-equity APIs (FMP/FINVIZ) and none
have point-in-time snapshots, so any backtest built on them silently uses restated data.
Classification and rejection reasons: `vendor/quant-skills/PROVENANCE.md`.

## Quarantined — never promote

| Skill | Why |
|---|---|
| `crypto-trading-signals` | Third-party API emitting entry/SL/TP/**leverage** with self-reported confidence. Unauditable, no trial registry, no DSR |
| `backtest-expert` (as-is) | Its `evaluate_backtest.py` scores a **pure in-sample curve fit** as "94/100 — Deploy"; OOS is not even an input |
| `strategy-pivot-designer` | Exists to iterate until the OOS improves — the prohibited loop |
| `edge-signal-aggregator` + `signal-postmortem` | Refit weights on realized outcomes = fitting on the evaluation period |
| `asset-allocation` (as-is) | Mean-variance / Black-Litterman on an estimated covariance is in-sample optimization |

Enforced by `tests/regression/test_quant_library_gate.py`.

## DO NOT

- Do NOT treat a library skill's verdict as a decision — it is a hypothesis.
- Do NOT run vendored scripts against live APIs and call the result a backtest.
- Do NOT promote a vendored skill without a `constitution_reviewed` row in `PROVENANCE.md`.
- Do NOT add a second Deflated Sharpe implementation — `services/common/metrics.py` is the SSOT.
- Do NOT import US-equity screeners; they were rejected deliberately, not overlooked.
