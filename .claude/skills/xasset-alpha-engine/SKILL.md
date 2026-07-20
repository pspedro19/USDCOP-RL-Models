---
name: xasset-alpha-engine
description: "Cross-asset quantitative engine for FX (majors and emerging markets), crypto, equity indices and commodities. NOTE: Deflated/Probabilistic Sharpe here are thin wrappers over services/common/metrics.py, which is the constitutional SSOT and the live release gate. Builds trend/carry/value signals on one comparable scale, sizes positions in correct instrument units (pips and lots for FX, contracts for futures, coins, shares), targets portfolio volatility with a covariance matrix, and validates backtests with purged cross-validation, Deflated Sharpe and PBO. Use when the user asks about EURUSD, USDMXN, USDBRL, USDZAR, USDTRY or any FX pair; about carry trades or interest rate differentials; about perpetual funding rates; about gold, XAUUSD or futures roll yield; about sizing a position in something that is not a US stock; about volatility targeting or risk parity across asset classes; or about whether a backtest is overfit. Also trigger on 'how many lots', 'pip value', 'is this backtest real', 'deflated sharpe', 'probability of backtest overfitting', 'emerging market carry', 'cross-asset portfolio'. Do NOT use for single-name US equity screening - the 02-screening-signals skills cover that better."
---

# Cross-Asset Alpha Engine

A quantitative engine that treats FX, crypto, equities and commodities as one
book rather than four screeners glued together.

## Why this exists

The rest of this collection is strong on US equities and thin everywhere else.
An audit found 27 of 35 signal-generating skills are structurally equity-only,
no FX spot data source exists anywhere in 118 skills, emerging-market FX
amounts to four lines of prose, and the position sizer returns *shares* for
every instrument — silently wrong for a 100,000-unit FX lot or a 100oz gold
contract.

This skill covers that gap, and refuses three specific failure modes found in
the existing tooling:

| Failure found | What this does instead |
|---|---|
| `evaluate_backtest.py` returns "Deploy" for an in-sample curve fit — out-of-sample is not an input | `validate()` takes OOS returns and trial count as **required** arguments |
| `continuous_kelly()` is uncapped; a shipped example recommends 18.5% risk on one trade | `fractional_kelly()` hard-rejects any fraction above 0.5 and caps output |
| Sizing returns "shares" regardless of instrument | Every position names its own unit; unknown symbols raise rather than defaulting |

## The core idea

Chart patterns do not transfer between a 24-hour FX pair, a perpetual swap, an
index future and a gold contract. **Carry does.** Every one of them has a
well-defined return-if-nothing-moves:

| Asset class | Carry is |
|---|---|
| FX (majors and EM) | interest rate differential, or forward points |
| Crypto perpetual | funding rate paid between longs and shorts |
| Equity index | dividend yield minus financing rate |
| Commodity | roll yield from the futures curve |

All four are annualised returns to a long position, so they rank in one
cross-section. Add time-series momentum (which has cross-asset evidence across
58 futures markets) and per-class value, scale everything by volatility, and
four asset classes become one portfolio.

## Quick start

```bash
cd scripts
python run_engine.py --verify     # self-check, exits 1 on mismatch
python run_engine.py --demo       # offline walkthrough of all four stages
python run_engine.py --live       # free keyless data: yfinance + ccxt + FRED
python run_engine.py --size EURUSD --entry 1.0850 --stop 1.0800 --risk 500
python -m pytest tests/ -q        # 74 tests
```

## Modules

| Module | Owns |
|---|---|
| `xasset/instruments.py` | Instrument registry and unit-correct position sizing |
| `xasset/carry.py` | Carry for each asset class, on one scale |
| `xasset/signals.py` | TSMOM trend, per-class value, signal combination |
| `xasset/sizing.py` | Volatility targeting, risk parity, capped Kelly |
| `xasset/validation.py` | Purged CV, PBO, and **thin wrappers** over the repo's PSR/DSR SSOT |
| `xasset/data.py` | Free keyless adapters, calendar alignment |

## Workflows

### Size a position in anything

```python
from xasset.instruments import get_instrument

get_instrument("EURUSD").size_from_risk(1.0850, 1.0800, risk_budget=500)
# 100,000 base_ccy_units (1.000 lots) | risk at stop 500.00

get_instrument("GC").size_from_risk(2400, 2380, risk_budget=5000)
# 2 contracts | risk at stop 4,000.00   <- floored; realised risk is BELOW budget
```

Always read `risk_at_stop`, not `risk_budget`. For futures they differ, because
a partial contract does not exist. When the budget cannot afford one contract,
`is_tradeable` is False rather than returning a fractional position.

For non-USD quote currencies pass `fx_rate_to_account`. Trading USDJPY from a
USD account needs `1/USDJPY`; getting it wrong scales the position by ~150x,
which is why it is explicit rather than inferred.

### Build a cross-asset book

```python
from xasset.data import fetch_prices
from xasset.sizing import realised_vol, vol_target_weights
from xasset.signals import blended_tsmom, Signal, combine_signals

pf = fetch_prices(["EURUSD", "USDMXN", "XAUUSD", "SPX", "BTCUSD"], period="5y")
rets = pf.align("intersect").returns()          # <- mandatory, see below
cov  = rets.cov().to_numpy() * 252
vols = {c: realised_vol(rets[c], halflife=60) for c in rets.columns}
sigs = [Signal(c, blended_tsmom(pf.prices[c].to_numpy()).value, "trend")
        for c in rets.columns]
book = vol_target_weights(combine_signals(sigs, {"trend": 1.0}), vols,
                          cov=cov, target_vol=0.10)
```

**Always `.align()` first.** Crypto trades 24/7, FX 24/5, equities on exchange
hours. Measured on 3 years of EURUSD/USDMXN/gold/SPX/BTC: only **68.5% of dates
are common to all five**, and forward-filling understates annualised volatility
by **17–19% on every asset**. Feed that into a vol target and every position is
~20% too large. `returns()` refuses ragged input unless you override it.

Pass `cov` whenever you can. Long gold, short dollar and long EM FX is one
dollar trade wearing three hats; without the covariance term the reported
portfolio vol is a lower bound.

### Validate a backtest

```python
from xasset.validation import validate
verdict = validate(oos_returns, trial_sharpes=every_sharpe_you_computed)
print(verdict.report())
```

Pass **every** trial Sharpe, including the discarded ones. That count is the
entire point — Deflated Sharpe haircuts by how hard you looked. Reporting only
the winner is the misconduct the metric exists to detect.

For strategy selection across variants, add `returns_matrix` (T × N) to get
PBO. Above 0.5 means your selection procedure is worse than random: the
in-sample winner tends to be the out-of-sample loser.

For model training, use `purged_kfold_splits` rather than plain K-fold. Any
strategy with a multi-day holding period has overlapping labels, and plain
K-fold leaks across the fold boundary, inflating out-of-sample Sharpe.

## Emerging markets

EM is where this differs most from the rest of the collection. See
`references/em-fx.md`. Three things the engine does automatically:

1. **Nominal carry is marked suspect when real carry is negative.** A 45%/yr
   TRY carry against 60% inflation is compensation for expected depreciation,
   not free money. Suspect estimates are excluded from the cross-sectional
   z-score by default — one 40% carry would otherwise dominate the ranking and
   pull the whole book toward the position most likely to gap.
2. **Forward-implied carry is preferred over rate differentials.** Where
   capital controls or NDF markets bind, covered interest parity breaks and the
   tradeable forward is the honest price.
3. **NDF and capital-control flags** on USDBRL, USDINR, USDCNH and USDTRY.

EM carry has strong negative skew. It pays steadily and then gaps. Volatility
targeting sizes on realised vol and therefore *understates* EM tail risk —
treat the vol-targeted weight as a ceiling, not a recommendation.

## Data

Free and keyless by design. The audit found 33 skills (28% of the collection)
bound to a single vendor that has already retired one endpoint, and 8 skills
depending on one individual's personal GitHub repo. This adds nothing to that.

| Source | Covers | Notes |
|---|---|---|
| yfinance | FX majors + EM, gold, indices, crypto daily | Indicative daily close; not tradeable rates, not intraday |
| ccxt | Crypto spot and **funding rates** | Funding is the carry leg |
| FRED CSV | US rates, 10y real yield | No key. Non-US coverage is patchy and lagged |

`data.TWELVEDATA_DRAFT` holds an unimplemented adapter design, deliberately
inert. It is deferred until this skill moves to the repo with the scraping and
API tooling, and it is a string constant rather than a stub function so it
cannot be mistaken for working code.

## Limits

Stated plainly, because the failure this collection exhibits most is
documentation describing code that does not exist.

- **No execution.** Signals and sizes only — no broker, no order router.
- **No intraday.** Daily bars. The free FX close is an indicative snapshot.
- **No transaction costs or market impact in the sizing path.** Nothing here
  estimates capacity. A vol-targeted EM book at size will not fill where the
  backtest assumes.
- **Value signals are weaker than trend and carry.** PPP has no timing content
  at all, and crypto has no consensus value definition — there is deliberately
  no crypto value function rather than an invented one.
- **`validate()` gates statistical plausibility, not economic sense.** It
  cannot tell you the signal is a proxy for something you already own.
- **Contract specs in the registry are the standard published values.** Verify
  against the exchange before trading; multipliers get revised.

## Related skills

- `05-risk-position-sizing/futures-position-sizer` — deeper per-symbol CME specs
- `10-instruments-valuation/currencies-and-fx` — CIP/UIP/PPP theory and worked examples
- `01-market-regime/crypto-regime-analyzer` — crypto regime scoring, keyless
- `02-screening-signals/multi-asset-trading-signals` — prose reference on FX and commodities
- `04-backtesting-validation/backtest-expert` — backtest methodology (see its OOS caveat above)
