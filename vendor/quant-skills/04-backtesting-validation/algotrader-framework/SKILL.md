---
name: algotrader-framework
description: Scaffolding framework and production knowledge base for building Python trading bots on Indian equity markets (NSE) via the Zerodha Kite API. Use when generating a trading bot skeleton, fetching Nifty index universes, choosing intraday vs positional structure, or avoiding known production failures (tick size rounding, VWAP daily reset, stop-loss lifecycle, position reconciliation, symbol cooldown). Applicable when user mentions Zerodha, kiteconnect, NSE, Nifty, or backtest-live parity.
---

# AlgoTrader Framework

A bot-scaffolding CLI plus a large body of documented production learnings for systematic trading on Indian equities. The value is concentrated in the knowledge files and the reference implementation — the CLI itself is a thin generator.

## Scope and Limits

**This framework is NSE/Zerodha-specific.** Session times (9:15–15:30 IST), circuit breakers, T+1 cash settlement, STT on the sell side, ₹ tick-size tiers, and the Nifty index universes are all hardcoded assumptions. For US or crypto markets, the signal logic and NUANCES gotchas about VWAP, candle completion, and stop-loss lifecycle still transfer; the broker, timing, and tax code do not.

**It is a scaffold, not a trading engine.** There is no backtest engine, no order router, and no live event loop. Generated bots are skeletons with `TODO` markers.

## When to Use This Skill

Use when:
- Starting a new Zerodha/NSE trading bot and wanting a sane file layout and battle-tested default parameters
- Fetching current Nifty 50/100/Midcap 150/Smallcap 250 constituents into JSON
- Deciding between intraday and positional structure
- Reviewing existing trading code against a list of known production failure modes
- Diagnosing a backtest-vs-live win-rate gap

Do not use for: broker-agnostic backtest methodology (use `backtest-expert`), or non-Indian markets without heavy adaptation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv/Scripts/activate
pip install -r requirements.txt
python algotrader.py help
```

Dependencies: `polars`, `kiteconnect`, `requests`, `beautifulsoup4`, `python-dotenv`, `structlog`. The CLI hard-fails on startup if the first four are missing.

`run.sh` and `start.sh` are bash wrappers that auto-create the venv. They assume `venv/bin/python` and will not work on Windows outside Git Bash — call `python algotrader.py` directly there.

Live trading needs Zerodha credentials in a `.env` file (`KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`) in the bot directory, never committed.

## Workflow

### 1. Read the Gotchas First

Read `NUANCES.md` before writing any code. It is 30 numbered production failures with the exact fix for each, and it is the single highest-value file here. The first ten are the ones that cause real money loss.

### 2. Build the Universe

```bash
python algotrader.py universe --indices nifty50,midcap150
```

Writes `universe/<index>.json`. Data is scraped from the NSE JSON endpoint with a cookie-priming session request; NSE blocks bots intermittently, in which case it silently falls back to a hardcoded constituent list from Jan 2026. **Check `last_updated` in the output** before trusting it for a live universe.

### 3. Scaffold the Bot

```bash
python algotrader.py wizard
```

Prompts for trading style, universe, strategy (momentum / mean reversion / fortress), capital, and risk tolerance, then writes `trading_bot_<timestamp>/` containing `config.json`, `main.py` (copied from the matching template), and `universe/`. Requires an interactive TTY.

Default parameters come from `get_default_parameters()` in `algotrader.py` and are the tested values from `KNOWLEDGE.md`: `rsi_long_min=45`, `rsi_long_max=65`, `adx_min=25`, `ema_fast=9`, `ema_slow=21`, `volume_mult=1.5`, `symbol_cooldown_minutes=45`. Intraday adds `target_pct=1.0` / `stop_loss_pct=0.5` / `max_hold_minutes=45`; positional uses `8.0` / `4.0` / `max_hold_days=30`.

### 4. Fill In the Logic

The generated `main.py` is a skeleton. Port the working implementation from `examples/full_system.py` — that is where the real signal generator lives.

### 5. Validate Before Going Live

Run the pre-flight checklist at the end of `NUANCES.md`. Paper trade first. Compare paper results against backtest and treat any win-rate gap as a parity bug, not variance.

## Choosing a Template

**`templates/minimal_intraday.py`** — same-day entries and exits, 1-second scan loop. Pick this when holds are measured in minutes to hours. It already implements `can_trade_symbol()` (45-min cooldown), `should_trade_now()` (blocks the 11:30–13:00 lunch lull and out-of-session times), and `round_to_tick()`. The trading loop body is `TODO`.

**`templates/minimal_positional.py`** — weekly rebalance on Friday after 15:00, holds of weeks. Pick this for momentum-ranked portfolio rotation. Much thinner than the intraday template; the entire rebalance body is `TODO`.

Swing trading has parameter defaults in the CLI but no template — the wizard routes anything non-intraday to the positional template.

## API Surface

**`algotrader.py` is a flat script, not an importable package.** The `from algotrader import ...` and `from algotrader.risk import ...` examples in `README.md` do not correspond to any shipped module — treat them as design sketches, not working code.

What actually exists:

| Command | Status |
|---|---|
| `wizard` (or no args) | Working — full interactive generator |
| `universe [--indices a,b]` | Working — fetches NSE, falls back to hardcoded list |
| `signal <SYMBOL>` | Stub — prints a placeholder |
| `check <file.py>` | Stub — prints a placeholder |
| `optimize <file.py>` | Stub — prints a placeholder |
| `help` | Working |

`skill.json` also advertises a `backtest` command and a `fix <issue>` command; neither is implemented in `main()`. The README's `compare_backtests()`, Kelly sizing, regime detection, and analytics examples are aspirational — the formulas are documented in `KNOWLEDGE.md`, but no code ships for them.

Real callable functions, in `examples/full_system.py`:

- `generate_fortress_signal(df, symbol, config) -> Optional[Signal]` — the 6-factor confirmation signal. Takes a Polars DataFrame that must already carry `open, high, low, close, volume, ema9, ema21, rsi, adx, vwap, avg_volume, atr`. Confidence accumulates across trend (0.25), ADX strength (0.15), RSI bounds (0.10), VWAP confluence (0.15), volume surge (0.10), and candle body. Trend, strength, and momentum are hard gates that return `None`.
- `can_trade_symbol`, `should_trade_now`, `is_candle_complete`, `round_to_tick`, `get_tick_size`
- `Signal` dataclass — symbol, direction, entry/stop/target, confidence, reason, strategy, timestamp, indicators

`fetch_index_constituents(index_name)` and `get_fallback_universe(index_name)` live in `algotrader.py`; `examples/universe_fetcher.py` is a standalone variant.

## Critical Gotchas

Condensed from `NUANCES.md` — read the file for the code fixes.

**Order rejection**: Round every price to the instrument's tick size (`round(price / tick) * tick`). Tick varies by price band (₹0.05 typical, ₹5.00 for high-priced names). This is cited as 90% of rejections. Re-download `https://api.kite.trade/instruments` daily; tick and lot sizes change.

**Naked positions**: When moving a stop loss, place the new order *then* cancel the old one. Cancel-then-place leaves the position unprotected if the placement fails. Two simultaneous stops is a safe failure mode; zero is not.

**Backtest-live divergence**: The top three causes are VWAP not resetting at 9:15 (it is a daily indicator, cumulative VWAP is meaningless), acting on incomplete candles (require a 500ms buffer past the candle boundary, else use `df[-2]`), and different data sources between backtest and live. Historical data is split-adjusted; live ticks are not.

**Restart correctness**: Reconcile positions against `kite.positions()['net']` on every startup. Zerodha is the source of truth; the local state file can be stale from a crash or a manual trade in the Kite app.

**Churn**: Enforce a 45-minute per-symbol cooldown after any exit. Without it, a marginal signal re-fires immediately after a stop-out and repeats the loss.

**Margin checks**: Use `margins['equity']['net']`, not `opening_balance`. The latter ignores margin already deployed.

**Indicator misuse**: ADX measures trend strength only. Gate on `adx > 25`, then take direction from the EMA relationship. Using ADX directionally drops win rate from ~58% to ~35%.

**Risk limits**: Size from stop distance, capped at 1% of capital per trade. Halve size after two consecutive losses; stop for the day after three.

**Performance**: Cache OHLCV as Parquet, not JSON (~28x). Vectorize with Polars expressions rather than row loops (~37x). Batch `kite.quote()` calls — Zerodha rate-limits at 3 req/sec.

## Bundled Files

- `NUANCES.md` — 30 production failures with fixes and a pre-launch checklist. Read first.
- `KNOWLEDGE.md` — 1,780 lines across 16 domains: exact indicator formulas, tested parameter values, rebalancing cost analysis, exit strategies, Kelly sizing, regime detection.
- `README.md` — architecture and intended feature set. Verify any code sample against the shipped files before relying on it.
- `QUICKSTART.md` — five-minute path; contains paths from the original author's machine.
- `examples/full_system.py` — the reference implementation. Most useful file after `NUANCES.md`.
- `examples/universe_fetcher.py` — standalone NSE constituent fetch.
- `templates/minimal_intraday.py`, `templates/minimal_positional.py` — bot skeletons.
- `SKILLS_SH_GUIDE.md`, `START_SCRIPT_GUIDE.md` — distribution and launcher docs, not needed to use the framework.

## Reminders

Performance figures quoted throughout (65% win rate, 51% CAGR) are the original author's backtest results on a specific universe and period. They are not reproducible guarantees — revalidate any strategy on your own data before committing capital.

The generated bot's `--mode live` flag is not implemented in the templates. Do not assume paper and live paths exist until you have written them.

`README.md` references a `LICENSE` file that is not present in this directory.
