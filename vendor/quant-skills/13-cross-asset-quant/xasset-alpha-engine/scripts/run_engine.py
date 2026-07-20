#!/usr/bin/env python3
"""Cross-asset alpha engine - runnable entry point.

Builds a vol-targeted book across FX (majors + EM), crypto, equities and gold
from trend + carry, then sizes one position in correct instrument units.

    python run_engine.py --demo             # offline, no network
    python run_engine.py --live             # fetches free data
    python run_engine.py --verify           # self-check, exits 1 on mismatch
    python run_engine.py --size EURUSD --entry 1.0850 --stop 1.0800 --risk 500

Every number printed is computed here. Nothing is illustrative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xasset.carry import (  # noqa: E402
    carry_to_zscore, commodity_roll_carry, crypto_perp_carry,
    equity_index_carry, fx_carry,
)
from xasset.instruments import get_instrument  # noqa: E402
from xasset.signals import Signal, blended_tsmom, combine_signals  # noqa: E402
from xasset.sizing import realised_vol, vol_target_weights  # noqa: E402
from xasset.validation import validate  # noqa: E402


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def demo_carry() -> dict[str, float]:
    """Carry across four asset classes, on one comparable scale."""
    _rule("1. CARRY - one definition, four asset classes")
    ests = [
        fx_carry("EURUSD", rate_base=0.0215, rate_quote=0.0433),
        fx_carry("AUDUSD", rate_base=0.0435, rate_quote=0.0433),
        fx_carry("USDMXN", rate_base=0.0433, rate_quote=0.1025, is_em=True,
                 annual_inflation_base=0.0290),
        fx_carry("USDTRY", rate_base=0.0433, rate_quote=0.4500, is_em=True,
                 annual_inflation_base=0.0290),
        crypto_perp_carry("BTCUSDT.P", funding_rate=0.000062),
        equity_index_carry("ES", dividend_yield=0.0128, financing_rate=0.0433),
        commodity_roll_carry("GC", front_price=4015.0, back_price=4048.0,
                             days_between=60),
    ]
    for e in ests:
        print(f"  {e}")

    print("\n  Cross-sectional z-scores (suspect estimates excluded):")
    z = carry_to_zscore(ests)
    for k, v in sorted(z.items(), key=lambda x: -x[1]):
        print(f"    {k:<12} {v:+.2f}")
    return z


def demo_portfolio() -> None:
    """Vol targeting across assets whose volatilities differ by ~9x."""
    _rule("2. VOL TARGETING - why equal notional is not a portfolio")
    signals = {"EURUSD": 0.8, "USDMXN": -0.6, "XAUUSD": 1.0, "SPX": 0.5, "BTCUSD": 0.7}
    vols = {"EURUSD": 0.07, "USDMXN": 0.12, "XAUUSD": 0.21, "SPX": 0.15, "BTCUSD": 0.48}

    print("  signal x conviction, and each asset's annualised vol:")
    for s in signals:
        print(f"    {s:<10} signal {signals[s]:+.2f}   vol {vols[s]:.0%}")

    res = vol_target_weights(signals, vols, target_vol=0.10)
    print(f"\n{res.report()}")
    print(
        "\n  Note EURUSD carries the largest weight despite a mid-sized signal:"
        "\n  at 7% vol it needs leverage to contribute its share of risk. Equal"
        "\n  notional would have made this a bitcoin fund with decorations."
    )


def demo_sizing() -> None:
    """The same risk budget, four asset classes, four different units."""
    _rule("3. POSITION SIZING - correct units per asset class")
    budget = 1000.0
    cases = [
        ("EURUSD", 1.0850, 1.0800, 1.0),
        ("USDMXN", 17.4100, 17.6000, 1 / 17.41),
        ("XAUUSD", 4015.0, 3975.0, 1.0),
        ("GC", 4015.0, 3975.0, 1.0),
        ("SPY", 660.0, 645.0, 1.0),
        ("BTCUSD", 65_447.0, 62_000.0, 1.0),
    ]
    print(f"  Risk budget {budget:,.0f} account-currency units per position.\n")
    for sym, entry, stop, fx in cases:
        pos = get_instrument(sym).size_from_risk(entry, stop, budget, fx)
        print(f"  {pos.describe()}")
    print(
        "\n  Four different units. A sizer that returns 'shares' for all six"
        "\n  is not being approximate - it is wrong in a way that looks fine."
    )


def demo_validation() -> None:
    """The gate that the collection's evaluate_backtest.py does not have."""
    _rule("4. VALIDATION - refusing an overfit backtest")
    rng = np.random.default_rng(20260720)

    print("  Case A: a good-looking curve, selected from 300 trials")
    r = rng.normal(0.0008, 0.008, 900)
    v = validate(r, trial_sharpes=list(rng.normal(0.0, 0.05, 300)))
    print("  " + v.report().replace("\n", "\n  "))

    print("\n  Case B: a genuine edge, 5 trials")
    r2 = rng.normal(0.0016, 0.006, 2500)
    v2 = validate(r2, trial_sharpes=list(rng.normal(0.0, 0.01, 5)))
    print("  " + v2.report().replace("\n", "\n  "))
    print(
        "\n  Same annualised Sharpe range. The difference is how hard you"
        "\n  looked before you found it - which is the input that the existing"
        "\n  evaluate_backtest.py never asks for."
    )


def run_live(period: str = "5y") -> int:
    """Build the book from free live data."""
    from xasset.data import DataUnavailable, fetch_crypto_funding, fetch_prices

    _rule("LIVE - free keyless sources")
    universe = ["EURUSD", "USDMXN", "USDZAR", "XAUUSD", "SPX", "BTCUSD"]
    try:
        pf = fetch_prices(universe, period=period)
    except (DataUnavailable, KeyError) as exc:
        print(f"  data unavailable: {exc}", file=sys.stderr)
        return 1

    print(pf.calendar_report())
    if pf.failed_symbols:
        print(f"  FAILED: {pf.failed_symbols}")

    aligned = pf.align("intersect")
    rets = aligned.returns()
    print(f"\n  aligned sample: {rets.shape[0]} days x {rets.shape[1]} assets")

    sigs: list[Signal] = []
    vols: dict[str, float] = {}
    print(f"\n  {'symbol':<10}{'vol':>9}{'trend':>9}")
    for c in rets.columns:
        try:
            t = blended_tsmom(aligned.prices[c].to_numpy())
            v = realised_vol(rets[c], halflife=60)
        except ValueError as exc:
            print(f"  {c:<10} skipped: {exc}")
            continue
        sigs.append(Signal(c, t.value, "trend", t.confidence, t.note))
        vols[c] = v
        print(f"  {c:<10}{v:>8.1%}{t.value:>9.2f}")

    try:
        funding = fetch_crypto_funding(("BTC/USDT:USDT",))
        for _, rate in funding.items():
            ce = crypto_perp_carry("BTCUSD", rate)
            print(f"\n  live crypto carry: {ce}")
    except DataUnavailable as exc:
        print(f"\n  funding unavailable ({exc}) - proceeding on trend only")

    conviction = combine_signals(sigs, weights={"trend": 1.0})
    cov = rets[list(vols)].cov().to_numpy() * 252
    res = vol_target_weights(conviction, vols, cov=cov, target_vol=0.10)
    print(f"\n{res.report()}")
    print(
        "\n  Covariance-aware: long gold / short USD / long EM FX is one dollar"
        "\n  trade wearing three hats, and the correlation term prices that in."
    )
    return 0


def verify() -> int:
    """Self-check. Exits non-zero on mismatch."""
    ok = True

    def check(label: str, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {label}: {got:.6f} (want {want:.6f})")

    _rule("VERIFY")
    p = get_instrument("EURUSD").size_from_risk(1.0850, 1.0800, 500.0)
    check("EURUSD 50-pip stop, $500 risk -> units", p.quantity, 100_000.0, 1e-6)
    check("  ... in standard lots", p.lots, 1.0, 1e-9)

    g = get_instrument("GC").size_from_risk(2400.0, 2380.0, 5000.0)
    check("Gold $20 stop, $5000 risk -> contracts", g.quantity, 2.0)
    check("  ... realised risk (floored)", g.risk_at_stop, 4000.0, 1e-6)

    c = fx_carry("EURUSD", 0.02, 0.05).annualised
    check("EURUSD carry at 2% vs 5%", c, -0.03, 1e-12)

    f = crypto_perp_carry("BTC", 0.0001).annualised
    check("Perp carry to long at 1bp/8h", f, -0.1095, 1e-6)

    rng = np.random.default_rng(1)
    v = validate(rng.normal(0.0008, 0.008, 900),
                 trial_sharpes=list(rng.normal(0, 0.05, 300)))
    print(f"  [{'OK' if not v.passed else 'FAIL'}] 300-trial curve fit is rejected")
    ok &= not v.passed

    print(f"\n  {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="offline walkthrough")
    ap.add_argument("--live", action="store_true", help="fetch free live data")
    ap.add_argument("--verify", action="store_true", help="self-check")
    ap.add_argument("--period", default="5y", help="history for --live")
    ap.add_argument("--size", metavar="SYMBOL", help="size a single position")
    ap.add_argument("--entry", type=float)
    ap.add_argument("--stop", type=float)
    ap.add_argument("--risk", type=float)
    ap.add_argument("--fx-rate", type=float, default=1.0,
                    help="quote ccy -> account ccy conversion")
    args = ap.parse_args()

    if args.size:
        if args.entry is None or args.stop is None or args.risk is None:
            ap.error("--size requires --entry, --stop and --risk")
        pos = get_instrument(args.size).size_from_risk(
            args.entry, args.stop, args.risk, args.fx_rate
        )
        print(pos.describe())
        return 0 if pos.is_tradeable else 1

    if args.verify:
        return verify()
    if args.live:
        return run_live(args.period)

    demo_carry()
    demo_portfolio()
    demo_sizing()
    demo_validation()
    print("\nRun --live for real data, --verify for the self-check.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
