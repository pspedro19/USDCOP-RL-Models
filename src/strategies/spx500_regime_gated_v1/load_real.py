"""Real-data loader for the SPX500 engine — replaces the synthetic scaffold.

Contract: CTR-SPX500-REALDATA-001

The engine was wired against `datagen.generate()`, which its own docstring labels
"SINTÉTICO ... NO evidencia de alfa". Meanwhile a real SPY snapshot already existed at
`data/snapshots/public_daily/spx500_daily.parquet` (1,643 rows, 2020-01-01 → 2026-07-17, with
an `available_at` column). Nothing joined the two, so every metric the strategy produced
measured the wiring, not the market.

This module produces the SAME column contract the engine expects (SDD-006), so the rest of the
pipeline — regimes, policies, gates — is untouched:

    open_to_open_return : realized return open_t → open_{t+1}   (PnL)
    close               : total-return price level              (signals)
    vix                 : risk state
    macro_stress        : exogenous macro driver

## What is honest here and what is not

**Honest:** the price. `adj_close` is dividend-adjusted, which is the total-return series
SDD-000 §4 requires — using `^GSPC` price-only would inflate alpha by ~1.8%/yr.

**NOT honest, and marked as such:** `vix` and `macro_stress`. FRED:VIXCLS, NFCI and HY-OAS are
not in this snapshot. Rather than silently substituting a synthetic driver — which would let a
"real data" run quietly keep a fabricated regime signal — this loader DERIVES a volatility
proxy from the price itself and flags it. A regime classifier fed realized vol instead of
implied vol is a different, weaker model, and the artifact says so.

**`available_at` is present but reconstructed** (the public adapter stamps `close + 1 day`),
so it is not a true point-in-time vintage. Per the constitution the maximum status this can
reach is `research_validated`, never `production`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / "data" / "snapshots" / "public_daily" / "spx500_daily.parquet"

# Realized-vol window used for the VIX proxy. 21 trading days ~ one month, matching VIX's
# 30-calendar-day horizon. Declared here as a prior, not tuned.
VOL_WINDOW = 21
ANNUALIZER = np.sqrt(252.0)


class RealDataUnavailable(RuntimeError):
    """Raised instead of falling back to synthetic data.

    A silent fallback is how a demo scaffold ends up producing numbers someone quotes.
    """


def load_real(*, snapshot: Path | None = None) -> pd.DataFrame:
    path = snapshot or SNAPSHOT
    if not path.is_file():
        raise RealDataUnavailable(
            f"No SPY snapshot at {path}. Refusing to fall back to datagen.generate(): the "
            "engine would run and report metrics that describe a synthetic series. Acquire "
            "the snapshot first (scripts/data/acquire_public_snapshots.py --asset spx500)."
        )

    d = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "adj_close"):
        if col not in d.columns:
            raise RealDataUnavailable(f"snapshot lacks required column `{col}`")

    out = pd.DataFrame(index=range(len(d)))
    out["timestamp"] = pd.to_datetime(d["timestamp"])
    out["available_at"] = pd.to_datetime(d["available_at"]) if "available_at" in d else pd.NaT

    # Total-return level drives the signals (SDD-000 §4).
    out["close"] = d["adj_close"].astype(float).to_numpy()

    # PnL is open-to-open: enter at tomorrow's open on today's close signal. Computing it any
    # other way would let a signal act on a price it could not have traded at.
    op = d["open"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o = np.append(op[1:] / op[:-1] - 1.0, np.nan)
    out["open_to_open_return"] = np.nan_to_num(o2o, nan=0.0)

    # PROXY, not VIX. Realized vol of total-return closes, annualized, shifted one day so a
    # bar never sees its own volatility.
    ret = pd.Series(out["close"]).pct_change()
    out["vix"] = (ret.rolling(VOL_WINDOW).std() * ANNUALIZER * 100.0).shift(1).bfill().to_numpy()

    # PROXY, not NFCI/HY-OAS. Z-score of the vol proxy: a crude "is stress elevated" driver.
    v = pd.Series(out["vix"])
    out["macro_stress"] = ((v - v.rolling(252).mean()) / v.rolling(252).std()).shift(1) \
        .fillna(0.0).to_numpy()

    out.attrs["data_class"] = "real_price_proxy_drivers"
    out.attrs["proxies"] = {
        "vix": f"realized vol {VOL_WINDOW}d annualized (FRED:VIXCLS absent from snapshot)",
        "macro_stress": "252d z-score of the vol proxy (FRED:NFCI / HY-OAS absent)",
    }
    out.attrs["point_in_time"] = False
    out.attrs["pit_note"] = (
        "available_at is reconstructed by the public adapter (close + 1d), not a provider "
        "vintage; max attainable status is research_validated"
    )
    return out


def load(*, real: bool = True) -> pd.DataFrame:
    """Entry point for the pipeline. `real=False` must be an explicit, visible choice."""
    if real:
        return load_real()
    from src.strategies.spx500_regime_gated_v1 import datagen
    df = datagen.generate()
    df.attrs["data_class"] = "SYNTHETIC"
    return df
