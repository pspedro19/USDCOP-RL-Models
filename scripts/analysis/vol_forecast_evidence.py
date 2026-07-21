"""H-VOLF-01 — does HAR-RV beat persistence at forecasting 5-day volatility?

Contract: CTR-QUANT-EVIDENCE-001 · Hypothesis pre-registered 2026-07-21 in all four asset
registries BEFORE this file was written.

Direction is closed (best model x horizon cell: p_adj = 1.0 over 63 cells). The sanctioned
repurpose of forecasting is VOLATILITY, which feeds sizing -- the one lever with demonstrated
value here (2026: every strategy loses less than its underlying).

The honest bar is NOT "does it predict" -- volatility is so autocorrelated that anything
"predicts" it -- but "does it beat persistence" (sigma_hat_{t+5} = sigma_t). If it cannot,
persistence IS the estimator and this avenue closes. That is an acceptable result and, given
that H-VOL-01 (EWMA in the sizer) already failed, the expected one.

## Locked-down design (each knob is a registered prior, not a choice)

  Model      HAR-RV (Corsi 2009): OLS of forward 5d realized variance on realized-variance
             components over 1/5/22 days. Three slopes and an intercept. The 1/5/22 windows
             are THE standard, not swept. No other model is run: 9 models x 7 horizons on a
             new target would be the directional mistake again, at 63 trials.
  Horizon    5 days -- the only horizon the sizing layer consumes.
  Fit        Walk-forward, expanding, refit every 21 obs, strictly causal (fit uses <= t).
  Metric     QLIKE on variance (Patton 2011). RMSE reported alongside, never as the verdict.
  Baselines  Persistence RV-20 and persistence EWMA(0.94) -- always in the same table.
  Verdict    OOS-2025 only. Reject H0 iff mean QLIKE(HAR) < mean QLIKE(EWMA) AND the paired
             block-bootstrap CI95 of the loss differential excludes zero. 2026 is reported
             as context, never as the verdict (it is the forward).

Run: python -m scripts.analysis.vol_forecast_evidence
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from services.common.metrics import qlike_loss  # noqa: E402

H = 5                # forecast horizon in trading days (the sizing window)
REFIT_EVERY = 21     # walk-forward refit cadence
LAM = 0.94           # RiskMetrics daily decay -- prior, not swept
HAR_WINDOWS = (1, 5, 22)   # Corsi's daily/weekly/monthly -- prior, not swept
OOS = 2025

SOURCES = {
    "usdcop": ("seeds/latest/usdcop_daily_ohlcv.parquet", "time", "close"),
    "xauusd": ("seeds/latest/xauusd_daily_ohlcv.parquet", "time", "close"),
    "btcusdt": ("seeds/latest/btcusdt_daily_ohlcv.parquet", "time", "close"),
    "spx500": ("data/snapshots/public_daily/spx500_daily.parquet", "timestamp", "adj_close"),
}


def _load_returns(asset: str) -> pd.Series:
    path, tcol, ccol = SOURCES[asset]
    d = pd.read_parquet(REPO / path).sort_values(tcol)
    t = pd.to_datetime(d[tcol])
    t = t.dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t
    px = d[ccol].astype(float)
    r = np.log(px / px.shift(1))
    s = pd.Series(r.to_numpy(), index=pd.DatetimeIndex(t)).dropna()
    return s[~s.index.duplicated(keep="last")]


def _ewma_var(r: np.ndarray, lam: float = LAM) -> np.ndarray:
    """Causal EWMA daily variance: var_t is built from r_{<t} only."""
    out = np.empty(len(r))
    var = float(np.var(r[:20])) or 1e-8
    for i in range(len(r)):
        out[i] = var
        var = lam * var + (1 - lam) * r[i] ** 2
    return out


def run_asset(asset: str) -> dict:
    r = _load_returns(asset)
    rv = r ** 2  # daily realized-variance proxy (squared close-to-close return)

    # Forward target: mean daily variance over the NEXT H days. Everything stays in daily
    # variance units so QLIKE compares like with like -- no annualization inside the loss.
    target = rv.rolling(H).mean().shift(-H)

    feats = pd.DataFrame({f"rv_{w}": rv.rolling(w).mean() for w in HAR_WINDOWS}, index=r.index)
    pers_rv20 = rv.rolling(20).mean()          # persistence: today's 20d variance
    pers_ewma = pd.Series(_ewma_var(r.to_numpy()), index=r.index)

    df = pd.concat([feats, target.rename("y"), pers_rv20.rename("p_rv20"),
                    pers_ewma.rename("p_ewma")], axis=1).dropna()

    # Walk-forward HAR: strictly causal, expanding, refit every REFIT_EVERY rows. Predictions
    # exist only where the fit saw nothing later than t.
    X = df[[f"rv_{w}" for w in HAR_WINDOWS]].to_numpy()
    y = df["y"].to_numpy()
    n = len(df)
    start = max(252, int(np.searchsorted(df.index, pd.Timestamp("2024-12-31"))))
    har = np.full(n, np.nan)
    coef = None
    for i in range(start, n):
        if coef is None or (i - start) % REFIT_EVERY == 0:
            Xi = np.column_stack([np.ones(i), X[:i]])
            coef, *_ = np.linalg.lstsq(Xi, y[:i], rcond=None)
        har[i] = max(float(coef[0] + X[i] @ coef[1:]), 1e-12)

    out = {"asset": asset, "n_total": n, "h": H, "har_windows": list(HAR_WINDOWS),
           "lambda": LAM, "windows": {}}
    for label, yr in (("OOS_2025", OOS), ("LIVE_2026_context_only", 2026)):
        m = (df.index.year == yr) & ~np.isnan(har)
        if m.sum() < 30:
            out["windows"][label] = {"n": int(m.sum()), "status": "insuficiente"}
            continue
        yt = y[m]
        preds = {"har": har[m], "p_ewma": df["p_ewma"].to_numpy()[m],
                 "p_rv20": df["p_rv20"].to_numpy()[m]}
        losses = {name: qlike_loss(yt, p) for name, p in preds.items()}
        w = {"n": int(m.sum())}
        for name in preds:
            w[f"qlike_{name}"] = round(float(np.nanmean(losses[name])), 6)
            w[f"rmse_{name}"] = round(float(np.sqrt(np.nanmean((yt - preds[name]) ** 2))), 12)

        # Paired moving-block bootstrap of the loss differential HAR - EWMA. The differential,
        # not the two means: vol-loss series are wildly heteroscedastic and pairing removes
        # the common component.
        d = losses["har"] - losses["p_ewma"]
        d = d[~np.isnan(d)]
        rng = np.random.default_rng(42)
        block, boots = 20, 5000
        if len(d) > block + 1:
            means = np.empty(boots)
            for b in range(boots):
                starts = rng.integers(0, len(d) - block + 1, size=(len(d) // block) + 1)
                idx = np.concatenate([np.arange(s, s + block) for s in starts])[:len(d)]
                means[b] = d[idx].mean()
            lo, hi = np.percentile(means, [2.5, 97.5])
            w["qlike_diff_har_minus_ewma"] = round(float(d.mean()), 6)
            w["ci95"] = [round(float(lo), 6), round(float(hi), 6)]
            w["excludes_zero"] = bool(hi < 0 or lo > 0)
        out["windows"][label] = w

    oos = out["windows"].get("OOS_2025", {})
    out["verdict"] = {
        "h0_rejected": bool(oos.get("qlike_har", 9e9) < oos.get("qlike_p_ewma", 0)
                            and oos.get("excludes_zero") and oos.get("ci95", [0])[0] < 0),
        "criterion": "OOS-2025: mean QLIKE(HAR) < QLIKE(EWMA) AND paired block-bootstrap "
                     "CI95 of the differential excludes zero",
    }
    return out


def main() -> int:
    print("=" * 88)
    print(f"H-VOLF-01 — HAR-RV vs persistencia, h={H}d, veredicto sobre OOS-{OOS}")
    print("=" * 88)
    results = []
    for asset in SOURCES:
        try:
            res = run_asset(asset)
        except Exception as e:  # noqa: BLE001
            print(f"  {asset:9} ERROR: {e}")
            continue
        results.append(res)
        o = res["windows"].get("OOS_2025", {})
        if "qlike_har" not in o:
            print(f"  {asset:9} OOS insuficiente ({o.get('n', 0)} obs)")
            continue
        verdict = "RECHAZA H0" if res["verdict"]["h0_rejected"] else "NO_RECHAZA"
        print(f"  {asset:9} QLIKE  HAR={o['qlike_har']:.4f}  EWMA={o['qlike_p_ewma']:.4f}  "
              f"RV20={o['qlike_p_rv20']:.4f}  diff={o.get('qlike_diff_har_minus_ewma')}  "
              f"IC95={o.get('ci95')}  -> {verdict}")

    out = REPO / ".claude" / "evidence" / "volf" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "h_volf_01.json").write_text(json.dumps({
        "hypothesis": "H-VOLF-01", "results": results,
        "evidence_class": "research_only", "promotion_eligible": False,
        "note": ("Verdict is OOS-2025 only; 2026 is context (it is the forward). No economic "
                 "test is run for any asset whose H0 stands -- a sizing trial is only paid "
                 "for once QLIKE is won."),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nartefacto -> {out / 'h_volf_01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
