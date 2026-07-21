"""H-COP-XLEAD-01 + H-DIR-FUND-01 — the two pre-registered directional data unlocks.

Contract: CTR-QUANT-EVIDENCE-001 · Both hypotheses registered BEFORE this file existed
(XLEAD on 2026-07-06, FUND on 2026-07-21).

Direction on the EXISTING data is closed (27 trials + full sweep, best cell p_adj = 1.0).
These two runs do not reopen that question: each adds information the closed question never
saw — cross-asset EM leads for COP (MXN/CLP print the dollar move before Bogota opens, per
em-fx), and perp funding for BTC (crowding, never tested directionally, only as a brake).

## Pre-signed design, one variable each, no knobs

  COP   Ridge+BR weekly walk-forward (the track's own loop), 5d horizon, expanding window:
        features WITHOUT vs WITH {usdmxn_ret_1d_lag, usdclp_ret_1d_lag}. Same Mondays, same
        model, same everything -- include_xlead is the single toggled variable.
  BTC   Ridge daily walk-forward, 5d horizon, refit every 21 obs:
        {ret_1d, ret_5d, ret_20d, rv_20} WITHOUT vs WITH z_funding(t-1).

  Verdict, both: OOS-2025 directional accuracy, decided by McNEMAR on the discordant weeks
  (paired: same periods, so the binomial on b vs c is exact). Reject H0 only if
  p < 0.05 AND DA_with > DA_without. Anything else => NO_RECHAZA, stated plainly.

If a hypothesis passes, NOTHING trades: the winner becomes a candidate NEW VERSION entering
the full promotion sequence (manifest, PIT, forward, DSR/PBO). v11 stays frozen either way.

Run: python -m scripts.analysis.directional_unlock_evidence
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

OOS = 2025
H = 5


def _mcnemar(hits_a: np.ndarray, hits_b: np.ndarray) -> dict:
    """Exact binomial McNemar on discordant pairs (same periods, paired by construction)."""
    from scipy import stats
    b = int(np.sum(hits_b & ~hits_a))   # WITH right, WITHOUT wrong
    c = int(np.sum(hits_a & ~hits_b))   # WITHOUT right, WITH wrong
    n = b + c
    p = float(stats.binomtest(b, n, 0.5).pvalue) if n > 0 else 1.0
    return {"discordant_with_right": b, "discordant_without_right": c, "p_mcnemar": round(p, 4)}


def run_cop() -> dict:
    """COP: the track's own dataset + enhance_v2, toggling include_xlead only."""
    from src.forecasting.dataset_loader import ForecastingDatasetLoader
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.enhance_v2 import enhance_features_v2
    from sklearn.linear_model import BayesianRidge, Ridge
    from sklearn.preprocessing import StandardScaler

    cfg = ForecastingSSOTConfig.load()
    df, feats = ForecastingDatasetLoader(cfg, project_root=REPO).load_dataset(target_horizon=H)
    out = {}
    for label, xlead in (("without", False), ("with", True)):
        d, cols = enhance_features_v2(df.copy(), feats, project_root=REPO,
                                      include_xlead=xlead)
        d = d.dropna(subset=cols + ["target_return_5d"]).reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        mondays = d[d["date"].dt.year == OOS].loc[d["date"].dt.dayofweek == 0, "date"]
        hits, dirs = [], []
        for monday in mondays:
            train = d[d["date"] < monday]
            row = d[d["date"] == monday]
            if len(train) < 200 or row.empty:
                continue
            X, y = train[cols].to_numpy(), train["target_return_5d"].to_numpy()
            sc = StandardScaler().fit(X)
            pr = Ridge(alpha=1.0).fit(sc.transform(X), y)
            pb = BayesianRidge(max_iter=300).fit(sc.transform(X), y)
            xr = sc.transform(row[cols].to_numpy())
            pred = float((pr.predict(xr)[0] + pb.predict(xr)[0]) / 2)
            realized = float(row["target_return_5d"].iloc[0])
            hits.append(np.sign(pred) == np.sign(realized) and realized != 0)
            dirs.append(int(np.sign(pred)))
        out[label] = {"n_weeks": len(hits), "da": round(float(np.mean(hits)), 4),
                      "hits": np.array(hits, dtype=bool)}
    mc = _mcnemar(out["without"]["hits"], out["with"]["hits"])
    return _verdict("H-COP-XLEAD-01", out, mc)


def run_btc() -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    px = pd.read_parquet(REPO / "seeds/latest/btcusdt_daily_ohlcv.parquet").sort_values("time")
    px["time"] = pd.to_datetime(px["time"]).dt.tz_localize(None)
    r = np.log(px["close"].astype(float) / px["close"].astype(float).shift(1))
    d = pd.DataFrame({"time": px["time"].to_numpy(), "ret_1d": r.to_numpy()})
    d["ret_5d"] = d["ret_1d"].rolling(5).sum()
    d["ret_20d"] = d["ret_1d"].rolling(20).sum()
    d["rv_20"] = d["ret_1d"].rolling(20).std()
    d["target"] = d["ret_1d"].rolling(H).sum().shift(-H)

    fund = pd.read_parquet(REPO / "seeds/latest/btcusdt_derivatives_daily.parquet")
    fund["date"] = pd.to_datetime(fund["date"]).dt.tz_localize(None)
    fund = fund.sort_values("date")
    fund["z_funding_lag"] = fund["funding_zscore"].shift(1)   # t-1: today's print unknowable
    d = pd.merge_asof(d.sort_values("time"), fund[["date", "z_funding_lag"]],
                      left_on="time", right_on="date", direction="backward").drop(columns=["date"])

    base = ["ret_1d", "ret_5d", "ret_20d", "rv_20"]
    out = {}
    for label, cols in (("without", base), ("with", base + ["z_funding_lag"])):
        dd = d.dropna(subset=cols + ["target"]).reset_index(drop=True)
        idx = dd.index[dd["time"].dt.year == OOS]
        hits = []
        coef = None
        sc = None
        for k, i in enumerate(idx):
            if coef is None or k % 21 == 0:
                tr = dd.iloc[:i]
                X, y = tr[cols].to_numpy(), tr["target"].to_numpy()
                sc = StandardScaler().fit(X)
                coef = Ridge(alpha=1.0).fit(sc.transform(X), y)
            pred = float(coef.predict(sc.transform(dd.loc[[i], cols].to_numpy()))[0])
            realized = float(dd.loc[i, "target"])
            hits.append(np.sign(pred) == np.sign(realized) and realized != 0)
        out[label] = {"n_days": len(hits), "da": round(float(np.mean(hits)), 4),
                      "hits": np.array(hits, dtype=bool)}
    n = min(len(out["without"]["hits"]), len(out["with"]["hits"]))
    mc = _mcnemar(out["without"]["hits"][:n], out["with"]["hits"][:n])
    return _verdict("H-DIR-FUND-01", out, mc)


def _verdict(hyp: str, out: dict, mc: dict) -> dict:
    da_w, da_wo = out["with"]["da"], out["without"]["da"]
    rejected = bool(mc["p_mcnemar"] < 0.05 and da_w > da_wo)
    res = {
        "hypothesis": hyp,
        "da_without": da_wo, "da_with": da_w, "delta_da": round(da_w - da_wo, 4),
        "n": out["with"].get("n_weeks", out["with"].get("n_days")),
        **mc,
        "h0_rejected": rejected,
        "criterion": "OOS-2025: McNemar p<0.05 AND DA_with > DA_without",
        "consequence": ("candidate NEW version -> full promotion sequence; nothing trades"
                        if rejected else "avenue closed on this data; stated plainly"),
    }
    for k in ("with", "without"):
        out[k].pop("hits", None)
    res["detail"] = out
    return res


def main() -> int:
    results = []
    for name, fn in (("cop_xlead", run_cop), ("btc_fund", run_btc)):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{name} ERROR: {e}")
            continue
        results.append(r)
        print(f"{r['hypothesis']}: DA sin={r['da_without']} con={r['da_with']} "
              f"delta={r['delta_da']} McNemar p={r['p_mcnemar']} -> "
              f"{'RECHAZA H0' if r['h0_rejected'] else 'NO_RECHAZA'}")

    out = REPO / ".claude" / "evidence" / "directional_unlock" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "results": results, "evidence_class": "research_only", "promotion_eligible": False,
    }, indent=2, default=str), encoding="utf-8")
    print(f"artefacto -> {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
