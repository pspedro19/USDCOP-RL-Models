"""Causal walk-forward audit for direct UP/DOWN USDCOP classification."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
from src.forecasting.contracts import HORIZONS

def main():
    cfg = ForecastingSSOTConfig.load()
    df, feature_cols = ForecastingDatasetLoader(cfg, project_root=ROOT).load_dataset()
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df[feature_cols].notna().all(axis=1)].reset_index(drop=True)
    X = df[feature_cols].to_numpy(float)
    close = df["close"].to_numpy(float)
    rows = []
    for h in HORIZONS:
        yret = np.log(np.roll(close, -h) / close)
        valid = np.arange(len(df)-h)
        y = (yret[valid] > 0).astype(int)
        xv = X[valid]
        n = len(valid); initial = int(n*0.6); step = (n-initial)//5
        pred, actual, baseline, dates, regimes = [], [], [], [], []
        for fold in range(5):
            train_end = initial + fold*step
            test_end = min(train_end+step, n)
            train_end = max(0, train_end-h)
            if train_end < 50 or test_end <= train_end: continue
            model = make_pipeline(SimpleImputer(), StandardScaler(),
                                   LogisticRegression(max_iter=1000, class_weight="balanced"))
            # Full-history training is retained after the 504-session adaptive
            # experiment proved unstable; the embargo remains causal.
            train_start = 0
            model.fit(xv[train_start:train_end], y[train_start:train_end])
            test_slice = slice(initial+fold*step, test_end)
            pred.extend(model.predict(xv[test_slice]))
            actual.extend(y[test_slice])
            majority = int(np.mean(y[train_start:train_end]) >= 0.5)
            baseline.extend([majority] * (test_end - (initial+fold*step)))
            dates.extend(df["date"].iloc[valid[initial+fold*step:test_end]].tolist())
            regimes.extend(df["volatility_20d"].iloc[valid[initial+fold*step:test_end]].tolist())
        da = float(np.mean(np.asarray(pred)==np.asarray(actual))) if actual else 0.5
        base_da = float(np.mean(np.asarray(baseline)==np.asarray(actual))) if actual else 0.5
        rows.append({"horizon_days": h, "n_oos": len(actual), "directional_da": da,
                     "majority_baseline_da": base_da,
                     "delta_vs_baseline": da-base_da,
                     "up_rate_oos": float(np.mean(actual)) if actual else np.nan})
        if actual:
            tmp = pd.DataFrame({"date": pd.to_datetime(dates), "pred": pred,
                                "actual": actual, "baseline": baseline, "vol20": regimes})
            tmp["vol_regime"] = np.where(tmp.vol20 <= tmp.vol20.median(), "low_vol", "high_vol")
            for regime, g in tmp.groupby("vol_regime"):
                rows.append({"horizon_days": h, "regime": regime, "n_oos": len(g),
                             "directional_da": float(np.mean(g.pred==g.actual)),
                             "majority_baseline_da": float(np.mean(g.baseline==g.actual)),
                             "delta_vs_baseline": float(np.mean(g.pred==g.actual)-np.mean(g.baseline==g.actual)),
                             "up_rate_oos": float(g.actual.mean())})
            for year, g in tmp.groupby(tmp.date.dt.year):
                rows.append({"horizon_days": h, "year": int(year), "n_oos": len(g),
                             "directional_da": float(np.mean(g.pred==g.actual)),
                             "majority_baseline_da": float(np.mean(g.baseline==g.actual)),
                             "delta_vs_baseline": float(np.mean(g.pred==g.actual)-np.mean(g.baseline==g.actual)),
                             "up_rate_oos": float(g.actual.mean())})
    out = pd.DataFrame(rows)
    path = ROOT/"reports"/"usdcop_directional_classifier_audit.csv"
    path.parent.mkdir(exist_ok=True); out.to_csv(path,index=False)
    print(out.to_string(index=False)); print(path)

if __name__ == "__main__": main()
