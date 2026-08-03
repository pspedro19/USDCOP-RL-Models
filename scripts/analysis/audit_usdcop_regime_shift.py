"""Audit causal feature-distribution shifts for USDCOP forecasting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

def main():
    cfg=ForecastingSSOTConfig.load(); df, cols=ForecastingDatasetLoader(cfg, project_root=ROOT).load_dataset()
    df=df.sort_values('date').reset_index(drop=True)
    rows=[]
    for i in range(252, len(df), 20):
        ref=df.iloc[i-252:i]; cur=df.iloc[max(0,i-60):i]
        scores=[]
        for c in cols:
            a=ref[c].dropna().to_numpy(float); b=cur[c].dropna().to_numpy(float)
            if len(a)>20 and len(b)>10:
                scale=np.std(a) or 1.0; scores.append(wasserstein_distance(a/scale,b/scale))
        rows.append({'date':df.date.iloc[i], 'shift_mean':float(np.mean(scores)), 'shift_max':float(np.max(scores))})
    out=pd.DataFrame(rows); out['shift_z']=(out.shift_mean-out.shift_mean.rolling(10,min_periods=3).mean())/(out.shift_mean.rolling(10,min_periods=3).std()+1e-9)
    path=ROOT/'reports'/'usdcop_regime_shift_audit.csv'; path.parent.mkdir(exist_ok=True); out.to_csv(path,index=False)
    print(out.tail(15).to_string(index=False)); print(path)
if __name__=='__main__': main()
