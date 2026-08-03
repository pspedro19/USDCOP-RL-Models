"""Two-stage causal feature selection: select on pre-2025, OOS 2025, then 2026."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.analysis._causal_backtest import matured_label_indices_before
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
from src.forecasting.contracts import HORIZONS

def fit_score(df, cols, h, start, end, train_end):
    close=df.close.to_numpy(float); X=df[cols].to_numpy(float)
    idx=np.arange(len(df)-h); y=(np.log(np.roll(close,-h)/close)>0).astype(int)
    test=idx[(idx>=start)&(idx<end)]; train=idx[idx<train_end-h]
    if len(test)<10 or len(train)<50: return None
    model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced'))
    model.fit(X[train],y[train]); pred=model.predict(X[test])
    base=int(np.mean(y[train])>=.5)
    return len(test),float(np.mean(pred==y[test])),float(np.mean(base==y[test]))

def main():
    cfg=ForecastingSSOTConfig.load(); df, cols=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset()
    df=df.sort_values('date').reset_index(drop=True); df=df[df[list(cols)].notna().all(axis=1)].reset_index(drop=True)
    dates=pd.to_datetime(df.date); future20=np.log(df.close.shift(-20)/df.close); y20=(future20>0).where(future20.notna()).astype(float).to_numpy()
    valid=matured_label_indices_before(dates,horizon=20,cutoff='2025-01-01'); valid=valid[~np.isnan(y20[valid])]; X=df[list(cols)].to_numpy(float)
    imp=SimpleImputer().fit(X[valid]); mi=mutual_info_classif(imp.transform(X[valid]),y20[valid].astype(int),random_state=7)
    order=np.argsort(mi)[::-1]; selected=[cols[i] for i in order[:8]]
    rows=[]
    for h in (5,10,15,20,25,30):
        for label,start,end,train_end in [('2025_OOS','2025-01-01','2026-01-01',len(df)),('2026_OOS','2026-01-01','2027-01-01',len(df))]:
            s=np.searchsorted(dates,pd.Timestamp(start)); e=np.searchsorted(dates,pd.Timestamp(end)); tr=s
            # 2025 model trains pre-2025; 2026 model retrains through 2025.
            score=fit_score(df,selected,h,s,e,tr)
            if score: rows.append({'period':label,'horizon':h,'n_oos':score[0],'da':score[1],'baseline_da':score[2],'delta':score[1]-score[2]})
    out=pd.DataFrame(rows); out.to_csv(ROOT/'reports'/'feature_selection_2025_oos_2026.csv',index=False)
    print('selected_features=',selected); print(out.to_string(index=False))
if __name__=='__main__': main()
