"""Causal audit of Colombian and cross-asset macro features by horizon."""
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

def main():
    cfg=ForecastingSSOTConfig.load(); price, base_cols=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset()
    price=price.sort_values('date').reset_index(drop=True)
    m=pd.read_parquet(ROOT/'data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet').reset_index()
    m=m.rename(columns={m.columns[0]:'date'}); m['date']=pd.to_datetime(m.date)
    # publication-safe: only macro observations strictly before the market date
    m['date']=m.date+pd.Timedelta(days=1)
    m=m.sort_values('date').drop_duplicates('date')
    macro_cols=['FXRT_SPOT_USDMXN_MEX_D_USDMXN','FXRT_SPOT_USDCLP_CHL_D_USDCLP','VOLT_VIX_USA_D_VIX','CRSK_SPREAD_EMBI_COL_D_EMBI','COMM_OIL_BRENT_GLB_D_BRENT','COMM_AGRI_COFFEE_GLB_D_COFFEE','COMM_METAL_GOLD_GLB_D_GOLD','FINC_RATE_IBR_OVERNIGHT_COL_D_IBR','POLR_POLICY_RATE_COL_M_TPM','FINC_BOND_YIELD10Y_COL_D_COL10Y','FINC_BOND_YIELD5Y_COL_D_COL5Y','EQTY_INDEX_COLCAP_COL_D_COLCAP']
    m=m[['date']+macro_cols]
    d=pd.merge_asof(price.sort_values('date'),m,on='date',direction='backward')
    feats=[]
    for c in macro_cols:
        d[c]=d[c].ffill()
        for lag in (1,5,20):
            name=f'{c}_chg{lag}'; d[name]=np.log(d[c]/d[c].shift(lag)); feats.append(name)
        feats.append(c)
    d=d.dropna(subset=feats).reset_index(drop=True)
    dates=pd.to_datetime(d.date); rows=[]
    for h in HORIZONS:
        close=d.close.to_numpy(float); y=(np.log(np.roll(close,-h)/close)>0).astype(int); valid=np.arange(len(d)-h)
        pre=matured_label_indices_before(dates,horizon=h,cutoff='2025-01-01'); ypre=y[pre]
        imp=SimpleImputer().fit(d[feats].iloc[pre]); mi=mutual_info_classif(imp.transform(d[feats].iloc[pre]),ypre,random_state=7)
        selected=[feats[i] for i in np.argsort(mi)[-12:]]
        for period,start,end in [('2025_OOS','2025-01-01','2026-01-01'),('2026_OOS','2026-01-01','2027-01-01')]:
            s=np.searchsorted(dates,pd.Timestamp(start)); e=np.searchsorted(dates,pd.Timestamp(end)); train=valid[valid<s-h]; test=valid[(valid>=s)&(valid<e)]
            if len(test)<20: continue
            model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced')); model.fit(d[selected].iloc[train],y[train]); pred=model.predict(d[selected].iloc[test]); base=int(np.mean(y[train])>=.5)
            rows.append({'period':period,'horizon':h,'n_oos':len(test),'da':float(np.mean(pred==y[test])),'baseline_da':float(np.mean(base==y[test])),'delta':float(np.mean(pred==y[test])-np.mean(base==y[test])),'selected_features':'|'.join(selected)})
    out=pd.DataFrame(rows); path=ROOT/'reports'/'colombia_macro_candidates_oos.csv'; out.to_csv(path,index=False); print(out[['period','horizon','n_oos','da','baseline_da','delta']].to_string(index=False)); print(path)
if __name__=='__main__': main()
