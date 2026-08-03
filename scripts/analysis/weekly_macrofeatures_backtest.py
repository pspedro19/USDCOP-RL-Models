"""Weekly causal backtest for macro candidate features, 2025-2026."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

def main():
 cfg=ForecastingSSOTConfig.load(); px,_=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset(); px=px.sort_values('date').reset_index(drop=True)
 m=pd.read_parquet(ROOT/'data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet').reset_index(); m=m.rename(columns={m.columns[0]:'date'}); m.date=pd.to_datetime(m.date)+pd.Timedelta(days=1); m=m.sort_values('date')
 raw=['CRSK_SPREAD_EMBI_COL_D_EMBI','VOLT_VIX_USA_D_VIX','EQTY_INDEX_COLCAP_COL_D_COLCAP','FINC_RATE_IBR_OVERNIGHT_COL_D_IBR','POLR_POLICY_RATE_COL_M_TPM','FINC_BOND_YIELD10Y_COL_D_COL10Y','FINC_BOND_YIELD5Y_COL_D_COL5Y','FXRT_SPOT_USDMXN_MEX_D_USDMXN','FXRT_SPOT_USDCLP_CHL_D_USDCLP','COMM_OIL_BRENT_GLB_D_BRENT','COMM_AGRI_COFFEE_GLB_D_COFFEE','COMM_METAL_GOLD_GLB_D_GOLD']
 m=m[['date']+raw].drop_duplicates('date'); d=pd.merge_asof(px,m,on='date',direction='backward')
 feats=[]
 for c in raw:
  d[c]=d[c].ffill(); feats.append(c)
  for l in (5,20): d[f'{c}_chg{l}']=np.log(d[c]/d[c].shift(l)); feats.append(f'{c}_chg{l}')
 d=d.dropna(subset=feats).reset_index(drop=True); dates=pd.to_datetime(d.date); weeks=d.assign(w=dates.dt.strftime('%G-W%V')).groupby('w').tail(1); rows=[]
 for _,cut in weeks[(weeks.date>='2025-01-01')].iterrows():
  ci=int(cut.name); train_end=ci
  for h in (1,5,10,15,20,25,30):
   y=(np.log(d.close.shift(-h)/d.close)>0).to_numpy(); valid=np.arange(0,ci-h); test=np.arange(ci,min(ci+5,len(d)-h))
   if len(valid)<100 or len(test)==0: continue
   tr=valid[:-h]; model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced')); model.fit(d[feats].iloc[tr],y[tr]); pred=model.predict(d[feats].iloc[test]); base=int(np.mean(y[tr])>=.5)
   rows.append({'week':cut.w,'horizon':h,'n_test':len(test),'macro_da':float(np.mean(pred==y[test])),'baseline_da':float(np.mean(base==y[test])),'delta':float(np.mean(pred==y[test])-np.mean(base==y[test]))})
 out=pd.DataFrame(rows); path=ROOT/'reports'/'weekly_macrofeatures_backtest_2025_2026.csv'; out.to_csv(path,index=False); print(out.groupby('horizon')[['macro_da','baseline_da','delta']].mean().to_string()); print(path)
if __name__=='__main__': main()
