"""Causal weekly HMM states for USDCOP (no future observations)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

def main():
 cfg=ForecastingSSOTConfig.load(); d,_=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset(); d=d.sort_values('date').reset_index(drop=True); d['week']=pd.to_datetime(d.date).dt.strftime('%G-W%V')
 rows=[]
 for _,cut in d.groupby('week').tail(1).iterrows():
  if cut.date<pd.Timestamp('2025-01-01'): continue
  hist=d[d.date<=cut.date].tail(252); ret=np.log(hist.close/hist.close.shift(1)).dropna();
  if len(ret)<100: continue
  model=GaussianHMM(n_components=3,covariance_type='diag',n_iter=200,random_state=42); model.fit(ret.to_numpy().reshape(-1,1)); states=model.predict(ret.to_numpy().reshape(-1,1)); means=model.means_.ravel(); order=np.argsort(means); labels={order[0]:'risk_on_appreciation',order[1]:'transition',order[2]:'risk_off_depreciation'}; state=labels[states[-1]]
  persistence=0
  for s in states[::-1]:
   if s != states[-1]: break
   persistence += 1
  rows.append({'week':cut.week,'date':cut.date,'state':state,'state_mean_return':float(means[states[-1]]),'persistence_days':persistence,'state_prob':float(model.predict_proba(ret.to_numpy().reshape(-1,1))[-1,states[-1]])})
 out=pd.DataFrame(rows); path=ROOT/'reports'/'weekly_hmm_regime_states_2025_2026.csv'; out.to_csv(path,index=False); print(out.groupby(['week']) .tail(1).to_string(index=False)); print('\nstate counts'); print(out.groupby('state').size()); print(path)
if __name__=='__main__': main()
