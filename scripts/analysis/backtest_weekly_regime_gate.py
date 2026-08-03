"""Evaluate regime-gated weekly macro forecasts without refitting on OOS."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.forecasting.regime_gate import compute_hurst_rs
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

def main():
 cfg=ForecastingSSOTConfig.load(); d,_=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset(); d=d.sort_values('date').reset_index(drop=True); d['week']=pd.to_datetime(d.date).dt.strftime('%G-W%V')
 weekly=d.groupby('week').tail(1); states=[]
 for _,r in weekly.iterrows():
  hist=d[d.date<=r.date].tail(60); ret=np.log(hist.close/hist.close.shift(1)).dropna().to_numpy(); h=compute_hurst_rs(ret); slope=np.polyfit(np.arange(len(hist)),hist.close,1)[0]/hist.close.mean(); vol_s=ret[-5:].std(); vol_l=ret.std(); ratio=vol_s/(vol_l or 1e-9)
  state='trending' if h>.52 and abs(slope)>0.0001 else ('mean_reverting' if h<.42 else 'indeterminate')
  states.append({'week':r.week,'hurst':h,'trend_slope':slope,'vol_ratio':ratio,'regime':state})
 reg=pd.DataFrame(states); pred=pd.read_csv(ROOT/'reports'/'weekly_macrofeatures_backtest_2025_2026.csv'); x=pred.merge(reg,on='week',how='left')
 rows=[]
 for (period,h),g in x.assign(period=lambda z:np.where(z.week.str.startswith('2025'),'2025_OOS','2026_FORWARD')).groupby(['period','horizon']):
  for name,subset in [('all',g),('trending',g[g.regime=='trending']),('non_mean_reverting',g[g.regime!='mean_reverting'])]:
   rows.append({'period':period,'horizon':h,'gate':name,'weeks':len(subset),'coverage':len(subset)/len(g),'da':subset.macro_da.mean() if len(subset) else np.nan,'baseline_da':subset.baseline_da.mean() if len(subset) else np.nan})
 out=pd.DataFrame(rows); path=ROOT/'reports'/'weekly_regime_gate_backtest.csv'; out.to_csv(path,index=False); print(out.to_string(index=False)); print(path)
if __name__=='__main__': main()
