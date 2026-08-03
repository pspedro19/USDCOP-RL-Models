"""Stable causal directional-regime detector with hysteresis inputs."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
def main():
 cfg=ForecastingSSOTConfig.load();d,_=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset();d=d.sort_values('date').reset_index(drop=True);d['ret']=np.log(d.close/d.close.shift(1));d['week']=pd.to_datetime(d.date).dt.strftime('%G-W%V');rows=[]
 for _,r in d.groupby('week').tail(1).iterrows():
  h=d[d.date<=r.date].ret.dropna(); recent=h.tail(20); ref=h.tail(120).head(100); p_recent=float((recent<0).mean());p_ref=float((ref<0).mean()); mean_recent=float(recent.mean());mean_ref=float(ref.mean()); score=(p_recent-p_ref)/np.sqrt(max(p_ref*(1-p_ref)/len(recent),1e-6)); state='risk_off' if score>1.5 and mean_recent<mean_ref else ('risk_on' if score<-1.5 and mean_recent>mean_ref else 'transition');rows.append({'week':r.week,'p_down_20':p_recent,'p_down_ref':p_ref,'mean_recent':mean_recent,'mean_ref':mean_ref,'direction_shift_z':score,'state':state})
 out=pd.DataFrame(rows);path=ROOT/'reports'/'weekly_direction_regime_states.csv';out.to_csv(path,index=False);print(out[out.week.str.startswith('2026')].tail(20).to_string(index=False));print(path)
if __name__=='__main__':main()
