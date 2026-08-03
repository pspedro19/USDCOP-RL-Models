"""Causal weekly backtest with regime-conditioned directional models."""
from pathlib import Path
import sys,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
def states_for(close):
 r=np.log(close/close.shift(1)); out=[]
 for i in range(len(close)):
  h=r.iloc[:i+1].dropna();
  if len(h)<120: out.append('transition');continue
  a=h.tail(20);b=h.tail(120).head(100);p=(a<0).mean();q=(b<0).mean();z=(p-q)/np.sqrt(max(q*(1-q)/len(a),1e-6));out.append('risk_off' if z>1.5 and a.mean()<b.mean() else ('risk_on' if z<-1.5 and a.mean()>b.mean() else 'transition'))
 return np.array(out)
def main():
 cfg=ForecastingSSOTConfig.load();d,cols=ForecastingDatasetLoader(cfg,project_root=ROOT).load_dataset();d=d.sort_values('date').reset_index(drop=True);d['week']=pd.to_datetime(d.date).dt.strftime('%G-W%V');d['state']=states_for(d.close);weeks=d.groupby('week').tail(1);rows=[]
 for _,cut in weeks[weeks.date>='2025-01-01'].iterrows():
  ci=cut.name
  for h in (1,5,10,15,20,25,30):
   y=(np.log(d.close.shift(-h)/d.close)>0).to_numpy();tr=np.arange(0,ci-h);te=np.arange(ci,min(ci+5,len(d)-h))
   if len(tr)<150 or len(te)==0:continue
   global_model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1200,class_weight='balanced'));global_model.fit(d[cols].iloc[tr],y[tr]);pred=[]
   for j in te:
    s=d.state.iloc[j]; idx=tr[d.state.iloc[tr]==s]; model=global_model
    if len(idx)>=80:
     model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1200,class_weight='balanced'));model.fit(d[cols].iloc[idx],y[idx])
    pred.append(int(model.predict(d[cols].iloc[[j]])[0]))
   base=int(y[tr].mean()>=.5);rows.append({'week':cut.week,'horizon':h,'regime':d.state.iloc[ci],'n_test':len(te),'da':np.mean(np.array(pred)==y[te]),'baseline_da':np.mean(base==y[te]),'delta':np.mean(np.array(pred)==y[te])-np.mean(base==y[te])})
 out=pd.DataFrame(rows);path=ROOT/'reports'/'weekly_regime_conditioned_backtest.csv';out.to_csv(path,index=False);print(out.groupby(['week']).head(1).tail(10).to_string(index=False));print(out.groupby('horizon')[['da','baseline_da','delta']].mean().to_string());print(path)
if __name__=='__main__':main()
