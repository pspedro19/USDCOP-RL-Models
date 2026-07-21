"""Inventory and quality metrics for seeds/backups produced by acquisition DAGs."""
from pathlib import Path
import json, os
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.claude/codex/evidence/acquisition-backups-audit.json'

def inspect(path):
    try: d=pd.read_parquet(path)
    except Exception as e: return {'path':str(path.relative_to(ROOT)),'readable':False,'error':str(e)}
    t=next((c for c in ('timestamp','time','date','fecha','created_at') if c in d.columns),None)
    r={'path':str(path.relative_to(ROOT)),'readable':True,'rows':len(d),'columns':list(d.columns),'time_column':t}
    if t:
        ts=pd.to_datetime(d[t],utc=True,errors='coerce').sort_values(); delta=ts.diff().dt.total_seconds()/86400
        r.update(start=ts.min().isoformat() if len(ts) else None,end=ts.max().isoformat() if len(ts) else None,
                 invalid_timestamps=int(ts.isna().sum()),duplicate_timestamps=int(ts.duplicated().sum()),
                 median_gap_days=float(delta.median()) if len(delta)>1 else None,p95_gap_days=float(delta.quantile(.95)) if len(delta)>1 else None)
    numeric=[c for c in ('open','high','low','close','volume') if c in d.columns]
    r['missing_pct']={c:round(float(d[c].isna().mean()*100),4) for c in numeric}
    if all(c in d for c in ('open','high','low','close')):
        r['ohlc_invalid']=int(((d['high']<d[['open','close']].max(axis=1))|(d['low']>d[['open','close']].min(axis=1))|(d[['open','high','low','close']]<=0).any(axis=1)).sum())
    return r

def main():
    paths=list((ROOT/'seeds/latest').glob('*.parquet'))+list((ROOT/'data/backups').rglob('*.parquet'))
    sources={
      'twelvedata': any('twelvedata' in p.read_text(errors='ignore').lower() for p in (ROOT/'airflow/dags').rglob('*.py')),
      'mt5': any('mt5' in p.read_text(errors='ignore').lower() for p in (ROOT/'airflow/dags').rglob('*.py')),
      'scraping': any(x in p.read_text(errors='ignore').lower() for p in (ROOT/'airflow/dags').rglob('*.py') for x in ('selenium','scrap','scraper')),
      'bcrp_suameca': any(x in p.read_text(errors='ignore').lower() for p in (ROOT/'airflow/dags').rglob('*.py') for x in ('bcrp','suameca')),
      'public_yahoo': (ROOT/'data/snapshots/public_daily').exists(),
    }
    result={'schema_version':1,'generated_at':pd.Timestamp.utcnow().isoformat(),'files':[inspect(p) for p in sorted(paths)],'sources_detected':sources,
            'summary':{'files':len(paths),'readable':None,'unreadable':None,'rows':None}}
    result['summary']['readable']=sum(x['readable'] for x in result['files']); result['summary']['unreadable']=len(paths)-result['summary']['readable']; result['summary']['rows']=sum(x.get('rows',0) for x in result['files'])
    result['decision']='REVIEW_REQUIRED' if result['summary']['unreadable'] or not all(sources.values()) else 'PASS'
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(result['summary']|{'decision':result['decision'],'sources':sources},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
