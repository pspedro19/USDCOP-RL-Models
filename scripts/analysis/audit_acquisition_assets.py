"""Audit acquisition methods, seeds, backfills and backups without external calls."""
from __future__ import annotations
import hashlib, json, os, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def frame_info(path: Path) -> dict:
    try:
        df = pd.read_parquet(path)
        date_cols = [c for c in df.columns if re.search(r'(date|time|timestamp|fecha)', str(c), re.I)]
        dates = []
        for c in date_cols:
            z = pd.to_datetime(df[c], errors='coerce', utc=True).dropna()
            if len(z): dates.extend([z.min(), z.max()])
        return {'rows': len(df), 'columns': len(df.columns), 'date_min': str(min(dates).date()) if dates else None,
                'date_max': str(max(dates).date()) if dates else None, 'sha256': sha256(path)}
    except Exception as e:
        return {'error': str(e), 'sha256': sha256(path)}

def main():
    out = {'generated_at': pd.Timestamp.utcnow().isoformat(), 'sources': {}, 'artifacts': [], 'decision': 'REVIEW_REQUIRED', 'blockers': []}
    scripts = list((ROOT/'airflow'/'dags').rglob('*.py')) + list((ROOT/'scripts').rglob('*.py'))
    text = '\n'.join(p.read_text(encoding='utf-8', errors='ignore') for p in scripts)
    methods = {'twelvedata': bool(re.search(r'twelve.?data', text, re.I)), 'mt5': bool(re.search(r'\bmt5\b|metatrader', text, re.I)),
               'bcrp': bool(re.search(r'\bbcrp\b', text, re.I)), 'suameca': bool(re.search(r'suameca', text, re.I)),
               'scraping': bool(re.search(r'scrap|selenium|investing\.com', text, re.I)), 'public_adapter': (ROOT/'scripts/data/acquire_public_snapshots.py').exists()}
    out['sources'] = methods
    for base in [ROOT/'seeds'/'latest', ROOT/'data'/'backups', ROOT/'data'/'snapshots']:
        for p in base.rglob('*.parquet'):
            info = frame_info(p); info.update({'path': str(p.relative_to(ROOT))})
            out['artifacts'].append(info)
    # Validate manifests where a hash is declared.
    for mp in list((ROOT/'seeds').rglob('*manifest*.json')) + list((ROOT/'data').rglob('*manifest*.json')):
        try:
            m=json.loads(mp.read_text(encoding='utf-8'))
            out.setdefault('manifests', []).append({'path':str(mp.relative_to(ROOT)), 'exists':True, 'declared':m})
        except Exception as e: out.setdefault('manifests', []).append({'path':str(mp.relative_to(ROOT)), 'error':str(e)})
    out['blockers'] += ['public snapshots declare pit_vintage=false and promotion_eligible=false']
    out['blockers'] += ['provider credentials/connectivity and live backfill execution were not exercised in this offline audit']
    out['blockers'] += ['backup restore drill and remote/offsite durability are not evidenced locally']
    (ROOT/'.claude'/'codex'/'evidence').mkdir(parents=True, exist_ok=True)
    (ROOT/'.claude'/'codex'/'evidence'/'acquisition-assets-audit.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'decision':out['decision'],'methods':methods,'artifacts':len(out['artifacts']),'blockers':len(out['blockers'])}, indent=2))

if __name__ == '__main__': main()
