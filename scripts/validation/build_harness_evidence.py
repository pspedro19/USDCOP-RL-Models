"""Build an auditable evidence bundle for harness runs (no secrets, no mutation)."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.claude/codex/evidence/harness_bundle'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name, cmd):
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=180)
    (OUT/f'{name}.stdout.log').write_text(p.stdout,encoding='utf-8'); (OUT/f'{name}.stderr.log').write_text(p.stderr,encoding='utf-8')
    return {'name':name,'command':' '.join(cmd),'returncode':p.returncode,'stdout':f'{name}.stdout.log','stderr':f'{name}.stderr.log'}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[run('market_statistics',[sys.executable,'scripts/analysis/audit_market_data_statistics.py']),run('acquisition_assets',[sys.executable,'scripts/analysis/audit_acquisition_assets.py']),run('reconciliation',[sys.executable,'scripts/analysis/reconcile_seed_backups.py']),run('manifest_validation',[sys.executable,'scripts/validation/validate_acquisition_manifests.py']),run('harness',[sys.executable,'.claude/codex/harness/harness_engine.py','--no-tests'])]
    evidence=[p for p in OUT.glob('*') if p.is_file()]
    index={'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'runs':runs,'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(evidence)]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Harness evidence bundle', '', f"Generated: {index['generated_at']}",'','Evidence is local/reproducible; no provider credentials or secrets are captured.','']
    md += ['## Runs','']+[f"- `{r['name']}`: exit `{r['returncode']}` — `{r['command']}`" for r in runs]
    md += ['','## Visual evidence','', '- Screenshot/video capture is **PENDING** until a browser session and authenticated sandbox are available.', '- The bundle contains command logs, JSON evidence and SHA-256 hashes; it does not claim a visual or live-provider run.','']
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'bundle':str(OUT),'runs':runs,'artifacts':len(index['artifacts'])},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
