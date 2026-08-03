#!/usr/bin/env python
"""Run production lifecycle gates from a JSON candidate manifest."""
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation.production_harness import evaluate, HarnessThresholds, write_evidence

def main():
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--champion"); p.add_argument("--out", default=".claude/codex/production-harness-latest.json"); p.add_argument("--kill-switch", action="store_true"); a=p.parse_args()
    candidate=json.load(open(a.manifest, encoding="utf-8")); champion=json.load(open(a.champion, encoding="utf-8")) if a.champion else None
    r=evaluate(candidate, champion, HarnessThresholds(), a.kill_switch); write_evidence(r,a.out); print(json.dumps({"go":r.go,"artifact_id":r.artifact_id,"reasons":r.reasons})); return 0 if r.go else 2
if __name__ == "__main__": raise SystemExit(main())
