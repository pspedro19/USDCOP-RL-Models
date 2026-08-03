"""CLI for quantitative harness; accepts JSON dataset descriptors."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation.quant_harness import run_harness

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    with open(ns.input, encoding="utf-8") as f: payload = json.load(f)
    result = run_harness(payload.get("datasets", {}), **payload.get("kwargs", {}))
    with open(ns.output, "w", encoding="utf-8") as f: json.dump(result, f, indent=2, default=str)
    return 0 if result["passed"] else 1
if __name__ == "__main__": raise SystemExit(main())
