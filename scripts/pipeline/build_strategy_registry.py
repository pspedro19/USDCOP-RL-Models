#!/usr/bin/env python
"""Build the dynamic strategy registry (registry.json + per-strategy manifest.json).

Implements the discovery half of CTR-STRAT-REGISTRY-001. Scans the dashboard data dir,
synthesizes manifests for legacy strategies (migration shim), and writes the registry index
the frontend reads to build itself dynamically — no hardcoded strategy_id / symbol / year.

Usage:
    python scripts/build_strategy_registry.py
    python scripts/build_strategy_registry.py --data-dir path/to/public/data
    python scripts/build_strategy_registry.py --check   # validate only, do not write
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/pipeline/<this> → repo root (reorg fix)

# Load the manifest contract as a standalone leaf module by file path. This deliberately bypasses
# the `src` package __init__ chain (which eager-imports the ML/forecasting stack and is currently
# import-broken on incomplete checkouts). The registry tool has zero ML dependencies by design.
_MANIFEST_PATH = REPO_ROOT / "src" / "contracts" / "strategy_manifest.py"
_spec = importlib.util.spec_from_file_location("strategy_manifest", _MANIFEST_PATH)
strategy_manifest = importlib.util.module_from_spec(_spec)
sys.modules["strategy_manifest"] = strategy_manifest  # required so @dataclass can resolve annotations
_spec.loader.exec_module(strategy_manifest)
RegistryBuilder = strategy_manifest.RegistryBuilder

DEFAULT_DATA_DIR = REPO_ROOT / "usdcop-trading-dashboard" / "public" / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the dynamic strategy registry")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dashboard public/data dir")
    parser.add_argument("--check", action="store_true", help="Validate only; do not write files")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"[registry] ERROR: data dir does not exist: {data_dir}", file=sys.stderr)
        return 1

    generated_at = datetime.now(timezone.utc).isoformat()
    builder = RegistryBuilder(data_dir, generated_at=generated_at)
    index = builder.build(write_manifests=not args.check)

    print(f"[registry] discovered {len(index.strategies)} strateg(ies), {len(index.assets)} asset(s)")
    for s in index.strategies:
        print(f"[registry]   - {s.strategy_id} ({s.asset_id}, {s.status}) years={s.backtest_years} "
              f"replay={s.has_replay} prod={s.has_production}")
    print(f"[registry] default = {index.default}")

    if args.check:
        print("[registry] --check: no files written")
        return 0

    out = builder.write(index)
    print(f"[registry] wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
