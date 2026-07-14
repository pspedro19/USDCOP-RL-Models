"""Backfill macro_snapshots / macro_charts into existing Gold & BTC weekly JSONs.

The multi-asset analysis generator originally emitted empty ``macro_snapshots`` /
``macro_charts`` for Gold/BTC, so their /analysis pages showed no macro charts
(unlike USD/COP). This script patches the already-published per-asset weekly
files in place using the SAME shared builder the generator now uses
(src/analysis/asset_macro_charts.py) — real global-driver data, no DB, no news
re-fetch. Idempotent: re-running recomputes and overwrites the two blocks only.

Usage:
  python -m scripts.ops.patch_asset_macro_charts                 # xauusd + btcusdt, all weeks
  python -m scripts.ops.patch_asset_macro_charts --asset xauusd
  python -m scripts.ops.patch_asset_macro_charts --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from src.analysis.asset_macro_charts import ASSET_DRIVERS, build_macro_blocks

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "usdcop-trading-dashboard/public/data/analysis"


def _patch_file(fp: Path, asset_id: str, dry_run: bool) -> bool:
    with open(fp, encoding="utf-8") as f:
        view = json.load(f)

    week_end = view.get("week_end")
    if not week_end:
        logger.warning("  %s: no week_end, skipped", fp.name)
        return False

    snapshots, charts = build_macro_blocks(date.fromisoformat(week_end), asset_id)
    if not charts:
        logger.warning("  %s: builder returned no charts (macro data?), skipped", fp.name)
        return False

    view["macro_snapshots"] = snapshots
    view["macro_charts"] = charts

    if dry_run:
        logger.info("  %s: would write %d drivers", fp.name, len(charts))
        return True

    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(view, f, ensure_ascii=False, indent=1)
    tmp.replace(fp)
    logger.info("  %s: %d drivers (%s)", fp.name, len(charts), ", ".join(charts))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSET_DRIVERS), default=None,
                    help="single asset (default: all with drivers)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assets = [args.asset] if args.asset else sorted(ASSET_DRIVERS)
    grand = 0
    for asset_id in assets:
        adir = ANALYSIS_ROOT / asset_id
        if not adir.is_dir():
            logger.warning("%s: dir not found (%s), skipped", asset_id, adir)
            continue
        files = sorted(adir.glob("weekly_*_W*.json"))
        logger.info("%s: %d weekly files", asset_id, len(files))
        patched = sum(_patch_file(fp, asset_id, args.dry_run) for fp in files)
        logger.info("%s: patched %d/%d", asset_id, patched, len(files))
        grand += patched
    logger.info("Done. %d files %s.", grand, "would be patched" if args.dry_run else "patched")


if __name__ == "__main__":
    main()
