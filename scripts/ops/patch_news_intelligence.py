"""
Backfill news_intelligence + political_bias_analysis into published weekly JSONs
================================================================================
Offline patcher for the USD/COP `/analysis` weekly view. For every
`public/data/analysis/weekly_YYYY_WXX.json` that is missing the news-intelligence
clusters (or the political-bias block), recompute them from the available article
pool (GDELT/Google/Investing CSVs + `news_articles` DB, via the same merged loader
the generator uses) and inject them — so `NewsClusterCard` and
`BiasDistributionCard` render without waiting for a full weekly regeneration.

This mirrors `scripts/ops/patch_asset_macro_charts.py`. It is idempotent and
non-destructive: weeks with no articles in-window are left untouched, and existing
non-empty `news_intelligence.clusters` are preserved unless `--force`.

Usage:
    python -m scripts.ops.patch_news_intelligence               # patch missing weeks
    python -m scripts.ops.patch_news_intelligence --force        # recompute all weeks
    python -m scripts.ops.patch_news_intelligence --year 2026    # limit to a year
    python -m scripts.ops.patch_news_intelligence --dry-run      # report, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "usdcop-trading-dashboard/public/data/analysis"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("patch_news_intelligence")

_WEEK_RE = re.compile(r"weekly_(\d{4})_W(\d{2})\.json$")


def _iter_week_files(year: int | None):
    for path in sorted(ANALYSIS_DIR.glob("weekly_*.json")):
        m = _WEEK_RE.search(path.name)
        if not m:
            continue
        iso_year, iso_week = int(m.group(1)), int(m.group(2))
        if year and iso_year != year:
            continue
        yield path, iso_year, iso_week


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=None, help="Limit to a single ISO year")
    ap.add_argument("--force", action="store_true", help="Recompute even if clusters exist")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = ap.parse_args()

    from src.analysis.weekly_generator import WeeklyAnalysisGenerator
    from src.contracts.analysis_schema import _sanitize_for_json

    # dry_run=True → deterministic (no LLM cost); still reads the full article pool.
    gen = WeeklyAnalysisGenerator(dry_run=True)

    patched = skipped = empty = 0
    for path, iso_year, iso_week in _iter_week_files(args.year):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("skip %s: unreadable (%s)", path.name, e)
            continue

        has_clusters = bool(data.get("news_intelligence", {}).get("clusters"))
        has_bias = bool(data.get("political_bias_analysis", {}).get("total_analyzed"))
        if has_clusters and has_bias and not args.force:
            skipped += 1
            continue

        start = date.fromisocalendar(iso_year, iso_week, 1)
        end = date.fromisocalendar(iso_year, iso_week, 5)
        ni, bias = gen._compute_news_intelligence(start, end)

        if not (ni and ni.get("clusters")) and not (bias and bias.get("total_analyzed")):
            empty += 1
            logger.info("%s: no articles in window — left untouched", path.name)
            continue

        changed = False
        if ni and ni.get("clusters") and (not has_clusters or args.force):
            data["news_intelligence"] = ni
            changed = True
        if bias and bias.get("total_analyzed") and (not has_bias or args.force):
            data["political_bias_analysis"] = bias
            changed = True

        if not changed:
            skipped += 1
            continue

        n_clusters = len(ni.get("clusters", [])) if ni else 0
        n_bias = bias.get("total_analyzed", 0) if bias else 0
        logger.info("%s: %d clusters, %d articles bias-analyzed%s",
                    path.name, n_clusters, n_bias, " (dry-run)" if args.dry_run else "")
        if not args.dry_run:
            path.write_text(
                json.dumps(_sanitize_for_json(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        patched += 1

    logger.info("Done: %d patched, %d skipped, %d empty-window", patched, skipped, empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
