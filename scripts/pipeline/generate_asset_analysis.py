"""
generate_asset_analysis.py — CLI for multi-asset (Gold/BTC) weekly analysis.

Produces the same WeeklyViewData JSON the /analysis page consumes, namespaced
per asset (public/data/analysis/<asset>/), using ONLY real data (OHLCV seeds,
computed technicals, published strategy trades, live GDELT news).

Examples
--------
  # One asset, one week
  python scripts/pipeline/generate_asset_analysis.py --asset xauusd --week 2026-W05

  # Backfill a range for one asset
  python scripts/pipeline/generate_asset_analysis.py --asset btcusdt --from 2026-W01 --to 2026-W27

  # All configured assets, all weeks present in their OHLCV for a year
  python scripts/pipeline/generate_asset_analysis.py --all-assets --year 2026

  # Skip live news (fast/offline) — technicals + signals still real
  python scripts/pipeline/generate_asset_analysis.py --asset xauusd --year 2026 --no-news
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.asset_analysis_generator import AssetAnalysisGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_asset_analysis")

_WEEK_RE = re.compile(r"^(\d{4})-W(\d{1,2})$")


def _parse_week(s: str) -> tuple[int, int]:
    m = _WEEK_RE.match(s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid week '{s}', expected YYYY-Www")
    return int(m.group(1)), int(m.group(2))


def _weeks_present(gen: AssetAnalysisGenerator, asset: str, year: int) -> list[int]:
    """ISO weeks of `year` that actually have OHLCV bars for the asset."""
    p = gen.profile(asset)
    df = gen._load_ohlcv(p)
    iso = df["time"].dt.isocalendar()
    mask = iso["year"] == year
    return sorted(set(int(w) for w in iso["week"][mask].unique()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-asset weekly analysis generator")
    ap.add_argument("--asset", type=str, help="Asset id (xauusd, btcusdt)")
    ap.add_argument("--all-assets", action="store_true", help="Generate for every configured asset")
    ap.add_argument("--week", type=str, help="Single ISO week YYYY-Www")
    ap.add_argument("--from", dest="from_week", type=str, help="Range start YYYY-Www")
    ap.add_argument("--to", dest="to_week", type=str, help="Range end YYYY-Www")
    ap.add_argument("--year", type=int, help="Generate all weeks present in OHLCV for this year")
    ap.add_argument("--no-news", action="store_true", help="Skip live GDELT news fetch")
    ap.add_argument("--config", type=str, help="Path to analysis_assets.yaml")
    args = ap.parse_args()

    gen = AssetAnalysisGenerator(config_path=args.config)

    if args.all_assets:
        assets = gen.known_assets()
    elif args.asset:
        assets = [args.asset]
    else:
        ap.error("Provide --asset <id> or --all-assets")
        return 2

    generated_at = datetime.now(timezone.utc).isoformat()
    total_written = 0

    for asset in assets:
        try:
            gen.profile(asset)
        except KeyError as e:
            logger.error(str(e))
            return 2

        # Resolve the target weeks
        if args.week:
            y, w = _parse_week(args.week)
            targets = [(y, w)]
        elif args.from_week and args.to_week:
            y1, w1 = _parse_week(args.from_week)
            y2, w2 = _parse_week(args.to_week)
            targets = []
            for y in range(y1, y2 + 1):
                lo = w1 if y == y1 else 1
                hi = w2 if y == y2 else 53
                for w in range(lo, hi + 1):
                    try:
                        date.fromisocalendar(y, w, 1)
                        targets.append((y, w))
                    except ValueError:
                        pass
        elif args.year:
            targets = [(args.year, w) for w in _weeks_present(gen, asset, args.year)]
        else:
            ap.error("Provide --week, --from/--to, or --year")
            return 2

        written = 0
        for (y, w) in targets:
            try:
                view = gen.generate_week(asset, y, w, skip_news=args.no_news)
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] %s-W%02d failed: %s", asset, y, w, e)
                continue
            if view is None:
                continue
            out = gen.write_week(asset, view, generated_at)
            written += 1
            logger.info("[%s] wrote %s (news=%d, sentiment=%s)",
                        asset, out.name, view["news_context"]["article_count"],
                        view["weekly_summary"]["sentiment"])

        if written:
            idx = gen.rebuild_index(asset)
            logger.info("[%s] rebuilt index (%s) with %d weeks", asset, idx.name, written)
        total_written += written

    logger.info("Done. %d weekly files written across %d asset(s).", total_written, len(assets))
    return 0 if total_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
