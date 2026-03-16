#!/usr/bin/env python3
"""
Generate Weekly Analysis CLI (SDD-07)
========================================
Generates AI-powered weekly analysis with macro charts and daily timeline.

Usage:
    # Generate for current week
    python scripts/generate_weekly_analysis.py

    # Generate for specific week
    python scripts/generate_weekly_analysis.py --week 2026-W09

    # Generate for specific date (daily only)
    python scripts/generate_weekly_analysis.py --date 2026-02-25

    # Backfill multiple weeks
    python scripts/generate_weekly_analysis.py --from 2026-W06 --to 2026-W09

    # Dry run (no LLM calls)
    python scripts/generate_weekly_analysis.py --dry-run

    # Export only (skip LLM, use cached data)
    python scripts/generate_weekly_analysis.py --export-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env for API keys (Azure OpenAI, Anthropic)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed, rely on shell env vars

from src.analysis.weekly_generator import WeeklyAnalysisGenerator


def parse_week_string(week_str: str) -> tuple[int, int]:
    """Parse 'YYYY-WNN' format to (year, week)."""
    parts = week_str.split("-W")
    if len(parts) != 2:
        raise ValueError(f"Invalid week format: {week_str}. Expected YYYY-WNN")
    return int(parts[0]), int(parts[1])


def print_summary(generator: WeeklyAnalysisGenerator, weeks_generated: int, elapsed: float):
    """Print a summary of the generation run."""
    output_dir = generator.export_dir
    print(f"\n{'='*60}")
    print(f"  Analysis Generation Complete")
    print(f"{'='*60}")
    print(f"  Weeks generated:  {weeks_generated}")
    print(f"  Charts generated: {generator._charts_generated}")
    print(f"  Total LLM cost:   ${generator.llm.total_cost:.4f}")
    print(f"  Total tokens:     {generator.llm.total_tokens}")
    print(f"  Elapsed time:     {elapsed:.1f}s")
    print(f"  Output directory:  {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate weekly analysis for USDCOP trading")
    parser.add_argument("--date", type=str, help="Generate daily analysis for specific date (YYYY-MM-DD)")
    parser.add_argument("--week", type=str, help="Generate weekly analysis (YYYY-WNN)")
    parser.add_argument("--from", dest="from_week", type=str, help="Backfill start week (YYYY-WNN)")
    parser.add_argument("--to", dest="to_week", type=str, help="Backfill end week (YYYY-WNN)")
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls, generate placeholder text")
    parser.add_argument("--export-only", action="store_true", help="Skip LLM, use cached data")
    parser.add_argument("--config", type=str, help="Path to analysis SSOT config")
    parser.add_argument("--legacy-pipeline", action="store_true", help="Use existing pipeline only (no LangGraph)")
    parser.add_argument("--langgraph-only", action="store_true", help="Use LangGraph pipeline only (skip existing LLM)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Determine execution mode
    if args.legacy_pipeline:
        mode = "legacy"
    elif args.langgraph_only:
        mode = "langgraph"
    else:
        mode = "default"

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create generator
    generator = WeeklyAnalysisGenerator(
        config_path=args.config,
        dry_run=args.dry_run,
        mode=mode,
    )

    overall_start = time.time()

    if args.date:
        # Single date
        target = date.fromisoformat(args.date)
        record = generator.generate_for_date(target)
        print(f"\n=== Daily Analysis: {target} ===")
        print(f"Headline: {record.headline}")
        print(f"Sentiment: {record.sentiment}")
        print(f"USD/COP: {record.usdcop_close}")
        print(f"H1 signal: {record.h1_signal}")
        print(f"H5 status: {record.h5_status}")
        print(f"News highlights: {len(record.news_highlights)}")
        print(f"Events: {len(record.economic_events)}")
        print(f"Tokens: {record.llm_tokens_used}, Cost: ${record.llm_cost_usd}")
        return

    if args.from_week and args.to_week:
        # Backfill range
        start_year, start_week = parse_week_string(args.from_week)
        end_year, end_week = parse_week_string(args.to_week)

        weeks_generated = 0
        weeks_failed = 0
        current_year, current_week = start_year, start_week
        while (current_year, current_week) <= (end_year, end_week):
            try:
                generator.generate_for_week(current_year, current_week)
                weeks_generated += 1
            except Exception as e:
                logging.error(f"Failed for {current_year}-W{current_week:02d}: {e}")
                weeks_failed += 1

            # Advance to next week
            current_week += 1
            if current_week > 52:
                current_week = 1
                current_year += 1

        elapsed = time.time() - overall_start
        print_summary(generator, weeks_generated, elapsed)
        if weeks_failed > 0:
            print(f"  WARNING: {weeks_failed} week(s) failed")
        return

    # Default: current week or specified week
    if args.week:
        year, week = parse_week_string(args.week)
    else:
        today = date.today()
        iso_cal = today.isocalendar()
        year, week = iso_cal[0], iso_cal[1]

    export = generator.generate_for_week(year, week)

    elapsed = time.time() - overall_start
    print(f"\n=== Weekly Analysis: {year}-W{week:02d} (mode={mode}) ===")
    print(f"Daily entries:   {len(export.daily_entries)}")
    print(f"Macro variables: {len(export.macro_snapshots)}")
    print(f"Charts:          {len(export.macro_charts)}")
    print(f"H5 signal:       {export.signals.get('h5', {}).get('direction', 'N/A')}")
    print(f"H1 signal:       {export.signals.get('h1', {}).get('direction', 'N/A')}")
    print(f"News articles:   {export.news_context.get('article_count', 0)}")
    print(f"Events:          {len(export.upcoming_events)}")
    # LangGraph enrichment status
    if hasattr(export, "technical_analysis") and export.technical_analysis:
        print(f"TA bias:         {export.technical_analysis.get('dominant_bias', 'N/A')}")
    if hasattr(export, "macro_regime") and export.macro_regime:
        regime = export.macro_regime.get("regime", {})
        print(f"Macro regime:    {regime.get('label', 'N/A')}")
    if hasattr(export, "quality_score") and export.quality_score is not None:
        print(f"Quality score:   {export.quality_score:.2f}")
    print(f"Total LLM cost:  ${generator.llm.total_cost:.4f}")
    print(f"Total tokens:    {generator.llm.total_tokens}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print(f"Output:          {generator.export_dir}")


if __name__ == "__main__":
    main()
