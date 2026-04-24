#!/usr/bin/env python3
"""
Analysis Quality Verifier (SDD)
================================
Post-backfill validation: checks every weekly JSON meets minimum standards.

Run: python3 scripts/verify_analysis_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Weeks where news gap is expected and accepted
# W09 (Feb 24-28): last week of historical CSVs, may have some
# W10 (Mar 2-6): gap between historical CSVs (end Feb 26) and live pipeline (start Mar 16)
# W11 (Mar 9-13): same gap
GAP_WEEKS = {"2026-W09", "2026-W10", "2026-W11"}

CHECKS = {
    "has_weekly_summary": lambda j: bool(
        j.get("weekly_summary", {}).get("markdown")
        or j.get("weekly_summary", {}).get("summary_markdown")
    ),
    "has_headline": lambda j: bool(
        j.get("weekly_summary", {}).get("headline")
    ),
    "has_sentiment": lambda j: j.get("weekly_summary", {}).get("sentiment") is not None,
    "has_fuentes_urls": lambda j: _has_fuentes_with_urls(j),
    "has_news_or_gap": lambda j: (
        _count_news(j) > 0
        or _get_week_label(j, j.get("_filename", "")) in GAP_WEEKS
    ),
    "has_macro_snapshots": lambda j: len(j.get("macro_snapshots", {})) >= 4,
    "has_daily_entries": lambda j: len(j.get("daily_entries", [])) >= 1,
    "no_lfs_pointer": lambda j: "git-lfs.github.com" not in json.dumps(j)[:200],
    "json_no_infinity": lambda j: "Infinity" not in json.dumps(j),
    "json_no_nan": lambda j: "NaN" not in json.dumps(j),
    "analysis_not_empty": lambda j: _analysis_length(j) > 100,
}


def _has_fuentes_with_urls(j: dict) -> bool:
    """Check if the markdown contains a Fuentes section with real URLs."""
    md = (
        j.get("weekly_summary", {}).get("markdown", "")
        or j.get("weekly_summary", {}).get("summary_markdown", "")
        or ""
    )
    if "### Fuentes" not in md and "**Datos Macro:**" not in md:
        return False
    fuentes_part = md.split("### Fuentes")[-1] if "### Fuentes" in md else md.split("**Datos Macro:**")[-1]
    return "https://" in fuentes_part


def _count_news(j: dict) -> int:
    """Count total news highlights across daily entries + week-level context."""
    daily_count = sum(
        len(de.get("news_highlights", []))
        for de in j.get("daily_entries", [])
    )
    week_count = j.get("news_context", {}).get("article_count", 0)
    return daily_count + week_count


def _get_week_label(j: dict, filename: str = "") -> str:
    """Extract week label like '2026-W09' from the JSON or filename."""
    ws = j.get("weekly_summary", {})
    year = ws.get("iso_year", j.get("iso_year", 0))
    week = ws.get("iso_week", j.get("iso_week", 0))
    if year and week:
        return f"{year}-W{week:02d}"
    # Fallback: parse from filename (e.g., "weekly_2026_W10.json" -> "2026-W10")
    if filename:
        import re
        m = re.search(r"(\d{4})_W(\d{2})", filename)
        if m:
            return f"{m.group(1)}-W{m.group(2)}"
    return ""


def _analysis_length(j: dict) -> int:
    """Get length of the main analysis text."""
    md = (
        j.get("weekly_summary", {}).get("markdown", "")
        or j.get("weekly_summary", {}).get("summary_markdown", "")
        or ""
    )
    return len(md)


def verify_all(analysis_dir: str | None = None) -> dict:
    """Run all checks on all weekly JSONs."""
    if analysis_dir is None:
        # Try common locations
        candidates = [
            Path("usdcop-trading-dashboard/public/data/analysis"),
            Path("public/data/analysis"),
            Path("data/analysis"),
        ]
        for c in candidates:
            if c.exists():
                analysis_dir = c
                break
        else:
            print("ERROR: Could not find analysis directory")
            sys.exit(1)
    else:
        analysis_dir = Path(analysis_dir)

    results = {}
    files = sorted(analysis_dir.glob("weekly_2026_W*.json"))

    if not files:
        print(f"No weekly JSON files found in {analysis_dir}")
        sys.exit(1)

    print(f"Checking {len(files)} weekly files in {analysis_dir}\n")

    for path in files:
        week = path.stem.replace("weekly_", "").replace("_", "-")
        try:
            data = json.loads(path.read_text())
            data["_filename"] = path.name  # Inject filename for gap week detection
        except json.JSONDecodeError as e:
            results[week] = {"status": "❌", "failures": [f"invalid JSON: {e}"]}
            print(f"❌ {week}: invalid JSON")
            continue

        failures = []
        for check_name, check_fn in CHECKS.items():
            try:
                if not check_fn(data):
                    failures.append(check_name)
            except Exception as e:
                failures.append(f"{check_name} (ERROR: {e})")

        status = "✅" if not failures else "❌"
        results[week] = {"status": status, "failures": failures}

        if failures:
            print(f"{status} {week}: FAILED — {', '.join(failures)}")
        else:
            print(f"{status} {week}: all {len(CHECKS)} checks passed")

    total = len(results)
    passed = sum(1 for r in results.values() if r["status"] == "✅")
    print(f"\n{'=' * 50}")
    print(f"Result: {passed}/{total} weeks passed all checks")

    if passed < total:
        failed_weeks = [w for w, r in results.items() if r["status"] != "✅"]
        print(f"Failed: {', '.join(failed_weeks)}")

    return results


if __name__ == "__main__":
    verify_all()
