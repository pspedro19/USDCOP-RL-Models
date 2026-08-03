"""Post-run Excel exporter for the sealed intraday LatAm directional audit.

The registered statistical run writes CSVs before Excel.  Audit timestamp
columns are timezone-aware, which Excel cannot store natively.  This exporter
does not refit models or recompute outcomes: it reads the frozen CSV artifacts
as strings and packages those exact values into the requested workbook.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_forward_flow_directional_audit import (
    safe_json,
    sha256_file,
    write_workbook,
)
from scripts.analysis.usdcop_intraday_latam_directional_audit import (
    build_research_frame,
)


DEFAULT_CONFIG = (
    ROOT / "config/forecast_experiments/usdcop_intraday_latam_lead_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = ROOT / config["outputs"]["directory"]
    predictions = pd.read_csv(output_dir / config["outputs"]["predictions"])
    summary = pd.read_csv(output_dir / config["outputs"]["summary"])
    tests = pd.read_csv(output_dir / config["outputs"]["paired_tests"])
    decisions = pd.read_csv(output_dir / "horizon_decisions.csv")
    manifest_path = output_dir / config["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, _, coverage = build_research_frame(config)
    feature_coverage = pd.DataFrame(coverage)
    feature_coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    overview = pd.DataFrame([
        {"field": "experiment_id", "value": manifest["experiment_id"]},
        {"field": "evidence_class", "value": manifest["evidence_class"]},
        {"field": "all_7_horizons_tested", "value": True},
        {"field": "trials_opened", "value": manifest["directional_trials_opened"]},
        {"field": "primary_gate_pass_count", "value": manifest["primary_gate_pass_count"]},
        {"field": "shadow_candidate_count", "value": manifest["prospective_shadow_candidate_count"]},
        {"field": "capital_authorized", "value": False},
        {"field": "config_sha256", "value": manifest["config_sha256"]},
        {"field": "warning", "value": manifest["warning"]},
    ])
    workbook_path = output_dir / config["outputs"]["workbook"]
    write_workbook(
        workbook_path,
        overview,
        decisions,
        summary,
        tests,
        predictions,
        feature_coverage,
    )
    manifest["workbook_export"] = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "post_run_csv_packaging_no_refit_no_metric_recompute",
        "exporter_sha256": sha256_file(Path(__file__)),
        "workbook_sha256": sha256_file(workbook_path),
    }
    manifest_path.write_text(
        json.dumps(safe_json(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(workbook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
