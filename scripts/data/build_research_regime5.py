#!/usr/bin/env python
"""Prepare the new schema and audit readiness without fitting or provider calls.

Actual construction is exposed by regime5_bundle.fit_development/session/export_bundle
for a future admitted build. This command deliberately cannot bypass missing
publication evidence, trial reconciliation or the new sanity/experiment freeze.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.macro_evidence import canonical_json, immutable_write  # noqa: E402
from src.research.observation_contract import REGIME5_VERSION, observation_contract  # noqa: E402
from src.research.regime5_bundle import safe_path  # noqa: E402
from src.research.research_readiness import audit_preparation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/research/research_regime5_v1.yaml"
    )
    parser.add_argument("--export-schema", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new immutable readiness report")
    args = parser.parse_args()
    if args.export_schema:
        immutable_write(
            ROOT / "config/research/observation_regime5_v1.json",
            canonical_json(observation_contract(REGIME5_VERSION).to_dict()),
        )
    result = audit_preparation(args.config)
    immutable_write(safe_path(args.output), canonical_json(result))
    print("retrospective_delivery_ready=" + str(result["retrospective_delivery_ready"]))
    print("training_ready=False; pilot_executed=False; no fit or API calls")
    for name, check in result["checks"].items():
        print(f"{name}: {check['status']}" + (f" — {check['reason']}" if "reason" in check else ""))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
