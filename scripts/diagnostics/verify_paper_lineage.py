"""Report RESOLVED/BROKEN/ABSENT for one paper-ledger strategy lineage path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database import get_psycopg2_connection  # noqa: E402
from src.lineage.paper_path import PaperPathStatus, verify_paper_path  # noqa: E402

DEFAULT_LEDGER = ROOT / "usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json"
EXIT_CODES = {
    PaperPathStatus.RESOLVED: 0,
    PaperPathStatus.BROKEN: 1,
    PaperPathStatus.ABSENT: 2,
}


def _has_lineage_declaration(ledger: dict, strategy_id: str) -> bool:
    strategies = ledger.get("strategies")
    if not isinstance(strategies, dict):
        return False
    strategy = strategies.get(strategy_id)
    return isinstance(strategy, dict) and strategy.get("lineage") is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-id", default="smart_simple_v11")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    if _has_lineage_declaration(ledger, args.strategy_id):
        with get_psycopg2_connection() as connection:
            result = verify_paper_path(connection, ledger, strategy_id=args.strategy_id)
    else:
        result = verify_paper_path(None, ledger, strategy_id=args.strategy_id)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "verified": result.verified,
                "coverage": 1 if result.verified else 0,
                "strategy_id": result.strategy_id,
                "node_ids": result.node_ids,
                "detail": result.detail,
            },
            sort_keys=True,
        )
    )
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
