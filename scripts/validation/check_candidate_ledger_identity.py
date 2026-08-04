"""Read-only independent verifier for the tracked 2026 candidate paper ledger."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LEDGER = REPO / "usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json"
PRODUCER = REPO / "scripts/pipeline/candidates_paper_ledger.py"


def main() -> int:
    from src.identity.candidate_ledger import verify_candidate_ledger

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    producer_hash = "sha256:" + hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    identity = verify_candidate_ledger(ledger, producer_code_hash=producer_hash)
    print(
        "candidate ledger identity OK: "
        f"semantic_hash={identity['semantic_hash']} "
        f"decision_fingerprint={identity['decision_fingerprint']} "
        f"derivation_id={identity['derivation_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
