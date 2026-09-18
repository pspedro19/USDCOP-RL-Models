"""Read-only ledger integrity and PAPER-net coverage, not a forward certificate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .paths import DECISIONS_PATH, SETTLEMENTS_PATH
from .settlement_accounting import (
    aggregate,
    api_cost_summary,
    candidate_from_sources,
    check_chain,
    check_decision_schedules,
    identity,
    read_ledger,
    validate_accounting,
)


def audit_snapshot(decisions: list[dict], settlements: list[dict]) -> dict:
    """Every net row links to original decisions; unknown never becomes zero."""
    check_chain(decisions)
    check_chain(settlements)
    check_decision_schedules(decisions)
    groups = defaultdict(list)
    excluded = 0
    usable_ids = set()
    treatment_identities = defaultdict(set)
    for row in decisions:
        try:
            day, arm, provider, model, prereg = identity(row)
        except (KeyError, TypeError, ValueError, AttributeError):
            excluded += 1
            continue
        treatment_identities[arm].add((provider, model, prereg))
        if row.get("bar_index") is not None and row.get("decision_schedule") != "first_bar_hold":
            groups[(day, arm)].append(row)
        else:
            try:
                candidate_from_sources([row])
            except (KeyError, TypeError, ValueError):
                excluded += 1
            else:
                usable_ids.add(row["decision_id"])
    for group in groups.values():
        candidate = aggregate(group)
        if candidate is None:
            excluded += len(group)
        else:
            usable_ids.add(candidate["decision_id"])
    if any(len(values) > 1 for values in treatment_identities.values()):
        raise ValueError("provider/model/preregistration changed within an arm; separate treatment required")
    net_rows = []
    unknown = 0
    for row in settlements:
        net = validate_accounting(row, decisions)
        if net is None:
            unknown += 1
            continue
        if row["decision_id"] not in usable_ids:
            raise ValueError("accounted settlement is not an admissible decision/session")
        net_rows.append({
            "session_date": row["session_date"], "decision_id": row["decision_id"],
            "net_return": net, "settlement_sha256": row["record_hash"],
        })
    return {
        "decision_records": len(decisions), "admissible_session_arms": len(usable_ids),
        "excluded_decision_records": excluded, "settlement_records": len(settlements),
        "net_verified_sessions": len(net_rows), "legacy_net_unknown": unknown,
        "unsettled_admissible_session_arms": len(usable_ids - {r["decision_id"] for r in settlements}),
        "api_cost": api_cost_summary(decisions),
        "distinct_input_hashes": len(Counter(row.get("prompt_sha256") for row in decisions)),
        "input_hash_note": "Context/observation hashes are not necessarily prompt-template versions.",
        "daily_returns": sorted(net_rows, key=lambda row: (row["session_date"], row["decision_id"])),
        "scientific_ready": False,
        "scope": "paper_assumed_costs; no executable quote, PIT, durable seal or planned-coverage certificate",
    }


def main() -> int:
    try:
        result = audit_snapshot(read_ledger(DECISIONS_PATH), read_ledger(SETTLEMENTS_PATH))
    except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        print(f"FAIL ledger/accounting: {exc}")
        return 1
    print("OK chain/content/accounting validation; this is not scientific readiness")
    print(json.dumps({k: v for k, v in result.items() if k != "daily_returns"},
                     sort_keys=True, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
