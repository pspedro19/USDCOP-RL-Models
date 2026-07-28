#!/usr/bin/env python
"""Governed CLI for non-idempotent research-family lifecycle and trial charging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.research.qlab import FamilyStore, TrialCharge, TrialLedger

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = FamilyStore(ROOT / "registries" / "families")
LEDGER = TrialLedger(ROOT / "registries" / "ledger.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlab")
    sub = parser.add_subparsers(dest="command", required=True)
    declare = sub.add_parser("family-declare")
    declare.add_argument("family_id")
    declare.add_argument("--kind", choices=("forecast", "action"), required=True)
    declare.add_argument("--cluster", required=True)
    declare.add_argument("--asset", required=True)
    declare.add_argument("--question", required=True)
    declare.add_argument("--bar", required=True)

    screen = sub.add_parser("screen")
    screen.add_argument("family_id")
    screen.add_argument("--charge-trial", action="store_true", required=True)
    screen.add_argument("--trial-id", required=True)
    screen.add_argument("--asset", required=True)
    screen.add_argument("--variant", required=True)
    screen.add_argument("--cutoff", required=True)
    screen.add_argument("--result", default="pending")
    screen.add_argument("--source")
    screen.add_argument("--code-hash")
    screen.add_argument("--data-hash")
    screen.add_argument("--note")

    for name, target in (("freeze", "FROZEN"), ("promote", "PROMOTED"), ("close", "CLOSED")):
        command = sub.add_parser(name)
        command.set_defaults(target_state=target)
        command.add_argument("family_id")
        command.add_argument("--reason", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "family-declare":
        path = FAMILIES.declare(
            {
                "family_id": args.family_id,
                "kind": args.kind,
                "cluster_id": args.cluster,
                "asset": args.asset,
                "question": args.question,
                "bar": args.bar,
                "provenance": {"crosses_wall": False},
            }
        )
        print(json.dumps({"status": "DECLARED", "path": str(path)}))
        return 0
    if args.command == "screen":
        family = FAMILIES.load(args.family_id)
        state = str(family.get("state", "DECLARED")).upper()
        if state not in {"DECLARED", "SCREENING"}:
            raise SystemExit(f"family state {state} does not admit new screening trials")
        row = LEDGER.charge(
            TrialCharge(
                trial_id=args.trial_id,
                family=args.family_id,
                asset=args.asset,
                cluster=str(family["cluster_id"]),
                kind=str(family["kind"]),
                variant=args.variant,
                cutoff=args.cutoff,
                result=args.result,
                source=args.source,
                code_hash=args.code_hash,
                data_hash=args.data_hash,
                note=args.note,
            )
        )
        FAMILIES.record_trial(args.family_id, row)
        if state == "DECLARED":
            FAMILIES.transition(
                args.family_id, "SCREENING", reason="first charged screening trial"
            )
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0
    path = FAMILIES.transition(args.family_id, args.target_state, reason=args.reason)
    print(json.dumps({"status": args.target_state, "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
