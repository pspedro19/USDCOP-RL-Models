#!/usr/bin/env python
"""Governed CLI for non-idempotent research-family lifecycle and trial charging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.point_in_time import ResearchEnvironment, read_point_in_time
from src.research.qlab import FamilyStore, TrialCharge, TrialLedger

FAMILIES = FamilyStore(ROOT / "registries" / "families")
LEDGER = TrialLedger(ROOT / "registries" / "ledger.jsonl")


def _normalize_cutoff(value: str) -> str:
    """Make date-only CLI cutoffs explicit and timezone-aware."""
    if len(value) == 10:
        return f"{value}T23:59:59.999999Z"
    return value


def _source_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise SystemExit(f"screening source does not exist: {path}")
    return path


def _rows_from_source(*, source_path: Path, **_: Any) -> list[Mapping[str, Any]]:
    """Read a finite evidence file; cutoff enforcement stays in the PIT layer.

    The reader deliberately returns every row. ``read_point_in_time`` passes the
    cutoff into this function and then independently checks the materialized
    result, so a reader that ignores or mishandles pushdown cannot leak a future
    observation into screening.
    """

    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        rows: Any = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    elif suffix == ".csv":
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix in {".parquet", ".pq"}:
        import pandas as pd

        rows = pd.read_parquet(source_path).to_dict(orient="records")
    else:
        raise SystemExit(
            f"unsupported screening source {source_path}; "
            "expected .jsonl, .json, .csv or .parquet"
        )

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SystemExit("screening source must contain a sequence of row objects")
    materialized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SystemExit(f"screening source row {index} is not an object")
        materialized.append(row)
    return materialized


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
    screen.add_argument(
        "--source",
        required=True,
        help="Point-in-time evidence file (.jsonl/.json/.csv/.parquet)",
    )
    screen.add_argument("--available-at-field", default="available_at")
    screen.add_argument("--code-hash")
    screen.add_argument("--data-hash")
    screen.add_argument("--note")

    for name, target in (("freeze", "FROZEN"), ("promote", "PROMOTED"), ("close", "CLOSED")):
        command = sub.add_parser(name)
        command.set_defaults(target_state=target)
        command.add_argument("family_id")
        command.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        cutoff = _normalize_cutoff(args.cutoff)
        source_path = _source_path(args.source)
        rows = read_point_in_time(
            _rows_from_source,
            cutoff=cutoff,
            environment=ResearchEnvironment.SCREENING,
            available_at_field=args.available_at_field,
            source_path=source_path,
        )
        if not rows:
            raise SystemExit("screening source has zero rows at the declared cutoff")
        source_digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
        row = LEDGER.charge(
            TrialCharge(
                trial_id=args.trial_id,
                family=args.family_id,
                asset=args.asset,
                cluster=str(family["cluster_id"]),
                kind=str(family["kind"]),
                variant=args.variant,
                cutoff=cutoff,
                result=args.result,
                source=str(source_path),
                code_hash=args.code_hash,
                data_hash=args.data_hash or source_digest,
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
