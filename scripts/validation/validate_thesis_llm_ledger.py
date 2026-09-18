#!/usr/bin/env python
"""Validate an LLM decision ledger before settlement/reporting."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HASH = re.compile(r"^[0-9a-f]{64}$")
SECRET_FIELDS = {"api_key", "authorization", "secret", "token", "raw_response"}


def validate(path: Path, *, require_complete_sessions: bool = False,
             expected_dataset_sha256: str | None = None,
             expected_dataset_block: str | None = None,
             forbid_retrospective: bool = False) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: row is not an object")
            missing = {"decision_id", "session_date", "bar", "prompt_hash", "raw_response_sha256",
                       "model_id", "prompt_version", "max_tokens", "temperature", "top_p"} - row.keys()
            if missing:
                raise ValueError(f"line {line_no}: missing fields {sorted(missing)}")
            if any(field in row for field in SECRET_FIELDS):
                raise ValueError(f"line {line_no}: secret/raw response field present")
            if not HASH.fullmatch(str(row["prompt_hash"])) or not HASH.fullmatch(str(row["raw_response_sha256"])):
                raise ValueError(f"line {line_no}: invalid hash")
            if (expected_dataset_sha256 is not None
                    and row.get("dataset_sha256") != expected_dataset_sha256):
                raise ValueError(f"line {line_no}: dataset hash mismatch")
            if expected_dataset_block is not None and row.get("dataset_block") != expected_dataset_block:
                raise ValueError(f"line {line_no}: dataset block mismatch")
            if forbid_retrospective and row.get("retrospective") is not False:
                raise ValueError(f"line {line_no}: retrospective row forbidden")
            if not isinstance(row["bar"], int) or not 0 <= row["bar"] <= 58:
                raise ValueError(f"line {line_no}: bar outside [0,58]")
            if float(row["temperature"]) != 0.10 or float(row["top_p"]) != 0.90 or int(row["max_tokens"]) != 256:
                raise ValueError(f"line {line_no}: sampling parameters differ from preregistration")
            rows.append(row)
    ids = [str(row["decision_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate decision_id")
    by_session: dict[str, set[int]] = {}
    for row in rows:
        by_session.setdefault(str(row["session_date"]), set()).add(int(row["bar"]))
    incomplete = {session: sorted(set(range(59)) - bars)
                  for session, bars in by_session.items() if bars != set(range(59))}
    if require_complete_sessions and incomplete:
        raise ValueError(f"incomplete sessions: {sorted(incomplete)}")
    return {"rows": len(rows), "sessions": len(by_session), "incomplete_sessions": incomplete,
            "valid": True, "secrets_checked": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--require-complete-sessions", action="store_true")
    parser.add_argument("--expected-dataset-sha256", type=str,
                        help="optional portable hash required on every ledger row")
    parser.add_argument("--expected-dataset-block", type=str,
                        help="optional dataset block required on every ledger row")
    parser.add_argument("--forbid-retrospective", action="store_true",
                        help="reject rows not explicitly marked retrospective=false")
    args = parser.parse_args()
    try:
        result = validate(args.ledger, require_complete_sessions=args.require_complete_sessions,
                          expected_dataset_sha256=args.expected_dataset_sha256,
                          expected_dataset_block=args.expected_dataset_block,
                          forbid_retrospective=args.forbid_retrospective)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"llm_ledger_invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
