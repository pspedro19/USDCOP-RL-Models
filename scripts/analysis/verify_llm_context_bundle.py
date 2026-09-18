"""Verify counts, dataset binding and retrospective labels of exported LLM contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(path: Path, expected_hash: str, expected_block: str, expected_count: int,
           retrospective: bool) -> int:
    count = 0
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if row.get("dataset_sha256") != expected_hash:
                errors.append(f"line {line_no}: dataset hash mismatch")
            if row.get("dataset_block") != expected_block:
                errors.append(f"line {line_no}: block mismatch")
            if bool(row.get("retrospective")) is not retrospective:
                errors.append(f"line {line_no}: retrospective label mismatch")
            if count == 1 and row.get("feature_order") is None:
                errors.append("first context has no feature_order")
    if count != expected_count:
        errors.append(f"context count {count} != {expected_count}")
    if errors:
        for error in errors[:10]:
            print(f"FAIL: {error}")
        return 1
    print(f"CONTEXT_BUNDLE_PASS {path.name} count={count} block={expected_block}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.portable.read_bytes()).hexdigest()
    rc = verify(args.selection, digest, "selection", 13_334, True)
    rc |= verify(args.holdout, digest, "holdout", 24_780, True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
