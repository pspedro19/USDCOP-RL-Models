"""Append-only, hash-chained JSONL ledger.

JSONL rather than a database for one reason: it survives. A ``.jsonl`` file is
readable in ten years with `cat`, diffs cleanly in git, and needs no running
service to audit. The chain gives it the integrity property a database would have
given through permissions.

Two invariants are enforced on every append:

1. **No duplicate key.** One decision per session, one settlement per decision.
   A re-run that would overwrite raises instead.
2. **Chain continuity.** ``prev_hash`` of the new record must equal the tip.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .canonical import GENESIS_HASH, chain_hash


class LedgerError(RuntimeError):
    """Raised when an append would violate an invariant."""


class Ledger:
    """A single append-only chained file.

    Args:
        path: JSONL file. Created on first append.
        key_field: field whose value must be unique across records.
    """

    def __init__(self, path: str | Path, key_field: str) -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- reading ----------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield records in write order. Empty if the file does not exist yet."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def tip_hash(self) -> str:
        """Hash of the last record, or the genesis anchor for an empty ledger."""
        last = None
        for record in self:
            last = record
        return GENESIS_HASH if last is None else last["record_hash"]

    def next_seq(self) -> int:
        return sum(1 for _ in self)

    def keys(self) -> set[str]:
        return {record[self.key_field] for record in self}

    # ---------- writing ----------

    def append(self, record: Any) -> dict[str, Any]:
        """Chain and append one dataclass record.

        Returns the serialized dict actually written, hashes included.

        Raises:
            LedgerError: on duplicate key.
        """
        key = getattr(record, self.key_field)
        if key in self.keys():
            raise LedgerError(
                f"{self.key_field}={key!r} already in {self.path.name}. "
                "Records are immutable; delete the file only if you are "
                "deliberately discarding the whole chain."
            )

        record.prev_hash = self.tip_hash()
        record.seq = self.next_seq()
        record.record_hash = chain_hash(record.prev_hash, record.hashable_payload())

        payload = asdict(record)
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True)

        # Append + fsync. Without the flush/fsync a crash between the write and
        # the OS flush can truncate the last record, and a truncated JSON line
        # breaks the chain in a way that looks like tampering.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        return payload

    # ---------- auditing ----------

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain end to end.

        Returns:
            ``(True, message)`` if intact, ``(False, reason)`` at the first break.
        """
        expected_prev = GENESIS_HASH
        count = 0

        for index, record in enumerate(self):
            stored_hash = record.get("record_hash")
            stored_prev = record.get("prev_hash")

            if stored_prev != expected_prev:
                return False, (
                    f"record {index} ({record.get(self.key_field)}): prev_hash "
                    f"{stored_prev[:12]}... does not match the previous record's "
                    f"hash {expected_prev[:12]}..."
                )

            payload = {
                k: v for k, v in record.items()
                if k not in ("prev_hash", "record_hash")
            }
            recomputed = chain_hash(stored_prev, payload)
            if recomputed != stored_hash:
                return False, (
                    f"record {index} ({record.get(self.key_field)}): content was "
                    f"modified after writing (hash mismatch)"
                )

            expected_prev = stored_hash
            count += 1

        return True, f"chain intact, {count} record(s), tip {expected_prev[:12]}..."
