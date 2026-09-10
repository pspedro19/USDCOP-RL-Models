"""Resolve one typed market quarantine through the governed C027 correction boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.ingest_asset_ohlcv import _db_conn  # noqa: E402
from src.data_quality.corrections import (  # noqa: E402
    CorrectionRequest,
    apply_market_correction,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quarantine_id")
    parser.add_argument("--revision-type", required=True)
    parser.add_argument("--corrected-record", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--compared-provider")
    args = parser.parse_args(argv)

    record = json.loads(args.corrected_record.read_text(encoding="utf-8"))
    request = CorrectionRequest(
        revision_type=args.revision_type,
        corrected_record=record,
        reason=args.reason,
        corrected_by=args.actor,
        compared_provider_id=args.compared_provider,
    )
    conn = _db_conn()
    try:
        result = apply_market_correction(
            conn, quarantine_id=args.quarantine_id, request=request
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "correction_event_id": result.correction_event_id,
                "canonical_count": result.canonical_count,
                "idempotent": result.idempotent,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
