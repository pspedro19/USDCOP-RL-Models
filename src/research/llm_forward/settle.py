"""Job 2 — settle sealed decisions against realized prices.

Runs after the close, as a separate process. It reads the decision ledger and
writes to a different file; it never opens ``decisions.jsonl`` for writing.

    python -m llmfwd.settle --price-csv data/prices/usdcop_5m.csv

Only decisions that are (a) sealed before the open and (b) not already settled
are processed. Refusing to re-settle matters: a settlement you can re-run is a
settlement you can re-run until you like the number.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from .ledger import Ledger, LedgerError
from .schema import SettlementRecord, utc_now_iso

from .paths import DECISIONS_PATH, SETTLEMENTS_PATH  # rutas centralizadas

DIRECTION_SIGN = {"long": 1.0, "short": -1.0, "flat": 0.0}


def load_session_bars(price_csv: Path) -> dict[str, list[tuple[datetime, float]]]:
    """Group 5-minute bars by session date.

    Expects columns ``timestamp_utc`` (ISO-8601) and ``close``. A dict of lists
    rather than a DataFrame keeps this module dependency-free — pandas is a fine
    tool but a settlement job that can run on a bare interpreter is one less
    thing to break in two years.
    """
    sessions: dict[str, list[tuple[datetime, float]]] = {}

    with price_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["timestamp_utc"])
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            session_date = stamp.astimezone(timezone.utc).date().isoformat()
            sessions.setdefault(session_date, []).append((stamp, float(row["close"])))

    for bars in sessions.values():
        bars.sort(key=lambda pair: pair[0])
    return sessions


def run(price_csv: Path, min_bars: int = 10) -> int:
    decisions = Ledger(DECISIONS_PATH, "decision_id")
    settlements = Ledger(SETTLEMENTS_PATH, "decision_id")

    already_settled = settlements.keys()
    sessions = load_session_bars(price_csv)
    settled_count = 0

    for record in decisions:
        decision_id = record["decision_id"]

        if decision_id in already_settled:
            continue
        if not record["sealed_before_open"]:
            print(f"  [skip] {decision_id}: not sealed before open")
            continue
        if record["abstained"]:
            print(f"  [skip] {decision_id}: abstained, nothing to settle")
            continue

        bars = sessions.get(record["session_date"], [])
        if len(bars) < min_bars:
            print(f"  [wait] {decision_id}: only {len(bars)} bar(s) available")
            continue

        open_price = bars[0][1]
        close_price = bars[-1][1]
        realized = (close_price - open_price) / open_price
        sign = DIRECTION_SIGN[record["decision"]["direction"]]

        settlement = SettlementRecord(
            seq=-1,
            decision_id=decision_id,
            session_date=record["session_date"],
            settled_at_utc=utc_now_iso(),
            open_price=open_price,
            close_price=close_price,
            realized_return=round(realized, 8),
            # Gross of costs, on purpose. Costs are applied downstream from a
            # documented model so the same settlement can be re-analysed under
            # different cost assumptions without being rewritten.
            signed_return=round(realized * sign, 8),
            bars_observed=len(bars),
        )

        try:
            written = settlements.append(settlement)
        except LedgerError as exc:
            print(f"  [refused] {exc}")
            continue

        print(
            f"  settled {decision_id}: signed_return="
            f"{settlement.signed_return:+.5f} hash={written['record_hash'][:12]}..."
        )
        settled_count += 1

    print(f"\n{settled_count} settlement(s) written")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Settle sealed decisions.")
    parser.add_argument("--price-csv", required=True, type=Path)
    parser.add_argument("--min-bars", type=int, default=10)
    args = parser.parse_args()
    return run(args.price_csv, args.min_bars)


if __name__ == "__main__":
    raise SystemExit(main())
