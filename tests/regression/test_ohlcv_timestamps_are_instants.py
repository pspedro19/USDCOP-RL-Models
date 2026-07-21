"""Session-pair OHLCV timestamps must be true instants, mono-convention.

Contract: CTR-DQ-TZ-001

Found live on 2026-07-21: `usdcop_m5_ohlcv` mixed THREE timestamp conventions. The realtime
source stored true instants (session = 13:00-17:55 UTC); the backfill/gap-fill/manual sources
stored COT WALL CLOCK mislabeled as UTC (session appearing at 08:00-12:55 UTC = pre-dawn
Bogota instants). 15,865 rows were shifted +5h by `scripts/ops/fix_tz_wall_cot_rows.py`;
815 collisions resolved in favor of the instant-true row; 86 off-session strays deleted.

Root cause: TwelveData returns wall-clock strings in the REQUESTED timezone. The COP/MXN
fetch (api_tz=America/Bogota) parsed them naive and inserted them naive, so postgres read
Bogota wall time as UTC. BRL escaped only because its documented request-in-UTC quirk went
through a branch that localizes.

Why each convention survived alone: both are internally consistent with SOME reading of
"session", so no per-row validator complains. Only the DISTRIBUTION exposes the defect —
a session pair whose stored hours-of-day span both 8-12 and 13-17 UTC is speaking two
languages. This test checks the distribution.

DB-dependent: skips when postgres is unreachable (CI without the stack). The ingest-side
guard (localize before insert) is tested by construction in the backfill DAG source check.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SESSION_SYMBOLS = ("USD/COP", "USD/MXN", "USD/BRL")
SESSION_UTC_HOURS = (13, 17)   # 8:00-12:55 America/Bogota (no DST in Colombia)


def _conn():
    try:
        import psycopg2
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            user=os.environ.get("POSTGRES_USER", "admin"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            connect_timeout=3,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres unreachable: {e}")


def test_session_pairs_are_mono_convention_instants():
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol, source,
                   COUNT(*) FILTER (WHERE EXTRACT(hour FROM time AT TIME ZONE 'UTC')
                                    NOT BETWEEN %s AND %s) AS off_session
            FROM usdcop_m5_ohlcv
            WHERE symbol IN %s
            GROUP BY symbol, source
            HAVING COUNT(*) FILTER (WHERE EXTRACT(hour FROM time AT TIME ZONE 'UTC')
                                    NOT BETWEEN %s AND %s) > 0
            """,
            (SESSION_UTC_HOURS[0], SESSION_UTC_HOURS[1], SESSION_SYMBOLS,
             SESSION_UTC_HOURS[0], SESSION_UTC_HOURS[1]),
        )
        offenders = cur.fetchall()
    finally:
        conn.close()

    assert not offenders, (
        f"session-pair rows outside 13-17 UTC (8:00-12:55 COT instants): {offenders}. "
        "Either an ingest path regressed to wall-clock-as-UTC (check the localize guard in "
        "l0_ohlcv_backfill.fetch_ohlcv_data) or a new source arrived with its own convention. "
        "Two conventions in one column invert intraday ordering across sources."
    )


def test_backfill_localizes_naive_wall_clock():
    """The ingest-side guard must stay: naive COT wall times get localized before insert."""
    src = (ROOT / "airflow" / "dags" / "l0_ohlcv_backfill.py").read_text(
        encoding="utf-8", errors="replace")
    assert "elif bar_time.tzinfo is None" in src and "COT_TZ.localize(bar_time)" in src, (
        "the backfill lost its localize guard: naive TwelveData wall-clock strings will be "
        "inserted as UTC instants again, re-poisoning the table the migration just cleaned"
    )
