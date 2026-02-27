#!/usr/bin/env python3
"""
cron_monitor.py — Monitor active H5 position for TP/HS triggers
================================================================
Cron replacement for Airflow DAG: forecast_h5_l7_multiday_executor (monitor task)

Checks the latest M5 bar against the active execution's take-profit and
hard-stop levels. If triggered, closes the position and updates both
forecast_h5_subtrades and forecast_h5_executions.

No re-entry after exit (Smart Simple v1.0 policy).

Schedule: Every 30 min, 08:00-13:00 COT (= 13:00-18:00 UTC), Mon-Fri

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0 (cron)
Date: 2026-02-17
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import psycopg2
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cron_monitor")

COT_TZ = pytz.timezone("America/Bogota")

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ["POSTGRES_PASSWORD"],
    )


# ---------------------------------------------------------------------------
# Queries (mirrored from forecast_h5_l7_multiday_executor.py)
# ---------------------------------------------------------------------------

def get_current_week_signal(cur):
    """Get the most recent H5 signal with TP/HS levels."""
    cur.execute("""
        SELECT id, signal_date, direction, adjusted_leverage,
               inference_week, inference_year, collapse_flag,
               hard_stop_pct, take_profit_pct, skip_trade,
               confidence_tier
        FROM forecast_h5_signals
        WHERE signal_date <= CURRENT_DATE
        ORDER BY signal_date DESC
        LIMIT 1
    """)
    return cur.fetchone()


def get_execution(cur, signal_date):
    """Get the execution record for a given signal date."""
    cur.execute("""
        SELECT id, status, direction, leverage
        FROM forecast_h5_executions
        WHERE signal_date = %s
    """, (signal_date,))
    return cur.fetchone()


def get_active_subtrade(cur, execution_id):
    """Get the open subtrade for an execution."""
    cur.execute("""
        SELECT id, subtrade_index, direction, entry_price, entry_timestamp,
               bar_count
        FROM forecast_h5_subtrades
        WHERE execution_id = %s AND exit_price IS NULL
        ORDER BY subtrade_index DESC
        LIMIT 1
    """, (execution_id,))
    return cur.fetchone()


def get_latest_bar(cur):
    """Get the most recent USD/COP M5 bar."""
    cur.execute("""
        SELECT time, open, high, low, close
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time DESC
        LIMIT 1
    """)
    return cur.fetchone()


# ---------------------------------------------------------------------------
# TP/HS check (inline to avoid import issues; logic from adaptive_stops.py)
# ---------------------------------------------------------------------------

def check_hard_stop(direction, entry_price, bar_high, bar_low, hs_pct):
    """LONG: bar_low <= entry*(1-hs). SHORT: bar_high >= entry*(1+hs)."""
    if direction == 1:
        return bar_low <= entry_price * (1 - hs_pct)
    else:
        return bar_high >= entry_price * (1 + hs_pct)


def check_take_profit(direction, entry_price, bar_high, bar_low, tp_pct):
    """LONG: bar_high >= entry*(1+tp). SHORT: bar_low <= entry*(1-tp)."""
    if direction == 1:
        return bar_high >= entry_price * (1 + tp_pct)
    else:
        return bar_low <= entry_price * (1 - tp_pct)


def get_exit_price(direction, entry_price, reason, hs_pct, tp_pct, bar_close):
    """Compute exit price: TP/HS are limit-order fills, week_end uses close."""
    if reason == "take_profit":
        return entry_price * (1 + tp_pct) if direction == 1 else entry_price * (1 - tp_pct)
    elif reason == "hard_stop":
        return entry_price * (1 - hs_pct) if direction == 1 else entry_price * (1 + hs_pct)
    else:
        return bar_close


# ---------------------------------------------------------------------------
# Main monitor logic
# ---------------------------------------------------------------------------

def monitor():
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # 1. Get current week's signal
        signal = get_current_week_signal(cur)
        if not signal:
            logger.info("No H5 signal found. Nothing to monitor.")
            return

        (signal_id, signal_date, direction, leverage, week, year,
         collapse, hard_stop_pct, take_profit_pct, skip_trade,
         confidence_tier) = signal

        if skip_trade or collapse:
            logger.info(f"Signal {signal_date} is skip/collapse. Nothing to monitor.")
            return

        if leverage is None:
            logger.info("No leverage set yet (vol targeting hasn't run). Skipping.")
            return

        # 2. Get execution record
        exec_row = get_execution(cur, signal_date)
        if not exec_row:
            logger.info(f"No execution for signal {signal_date}. Nothing to monitor.")
            return

        exec_id, status, exec_dir, exec_lev = exec_row

        if status in ("closed", "paused", "error"):
            logger.info(f"Execution {exec_id} already {status}. Nothing to monitor.")
            return

        # 3. Get active subtrade
        sub = get_active_subtrade(cur, exec_id)
        if not sub:
            logger.info(f"No active subtrade for execution {exec_id}. Already exited by TP/HS.")
            return

        sub_id, sub_idx, sub_dir, entry_price, entry_ts, bar_count = sub

        # 4. Get latest bar
        bar = get_latest_bar(cur)
        if not bar:
            logger.warning("No USD/COP M5 bars in database.")
            return

        bar_time, bar_open, bar_high, bar_low, bar_close = bar

        # Convert Decimal to float for arithmetic
        entry_price = float(entry_price)
        bar_open = float(bar_open)
        bar_high = float(bar_high)
        bar_low = float(bar_low)
        bar_close = float(bar_close)
        hard_stop_pct = float(hard_stop_pct) if hard_stop_pct else 0.0
        take_profit_pct = float(take_profit_pct) if take_profit_pct else 0.0
        leverage = float(leverage) if leverage else 1.0

        logger.info(
            f"Monitoring exec={exec_id}, sub={sub_id}, "
            f"dir={'LONG' if sub_dir == 1 else 'SHORT'}, "
            f"entry={entry_price:.2f}, bar_close={bar_close:.2f}, "
            f"TP={take_profit_pct:.4f}, HS={hard_stop_pct:.4f}"
        )

        # 5. Check hard stop first (worst case takes priority)
        if check_hard_stop(sub_dir, entry_price, bar_high, bar_low, hard_stop_pct):
            exit_price = get_exit_price(sub_dir, entry_price, "hard_stop",
                                        hard_stop_pct, take_profit_pct, bar_close)
            raw_pnl = sub_dir * (exit_price - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'hard_stop',
                    trailing_state = 'triggered', bar_count = %s,
                    pnl_pct = %s, pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, (bar_count or 0) + 1, lev_pnl, raw_pnl, sub_id))

            cur.execute("""
                UPDATE forecast_h5_executions
                SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                    exit_reason = 'hard_stop', week_pnl_pct = %s,
                    week_pnl_unleveraged_pct = %s, n_subtrades = 1,
                    updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, lev_pnl, raw_pnl, exec_id))

            conn.commit()
            logger.info(
                f"HARD STOP TRIGGERED: exit={exit_price:.2f}, "
                f"pnl={lev_pnl:+.4f}% (unlev={raw_pnl:+.4f}%)"
            )
            return

        # 6. Check take profit
        if check_take_profit(sub_dir, entry_price, bar_high, bar_low, take_profit_pct):
            exit_price = get_exit_price(sub_dir, entry_price, "take_profit",
                                        hard_stop_pct, take_profit_pct, bar_close)
            raw_pnl = sub_dir * (exit_price - entry_price) / entry_price
            lev_pnl = raw_pnl * leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'take_profit',
                    trailing_state = 'triggered', bar_count = %s,
                    pnl_pct = %s, pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, (bar_count or 0) + 1, lev_pnl, raw_pnl, sub_id))

            cur.execute("""
                UPDATE forecast_h5_executions
                SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                    exit_reason = 'take_profit', week_pnl_pct = %s,
                    week_pnl_unleveraged_pct = %s, n_subtrades = 1,
                    updated_at = NOW()
                WHERE id = %s
            """, (exit_price, bar_time, lev_pnl, raw_pnl, exec_id))

            conn.commit()
            logger.info(
                f"TAKE PROFIT TRIGGERED: exit={exit_price:.2f}, "
                f"pnl={lev_pnl:+.4f}% (unlev={raw_pnl:+.4f}%)"
            )
            return

        # 7. Still monitoring — log distance to TP/HS
        if sub_dir == 1:  # LONG
            dist_tp = (bar_close / entry_price - 1) / take_profit_pct * 100 if take_profit_pct else 0
            dist_hs = (1 - bar_close / entry_price) / hard_stop_pct * 100 if hard_stop_pct else 0
        else:  # SHORT
            dist_tp = (1 - bar_close / entry_price) / take_profit_pct * 100 if take_profit_pct else 0
            dist_hs = (bar_close / entry_price - 1) / hard_stop_pct * 100 if hard_stop_pct else 0

        # Update bar count
        cur.execute("""
            UPDATE forecast_h5_subtrades
            SET bar_count = %s, updated_at = NOW()
            WHERE id = %s
        """, ((bar_count or 0) + 1, sub_id))
        conn.commit()

        logger.info(
            f"MONITORING: bar={bar_time}, close={bar_close:.2f}, "
            f"entry={entry_price:.2f}, dist_to_TP={dist_tp:.1f}%, "
            f"dist_to_HS={dist_hs:.1f}%, bars={bar_count or 0 + 1}"
        )

    except Exception as e:
        logger.error(f"Monitor failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    now_cot = datetime.now(COT_TZ)
    logger.info(f"Position monitor starting at {now_cot.strftime('%Y-%m-%d %H:%M COT')}")

    # Skip weekends
    if now_cot.weekday() >= 5:
        logger.info("Weekend. Nothing to monitor.")
        return

    monitor()
    logger.info("Position monitor complete")


if __name__ == "__main__":
    main()
