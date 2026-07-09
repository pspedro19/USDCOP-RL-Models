#!/usr/bin/env python3
"""
cron_week_end.py — Close open H5 position at Friday market close
================================================================
Cron replacement for Airflow DAG: forecast_h5_l7_multiday_executor (close_week)

Closes any open position at the latest M5 bar price.
Updates forecast_h5_subtrades (exit_reason='week_end') and
forecast_h5_executions (status='closed').

Schedule: Friday 12:50 COT (= 17:50 UTC)

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
logger = logging.getLogger("cron_week_end")

COT_TZ = pytz.timezone("America/Bogota")

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
# Close week logic (mirrored from forecast_h5_l7_multiday_executor.close_week)
# ---------------------------------------------------------------------------

def close_week():
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # 1. Get current week's signal
        cur.execute("""
            SELECT id, signal_date, direction, adjusted_leverage,
                   hard_stop_pct, take_profit_pct
            FROM forecast_h5_signals
            WHERE signal_date <= CURRENT_DATE
            ORDER BY signal_date DESC
            LIMIT 1
        """)
        signal = cur.fetchone()
        if not signal:
            logger.info("No signal found. Nothing to close.")
            return {"action": "no_signal"}

        signal_id, signal_date, direction, leverage, hs_pct, tp_pct = signal

        # 2. Get execution record
        cur.execute("""
            SELECT id, status, direction, leverage
            FROM forecast_h5_executions
            WHERE signal_date = %s
        """, (signal_date,))
        exec_row = cur.fetchone()

        if not exec_row:
            logger.info(f"No execution for signal {signal_date}. Nothing to close.")
            return {"action": "no_execution"}

        exec_id, status, exec_dir, exec_lev = exec_row

        if status in ("closed", "paused", "error"):
            logger.info(f"Execution {exec_id} already {status}.")
            return {"action": "already_closed"}

        # Use exec leverage if signal leverage is None
        use_leverage = exec_lev or leverage or 1.0

        # 3. Get latest bar for exit price
        cur.execute("""
            SELECT time, open, high, low, close
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            ORDER BY time DESC
            LIMIT 1
        """)
        bar = cur.fetchone()
        if not bar:
            logger.error("No USD/COP M5 bars in database. Cannot close.")
            return {"action": "no_bar"}

        bar_time, _, _, _, bar_close = bar

        # 4. Close active subtrade if any
        cur.execute("""
            SELECT id, direction, entry_price, bar_count
            FROM forecast_h5_subtrades
            WHERE execution_id = %s AND exit_price IS NULL
            ORDER BY subtrade_index DESC
            LIMIT 1
        """, (exec_id,))
        active_sub = cur.fetchone()

        if active_sub:
            sub_id, sub_dir, entry_price, bar_count = active_sub
            raw_pnl = sub_dir * (bar_close - entry_price) / entry_price
            lev_pnl = raw_pnl * use_leverage

            cur.execute("""
                UPDATE forecast_h5_subtrades
                SET exit_price = %s, exit_timestamp = %s, exit_reason = 'week_end',
                    trailing_state = 'expired', pnl_pct = %s,
                    pnl_unleveraged_pct = %s, updated_at = NOW()
                WHERE id = %s
            """, (bar_close, bar_time, lev_pnl, raw_pnl, sub_id))

            logger.info(
                f"Closed subtrade {sub_id}: entry={entry_price:.2f}, "
                f"exit={bar_close:.2f}, pnl={lev_pnl:+.4f}%"
            )

        # 5. Aggregate week PnL from all subtrades
        cur.execute("""
            SELECT COALESCE(SUM(pnl_pct), 0),
                   COALESCE(SUM(pnl_unleveraged_pct), 0),
                   COUNT(*)
            FROM forecast_h5_subtrades
            WHERE execution_id = %s AND exit_price IS NOT NULL
        """, (exec_id,))
        total_pnl, total_unlev, n_subs = cur.fetchone()

        # 6. Close the execution
        cur.execute("""
            UPDATE forecast_h5_executions
            SET status = 'closed', exit_price = %s, exit_timestamp = %s,
                exit_reason = 'week_end', week_pnl_pct = %s,
                week_pnl_unleveraged_pct = %s, n_subtrades = %s,
                updated_at = NOW()
            WHERE id = %s
        """, (bar_close, bar_time, total_pnl, total_unlev, n_subs, exec_id))

        conn.commit()

        logger.info(
            f"WEEK CLOSED: exec_id={exec_id}, signal_date={signal_date}, "
            f"exit_price={bar_close:.2f}, week_pnl={total_pnl:+.4f}%, "
            f"n_subtrades={n_subs}"
        )

        return {
            "action": "closed",
            "exec_id": exec_id,
            "week_pnl_pct": float(total_pnl),
            "exit_price": float(bar_close),
        }

    except Exception as e:
        conn.rollback()
        logger.error(f"Week-end close failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    now_cot = datetime.now(COT_TZ)
    logger.info(f"Week-end close starting at {now_cot.strftime('%Y-%m-%d %H:%M COT')}")

    # Safety: only run on Friday
    if now_cot.weekday() != 4:
        logger.warning(f"Today is {now_cot.strftime('%A')}, not Friday. Aborting.")
        return

    result = close_week()
    logger.info(f"Result: {result}")
    logger.info("Week-end close complete")


if __name__ == "__main__":
    main()
