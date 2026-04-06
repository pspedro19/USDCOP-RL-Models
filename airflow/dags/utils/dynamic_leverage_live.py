"""
Dynamic Leverage for Live Airflow DAGs.

Scales leverage based on rolling win rate and drawdown.
Reads recent execution history from forecast_h5_executions.

Mirrors src/forecasting/dynamic_leverage.py for DAG context.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_leverage_scaler(rolling_wr, current_dd_pct,
                            wr_full=60.0, wr_half=40.0, wr_pause=30.0,
                            dd_threshold=6.0):
    """
    Compute leverage scale factor [0.25, 1.0] from rolling WR and drawdown.
    """
    # WR-based scaling
    if rolling_wr >= wr_full:
        wr_factor = 1.0
    elif rolling_wr >= wr_half:
        wr_factor = 0.5 + 0.5 * (rolling_wr - wr_half) / max(wr_full - wr_half, 1)
    elif rolling_wr >= wr_pause:
        wr_factor = 0.25 + 0.25 * (rolling_wr - wr_pause) / max(wr_half - wr_pause, 1)
    else:
        wr_factor = 0.25

    # Drawdown penalty
    dd_factor = 0.5 if current_dd_pct > dd_threshold else 1.0

    return max(0.25, min(1.0, wr_factor * dd_factor))


def compute_dynamic_leverage_from_db(conn, lookback_weeks=8):
    """
    Compute dynamic leverage adjustment from DB execution history.

    Args:
        conn: psycopg2 connection
        lookback_weeks: Rolling window for WR calculation

    Returns:
        dict with rolling_wr, current_dd_pct, leverage_scaler
    """
    try:
        cur = conn.cursor()

        # Get recent closed executions
        cur.execute("""
            SELECT week_pnl_pct
            FROM forecast_h5_executions
            WHERE status = 'closed' AND week_pnl_pct IS NOT NULL
            ORDER BY signal_date DESC
            LIMIT %s
        """, (lookback_weeks,))

        rows = cur.fetchall()
        cur.close()

        if not rows or len(rows) < 3:
            logger.info(f"Dynamic leverage: insufficient history ({len(rows)} trades), using 1.0x")
            return {"rolling_wr": 100.0, "current_dd_pct": 0.0, "leverage_scaler": 1.0}

        pnls = [float(r[0]) for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        rolling_wr = (wins / len(pnls)) * 100

        # Compute current drawdown from equity curve
        equity = 10000.0
        peak = 10000.0
        for p in reversed(pnls):  # Oldest first
            equity *= (1 + p)
            peak = max(peak, equity)
        current_dd = (1 - equity / peak) * 100 if peak > 0 else 0

        scaler = compute_leverage_scaler(rolling_wr, current_dd)

        logger.info(f"Dynamic leverage: WR={rolling_wr:.1f}%, DD={current_dd:.1f}%, scaler={scaler:.2f}")
        return {
            "rolling_wr": round(rolling_wr, 1),
            "current_dd_pct": round(current_dd, 2),
            "leverage_scaler": round(scaler, 3),
        }

    except Exception as e:
        logger.error(f"Dynamic leverage computation failed: {e}")
        return {"rolling_wr": 100.0, "current_dd_pct": 0.0, "leverage_scaler": 1.0}
