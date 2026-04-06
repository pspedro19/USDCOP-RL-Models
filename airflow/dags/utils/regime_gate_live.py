"""
Regime Gate for Live Airflow DAGs.

Computes Hurst exponent from DB daily returns and classifies
the current market regime. Used by L5 vol-targeting to decide
whether to trade and at what sizing.

Mirrors src/forecasting/regime_gate.py logic for DAG context.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_hurst_rs(returns, max_k=20):
    """Hurst exponent via rescaled range (R/S) method."""
    N = len(returns)
    if N < 30:
        return 0.5

    RS = []
    for k in range(10, min(max_k + 1, N // 2)):
        rs_vals = []
        for start in range(0, N - k, k):
            chunk = returns[start:start + k]
            mean_c = np.mean(chunk)
            Y = np.cumsum(chunk - mean_c)
            R = np.max(Y) - np.min(Y)
            S = np.std(chunk, ddof=1)
            if S > 1e-10:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append((np.log(k), np.log(np.mean(rs_vals))))

    if len(RS) < 3:
        return 0.5

    x = np.array([r[0] for r in RS])
    y = np.array([r[1] for r in RS])
    H = float(np.polyfit(x, y, 1)[0])
    return max(0.0, min(1.0, H))


def classify_regime(hurst, hurst_trending=0.52, hurst_mean_rev=0.42):
    """Classify regime from Hurst exponent."""
    if hurst > hurst_trending:
        return "trending"
    elif hurst < hurst_mean_rev:
        return "mean_reverting"
    return "indeterminate"


def get_regime_sizing(regime, sizing_trending=1.0, sizing_indeterminate=0.40, sizing_mean_rev=0.0):
    """Get sizing multiplier for the current regime."""
    if regime == "trending":
        return sizing_trending
    elif regime == "mean_reverting":
        return sizing_mean_rev
    return sizing_indeterminate


def compute_regime_from_db(conn, lookback_days=60, symbol="USD/COP"):
    """
    Full regime computation from DB daily returns.

    Args:
        conn: psycopg2 connection
        lookback_days: Number of trading days for Hurst calculation
        symbol: FX pair symbol

    Returns:
        dict with hurst, regime, sizing_factor
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT close FROM (
                SELECT DISTINCT ON (time::date) time, close
                FROM usdcop_m5_ohlcv
                WHERE symbol = %s
                ORDER BY time::date DESC, time DESC
                LIMIT %s
            ) sub ORDER BY time ASC
        """, (symbol, lookback_days + 5))

        rows = cur.fetchall()
        cur.close()

        if len(rows) < 30:
            logger.warning(f"Insufficient data for Hurst: {len(rows)} rows")
            return {"hurst": 0.5, "regime": "indeterminate", "sizing_factor": 0.40}

        closes = np.array([float(r[0]) for r in rows])
        returns = np.diff(closes) / closes[:-1]

        hurst = compute_hurst_rs(returns[-lookback_days:])
        regime = classify_regime(hurst)
        sizing = get_regime_sizing(regime)

        logger.info(f"Regime gate: Hurst={hurst:.3f}, regime={regime}, sizing={sizing:.2f}")
        return {"hurst": round(hurst, 4), "regime": regime, "sizing_factor": sizing}

    except Exception as e:
        logger.error(f"Regime computation failed: {e}")
        return {"hurst": 0.5, "regime": "indeterminate", "sizing_factor": 0.40}
