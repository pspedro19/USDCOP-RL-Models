"""
Order Flow Endpoint Implementation
To be added to trading_analytics_api.py after line 967
"""


@app.get("/api/analytics/order-flow")
async def get_order_flow(
    symbol: str = "USDCOP",
    window: int = Query(60, description="Time window in seconds")
):
    """
    Get order flow metrics (buy/sell volume imbalance)

    Analyzes bid/ask volume over the specified time window
    to determine market pressure (buying vs selling).

    Returns:
        - buy_volume: Total volume on buy side
        - sell_volume: Total volume on sell side
        - buy_percent: Percentage of buy volume
        - sell_percent: Percentage of sell volume
        - imbalance: buy_percent - sell_percent
    """
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query recent candles from L0 database
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=window)

        query = """
        SELECT
            datetime,
            open,
            high,
            low,
            close,
            volume
        FROM market_data
        WHERE symbol = %s
          AND datetime >= %s
          AND datetime <= %s
          AND volume > 0
        ORDER BY datetime DESC
        """

        cursor.execute(query, (symbol, start_time, end_time))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            logger.warning(f"No order flow data found for {symbol} in last {window}s")
            return {
                "symbol": symbol,
                "timestamp": end_time.isoformat(),
                "window_seconds": window,
                "order_flow": {
                    "buy_volume": 0,
                    "sell_volume": 0,
                    "buy_percent": 50.0,
                    "sell_percent": 50.0,
                    "imbalance": 0.0
                },
                "data_available": False
            }

        # Calculate order flow using price action analysis
        # Buy pressure: When close > open (bullish candle)
        # Sell pressure: When close < open (bearish candle)

        buy_volume = 0
        sell_volume = 0

        for row in rows:
            vol = float(row['volume'])
            close = float(row['close'])
            open_price = float(row['open'])

            # Classify volume as buy or sell based on price action
            if close > open_price:
                # Bullish candle - buying pressure
                buy_volume += vol
            elif close < open_price:
                # Bearish candle - selling pressure
                sell_volume += vol
            else:
                # Neutral candle - split volume
                buy_volume += vol / 2
                sell_volume += vol / 2

        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            buy_pct = 50.0
            sell_pct = 50.0
            imbalance = 0.0
        else:
            buy_pct = (buy_volume / total_volume) * 100
            sell_pct = (sell_volume / total_volume) * 100
            imbalance = buy_pct - sell_pct

        return {
            "symbol": symbol,
            "timestamp": end_time.isoformat(),
            "window_seconds": window,
            "order_flow": {
                "buy_volume": round(buy_volume, 2),
                "sell_volume": round(sell_volume, 2),
                "buy_percent": round(buy_pct, 1),
                "sell_percent": round(sell_pct, 1),
                "imbalance": round(imbalance, 1)
            },
            "data_available": True,
            "candles_analyzed": len(rows)
        }

    except Exception as e:
        logger.error(f"Error calculating order flow: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to calculate order flow", "details": str(e)}
        )
