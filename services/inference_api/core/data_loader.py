"""
Data Loader for OHLCV and Macro data from PostgreSQL
"""

import asyncio
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import asyncpg
import pandas as pd
import numpy as np
from ..config import get_settings

settings = get_settings()

# Colombian holidays 2025-2026 (exclude from trading data)
# Must match the chart API to ensure consistency
COLOMBIA_HOLIDAYS = [
    '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
    '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
    '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
    '2025-12-08', '2025-12-25',
    '2026-01-01', '2026-01-12', '2026-03-23', '2026-04-02', '2026-04-03',
    '2026-05-01', '2026-05-18', '2026-06-08', '2026-06-15', '2026-06-29',
    '2026-07-20', '2026-08-07', '2026-08-17', '2026-10-12', '2026-11-02',
    '2026-11-16', '2026-12-08', '2026-12-25',
]

# Market hours (UTC times for old format data)
MARKET_OPEN_UTC = '13:00:00'
MARKET_CLOSE_UTC = '17:55:00'
# COT times for new format data
MARKET_OPEN_COT = '08:00:00'
MARKET_CLOSE_COT = '12:55:00'


class DataLoader:
    """
    Loads OHLCV and macro indicator data from PostgreSQL.
    Designed for async operation with connection pooling.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                min_size=2,
                max_size=10,
            )
        return self._pool

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def load_ohlcv(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "USD/COP"
    ) -> pd.DataFrame:
        """
        Load OHLCV data from usdcop_m5_ohlcv table.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Trading symbol (default USD/COP)

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        pool = await self._get_pool()

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Build holidays list for SQL
        holidays_sql = ", ".join([f"'{h}'" for h in COLOMBIA_HOLIDAYS])

        query = f"""
            SELECT
                time,
                open,
                high,
                low,
                close,
                COALESCE(volume, 0) as volume
            FROM usdcop_m5_ohlcv
            WHERE time >= $1
              AND time < ($2::timestamp + INTERVAL '1 day')
              AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
              AND DATE(time)::text NOT IN ({holidays_sql})
              AND (
                (time::time >= '{MARKET_OPEN_UTC}'::time AND time::time <= '{MARKET_CLOSE_UTC}'::time)
                OR
                (time::time >= '{MARKET_OPEN_COT}'::time AND time::time <= '{MARKET_CLOSE_COT}'::time)
              )
            ORDER BY time ASC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, start, end)

        if not rows:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame([dict(row) for row in rows])
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    async def load_macro(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load macro indicators from macro_indicators_daily table.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with macro indicator columns
        """
        pool = await self._get_pool()

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Query with correct column names from the table
        query = """
            SELECT
                fecha as date,
                fxrt_index_dxy_usa_d_dxy as dxy,
                volt_vix_usa_d_vix as vix,
                crsk_spread_embi_col_d_embi as embi,
                comm_oil_brent_glb_d_brent as brent,
                finc_bond_yield10y_usa_d_ust10y as treasury_10y,
                fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn
            FROM macro_indicators_daily
            WHERE fecha >= $1
              AND fecha <= $2
            ORDER BY fecha ASC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, start, end)

        if not rows:
            return pd.DataFrame(columns=["date", "dxy", "vix", "embi", "brent", "treasury_10y", "usdmxn"])

        df = pd.DataFrame([dict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"])

        # Convert numeric columns
        for col in ["dxy", "vix", "embi", "brent", "treasury_10y", "usdmxn"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    async def load_combined_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load and merge OHLCV with macro data.
        Macro data is forward-filled to 5-minute bars.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV and macro columns merged
        """
        # Load both datasets in parallel
        ohlcv_task = self.load_ohlcv(start_date, end_date)
        macro_task = self.load_macro(start_date, end_date)

        ohlcv_df, macro_df = await asyncio.gather(ohlcv_task, macro_task)

        if ohlcv_df.empty:
            raise ValueError(f"No OHLCV data found for {start_date} to {end_date}")

        # Add date column to OHLCV for merging
        ohlcv_df["date"] = ohlcv_df["time"].dt.date
        ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"])

        # Merge with macro data
        if not macro_df.empty:
            combined = ohlcv_df.merge(
                macro_df,
                on="date",
                how="left"
            )
            # Forward fill macro values (they're daily)
            macro_cols = ["dxy", "vix", "embi", "brent", "treasury_10y", "usdmxn"]
            for col in macro_cols:
                if col in combined.columns:
                    combined[col] = combined[col].ffill()
        else:
            combined = ohlcv_df
            # Add empty macro columns
            for col in ["dxy", "vix", "embi", "brent", "treasury_10y", "usdmxn"]:
                combined[col] = np.nan

        return combined

    async def get_data_stats(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Get statistics about available data for a date range.

        Returns:
            Dict with counts and date ranges
        """
        pool = await self._get_pool()

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        async with pool.acquire() as conn:
            # OHLCV stats
            ohlcv_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as bar_count,
                    MIN(time) as first_bar,
                    MAX(time) as last_bar
                FROM usdcop_m5_ohlcv
                WHERE time >= $1 AND time < ($2::timestamp + INTERVAL '1 day')
            """, start, end)

            # Macro stats
            macro_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as day_count,
                    MIN(fecha) as first_day,
                    MAX(fecha) as last_day
                FROM macro_indicators_daily
                WHERE fecha >= $1 AND fecha <= $2
            """, start, end)

        return {
            "ohlcv": {
                "bar_count": ohlcv_stats["bar_count"],
                "first_bar": str(ohlcv_stats["first_bar"]) if ohlcv_stats["first_bar"] else None,
                "last_bar": str(ohlcv_stats["last_bar"]) if ohlcv_stats["last_bar"] else None,
            },
            "macro": {
                "day_count": macro_stats["day_count"],
                "first_day": str(macro_stats["first_day"]) if macro_stats["first_day"] else None,
                "last_day": str(macro_stats["last_day"]) if macro_stats["last_day"] else None,
            }
        }
