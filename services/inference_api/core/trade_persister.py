"""
Trade Persister
Saves and retrieves trades from PostgreSQL
"""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import asyncpg
import logging
from ..config import get_settings
from .trade_simulator import Trade

settings = get_settings()
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD string to date object"""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


class TradePersister:
    """
    Handles persistence of trades to PostgreSQL trades_history table.
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

    async def get_trades(
        self,
        start_date: str,
        end_date: str,
        model_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get existing trades for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_id: Model ID to filter by

        Returns:
            List of trade dictionaries
        """
        pool = await self._get_pool()

        query = """
            SELECT
                id as trade_id,
                model_id,
                side,
                entry_price,
                exit_price,
                entry_time,
                exit_time,
                duration_bars,
                pnl_usd,
                pnl_pct,
                exit_reason,
                equity_at_entry,
                equity_at_exit
            FROM trades_history
            WHERE model_id = $1
              AND entry_time >= $2
              AND entry_time < ($3::timestamp + INTERVAL '1 day')
            ORDER BY entry_time ASC
        """

        start = parse_date(start_date)
        end = parse_date(end_date)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, model_id, start, end)

        trades = []
        for row in rows:
            trades.append({
                "trade_id": row["trade_id"],
                "model_id": row["model_id"],
                "timestamp": row["entry_time"].isoformat() if row["entry_time"] else None,
                "entry_time": row["entry_time"].isoformat() if row["entry_time"] else None,
                "exit_time": row["exit_time"].isoformat() if row["exit_time"] else None,
                "side": row["side"].lower() if row["side"] else "long",
                "entry_price": float(row["entry_price"]) if row["entry_price"] else 0,
                "exit_price": float(row["exit_price"]) if row["exit_price"] else None,
                "pnl": float(row["pnl_usd"]) if row["pnl_usd"] else 0,
                "pnl_usd": float(row["pnl_usd"]) if row["pnl_usd"] else 0,
                "pnl_percent": float(row["pnl_pct"]) if row["pnl_pct"] else 0,
                "pnl_pct": float(row["pnl_pct"]) if row["pnl_pct"] else 0,
                "status": "closed" if row["exit_time"] else "open",
                "duration_minutes": (row["duration_bars"] or 0) * 5,
                "exit_reason": row["exit_reason"],
                "equity_at_entry": float(row["equity_at_entry"]) if row["equity_at_entry"] else None,
                "equity_at_exit": float(row["equity_at_exit"]) if row["equity_at_exit"] else None,
                "entry_confidence": None,
                "exit_confidence": None,
            })

        return trades

    async def trades_exist(
        self,
        start_date: str,
        end_date: str,
        model_id: str,
        min_trades: int = 1
    ) -> bool:
        """
        Check if trades exist for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_id: Model ID
            min_trades: Minimum number of trades required

        Returns:
            True if sufficient trades exist
        """
        pool = await self._get_pool()

        query = """
            SELECT COUNT(*) as count
            FROM trades_history
            WHERE model_id = $1
              AND entry_time >= $2
              AND entry_time < ($3::timestamp + INTERVAL '1 day')
        """

        start = parse_date(start_date)
        end = parse_date(end_date)

        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, model_id, start, end)

        return result["count"] >= min_trades

    async def save_trades(
        self,
        trades: List[Trade],
        batch_size: int = 100
    ) -> int:
        """
        Save trades to database.

        Args:
            trades: List of Trade objects
            batch_size: Number of trades to insert per batch

        Returns:
            Number of trades saved
        """
        if not trades:
            return 0

        pool = await self._get_pool()

        insert_query = """
            INSERT INTO trades_history (
                model_id, side, entry_price, exit_price,
                entry_time, exit_time, duration_bars,
                pnl_usd, pnl_pct, exit_reason,
                equity_at_entry, equity_at_exit
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            )
            ON CONFLICT DO NOTHING
        """

        saved_count = 0

        async with pool.acquire() as conn:
            # Process in batches
            for i in range(0, len(trades), batch_size):
                batch = trades[i:i + batch_size]

                # Prepare batch data
                batch_data = []
                for trade in batch:
                    batch_data.append((
                        trade.model_id,
                        trade.side.upper(),
                        trade.entry_price,
                        trade.exit_price,
                        trade.entry_time,
                        trade.exit_time,
                        trade.duration_bars,
                        trade.pnl_usd,
                        trade.pnl_pct,
                        trade.exit_reason,
                        trade.equity_at_entry,
                        trade.equity_at_exit,
                    ))

                # Execute batch insert
                await conn.executemany(insert_query, batch_data)
                saved_count += len(batch)

                logger.info(f"Saved batch of {len(batch)} trades ({saved_count}/{len(trades)})")

        return saved_count

    async def delete_trades(
        self,
        start_date: str,
        end_date: str,
        model_id: str
    ) -> int:
        """
        Delete trades for a date range (for regeneration).

        Args:
            start_date: Start date
            end_date: End date
            model_id: Model ID

        Returns:
            Number of trades deleted
        """
        pool = await self._get_pool()

        query = """
            DELETE FROM trades_history
            WHERE model_id = $1
              AND entry_time >= $2
              AND entry_time < ($3::timestamp + INTERVAL '1 day')
        """

        start = parse_date(start_date)
        end = parse_date(end_date)

        async with pool.acquire() as conn:
            result = await conn.execute(query, model_id, start, end)

        # Parse "DELETE N" result
        deleted = int(result.split()[-1]) if result else 0
        logger.info(f"Deleted {deleted} trades for {model_id} from {start_date} to {end_date}")

        return deleted

    async def get_trade_count(
        self,
        start_date: str,
        end_date: str,
        model_id: str
    ) -> int:
        """Get count of trades for a date range"""
        pool = await self._get_pool()

        query = """
            SELECT COUNT(*) as count
            FROM trades_history
            WHERE model_id = $1
              AND entry_time >= $2
              AND entry_time < ($3::timestamp + INTERVAL '1 day')
        """

        start = parse_date(start_date)
        end = parse_date(end_date)

        async with pool.acquire() as conn:
            result = await conn.fetchrow(query, model_id, start, end)

        return result["count"]
