"""
Trades Router - Trade history and performance endpoints

Provides endpoints for:
- Listing trades with pagination and filtering
- Trading performance summary
- Most recent trade retrieval
"""

from datetime import datetime, timezone
from typing import List, Optional
import logging

from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel, Field
import asyncpg

from ..config import get_settings
from ..models.responses import TradeResponse

router = APIRouter(tags=["trades"])
logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# Pydantic Models
# =============================================================================


class TradesSummary(BaseModel):
    """Trading performance summary"""

    total_trades: int = Field(description="Total number of trades")
    winning_trades: int = Field(description="Number of profitable trades")
    losing_trades: int = Field(description="Number of losing trades")
    win_rate: float = Field(description="Win rate as percentage (0-100)")
    total_pnl: float = Field(description="Total profit/loss in USD")
    total_pnl_percent: float = Field(description="Total P&L as percentage of initial capital")
    avg_pnl: float = Field(description="Average P&L per trade")
    avg_duration_minutes: Optional[float] = Field(
        default=None, description="Average trade duration in minutes"
    )
    max_drawdown_percent: float = Field(
        default=0.0, description="Maximum drawdown as percentage"
    )
    sharpe_ratio: Optional[float] = Field(
        default=None, description="Sharpe ratio (if sufficient trades)"
    )
    best_trade_pnl: float = Field(default=0.0, description="Best trade P&L")
    worst_trade_pnl: float = Field(default=0.0, description="Worst trade P&L")
    period_start: Optional[str] = Field(default=None, description="Start of period")
    period_end: Optional[str] = Field(default=None, description="End of period")


class TradesListResponse(BaseModel):
    """Response for trades list endpoint"""

    trades: List[TradeResponse]
    total: int = Field(description="Total number of trades matching filters")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of trades per page")
    has_more: bool = Field(description="Whether more pages are available")


class LatestTradeResponse(BaseModel):
    """Response for latest trade endpoint"""

    trade: Optional[TradeResponse] = None
    message: str = "Success"


# =============================================================================
# Database Helpers
# =============================================================================


async def get_db_connection():
    """Get database connection."""
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable"
        )


def row_to_trade_response(row: asyncpg.Record) -> TradeResponse:
    """Convert database row to TradeResponse."""
    return TradeResponse(
        trade_id=row["id"],
        model_id=row.get("model_id", "ppo_primary"),
        timestamp=row["entry_time"].isoformat() if row["entry_time"] else "",
        entry_time=row["entry_time"].isoformat() if row["entry_time"] else "",
        exit_time=row["exit_time"].isoformat() if row.get("exit_time") else None,
        side=row.get("side", "long"),
        entry_price=float(row.get("entry_price", 0)),
        exit_price=float(row["exit_price"]) if row.get("exit_price") else None,
        pnl=float(row.get("pnl", 0)),
        pnl_usd=float(row.get("pnl_usd", row.get("pnl", 0))),
        pnl_percent=float(row.get("pnl_percent", row.get("pnl_pct", 0))),
        pnl_pct=float(row.get("pnl_pct", row.get("pnl_percent", 0))),
        status=row.get("status", "closed"),
        duration_minutes=row.get("duration_minutes"),
        exit_reason=row.get("exit_reason"),
        equity_at_entry=float(row["equity_at_entry"]) if row.get("equity_at_entry") else None,
        equity_at_exit=float(row["equity_at_exit"]) if row.get("equity_at_exit") else None,
        entry_confidence=float(row["entry_confidence"]) if row.get("entry_confidence") else None,
        exit_confidence=float(row["exit_confidence"]) if row.get("exit_confidence") else None,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/trades", response_model=TradesListResponse)
async def list_trades(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Trades per page"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    side: Optional[str] = Query(None, description="Filter by side (long/short)"),
    status: Optional[str] = Query(None, description="Filter by status (open/closed)"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    min_pnl: Optional[float] = Query(None, description="Minimum P&L filter"),
    max_pnl: Optional[float] = Query(None, description="Maximum P&L filter"),
    sort_by: str = Query("entry_time", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
):
    """
    List trades with pagination and filtering.

    Supports filtering by:
    - model_id: Filter by specific model
    - side: long or short
    - status: open or closed
    - start_date/end_date: Date range
    - min_pnl/max_pnl: P&L range

    Returns paginated results with total count.
    """
    conn = await get_db_connection()

    try:
        # Build query with filters
        conditions = []
        params = []
        param_count = 0

        if model_id:
            param_count += 1
            conditions.append(f"model_id = ${param_count}")
            params.append(model_id)

        if side:
            param_count += 1
            conditions.append(f"side = ${param_count}")
            params.append(side.lower())

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.lower())

        if start_date:
            param_count += 1
            conditions.append(f"entry_time >= ${param_count}")
            params.append(datetime.fromisoformat(start_date.replace("Z", "+00:00")))

        if end_date:
            param_count += 1
            conditions.append(f"entry_time <= ${param_count}")
            params.append(datetime.fromisoformat(end_date.replace("Z", "+00:00")))

        if min_pnl is not None:
            param_count += 1
            conditions.append(f"pnl >= ${param_count}")
            params.append(min_pnl)

        if max_pnl is not None:
            param_count += 1
            conditions.append(f"pnl <= ${param_count}")
            params.append(max_pnl)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate sort_by to prevent SQL injection
        allowed_sort_fields = {
            "entry_time", "exit_time", "pnl", "pnl_percent",
            "side", "status", "entry_price", "exit_price"
        }
        if sort_by not in allowed_sort_fields:
            sort_by = "entry_time"

        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM trades WHERE {where_clause}"
        total = await conn.fetchval(count_query, *params)

        # Get paginated trades
        offset = (page - 1) * page_size
        trades_query = f"""
            SELECT *
            FROM trades
            WHERE {where_clause}
            ORDER BY {sort_by} {sort_direction}
            LIMIT {page_size}
            OFFSET {offset}
        """

        rows = await conn.fetch(trades_query, *params)
        trades = [row_to_trade_response(row) for row in rows]

        has_more = (page * page_size) < total

        return TradesListResponse(
            trades=trades,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list trades: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list trades: {str(e)}")
    finally:
        await conn.close()


@router.get("/trades/summary", response_model=TradesSummary)
async def get_trades_summary(
    request: Request,
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
):
    """
    Get trading performance summary.

    Returns aggregated statistics including:
    - Total/winning/losing trades
    - Win rate
    - Total and average P&L
    - Sharpe ratio
    - Max drawdown
    - Best/worst trades
    """
    conn = await get_db_connection()

    try:
        # Build filter conditions
        conditions = []
        params = []
        param_count = 0

        if model_id:
            param_count += 1
            conditions.append(f"model_id = ${param_count}")
            params.append(model_id)

        if start_date:
            param_count += 1
            conditions.append(f"entry_time >= ${param_count}")
            params.append(datetime.fromisoformat(start_date.replace("Z", "+00:00")))

        if end_date:
            param_count += 1
            conditions.append(f"entry_time <= ${param_count}")
            params.append(datetime.fromisoformat(end_date.replace("Z", "+00:00")))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Query for summary statistics
        summary_query = f"""
            SELECT
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
                COUNT(*) FILTER (WHERE pnl <= 0) as losing_trades,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(AVG(pnl), 0) as avg_pnl,
                COALESCE(AVG(duration_minutes), 0) as avg_duration_minutes,
                COALESCE(MAX(pnl), 0) as best_trade_pnl,
                COALESCE(MIN(pnl), 0) as worst_trade_pnl,
                COALESCE(STDDEV(pnl), 0) as pnl_stddev,
                MIN(entry_time) as period_start,
                MAX(entry_time) as period_end
            FROM trades
            WHERE {where_clause}
        """

        row = await conn.fetchrow(summary_query, *params)

        total_trades = row["total_trades"] or 0
        winning_trades = row["winning_trades"] or 0
        losing_trades = row["losing_trades"] or 0
        total_pnl = float(row["total_pnl"] or 0)
        avg_pnl = float(row["avg_pnl"] or 0)
        avg_duration = float(row["avg_duration_minutes"] or 0) if row["avg_duration_minutes"] else None
        best_trade = float(row["best_trade_pnl"] or 0)
        worst_trade = float(row["worst_trade_pnl"] or 0)
        pnl_stddev = float(row["pnl_stddev"] or 0)

        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate Sharpe ratio (annualized, assuming daily trades)
        sharpe_ratio = None
        if total_trades >= 30 and pnl_stddev > 0:
            # Annualized Sharpe = (avg_return / std_return) * sqrt(252)
            sharpe_ratio = (avg_pnl / pnl_stddev) * (252 ** 0.5)

        # Calculate max drawdown (simplified - would need equity curve for accurate calc)
        max_drawdown_query = f"""
            SELECT
                COALESCE(MIN(
                    (equity_at_exit - MAX(equity_at_entry) OVER (ORDER BY entry_time ROWS UNBOUNDED PRECEDING))
                    / NULLIF(MAX(equity_at_entry) OVER (ORDER BY entry_time ROWS UNBOUNDED PRECEDING), 0) * 100
                ), 0) as max_drawdown
            FROM trades
            WHERE {where_clause} AND equity_at_entry IS NOT NULL
        """

        try:
            dd_row = await conn.fetchrow(max_drawdown_query, *params)
            max_drawdown = abs(float(dd_row["max_drawdown"] or 0)) if dd_row else 0.0
        except Exception:
            max_drawdown = 0.0

        # Calculate total P&L percent (assume initial capital from first trade)
        total_pnl_percent = 0.0
        if total_trades > 0:
            initial_equity_query = f"""
                SELECT equity_at_entry
                FROM trades
                WHERE {where_clause} AND equity_at_entry IS NOT NULL
                ORDER BY entry_time ASC
                LIMIT 1
            """
            eq_row = await conn.fetchrow(initial_equity_query, *params)
            if eq_row and eq_row["equity_at_entry"]:
                initial_equity = float(eq_row["equity_at_entry"])
                if initial_equity > 0:
                    total_pnl_percent = (total_pnl / initial_equity) * 100

        return TradesSummary(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_percent, 2),
            avg_pnl=round(avg_pnl, 2),
            avg_duration_minutes=round(avg_duration, 1) if avg_duration else None,
            max_drawdown_percent=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2) if sharpe_ratio else None,
            best_trade_pnl=round(best_trade, 2),
            worst_trade_pnl=round(worst_trade, 2),
            period_start=row["period_start"].isoformat() if row["period_start"] else None,
            period_end=row["period_end"].isoformat() if row["period_end"] else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trades summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trades summary: {str(e)}"
        )
    finally:
        await conn.close()


@router.get("/trades/latest", response_model=LatestTradeResponse)
async def get_latest_trade(
    request: Request,
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
):
    """
    Get the most recent trade.

    Returns the latest trade based on entry_time.
    Optionally filter by model_id.
    """
    conn = await get_db_connection()

    try:
        # Build query
        if model_id:
            query = """
                SELECT *
                FROM trades
                WHERE model_id = $1
                ORDER BY entry_time DESC
                LIMIT 1
            """
            row = await conn.fetchrow(query, model_id)
        else:
            query = """
                SELECT *
                FROM trades
                ORDER BY entry_time DESC
                LIMIT 1
            """
            row = await conn.fetchrow(query)

        if not row:
            return LatestTradeResponse(
                trade=None,
                message="No trades found"
            )

        trade = row_to_trade_response(row)

        return LatestTradeResponse(
            trade=trade,
            message="Success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest trade: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest trade: {str(e)}"
        )
    finally:
        await conn.close()


@router.get("/trades/{trade_id}", response_model=TradeResponse)
async def get_trade_by_id(
    trade_id: int,
    request: Request,
):
    """
    Get a specific trade by ID.

    Returns full trade details including entry/exit info and P&L.
    """
    conn = await get_db_connection()

    try:
        query = "SELECT * FROM trades WHERE id = $1"
        row = await conn.fetchrow(query, trade_id)

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Trade with id {trade_id} not found"
            )

        return row_to_trade_response(row)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade {trade_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get trade: {str(e)}"
        )
    finally:
        await conn.close()
