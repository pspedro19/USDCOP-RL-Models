"""Reconciliation Engine — compares internal DB signals vs exchange fills.

Reads H5/H1 executions from PostgreSQL, queries exchange fills via SignalBridge
API, and flags discrepancies (missed fills, slippage > threshold, qty mismatch).

Usage:
    engine = ReconciliationEngine(db_conn, signalbridge_url)
    result = engine.reconcile(date.today(), pipeline="h5")
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)

SLIPPAGE_WARN_BPS = 2.0  # Flag slippage > 2 bps


@dataclass
class InternalTrade:
    """A trade as recorded in our DB (signals/executions tables)."""

    signal_date: date
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    leverage: float
    pnl_pct: Optional[float]
    exit_reason: Optional[str]


@dataclass
class ExchangeFill:
    """A fill as returned by the exchange (via SignalBridge API)."""

    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    commission: float
    order_id: str


@dataclass
class ReconciliationItem:
    """Result of comparing one internal trade to its exchange fill."""

    signal_date: date
    pipeline: str
    direction: Optional[str]
    internal_entry_price: Optional[float]
    internal_exit_price: Optional[float]
    internal_leverage: Optional[float]
    internal_pnl_pct: Optional[float]
    internal_exit_reason: Optional[str]
    exchange_entry_price: Optional[float]
    exchange_exit_price: Optional[float]
    exchange_quantity: Optional[float]
    exchange_commission: Optional[float]
    match_status: str  # 'match', 'slippage', 'missed_fill', 'extra_fill', 'qty_mismatch'
    entry_slippage_bps: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    pnl_diff_pct: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class ReconciliationResult:
    """Summary of a reconciliation run."""

    run_date: date
    pipeline: str
    exchange: str
    signals_checked: int = 0
    matches: int = 0
    discrepancies: int = 0
    missed_fills: int = 0
    extra_fills: int = 0
    max_slippage_bps: Optional[float] = None
    avg_slippage_bps: Optional[float] = None
    items: list = field(default_factory=list)
    notes: Optional[str] = None


class ReconciliationEngine:
    """Compare internal trading signals against exchange fills."""

    def __init__(
        self,
        db_conn,
        signalbridge_url: str = "http://localhost:8085",
        slippage_threshold_bps: float = SLIPPAGE_WARN_BPS,
    ):
        self.db_conn = db_conn
        self.signalbridge_url = signalbridge_url.rstrip("/")
        self.slippage_threshold_bps = slippage_threshold_bps

    def reconcile(
        self,
        run_date: date,
        pipeline: str = "h5",
        exchange: str = "mexc",
        dry_run: bool = False,
    ) -> ReconciliationResult:
        """Run reconciliation for a specific date and pipeline.

        Args:
            run_date: The trading date to reconcile.
            pipeline: 'h1' or 'h5'.
            exchange: Exchange name (default 'mexc').
            dry_run: If True, don't persist results to DB.

        Returns:
            ReconciliationResult with match statistics.
        """
        result = ReconciliationResult(
            run_date=run_date, pipeline=pipeline, exchange=exchange
        )

        # 1. Fetch internal trades from DB
        internal_trades = self._fetch_internal_trades(run_date, pipeline)
        result.signals_checked = len(internal_trades)

        if not internal_trades:
            result.notes = f"No {pipeline} trades found for {run_date}"
            logger.info(result.notes)
            if not dry_run:
                self._persist_result(result)
            return result

        # 2. Fetch exchange fills from SignalBridge
        exchange_fills = self._fetch_exchange_fills(run_date, exchange)

        # 3. Match internal trades to exchange fills
        items = self._match_trades(internal_trades, exchange_fills, pipeline)
        result.items = items

        # 4. Compute statistics
        slippages = []
        for item in items:
            if item.match_status == "match":
                result.matches += 1
            elif item.match_status == "slippage":
                result.discrepancies += 1
                if item.entry_slippage_bps is not None:
                    slippages.append(abs(item.entry_slippage_bps))
            elif item.match_status == "missed_fill":
                result.missed_fills += 1
                result.discrepancies += 1
            elif item.match_status == "extra_fill":
                result.extra_fills += 1
                result.discrepancies += 1
            else:
                result.discrepancies += 1

        if slippages:
            result.max_slippage_bps = max(slippages)
            result.avg_slippage_bps = sum(slippages) / len(slippages)

        logger.info(
            f"Reconciliation {pipeline} {run_date}: "
            f"{result.matches} matches, {result.discrepancies} discrepancies, "
            f"{result.missed_fills} missed fills"
        )

        # 5. Persist results
        if not dry_run:
            self._persist_result(result)

        return result

    def _fetch_internal_trades(
        self, run_date: date, pipeline: str
    ) -> list[InternalTrade]:
        """Fetch trades from our DB for the given date and pipeline."""
        trades = []
        cur = self.db_conn.cursor()

        if pipeline == "h5":
            cur.execute(
                """
                SELECT signal_date, direction, entry_price, exit_price,
                       adjusted_leverage, pnl_pct, exit_reason
                FROM forecast_h5_executions
                WHERE signal_date = %s AND status = 'closed'
                ORDER BY signal_date
                """,
                (run_date,),
            )
        elif pipeline == "h1":
            cur.execute(
                """
                SELECT signal_date, direction, entry_price, exit_price,
                       leverage, pnl_pct, exit_reason
                FROM forecast_executions
                WHERE signal_date = %s AND status = 'closed'
                ORDER BY signal_date
                """,
                (run_date,),
            )
        else:
            logger.warning(f"Unknown pipeline: {pipeline}")
            return trades

        for row in cur.fetchall():
            trades.append(
                InternalTrade(
                    signal_date=row[0],
                    direction=row[1] or "SHORT",
                    entry_price=float(row[2]) if row[2] else 0.0,
                    exit_price=float(row[3]) if row[3] else None,
                    leverage=float(row[4]) if row[4] else 1.0,
                    pnl_pct=float(row[5]) if row[5] else None,
                    exit_reason=row[6],
                )
            )

        cur.close()
        return trades

    def _fetch_exchange_fills(
        self, run_date: date, exchange: str
    ) -> list[ExchangeFill]:
        """Fetch fills from SignalBridge API for the given date."""
        fills = []
        try:
            import urllib.request
            import json

            url = (
                f"{self.signalbridge_url}/api/signal-bridge/history"
                f"?date={run_date.isoformat()}&exchange={exchange}"
            )
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())

            for execution in data.get("executions", []):
                if execution.get("status") != "FILLED":
                    continue
                fills.append(
                    ExchangeFill(
                        timestamp=datetime.fromisoformat(
                            execution.get("created_at", "")
                        ),
                        side=execution.get("side", ""),
                        price=float(execution.get("filled_price", 0)),
                        quantity=float(execution.get("filled_qty", 0)),
                        commission=float(execution.get("commission", 0)),
                        order_id=execution.get("exchange_order_id", ""),
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to fetch exchange fills: {e}")

        return fills

    def _match_trades(
        self,
        internal: list[InternalTrade],
        exchange: list[ExchangeFill],
        pipeline: str,
    ) -> list[ReconciliationItem]:
        """Match internal trades to exchange fills."""
        items = []
        used_fills = set()

        for trade in internal:
            best_fill = None
            best_slippage = float("inf")

            # Find the exchange fill closest in price to the internal trade
            for i, fill in enumerate(exchange):
                if i in used_fills:
                    continue
                if trade.entry_price and fill.price:
                    slippage = abs(fill.price - trade.entry_price) / trade.entry_price
                    if slippage < best_slippage:
                        best_slippage = slippage
                        best_fill = (i, fill)

            if best_fill is None:
                # No matching fill found
                items.append(
                    ReconciliationItem(
                        signal_date=trade.signal_date,
                        pipeline=pipeline,
                        direction=trade.direction,
                        internal_entry_price=trade.entry_price,
                        internal_exit_price=trade.exit_price,
                        internal_leverage=trade.leverage,
                        internal_pnl_pct=trade.pnl_pct,
                        internal_exit_reason=trade.exit_reason,
                        exchange_entry_price=None,
                        exchange_exit_price=None,
                        exchange_quantity=None,
                        exchange_commission=None,
                        match_status="missed_fill",
                        notes="No matching exchange fill found",
                    )
                )
                continue

            fill_idx, fill = best_fill
            used_fills.add(fill_idx)

            entry_slippage_bps = (
                (fill.price - trade.entry_price) / trade.entry_price * 10000
                if trade.entry_price
                else None
            )

            if entry_slippage_bps and abs(entry_slippage_bps) > self.slippage_threshold_bps:
                status = "slippage"
            else:
                status = "match"

            items.append(
                ReconciliationItem(
                    signal_date=trade.signal_date,
                    pipeline=pipeline,
                    direction=trade.direction,
                    internal_entry_price=trade.entry_price,
                    internal_exit_price=trade.exit_price,
                    internal_leverage=trade.leverage,
                    internal_pnl_pct=trade.pnl_pct,
                    internal_exit_reason=trade.exit_reason,
                    exchange_entry_price=fill.price,
                    exchange_exit_price=None,
                    exchange_quantity=fill.quantity,
                    exchange_commission=fill.commission,
                    match_status=status,
                    entry_slippage_bps=entry_slippage_bps,
                )
            )

        # Extra fills (exchange fills with no internal match)
        for i, fill in enumerate(exchange):
            if i not in used_fills:
                items.append(
                    ReconciliationItem(
                        signal_date=run_date if internal else date.today(),
                        pipeline=pipeline,
                        direction=None,
                        internal_entry_price=None,
                        internal_exit_price=None,
                        internal_leverage=None,
                        internal_pnl_pct=None,
                        internal_exit_reason=None,
                        exchange_entry_price=fill.price,
                        exchange_exit_price=None,
                        exchange_quantity=fill.quantity,
                        exchange_commission=fill.commission,
                        match_status="extra_fill",
                        notes=f"Exchange fill with no internal signal: order {fill.order_id}",
                    )
                )

        return items

    def _persist_result(self, result: ReconciliationResult) -> None:
        """Write reconciliation results to the database."""
        cur = self.db_conn.cursor()
        try:
            # Insert run summary
            cur.execute(
                """
                INSERT INTO reconciliation_runs
                    (run_date, pipeline, exchange, status, signals_checked,
                     matches, discrepancies, missed_fills, extra_fills,
                     max_slippage_bps, avg_slippage_bps, notes, completed_at)
                VALUES (%s, %s, %s, 'completed', %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (run_date, pipeline, exchange) DO UPDATE SET
                    status = 'completed',
                    signals_checked = EXCLUDED.signals_checked,
                    matches = EXCLUDED.matches,
                    discrepancies = EXCLUDED.discrepancies,
                    missed_fills = EXCLUDED.missed_fills,
                    extra_fills = EXCLUDED.extra_fills,
                    max_slippage_bps = EXCLUDED.max_slippage_bps,
                    avg_slippage_bps = EXCLUDED.avg_slippage_bps,
                    notes = EXCLUDED.notes,
                    completed_at = NOW()
                RETURNING id
                """,
                (
                    result.run_date,
                    result.pipeline,
                    result.exchange,
                    result.signals_checked,
                    result.matches,
                    result.discrepancies,
                    result.missed_fills,
                    result.extra_fills,
                    result.max_slippage_bps,
                    result.avg_slippage_bps,
                    result.notes,
                ),
            )
            run_id = cur.fetchone()[0]

            # Insert items
            for item in result.items:
                cur.execute(
                    """
                    INSERT INTO reconciliation_items
                        (run_id, signal_date, pipeline, direction,
                         internal_entry_price, internal_exit_price,
                         internal_leverage, internal_pnl_pct, internal_exit_reason,
                         exchange_entry_price, exchange_exit_price,
                         exchange_quantity, exchange_commission,
                         match_status, entry_slippage_bps, exit_slippage_bps,
                         pnl_diff_pct, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        item.signal_date,
                        item.pipeline,
                        item.direction,
                        item.internal_entry_price,
                        item.internal_exit_price,
                        item.internal_leverage,
                        item.internal_pnl_pct,
                        item.internal_exit_reason,
                        item.exchange_entry_price,
                        item.exchange_exit_price,
                        item.exchange_quantity,
                        item.exchange_commission,
                        item.match_status,
                        item.entry_slippage_bps,
                        item.exit_slippage_bps,
                        item.pnl_diff_pct,
                        item.notes,
                    ),
                )

            self.db_conn.commit()
            logger.info(f"Persisted reconciliation run {run_id} with {len(result.items)} items")

        except Exception as e:
            self.db_conn.rollback()
            logger.error(f"Failed to persist reconciliation: {e}")
            raise
        finally:
            cur.close()
