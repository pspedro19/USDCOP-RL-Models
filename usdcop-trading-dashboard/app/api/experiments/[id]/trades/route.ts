/**
 * GET /api/experiments/[id]/trades - Get L4 backtest trades for exact replay
 *
 * Contract: CTR-L4-TRADES-API-001
 * Version: 1.0.0
 * Created: 2026-02-04
 *
 * Purpose: Retrieve the EXACT trades that L4 generated during backtest validation.
 * This ensures 100% consistency between the metrics shown in FloatingExperimentPanel
 * and the trades displayed in backtest replay chart.
 *
 * Data Source: backtest_trades table (populated by L4 DAG)
 */

import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Type for a backtest trade from database
interface BacktestTrade {
  id: number;
  proposalId: string;
  tradeId: number;
  modelId: string;
  timestamp: string;
  entryTime: string;
  exitTime: string | null;
  side: 'LONG' | 'SHORT' | 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number | null;
  pnl: number | null;
  pnlUsd: number | null;
  pnlPercent: number | null;
  status: 'open' | 'closed' | 'pending';
  durationMinutes: number | null;
  exitReason: string | null;
  equityAtEntry: number | null;
  equityAtExit: number | null;
  entryConfidence: number | null;
  exitConfidence: number | null;
  rawAction: number | null;
  createdAt: string;
}

// Response type
interface TradesResponse {
  proposalId: string;
  modelId: string;
  tradesCount: number;
  trades: BacktestTrade[];
  source: 'l4_backtest';  // Always 'l4_backtest' for this endpoint
  summary: {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    totalPnlUsd: number;
    firstTrade: string | null;
    lastTrade: string | null;
  };
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  // Validate authentication
  const skipAuth = process.env.SKIP_AUTH === 'true';
  if (!skipAuth) {
    const authResult = await protectApiRoute(request, { rateLimit: false });
    if (!authResult.authenticated) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const proposalId = params.id;

  try {
    // First, verify the proposal exists
    const proposalResult = await pool.query(`
      SELECT proposal_id, model_id
      FROM promotion_proposals
      WHERE proposal_id = $1
    `, [proposalId]);

    if (proposalResult.rows.length === 0) {
      return NextResponse.json(
        { error: 'Proposal not found', proposalId },
        { status: 404 }
      );
    }

    const { model_id: modelId } = proposalResult.rows[0];

    // Get all trades for this proposal, ordered by entry time
    const tradesResult = await pool.query(`
      SELECT
        id,
        proposal_id,
        trade_id,
        model_id,
        timestamp,
        entry_time,
        exit_time,
        side,
        entry_price,
        exit_price,
        pnl,
        pnl_usd,
        pnl_percent,
        status,
        duration_minutes,
        exit_reason,
        equity_at_entry,
        equity_at_exit,
        entry_confidence,
        exit_confidence,
        raw_action,
        created_at
      FROM backtest_trades
      WHERE proposal_id = $1
      ORDER BY entry_time ASC
    `, [proposalId]);

    // Calculate summary statistics
    const trades = tradesResult.rows;
    const winningTrades = trades.filter(t => (t.pnl_usd || 0) > 0);
    const losingTrades = trades.filter(t => (t.pnl_usd || 0) <= 0);
    const totalPnlUsd = trades.reduce((sum, t) => sum + (parseFloat(t.pnl_usd) || 0), 0);

    const response: TradesResponse = {
      proposalId,
      modelId,
      tradesCount: trades.length,
      source: 'l4_backtest',
      trades: trades.map(row => ({
        id: row.id,
        proposalId: row.proposal_id,
        tradeId: row.trade_id,
        modelId: row.model_id,
        timestamp: row.timestamp,
        entryTime: row.entry_time,
        exitTime: row.exit_time,
        side: row.side,
        entryPrice: parseFloat(row.entry_price),
        exitPrice: row.exit_price ? parseFloat(row.exit_price) : null,
        pnl: row.pnl ? parseFloat(row.pnl) : null,
        pnlUsd: row.pnl_usd ? parseFloat(row.pnl_usd) : null,
        pnlPercent: row.pnl_percent ? parseFloat(row.pnl_percent) : null,
        status: row.status,
        durationMinutes: row.duration_minutes,
        exitReason: row.exit_reason,
        equityAtEntry: row.equity_at_entry ? parseFloat(row.equity_at_entry) : null,
        equityAtExit: row.equity_at_exit ? parseFloat(row.equity_at_exit) : null,
        entryConfidence: row.entry_confidence ? parseFloat(row.entry_confidence) : null,
        exitConfidence: row.exit_confidence ? parseFloat(row.exit_confidence) : null,
        rawAction: row.raw_action ? parseFloat(row.raw_action) : null,
        createdAt: row.created_at,
      })),
      summary: {
        totalTrades: trades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        winRate: trades.length > 0 ? winningTrades.length / trades.length : 0,
        totalPnlUsd,
        firstTrade: trades.length > 0 ? trades[0].entry_time : null,
        lastTrade: trades.length > 0 ? trades[trades.length - 1].exit_time : null,
      },
    };

    // If no trades found, return empty result (not an error - L4 might not have run yet)
    if (trades.length === 0) {
      return NextResponse.json({
        ...response,
        message: 'No L4 trades found for this proposal. The backtest may not have been run yet, or trades were not persisted.',
      });
    }

    return NextResponse.json(response);

  } catch (error) {
    console.error('Error fetching L4 trades:', error);
    return NextResponse.json(
      { error: 'Failed to fetch trades', details: String(error) },
      { status: 500 }
    );
  }
}
