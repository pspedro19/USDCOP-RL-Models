/**
 * GET /api/production/monitor - Get all production monitor data
 *
 * Returns production model info, live state, equity curve, and pending experiments summary
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { validateApiAuth } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function GET(request: NextRequest) {
  // Validate authentication
  const authResult = await validateApiAuth(request);
  if (!authResult.authenticated) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // 1. Get production model
    const modelResult = await pool.query(`
      SELECT
        mr.model_id,
        mr.experiment_name,
        mr.stage,
        mr.promoted_at,
        mr.approved_by,
        mr.metrics,
        mr.lineage,
        ec.config_hash,
        ec.feature_order_hash
      FROM model_registry mr
      LEFT JOIN experiment_contracts ec ON ec.config_hash = mr.lineage->>'config_hash'
      WHERE mr.stage = 'production' AND mr.is_active = TRUE
      LIMIT 1
    `);

    const productionModel = modelResult.rows.length > 0 ? {
      modelId: modelResult.rows[0].model_id,
      experimentName: modelResult.rows[0].experiment_name,
      stage: modelResult.rows[0].stage,
      promotedAt: modelResult.rows[0].promoted_at,
      approvedBy: modelResult.rows[0].approved_by,
      configHash: modelResult.rows[0].config_hash || modelResult.rows[0].lineage?.config_hash,
      featureOrderHash: modelResult.rows[0].feature_order_hash || modelResult.rows[0].lineage?.feature_order_hash,
      modelHash: modelResult.rows[0].lineage?.model_hash,
      metrics: modelResult.rows[0].metrics || {},
      lineage: modelResult.rows[0].lineage || {},
    } : null;

    // 2. Get live inference state from inference_signals table
    const liveStateResult = await pool.query(`
      SELECT
        timestamp,
        action,
        confidence,
        price,
        model_id
      FROM inference_signals
      WHERE DATE(timestamp) = CURRENT_DATE
      ORDER BY timestamp DESC
      LIMIT 1
    `);

    const todayStatsResult = await pool.query(`
      SELECT
        COUNT(*) as today_trades,
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) - SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as today_pnl
      FROM execution_log
      WHERE DATE(executed_at) = CURRENT_DATE
    `);

    const positionResult = await pool.query(`
      SELECT position, entry_price, entry_time
      FROM current_positions
      WHERE symbol = 'USDCOP'
      ORDER BY entry_time DESC
      LIMIT 1
    `);

    // Determine if market is open (Colombia: 8am-4pm = 13:00-21:00 UTC)
    const now = new Date();
    const hour = now.getUTCHours();
    const day = now.getUTCDay();
    const isMarketOpen = day >= 1 && day <= 5 && hour >= 13 && hour < 21;

    const liveState = {
      lastSignal: liveStateResult.rows.length > 0 ? {
        timestamp: liveStateResult.rows[0].timestamp,
        action: liveStateResult.rows[0].action,
        confidence: parseFloat(liveStateResult.rows[0].confidence || '0'),
        price: parseFloat(liveStateResult.rows[0].price || '0'),
        modelId: liveStateResult.rows[0].model_id,
      } : null,
      currentPosition: positionResult.rows[0]?.position || 'NEUTRAL',
      positionEntryPrice: positionResult.rows[0]?.entry_price || null,
      positionEntryTime: positionResult.rows[0]?.entry_time || null,
      unrealizedPnL: 0, // Would need live price feed
      todayPnL: parseFloat(todayStatsResult.rows[0]?.today_pnl || '0'),
      todayTrades: parseInt(todayStatsResult.rows[0]?.today_trades || '0', 10),
      lastUpdateTime: new Date().toISOString(),
      isMarketOpen,
    };

    // 3. Get today's equity curve
    const equityResult = await pool.query(`
      SELECT
        timestamp,
        cumulative_pnl as equity,
        position,
        action as signal
      FROM equity_snapshots
      WHERE DATE(timestamp) = CURRENT_DATE
      ORDER BY timestamp ASC
    `);

    const equityCurve = equityResult.rows.length > 0 ? {
      sessionDate: new Date().toISOString().split('T')[0],
      startEquity: 100000, // Base equity
      currentEquity: 100000 + (equityResult.rows[equityResult.rows.length - 1]?.equity || 0),
      points: equityResult.rows.map(row => ({
        timestamp: row.timestamp,
        equity: 100000 + parseFloat(row.equity || '0'),
        position: row.position || 'NEUTRAL',
        signal: row.signal,
      })),
      highWaterMark: Math.max(...equityResult.rows.map(r => 100000 + parseFloat(r.equity || '0'))),
      maxDrawdown: 0, // Would calculate from equity curve
    } : null;

    // 4. Get pending experiments summary
    const pendingResult = await pool.query(`
      SELECT
        proposal_id,
        model_id,
        experiment_name,
        recommendation,
        confidence,
        created_at,
        EXTRACT(EPOCH FROM (expires_at - NOW())) / 3600 as hours_until_expiry
      FROM promotion_proposals
      WHERE status = 'PENDING_APPROVAL'
        AND (expires_at IS NULL OR expires_at > NOW())
      ORDER BY created_at DESC
      LIMIT 5
    `);

    const pendingSummary = {
      count: pendingResult.rowCount || 0,
      experiments: pendingResult.rows.map(row => ({
        proposalId: row.proposal_id,
        modelId: row.model_id,
        experimentName: row.experiment_name,
        recommendation: row.recommendation,
        confidence: parseFloat(row.confidence),
        hoursUntilExpiry: row.hours_until_expiry ? parseFloat(row.hours_until_expiry) : null,
        createdAt: row.created_at,
      })),
    };

    return NextResponse.json({
      productionModel,
      liveState,
      equityCurve,
      pendingSummary,
    });
  } catch (error) {
    console.error('Error fetching production monitor data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch production monitor data' },
      { status: 500 }
    );
  }
}
