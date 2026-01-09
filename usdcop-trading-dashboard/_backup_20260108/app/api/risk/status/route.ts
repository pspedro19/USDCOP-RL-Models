import { NextResponse } from 'next/server';

const TRADING_API_URL = process.env.NEXT_PUBLIC_TRADING_API_URL || 'http://localhost:8000';

/**
 * Risk Status API
 * Returns current risk management state for paper trading
 */
export async function GET() {
  try {
    // Try to fetch from backend
    const response = await fetch(`${TRADING_API_URL}/api/risk/status`, {
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
      cache: 'no-store'
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch (error) {
    // Backend not available, return demo data
  }

  // Demo data for paper trading mode
  const demoRiskStatus = {
    is_paper_trading: true,
    kill_switch_active: false,
    daily_blocked: false,
    cooldown_active: false,
    cooldown_remaining_minutes: 0,
    current_drawdown_pct: 1.2,
    daily_pnl_pct: 0.85,
    trade_count_today: 3,
    consecutive_losses: 0,
    limits: {
      max_drawdown_pct: 15,
      max_daily_loss_pct: 5,
      max_trades_per_day: 20,
      max_position_size: 1000,
      cooldown_minutes: 30
    },
    last_updated: new Date().toISOString(),
    source: 'demo'
  };

  return NextResponse.json(demoRiskStatus, {
    headers: { 'Cache-Control': 'no-store, max-age=0' }
  });
}
