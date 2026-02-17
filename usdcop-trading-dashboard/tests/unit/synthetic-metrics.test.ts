/**
 * Synthetic Backtest Metrics Validation
 * Verifies investor demo produces coherent, target metrics.
 */
import { describe, it, expect } from 'vitest';
import { generateSyntheticTrades, calculateBacktestSummary } from '@/lib/services/synthetic-backtest.service';

describe('Investor Demo Metrics - Full Year 2025', () => {
  const trades = generateSyntheticTrades({
    startDate: '2025-01-01',
    endDate: '2025-12-31',
    modelId: 'investor_demo',
  });
  const summary = calculateBacktestSummary(trades);

  it('generates ~120 trades (10/month)', () => {
    expect(trades.length).toBeGreaterThanOrEqual(100);
    expect(trades.length).toBeLessThanOrEqual(140);
  });

  it('targets ~33% annual return', () => {
    expect(summary.total_return_pct).toBeGreaterThan(25);
    expect(summary.total_return_pct).toBeLessThan(45);
  });

  it('achieves ~60% win rate', () => {
    expect(summary.win_rate).toBeGreaterThan(55);
    expect(summary.win_rate).toBeLessThan(65);
  });

  it('keeps max drawdown under 15%', () => {
    expect(summary.max_drawdown_pct).toBeLessThan(15);
  });

  it('produces positive Sharpe ratio', () => {
    expect(summary.sharpe_ratio).toBeGreaterThan(0.5);
  });

  it('has realistic average trade duration (30-240 min)', () => {
    expect(summary.avg_trade_duration_minutes).toBeGreaterThan(30);
    expect(summary.avg_trade_duration_minutes).toBeLessThan(240);
  });

  it('all trades have proper equity tracking', () => {
    for (const t of trades) {
      expect(t.equity_at_entry).toBeGreaterThan(0);
      expect(t.equity_at_exit).toBeGreaterThan(0);
      expect(t.entry_confidence).toBeGreaterThan(0);
      expect(t.entry_confidence).toBeLessThanOrEqual(1);
    }
  });

  it('equity curve ends higher than start', () => {
    const lastEquity = trades[trades.length - 1].equity_at_exit;
    expect(lastEquity).toBeGreaterThan(10000);
  });

  it('trades are on business days only', () => {
    for (const t of trades) {
      const day = new Date(t.entry_time).getUTCDay();
      expect(day).toBeGreaterThan(0); // not Sunday
      expect(day).toBeLessThan(6);    // not Saturday
    }
  });

  it('is deterministic (same seed = same results)', () => {
    const trades2 = generateSyntheticTrades({
      startDate: '2025-01-01',
      endDate: '2025-12-31',
      modelId: 'investor_demo',
    });
    expect(trades2.length).toBe(trades.length);
    expect(trades2[0].entry_price).toBe(trades[0].entry_price);
    expect(trades2[trades.length - 1].pnl_usd).toBe(trades[trades.length - 1].pnl_usd);
  });
});

describe('Investor Demo Metrics - H1 2025 (6 months)', () => {
  const trades = generateSyntheticTrades({
    startDate: '2025-01-01',
    endDate: '2025-06-30',
    modelId: 'demo_model',
  });
  const summary = calculateBacktestSummary(trades);

  it('generates ~60 trades for 6 months', () => {
    expect(trades.length).toBeGreaterThanOrEqual(50);
    expect(trades.length).toBeLessThanOrEqual(70);
  });

  it('targets ~16.5% return (half-year pro-rata)', () => {
    expect(summary.total_return_pct).toBeGreaterThan(10);
    expect(summary.total_return_pct).toBeLessThan(25);
  });

  it('print summary for inspection', () => {
    console.log('\n=== Full Year Summary ===');
    const fullYear = calculateBacktestSummary(generateSyntheticTrades({
      startDate: '2025-01-01', endDate: '2025-12-31', modelId: 'investor_demo',
    }));
    console.log(JSON.stringify(fullYear, null, 2));

    console.log('\n=== H1 2025 Summary ===');
    console.log(JSON.stringify(summary, null, 2));
  });
});
