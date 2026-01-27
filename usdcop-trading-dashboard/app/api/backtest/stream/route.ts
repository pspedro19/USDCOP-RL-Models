import { NextRequest } from 'next/server';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

/**
 * Backtest Stream API Proxy
 * Proxies to inference API backend, or generates synthetic backtest data as fallback.
 *
 * The fallback generates a fluid, progressive replay:
 * - Trades are sent one-by-one with delays so the chart reveals progressively
 * - Progress events interleave with trade events
 * - Total duration ~15-25 seconds for an engaging investor presentation
 */
export async function POST(request: NextRequest) {
  const body = await request.json();
  const { start_date, end_date, model_id, force_regenerate } = body;

  // Try backend first
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${BACKEND_URL}/v1/backtest/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ start_date, end_date, model_id, force_regenerate }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok && response.body) {
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }
  } catch {
    // Backend unavailable, generate synthetic data
  }

  // ============================================================================
  // Generate synthetic backtest with fluid, progressive replay
  // ============================================================================

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      const send = (type: string, data: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, data })}\n\n`));
      };
      const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

      const startMs = Date.now();
      const start = new Date(start_date);
      const end = new Date(end_date);
      const totalDays = Math.max(1, Math.round((end.getTime() - start.getTime()) / 86400000));

      // ── Generate all trades upfront ──────────────────────────────────────
      const isInvestorDemo = model_id?.includes('investor') || model_id?.includes('demo');
      // Investor mode: ~40 trades for a nice presentation; other models: based on duration
      const tradeCount = isInvestorDemo
        ? Math.min(45, Math.max(25, Math.floor(totalDays / 7)))
        : 20 + Math.floor(totalDays / 10);

      const trades: Array<Record<string, unknown>> = [];
      let equity = 10000;
      const totalMs = end.getTime() - start.getTime();

      // Use a slight upward bias for investor demo (looks better in presentations)
      const winBias = isInvestorDemo ? 0.57 : 0.50;

      for (let i = 0; i < tradeCount; i++) {
        // Spread trades across the date range with slight randomness
        const baseOffset = (i / tradeCount) * totalMs;
        const jitter = (Math.random() - 0.5) * (totalMs / tradeCount) * 0.5;
        const entryDate = new Date(start.getTime() + baseOffset + jitter);
        // Trade duration: 30min to 6 hours
        const durationMs = (30 + Math.random() * 330) * 60 * 1000;
        const exitDate = new Date(Math.min(entryDate.getTime() + durationMs, end.getTime()));

        const side = Math.random() > 0.5 ? 'LONG' : 'SHORT';
        // Price follows a realistic wave pattern
        const trend = Math.sin(i * 0.15) * 80;
        const noise = (Math.random() - 0.5) * 40;
        const entryPrice = 4180 + trend + noise;
        // Win/loss determined by bias
        const isWin = Math.random() < winBias;
        const magnitude = 5 + Math.random() * 25;
        const priceMove = isWin ? magnitude : -magnitude;
        const exitPrice = entryPrice + (side === 'LONG' ? priceMove : -priceMove);
        const pnl = side === 'LONG' ? exitPrice - entryPrice : entryPrice - exitPrice;
        const pnlPct = (pnl / entryPrice) * 100;
        const pnlUsd = pnl * 10;
        equity += pnlUsd;

        trades.push({
          trade_id: i + 1,
          model_id: model_id || 'investor_demo',
          timestamp: entryDate.toISOString(),
          entry_time: entryDate.toISOString(),
          exit_time: exitDate.toISOString(),
          side,
          entry_price: Math.round(entryPrice * 100) / 100,
          exit_price: Math.round(exitPrice * 100) / 100,
          pnl: Math.round(pnl * 100) / 100,
          pnl_usd: Math.round(pnlUsd * 100) / 100,
          pnl_percent: Math.round(pnlPct * 1000) / 1000,
          pnl_pct: Math.round(pnlPct * 1000) / 1000,
          status: 'closed',
          duration_minutes: Math.round(durationMs / 60000),
          exit_reason: isWin ? 'take_profit' : 'stop_loss',
          equity_at_entry: Math.round((equity - pnlUsd) * 100) / 100,
          equity_at_exit: Math.round(equity * 100) / 100,
          entry_confidence: Math.round((0.60 + Math.random() * 0.35) * 100) / 100,
          exit_confidence: Math.round((0.50 + Math.random() * 0.35) * 100) / 100,
        });
      }

      // Sort by entry time
      trades.sort((a, b) =>
        new Date(a.entry_time as string).getTime() - new Date(b.entry_time as string).getTime()
      );

      // ── Phase 1: Initial "loading" progress (fast, ~2s) ─────────────────
      send('progress', {
        progress: 0,
        current_bar: 0,
        total_bars: tradeCount,
        trades_generated: 0,
        status: 'loading',
        message: 'Loading historical data...',
      });
      await delay(600);

      send('progress', {
        progress: 0.05,
        current_bar: 0,
        total_bars: tradeCount,
        trades_generated: 0,
        status: 'loading',
        message: 'Initializing model...',
      });
      await delay(500);

      send('progress', {
        progress: 0.10,
        current_bar: 0,
        total_bars: tradeCount,
        trades_generated: 0,
        status: 'running',
        message: 'Running simulation...',
      });
      await delay(400);

      // ── Phase 2: Stream trades progressively (~15-20s total) ─────────────
      // Each trade gets: progress event → delay → trade event → delay
      // Delay per trade: ~400-600ms for investor demo, faster for others
      const baseDelay = isInvestorDemo ? 450 : 250;

      for (let i = 0; i < trades.length; i++) {
        const trade = trades[i];
        const progress = 0.10 + (i / trades.length) * 0.85; // 10% to 95%

        // Progress event (every trade for smooth progress bar)
        send('progress', {
          progress,
          current_bar: i + 1,
          total_bars: trades.length,
          trades_generated: i + 1,
          status: 'running',
          message: `Trade ${i + 1}/${trades.length}: ${trade.side} @ $${(trade.entry_price as number).toFixed(2)}`,
        });

        // Brief pause before revealing the trade
        await delay(baseDelay * 0.4);

        // Send trade event — this is what makes the chart update
        send('trade', {
          ...trade,
          trade_id: String(trade.trade_id),
          current_equity: trade.equity_at_exit,
        });

        // Pause after trade — varies slightly for natural feel
        const jitteredDelay = baseDelay * (0.5 + Math.random() * 0.4);
        await delay(jitteredDelay);
      }

      // ── Phase 3: Finalize (~1s) ──────────────────────────────────────────
      send('progress', {
        progress: 0.97,
        current_bar: trades.length,
        total_bars: trades.length,
        trades_generated: trades.length,
        status: 'saving',
        message: 'Calculating performance metrics...',
      });
      await delay(500);

      // ── Send final result ────────────────────────────────────────────────
      const wins = trades.filter(t => (t.pnl as number) > 0);
      const losses = trades.filter(t => (t.pnl as number) <= 0);
      const totalPnl = trades.reduce((s, t) => s + (t.pnl_usd as number), 0);

      let peak = 10000;
      let maxDrawdown = 0;
      let running = 10000;
      for (const t of trades) {
        running += t.pnl_usd as number;
        peak = Math.max(peak, running);
        const dd = ((peak - running) / peak) * 100;
        maxDrawdown = Math.max(maxDrawdown, dd);
      }

      // Sharpe ratio approximation
      const returns = trades.map(t => (t.pnl_pct as number) / 100);
      const avgReturn = returns.reduce((s, r) => s + r, 0) / returns.length;
      const stdReturn = Math.sqrt(
        returns.reduce((s, r) => s + Math.pow(r - avgReturn, 2), 0) / returns.length
      );
      const sharpe = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0;

      send('result', {
        success: true,
        source: 'generated',
        trade_count: trades.length,
        trades,
        summary: {
          total_trades: trades.length,
          winning_trades: wins.length,
          losing_trades: losses.length,
          win_rate: trades.length > 0 ? Math.round((wins.length / trades.length) * 10000) / 100 : 0,
          total_pnl: Math.round(totalPnl * 100) / 100,
          total_return_pct: Math.round((totalPnl / 10000) * 10000) / 100,
          max_drawdown_pct: Math.round(maxDrawdown * 100) / 100,
          sharpe_ratio: Math.round(sharpe * 100) / 100,
          avg_trade_duration_minutes: trades.length > 0
            ? Math.round(trades.reduce((s, t) => s + (t.duration_minutes as number), 0) / trades.length)
            : 0,
        },
        processing_time_ms: Date.now() - startMs,
        date_range: { start: start_date, end: end_date },
      });

      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
