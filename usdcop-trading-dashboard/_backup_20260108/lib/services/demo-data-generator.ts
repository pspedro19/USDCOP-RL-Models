/**
 * Demo Data Generator
 * ====================
 * Generates realistic demo data for SAC and LLM models.
 * Used when a demo model is selected (isRealData === false).
 */

// ============================================================================
// Types
// ============================================================================

export interface DemoSignal {
  strategy_code: string;
  timestamp: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entry_price: number;
  size: number;
  side: 'LONG' | 'SHORT' | 'FLAT';
}

export interface DemoTrade {
  trade_id: number;
  strategy_code: string;
  timestamp: string;
  entry_time: string;
  exit_time: string | null;
  side: 'BUY' | 'SELL';
  entry_price: number;
  exit_price: number | null;
  pnl: number;
  pnl_pct: number;
  duration_minutes: number;
  status: 'OPEN' | 'CLOSED';
}

export interface DemoEquityPoint {
  timestamp: string;
  equity_value: number;
  return_pct: number;
  drawdown_pct: number;
}

export interface DemoPosition {
  strategy_code: string;
  strategy_name: string;
  side: 'long' | 'short' | 'flat';
  entry_price: number | null;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_time: string | null;
  duration_minutes: number;
}

export interface DemoMetrics {
  modelId: string;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  pnl_month: number;
  pnl_month_pct: number;
}

// ============================================================================
// Model-specific configuration
// ============================================================================

const MODEL_CONFIGS: Record<string, {
  name: string;
  baseReturn: number;
  volatility: number;
  winRate: number;
  signalFrequency: number; // signals per hour
  avgTradeDuration: number; // minutes
}> = {
  'sac_v1_demo': {
    name: 'SAC V1 Demo',
    baseReturn: 0.08, // 8% monthly return
    volatility: 0.15,
    winRate: 0.54,
    signalFrequency: 2,
    avgTradeDuration: 45,
  },
  'llm_claude_demo': {
    name: 'LLM Claude Demo',
    baseReturn: 0.06, // 6% monthly return
    volatility: 0.12,
    winRate: 0.51,
    signalFrequency: 1.5,
    avgTradeDuration: 60,
  },
};

// ============================================================================
// Random utilities
// ============================================================================

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function normalRandom(rand: () => number, mean: number = 0, stdDev: number = 1): number {
  const u1 = rand();
  const u2 = rand();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * stdDev + mean;
}

// ============================================================================
// Demo Data Generators
// ============================================================================

/**
 * Generate demo signals for a model
 */
export function generateDemoSignals(
  modelId: string,
  count: number = 20,
  basePrice: number = 4285
): DemoSignal[] {
  const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['sac_v1_demo'];
  const seed = modelId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = seededRandom(seed + Date.now() % 10000);

  const signals: DemoSignal[] = [];
  let currentPrice = basePrice;
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(now.getTime() - (count - i) * 5 * 60 * 1000);
    currentPrice += normalRandom(rand, 0, 5);

    const signalRand = rand();
    let signal: 'BUY' | 'SELL' | 'HOLD';
    let side: 'LONG' | 'SHORT' | 'FLAT';

    if (signalRand > 0.7) {
      signal = 'BUY';
      side = 'LONG';
    } else if (signalRand > 0.4) {
      signal = 'SELL';
      side = 'SHORT';
    } else {
      signal = 'HOLD';
      side = 'FLAT';
    }

    signals.push({
      strategy_code: modelId,
      timestamp: timestamp.toISOString(),
      signal,
      confidence: 50 + rand() * 40,
      entry_price: Math.round(currentPrice * 100) / 100,
      size: 1,
      side,
    });
  }

  return signals;
}

/**
 * Generate demo trades for a model
 */
export function generateDemoTrades(
  modelId: string,
  count: number = 20,
  basePrice: number = 4285
): DemoTrade[] {
  const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['sac_v1_demo'];
  const seed = modelId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = seededRandom(seed);

  const trades: DemoTrade[] = [];
  let currentPrice = basePrice;
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const entryTime = new Date(now.getTime() - (count - i) * config.avgTradeDuration * 60 * 1000);
    const duration = Math.round(config.avgTradeDuration * (0.5 + rand()));
    const exitTime = new Date(entryTime.getTime() + duration * 60 * 1000);

    const entryPrice = currentPrice + normalRandom(rand, 0, 3);
    const priceChange = normalRandom(rand, config.baseReturn / count, config.volatility / Math.sqrt(count));
    const exitPrice = entryPrice * (1 + priceChange / 100);

    const side: 'BUY' | 'SELL' = rand() > 0.5 ? 'BUY' : 'SELL';
    const pnlMultiplier = side === 'BUY' ? 1 : -1;
    const pnl = (exitPrice - entryPrice) * pnlMultiplier * 1000; // assuming $1000 position
    const pnlPct = ((exitPrice - entryPrice) / entryPrice) * 100 * pnlMultiplier;

    currentPrice = exitPrice;

    trades.push({
      trade_id: i + 1,
      strategy_code: modelId,
      timestamp: entryTime.toISOString(),
      entry_time: entryTime.toISOString(),
      exit_time: exitTime.toISOString(),
      side,
      entry_price: Math.round(entryPrice * 100) / 100,
      exit_price: Math.round(exitPrice * 100) / 100,
      pnl: Math.round(pnl * 100) / 100,
      pnl_pct: Math.round(pnlPct * 100) / 100,
      duration_minutes: duration,
      status: 'CLOSED',
    });
  }

  return trades.reverse(); // Most recent first
}

/**
 * Generate demo equity curve for a model
 */
export function generateDemoEquityCurve(
  modelId: string,
  hours: number = 24,
  initialCapital: number = 10000
): DemoEquityPoint[] {
  const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['sac_v1_demo'];
  const seed = modelId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = seededRandom(seed);

  const points: DemoEquityPoint[] = [];
  let equity = initialCapital;
  let peakEquity = initialCapital;
  const now = new Date();
  const intervalMinutes = 5;
  const totalPoints = (hours * 60) / intervalMinutes;

  for (let i = 0; i < totalPoints; i++) {
    const timestamp = new Date(now.getTime() - (totalPoints - i) * intervalMinutes * 60 * 1000);

    // Generate return with trend + random component
    const dailyReturn = config.baseReturn / (24 * 12); // per 5-min bar
    const randomReturn = normalRandom(rand, dailyReturn, config.volatility / Math.sqrt(24 * 12));
    equity = equity * (1 + randomReturn / 100);

    // Track peak for drawdown calculation
    peakEquity = Math.max(peakEquity, equity);
    const drawdownPct = ((peakEquity - equity) / peakEquity) * 100;

    points.push({
      timestamp: timestamp.toISOString(),
      equity_value: Math.round(equity * 100) / 100,
      return_pct: Math.round(((equity - initialCapital) / initialCapital) * 10000) / 100,
      drawdown_pct: Math.round(drawdownPct * 100) / 100,
    });
  }

  return points;
}

/**
 * Generate demo position for a model
 */
export function generateDemoPosition(
  modelId: string,
  currentPrice: number = 4285
): DemoPosition {
  const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['sac_v1_demo'];
  const seed = modelId.split('').reduce((a, c) => a + c.charCodeAt(0), 0) + Date.now() % 1000;
  const rand = seededRandom(seed);

  const sideRand = rand();
  let side: 'long' | 'short' | 'flat';

  if (sideRand > 0.6) {
    side = 'long';
  } else if (sideRand > 0.3) {
    side = 'short';
  } else {
    side = 'flat';
  }

  if (side === 'flat') {
    return {
      strategy_code: modelId,
      strategy_name: config.name,
      side: 'flat',
      entry_price: null,
      current_price: currentPrice,
      unrealized_pnl: 0,
      unrealized_pnl_pct: 0,
      entry_time: null,
      duration_minutes: 0,
    };
  }

  const entryPrice = currentPrice * (1 + normalRandom(rand, 0, 0.002));
  const pnlMultiplier = side === 'long' ? 1 : -1;
  const unrealizedPnl = (currentPrice - entryPrice) * pnlMultiplier * 1000;
  const unrealizedPnlPct = ((currentPrice - entryPrice) / entryPrice) * 100 * pnlMultiplier;

  const entryTime = new Date(Date.now() - Math.random() * config.avgTradeDuration * 60 * 1000);

  return {
    strategy_code: modelId,
    strategy_name: config.name,
    side,
    entry_price: Math.round(entryPrice * 100) / 100,
    current_price: currentPrice,
    unrealized_pnl: Math.round(unrealizedPnl * 100) / 100,
    unrealized_pnl_pct: Math.round(unrealizedPnlPct * 100) / 100,
    entry_time: entryTime.toISOString(),
    duration_minutes: Math.round((Date.now() - entryTime.getTime()) / 60000),
  };
}

/**
 * Generate demo metrics for a model
 */
export function generateDemoMetrics(modelId: string): DemoMetrics {
  const config = MODEL_CONFIGS[modelId] || MODEL_CONFIGS['sac_v1_demo'];
  const seed = modelId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const rand = seededRandom(seed);

  return {
    modelId,
    sharpe_ratio: 1.2 + rand() * 0.8,
    max_drawdown_pct: 8 + rand() * 10,
    win_rate: config.winRate * 100,
    total_trades: Math.round(50 + rand() * 150),
    profit_factor: 1.2 + rand() * 0.5,
    pnl_month: 500 + rand() * 1000,
    pnl_month_pct: config.baseReturn * 100 * (0.8 + rand() * 0.4),
  };
}

/**
 * Check if a model uses demo data
 */
export function isDemoModel(modelId: string): boolean {
  return modelId.includes('demo') || modelId in MODEL_CONFIGS;
}

export default {
  generateDemoSignals,
  generateDemoTrades,
  generateDemoEquityCurve,
  generateDemoPosition,
  generateDemoMetrics,
  isDemoModel,
};
