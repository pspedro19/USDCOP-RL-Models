import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

interface L5Prediction {
  timestamp: string;
  prediction: number;
  confidence: number;
  action: string;
  price: number;
  features?: any;
  metadata?: any;
}

interface TechnicalIndicators {
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
  };
  volume: number;
  sma20: number;
  sma50: number;
  ema12: number;
  ema26: number;
  stochastic: {
    k: number;
    d: number;
  };
  atr: number;
  adx: number;
}

interface TradingSignal {
  id: string;
  timestamp: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string[];
  riskScore: number;
  expectedReturn: number;
  timeHorizon: string;
  modelSource: string;
  latency: number;
  technicalIndicators?: TechnicalIndicators;
  mlPrediction?: L5Prediction;
  // New fields for tipo_accion tracking
  dataType?: 'backtest' | 'out_of_sample' | 'live';
  model_id?: string;
}

interface SignalPerformance {
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  totalSignals: number;
  activeSignals: number;
}

// NO RANDOM VALUES - Real calculations or null
// Calculate Stochastic Oscillator from actual price data
function calculateStochastic(prices: number[], period: number = 14): { k: number, d: number } {
  if (prices.length < period) {
    return { k: 50, d: 50 }; // Neutral when insufficient data
  }

  const recentPrices = prices.slice(-period);
  const high = Math.max(...recentPrices);
  const low = Math.min(...recentPrices);
  const close = recentPrices[recentPrices.length - 1];

  // %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) × 100
  const k = high !== low ? ((close - low) / (high - low)) * 100 : 50;

  // %D is typically 3-period SMA of %K (simplified to same as K for now)
  // In production, you'd calculate this from multiple K values
  const d = k;

  return { k, d };
}

// Calculate ADX (Average Directional Index) from price data
function calculateADX(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) {
    return 25; // Neutral ADX when insufficient data (20-40 range is normal)
  }

  // Simplified ADX calculation based on price movement strength
  let sumDX = 0;
  for (let i = prices.length - period; i < prices.length - 1; i++) {
    const priceChange = Math.abs(prices[i + 1] - prices[i]);
    const priceRange = prices[i];
    sumDX += (priceChange / priceRange) * 100;
  }

  const adx = sumDX / period;
  return Math.max(0, Math.min(100, adx)); // Clamp between 0-100
}

// Calculate technical indicators from price data
function calculateTechnicalIndicators(prices: number[], volumes: number[]): TechnicalIndicators {
  const currentPrice = prices[prices.length - 1];
  const period14 = Math.min(14, prices.length);
  const period20 = Math.min(20, prices.length);

  // RSI calculation (simplified)
  const gains = [];
  const losses = [];
  for (let i = 1; i < period14 && i < prices.length; i++) {
    const change = prices[i] - prices[i-1];
    if (change > 0) {
      gains.push(change);
      losses.push(0);
    } else {
      gains.push(0);
      losses.push(Math.abs(change));
    }
  }
  const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / gains.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  // Simple moving averages
  const sma20 = prices.slice(-period20).reduce((a, b) => a + b, 0) / Math.min(period20, prices.length);
  const sma50 = prices.slice(-Math.min(50, prices.length)).reduce((a, b) => a + b, 0) / Math.min(50, prices.length);

  // EMA calculation (simplified)
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);

  // MACD
  const macd = ema12 - ema26;
  const signal = calculateEMA([macd], 9); // Simplified signal line
  const histogram = macd - signal;

  // Bollinger Bands
  const stdDev = Math.sqrt(prices.slice(-period20).reduce((sum, price) => {
    return sum + Math.pow(price - sma20, 2);
  }, 0) / Math.min(period20, prices.length));

  // NO RANDOM VALUES - Real calculations or null
  const stochastic = calculateStochastic(prices, 14);
  const adx = calculateADX(prices, 14);

  return {
    rsi: Math.max(0, Math.min(100, rsi)),
    macd: {
      macd,
      signal,
      histogram
    },
    bollinger: {
      upper: sma20 + (2 * stdDev),
      middle: sma20,
      lower: sma20 - (2 * stdDev)
    },
    volume: volumes[volumes.length - 1] || 1000000,
    sma20,
    sma50,
    ema12,
    ema26,
    stochastic: {
      k: stochastic.k,
      d: stochastic.d
    },
    atr: stdDev * 2, // Simplified ATR
    adx: adx
  };
}

function calculateEMA(prices: number[], period: number): number {
  const k = 2 / (period + 1);
  let ema = prices[0];
  for (let i = 1; i < prices.length; i++) {
    ema = (prices[i] * k) + (ema * (1 - k));
  }
  return ema;
}

// Generate signal reasoning based on indicators and ML prediction
function generateSignalReasoning(indicators: TechnicalIndicators, prediction: L5Prediction): string[] {
  const reasons: string[] = [];
  
  // RSI signals
  if (indicators.rsi < 30) {
    reasons.push(`RSI oversold (${indicators.rsi.toFixed(1)})`);
  } else if (indicators.rsi > 70) {
    reasons.push(`RSI overbought (${indicators.rsi.toFixed(1)})`);
  }

  // MACD signals
  if (indicators.macd.histogram > 0 && indicators.macd.macd > indicators.macd.signal) {
    reasons.push('MACD bullish crossover');
  } else if (indicators.macd.histogram < 0 && indicators.macd.macd < indicators.macd.signal) {
    reasons.push('MACD bearish crossover');
  }

  // Price vs moving averages
  const currentPrice = prediction.price;
  if (currentPrice > indicators.sma20 && indicators.sma20 > indicators.sma50) {
    reasons.push('Price above SMA20, bullish trend');
  } else if (currentPrice < indicators.sma20 && indicators.sma20 < indicators.sma50) {
    reasons.push('Price below SMA20, bearish trend');
  }

  // Bollinger Bands
  if (currentPrice < indicators.bollinger.lower) {
    reasons.push(`Support at Bollinger lower band (${indicators.bollinger.lower.toFixed(2)})`);
  } else if (currentPrice > indicators.bollinger.upper) {
    reasons.push(`Resistance at Bollinger upper band (${indicators.bollinger.upper.toFixed(2)})`);
  }

  // ML prediction confidence
  if (prediction.confidence > 0.8) {
    reasons.push(`High ML confidence (${(prediction.confidence * 100).toFixed(1)}%)`);
  }

  // Volume analysis
  if (indicators.volume > 1500000) {
    reasons.push('Volume spike detected');
  } else if (indicators.volume < 500000) {
    reasons.push('Low volume period');
  }

  return reasons.length > 0 ? reasons : ['Mixed signals', 'Awaiting confirmation'];
}

// Convert L5 prediction to trading signal
function convertL5PredictionToSignal(prediction: L5Prediction, indicators: TechnicalIndicators): TradingSignal {
  const signalType = prediction.action?.toLowerCase() || 'hold';
  let type: 'BUY' | 'SELL' | 'HOLD';
  
  if (signalType.includes('buy') || prediction.prediction > 0.6) {
    type = 'BUY';
  } else if (signalType.includes('sell') || prediction.prediction < 0.4) {
    type = 'SELL';
  } else {
    type = 'HOLD';
  }

  const confidence = Math.max(50, Math.min(95, prediction.confidence * 100));
  const price = prediction.price;
  
  // Calculate stop loss and take profit based on ATR
  const atr = indicators.atr;
  let stopLoss, takeProfit;
  
  if (type === 'BUY') {
    stopLoss = price - (atr * 2);
    takeProfit = price + (atr * 3);
  } else if (type === 'SELL') {
    stopLoss = price + (atr * 2);
    takeProfit = price - (atr * 3);
  }

  const reasoning = generateSignalReasoning(indicators, prediction);
  
  // Calculate risk score (1-10)
  const volatilityRisk = Math.min(5, (atr / price) * 1000); // Normalized volatility
  const confidenceRisk = (100 - confidence) / 20; // Risk based on confidence
  const rsiRisk = Math.abs(indicators.rsi - 50) / 25; // Risk from extreme RSI
  const riskScore = Math.max(1, Math.min(10, volatilityRisk + confidenceRisk + rsiRisk));

  // Expected return based on take profit vs stop loss
  const expectedReturn = type === 'HOLD' ? 0 : 
    type === 'BUY' ? ((takeProfit! - price) / price) :
    ((price - takeProfit!) / price);

  // NO RANDOM VALUES - Real calculations or null
  // Generate deterministic ID from timestamp and price (not random)
  const idHash = `${prediction.timestamp}_${price.toFixed(2)}_${type}`.split('').reduce((hash, char) => {
    return ((hash << 5) - hash) + char.charCodeAt(0);
  }, 0).toString(36);

  return {
    id: `sig_${Date.now()}_${idHash}`,
    timestamp: prediction.timestamp,
    type,
    confidence,
    price,
    stopLoss,
    takeProfit,
    reasoning,
    riskScore,
    expectedReturn,
    timeHorizon: confidence > 80 ? '15-30 min' : '5-15 min',
    modelSource: 'L5_PPO_LSTM_v2.1',
    latency: 0, // NO FAKE LATENCY - Real latency measured at API level
    technicalIndicators: indicators,
    mlPrediction: prediction
  };
}

/**
 * Trading Signals from Database
 *
 * Fetches real trading signals from dw.fact_rl_inference table.
 * Supports both backtest and out-of-sample data tracking.
 */

// Backtest end date - signals before this are backtest, after are out-of-sample
const BACKTEST_END_DATE = new Date('2025-09-30T00:00:00Z');

async function handler(request: NextRequest) {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '50');
    const modelId = searchParams.get('model_id') || 'ppo_v1';

    const client = await pool.connect();

    try {
      // FIRST: the PRODUCTION strategy's signals (H5 Smart Simple executions — entry/exit
      // markers from the restored DB). The RL sources below are legacy: after a cold boot
      // trades_history is empty (regenerable, not backed up) and the chart showed
      // "Signal source unavailable" for the strategy that IS live (found 2026-07-07).
      try {
        const h5 = await client.query(
          `
          SELECT id, entry_timestamp AS timestamp, direction, entry_price AS price,
                 'entry' AS signal_type, confidence_tier, exit_reason
          FROM forecast_h5_executions
          WHERE entry_timestamp IS NOT NULL
          UNION ALL
          SELECT id, exit_timestamp AS timestamp, -direction AS direction, exit_price AS price,
                 'exit' AS signal_type, confidence_tier, exit_reason
          FROM forecast_h5_executions
          WHERE exit_timestamp IS NOT NULL
          ORDER BY timestamp DESC
          LIMIT $1
          `,
          [limit],
        );
        if (h5.rows.length > 0) {
          const signals: TradingSignal[] = h5.rows.map((row) => {
            const isEntry = row.signal_type === 'entry';
            const dir = Number(row.direction);
            const type: 'BUY' | 'SELL' | 'HOLD' = dir > 0 ? 'BUY' : dir < 0 ? 'SELL' : 'HOLD';
            return {
              id: `h5_${row.id}_${row.signal_type}`,
              timestamp: row.timestamp,
              type,
              confidence: row.confidence_tier === 'HIGH' ? 85 : row.confidence_tier === 'MEDIUM' ? 70 : 60,
              price: parseFloat(row.price || '0'),
              reasoning: [
                'Estrategia: Smart Simple v2.0 (H5 semanal)',
                `Señal: ${isEntry ? 'Entrada' : 'Salida'}`,
                ...(isEntry ? [] : [`Motivo: ${row.exit_reason ?? 'n/a'}`]),
              ],
              riskScore: 3,
              expectedReturn: 0,
              timeHorizon: 'semanal',
              modelSource: 'smart_simple_v11',
              latency: 0,
              dataType: 'out_of_sample',
              model_id: 'smart_simple_v11',
            };
          });
          return NextResponse.json({
            success: true,
            signals,
            metadata: { source: 'forecast_h5_executions', model_id: modelId,
                        count: signals.length, latency: measureLatency(startTime) },
          }, { headers: { 'Cache-Control': 'no-store, max-age=0' } });
        }
      } catch (h5err) {
        console.warn('[TradingSignals] H5 source unavailable, falling back:', (h5err as Error).message);
      }

      // Legacy RL sources — try trades_history (paper trading results)
      const tradesResult = await client.query(
        `
        SELECT
          id,
          entry_time as timestamp,
          side as action,
          0.75 as confidence,
          entry_price as price,
          CASE WHEN side = 'LONG' THEN 0.8 ELSE -0.8 END as action_raw,
          model_id,
          'entry' as signal_type
        FROM public.trades_history
        WHERE model_id = $1
        UNION ALL
        SELECT
          id,
          exit_time as timestamp,
          CASE WHEN side = 'LONG' THEN 'EXIT_LONG' ELSE 'EXIT_SHORT' END as action,
          0.75 as confidence,
          exit_price as price,
          0 as action_raw,
          model_id,
          'exit' as signal_type
        FROM public.trades_history
        WHERE model_id = $1 AND exit_time IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT $2
        `,
        [modelId, limit]
      );

      // If trades_history has data, use it
      if (tradesResult.rows.length > 0) {
        const signals: TradingSignal[] = tradesResult.rows.map((row, index) => {
          const timestamp = new Date(row.timestamp);
          const isBacktest = timestamp <= BACKTEST_END_DATE;

          // Map action to signal type
          let type: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
          const action = row.action?.toUpperCase() || '';
          if (action === 'LONG' || action === 'BUY') {
            type = 'BUY';
          } else if (action === 'SHORT' || action === 'SELL') {
            type = 'SELL';
          } else if (action.includes('EXIT')) {
            type = 'HOLD'; // Exit signals shown as HOLD
          }

          const confidence = 75;
          const price = parseFloat(row.price || '0');

          return {
            id: `sig_${row.id}_${row.signal_type}`,
            timestamp: row.timestamp,
            type,
            confidence,
            price,
            reasoning: [
              `Model: ${row.model_id}`,
              `Signal: ${row.signal_type === 'entry' ? 'Entry' : 'Exit'}`,
              `Action: ${row.action}`,
              isBacktest ? 'Backtest signal' : 'Paper trading signal'
            ],
            riskScore: 3,
            expectedReturn: type === 'BUY' ? 0.015 : type === 'SELL' ? -0.015 : 0,
            timeHorizon: '5-60 min',
            modelSource: row.model_id || 'PPO',
            latency: 0,
            dataType: isBacktest ? 'backtest' : 'out_of_sample',
            model_id: row.model_id,
          };
        });

        const latency = measureLatency(startTime);
        return NextResponse.json({
          success: true,
          signals,
          metadata: {
            source: 'trades_history',
            model_id: modelId,
            count: signals.length,
            latency,
          }
        }, {
          headers: { 'Cache-Control': 'no-store, max-age=0' }
        });
      }

      // Fallback: dw.fact_rl_inference — schema per init-scripts (cold-boot truth):
      // (id, timestamp, bar_number, action, observation, model_version, inference_time_ms).
      // The old query referenced pre-rebuild columns (inference_id/timestamp_utc/…) and
      // SQL-errored into "Signal source unavailable" after every cold boot (2026-07-07).
      let result: { rows: Array<Record<string, unknown>> } = { rows: [] };
      try {
        result = await client.query(
          `
          SELECT
            id,
            timestamp,
            action::text as action,
            0.5 as confidence,
            0 as price,
            0 as action_raw,
            model_version as model_id
          FROM dw.fact_rl_inference
          WHERE model_version = $1
          ORDER BY timestamp DESC
          LIMIT $2
          `,
          [modelId, limit]
        );
      } catch (rlErr) {
        // No RL data / schema drift → clean empty (never fabricate, never scare the UI).
        console.warn('[TradingSignals] RL fallback unavailable:', (rlErr as Error).message);
        return NextResponse.json({
          success: true,
          signals: [],
          metadata: { source: 'empty', model_id: modelId, count: 0,
                      latency: measureLatency(startTime) },
        }, { headers: { 'Cache-Control': 'no-store, max-age=0' } });
      }

      // Transform database rows to TradingSignal format
      const signals: TradingSignal[] = result.rows.map((row, index) => {
        const timestamp = new Date(row.timestamp);
        const isBacktest = timestamp <= BACKTEST_END_DATE;

        // Map action to signal type
        let type: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
        const action = row.action?.toLowerCase() || '';
        if (action === 'long' || action === 'buy' || row.action_raw > 0.5) {
          type = 'BUY';
        } else if (action === 'short' || action === 'sell' || row.action_raw < -0.5) {
          type = 'SELL';
        }

        const confidence = parseFloat(row.confidence || '0.5') * 100;
        const price = parseFloat(row.price || '0');

        return {
          id: row.id || `sig_${index}`,
          timestamp: row.timestamp,
          type,
          confidence: Math.round(confidence),
          price,
          reasoning: [
            `Model: ${row.model_id}`,
            `Action: ${row.action_discretized || 'Unknown'}`,
            isBacktest ? 'Backtest signal' : 'Out-of-sample signal'
          ],
          riskScore: Math.round((1 - (confidence / 100)) * 10),
          expectedReturn: type === 'BUY' ? 0.015 : type === 'SELL' ? -0.015 : 0,
          timeHorizon: '5-15 min',
          modelSource: row.model_id || 'PPO_V19',
          latency: 0,
          dataType: isBacktest ? 'backtest' : 'out_of_sample',
          model_id: row.model_id,
        };
      });

      const latency = measureLatency(startTime);

      // Return in expected format with success and signals fields
      return NextResponse.json({
        success: true,
        signals,
        metadata: {
          source: 'database',
          model_id: modelId,
          count: signals.length,
          latency,
        }
      }, {
        headers: { 'Cache-Control': 'no-store, max-age=0' }
      });

    } finally {
      client.release();
    }

  } catch (error: any) {
    console.error('[TradingSignals] Database error:', error.message);
    const latency = measureLatency(startTime);

    // NEVER fabricate signals on failure — the live chart must not render
    // simulated data (audit A5-01). Surface an explicit error/empty state so
    // the UI shows "unavailable" instead of fake trades.
    const modelId = request.nextUrl.searchParams.get('model_id') || 'ppo_v1';
    return NextResponse.json({
      success: false,
      signals: [],
      error: 'Signal source unavailable',
      metadata: {
        source: 'error',
        model_id: modelId,
        count: 0,
        latency,
      }
    }, {
      status: 503,
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });
  }
}


// SECURITY: Protect endpoint with authentication
// Allow admin, trader, and viewer roles to access trading signals
export const GET = withAuth(handler, {
  requiredRole: ['admin', 'trader', 'viewer'],
});