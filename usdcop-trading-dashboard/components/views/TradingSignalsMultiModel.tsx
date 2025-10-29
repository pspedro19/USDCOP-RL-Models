'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';
import { Activity, RefreshCw } from 'lucide-react';
import EquityCurveChart from './EquityCurveChart';
import ModelSignalCard from './ModelSignalCard';
import PerformanceMetricsGrid from './PerformanceMetricsGrid';
// import OrderBook from './OrderBookDisabled'; // REMOVED - Component completely eliminated
import TradeHistory from './TradeHistory';

// ============================================================================
// INTERFACES
// ============================================================================

interface StrategyEquityCurve {
  timestamp: string;
  RL_PPO: number;
  ML_LGBM: number;
  ML_XGB: number | null;
  LLM_CLAUDE: number | null;
  PORTFOLIO: number;
  CAPITAL_INICIAL: number;
}

interface ModelSignal {
  signal_id: number;
  timestamp_utc: string;
  strategy_code: string;
  strategy_name: string;
  strategy_type: 'RL' | 'ML' | 'LLM';
  signal: 'long' | 'short' | 'flat' | 'close';
  side: 'buy' | 'sell' | 'hold';
  size: number;
  confidence: number;
  entry_price: number;
  stop_loss: number | null;
  take_profit: number | null;
  risk_usd: number;
  notional_usd: number;
  reasoning: string;
  features_snapshot: Record<string, number> | null;
}

interface StrategyPerformance {
  strategy_code: string;
  strategy_name: string;
  strategy_type: 'RL' | 'ML' | 'LLM';
  current_equity: number;
  cash_balance: number;
  open_positions: number;
  total_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  today_return_pct: number | null;
  today_trades: number | null;
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function TradingSignalsMultiModel() {
  // ========== STATE ==========
  const [equityCurves, setEquityCurves] = useState<StrategyEquityCurve[]>([]);
  const [currentSignals, setCurrentSignals] = useState<Record<string, ModelSignal>>({});
  const [performance, setPerformance] = useState<StrategyPerformance[]>([]);
  const [loading, setLoading] = useState(true);
  const [isInitialLoad, setIsInitialLoad] = useState(true); // Track first load vs refresh
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // ========== DATA FETCHING ==========
  const fetchEquityCurves = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout

      const response = await fetch('/api/trading/equity-curves/multi-strategy?hours=24&resolution=5m', {
        cache: 'no-store',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setEquityCurves(data.equity_curves || []);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.warn('[TradingSignals] Equity curves timed out - skipping');
      } else {
        console.error('[TradingSignals] Error fetching equity curves:', err);
      }
      setEquityCurves([]); // Set empty if fails
    }
  }, []);

  const fetchCurrentSignals = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout

      const response = await fetch('/api/trading/signals/multi-strategy', {
        cache: 'no-store',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Group latest signal by strategy
      const signalsByStrategy: Record<string, ModelSignal> = {};
      (data.signals || []).forEach((sig: ModelSignal) => {
        signalsByStrategy[sig.strategy_code] = sig;
      });

      setCurrentSignals(signalsByStrategy);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.warn('[TradingSignals] Signals timed out - skipping');
      } else {
        console.error('[TradingSignals] Error fetching signals:', err);
      }
    }
  }, []);

  const fetchPerformance = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout

      const response = await fetch('/api/trading/performance/multi-strategy?period_days=30', {
        cache: 'no-store',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setPerformance(data.strategies || []);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.warn('[TradingSignals] Performance timed out - skipping');
      } else {
        console.error('[TradingSignals] Error fetching performance:', err);
      }
      setPerformance([]); // Set empty if fails
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      await Promise.all([
        fetchEquityCurves(),
        fetchCurrentSignals(),
        fetchPerformance()
      ]);

      setLastUpdate(new Date());
      setIsInitialLoad(false); // Mark initial load as complete
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error fetching data');
    } finally {
      setLoading(false);
    }
  }, [fetchEquityCurves, fetchCurrentSignals, fetchPerformance]);

  // ========== EFFECTS ==========
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      console.log('[TradingSignals] Auto-refreshing data...');
      refreshAll();
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshAll]);

  // ========== LOADING STATE (ONLY FIRST LOAD) ==========
  if (isInitialLoad && loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-cyan-500/20">
        <div className="text-center">
          <Activity className="h-10 w-10 mx-auto mb-4 animate-pulse text-cyan-500" />
          <p className="text-cyan-500 font-mono text-lg">Loading Multi-Model Trading Signals...</p>
          <p className="text-slate-500 font-mono text-sm mt-2">Fetching equity curves, signals, and performance data</p>
        </div>
      </div>
    );
  }

  // ========== ERROR STATE ==========
  if (error && equityCurves.length === 0 && performance.length === 0) {
    return (
      <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-400">
          <RefreshCw className="h-5 w-5" />
          <div>
            <p className="font-mono font-bold">Error Loading Trading Signals</p>
            <p className="text-sm mt-1">{error}</p>
            <button
              onClick={refreshAll}
              className="mt-3 px-4 py-2 bg-red-900/40 hover:bg-red-900/60 rounded text-sm transition-all"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ========== MAIN RENDER ==========
  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen max-w-[1800px] mx-auto">

      {/* ========== HEADER ========== */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-900 border border-cyan-500/20 rounded-xl p-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-cyan-500 font-mono">
              MULTI-MODEL TRADING SIGNALS
            </h1>
            <p className="text-slate-400 mt-2">
              Real-time signals from RL, ML, and LLM strategies • Live equity tracking
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Badge className="bg-green-500/20 text-green-400 border-green-500/50 animate-pulse">
              LIVE
            </Badge>

            {lastUpdate && (
              <div className="text-xs text-slate-400 font-mono">
                Updated: {lastUpdate.toLocaleTimeString()}
              </div>
            )}

            <button
              onClick={refreshAll}
              className="p-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-600/50 text-slate-300 transition-all"
              title="Refresh data"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-1 rounded-lg border text-xs font-mono ${
                autoRefresh
                  ? 'bg-green-500/20 border-green-500/50 text-green-400'
                  : 'bg-slate-800/50 border-slate-600/50 text-slate-400'
              }`}
            >
              Auto {autoRefresh ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>
      </motion.div>

      {/* ========== EQUITY CURVE CHART ========== */}
      <EquityCurveChart data={equityCurves} />

      {/* ========== ORDER BOOK ========== */}
      {/* OrderBook component removed - was displaying simulated Math.random() data */}
      {/* Alternative: VolumeProfileChart can be added here if needed */}

      {/* ========== CURRENT SIGNALS (Model Cards) ========== */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <ModelSignalCard
          strategyCode="RL_PPO"
          strategyName="RL PPO Agent"
          signal={currentSignals['RL_PPO']}
          color="blue"
        />

        <ModelSignalCard
          strategyCode="ML_LGBM"
          strategyName="ML LightGBM"
          signal={currentSignals['ML_LGBM']}
          color="purple"
        />

        <ModelSignalCard
          strategyCode="ML_XGB"
          strategyName="ML XGBoost"
          signal={currentSignals['ML_XGB']}
          color="orange"
        />

        <ModelSignalCard
          strategyCode="LLM_CLAUDE"
          strategyName="LLM Claude"
          signal={currentSignals['LLM_CLAUDE']}
          color="amber"
        />

        <ModelSignalCard
          strategyCode="ENSEMBLE"
          strategyName="Ensemble"
          signal={currentSignals['ENSEMBLE']}
          color="slate"
        />
      </div>

      {/* ========== PERFORMANCE METRICS GRID ========== */}
      <PerformanceMetricsGrid strategies={performance} />

      {/* ========== TRADE HISTORY ========== */}
      <TradeHistory />

    </div>
  );
}
