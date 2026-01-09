'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { motion } from 'framer-motion';
import { Activity, RefreshCw, Radio, Wifi, WifiOff, BarChart3, GitCompare, Trophy } from 'lucide-react';
import EquityCurveChart from './EquityCurveChart';
import ModelSignalCard from './ModelSignalCard';
import PerformanceMetricsGrid from './PerformanceMetricsGrid';
import TradeHistory from './TradeHistory';
import LiveVsBacktestComparison from './LiveVsBacktestComparison';
import ModelComparisonPanel from './ModelComparisonPanel';
import { useEquityCurveStream } from '@/hooks/useEquityCurveStream';

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
  const [currentSignals, setCurrentSignals] = useState<Record<string, ModelSignal>>({});
  const [performance, setPerformance] = useState<StrategyPerformance[]>([]);
  const [loading, setLoading] = useState(true);
  const [isInitialLoad, setIsInitialLoad] = useState(true); // Track first load vs refresh
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true); // Enable SSE by default

  // View toggle for different dashboard sections
  type DashboardView = 'signals' | 'comparison' | 'liveBacktest';
  const [activeView, setActiveView] = useState<DashboardView>('signals');

  // ========== SSE STREAMING FOR EQUITY CURVES ==========
  const {
    data: streamingEquityData,
    latestValues: streamingLatestValues,
    isConnected: streamingConnected,
    connectionType,
    lastUpdate: streamingLastUpdate,
    error: streamingError,
    reconnect: reconnectStream
  } = useEquityCurveStream({
    strategies: ['RL_PPO', 'ML_LGBM', 'ML_XGB', 'LLM_CLAUDE'],
    enabled: useStreaming,
    fallbackInterval: 10000
  });

  // Transform streaming data to chart format
  const equityCurves = useMemo(() => {
    if (!useStreaming || Object.keys(streamingEquityData).length === 0) {
      return [];
    }

    // Get all unique timestamps
    const allTimestamps = new Set<string>();
    Object.values(streamingEquityData).forEach(points => {
      points.forEach(p => allTimestamps.add(p.timestamp));
    });

    // Sort timestamps
    const sortedTimestamps = Array.from(allTimestamps).sort();

    // Build chart data
    return sortedTimestamps.map(timestamp => {
      const point: StrategyEquityCurve = {
        timestamp,
        RL_PPO: 10000,
        ML_LGBM: 10000,
        ML_XGB: null,
        LLM_CLAUDE: null,
        PORTFOLIO: 40000,
        CAPITAL_INICIAL: 10000
      };

      // Find data for each strategy at this timestamp
      Object.entries(streamingEquityData).forEach(([code, points]) => {
        const dataPoint = points.find(p => p.timestamp === timestamp);
        if (dataPoint) {
          if (code === 'RL_PPO') point.RL_PPO = dataPoint.equity_value;
          else if (code === 'ML_LGBM') point.ML_LGBM = dataPoint.equity_value;
          else if (code === 'ML_XGB') point.ML_XGB = dataPoint.equity_value;
          else if (code === 'LLM_CLAUDE') point.LLM_CLAUDE = dataPoint.equity_value;
        }
      });

      // Calculate portfolio as sum of individual strategies
      point.PORTFOLIO = point.RL_PPO + point.ML_LGBM + (point.ML_XGB || 10000) + (point.LLM_CLAUDE || 10000);

      return point;
    });
  }, [useStreaming, streamingEquityData]);

  // Fallback: Fetch equity curves via polling when streaming is disabled
  const [polledEquityCurves, setPolledEquityCurves] = useState<StrategyEquityCurve[]>([]);

  const fetchEquityCurves = useCallback(async () => {
    if (useStreaming) return; // Skip if using streaming

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const response = await fetch('/api/trading/equity-curves/multi-strategy?hours=24&resolution=5m', {
        cache: 'no-store',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setPolledEquityCurves(data.equity_curves || []);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.warn('[TradingSignals] Equity curves timed out - skipping');
      } else {
        console.error('[TradingSignals] Error fetching equity curves:', err);
      }
      setPolledEquityCurves([]);
    }
  }, [useStreaming]);

  // Get final equity curves based on mode
  const finalEquityCurves = useMemo(() => {
    return useStreaming ? equityCurves : polledEquityCurves;
  }, [useStreaming, equityCurves, polledEquityCurves]);

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
            {/* Streaming Status Badge */}
            {useStreaming && (
              <Badge
                className={`flex items-center gap-1.5 ${
                  streamingConnected
                    ? connectionType === 'sse'
                      ? 'bg-green-500/20 text-green-400 border-green-500/50'
                      : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
                    : 'bg-red-500/20 text-red-400 border-red-500/50'
                }`}
              >
                {streamingConnected ? (
                  connectionType === 'sse' ? (
                    <>
                      <Radio className="h-3 w-3 animate-pulse" />
                      SSE LIVE
                    </>
                  ) : (
                    <>
                      <Wifi className="h-3 w-3" />
                      POLLING
                    </>
                  )
                ) : (
                  <>
                    <WifiOff className="h-3 w-3" />
                    DISCONNECTED
                  </>
                )}
              </Badge>
            )}

            {!useStreaming && (
              <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                POLLING
              </Badge>
            )}

            {(streamingLastUpdate || lastUpdate) && (
              <div className="text-xs text-slate-400 font-mono">
                Updated: {new Date(streamingLastUpdate || lastUpdate!).toLocaleTimeString()}
              </div>
            )}

            <button
              onClick={useStreaming ? reconnectStream : refreshAll}
              className="p-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-600/50 text-slate-300 transition-all"
              title={useStreaming ? 'Reconnect stream' : 'Refresh data'}
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>

            {/* Streaming Toggle */}
            <button
              onClick={() => setUseStreaming(!useStreaming)}
              className={`px-3 py-1 rounded-lg border text-xs font-mono ${
                useStreaming
                  ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400'
                  : 'bg-slate-800/50 border-slate-600/50 text-slate-400'
              }`}
              title="Toggle SSE streaming vs polling"
            >
              Stream {useStreaming ? 'ON' : 'OFF'}
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

        {/* ========== VIEW TABS ========== */}
        <div className="mt-6 flex items-center gap-2 border-t border-slate-700/50 pt-4">
          <button
            onClick={() => setActiveView('signals')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-mono transition-all ${
              activeView === 'signals'
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50'
                : 'bg-slate-800/50 text-slate-400 border border-slate-600/50 hover:text-slate-200'
            }`}
          >
            <Activity className="h-4 w-4" />
            Trading Signals
          </button>
          <button
            onClick={() => setActiveView('comparison')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-mono transition-all ${
              activeView === 'comparison'
                ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50'
                : 'bg-slate-800/50 text-slate-400 border border-slate-600/50 hover:text-slate-200'
            }`}
          >
            <Trophy className="h-4 w-4" />
            Model Arena
          </button>
          <button
            onClick={() => setActiveView('liveBacktest')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-mono transition-all ${
              activeView === 'liveBacktest'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50'
                : 'bg-slate-800/50 text-slate-400 border border-slate-600/50 hover:text-slate-200'
            }`}
          >
            <GitCompare className="h-4 w-4" />
            Live vs Backtest
          </button>
        </div>
      </motion.div>

      {/* ========== EQUITY CURVE CHART (Always visible) ========== */}
      <EquityCurveChart data={finalEquityCurves} />

      {/* ========== CONDITIONAL VIEW RENDERING ========== */}

      {/* VIEW: Trading Signals (Default) */}
      {activeView === 'signals' && (
        <>
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
        </>
      )}

      {/* VIEW: Model Comparison Arena */}
      {activeView === 'comparison' && (
        <ModelComparisonPanel showTitle={false} highlightBest={true} />
      )}

      {/* VIEW: Live vs Backtest Comparison */}
      {activeView === 'liveBacktest' && (
        <LiveVsBacktestComparison showAllModels={true} />
      )}

    </div>
  );
}
