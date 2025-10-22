'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp, TrendingDown, Activity, Clock, Target, Zap,
  BarChart3, AlertTriangle, Play, Pause, RotateCcw, Download,
  Signal, Database, Globe, Layers, Eye, EyeOff
} from 'lucide-react';
import RealDataTradingChart from '../charts/RealDataTradingChart';
import RealTimePriceDisplay from '../realtime/RealTimePriceDisplay';
import { useDbStats } from '@/hooks/useDbStats';
import { MarketDataService } from '@/lib/services/market-data-service';

// KPI Data Generator with dynamic session P&L
const generateKPIData = async () => {
  // Fetch session P&L from Analytics API
  let pnlIntraday = 0;
  let pnlPercent = 0;

  try {
    // ✅ Use Next.js API proxy (no direct backend calls)
    const response = await fetch(`/api/analytics/session-pnl?symbol=USDCOP`);

    if (response.ok) {
      const data = await response.json();
      pnlIntraday = data.session_pnl || 0;
      pnlPercent = data.session_pnl_percent || 0;
    }
  } catch (error) {
    console.error('Error fetching session P&L:', error);
  }

  return {
    session: {
      timeRange: '08:00–12:55',
      pnlIntraday, // ✅ From /api/analytics/session-pnl
      pnlPercent,  // ✅ From /api/analytics/session-pnl
      tradesEpisode: 0, // ⚠️ TODO: Add to /api/analytics/session-pnl endpoint
      targetRange: '2–10',
      avgHolding: 0, // ⚠️ TODO: Add to /api/analytics/session-pnl endpoint
      holdingRange: '5–25 barras',
      actionBalance: { sell: 0, buy: 0 }, // ⚠️ TODO: Add to /api/analytics/session-pnl endpoint
      drawdownIntraday: 0 // ⚠️ TODO: Add to /api/analytics/session-pnl endpoint
    },
    execution: {
      vwapVsFill: 0, // ⚠️ TODO: Create /api/analytics/execution-metrics endpoint
      spreadEffective: 0, // ⚠️ TODO: Create /api/analytics/execution-metrics endpoint
      slippage: 0, // ⚠️ TODO: Create /api/analytics/execution-metrics endpoint
      turnCost: 0, // ⚠️ TODO: Create /api/analytics/execution-metrics endpoint
      fillRatio: 0 // ⚠️ TODO: Create /api/analytics/execution-metrics endpoint
    },
    latency: {
      p50: 0, // ⚠️ TODO: Create /api/analytics/latency-metrics endpoint
      p95: 0, // ⚠️ TODO: Create /api/analytics/latency-metrics endpoint
      p99: 0, // ⚠️ TODO: Create /api/analytics/latency-metrics endpoint
      onnxP99: 0 // ⚠️ TODO: Create /api/analytics/latency-metrics endpoint
    },
    marketState: {
      status: 'LOADING',
      latency: '-',
      clockSkew: '-',
      lastUpdate: '-'
    }
  };

  // Fetch market status from API (8AM-12:55PM COT dynamic check)
  try {
    const health = await MarketDataService.checkAPIHealth();
    if (health.market_status) {
      return {
        ...baseData,
        marketState: {
          status: health.market_status.is_open ? 'OPEN' : 'CLOSED',
          latency: health.response_time_ms ? `${health.response_time_ms}ms` : '-',
          clockSkew: '-', // ⚠️ TODO: Add clock_skew to Trading API health endpoint
          lastUpdate: new Date().toLocaleTimeString()
        }
      };
    }
  } catch (error) {
    console.error('[EnhancedTradingTerminal] Error fetching market status:', error);
  }

  return baseData;
};

const EnhancedTradingTerminal: React.FC = () => {
  const [kpiData, setKpiData] = useState<any>({
    session: { timeRange: '08:00–12:55', pnlIntraday: 0, pnlPercent: 0, tradesEpisode: 0, targetRange: '2–10', avgHolding: 0, holdingRange: '5–25 barras', actionBalance: { sell: 0, buy: 0 }, drawdownIntraday: 0 },
    execution: { vwapVsFill: 0, spreadEffective: 0, slippage: 0, turnCost: 0, fillRatio: 0 },
    latency: { p50: 0, p95: 0, p99: 0, onnxP99: 0 },
    marketState: { status: 'LOADING', latency: '-', clockSkew: '-', lastUpdate: '-' }
  });
  const [isReplaying, setIsReplaying] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(true);
  const { stats: dbStats } = useDbStats(60000); // Refresh every 60 seconds

  // Load initial data and update every 5 seconds
  useEffect(() => {
    const loadData = async () => {
      const data = await generateKPIData();
      setKpiData(data);
    };

    // Load immediately
    loadData();

    // Update every 5 seconds
    const interval = setInterval(() => {
      loadData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (value: number, threshold: number, reverse = false) => {
    const isGood = reverse ? value < threshold : value > threshold;
    return isGood ? 'text-green-400' : 'text-red-400';
  };

  const getBadgeColor = (value: number, good: number, warning: number) => {
    if (value <= good) return 'bg-green-500/20 text-green-400 border-green-500/30';
    if (value <= warning) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    return 'bg-red-500/20 text-red-400 border-red-500/30';
  };

  return (
    <div className="h-full flex flex-col bg-slate-950/50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-bold text-white">Trading Terminal</h2>
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${kpiData.marketState.status === 'OPEN' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-slate-400">{kpiData.marketState.status}</span>
            <span className="text-slate-500">•</span>
            <span className="text-cyan-400">{kpiData.marketState.latency}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          
          <button
            onClick={() => setIsReplaying(!isReplaying)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
              isReplaying 
                ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                : 'bg-green-500/20 text-green-400 border border-green-500/30'
            }`}
          >
            {isReplaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span className="text-sm font-medium">Replay</span>
          </button>

          <button className="flex items-center gap-2 px-3 py-2 bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-500/30 transition-all">
            <Download className="w-4 h-4" />
            <span className="text-sm font-medium">Export</span>
          </button>
        </div>
      </div>

      {/* KPI Dashboard */}
      <div className="flex-1 p-4 space-y-4 overflow-y-auto">
        {/* Session KPIs */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-cyan-400" />
            KPIs Sesión ({kpiData.session.timeRange})
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">P&L Intradía</div>
              <div className={`text-xl font-bold ${kpiData.session.pnlIntraday >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {kpiData.session.pnlIntraday >= 0 ? '+' : ''}${kpiData.session.pnlIntraday.toFixed(2)}
              </div>
              <div className={`text-sm ${kpiData.session.pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ({kpiData.session.pnlPercent >= 0 ? '+' : ''}{kpiData.session.pnlPercent}%)
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Trades/Episodio</div>
              <div className="text-xl font-bold text-white">{kpiData.session.tradesEpisode}</div>
              <div className="text-sm text-slate-500">Target: {kpiData.session.targetRange}</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Avg Holding</div>
              <div className="text-xl font-bold text-white">{kpiData.session.avgHolding}</div>
              <div className="text-sm text-slate-500">{kpiData.session.holdingRange}</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Action Balance</div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1">
                  <TrendingDown className="w-4 h-4 text-red-400" />
                  <span className="text-red-400">{kpiData.session.actionBalance.sell}%</span>
                </div>
                <div className="text-slate-500">/</div>
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  <span className="text-green-400">{kpiData.session.actionBalance.buy}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <div className="bg-slate-800/50 rounded-lg p-3 w-full md:w-1/2">
              <div className="text-sm text-slate-400">Drawdown Intradía</div>
              <div className="text-xl font-bold text-red-400">{kpiData.session.drawdownIntraday}%</div>
            </div>
          </div>
        </div>

        {/* Execution & Costs */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-purple-400" />
            Ejecución/Costos
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">VWAP vs Fill</div>
              <div className="text-xl font-bold text-white">{kpiData.execution.vwapVsFill}</div>
              <div className="text-sm text-slate-500">bps</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Spread Efectivo</div>
              <div className="text-xl font-bold text-white">{kpiData.execution.spreadEffective}</div>
              <div className="text-sm text-slate-500">bps/hora</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Slippage</div>
              <div className="text-xl font-bold text-white">{kpiData.execution.slippage}</div>
              <div className="text-sm text-slate-500">bps/hora</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Turn Cost</div>
              <div className="text-xl font-bold text-white">{kpiData.execution.turnCost}</div>
              <div className="text-sm text-slate-500">bps/trade</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Fill Ratio</div>
              <div className={`text-xl font-bold ${getStatusColor(kpiData.execution.fillRatio, 90)}`}>
                {kpiData.execution.fillRatio}%
              </div>
            </div>
          </div>
        </div>

        {/* Latency */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Latencia
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">p50 Ejecución</div>
              <div className={`text-xl font-bold ${getBadgeColor(kpiData.latency.p50, 50, 80).includes('text-green') ? 'text-green-400' : getBadgeColor(kpiData.latency.p50, 50, 80).includes('text-yellow') ? 'text-yellow-400' : 'text-red-400'}`}>
                {kpiData.latency.p50}ms
              </div>
              <div className="text-sm text-slate-500">Target: &lt;100ms</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">p95 Ejecución</div>
              <div className={`text-xl font-bold ${getBadgeColor(kpiData.latency.p95, 80, 95).includes('text-green') ? 'text-green-400' : getBadgeColor(kpiData.latency.p95, 80, 95).includes('text-yellow') ? 'text-yellow-400' : 'text-red-400'}`}>
                {kpiData.latency.p95}ms
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">p99 Ejecución</div>
              <div className={`text-xl font-bold ${getBadgeColor(kpiData.latency.p99, 90, 100).includes('text-green') ? 'text-green-400' : getBadgeColor(kpiData.latency.p99, 90, 100).includes('text-yellow') ? 'text-yellow-400' : 'text-red-400'}`}>
                {kpiData.latency.p99}ms
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">ONNX p99</div>
              <div className={`text-xl font-bold ${getBadgeColor(kpiData.latency.onnxP99, 15, 20).includes('text-green') ? 'text-green-400' : getBadgeColor(kpiData.latency.onnxP99, 15, 20).includes('text-yellow') ? 'text-yellow-400' : 'text-red-400'}`}>
                {kpiData.latency.onnxP99}ms
              </div>
              <div className="text-sm text-slate-500">Target: &lt;20ms</div>
            </div>
          </div>
        </div>

        {/* Market State */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5 text-blue-400" />
            Market State
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Status</div>
              <div className={`text-xl font-bold ${kpiData.marketState.status === 'OPEN' ? 'text-green-400' : 'text-red-400'}`}>
                {kpiData.marketState.status}
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Latencia</div>
              <div className="text-xl font-bold text-cyan-400">{kpiData.marketState.latency}</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Clock Skew</div>
              <div className="text-xl font-bold text-white">{kpiData.marketState.clockSkew}</div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-sm text-slate-400">Última Actualización</div>
              <div className="text-xl font-bold text-white">{kpiData.marketState.lastUpdate}</div>
            </div>
          </div>
        </div>

        {/* Real-Time Price Display */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-orange-400" />
            Precio USD/COP en Tiempo Real
          </h3>
          <RealTimePriceDisplay symbol="USDCOP" />
        </div>

        {/* Real Trading Chart - SIEMPRE VISIBLE */}
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-orange-400" />
            Gráfico USD/COP - Datos Históricos Reales ({dbStats.totalRecords.toLocaleString()} registros)
          </h3>

          <RealDataTradingChart
            symbol="USDCOP"
            timeframe="5m"
            height={500}
            className="w-full"
          />
        </div>
      </div>
    </div>
  );
};

export default EnhancedTradingTerminal;