'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp, TrendingDown, Activity, Clock, Target, Zap, 
  BarChart3, AlertTriangle, Play, Pause, RotateCcw, Download,
  Signal, Database, Globe, Layers, Eye, EyeOff
} from 'lucide-react';

// KPI Mock Data Generator
const generateKPIData = () => ({
  session: {
    timeRange: '08:00–12:55',
    pnlIntraday: 1247.85,
    pnlPercent: 2.34,
    tradesEpisode: 7,
    targetRange: '2–10',
    avgHolding: 12,
    holdingRange: '5–25 barras',
    actionBalance: { sell: 45, buy: 55 },
    drawdownIntraday: -2.1
  },
  execution: {
    vwapVsFill: 1.2, // bps
    spreadEffective: 4.8, // bps per hour
    slippage: 2.1, // bps per hour
    turnCost: 8.5, // bps per trade
    fillRatio: 94.2 // %
  },
  latency: {
    p50: 45, // ms
    p95: 78, // ms
    p99: 95, // ms
    onnxP99: 12 // ms
  },
  marketState: {
    status: 'OPEN',
    latency: '<5ms',
    clockSkew: '0.2ms',
    lastUpdate: new Date().toLocaleTimeString()
  }
});

const EnhancedTradingTerminal: React.FC = () => {
  const [kpiData, setKpiData] = useState(generateKPIData());
  const [isReplaying, setIsReplaying] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Update data every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setKpiData(generateKPIData());
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
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`p-2 rounded-lg transition-all ${showAdvanced ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-800/50 text-slate-400 hover:text-white'}`}
          >
            {showAdvanced ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
          
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

        {/* Advanced View */}
        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/30"
            >
              <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-orange-400" />
                Vista Avanzada
              </h3>
              
              <div className="text-center text-slate-400 py-8">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 text-slate-600" />
                <p>Métricas avanzadas, gráficos detallados y análisis en tiempo real</p>
                <p className="text-sm mt-2">Chart integrado, order book, y controles de replay</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EnhancedTradingTerminal;