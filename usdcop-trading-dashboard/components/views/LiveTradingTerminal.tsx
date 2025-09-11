'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, TrendingUp, TrendingDown, Volume2, Clock, Target, 
  BarChart3, Zap, Signal, AlertCircle, Play, Pause, Square,
  Maximize2, Download, Share2, Camera, MousePointer, Crosshair,
  LineChart, CandlestickChart, AreaChart, Settings, Filter,
  Eye, EyeOff, Layers, Grid, Ruler, Move, RotateCcw, Trash2
} from 'lucide-react';

// Real-time market data simulation
const useRealTimeMarketData = () => {
  const [data, setData] = useState({
    price: 4150.25,
    change: 15.75,
    changePercent: 0.38,
    volume: 2847329,
    high24h: 4165.50,
    low24h: 3890.25,
    spread: 21.2,
    vwapError: 2.8,
    pegRate: 3.2,
    timestamp: new Date(),
    orderFlow: {
      buy: 58.3,
      sell: 41.7,
      imbalance: 16.6
    },
    technicals: {
      rsi: 68.3,
      macd: 0.245,
      bollinger: {
        upper: 4180.50,
        middle: 4125.75,
        lower: 4071.00
      },
      ema20: 4142.30,
      ema50: 4125.80,
      ema200: 4098.45
    }
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const priceChange = (Math.random() - 0.5) * 5;
        const newPrice = Math.max(3800, Math.min(4300, prev.price + priceChange));
        
        return {
          ...prev,
          price: newPrice,
          change: newPrice - prev.price,
          changePercent: ((newPrice - prev.price) / prev.price) * 100,
          volume: prev.volume + Math.floor(Math.random() * 10000),
          spread: 18 + Math.random() * 6,
          vwapError: Math.random() * 5,
          pegRate: Math.random() * 8,
          timestamp: new Date(),
          orderFlow: {
            buy: 40 + Math.random() * 40,
            sell: 40 + Math.random() * 40,
            imbalance: (Math.random() - 0.5) * 30
          },
          technicals: {
            ...prev.technicals,
            rsi: Math.max(20, Math.min(80, prev.technicals.rsi + (Math.random() - 0.5) * 2)),
            macd: prev.technicals.macd + (Math.random() - 0.5) * 0.02
          }
        };
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return data;
};

// Trading Session Status
const useTradingSession = () => {
  const [session, setSession] = useState({
    isActive: true,
    coverage: 95.8,
    hoursRemaining: '2h 35m',
    sessionType: 'Premium Window',
    nextEvent: 'Market Close',
    latency: {
      inference: 18,
      e2e: 87
    }
  });

  return session;
};

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'optimal' | 'warning' | 'critical';
  icon: React.ComponentType<any>;
  trend?: 'up' | 'down' | 'neutral';
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, value, subtitle, status, icon: Icon, trend 
}) => {
  const statusColors = {
    optimal: 'border-market-up text-market-up shadow-market-up',
    warning: 'border-fintech-purple-400 text-fintech-purple-400 shadow-glow-purple',
    critical: 'border-market-down text-market-down shadow-market-down'
  };

  const bgColors = {
    optimal: 'from-market-up/10 to-market-up/5',
    warning: 'from-fintech-purple-400/10 to-fintech-purple-400/5',
    critical: 'from-market-down/10 to-market-down/5'
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-6 rounded-xl border ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        {trend && (
          <div className={`${trend === 'up' ? 'text-market-up' : trend === 'down' ? 'text-market-down' : 'text-fintech-dark-400'}`}>
            {trend === 'up' && <TrendingUp className="w-4 h-4" />}
            {trend === 'down' && <TrendingDown className="w-4 h-4" />}
          </div>
        )}
      </div>
      
      <div className="text-2xl font-bold text-white mb-1">
        {typeof value === 'number' ? value.toFixed(2) : value}
      </div>
      <div className="text-xs text-fintech-dark-300">{subtitle}</div>
    </motion.div>
  );
};

const ChartToolbar: React.FC<{
  activeTools: string[];
  onToolToggle: (tool: string) => void;
  onTimeframeChange: (timeframe: string) => void;
  activeTimeframe: string;
}> = ({ activeTools, onToolToggle, onTimeframeChange, activeTimeframe }) => {
  const timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];
  
  const drawingTools = [
    { id: 'line', icon: MousePointer, label: 'Line' },
    { id: 'rect', icon: Move, label: 'Rectangle' },
    { id: 'circle', icon: Target, label: 'Circle' },
    { id: 'fib', icon: Ruler, label: 'Fibonacci' },
    { id: 'trend', icon: TrendingUp, label: 'Trendline' },
  ];

  const indicators = [
    { id: 'bollinger', label: 'Bollinger Bands' },
    { id: 'ema', label: 'EMA 20/50/200' },
    { id: 'rsi', label: 'RSI' },
    { id: 'macd', label: 'MACD' },
    { id: 'volume', label: 'Volume Profile' },
  ];

  return (
    <div className="flex items-center justify-between p-4 glass-surface rounded-xl border border-fintech-dark-700">
      {/* Timeframes */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Timeframe:</span>
        <div className="flex items-center gap-1 bg-fintech-dark-800 rounded-lg p-1">
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange(tf)}
              className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                activeTimeframe === tf 
                  ? 'bg-fintech-cyan-500 text-white shadow-glow-cyan' 
                  : 'text-fintech-dark-400 hover:text-white'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Drawing Tools */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Tools:</span>
        <div className="flex items-center gap-1">
          {drawingTools.map((tool) => (
            <button
              key={tool.id}
              onClick={() => onToolToggle(tool.id)}
              className={`p-2 rounded-lg transition-all ${
                activeTools.includes(tool.id)
                  ? 'bg-fintech-cyan-500/20 text-fintech-cyan-400 border border-fintech-cyan-500/30'
                  : 'bg-fintech-dark-800/50 text-fintech-dark-400 hover:text-white'
              }`}
              title={tool.label}
            >
              <tool.icon className="w-4 h-4" />
            </button>
          ))}
        </div>
      </div>

      {/* Indicators */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-fintech-dark-300 mr-2">Indicators:</span>
        <div className="flex items-center gap-1">
          {indicators.map((indicator) => (
            <button
              key={indicator.id}
              onClick={() => onToolToggle(indicator.id)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                activeTools.includes(indicator.id)
                  ? 'bg-fintech-purple-400/20 text-fintech-purple-400 border border-fintech-purple-400/30'
                  : 'bg-fintech-dark-800/50 text-fintech-dark-400 hover:text-white'
              }`}
            >
              {indicator.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default function LiveTradingTerminal() {
  const marketData = useRealTimeMarketData();
  const session = useTradingSession();
  const [activeTools, setActiveTools] = useState(['bollinger', 'ema', 'volume']);
  const [activeTimeframe, setActiveTimeframe] = useState('M5');
  const [isPlaying, setIsPlaying] = useState(true);

  const handleToolToggle = useCallback((tool: string) => {
    setActiveTools(prev => 
      prev.includes(tool) 
        ? prev.filter(t => t !== tool)
        : [...prev, tool]
    );
  }, []);

  const handleTimeframeChange = useCallback((timeframe: string) => {
    setActiveTimeframe(timeframe);
  }, []);

  // RL Metrics (Discrete Actions: Sell/Hold/Buy)
  const rlMetrics = {
    tradesPerEpisode: 6,
    avgHolding: 12,
    actionBalance: { sell: 18.5, hold: 63.2, buy: 18.3 },
    spreadCaptured: 19.8,
    pegRate: 3.2,
    vwapError: 2.8
  };

  const getMetricStatus = (metric: string, value: number) => {
    const thresholds = {
      tradesPerEpisode: { optimal: [2, 10], warning: [1, 12] },
      avgHolding: { optimal: [5, 25], warning: [3, 30] },
      spreadCaptured: { optimal: [0, 21.5], warning: [0, 25] },
      pegRate: { optimal: [0, 5], warning: [0, 20] },
      vwapError: { optimal: [0, 3], warning: [0, 5] }
    };

    const threshold = thresholds[metric as keyof typeof thresholds];
    if (!threshold) return 'optimal';

    if (value >= threshold.optimal[0] && value <= threshold.optimal[1]) return 'optimal';
    if (value >= threshold.warning[0] && value <= threshold.warning[1]) return 'warning';
    return 'critical';
  };

  return (
    <div className="min-h-screen bg-fintech-dark-950 p-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Live Trading Terminal</h1>
            <p className="text-fintech-dark-300">USD/COP Professional Trading - Discrete RL Actions (Sell/Hold/Buy)</p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Session Status */}
            <div className="glass-surface px-4 py-2 rounded-xl border border-market-up shadow-market-up">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                <span className="text-market-up font-medium">{session.sessionType}</span>
                <span className="text-fintech-dark-300 text-sm">• {session.hoursRemaining} left</span>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-2 rounded-lg transition-all ${
                  isPlaying 
                    ? 'bg-market-down/20 text-market-down border border-market-down/30'
                    : 'bg-market-up/20 text-market-up border border-market-up/30'
                }`}
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              
              <button className="p-2 bg-fintech-dark-800/50 hover:bg-fintech-dark-700/50 rounded-lg border border-fintech-dark-600/50 text-fintech-dark-300 transition-all">
                <Square className="w-5 h-5" />
              </button>
              
              <button className="p-2 bg-fintech-dark-800/50 hover:bg-fintech-dark-700/50 rounded-lg border border-fintech-dark-600/50 text-fintech-dark-300 transition-all">
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Price Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-surface p-4 rounded-xl border border-fintech-dark-700 mb-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-8">
            {/* Main Price */}
            <div className="flex items-center gap-4">
              <div>
                <div className="text-4xl font-bold text-white">
                  ${marketData.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                </div>
                <div className="text-sm text-fintech-dark-400">USD/COP</div>
              </div>
              
              <div className={`flex items-center gap-2 ${marketData.change >= 0 ? 'text-market-up' : 'text-market-down'}`}>
                {marketData.change >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                <div>
                  <div className="text-xl font-bold">
                    {marketData.change >= 0 ? '+' : ''}{marketData.change.toFixed(2)}
                  </div>
                  <div className="text-sm">
                    ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
                  </div>
                </div>
              </div>
            </div>

            {/* Market Stats */}
            <div className="flex items-center gap-6 text-sm">
              <div>
                <div className="text-fintech-dark-400">Volume 24H</div>
                <div className="text-white font-bold">{(marketData.volume / 1000000).toFixed(2)}M</div>
              </div>
              
              <div>
                <div className="text-fintech-dark-400">Spread</div>
                <div className={`font-bold ${marketData.spread <= 21.5 ? 'text-market-up' : 'text-market-down'}`}>
                  {marketData.spread.toFixed(1)} bps
                </div>
              </div>
              
              <div>
                <div className="text-fintech-dark-400">VWAP Error</div>
                <div className={`font-bold ${marketData.vwapError <= 3 ? 'text-market-up' : 'text-market-down'}`}>
                  {marketData.vwapError.toFixed(1)} bps
                </div>
              </div>

              <div>
                <div className="text-fintech-dark-400">Peg Rate</div>
                <div className={`font-bold ${marketData.pegRate <= 5 ? 'text-market-up' : 'text-market-down'}`}>
                  {marketData.pegRate.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Real-time indicators */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Signal className="w-5 h-5 text-fintech-cyan-500" />
              <div>
                <div className="text-sm text-fintech-dark-400">Last Update</div>
                <div className="text-white font-medium">{marketData.timestamp.toLocaleTimeString()}</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Critical RL Performance Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
          <Activity className="w-6 h-6 text-fintech-cyan-500" />
          Live Trading Performance
        </h2>
        
        {/* Critical Metrics - Always Visible */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-6">
          <MetricCard
            title="Spread Captured"
            value={`${rlMetrics.spreadCaptured} bps`}
            subtitle="<21.5 bps target"
            status={getMetricStatus('spreadCaptured', rlMetrics.spreadCaptured) as any}
            icon={Target}
            trend="down"
          />
          
          <MetricCard
            title="Peg Rate"
            value={`${rlMetrics.pegRate}%`}
            subtitle="<5% optimal, <20% limit"
            status={getMetricStatus('pegRate', rlMetrics.pegRate) as any}
            icon={AlertCircle}
            trend="up"
          />
          
          <MetricCard
            title="Trades/Episode"
            value={rlMetrics.tradesPerEpisode}
            subtitle="Target: 2-10 trades"
            status={getMetricStatus('tradesPerEpisode', rlMetrics.tradesPerEpisode) as any}
            icon={Activity}
            trend="neutral"
          />
        </div>
        
        {/* Secondary Metrics - Expandable */}
        <details className="group">
          <summary className="cursor-pointer list-none mb-4">
            <div className="flex items-center gap-3 text-fintech-dark-300 hover:text-white transition-colors">
              <span className="text-sm font-medium">Action Balance & Timing Details</span>
              <div className="group-open:rotate-90 transition-transform">▶</div>
            </div>
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <MetricCard
              title="Avg Holding"
              value={`${rlMetrics.avgHolding} bars`}
              subtitle="5M bars (5-25 target)"
              status={getMetricStatus('avgHolding', rlMetrics.avgHolding) as any}
              icon={Clock}
              trend="neutral"
            />
            
            <MetricCard
              title="Sell Actions"
              value={`${rlMetrics.actionBalance.sell}%`}
              subtitle="≥5% balance target"
              status={rlMetrics.actionBalance.sell >= 5 ? 'optimal' : 'critical'}
              icon={TrendingDown}
              trend="neutral"
            />
            
            <MetricCard
              title="Buy Actions"
              value={`${rlMetrics.actionBalance.buy}%`}
              subtitle="≥5% balance target"
              status={rlMetrics.actionBalance.buy >= 5 ? 'optimal' : 'critical'}
              icon={TrendingUp}
              trend="neutral"
            />
          </div>
        </details>
      </motion.div>

      {/* Chart Toolbar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mb-6"
      >
        <ChartToolbar
          activeTools={activeTools}
          onToolToggle={handleToolToggle}
          onTimeframeChange={handleTimeframeChange}
          activeTimeframe={activeTimeframe}
        />
      </motion.div>

      {/* Main Chart Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-6 mb-6"
        style={{ height: '600px' }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <h3 className="text-xl font-bold text-white">USD/COP Real-Time Chart</h3>
            <div className="flex items-center gap-2 text-sm text-fintech-dark-400">
              <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
              <span>Premium Window 08:00-12:55 COT • Coverage: {session.coverage}%</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button className="p-2 bg-fintech-dark-800/50 hover:bg-fintech-dark-700/50 rounded-lg text-fintech-dark-400 hover:text-white transition-all">
              <Camera className="w-4 h-4" />
            </button>
            <button className="p-2 bg-fintech-dark-800/50 hover:bg-fintech-dark-700/50 rounded-lg text-fintech-dark-400 hover:text-white transition-all">
              <Share2 className="w-4 h-4" />
            </button>
            <button className="p-2 bg-fintech-dark-800/50 hover:bg-fintech-dark-700/50 rounded-lg text-fintech-dark-400 hover:text-white transition-all">
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Chart Placeholder with Technical Indicators */}
        <div className="h-full bg-gradient-to-br from-fintech-dark-900/40 to-fintech-dark-800/40 rounded-xl border border-fintech-dark-700/30 flex items-center justify-center relative">
          <div className="text-center">
            <BarChart3 className="w-20 h-20 text-fintech-cyan-400 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-white mb-2">Professional TradingView-Style Chart</h3>
            <p className="text-fintech-dark-400 mb-6">Real-time USD/COP candlesticks with ML overlays</p>
            
            {/* Active Indicators Display */}
            <div className="flex items-center justify-center gap-6 text-sm">
              {activeTools.includes('bollinger') && (
                <span className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-market-up rounded-full"></div>
                  Bollinger Bands
                </span>
              )}
              {activeTools.includes('ema') && (
                <span className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-fintech-purple-400 rounded-full"></div>
                  EMA 20/50/200
                </span>
              )}
              {activeTools.includes('volume') && (
                <span className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-fintech-cyan-400 rounded-full"></div>
                  Volume Profile
                </span>
              )}
            </div>
          </div>

          {/* Overlays */}
          <div className="absolute top-4 left-4 space-y-2">
            <div className="glass-surface p-3 rounded-lg">
              <div className="text-sm text-fintech-dark-300">WTI & DXY Correlation</div>
              <div className="text-white font-medium">r = 0.73 (Strong)</div>
            </div>
          </div>

          <div className="absolute top-4 right-4 space-y-2">
            <div className="glass-surface p-3 rounded-lg">
              <div className="text-sm text-fintech-dark-300">Order Flow Imbalance</div>
              <div className={`text-lg font-bold ${marketData.orderFlow.imbalance > 0 ? 'text-market-up' : 'text-market-down'}`}>
                {marketData.orderFlow.imbalance > 0 ? '+' : ''}{marketData.orderFlow.imbalance.toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="absolute bottom-4 left-4 space-y-2">
            <div className="glass-surface p-3 rounded-lg">
              <div className="text-sm text-fintech-dark-300">RSI(14)</div>
              <div className={`text-lg font-bold ${
                marketData.technicals.rsi > 70 ? 'text-market-down' : 
                marketData.technicals.rsi < 30 ? 'text-market-up' : 
                'text-fintech-cyan-400'
              }`}>
                {marketData.technicals.rsi.toFixed(1)}
              </div>
            </div>
          </div>

          <div className="absolute bottom-4 right-4 space-y-2">
            <div className="glass-surface p-3 rounded-lg">
              <div className="text-sm text-fintech-dark-300">Latency</div>
              <div className="space-y-1">
                <div className="text-sm">ONNX: <span className="text-market-up font-medium">{session.latency.inference}ms</span></div>
                <div className="text-sm">E2E: <span className="text-market-up font-medium">{session.latency.e2e}ms</span></div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Bottom Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-surface p-4 rounded-xl border border-fintech-dark-700"
      >
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-fintech-dark-400">Session Coverage:</span>
              <span className="text-fintech-cyan-400 font-bold">{session.coverage}% of 60 bars</span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-fintech-dark-400">Slippage Est:</span>
              <span className="text-white font-bold">2.3 bps</span>
            </div>
            
            <div className="flex items-center gap-2">
              <span className="text-fintech-dark-400">Action Balance:</span>
              <span className="text-market-down font-bold">S:{rlMetrics.actionBalance.sell}%</span>
              <span className="text-fintech-dark-400 font-bold">H:{rlMetrics.actionBalance.hold}%</span>
              <span className="text-market-up font-bold">B:{rlMetrics.actionBalance.buy}%</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
              <span className="text-market-up">Real-time Data Stream</span>
            </div>
            
            <div className="text-fintech-dark-400">
              Premium Window Active • Next: {session.nextEvent}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}