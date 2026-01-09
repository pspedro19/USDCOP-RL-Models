'use client';

import React, { useState, useCallback } from 'react';
import RealDataTradingChart from '@/components/charts/RealDataTradingChart';
import { motion } from 'framer-motion';
import { useRLMetrics } from '@/hooks/useAnalytics';
import {
  Activity, TrendingUp, TrendingDown, Target, AlertCircle, Clock,
  Play, Pause, Square, Maximize2, Signal
} from 'lucide-react';
import { useRealTimeMarketData } from '@/hooks/trading/useRealTimeMarketData';
import { useTradingSession } from '@/hooks/trading/useTradingSession';
import { useExecutionMetrics } from '@/hooks/trading/useExecutionMetrics';
import { MetricCard, ChartToolbar } from '@/components/trading';
import { RiskStatusCard } from '@/components/trading/RiskStatusCard';

export default function LiveTradingTerminal() {
  const marketData = useRealTimeMarketData();
  const session = useTradingSession();
  const executionMetrics = useExecutionMetrics();
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

  const { metrics: rlMetricsData, isLoading: rlMetricsLoading } = useRLMetrics('USDCOP', 30);

  const rlMetrics = rlMetricsData || {
    tradesPerEpisode: 0,
    avgHolding: 0,
    actionBalance: { sell: 0, hold: 0, buy: 0 },
    spreadCaptured: 0,
    pegRate: 0,
    vwapError: 0
  };

  const getMetricStatus = (metric: string, value: number): 'optimal' | 'warning' | 'critical' => {
    const thresholds: Record<string, { optimal: [number, number], warning: [number, number] }> = {
      tradesPerEpisode: { optimal: [2, 10], warning: [1, 12] },
      avgHolding: { optimal: [5, 25], warning: [3, 30] },
      spreadCaptured: { optimal: [0, 21.5], warning: [0, 25] },
      pegRate: { optimal: [0, 5], warning: [0, 20] },
      vwapError: { optimal: [0, 3], warning: [0, 5] }
    };

    const threshold = thresholds[metric];
    if (!threshold) return 'optimal';

    if (value >= threshold.optimal[0] && value <= threshold.optimal[1]) return 'optimal';
    if (value >= threshold.warning[0] && value <= threshold.warning[1]) return 'warning';
    return 'critical';
  };

  return (
    <div className="min-h-screen bg-fintech-dark-950 p-6">
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
            <div className="glass-surface px-4 py-2 rounded-xl border border-market-up shadow-market-up">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                <span className="text-market-up font-medium">{session.sessionType}</span>
                <span className="text-fintech-dark-300 text-sm">• {session.hoursRemaining} left</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-2 rounded-lg transition-all ${isPlaying
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

      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-surface p-4 rounded-xl border border-fintech-dark-700 mb-6"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-8">
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

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          <div className="xl:col-span-3">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
              <Activity className="w-6 h-6 text-fintech-cyan-500" />
              Live Trading Performance
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-6">
              <MetricCard
                title="Spread Captured"
                value={`${rlMetrics.spreadCaptured} bps`}
                subtitle="<21.5 bps target"
                status={getMetricStatus('spreadCaptured', rlMetrics.spreadCaptured)}
                icon={Target}
                trend="down"
              />

              <MetricCard
                title="Peg Rate"
                value={`${rlMetrics.pegRate}%`}
                subtitle="<5% optimal, <20% limit"
                status={getMetricStatus('pegRate', rlMetrics.pegRate)}
                icon={AlertCircle}
                trend="up"
              />

              <MetricCard
                title="Trades/Episode"
                value={rlMetrics.tradesPerEpisode}
                subtitle="Target: 2-10 trades"
                status={getMetricStatus('tradesPerEpisode', rlMetrics.tradesPerEpisode)}
                icon={Activity}
                trend="neutral"
              />
            </div>

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
                  status={getMetricStatus('avgHolding', rlMetrics.avgHolding)}
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
          </div>

          <div className="xl:col-span-1">
            <RiskStatusCard />
          </div>
        </div>
      </motion.div>

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
        </div>

        <RealDataTradingChart
          symbol="USDCOP"
          timeframe="5m"
          height={500}
          className="h-full"
          showSignals={true}
          showPositions={true}
        />
      </motion.div>

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
              <span className="text-white font-bold">
                {executionMetrics.avgSlippage > 0
                  ? `${executionMetrics.avgSlippage.toFixed(1)} bps`
                  : '-- bps'}
              </span>
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
