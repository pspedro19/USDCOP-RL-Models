'use client';

/**
 * Multi-Timeframe Analysis Dashboard
 * Professional multi-timeframe view with synchronized analysis
 * Features: 5M, 15M, 1H, 4H, 1D charts with trend alignment and confluence analysis
 */

import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, ResponsiveContainer, YAxis, XAxis, CartesianGrid, Tooltip } from 'recharts';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Maximize2,
  Minimize2,
  RefreshCw,
  Target,
  Clock,
  Zap,
  CheckCircle,
  AlertTriangle,
  Eye,
  EyeOff,
  Settings,
  Grid3X3,
  Signal
} from 'lucide-react';
import * as TI from 'technicalindicators';

interface OHLCData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TimeframeData {
  timeframe: string;
  data: OHLCData[];
  trend: 'bullish' | 'bearish' | 'neutral';
  strength: number; // 0-100
  rsi: number;
  macd: { value: number; signal: number; histogram: number };
  support: number;
  resistance: number;
  key_levels: number[];
}

interface MultiTimeframeProps {
  rawData: OHLCData[];
  height?: number;
}

type Timeframe = '5M' | '15M' | '1H' | '4H' | '1D' | '1W';

export const MultiTimeframeAnalysis: React.FC<MultiTimeframeProps> = ({
  rawData,
  height = 300
}) => {
  const [selectedTimeframes, setSelectedTimeframes] = useState<Timeframe[]>(['5M', '15M', '1H', '4H']);
  const [syncCrosshairs, setSyncCrosshairs] = useState(true);
  const [showConfluence, setShowConfluence] = useState(true);
  const [expandedChart, setExpandedChart] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showIndicators, setShowIndicators] = useState(true);

  // Aggregate data for different timeframes
  const aggregateTimeframeData = useCallback((data: OHLCData[], timeframe: Timeframe): OHLCData[] => {
    if (!data || data.length === 0) return [];

    const aggregated: OHLCData[] = [];
    let periodMinutes: number;

    switch (timeframe) {
      case '5M': periodMinutes = 5; break;
      case '15M': periodMinutes = 15; break;
      case '1H': periodMinutes = 60; break;
      case '4H': periodMinutes = 240; break;
      case '1D': periodMinutes = 1440; break;
      case '1W': periodMinutes = 10080; break;
      default: periodMinutes = 5;
    }

    // Assuming base data is 5-minute intervals
    const candlesPerPeriod = periodMinutes / 5;

    for (let i = 0; i < data.length; i += candlesPerPeriod) {
      const slice = data.slice(i, Math.min(i + candlesPerPeriod, data.length));
      
      if (slice.length === 0) continue;

      const open = slice[0].open;
      const high = Math.max(...slice.map(d => d.high));
      const low = Math.min(...slice.map(d => d.low));
      const close = slice[slice.length - 1].close;
      const volume = slice.reduce((sum, d) => sum + (d.volume || 0), 0);

      aggregated.push({
        datetime: slice[0].datetime,
        open,
        high,
        low,
        close,
        volume
      });
    }

    return aggregated;
  }, []);

  // Calculate technical analysis for each timeframe
  const calculateTechnicalData = useCallback((data: OHLCData[]): Omit<TimeframeData, 'timeframe' | 'data'> => {
    if (data.length < 20) {
      return {
        trend: 'neutral',
        strength: 0,
        rsi: 50,
        macd: { value: 0, signal: 0, histogram: 0 },
        support: 0,
        resistance: 0,
        key_levels: []
      };
    }

    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);

    // RSI
    let rsi = 50;
    try {
      const rsiValues = TI.RSI.calculate({ values: closes, period: 14 });
      rsi = rsiValues[rsiValues.length - 1] || 50;
    } catch (error) {
      console.warn('RSI calculation failed:', error);
    }

    // MACD
    let macd = { value: 0, signal: 0, histogram: 0 };
    try {
      const macdData = TI.MACD.calculate({
        values: closes,
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
        SimpleMAOscillator: false,
        SimpleMASignal: false
      });
      const latest = macdData[macdData.length - 1];
      if (latest) {
        macd = {
          value: latest.MACD || 0,
          signal: latest.signal || 0,
          histogram: latest.histogram || 0
        };
      }
    } catch (error) {
      console.warn('MACD calculation failed:', error);
    }

    // Trend determination
    const smaShort = closes.slice(-10).reduce((a, b) => a + b, 0) / 10;
    const smaLong = closes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    let strength = 0;

    if (smaShort > smaLong && rsi > 50 && macd.value > macd.signal) {
      trend = 'bullish';
      strength = Math.min(100, (rsi - 50) * 2 + Math.abs(macd.histogram) * 100);
    } else if (smaShort < smaLong && rsi < 50 && macd.value < macd.signal) {
      trend = 'bearish';
      strength = Math.min(100, (50 - rsi) * 2 + Math.abs(macd.histogram) * 100);
    } else {
      strength = 50 - Math.abs(rsi - 50);
    }

    // Support and Resistance levels
    const recentData = data.slice(-50);
    const recentHighs = recentData.map(d => d.high).sort((a, b) => b - a);
    const recentLows = recentData.map(d => d.low).sort((a, b) => a - b);
    
    const resistance = recentHighs.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
    const support = recentLows.slice(0, 5).reduce((a, b) => a + b, 0) / 5;

    // Key levels (pivot points, psychological levels)
    const currentPrice = closes[closes.length - 1];
    const key_levels = [
      support,
      support + (resistance - support) * 0.382,
      support + (resistance - support) * 0.5,
      support + (resistance - support) * 0.618,
      resistance
    ].filter(level => Math.abs(level - currentPrice) / currentPrice > 0.005); // Filter out levels too close to current price

    return {
      trend,
      strength: Math.round(strength),
      rsi: Math.round(rsi),
      macd,
      support: Math.round(support * 100) / 100,
      resistance: Math.round(resistance * 100) / 100,
      key_levels
    };
  }, []);

  // Process all timeframe data
  const timeframeAnalysis = useMemo((): TimeframeData[] => {
    if (!rawData || rawData.length === 0) return [];

    console.log(`[MultiTimeframe] Processing ${rawData.length} data points for ${selectedTimeframes.length} timeframes`);

    return selectedTimeframes.map(timeframe => {
      const aggregatedData = aggregateTimeframeData(rawData, timeframe);
      const technicalData = calculateTechnicalData(aggregatedData);
      
      return {
        timeframe,
        data: aggregatedData,
        ...technicalData
      };
    });
  }, [rawData, selectedTimeframes, aggregateTimeframeData, calculateTechnicalData]);

  // Calculate confluence analysis
  const confluenceAnalysis = useMemo(() => {
    if (timeframeAnalysis.length === 0) return null;

    const bullishCount = timeframeAnalysis.filter(tf => tf.trend === 'bullish').length;
    const bearishCount = timeframeAnalysis.filter(tf => tf.trend === 'bearish').length;
    const avgStrength = timeframeAnalysis.reduce((sum, tf) => sum + tf.strength, 0) / timeframeAnalysis.length;
    const avgRsi = timeframeAnalysis.reduce((sum, tf) => sum + tf.rsi, 0) / timeframeAnalysis.length;

    const overallTrend = bullishCount > bearishCount ? 'bullish' : 
                        bearishCount > bullishCount ? 'bearish' : 'neutral';

    const confluence = Math.abs(bullishCount - bearishCount) / timeframeAnalysis.length;

    // Find common support/resistance levels
    const allLevels = timeframeAnalysis.flatMap(tf => [tf.support, tf.resistance, ...tf.key_levels]);
    const levelGroups: { level: number; count: number }[] = [];
    
    allLevels.forEach(level => {
      const existing = levelGroups.find(group => Math.abs(group.level - level) / level < 0.01);
      if (existing) {
        existing.count++;
        existing.level = (existing.level * (existing.count - 1) + level) / existing.count;
      } else {
        levelGroups.push({ level, count: 1 });
      }
    });

    const significantLevels = levelGroups
      .filter(group => group.count >= 2)
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    return {
      overallTrend,
      confluence: Math.round(confluence * 100),
      avgStrength: Math.round(avgStrength),
      avgRsi: Math.round(avgRsi),
      bullishTimeframes: bullishCount,
      bearishTimeframes: bearishCount,
      significantLevels: significantLevels.map(group => ({
        level: Math.round(group.level * 100) / 100,
        strength: group.count
      }))
    };
  }, [timeframeAnalysis]);

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30';
      case 'bearish': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
    }
  };

  const getStrengthColor = (strength: number) => {
    if (strength > 70) return 'text-emerald-400';
    if (strength > 40) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatTimeframe = (tf: string) => {
    const labels: { [key: string]: string } = {
      '5M': '5 min',
      '15M': '15 min',
      '1H': '1 hour',
      '4H': '4 hours',
      '1D': '1 day',
      '1W': '1 week'
    };
    return labels[tf] || tf;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="space-y-6"
    >
      {/* Control Panel */}
      <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <motion.h3
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
            >
              Multi-Timeframe Analysis
            </motion.h3>
            <Badge className="bg-slate-800 text-slate-300 border-slate-600">
              {selectedTimeframes.length} timeframes
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSyncCrosshairs(!syncCrosshairs)}
              className={`${syncCrosshairs ? 'text-cyan-400' : 'text-slate-400'} hover:text-white`}
            >
              <Target className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowConfluence(!showConfluence)}
              className={`${showConfluence ? 'text-purple-400' : 'text-slate-400'} hover:text-white`}
            >
              <Signal className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowIndicators(!showIndicators)}
              className={`${showIndicators ? 'text-emerald-400' : 'text-slate-400'} hover:text-white`}
            >
              <Activity className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Timeframe Selection */}
        <div className="flex flex-wrap gap-2 mb-6">
          {(['5M', '15M', '1H', '4H', '1D', '1W'] as Timeframe[]).map(tf => (
            <motion.button
              key={tf}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                setSelectedTimeframes(prev =>
                  prev.includes(tf)
                    ? prev.filter(t => t !== tf)
                    : [...prev, tf].slice(0, 6) // Max 6 timeframes
                );
              }}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                selectedTimeframes.includes(tf)
                  ? 'bg-cyan-500 text-white shadow-lg'
                  : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:border-slate-600/50 hover:text-slate-300'
              }`}
            >
              {tf}
            </motion.button>
          ))}
        </div>

        {/* Confluence Analysis */}
        {showConfluence && confluenceAnalysis && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="bg-gradient-to-r from-slate-800/30 to-slate-700/30 rounded-xl p-4 border border-slate-600/20"
          >
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Signal className="w-5 h-5 text-purple-400" />
              Confluence Analysis
            </h4>
            
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
              <div className="text-center">
                <div className={`text-2xl font-bold mb-1 ${getTrendColor(confluenceAnalysis.overallTrend).split(' ')[0]}`}>
                  {confluenceAnalysis.overallTrend.toUpperCase()}
                </div>
                <div className="text-xs text-slate-400">Overall Trend</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-cyan-400 mb-1">
                  {confluenceAnalysis.confluence}%
                </div>
                <div className="text-xs text-slate-400">Confluence</div>
              </div>
              
              <div className="text-center">
                <div className={`text-2xl font-bold mb-1 ${getStrengthColor(confluenceAnalysis.avgStrength)}`}>
                  {confluenceAnalysis.avgStrength}
                </div>
                <div className="text-xs text-slate-400">Avg Strength</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400 mb-1">
                  {confluenceAnalysis.bullishTimeframes}
                </div>
                <div className="text-xs text-slate-400">Bullish TFs</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400 mb-1">
                  {confluenceAnalysis.bearishTimeframes}
                </div>
                <div className="text-xs text-slate-400">Bearish TFs</div>
              </div>
            </div>

            {/* Significant Levels */}
            {confluenceAnalysis.significantLevels.length > 0 && (
              <div>
                <h5 className="text-sm font-semibold text-slate-300 mb-2">Key Confluence Levels</h5>
                <div className="flex flex-wrap gap-2">
                  {confluenceAnalysis.significantLevels.map((level, index) => (
                    <Badge
                      key={index}
                      className={`${
                        level.strength > 3 
                          ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                          : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                      }`}
                    >
                      ${level.level} ({level.strength}x)
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </Card>

      {/* Timeframe Charts Grid */}
      <div className={`grid gap-4 ${
        expandedChart 
          ? 'grid-cols-1' 
          : selectedTimeframes.length <= 2 
            ? 'grid-cols-1 md:grid-cols-2' 
            : selectedTimeframes.length <= 4
              ? 'grid-cols-2 md:grid-cols-2 lg:grid-cols-4'
              : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6'
      }`}>
        {timeframeAnalysis.map((tfData, index) => (
          <motion.div
            key={tfData.timeframe}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: index * 0.1 }}
          >
            <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <h4 className="text-lg font-semibold text-white">
                    {tfData.timeframe}
                  </h4>
                  <Badge className={getTrendColor(tfData.trend)}>
                    {tfData.trend}
                  </Badge>
                </div>
                
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setExpandedChart(
                      expandedChart === tfData.timeframe ? null : tfData.timeframe
                    )}
                    className="text-slate-400 hover:text-white p-1"
                  >
                    {expandedChart === tfData.timeframe ? (
                      <Minimize2 className="w-3 h-3" />
                    ) : (
                      <Maximize2 className="w-3 h-3" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Mini Chart */}
              <div className={`${expandedChart === tfData.timeframe ? 'h-96' : 'h-32'} mb-3`}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={tfData.data.slice(-50)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis 
                      dataKey="datetime" 
                      hide={!expandedChart}
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      stroke="#9CA3AF"
                      fontSize={10}
                    />
                    <YAxis 
                      domain={['dataMin - 10', 'dataMax + 10']}
                      hide={!expandedChart}
                      stroke="#9CA3AF"
                      fontSize={10}
                    />
                    {expandedChart === tfData.timeframe && (
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #475569',
                          borderRadius: '8px'
                        }}
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                      />
                    )}
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke={tfData.trend === 'bullish' ? '#10b981' : tfData.trend === 'bearish' ? '#ef4444' : '#f59e0b'}
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Technical Summary */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Strength:</span>
                  <span className={`font-bold ${getStrengthColor(tfData.strength)}`}>
                    {tfData.strength}%
                  </span>
                </div>
                
                {showIndicators && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">RSI:</span>
                      <span className={`font-mono ${
                        tfData.rsi > 70 ? 'text-red-400' : 
                        tfData.rsi < 30 ? 'text-emerald-400' : 'text-slate-300'
                      }`}>
                        {tfData.rsi}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">MACD:</span>
                      <span className={`font-mono text-xs ${
                        tfData.macd.value > tfData.macd.signal ? 'text-emerald-400' : 'text-red-400'
                      }`}>
                        {tfData.macd.histogram > 0 ? '▲' : '▼'} {Math.abs(tfData.macd.histogram).toFixed(3)}
                      </span>
                    </div>
                  </>
                )}
                
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Support:</span>
                  <span className="text-emerald-400 font-mono text-xs">
                    ${tfData.support}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Resistance:</span>
                  <span className="text-red-400 font-mono text-xs">
                    ${tfData.resistance}
                  </span>
                </div>

                {/* Data points info */}
                <div className="pt-2 border-t border-slate-700/30">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>{formatTimeframe(tfData.timeframe)}</span>
                    <span>{tfData.data.length} bars</span>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* No Data Message */}
      {timeframeAnalysis.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-400 mb-2">No Analysis Data</h3>
          <p className="text-slate-500">
            Please select at least one timeframe to begin multi-timeframe analysis.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default MultiTimeframeAnalysis;