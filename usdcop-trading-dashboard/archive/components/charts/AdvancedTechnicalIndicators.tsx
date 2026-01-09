'use client';

/**
 * Advanced Technical Indicators Component
 * Professional trading indicators with TradingView-style calculations
 * Includes: RSI, MACD, Stochastic, ATR, Williams %R, CCI, ADX, Bollinger Bands
 */

import React, { useMemo, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ComposedChart } from 'recharts';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target, 
  Gauge, 
  BarChart3, 
  Zap,
  Signal,
  Eye,
  EyeOff,
  Settings,
  RefreshCw,
  ChevronDown,
  ChevronUp
} from 'lucide-react';

// Import technical indicators library
import * as TI from 'technicalindicators';

interface OHLCData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TechnicalIndicatorsProps {
  data: OHLCData[];
  height?: number;
  showGrid?: boolean;
}

interface IndicatorData {
  datetime: string;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
  macdHistogram?: number;
  stochK?: number;
  stochD?: number;
  atr?: number;
  williamsR?: number;
  cci?: number;
  adx?: number;
  bbUpper?: number;
  bbMiddle?: number;
  bbLower?: number;
  obv?: number;
  mfi?: number;
  price: number;
  volume: number;
}

export const AdvancedTechnicalIndicators: React.FC<TechnicalIndicatorsProps> = ({
  data,
  height = 400,
  showGrid = true
}) => {
  const [activeIndicators, setActiveIndicators] = useState({
    rsi: true,
    macd: true,
    stochastic: false,
    atr: false,
    williamsR: false,
    cci: false,
    adx: false,
    bollinger: true,
    obv: false,
    mfi: false
  });

  const [selectedTimeframe, setSelectedTimeframe] = useState(14);
  const [isExpanded, setIsExpanded] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Calculate all technical indicators
  const indicatorData = useMemo(() => {
    if (!data || data.length < 50) return [];

    console.log(`[TechnicalIndicators] Calculating indicators for ${data.length} data points`);

    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const volumes = data.map(d => d.volume || 0);
    const opens = data.map(d => d.open);

    // RSI Calculation
    let rsiValues: number[] = [];
    try {
      rsiValues = TI.RSI.calculate({ 
        values: closes, 
        period: selectedTimeframe 
      });
    } catch (error) {
      console.warn('RSI calculation failed:', error);
    }

    // MACD Calculation
    let macdData: any[] = [];
    try {
      macdData = TI.MACD.calculate({
        values: closes,
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
        SimpleMAOscillator: false,
        SimpleMASignal: false
      });
    } catch (error) {
      console.warn('MACD calculation failed:', error);
    }

    // Stochastic Oscillator
    let stochData: any[] = [];
    try {
      stochData = TI.Stochastic.calculate({
        high: highs,
        low: lows,
        close: closes,
        period: selectedTimeframe,
        signalPeriod: 3
      });
    } catch (error) {
      console.warn('Stochastic calculation failed:', error);
    }

    // ATR (Average True Range)
    let atrValues: number[] = [];
    try {
      atrValues = TI.ATR.calculate({
        high: highs,
        low: lows,
        close: closes,
        period: selectedTimeframe
      });
    } catch (error) {
      console.warn('ATR calculation failed:', error);
    }

    // Williams %R
    let williamsRValues: number[] = [];
    try {
      williamsRValues = TI.WilliamsR.calculate({
        high: highs,
        low: lows,
        close: closes,
        period: selectedTimeframe
      });
    } catch (error) {
      console.warn('Williams %R calculation failed:', error);
    }

    // CCI (Commodity Channel Index)
    let cciValues: number[] = [];
    try {
      cciValues = TI.CCI.calculate({
        high: highs,
        low: lows,
        close: closes,
        period: selectedTimeframe
      });
    } catch (error) {
      console.warn('CCI calculation failed:', error);
    }

    // ADX (Average Directional Index)
    let adxData: any[] = [];
    try {
      adxData = TI.ADX.calculate({
        high: highs,
        low: lows,
        close: closes,
        period: selectedTimeframe
      });
    } catch (error) {
      console.warn('ADX calculation failed:', error);
    }

    // Bollinger Bands
    let bbData: any[] = [];
    try {
      bbData = TI.BollingerBands.calculate({
        values: closes,
        period: 20,
        stdDev: 2
      });
    } catch (error) {
      console.warn('Bollinger Bands calculation failed:', error);
    }

    // OBV (On Balance Volume)
    let obvValues: number[] = [];
    try {
      obvValues = TI.OBV.calculate({
        close: closes,
        volume: volumes
      });
    } catch (error) {
      console.warn('OBV calculation failed:', error);
    }

    // MFI (Money Flow Index)
    let mfiValues: number[] = [];
    try {
      mfiValues = TI.MFI.calculate({
        high: highs,
        low: lows,
        close: closes,
        volume: volumes,
        period: selectedTimeframe
      });
    } catch (error) {
      console.warn('MFI calculation failed:', error);
    }

    // Combine all indicators
    const combinedData: IndicatorData[] = data.map((item, index) => {
      const result: IndicatorData = {
        datetime: item.datetime,
        price: item.close,
        volume: item.volume || 0
      };

      // Add RSI
      const rsiIndex = index - (selectedTimeframe - 1);
      if (rsiIndex >= 0 && rsiIndex < rsiValues.length) {
        result.rsi = rsiValues[rsiIndex];
      }

      // Add MACD
      const macdIndex = index - (26 - 1); // MACD starts after slow period
      if (macdIndex >= 0 && macdIndex < macdData.length) {
        result.macd = macdData[macdIndex].MACD;
        result.macdSignal = macdData[macdIndex].signal;
        result.macdHistogram = macdData[macdIndex].histogram;
      }

      // Add Stochastic
      const stochIndex = index - (selectedTimeframe - 1);
      if (stochIndex >= 0 && stochIndex < stochData.length) {
        result.stochK = stochData[stochIndex].k;
        result.stochD = stochData[stochIndex].d;
      }

      // Add ATR
      const atrIndex = index - (selectedTimeframe - 1);
      if (atrIndex >= 0 && atrIndex < atrValues.length) {
        result.atr = atrValues[atrIndex];
      }

      // Add Williams %R
      const wrIndex = index - (selectedTimeframe - 1);
      if (wrIndex >= 0 && wrIndex < williamsRValues.length) {
        result.williamsR = williamsRValues[wrIndex];
      }

      // Add CCI
      const cciIndex = index - (selectedTimeframe - 1);
      if (cciIndex >= 0 && cciIndex < cciValues.length) {
        result.cci = cciValues[cciIndex];
      }

      // Add ADX
      const adxIndex = index - (selectedTimeframe - 1);
      if (adxIndex >= 0 && adxIndex < adxData.length) {
        result.adx = adxData[adxIndex].adx;
      }

      // Add Bollinger Bands
      const bbIndex = index - (20 - 1); // BB period is 20
      if (bbIndex >= 0 && bbIndex < bbData.length) {
        result.bbUpper = bbData[bbIndex].upper;
        result.bbMiddle = bbData[bbIndex].middle;
        result.bbLower = bbData[bbIndex].lower;
      }

      // Add OBV
      if (index < obvValues.length) {
        result.obv = obvValues[index];
      }

      // Add MFI
      const mfiIndex = index - (selectedTimeframe - 1);
      if (mfiIndex >= 0 && mfiIndex < mfiValues.length) {
        result.mfi = mfiValues[mfiIndex];
      }

      return result;
    });

    console.log(`[TechnicalIndicators] Generated ${combinedData.length} indicator data points`);
    return combinedData.filter(item => item.rsi !== undefined); // Filter out incomplete data
  }, [data, selectedTimeframe]);

  // Calculate current signals and trends
  const currentSignals = useMemo(() => {
    if (indicatorData.length < 2) return {};

    const latest = indicatorData[indicatorData.length - 1];
    const previous = indicatorData[indicatorData.length - 2];

    return {
      rsi: {
        value: latest.rsi,
        signal: latest.rsi! > 70 ? 'OVERBOUGHT' : latest.rsi! < 30 ? 'OVERSOLD' : 'NEUTRAL',
        trend: latest.rsi! > previous.rsi! ? 'UP' : 'DOWN'
      },
      macd: {
        value: latest.macd,
        signal: latest.macd! > latest.macdSignal! ? 'BULLISH' : 'BEARISH',
        histogram: latest.macdHistogram,
        trend: latest.macd! > previous.macd! ? 'UP' : 'DOWN'
      },
      stochastic: {
        k: latest.stochK,
        d: latest.stochD,
        signal: latest.stochK! > 80 ? 'OVERBOUGHT' : latest.stochK! < 20 ? 'OVERSOLD' : 'NEUTRAL'
      },
      atr: {
        value: latest.atr,
        trend: latest.atr! > previous.atr! ? 'INCREASING' : 'DECREASING'
      }
    };
  }, [indicatorData]);

  const refreshIndicators = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1000);
  };

  const toggleIndicator = (indicator: keyof typeof activeIndicators) => {
    setActiveIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BULLISH':
      case 'OVERSOLD':
        return 'text-emerald-400';
      case 'BEARISH':
      case 'OVERBOUGHT':
        return 'text-red-400';
      case 'NEUTRAL':
      default:
        return 'text-yellow-400';
    }
  };

  const getSignalBadgeColor = (signal: string) => {
    switch (signal) {
      case 'BULLISH':
      case 'OVERSOLD':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'BEARISH':
      case 'OVERBOUGHT':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'NEUTRAL':
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="space-y-6"
    >
      {/* Header Panel */}
      <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5" />
        
        <div className="flex items-center justify-between mb-4 relative z-10">
          <div className="flex items-center gap-4">
            <motion.h3
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
            >
              Technical Indicators
            </motion.h3>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-white transition-all duration-200"
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </motion.button>

            <Badge className="bg-slate-800/50 text-slate-300 border-slate-600/50">
              {indicatorData.length} data points
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {/* Timeframe selector */}
            <div className="flex items-center gap-2 bg-slate-800/50 rounded-lg p-1">
              {[9, 14, 21, 50].map(period => (
                <button
                  key={period}
                  onClick={() => setSelectedTimeframe(period)}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-all duration-200 ${
                    selectedTimeframe === period
                      ? 'bg-cyan-500 text-white'
                      : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                  }`}
                >
                  {period}
                </button>
              ))}
            </div>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={refreshIndicators}
              disabled={refreshing}
              className="p-2 rounded-lg bg-slate-800/50 hover:bg-emerald-500/20 text-slate-400 hover:text-emerald-400 transition-all duration-200"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </motion.button>
          </div>
        </div>

        {/* Current Signals Summary */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-4 relative z-10"
            >
              {/* RSI Signal */}
              {activeIndicators.rsi && currentSignals.rsi && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-400 font-semibold">RSI ({selectedTimeframe})</span>
                    <Activity className="w-4 h-4 text-purple-400" />
                  </div>
                  <div className="text-2xl font-mono font-bold text-white mb-1">
                    {currentSignals.rsi.value?.toFixed(1)}
                  </div>
                  <Badge className={`text-xs ${getSignalBadgeColor(currentSignals.rsi.signal)}`}>
                    {currentSignals.rsi.signal}
                  </Badge>
                </motion.div>
              )}

              {/* MACD Signal */}
              {activeIndicators.macd && currentSignals.macd && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 }}
                  className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-400 font-semibold">MACD</span>
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="text-2xl font-mono font-bold text-white mb-1">
                    {currentSignals.macd.value?.toFixed(3)}
                  </div>
                  <Badge className={`text-xs ${getSignalBadgeColor(currentSignals.macd.signal)}`}>
                    {currentSignals.macd.signal}
                  </Badge>
                </motion.div>
              )}

              {/* Stochastic Signal */}
              {activeIndicators.stochastic && currentSignals.stochastic && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                  className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-400 font-semibold">Stochastic</span>
                    <Target className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div className="text-2xl font-mono font-bold text-white mb-1">
                    {currentSignals.stochastic.k?.toFixed(1)}
                  </div>
                  <Badge className={`text-xs ${getSignalBadgeColor(currentSignals.stochastic.signal)}`}>
                    {currentSignals.stochastic.signal}
                  </Badge>
                </motion.div>
              )}

              {/* ATR Signal */}
              {activeIndicators.atr && currentSignals.atr && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 }}
                  className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-400 font-semibold">ATR</span>
                    <Gauge className="w-4 h-4 text-orange-400" />
                  </div>
                  <div className="text-2xl font-mono font-bold text-white mb-1">
                    {currentSignals.atr.value?.toFixed(2)}
                  </div>
                  <Badge className={`text-xs ${currentSignals.atr.trend === 'INCREASING' ? 'bg-red-500/20 text-red-400 border-red-500/30' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'}`}>
                    {currentSignals.atr.trend}
                  </Badge>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Indicator Toggle Controls */}
        <div className="flex flex-wrap gap-2 mt-4 relative z-10">
          {Object.entries(activeIndicators).map(([key, active]) => (
            <motion.button
              key={key}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => toggleIndicator(key as keyof typeof activeIndicators)}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 ${
                active
                  ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 text-white border border-cyan-500/30'
                  : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:border-slate-600/50 hover:text-slate-300'
              }`}
            >
              {key.toUpperCase()}
            </motion.button>
          ))}
        </div>
      </Card>

      {/* Individual Indicator Charts */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* RSI Chart */}
            {activeIndicators.rsi && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Activity className="w-5 h-5 text-purple-400" />
                      RSI ({selectedTimeframe})
                    </h4>
                    <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
                      Momentum
                    </Badge>
                  </div>
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={indicatorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                        <XAxis 
                          dataKey="datetime" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis 
                          domain={[0, 100]}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #475569',
                            borderRadius: '8px',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
                          }}
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                        />
                        <Line
                          type="monotone"
                          dataKey="rsi"
                          stroke="#a855f7"
                          strokeWidth={2}
                          dot={false}
                          strokeDasharray="0"
                        />
                        {/* Overbought/Oversold lines */}
                        <Line
                          type="monotone"
                          dataKey={() => 70}
                          stroke="#ef4444"
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey={() => 30}
                          stroke="#10b981"
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* MACD Chart */}
            {activeIndicators.macd && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-cyan-400" />
                      MACD
                    </h4>
                    <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                      Trend
                    </Badge>
                  </div>
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={indicatorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                        <XAxis 
                          dataKey="datetime" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis stroke="#9CA3AF" fontSize={10} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #475569',
                            borderRadius: '8px'
                          }}
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                        />
                        <Bar
                          dataKey="macdHistogram"
                          fill="#06b6d4"
                          opacity={0.6}
                        />
                        <Line
                          type="monotone"
                          dataKey="macd"
                          stroke="#06b6d4"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="macdSignal"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Bollinger Bands Chart */}
            {activeIndicators.bollinger && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-emerald-400" />
                      Bollinger Bands
                    </h4>
                    <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                      Volatility
                    </Badge>
                  </div>
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={indicatorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                        <XAxis 
                          dataKey="datetime" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis stroke="#9CA3AF" fontSize={10} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #475569',
                            borderRadius: '8px'
                          }}
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                        />
                        <Area
                          type="monotone"
                          dataKey="bbUpper"
                          stroke="#10b981"
                          fill="transparent"
                          strokeWidth={1}
                          strokeDasharray="3 3"
                        />
                        <Line
                          type="monotone"
                          dataKey="bbMiddle"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="price"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Area
                          type="monotone"
                          dataKey="bbLower"
                          stroke="#10b981"
                          fill="transparent"
                          strokeWidth={1}
                          strokeDasharray="3 3"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Stochastic Oscillator Chart */}
            {activeIndicators.stochastic && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Target className="w-5 h-5 text-emerald-400" />
                      Stochastic Oscillator
                    </h4>
                    <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                      Momentum
                    </Badge>
                  </div>
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={indicatorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                        <XAxis 
                          dataKey="datetime" 
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          stroke="#9CA3AF"
                          fontSize={10}
                        />
                        <YAxis 
                          domain={[0, 100]}
                          stroke="#9CA3AF" 
                          fontSize={10} 
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #475569',
                            borderRadius: '8px'
                          }}
                          labelFormatter={(value) => new Date(value).toLocaleString()}
                        />
                        <Line
                          type="monotone"
                          dataKey="stochK"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="stochD"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />
                        {/* Overbought/Oversold lines */}
                        <Line
                          type="monotone"
                          dataKey={() => 80}
                          stroke="#ef4444"
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey={() => 20}
                          stroke="#ef4444"
                          strokeWidth={1}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default AdvancedTechnicalIndicators;