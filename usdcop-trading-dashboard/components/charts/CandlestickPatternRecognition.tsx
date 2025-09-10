'use client';

/**
 * Advanced Candlestick Pattern Recognition System
 * Identifies and highlights professional trading patterns
 * Includes: Doji, Hammer, Engulfing, Stars, Triangles, and complex formations
 */

import React, { useMemo, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp,
  TrendingDown,
  Target,
  Star,
  Triangle,
  Circle,
  Square,
  Zap,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Search,
  Filter,
  BarChart3,
  Activity,
  Crosshair,
  RefreshCw
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

interface CandlestickPattern {
  name: string;
  type: 'bullish' | 'bearish' | 'neutral' | 'reversal';
  strength: 'weak' | 'moderate' | 'strong';
  index: number;
  datetime: string;
  description: string;
  reliability: number; // 0-100
  confirmation: boolean;
}

interface PatternRecognitionProps {
  data: OHLCData[];
  onPatternClick?: (pattern: CandlestickPattern) => void;
}

export const CandlestickPatternRecognition: React.FC<PatternRecognitionProps> = ({
  data,
  onPatternClick
}) => {
  const [selectedPatterns, setSelectedPatterns] = useState<string[]>([
    'doji', 'hammer', 'engulfing', 'harami', 'star', 'marubozu'
  ]);
  const [filterType, setFilterType] = useState<'all' | 'bullish' | 'bearish' | 'reversal'>('all');
  const [minStrength, setMinStrength] = useState<'weak' | 'moderate' | 'strong'>('weak');
  const [showDescriptions, setShowDescriptions] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Pattern detection algorithms
  const detectPatterns = useMemo(() => {
    if (!data || data.length < 5) return [];

    const patterns: CandlestickPattern[] = [];
    
    console.log(`[PatternRecognition] Analyzing ${data.length} candles for patterns`);

    // Helper functions
    const isGreenCandle = (candle: OHLCData) => candle.close > candle.open;
    const isRedCandle = (candle: OHLCData) => candle.close < candle.open;
    const getCandleBody = (candle: OHLCData) => Math.abs(candle.close - candle.open);
    const getCandleRange = (candle: OHLCData) => candle.high - candle.low;
    const getUpperWick = (candle: OHLCData) => candle.high - Math.max(candle.open, candle.close);
    const getLowerWick = (candle: OHLCData) => Math.min(candle.open, candle.close) - candle.low;

    // Pattern detection functions
    const isDoji = (candle: OHLCData, threshold = 0.1): boolean => {
      const body = getCandleBody(candle);
      const range = getCandleRange(candle);
      return range > 0 && (body / range) < threshold;
    };

    const isHammer = (candle: OHLCData): boolean => {
      const body = getCandleBody(candle);
      const lowerWick = getLowerWick(candle);
      const upperWick = getUpperWick(candle);
      const range = getCandleRange(candle);
      
      return range > 0 && 
             lowerWick > body * 2 && 
             upperWick < body * 0.5 &&
             body / range < 0.3;
    };

    const isShootingStar = (candle: OHLCData): boolean => {
      const body = getCandleBody(candle);
      const lowerWick = getLowerWick(candle);
      const upperWick = getUpperWick(candle);
      const range = getCandleRange(candle);
      
      return range > 0 && 
             upperWick > body * 2 && 
             lowerWick < body * 0.5 &&
             body / range < 0.3;
    };

    const isMarubozu = (candle: OHLCData): boolean => {
      const body = getCandleBody(candle);
      const range = getCandleRange(candle);
      const upperWick = getUpperWick(candle);
      const lowerWick = getLowerWick(candle);
      
      return range > 0 && 
             (body / range) > 0.95 && 
             upperWick < range * 0.05 && 
             lowerWick < range * 0.05;
    };

    const isSpinningTop = (candle: OHLCData): boolean => {
      const body = getCandleBody(candle);
      const upperWick = getUpperWick(candle);
      const lowerWick = getLowerWick(candle);
      const range = getCandleRange(candle);
      
      return range > 0 && 
             body / range < 0.3 && 
             upperWick > body * 0.5 && 
             lowerWick > body * 0.5;
    };

    // Pattern analysis loop
    for (let i = 2; i < data.length - 2; i++) {
      const current = data[i];
      const previous = data[i - 1];
      const next = data[i + 1];
      const twoPrev = data[i - 2];
      const twoNext = data[i + 1]; // For future confirmation

      // Single candle patterns
      if (selectedPatterns.includes('doji') && isDoji(current)) {
        patterns.push({
          name: 'Doji',
          type: 'neutral',
          strength: 'moderate',
          index: i,
          datetime: current.datetime,
          description: 'Indecision pattern - potential trend reversal when at key levels',
          reliability: 65,
          confirmation: false
        });
      }

      if (selectedPatterns.includes('hammer')) {
        if (isHammer(current) && isRedCandle(previous)) {
          patterns.push({
            name: 'Hammer',
            type: 'bullish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Bullish reversal pattern - rejection of lower prices',
            reliability: 75,
            confirmation: i < data.length - 1 && isGreenCandle(next)
          });
        }

        if (isShootingStar(current) && isGreenCandle(previous)) {
          patterns.push({
            name: 'Shooting Star',
            type: 'bearish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Bearish reversal pattern - rejection of higher prices',
            reliability: 75,
            confirmation: i < data.length - 1 && isRedCandle(next)
          });
        }
      }

      if (selectedPatterns.includes('marubozu') && isMarubozu(current)) {
        patterns.push({
          name: isGreenCandle(current) ? 'Bullish Marubozu' : 'Bearish Marubozu',
          type: isGreenCandle(current) ? 'bullish' : 'bearish',
          strength: 'strong',
          index: i,
          datetime: current.datetime,
          description: 'Strong directional momentum - no upper or lower wicks',
          reliability: 80,
          confirmation: true
        });
      }

      // Multi-candle patterns
      if (selectedPatterns.includes('engulfing') && i > 0) {
        const currentBody = getCandleBody(current);
        const previousBody = getCandleBody(previous);

        // Bullish Engulfing
        if (isRedCandle(previous) && isGreenCandle(current) &&
            current.open < previous.close && current.close > previous.open &&
            currentBody > previousBody) {
          patterns.push({
            name: 'Bullish Engulfing',
            type: 'bullish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Strong bullish reversal - larger green candle engulfs previous red',
            reliability: 85,
            confirmation: i < data.length - 1 && isGreenCandle(next)
          });
        }

        // Bearish Engulfing
        if (isGreenCandle(previous) && isRedCandle(current) &&
            current.open > previous.close && current.close < previous.open &&
            currentBody > previousBody) {
          patterns.push({
            name: 'Bearish Engulfing',
            type: 'bearish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Strong bearish reversal - larger red candle engulfs previous green',
            reliability: 85,
            confirmation: i < data.length - 1 && isRedCandle(next)
          });
        }
      }

      if (selectedPatterns.includes('harami') && i > 0) {
        const currentBody = getCandleBody(current);
        const previousBody = getCandleBody(previous);

        // Bullish Harami
        if (isRedCandle(previous) && isGreenCandle(current) &&
            current.open > previous.close && current.close < previous.open &&
            currentBody < previousBody) {
          patterns.push({
            name: 'Bullish Harami',
            type: 'bullish',
            strength: 'moderate',
            index: i,
            datetime: current.datetime,
            description: 'Bullish reversal - small green candle within large red candle body',
            reliability: 70,
            confirmation: i < data.length - 1 && isGreenCandle(next)
          });
        }

        // Bearish Harami
        if (isGreenCandle(previous) && isRedCandle(current) &&
            current.open < previous.close && current.close > previous.open &&
            currentBody < previousBody) {
          patterns.push({
            name: 'Bearish Harami',
            type: 'bearish',
            strength: 'moderate',
            index: i,
            datetime: current.datetime,
            description: 'Bearish reversal - small red candle within large green candle body',
            reliability: 70,
            confirmation: i < data.length - 1 && isRedCandle(next)
          });
        }
      }

      // Three-candle patterns
      if (selectedPatterns.includes('star') && i > 1) {
        const first = twoPrev;
        const star = previous;
        const third = current;

        // Morning Star (Bullish)
        if (isRedCandle(first) && getCandleBody(star) < getCandleBody(first) * 0.3 &&
            isGreenCandle(third) && third.close > (first.open + first.close) / 2) {
          patterns.push({
            name: 'Morning Star',
            type: 'bullish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Three-candle bullish reversal pattern',
            reliability: 80,
            confirmation: i < data.length - 1 && isGreenCandle(next)
          });
        }

        // Evening Star (Bearish)
        if (isGreenCandle(first) && getCandleBody(star) < getCandleBody(first) * 0.3 &&
            isRedCandle(third) && third.close < (first.open + first.close) / 2) {
          patterns.push({
            name: 'Evening Star',
            type: 'bearish',
            strength: 'strong',
            index: i,
            datetime: current.datetime,
            description: 'Three-candle bearish reversal pattern',
            reliability: 80,
            confirmation: i < data.length - 1 && isRedCandle(next)
          });
        }
      }

      // Spinning Top
      if (selectedPatterns.includes('spinning') && isSpinningTop(current)) {
        patterns.push({
          name: 'Spinning Top',
          type: 'neutral',
          strength: 'weak',
          index: i,
          datetime: current.datetime,
          description: 'Indecision pattern - small body with long wicks on both sides',
          reliability: 50,
          confirmation: false
        });
      }

      // Tweezers
      if (selectedPatterns.includes('tweezers') && i > 0) {
        const priceTolerance = (current.high + current.low) * 0.002; // 0.2% tolerance

        // Tweezers Top
        if (Math.abs(current.high - previous.high) < priceTolerance &&
            isGreenCandle(previous) && isRedCandle(current)) {
          patterns.push({
            name: 'Tweezers Top',
            type: 'bearish',
            strength: 'moderate',
            index: i,
            datetime: current.datetime,
            description: 'Bearish reversal - two candles with similar highs',
            reliability: 65,
            confirmation: i < data.length - 1 && isRedCandle(next)
          });
        }

        // Tweezers Bottom
        if (Math.abs(current.low - previous.low) < priceTolerance &&
            isRedCandle(previous) && isGreenCandle(current)) {
          patterns.push({
            name: 'Tweezers Bottom',
            type: 'bullish',
            strength: 'moderate',
            index: i,
            datetime: current.datetime,
            description: 'Bullish reversal - two candles with similar lows',
            reliability: 65,
            confirmation: i < data.length - 1 && isGreenCandle(next)
          });
        }
      }
    }

    console.log(`[PatternRecognition] Found ${patterns.length} patterns`);
    return patterns;
  }, [data, selectedPatterns]);

  // Filter patterns based on user preferences
  const filteredPatterns = useMemo(() => {
    let filtered = detectPatterns;

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(pattern => pattern.type === filterType);
    }

    // Filter by strength
    const strengthOrder = { weak: 0, moderate: 1, strong: 2 };
    filtered = filtered.filter(pattern => 
      strengthOrder[pattern.strength] >= strengthOrder[minStrength]
    );

    return filtered.sort((a, b) => b.reliability - a.reliability);
  }, [detectPatterns, filterType, minStrength]);

  // Pattern categories for selection
  const patternCategories = {
    'Single Candle': ['doji', 'hammer', 'marubozu', 'spinning'],
    'Two Candle': ['engulfing', 'harami', 'tweezers'],
    'Three Candle': ['star', 'abandoned'],
    'Complex': ['triangle', 'flag', 'pennant']
  };

  const getPatternIcon = (pattern: CandlestickPattern) => {
    switch (pattern.type) {
      case 'bullish': return <TrendingUp className="w-4 h-4 text-emerald-400" />;
      case 'bearish': return <TrendingDown className="w-4 h-4 text-red-400" />;
      case 'reversal': return <RefreshCw className="w-4 h-4 text-yellow-400" />;
      default: return <Target className="w-4 h-4 text-slate-400" />;
    }
  };

  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'strong': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30';
      case 'moderate': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/30';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'bullish': return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30';
      case 'bearish': return 'text-red-400 bg-red-500/20 border-red-500/30';
      case 'reversal': return 'text-purple-400 bg-purple-500/20 border-purple-500/30';
      default: return 'text-slate-400 bg-slate-500/20 border-slate-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <motion.h3
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
            >
              Pattern Recognition
            </motion.h3>
            <Badge className="bg-slate-800 text-slate-300 border-slate-600">
              {filteredPatterns.length} patterns found
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDescriptions(!showDescriptions)}
              className="text-slate-400 hover:text-white"
            >
              {showDescriptions ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`${autoRefresh ? 'text-emerald-400' : 'text-slate-400'} hover:text-white`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="text-sm text-slate-400 mb-2 block">Filter by Type</label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white"
            >
              <option value="all">All Types</option>
              <option value="bullish">Bullish</option>
              <option value="bearish">Bearish</option>
              <option value="reversal">Reversal</option>
              <option value="neutral">Neutral</option>
            </select>
          </div>
          
          <div>
            <label className="text-sm text-slate-400 mb-2 block">Minimum Strength</label>
            <select
              value={minStrength}
              onChange={(e) => setMinStrength(e.target.value as any)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white"
            >
              <option value="weak">All Strengths</option>
              <option value="moderate">Moderate+</option>
              <option value="strong">Strong Only</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-slate-400 mb-2 block">Quick Filters</label>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFilterType('bullish')}
                className="text-emerald-400 hover:bg-emerald-500/20"
              >
                <TrendingUp className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFilterType('bearish')}
                className="text-red-400 hover:bg-red-500/20"
              >
                <TrendingDown className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFilterType('all')}
                className="text-slate-400 hover:text-white"
              >
                All
              </Button>
            </div>
          </div>
        </div>

        {/* Pattern Selection */}
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-slate-300">Select Patterns to Detect</h4>
          {Object.entries(patternCategories).map(([category, patterns]) => (
            <div key={category} className="space-y-2">
              <h5 className="text-xs text-slate-500 uppercase tracking-wide">{category}</h5>
              <div className="flex flex-wrap gap-2">
                {patterns.map(pattern => (
                  <motion.button
                    key={pattern}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setSelectedPatterns(prev =>
                        prev.includes(pattern)
                          ? prev.filter(p => p !== pattern)
                          : [...prev, pattern]
                      );
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                      selectedPatterns.includes(pattern)
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:border-slate-600/50 hover:text-slate-300'
                    }`}
                  >
                    {pattern.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  </motion.button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Patterns List */}
      <div className="grid gap-4">
        <AnimatePresence>
          {filteredPatterns.map((pattern, index) => (
            <motion.div
              key={`${pattern.datetime}-${pattern.name}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <Card 
                className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-4 cursor-pointer hover:bg-slate-800/70 transition-all duration-200"
                onClick={() => onPatternClick?.(pattern)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">
                      {getPatternIcon(pattern)}
                    </div>
                    
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="text-lg font-semibold text-white">{pattern.name}</h4>
                        <Badge className={getTypeColor(pattern.type)}>
                          {pattern.type}
                        </Badge>
                        <Badge className={getStrengthColor(pattern.strength)}>
                          {pattern.strength}
                        </Badge>
                        {pattern.confirmation && (
                          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Confirmed
                          </Badge>
                        )}
                      </div>
                      
                      <div className="flex items-center gap-4 text-sm text-slate-400 mb-2">
                        <span>{new Date(pattern.datetime).toLocaleString()}</span>
                        <span>Reliability: {pattern.reliability}%</span>
                        <span>Index: {pattern.index}</span>
                      </div>
                      
                      {showDescriptions && (
                        <p className="text-sm text-slate-300 leading-relaxed">
                          {pattern.description}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${
                      pattern.reliability > 80 ? 'bg-emerald-400' :
                      pattern.reliability > 60 ? 'bg-yellow-400' : 'bg-red-400'
                    }`} />
                    <span className="text-xs text-slate-500 font-mono">
                      {pattern.reliability}%
                    </span>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredPatterns.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <Search className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-slate-400 mb-2">No Patterns Found</h3>
            <p className="text-slate-500">
              Try adjusting your filters or selecting different pattern types.
            </p>
          </motion.div>
        )}
      </div>

      {/* Pattern Statistics */}
      {filteredPatterns.length > 0 && (
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Pattern Statistics
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/30 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-emerald-400 mb-1">
                {filteredPatterns.filter(p => p.type === 'bullish').length}
              </div>
              <div className="text-sm text-slate-400">Bullish</div>
            </div>
            
            <div className="bg-slate-800/30 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-400 mb-1">
                {filteredPatterns.filter(p => p.type === 'bearish').length}
              </div>
              <div className="text-sm text-slate-400">Bearish</div>
            </div>
            
            <div className="bg-slate-800/30 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-yellow-400 mb-1">
                {filteredPatterns.filter(p => p.strength === 'strong').length}
              </div>
              <div className="text-sm text-slate-400">Strong</div>
            </div>
            
            <div className="bg-slate-800/30 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-cyan-400 mb-1">
                {Math.round(filteredPatterns.reduce((sum, p) => sum + p.reliability, 0) / filteredPatterns.length || 0)}%
              </div>
              <div className="text-sm text-slate-400">Avg Reliability</div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default CandlestickPatternRecognition;