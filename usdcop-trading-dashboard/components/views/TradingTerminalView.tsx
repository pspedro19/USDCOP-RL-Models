/**
 * Complete Trading Terminal View with Integrated Replay Controls
 * Combines chart, replay controls, and data management in one interface
 */

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';
import { ReplayControls } from '@/components/controls/ReplayControls';
import { useMarketStore } from '@/lib/store/market-store';
import { enhancedDataService, EnhancedCandle } from '@/lib/services/enhanced-data-service';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  BarChart3, 
  Clock, 
  AlertCircle,
  Database,
  Signal,
  Globe,
  Layers,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

// Dynamic imports for performance
const InteractiveTradingChart = dynamic(
  () => import('@/components/charts/InteractiveTradingChart').then(mod => ({ default: mod.InteractiveTradingChart })), 
  {
    ssr: false,
    loading: () => (
      <div className="h-full bg-slate-900/50 animate-pulse rounded-xl flex items-center justify-center">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-2 animate-pulse" />
          <p className="text-slate-400">Loading Chart...</p>
        </div>
      </div>
    )
  }
);

export const TradingTerminalView: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRealtime, setIsRealtime] = useState(false);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market' | 'after-hours'>('closed');
  const [historicalData, setHistoricalData] = useState<EnhancedCandle[]>([]);
  const [displayData, setDisplayData] = useState<EnhancedCandle[]>([]);
  const [dataStats, setDataStats] = useState({
    total: 0,
    minio: 0,
    twelvedata: 0,
    realtime: 0,
    dateRange: {
      start: '',
      end: ''
    }
  });

  const {
    candles,
    setCandles,
    appendCandle,
    currentPrice,
    priceChange,
    volume24h,
    high24h,
    low24h,
    connectionStatus,
    setConnectionStatus,
    dataSource,
    setDataSource
  } = useMarketStore();

  // Calculate price change percentage
  const priceChangePercent = candles.length > 1 
    ? ((candles[candles.length - 1].close - candles[0].close) / candles[0].close) * 100 
    : 0;

  // Load initial data on mount
  useEffect(() => {
    loadInitialData();
    checkMarketStatus();
    
    // Check market status every minute
    const statusInterval = setInterval(checkMarketStatus, 60000);
    
    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  // Update data stats when candles change
  useEffect(() => {
    if (candles.length > 0) {
      const stats = {
        total: candles.length,
        minio: candles.filter(d => d.source === 'minio').length,
        twelvedata: candles.filter(d => d.source === 'twelvedata').length,
        realtime: candles.filter(d => d.source === 'realtime').length,
        dateRange: {
          start: candles[0]?.datetime || '',
          end: candles[candles.length - 1]?.datetime || ''
        }
      };
      setDataStats(stats);
      
      console.log(`[TradingTerminal] Data stats updated:`, stats);
      if (stats.total > 80000) {
        console.log(`[TradingTerminal] ✅ Full historical dataset loaded: ${stats.total.toLocaleString()} points from ${stats.dateRange.start} to ${stats.dateRange.end}`);
      }
    }
  }, [candles]);

  const loadInitialData = async () => {
    console.log('[TradingTerminal] Loading initial market data...');
    setIsLoading(true);
    
    try {
      // Load from API first
      const response = await fetch('/api/data/historical');
      let data: EnhancedCandle[] = [];
      
      if (response.ok) {
        const result = await response.json();
        data = result.data || [];
        console.log(`[TradingTerminal] Loaded ${data.length} historical points from API`);
      } else {
        // Fallback to service
        data = await enhancedDataService.loadCompleteHistory();
        console.log(`[TradingTerminal] Loaded ${data.length} historical points from service`);
      }
      
      // Process and clean data
      const uniqueDataMap = new Map();
      data.forEach(item => {
        const timestamp = new Date(item.datetime).getTime();
        if (!uniqueDataMap.has(timestamp) || item.close > 0) {
          uniqueDataMap.set(timestamp, item);
        }
      });
      
      const cleanedData = Array.from(uniqueDataMap.values())
        .sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
      
      console.log(`[TradingTerminal] Cleaned data: ${cleanedData.length} unique points`);
      
      // Store and display ALL historical data
      setHistoricalData(cleanedData);
      setDisplayData(cleanedData);
      setCandles(cleanedData);
      
      if (cleanedData.length > 0) {
        const firstPoint = cleanedData[0];
        const lastPoint = cleanedData[cleanedData.length - 1];
        console.log(`[TradingTerminal] Full data range: ${firstPoint.datetime} to ${lastPoint.datetime}`);
        
        // Log milestone if we have the full dataset
        if (cleanedData.length > 80000) {
          console.log(`[TradingTerminal] 🎉 COMPLETE DATASET LOADED: ${cleanedData.length.toLocaleString()} points spanning 2020-2025`);
        }
      }
      
    } catch (error) {
      console.error('[TradingTerminal] Failed to load data:', error);
      setHistoricalData([]);
      setDisplayData([]);
      setCandles([]);
    } finally {
      setIsLoading(false);
    }
  };

  const checkMarketStatus = () => {
    const now = new Date();
    const hour = now.getHours();
    const minutes = now.getMinutes();
    const totalMinutes = hour * 60 + minutes;
    const day = now.getDay();
    
    if (day === 0 || day === 6) {
      setMarketStatus('closed');
      return;
    }
    
    // Trading hours: 8:00 AM - 12:55 PM (Colombia time)
    if (totalMinutes >= 480 && totalMinutes <= 775) {
      setMarketStatus('open');
    } else if (totalMinutes >= 420 && totalMinutes < 480) {
      setMarketStatus('pre-market');
    } else if (totalMinutes > 775 && totalMinutes <= 840) {
      setMarketStatus('after-hours');
    } else {
      setMarketStatus('closed');
    }
  };

  const handleAlignedDatasetClick = async () => {
    console.log('[TradingTerminal] Aligning dataset with real-time data...');
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/market/realtime?action=align');
      if (!response.ok) throw new Error('Failed to fetch aligned data');
      
      const result = await response.json();
      const allData = result.data || [];
      const meta = result.meta || {};
      
      console.log(`[TradingTerminal] Aligned dataset:`, meta);
      
      if (allData.length > 0) {
        setHistoricalData(allData);
        setDisplayData(allData);
        setCandles(allData);
        
        console.log(`[TradingTerminal] ✅ Dataset aligned with ${allData.length.toLocaleString()} total points`);
      }
    } catch (error) {
      console.error('[TradingTerminal] Failed to align dataset:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRealtimeToggle = (enabled: boolean) => {
    console.log(`[TradingTerminal] Real-time updates ${enabled ? 'enabled' : 'disabled'}`);
    setIsRealtime(enabled);
    
    if (enabled) {
      // Start real-time updates
      fetch('/api/market/realtime?action=start')
        .then(res => res.json())
        .then(result => {
          console.log('[TradingTerminal] Real-time updates started:', result.schedule);
        });
    } else {
      // Stop real-time updates
      fetch('/api/market/realtime?action=stop')
        .then(res => res.json())
        .then(result => {
          console.log('[TradingTerminal] Real-time updates stopped');
        });
    }
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/10 via-transparent to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-900/10 via-transparent to-transparent pointer-events-none" />
      
      {/* Header with enhanced market info */}
      <motion.div 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-slate-900/50 backdrop-blur-xl border-b border-slate-700/50 p-4 relative z-10"
      >
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
                <TrendingUp className="w-7 h-7 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full border-2 border-slate-900 animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                USD/COP Trading Terminal
              </h1>
              <p className="text-slate-400">Complete historical dataset with replay controls</p>
            </div>
          </div>
          
          {/* Enhanced Price Display */}
          <div className="flex items-center gap-4">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-3 border border-slate-700/50">
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-bold text-cyan-400">
                  ${currentPrice.toLocaleString()}
                </span>
                <span className={`text-sm font-medium px-2 py-1 rounded ${
                  priceChangePercent >= 0 
                    ? 'text-emerald-300 bg-emerald-500/20' 
                    : 'text-red-300 bg-red-500/20'
                }`}>
                  {priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
                </span>
              </div>
            </div>
            
            {/* Market Status */}
            <div className={`flex items-center gap-2 px-3 py-2 rounded-xl backdrop-blur-sm border ${
              marketStatus === 'open' 
                ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                : marketStatus === 'pre-market' || marketStatus === 'after-hours'
                ? 'bg-amber-500/20 border-amber-500/30 text-amber-300'
                : 'bg-red-500/20 border-red-500/30 text-red-300'
            }`}>
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                marketStatus === 'open' ? 'bg-emerald-400'
                : marketStatus === 'pre-market' || marketStatus === 'after-hours' ? 'bg-amber-400'
                : 'bg-red-400'
              }`} />
              <span className="text-xs font-medium uppercase tracking-wider">
                {marketStatus.replace('-', ' ')}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main content area with ReplayControls and Chart */}
      <div className="flex-1 flex flex-col lg:flex-row gap-4 p-4 relative z-10">
        {/* Left side - Replay Controls and Data Stats */}
        <motion.div 
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="lg:w-96 flex flex-col gap-4"
        >
          {/* Replay Controls */}
          <ReplayControls
            onAlignedDatasetClick={handleAlignedDatasetClick}
            onRealtimeToggle={handleRealtimeToggle}
          />
          
          {/* Extended Data Statistics */}
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider font-bold mb-4">
              Dataset Statistics
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Total Points</span>
                <span className="text-white font-mono font-bold">
                  {dataStats.total.toLocaleString()}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm flex items-center gap-1">
                  <Database size={12} />
                  Historical (MinIO)
                </span>
                <span className="text-emerald-400 font-mono">
                  {dataStats.minio.toLocaleString()}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm flex items-center gap-1">
                  <Globe size={12} />
                  TwelveData API
                </span>
                <span className="text-purple-400 font-mono">
                  {dataStats.twelvedata.toLocaleString()}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm flex items-center gap-1">
                  <Signal size={12} />
                  Real-time
                </span>
                <span className="text-yellow-400 font-mono">
                  {dataStats.realtime.toLocaleString()}
                </span>
              </div>
              
              {dataStats.dateRange.start && (
                <div className="pt-3 border-t border-slate-700/50">
                  <div className="text-slate-400 text-xs mb-2">Date Range</div>
                  <div className="text-xs text-slate-300 font-mono">
                    <div>{new Date(dataStats.dateRange.start).toLocaleDateString()}</div>
                    <div className="text-slate-500">to</div>
                    <div>{new Date(dataStats.dateRange.end).toLocaleDateString()}</div>
                  </div>
                </div>
              )}
              
              {dataStats.total > 80000 && (
                <div className="mt-3 p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                  <div className="text-emerald-400 text-xs font-semibold flex items-center gap-1">
                    <Activity size={10} />
                    Complete Dataset Active
                  </div>
                  <div className="text-emerald-300 text-xs mt-1">
                    Full 2020-2025 historical data loaded
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Right side - Trading Chart */}
        <motion.div 
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex-1 flex flex-col"
        >
          {isLoading ? (
            <div className="h-full bg-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-8 flex items-center justify-center">
              <div className="text-center">
                <motion.div
                  className="w-16 h-16 border-4 border-slate-700 border-t-cyan-500 rounded-full mx-auto mb-4"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
                <h3 className="text-xl font-bold text-white mb-2">Loading Market Data</h3>
                <p className="text-slate-400">Connecting to data sources...</p>
              </div>
            </div>
          ) : displayData.length > 0 ? (
            <div className="h-full bg-slate-900/30 backdrop-blur-xl border border-slate-700/50 rounded-xl overflow-hidden relative">
              {/* Chart Header */}
              <div className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/30 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Layers className="w-5 h-5 text-cyan-400" />
                      <span className="text-white font-semibold">USD/COP • 5min</span>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg">
                      <BarChart3 className="w-4 h-4 text-slate-400" />
                      <span className="text-sm text-slate-300">
                        {displayData.length.toLocaleString()} candles
                      </span>
                    </div>
                  </div>
                  
                  {isRealtime && (
                    <motion.div
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="flex items-center gap-2 px-3 py-1 bg-emerald-500/20 rounded-lg border border-emerald-500/30"
                    >
                      <Signal className="w-3 h-3 text-emerald-400" />
                      <span className="text-xs text-emerald-300 font-medium">LIVE</span>
                    </motion.div>
                  )}
                </div>
              </div>
              
              {/* Chart Container */}
              <div className="h-full pt-16">
                <InteractiveTradingChart 
                  data={displayData}
                  isRealtime={isRealtime}
                  onRangeChange={(start, end) => {
                    console.log(`[Chart] Range: ${start.toISOString()} to ${end.toISOString()}`);
                  }}
                />
              </div>
            </div>
          ) : (
            <div className="h-full bg-slate-900/50 backdrop-blur-xl border border-slate-700/50 rounded-xl p-8 flex items-center justify-center">
              <div className="text-center">
                <AlertCircle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">No Data Available</h3>
                <p className="text-slate-400 mb-6">Unable to load market data. Check connection and try again.</p>
                <motion.button
                  onClick={loadInitialData}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-purple-600 text-white rounded-xl font-semibold"
                >
                  Reload Data
                </motion.button>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Bottom Status Bar */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="bg-slate-900/50 backdrop-blur-xl border-t border-slate-700/50 p-4 relative z-10"
      >
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Left side - Connection and Data Info */}
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
              connectionStatus === 'connected' 
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'bg-red-500/20 text-red-300'
            }`}>
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                connectionStatus === 'connected' ? 'bg-emerald-400' : 'bg-red-400'
              }`} />
              <span className="text-xs font-medium">{connectionStatus}</span>
            </div>
            
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg">
              <Clock className="w-3 h-3 text-slate-400" />
              <span className="text-xs text-slate-300">5min intervals</span>
            </div>
          </div>
          
          {/* Right side - Market Metrics */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg">
              <BarChart3 className="w-3 h-3 text-cyan-400" />
              <span className="text-xs text-slate-400">Vol</span>
              <span className="text-xs text-cyan-300 font-mono">
                ${(volume24h / 1000000).toFixed(1)}M
              </span>
            </div>
            
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg">
              <ArrowUpRight className="w-3 h-3 text-emerald-400" />
              <span className="text-xs text-slate-400">H</span>
              <span className="text-xs text-emerald-300 font-mono">${high24h.toFixed(2)}</span>
            </div>
            
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg">
              <ArrowDownRight className="w-3 h-3 text-red-400" />
              <span className="text-xs text-slate-400">L</span>
              <span className="text-xs text-red-300 font-mono">${low24h.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default TradingTerminalView;