/**
 * Enhanced Trading Dashboard
 * Ultra-dynamic interface with real-time updates and historical replay
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import { useMarketStore } from '@/lib/store/market-store';
import { enhancedDataService, EnhancedCandle } from '@/lib/services/enhanced-data-service';
import { AnimatedSidebar } from '@/components/ui/AnimatedSidebar';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
// Removed imports for deleted components
// import { ChartErrorBoundary } from '@/components/common/ErrorBoundary';
// import PerformanceStatus from '@/components/status/PerformanceStatus';
import { TrendingUp, TrendingDown, Activity, BarChart3, Clock, AlertCircle, Zap, Signal, Target, Shield, Layers, Globe, ArrowUpRight, ArrowDownRight } from 'lucide-react';

// Dynamic imports for performance
const LightweightChart = dynamic(() => import('@/components/charts/LightweightChart'), {
  ssr: false,
  loading: () => <div className="h-full bg-gray-900 animate-pulse rounded-xl" />
});

const InteractiveTradingChart = dynamic(() => import('@/components/charts/InteractiveTradingChart').then(mod => ({ default: mod.InteractiveTradingChart })), {
  ssr: false,
  loading: () => <div className="h-full bg-gray-900 animate-pulse rounded-xl" />
});

// NO SYNTHETIC DATA GENERATION ALLOWED
// All data must be real from MinIO or TwelveData API

const EnhancedTradingDashboard: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRealtime, setIsRealtime] = useState(false);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market' | 'after-hours'>('closed');
  const [replayProgress, setReplayProgress] = useState(0);
  const [historicalData, setHistoricalData] = useState<EnhancedCandle[]>([]);
  const [displayData, setDisplayData] = useState<EnhancedCandle[]>([]);
  const [useInteractiveChart, setUseInteractiveChart] = useState(true); // Use interactive chart by default
  const replayIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const realtimeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const replayIndexRef = useRef(0);
  
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
      stopReplay();
      stopRealtime();
    };
  }, []);

  const loadInitialData = async () => {
    setIsLoading(true);
    try {
      // Try to load from API first
      const response = await fetch('/api/data/historical');
      let data: EnhancedCandle[] = [];
      
      if (response.ok) {
        const result = await response.json();
        data = result.data || [];
        console.log(`[Dashboard] Loaded ${data.length} historical data points from API`);
        // Log date range to debug
        if (data.length > 0) {
          console.log(`[Dashboard] Date range: ${data[0].datetime} to ${data[data.length - 1].datetime}`);
          console.log(`[Dashboard] First data point:`, data[0]);
          console.log(`[Dashboard] Last data point:`, data[data.length - 1]);
        }
      } else {
        // Fallback to service if API fails
        data = await enhancedDataService.loadCompleteHistory();
        console.log(`[Dashboard] Loaded ${data.length} historical data points from service`);
      }
      
      // Process data in chunks to avoid blocking
      requestAnimationFrame(() => {
        // Remove duplicates and sort data by datetime
        const uniqueDataMap = new Map();
        data.forEach(item => {
          const timestamp = new Date(item.datetime).getTime();
          if (!uniqueDataMap.has(timestamp) || item.close > 0) { // Prefer non-zero data
            uniqueDataMap.set(timestamp, item);
          }
        });
        
        // Convert map back to array and sort
        const cleanedData = Array.from(uniqueDataMap.values())
          .sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
        
        console.log(`[Dashboard] Cleaned data: ${cleanedData.length} unique points from ${data.length} total`);
        
        // Store full historical data
        setHistoricalData(cleanedData);
        
        // Display ALL historical data (2020-2025)
        console.log(`[Dashboard] Setting display data: ${cleanedData.length} points`);
        if (cleanedData.length > 0) {
          const firstPoint = cleanedData[0];
          const lastPoint = cleanedData[cleanedData.length - 1];
          console.log(`[Dashboard] Full data range: ${firstPoint.datetime} to ${lastPoint.datetime}`);
        }
        
        setCandles(cleanedData); // No slicing - show ALL data
        setDisplayData(cleanedData); // No slicing - show ALL data
      });
    } catch (error) {
      console.error('[Dashboard] ❌ Failed to load data:', error);
      console.error('[Dashboard] NO mock data will be generated - using only real data');
      // NO MOCK DATA - Better to show empty than fake data
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
    
    // Market closed on weekends
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

  const handlePlayPause = () => {
    if (isPlaying) {
      stopReplay();
    } else {
      startReplay();
    }
  };

  const startReplay = () => {
    if (historicalData.length === 0) return;
    
    console.log('[Dashboard] Starting FULL historical replay from 2020');
    setIsPlaying(true);
    stopRealtime(); // Stop realtime if active
    
    // Start from the BEGINNING (index 0) to replay ALL historical data from 2020
    const startIndex = 0; // Start from 2020!
    console.log(`[Dashboard] Starting replay from beginning: ${historicalData.length} total points`);
    
    // Initialize with first batch of data
    const initialBatchSize = 500; // Show first 500 candles
    const initialData = historicalData.slice(0, initialBatchSize);
    setDisplayData(initialData);
    setCandles(initialData);
    replayIndexRef.current = initialBatchSize;
    
    // Replay with adaptive batch size for smoother animation
    replayIntervalRef.current = setInterval(() => {
      if (replayIndexRef.current >= historicalData.length) {
        console.log('[Dashboard] Replay complete');
        stopReplay();
        return;
      }
      
      // Adaptive batch size based on remaining data
      const remaining = historicalData.length - replayIndexRef.current;
      // Larger batch sizes for faster replay of 84,455 points
      const batchSize = Math.min(remaining > 10000 ? 500 : remaining > 1000 ? 100 : 20, remaining);
      const endIndex = replayIndexRef.current + batchSize;
      
      // Use requestAnimationFrame for smoother updates
      requestAnimationFrame(() => {
        const newData = historicalData.slice(0, endIndex); // Show from beginning to current position
        setDisplayData(newData);
        setCandles(newData); // Keep ALL data for full historical view
        
        // Update progress
        const progress = (endIndex / historicalData.length) * 100;
        setReplayProgress(progress);
      });
      
      replayIndexRef.current = endIndex;
    }, 50); // Faster updates but smaller batches
  };

  const stopReplay = () => {
    if (replayIntervalRef.current) {
      clearInterval(replayIntervalRef.current);
      replayIntervalRef.current = null;
    }
    setIsPlaying(false);
    setReplayProgress(0);
  };

  const processAlignedData = (allData: any[]) => {
    // Find last available date in dataset
    const lastDataPoint = allData[allData.length - 1];
    const lastDate = new Date(lastDataPoint.datetime);
    const today = new Date();
    
    // Calculate gap in days
    const gapInMs = today.getTime() - lastDate.getTime();
    const gapInDays = Math.floor(gapInMs / (1000 * 60 * 60 * 24));
    
    console.log(`[Dashboard] Last data: ${lastDate.toISOString()}, Today: ${today.toISOString()}`);
    console.log(`[Dashboard] Data gap: ${gapInDays} days`);
    console.log(`[Dashboard] Total data points received: ${allData.length}`);
    
    // IMPORTANT: The backend already filtered for trading hours
    // DO NOT filter again in the frontend - just use the data as-is
    const alignedData = allData;
    
    console.log(`[Dashboard] Using all ${alignedData.length} points from backend (already filtered for trading hours)`);
    
    // Store all the aligned data
    setHistoricalData(alignedData);
    
    // Display ALL historical data - no slicing!
    const dataToDisplay = alignedData; // Show ALL data from 2020-2025
    
    console.log(`[Dashboard] Displaying last ${dataToDisplay.length} points`);
    
    console.log(`[Dashboard] Displaying last ${dataToDisplay.length} points from dataset`);
    if (dataToDisplay.length > 0) {
      const firstPoint = dataToDisplay[0];
      const lastPoint = dataToDisplay[dataToDisplay.length - 1];
      console.log(`[Dashboard] Display range: ${firstPoint.datetime} to ${lastPoint.datetime}`);
      console.log(`[Dashboard] First displayed: ${new Date(firstPoint.datetime).toLocaleString()} - Close: ${firstPoint.close}`);
      console.log(`[Dashboard] Last displayed: ${new Date(lastPoint.datetime).toLocaleString()} - Close: ${lastPoint.close}`);
      
      // Check if we have today's data
      const today = new Date();
      const todayStr = today.toISOString().split('T')[0];
      const hasToday = dataToDisplay.some((p: any) => p.datetime.startsWith(todayStr));
      console.log(`[Dashboard] Has today's data (${todayStr}): ${hasToday}`);
    }
    
    setDisplayData(dataToDisplay);
    setCandles(dataToDisplay);
    
    // Only start real-time if data is recent (less than 1 day old) AND market is open
    const dataIsRecent = gapInDays <= 1;
    const currentHour = today.getHours();
    const currentMinutes = today.getMinutes();
    const currentTotalMinutes = currentHour * 60 + currentMinutes;
    const currentDay = today.getDay();
    const marketIsOpen = currentDay >= 1 && currentDay <= 5 && currentTotalMinutes >= 480 && currentTotalMinutes <= 775;
    
    if (dataIsRecent && marketIsOpen) {
      console.log('[Dashboard] ✅ Data is recent and market is open, starting real-time updates');
      startRealtime();
    } else if (!dataIsRecent) {
      console.log(`[Dashboard] ⚠️ Data is ${gapInDays} days old, real-time disabled until fresh data is available`);
    } else if (!marketIsOpen) {
      console.log('[Dashboard] ⏸️ Market is closed, real-time updates disabled');
    }
  };

  const handleReset = () => {
    console.log('[Dashboard] Resetting to current data');
    stopReplay();
    stopRealtime();
    
    // Reset to show last 7 days of data
    const today = new Date();
    const daysToShow = 7;
    const cutoffDate = new Date(today);
    cutoffDate.setDate(cutoffDate.getDate() - daysToShow);
    
    const recentData = historicalData.filter((point: any) => {
      const pointDate = new Date(point.datetime);
      return pointDate >= cutoffDate;
    });
    
    const dataToDisplay = recentData; // Display ALL data without limits
    
    console.log(`[Dashboard] Reset: Showing ${dataToDisplay.length} points from last ${daysToShow} days`);
    setDisplayData(dataToDisplay);
    setCandles(dataToDisplay);
    setReplayProgress(0);
  };

  const handleAlignDataset = async () => {
    console.log('[Dashboard] Starting complete data alignment process...');
    setIsLoading(true);
    
    try {
      // SIMPLIFIED: Go directly to align endpoint which handles everything
      console.log('[Dashboard] Fetching aligned dataset with real-time data...');
      const response = await fetch('/api/market/realtime?action=align');
      if (!response.ok) throw new Error('Failed to fetch aligned data');
      
      const result = await response.json();
      const allData = result.data || [];
      const meta = result.meta || {};
      
      console.log(`[Dashboard] Aligned dataset loaded:`);
      console.log(`  - Total: ${meta.total} points`);
      console.log(`  - Historical: ${meta.historical} points`);
      console.log(`  - Realtime: ${meta.realtime} points`);
      console.log(`  - Date range: ${meta.startDate} to ${meta.endDate}`);
      
      if (allData.length === 0) {
        console.error('[Dashboard] No data available');
        return;
      }
      
      // Log the last 5 data points to verify we have today's data
      console.log('[Dashboard] Last 5 data points received:');
      const last5 = allData.slice(-5);
      last5.forEach((point: any) => {
        console.log(`  ${point.datetime}: Close=${point.close}, Source=${point.source}`);
      });
      
      processAlignedData(allData);
    } catch (error) {
      console.error('[Dashboard] Failed to align dataset:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startRealtime = () => {
    if (marketStatus === 'closed') {
      console.log('[Dashboard] Market is closed, real-time updates disabled');
      return;
    }
    
    setIsRealtime(true);
    stopReplay(); // Stop replay if active
    
    // Start realtime updates on the server
    fetch('/api/market/realtime?action=start')
      .then(res => res.json())
      .then(result => {
        console.log('[Dashboard] Real-time updates configured:', result.schedule);
      });
    
    // Update every minute to check for new 5-minute candles
    realtimeIntervalRef.current = setInterval(async () => {
      const now = new Date();
      const minutes = now.getMinutes();
      
      // Only fetch at 5-minute intervals (00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
      if (minutes % 5 === 0) {
        if (marketStatus === 'closed') {
          console.log('[Dashboard] Market closed, stopping real-time updates');
          stopRealtime();
          return;
        }
        
        try {
          // Fetch latest data point from TwelveData
          const response = await fetch('/api/market/realtime?action=fetch');
          if (response.ok) {
            const result = await response.json();
            const latestData = result.data || [];
            
            if (latestData.length > 0) {
              const latestPoint = latestData[latestData.length - 1];
              console.log(`[Dashboard] New real-time candle at ${now.toLocaleTimeString()}:`, latestPoint);
              
              // Check if this is a new candle (not duplicate)
              const exists = displayData.some(d => d.datetime === latestPoint.datetime);
              if (!exists) {
                appendCandle(latestPoint);
                setDisplayData(prev => [...prev, latestPoint]); // Keep ALL data
                setHistoricalData(prev => [...prev, latestPoint]);
              }
            } else {
              console.error('[Dashboard] No real-time data available - API key may be invalid');
            }
          } else {
            const errorData = await response.json();
            console.error('[Dashboard] API Error:', errorData.message || 'Unknown error');
            // Show error once per session
            if (!window.apiErrorShown) {
              window.apiErrorShown = true;
              alert('⚠️ No se pueden obtener datos en tiempo real\n\n' +
                    'Necesitas una API key válida de TwelveData (es GRATIS)\n\n' +
                    '1. Ve a: https://twelvedata.com/pricing\n' +
                    '2. Regístrate gratis\n' +
                    '3. Actualiza .env.local:\n' +
                    '   TWELVE_DATA_API_KEY=tu_clave_aqui\n' +
                    '4. Reinicia el servidor');
            }
          }
        } catch (error) {
          console.error('[Dashboard] Real-time update failed:', error);
        }
      }
    }, 60000); // Check every minute
    
    console.log('[Dashboard] Real-time updates started (5-minute candles at :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55)');
  };

  const stopRealtime = () => {
    if (realtimeIntervalRef.current) {
      clearInterval(realtimeIntervalRef.current);
      realtimeIntervalRef.current = null;
    }
    setIsRealtime(false);
  };

  const handleDataSourceChange = async (source: 'l0' | 'l1' | 'mock') => {
    if (source === 'mock') {
      console.error('[Dashboard] ❌ Mock data is PROHIBITED. Only real data allowed.');
      alert('Mock data is not allowed. Please use L0 or L1 real data sources.');
      return;
    }
    
    console.log(`[Dashboard] Switching data source to ${source}`);
    setDataSource(source);
    
    // Reload data from new source - REAL DATA ONLY
    setIsLoading(true);
    try {
      const response = await fetch(`/api/data/historical?source=${source}`);
      let data: EnhancedCandle[] = [];
      
      if (response.ok) {
        const result = await response.json();
        data = result.data || [];
        console.log(`[Dashboard] Loaded ${data.length} REAL data points from ${source}`);
      } else {
        console.error(`[Dashboard] Failed to load data from ${source}`);
        data = [];
      }
      
      setHistoricalData(data);
      const recentData = data; // Show ALL historical data
      setCandles(recentData);
      setDisplayData(recentData);
    } catch (error) {
      console.error('[Dashboard] Failed to switch data source:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-hidden relative">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-900/20 via-transparent to-transparent" />
      {/* Animated Sidebar */}
      <AnimatedSidebar
        onPlayPause={handlePlayPause}
        onReset={handleReset}
        onAlignDataset={handleAlignDataset}
        isPlaying={isPlaying}
        isRealtime={isRealtime}
        dataSource={dataSource}
        onDataSourceChange={handleDataSourceChange}
        marketStatus={marketStatus}
        currentPrice={currentPrice}
        priceChange={priceChange}
        priceChangePercent={priceChangePercent}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <motion.header 
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="bg-slate-900/30 backdrop-blur-xl border-b border-slate-700/50 px-4 sm:px-6 py-3 sm:py-4 relative overflow-hidden"
        >
          {/* Header Glow Effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-amber-500/5" />
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-0 relative z-10">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-6">
              <motion.div 
                className="flex items-center gap-4"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.5 }}
              >
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/25">
                    <TrendingUp className="w-7 h-7 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full border-2 border-slate-900 animate-pulse" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">USD/COP Trading Terminal</h1>
                  <p className="text-slate-400 text-sm">Professional Trading Platform</p>
                </div>
              </motion.div>
              
              {/* Enhanced Status Indicators */}
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                {isPlaying && (
                  <motion.div
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 text-amber-300 rounded-xl backdrop-blur-sm shadow-lg shadow-amber-500/10"
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Activity className="w-4 h-4" />
                    </motion.div>
                    <span className="text-sm font-medium">Replaying {replayProgress.toFixed(1)}%</span>
                    <div className="w-16 h-1 bg-amber-900/50 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-amber-400 to-orange-400 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${replayProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </motion.div>
                )}
                
                {isRealtime && (
                  <motion.div
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0, opacity: 0 }}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500/20 to-green-500/20 border border-emerald-500/30 text-emerald-300 rounded-xl backdrop-blur-sm shadow-lg shadow-emerald-500/10"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <Signal className="w-4 h-4" />
                    </motion.div>
                    <span className="text-sm font-medium">Live Data</span>
                    <motion.div
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="flex items-center gap-1"
                    >
                      <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                      <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                      <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                    </motion.div>
                  </motion.div>
                )}
                
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl shadow-lg"
                >
                  <Clock className="w-4 h-4 text-slate-400" />
                  <span className="text-sm text-slate-300 font-mono">{new Date().toLocaleTimeString()}</span>
                </motion.div>
                
                {/* Market Status Indicator */}
                <motion.div 
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl backdrop-blur-sm border shadow-lg ${
                    marketStatus === 'open' 
                      ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 shadow-emerald-500/10'
                      : marketStatus === 'pre-market' || marketStatus === 'after-hours'
                      ? 'bg-amber-500/20 border-amber-500/30 text-amber-300 shadow-amber-500/10'
                      : 'bg-red-500/20 border-red-500/30 text-red-300 shadow-red-500/10'
                  }`}
                >
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className={`w-2 h-2 rounded-full ${
                      marketStatus === 'open' ? 'bg-emerald-400'
                      : marketStatus === 'pre-market' || marketStatus === 'after-hours' ? 'bg-amber-400'
                      : 'bg-red-400'
                    }`}
                  />
                  <span className="text-xs font-medium uppercase tracking-wider">
                    {marketStatus.replace('-', ' ')}
                  </span>
                </motion.div>
              </div>
            </div>
            
            {/* Performance Status */}
            {/* <PerformanceStatus /> */}
          </div>
        </motion.header>
        
        {/* Chart Area */}
        <motion.div 
          className="flex-1 p-3 sm:p-6 overflow-hidden relative"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
        >
          {/* Floating Performance Metrics */}
          <div className="absolute top-4 right-4 sm:top-8 sm:right-8 z-10 flex flex-col gap-3 hidden lg:flex">
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="bg-slate-900/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-4 shadow-2xl"
            >
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 bg-gradient-to-br from-green-400 to-emerald-500 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-4 h-4 text-white" />
                </div>
                <span className="text-slate-300 text-sm font-medium">24H Performance</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                  ${currentPrice.toLocaleString()}
                </span>
                <span className={`text-sm font-medium px-2 py-1 rounded-lg ${
                  priceChangePercent >= 0 
                    ? 'text-emerald-300 bg-emerald-500/20' 
                    : 'text-red-300 bg-red-500/20'
                }`}>
                  {priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
                </span>
              </div>
            </motion.div>
            
            {/* Data Quality Indicator */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7 }}
              className="bg-slate-900/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-3 shadow-2xl"
            >
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-slate-400 uppercase tracking-wider">Data Quality</span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((bar) => (
                    <motion.div
                      key={bar}
                      className="w-1 bg-gradient-to-t from-blue-600 to-blue-400 rounded-full"
                      initial={{ height: 0 }}
                      animate={{ height: `${Math.random() * 12 + 4}px` }}
                      transition={{ delay: 0.8 + bar * 0.1, duration: 0.5 }}
                    />
                  ))}
                </div>
                <span className="text-xs text-blue-300 font-medium">Premium</span>
              </div>
            </motion.div>
          </div>

          <div className="relative">
            {isLoading ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="h-full bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-8 shadow-2xl relative overflow-hidden"
              >
                {/* Loading Background Pattern */}
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5" />
                <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
                
                <div className="h-full flex items-center justify-center relative z-10">
                  <div className="text-center">
                    {/* Enhanced Loading Spinner */}
                    <motion.div
                      className="relative w-20 h-20 mx-auto mb-8"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <div className="absolute inset-0 border-4 border-slate-700 rounded-full" />
                      <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 border-r-purple-500 rounded-full" />
                      <motion.div
                        className="absolute inset-2 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        <BarChart3 className="w-8 h-8 text-blue-400" />
                      </motion.div>
                    </motion.div>
                    
                    <motion.h3
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent mb-2"
                    >
                      Loading Market Data
                    </motion.h3>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="text-slate-400 mb-6"
                    >
                      Connecting to financial data streams...
                    </motion.p>
                    
                    {/* Loading Progress Indicators */}
                    <div className="space-y-3 max-w-xs mx-auto">
                      {['Historical Data', 'Real-time Feed', 'Market Analysis'].map((step, index) => (
                        <motion.div
                          key={step}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.7 + index * 0.2 }}
                          className="flex items-center gap-3"
                        >
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: index * 0.3 }}
                            className="w-4 h-4 border-2 border-slate-600 border-t-blue-400 rounded-full"
                          />
                          <span className="text-sm text-slate-300">{step}</span>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : displayData.length > 0 ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="h-full bg-gradient-to-br from-slate-900/30 to-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-3xl shadow-2xl overflow-hidden relative"
              >
                {/* Chart Header */}
                <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-slate-900/50 to-slate-800/50 backdrop-blur-sm border-b border-slate-700/30 p-4 z-20">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        <Layers className="w-5 h-5 text-blue-400" />
                        <span className="text-white font-semibold">USD/COP • 5min</span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <Globe className="w-4 h-4 text-slate-400" />
                        <span className="text-sm text-slate-300">{displayData.length.toLocaleString()} candles</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {isRealtime && (
                        <motion.div
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 2, repeat: Infinity }}
                          className="flex items-center gap-2 px-2 py-1 bg-emerald-500/20 rounded-lg border border-emerald-500/30"
                        >
                          <Signal className="w-3 h-3 text-emerald-400" />
                          <span className="text-xs text-emerald-300">LIVE</span>
                        </motion.div>
                      )}
                    </div>
                  </div>
                </div>
                
                {/* Chart Container with enhanced styling */}
                <div className="h-full pt-16 relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/2 to-purple-500/2" />
                  <InteractiveTradingChart 
                    data={displayData}
                    isRealtime={isRealtime}
                    onRangeChange={(start, end) => {
                      console.log(`[Chart] Range changed: ${start.toISOString()} to ${end.toISOString()}`);
                    }}
                  />
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="h-full bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-8 shadow-2xl relative overflow-hidden"
              >
                {/* Empty State Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-orange-500/5" />
                <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
                
                <div className="h-full flex items-center justify-center relative z-10">
                  <div className="text-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                      className="w-24 h-24 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-3xl flex items-center justify-center mx-auto mb-6 border border-amber-500/20"
                    >
                      <motion.div
                        animate={{ y: [-5, 5, -5] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        <AlertCircle className="w-12 h-12 text-amber-400" />
                      </motion.div>
                    </motion.div>
                    
                    <motion.h3
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent mb-2"
                    >
                      No Market Data Available
                    </motion.h3>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                      className="text-slate-400 mb-8 max-w-md mx-auto"
                    >
                      Unable to load trading data. Please check your connection and try again.
                    </motion.p>
                    
                    <motion.button
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 }}
                      onClick={loadInitialData}
                      whileHover={{ scale: 1.05, y: -2 }}
                      whileTap={{ scale: 0.95 }}
                      className="group relative px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl hover:from-blue-500 hover:to-purple-500 transition-all duration-300 shadow-lg shadow-blue-500/25 font-semibold"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-2xl opacity-0 group-hover:opacity-20 transition-opacity" />
                      <span className="relative flex items-center gap-2">
                        <Target className="w-5 h-5" />
                        Load Market Data
                      </span>
                    </motion.button>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
        <motion.footer initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.3, duration: 0.6 }} className="bg-slate-900/30 backdrop-blur-xl border-t border-slate-700/50 px-4 sm:px-6 py-3 sm:py-4 relative overflow-hidden">
          {/* Footer Glow Effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-emerald-500/5" />
          <div className="absolute bottom-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
          
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 lg:gap-0 relative z-10">
            {/* Left Side - Data Metrics */}
            <div className="flex flex-wrap items-center gap-3 sm:gap-6">
              <motion.div 
                className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-xl border border-slate-700/30"
                whileHover={{ scale: 1.02 }}
              >
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span className="text-sm text-slate-300 font-medium">
                  {displayData.length.toLocaleString()} Points
                </span>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-xl border border-slate-700/30"
                whileHover={{ scale: 1.02 }}
              >
                <div className={`w-2 h-2 rounded-full ${
                  dataSource === 'l1' ? 'bg-emerald-400' : 
                  dataSource === 'l0' ? 'bg-blue-400' : 'bg-amber-400'
                }`} />
                <span className="text-sm text-slate-300 font-medium">
                  {dataSource.toUpperCase()} Source
                </span>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-xl border border-slate-700/30"
                whileHover={{ scale: 1.02 }}
              >
                <Clock className="w-3 h-3 text-slate-400" />
                <span className="text-sm text-slate-300 font-medium">5min Interval</span>
              </motion.div>
            </div>
            
            {/* Right Side - Market Metrics */}
            <div className="flex flex-wrap items-center gap-2 sm:gap-4">
              <motion.div 
                className="flex items-center gap-2 px-3 py-2 bg-slate-800/30 rounded-lg border border-slate-700/30"
                whileHover={{ y: -1 }}
              >
                <BarChart3 className="w-4 h-4 text-blue-400" />
                <span className="text-xs text-slate-400">Vol</span>
                <span className="text-sm text-white font-mono">
                  ${(volume24h / 1000000).toFixed(2)}M
                </span>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-2 px-3 py-2 bg-slate-800/30 rounded-lg border border-slate-700/30"
                whileHover={{ y: -1 }}
              >
                <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-slate-400">H</span>
                <span className="text-sm text-emerald-300 font-mono">
                  ${high24h.toFixed(2)}
                </span>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-2 px-3 py-2 bg-slate-800/30 rounded-lg border border-slate-700/30"
                whileHover={{ y: -1 }}
              >
                <ArrowDownRight className="w-4 h-4 text-red-400" />
                <span className="text-xs text-slate-400">L</span>
                <span className="text-sm text-red-300 font-mono">
                  ${low24h.toFixed(2)}
                </span>
              </motion.div>
              
              {/* Connection Status */}
              <motion.div 
                className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
                  connectionStatus === 'connected' 
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                    : connectionStatus === 'connecting'
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                    : 'bg-red-500/10 border-red-500/30 text-red-300'
                }`}
                whileHover={{ scale: 1.05 }}
              >
                <motion.div
                  className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-emerald-400'
                    : connectionStatus === 'connecting' ? 'bg-amber-400'
                    : 'bg-red-400'
                  }`}
                  animate={{ 
                    scale: connectionStatus === 'connecting' ? [1, 1.3, 1] : 1,
                    opacity: connectionStatus === 'connecting' ? [1, 0.5, 1] : 1
                  }}
                  transition={{ 
                    duration: 1.5, 
                    repeat: connectionStatus === 'connecting' ? Infinity : 0 
                  }}
                />
                <span className="text-xs font-medium capitalize">
                  {connectionStatus}
                </span>
              </motion.div>
            </div>
          </div>
        </motion.footer>
      </div>
    </div>
  );
};

export default EnhancedTradingDashboard;