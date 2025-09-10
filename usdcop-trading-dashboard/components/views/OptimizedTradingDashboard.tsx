/**
 * Optimized Trading Dashboard for Large Datasets
 * Handles 84K+ data points with virtualization, lazy loading, and performance optimizations
 */

import React, { useState, useEffect, useCallback, useRef, memo } from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  Activity, 
  BarChart3, 
  Clock, 
  AlertCircle,
  Zap,
  Signal,
  Globe,
  ArrowUpRight,
  ArrowDownRight,
  Settings
} from 'lucide-react';

// Hooks for performance optimizations
import { 
  useChartOptimization, 
  useDebouncedUpdate, 
  useOptimizedAnimation,
  usePerformanceMonitor,
  useBatchedUpdates 
} from '@/lib/hooks/useEfficiencientRendering';
import { lazyDataService } from '@/lib/services/lazy-data-service';
import { animationOptimizer } from '@/lib/utils/animation-optimizer';

// Dynamic imports with optimized loading
const VirtualizedChart = dynamic(() => 
  import('@/components/charts/VirtualizedChart').then(mod => ({ default: mod.VirtualizedChart })), 
  {
    ssr: false,
    loading: () => (
      <div className="h-[500px] bg-slate-900/50 animate-pulse rounded-xl flex items-center justify-center">
        <div className="text-slate-400">Loading optimized chart...</div>
      </div>
    )
  }
);

const AnimatedSidebar = dynamic(() => 
  import('@/components/ui/AnimatedSidebar').then(mod => ({ default: mod.AnimatedSidebar })), 
  {
    ssr: false,
    loading: () => <div className="w-16 bg-slate-900 animate-pulse" />
  }
);

interface OptimizedTradingDashboardProps {
  initialData?: any[];
  enableRealtime?: boolean;
}

// Memoized performance status component
const PerformanceStatus = memo(({ metrics }: { metrics: any }) => (
  <motion.div
    initial={{ opacity: 0, x: 50 }}
    animate={{ opacity: 1, x: 0 }}
    className="bg-slate-900/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-3 shadow-2xl"
  >
    <div className="flex items-center gap-2 mb-2">
      <Zap className="w-4 h-4 text-yellow-400" />
      <span className="text-xs text-slate-400 uppercase tracking-wider">Performance</span>
    </div>
    <div className="space-y-1 text-xs">
      <div className="flex justify-between">
        <span className="text-slate-500">Render:</span>
        <span className="text-yellow-300 font-mono">{metrics.renderTime.toFixed(1)}ms</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-500">Memory:</span>
        <span className="text-blue-300 font-mono">{metrics.memoryUsage.toFixed(1)}MB</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-500">Points:</span>
        <span className="text-green-300 font-mono">{metrics.dataPoints?.toLocaleString() || '0'}</span>
      </div>
    </div>
  </motion.div>
));

const OptimizedTradingDashboard: React.FC<OptimizedTradingDashboardProps> = ({
  initialData = [],
  enableRealtime = false
}) => {
  // Performance monitoring
  const { metrics, startMeasure, endMeasure } = usePerformanceMonitor();
  
  // State management with performance optimizations
  const [rawData, setRawData] = useState<any[]>(initialData);
  const [isLoading, setIsLoading] = useState(false);
  const [isRealtime, setIsRealtime] = useState(enableRealtime);
  const [viewRange, setViewRange] = useState<[number, number]>([0, 2000]); // Show last 2k points initially
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market' | 'after-hours'>('closed');
  
  // Refs for cleanup
  const realtimeIntervalRef = useRef<NodeJS.Timeout>();
  const abortControllerRef = useRef<AbortController>();

  // Optimized data processing with chunking and decimation
  const { data: optimizedData, metrics: dataMetrics } = useChartOptimization(rawData, {
    maxPoints: 20000, // Limit to 20k points for smooth rendering
    aggregation: 'ohlc',
    enableDecimation: rawData.length > 20000
  });

  // Debounced view range to prevent excessive updates
  const debouncedViewRange = useDebouncedUpdate(viewRange, 100);

  // Get visible data slice based on view range
  const visibleData = React.useMemo(() => {
    const [start, end] = debouncedViewRange;
    return optimizedData.slice(Math.max(0, start), Math.min(optimizedData.length, end));
  }, [optimizedData, debouncedViewRange]);

  // Animation configuration based on data size
  const { variants, config: animationConfig } = useOptimizedAnimation(visibleData.length);

  // Batched updates for performance
  const batchUpdate = useBatchedUpdates();

  // Calculate performance metrics
  const currentPrice = optimizedData.length > 0 ? optimizedData[optimizedData.length - 1]?.close || 0 : 0;
  const priceChange = optimizedData.length > 1 
    ? currentPrice - optimizedData[0]?.close 
    : 0;
  const priceChangePercent = optimizedData.length > 1 && optimizedData[0]?.close 
    ? (priceChange / optimizedData[0].close) * 100 
    : 0;

  // Initialize lazy data service
  useEffect(() => {
    if (rawData.length > 0) {
      lazyDataService.initialize(rawData.length);
    }
  }, [rawData.length]);

  // Optimized data loading with chunking
  const loadData = useCallback(async (startIndex = 0, chunkSize = 5000) => {
    startMeasure();
    setIsLoading(true);

    try {
      // Cancel any pending requests
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      abortControllerRef.current = new AbortController();

      // Load data in chunks for better performance
      const response = await fetch(`/api/data/historical?start=${startIndex}&limit=${chunkSize}`, {
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error('Failed to load data');

      const result = await response.json();
      const newData = result.data || [];

      // Batch the state update
      batchUpdate(() => {
        if (startIndex === 0) {
          setRawData(newData);
        } else {
          setRawData(prev => [...prev, ...newData]);
        }
      });

      console.log(`[OptimizedDashboard] Loaded ${newData.length} points (chunk ${Math.floor(startIndex / chunkSize)})`);

    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('[OptimizedDashboard] Data loading error:', error);
      }
    } finally {
      setIsLoading(false);
      endMeasure('data-load');
    }
  }, [startMeasure, endMeasure, batchUpdate]);

  // Initial data load
  useEffect(() => {
    if (rawData.length === 0) {
      loadData();
    }
  }, [loadData, rawData.length]);

  // Optimized real-time updates
  const startRealtime = useCallback(() => {
    if (marketStatus === 'closed') return;

    setIsRealtime(true);
    
    realtimeIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch('/api/market/realtime?action=fetch');
        if (response.ok) {
          const result = await response.json();
          const latestData = result.data || [];

          if (latestData.length > 0) {
            const latestPoint = latestData[latestData.length - 1];
            
            batchUpdate(() => {
              setRawData(prev => {
                // Check for duplicates
                const exists = prev.some(d => d.datetime === latestPoint.datetime);
                if (exists) return prev;
                
                // Add new point and maintain reasonable size
                const updated = [...prev, latestPoint];
                return updated.slice(-50000); // Keep last 50k points max
              });
            });
          }
        }
      } catch (error) {
        console.error('[OptimizedDashboard] Real-time update error:', error);
      }
    }, 30000); // Update every 30 seconds
  }, [marketStatus, batchUpdate]);

  const stopRealtime = useCallback(() => {
    if (realtimeIntervalRef.current) {
      clearInterval(realtimeIntervalRef.current);
      realtimeIntervalRef.current = undefined;
    }
    setIsRealtime(false);
  }, []);

  // Handle view range changes for virtual scrolling
  const handleRangeChange = useCallback((start: Date, end: Date) => {
    const startIndex = optimizedData.findIndex(d => new Date(d.datetime) >= start);
    const endIndex = optimizedData.findIndex(d => new Date(d.datetime) > end);
    
    if (startIndex >= 0 && endIndex >= 0) {
      setViewRange([startIndex, endIndex]);
    }
  }, [optimizedData]);

  // Check market status
  useEffect(() => {
    const checkStatus = () => {
      const now = new Date();
      const hour = now.getHours();
      const day = now.getDay();
      
      if (day === 0 || day === 6) {
        setMarketStatus('closed');
      } else if (hour >= 8 && hour < 13) {
        setMarketStatus('open');
      } else {
        setMarketStatus('closed');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRealtime();
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      animationOptimizer.dispose();
    };
  }, [stopRealtime]);

  return (
    <div className="flex flex-col lg:flex-row h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-hidden relative">
      {/* Background patterns */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-900/20 via-transparent to-transparent" />

      {/* Sidebar */}
      <AnimatedSidebar
        onPlayPause={() => {}} // Implement if needed
        onReset={() => setViewRange([Math.max(0, optimizedData.length - 2000), optimizedData.length])}
        onAlignDataset={() => loadData(0, 10000)}
        isPlaying={false}
        isRealtime={isRealtime}
        dataSource="l0"
        onDataSourceChange={() => {}}
        marketStatus={marketStatus}
        currentPrice={currentPrice}
        priceChange={priceChange}
        priceChangePercent={priceChangePercent}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <motion.header
          {...variants.container}
          className="bg-slate-900/30 backdrop-blur-xl border-b border-slate-700/50 px-6 py-4 relative overflow-hidden"
        >
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center gap-6">
              <motion.div className="flex items-center gap-4" {...variants.item}>
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <TrendingUp className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                    USD/COP Optimized
                  </h1>
                  <p className="text-slate-400 text-sm">High-Performance Trading Platform</p>
                </div>
              </motion.div>

              {/* Status indicators */}
              <div className="flex items-center gap-3">
                {isRealtime && (
                  <motion.div
                    {...variants.item}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 rounded-xl backdrop-blur-sm"
                  >
                    <Signal className="w-4 h-4" />
                    <span className="text-sm font-medium">Live Data</span>
                  </motion.div>
                )}

                <motion.div
                  {...variants.item}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl"
                >
                  <Clock className="w-4 h-4 text-slate-400" />
                  <span className="text-sm text-slate-300 font-mono">
                    {new Date().toLocaleTimeString()}
                  </span>
                </motion.div>

                <motion.div
                  {...variants.item}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl border ${
                    marketStatus === 'open' 
                      ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                      : 'bg-red-500/20 border-red-500/30 text-red-300'
                  }`}
                >
                  <div className={`w-2 h-2 rounded-full ${
                    marketStatus === 'open' ? 'bg-emerald-400' : 'bg-red-400'
                  }`} />
                  <span className="text-xs font-medium uppercase">
                    {marketStatus.replace('-', ' ')}
                  </span>
                </motion.div>
              </div>
            </div>

            {/* Data metrics */}
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4" />
                <span>{rawData.length.toLocaleString()} total</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                <span>{visibleData.length.toLocaleString()} visible</span>
              </div>
              {dataMetrics.compressionRatio < 1 && (
                <div className="flex items-center gap-2 px-2 py-1 bg-blue-500/20 rounded-lg">
                  <Zap className="w-3 h-3" />
                  <span className="text-blue-300">
                    {(dataMetrics.compressionRatio * 100).toFixed(0)}% optimized
                  </span>
                </div>
              )}
            </div>
          </div>
        </motion.header>

        {/* Chart area */}
        <motion.div
          className="flex-1 p-6 overflow-hidden relative"
          {...variants.container}
        >
          {/* Performance metrics overlay */}
          <div className="absolute top-8 right-8 z-10 flex flex-col gap-3 hidden lg:flex">
            <PerformanceStatus metrics={{
              ...metrics,
              dataPoints: visibleData.length
            }} />
          </div>

          {isLoading ? (
            <motion.div
              {...variants.container}
              className="h-full bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-8 shadow-2xl"
            >
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="w-16 h-16 border-4 border-slate-700 border-t-blue-500 rounded-full mx-auto mb-4"
                  />
                  <h3 className="text-xl font-bold text-white mb-2">Loading Optimized Data</h3>
                  <p className="text-slate-400">Processing {rawData.length.toLocaleString()} data points...</p>
                </div>
              </div>
            </motion.div>
          ) : visibleData.length > 0 ? (
            <motion.div
              {...variants.container}
              className="h-full bg-gradient-to-br from-slate-900/30 to-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-3xl shadow-2xl overflow-hidden"
            >
              <VirtualizedChart
                data={visibleData}
                isRealtime={isRealtime}
                onRangeChange={handleRangeChange}
                chunkSize={2000}
                initialViewSize={1000}
              />
            </motion.div>
          ) : (
            <motion.div
              {...variants.container}
              className="h-full bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-8 shadow-2xl"
            >
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <AlertCircle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-white mb-2">No Data Available</h3>
                  <p className="text-slate-400 mb-6">Unable to load market data</p>
                  <Button
                    onClick={() => loadData()}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500"
                  >
                    Reload Data
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Footer with metrics */}
        <motion.footer
          {...variants.container}
          className="bg-slate-900/30 backdrop-blur-xl border-t border-slate-700/50 px-6 py-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-xl">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span className="text-sm text-slate-300">
                  {visibleData.length.toLocaleString()} Points
                </span>
              </div>
              <div className="flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-xl">
                <Clock className="w-3 h-3 text-slate-400" />
                <span className="text-sm text-slate-300">5min Interval</span>
              </div>
            </div>

            {/* Price metrics */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-800/30 rounded-lg">
                <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                <span className="text-sm text-emerald-300 font-mono">
                  ${currentPrice.toFixed(2)}
                </span>
              </div>
              <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
                priceChange >= 0 
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-red-500/20 text-red-300'
              }`}>
                <span className="text-sm font-mono">
                  {priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </motion.footer>
      </div>
    </div>
  );
};

export default memo(OptimizedTradingDashboard);