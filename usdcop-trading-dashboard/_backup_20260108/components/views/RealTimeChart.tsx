'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { fetchTimeSeries, wsClient, PriceData } from '@/lib/services/twelvedata';
import { pipelineDataService } from '@/lib/services/pipeline-data-client';
import { getTradingSessionInfo, formatTimeToOpen } from '@/lib/utils/trading-hours';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Clock,
  Wifi,
  WifiOff,
  Database,
  AlertTriangle,
  Zap,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { motion, AnimatePresence } from 'framer-motion';

// Import the working InteractiveTradingChart component
const InteractiveTradingChart = dynamic(
  () => import('@/components/charts/InteractiveTradingChart').then(mod => ({ default: mod.InteractiveTradingChart })),
  { ssr: false }
);

// Professional chart interface for OHLC data
interface CandleData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

export default function RealTimeChart() {
  // Core state
  const [data, setData] = useState<CandleData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Real-time update states
  const [secondsUntilNextUpdate, setSecondsUntilNextUpdate] = useState<number>(300);
  const [isAutoUpdating, setIsAutoUpdating] = useState<boolean>(false);
  const [last20Values, setLast20Values] = useState<CandleData[]>([]);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  
  // Market state
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market'>('closed');
  
  // Refs
  const wsUnsubscribe = useRef<(() => void) | null>(null);
  const autoUpdateInterval = useRef<NodeJS.Timeout | null>(null);
  const countdownInterval = useRef<NodeJS.Timeout | null>(null);

  // Professional market status checking
  const checkTradingHours = useCallback(() => {
    const now = new Date();
    const colombiaTime = new Date(now.toLocaleString("en-US", {timeZone: "America/Bogota"}));
    const day = colombiaTime.getDay();
    const hours = colombiaTime.getHours();
    const minutes = colombiaTime.getMinutes();
    const currentTime = hours * 60 + minutes;
    
    // Monday to Friday, 8:00 AM to 12:55 PM Colombian time
    if (day >= 1 && day <= 5 && currentTime >= 480 && currentTime <= 775) {
      setMarketStatus('open');
      return true;
    } else if (day >= 1 && day <= 5 && currentTime >= 420 && currentTime < 480) {
      setMarketStatus('pre-market');
      return false;
    } else {
      setMarketStatus('closed');
      return false;
    }
  }, []);

  // Calculate seconds until next aligned 5-minute mark
  const calculateSecondsToNextUpdate = useCallback(() => {
    const now = new Date();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();
    const nextUpdateMinute = Math.ceil(minutes / 5) * 5;
    const minutesUntilUpdate = nextUpdateMinute - minutes;
    const secondsUntilUpdate = (minutesUntilUpdate * 60) - seconds;
    return secondsUntilUpdate > 0 ? secondsUntilUpdate : 300;
  }, []);

  // Professional real-time data fetching with smooth updates
  const fetchRealtimeData = useCallback(async () => {
    if (!checkTradingHours()) return;
    
    try {
      console.log('[RealTimeChart] Fetching clean 5-minute update...');
      const latestData = await pipelineDataService.getRealtimeUpdate();
      
      if (latestData) {
        const candleData: CandleData = {
          ...latestData,
          timestamp: new Date(latestData.datetime).getTime()
        };
        
        setData(prev => {
          // Prevent duplicates and ensure clean 5-minute intervals
          const newData = [...prev];
          const lastCandle = newData[newData.length - 1];
          
          // Only add if it's a new 5-minute interval
          if (!lastCandle || new Date(candleData.datetime).getMinutes() !== new Date(lastCandle.datetime).getMinutes()) {
            newData.push(candleData);
            
            // Update last 20 values for table
            const last20 = newData.slice(-20);
            setLast20Values(last20);
            
            return newData.sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
          }
          
          return prev;
        });
        
        // Smooth price animation
        setCurrentPrice(prev => {
          const newPrice = Number(latestData.close);
          return newPrice;
        });
        
        setLastUpdateTime(new Date());
        
        // Calculate price change from previous close
        if (data.length > 0) {
          const previousClose = Number(data[data.length - 1].close);
          setPriceChange(((Number(latestData.close) - previousClose) / previousClose) * 100);
        }
      }
    } catch (error) {
      console.error('[RealTimeChart] Error fetching real-time data:', error);
    }
  }, [checkTradingHours, data]);

  // Professional data loading with Bloomberg-style initialization
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log('[RealTimeChart] Loading recent historical data (last 7 days)...');
        const endDate = new Date(); // Today
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - 7); // Last 7 days instead of from 2020

        const alignedData = await pipelineDataService.loadL0Data(startDate, endDate, 'align');

        if (alignedData.length > 0) {
          console.log(`[RealTimeChart] Loaded ${alignedData.length} professional data points`);

          const processedData: CandleData[] = alignedData
            .sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime())
            .map(item => ({
              ...item,
              timestamp: new Date(item.datetime).getTime()
            }));

          setData(processedData);
          setCurrentPrice(Number(processedData[processedData.length - 1].close));

          // Update last 20 for table display
          setLast20Values(processedData.slice(-20));

          // Calculate professional price change
          if (processedData.length > 1) {
            const latest = processedData[processedData.length - 1];
            const previous = processedData[processedData.length - 2];
            setPriceChange(((Number(latest.close) - Number(previous.close)) / Number(previous.close)) * 100);
          }

          // Set aligned update time
          const alignedTime = new Date();
          alignedTime.setSeconds(0, 0);
          alignedTime.setMinutes(Math.floor(alignedTime.getMinutes() / 5) * 5);
          setLastUpdateTime(alignedTime);
        }

        setLoading(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error loading chart data';
        console.error('[RealTimeChart] Error loading professional data:', err);
        setError(errorMessage);
        setLoading(false);
      }
    };

    loadInitialData();
    
    // Check market status initially and every minute
    checkTradingHours();
    const statusInterval = setInterval(checkTradingHours, 60000);
    
    return () => {
      clearInterval(statusInterval);
      if (wsUnsubscribe.current) {
        wsUnsubscribe.current();
      }
    };
  }, [checkTradingHours]);

  // SMART real-time updates - ALIGNED to 5-minute intervals (:00, :05, :10, :15, etc.)
  useEffect(() => {
    const isInTradingHours = checkTradingHours();
    setIsAutoUpdating(isInTradingHours);

    if (isInTradingHours) {
      // Calculate next 5-minute interval in COT timezone
      const calculateNextInterval = () => {
        const now = new Date();
        // Convert to COT to get correct minute alignment
        const cotFormatter = new Intl.DateTimeFormat('en-US', {
          timeZone: 'America/Bogota',
          hour: 'numeric',
          minute: 'numeric',
          second: 'numeric',
          hour12: false
        });

        const parts = cotFormatter.formatToParts(now);
        const currentMinute = parseInt(parts.find(p => p.type === 'minute')?.value || '0');
        const currentSecond = parseInt(parts.find(p => p.type === 'second')?.value || '0');

        // Calculate minutes to next 5-minute mark
        const minutesToNext = 5 - (currentMinute % 5);
        const minutesOffset = minutesToNext === 5 ? 0 : minutesToNext;

        // Seconds until next 5-minute mark + 10 second buffer for data availability
        const secondsToNext = (minutesOffset * 60) - currentSecond + 10;

        return secondsToNext > 0 ? secondsToNext : (300 + 10); // Next interval or 5min + buffer
      };

      // Initial fetch
      fetchRealtimeData();

      // Smart scheduling: Align to next :00, :05, :10, :15, etc. + 10s buffer
      const secondsToNextInterval = calculateNextInterval();

      console.log(`[RealTimeChart] Smart refresh: Next update in ${secondsToNextInterval}s (aligned to 5-min interval + 10s buffer)`);

      // First aligned update
      const alignedTimeout = setTimeout(() => {
        fetchRealtimeData();

        // After first aligned fetch, continue every 5 minutes exactly
        autoUpdateInterval.current = setInterval(() => {
          if (checkTradingHours()) {
            fetchRealtimeData();
          } else {
            setIsAutoUpdating(false);
          }
        }, 5 * 60 * 1000); // Exactly 5 minutes
      }, secondsToNextInterval * 1000);

      // Professional countdown timer (updates every second)
      countdownInterval.current = setInterval(() => {
        setSecondsUntilNextUpdate(calculateSecondsToNextUpdate());
      }, 1000);

      return () => {
        clearTimeout(alignedTimeout);
        if (autoUpdateInterval.current) {
          clearInterval(autoUpdateInterval.current);
        }
        if (countdownInterval.current) {
          clearInterval(countdownInterval.current);
        }
      };
    }

    return () => {
      if (autoUpdateInterval.current) {
        clearInterval(autoUpdateInterval.current);
      }
      if (countdownInterval.current) {
        clearInterval(countdownInterval.current);
      }
    };
  }, [checkTradingHours, fetchRealtimeData, calculateSecondsToNextUpdate]);


  // Professional data alignment handler
  const handleAlignDataset = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('[RealTimeChart] Aligning professional dataset...');
      const now = new Date();
      const startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

      const alignedData = await pipelineDataService.loadL0Data(startDate, now, 'align');

      if (alignedData.length > 0) {
        console.log(`[RealTimeChart] Professional dataset aligned: ${alignedData.length} points`);

        const processedData: CandleData[] = alignedData.map(item => ({
          ...item,
          timestamp: new Date(item.datetime).getTime()
        }));

        setData(processedData);
        setCurrentPrice(Number(processedData[processedData.length - 1].close));
        setLast20Values(processedData.slice(-20));

        const alignedTime = new Date();
        alignedTime.setSeconds(0, 0);
        alignedTime.setMinutes(Math.floor(alignedTime.getMinutes() / 5) * 5);
        setLastUpdateTime(alignedTime);

        if (processedData.length > 1) {
          const prevPrice = Number(processedData[processedData.length - 2].close);
          const change = ((currentPrice - prevPrice) / prevPrice) * 100;
          setPriceChange(change);
        }
      }

      setLoading(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error aligning dataset';
      console.error('[RealTimeChart] Error aligning professional dataset:', err);
      setError(errorMessage);
      setLoading(false);
    }
  }, [currentPrice]);

  // Professional calculations
  const high24h = data.length > 0 ? Math.max(...data.map(d => Number(d.high))) : 0;
  const low24h = data.length > 0 ? Math.min(...data.map(d => Number(d.low))) : 0;
  const sessionInfo = getTradingSessionInfo();
  
  // Professional price formatting
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price).replace('COP', '').trim() + ' COP';
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <Card className="bg-[var(--bg-elevated)] border-[var(--chart-grid)]">
          <CardContent className="flex items-center justify-center py-24">
            <div className="text-center">
              <RefreshCw className="h-12 w-12 animate-spin text-[var(--market-up)] mx-auto mb-4" />
              <p className="text-[var(--text-primary)] text-xl font-semibold mb-2">Loading Professional Chart</p>
              <p className="text-[var(--text-secondary)] text-sm">Initializing Bloomberg Terminal style</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Card className="bg-[var(--bg-elevated)] border-[var(--market-down)] border-2">
          <CardContent className="py-12 text-center">
            <AlertCircle className="h-12 w-12 text-[var(--market-down)] mx-auto mb-4" />
            <p className="text-[var(--market-down)] text-xl font-semibold mb-2">Error Loading Chart Data</p>
            <p className="text-[var(--text-secondary)] text-sm mb-6">{error}</p>
            <Button
              onClick={handleAlignDataset}
              className="bg-[var(--market-up)] hover:bg-[var(--market-up)]/80 text-white"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 bg-[var(--bg-primary)] min-h-screen p-4">
      {/* Professional Market Status Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl p-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            {/* Market Status */}
            <motion.div 
              className={`flex items-center space-x-3 px-4 py-2 rounded-xl backdrop-blur-sm ${
                marketStatus === 'open' 
                  ? 'bg-[var(--up-10)] border border-[var(--market-up)]' 
                  : marketStatus === 'pre-market'
                  ? 'bg-[var(--down-10)] border border-[var(--status-delayed)]'
                  : 'bg-[var(--down-10)] border border-[var(--market-down)]'
              }`}
              animate={marketStatus === 'open' ? { scale: [1, 1.02, 1] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            >
              {marketStatus === 'open' ? (
                <Wifi className="h-4 w-4 text-[var(--market-up)]" />
              ) : (
                <WifiOff className="h-4 w-4 text-[var(--market-down)]" />
              )}
              <span className="font-mono text-sm font-semibold text-[var(--text-primary)]">
                {marketStatus === 'open' ? 'MARKET OPEN' : 
                 marketStatus === 'pre-market' ? 'PRE-MARKET' : 'MARKET CLOSED'}
              </span>
            </motion.div>
            
            {/* Trading Hours */}
            <div className="text-xs text-[var(--text-secondary)]">
              <div className="font-mono">COT 08:00 - 12:55</div>
              <div className="text-[var(--text-tertiary)]">Monday - Friday</div>
            </div>
            
            {/* Live Indicator */}
            {lastUpdateTime && marketStatus === 'open' && (
              <motion.div 
                className="flex items-center space-x-2 px-3 py-1 bg-[var(--up-10)] rounded-lg border border-[var(--market-up)]" 
                animate={{ opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <div className="w-2 h-2 rounded-full bg-[var(--market-up)] animate-pulse" />
                <span className="text-xs font-mono text-[var(--market-up)] font-medium">
                  LIVE • {lastUpdateTime.toLocaleTimeString('en-US', { 
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                  })}
                </span>
              </motion.div>
            )}
            
            {/* Countdown Timer */}
            {isAutoUpdating && (
              <motion.div 
                className="flex items-center space-x-2 px-3 py-1 bg-[var(--bg-overlay)] rounded-lg"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
              >
                <Clock className="h-3 w-3 text-[var(--text-secondary)]" />
                <span className="text-xs font-mono text-[var(--text-secondary)]">
                  Next: {Math.floor(secondsUntilNextUpdate / 60)}:{(secondsUntilNextUpdate % 60).toString().padStart(2, '0')}
                </span>
              </motion.div>
            )}
          </div>
          
          {/* Actions */}
          <Button
            onClick={handleAlignDataset}
            className="bg-[var(--bg-interactive)] hover:bg-[var(--bg-overlay)] border border-[var(--chart-grid)] text-[var(--text-primary)] transition-all duration-200"
            size="sm"
          >
            <Database className="h-4 w-4 mr-2" />
            <span className="font-mono text-xs">ALIGN DATA</span>
          </Button>
        </div>
      </motion.div>
      

      {/* Professional Price Cards with Glass-morphism */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-4 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* Current Price */}
        <motion.div 
          className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl p-4 relative overflow-hidden"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[var(--text-secondary)] uppercase tracking-wide">Current Price</span>
            <div className="flex space-x-1">
              <div className={`w-2 h-2 rounded-full animate-pulse ${
                marketStatus === 'open' ? 'bg-[var(--market-up)]' : 'bg-[var(--market-down)]'
              }`} />
              <Activity className="h-3 w-3 text-[var(--text-secondary)]" />
            </div>
          </div>
          <motion.div 
            className="text-2xl font-mono font-bold text-[var(--text-primary)] mb-1"
            animate={{ scale: currentPrice !== 0 ? [1, 1.05, 1] : 1 }}
            transition={{ duration: 0.3 }}
          >
            {formatPrice(currentPrice)}
          </motion.div>
          <div className={`flex items-center text-sm font-mono ${
            priceChange >= 0 ? 'text-[var(--market-up)]' : 'text-[var(--market-down)]'
          }`}>
            {priceChange >= 0 ? 
              <TrendingUp className="h-3 w-3 mr-1" /> : 
              <TrendingDown className="h-3 w-3 mr-1" />
            }
            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
          </div>
          {lastUpdateTime && (
            <div className="text-xs text-[var(--text-tertiary)] mt-2 font-mono">
              {lastUpdateTime.toLocaleString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
              })}
            </div>
          )}
        </motion.div>

        {/* Session High */}
        <motion.div 
          className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl p-4 relative overflow-hidden"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[var(--text-secondary)] uppercase tracking-wide">Session High</span>
            <TrendingUp className="h-3 w-3 text-[var(--market-up)]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[var(--market-up)] mb-1">
            {formatPrice(high24h)}
          </div>
        </motion.div>

        {/* Session Low */}
        <motion.div 
          className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl p-4 relative overflow-hidden"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[var(--text-secondary)] uppercase tracking-wide">Session Low</span>
            <TrendingDown className="h-3 w-3 text-[var(--market-down)]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[var(--market-down)] mb-1">
            {formatPrice(low24h)}
          </div>
        </motion.div>

        {/* Data Points */}
        <motion.div 
          className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl p-4 relative overflow-hidden"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[var(--text-secondary)] uppercase tracking-wide">Data Points</span>
            <Database className="h-3 w-3 text-[var(--text-secondary)]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[var(--text-primary)] mb-1">
            {data.length.toLocaleString()}
          </div>
          <div className="text-xs text-[var(--text-tertiary)] font-mono">
            5-minute intervals
          </div>
        </motion.div>
      </motion.div>

      {/* Professional Chart with Full Historical Data */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4 }}
        className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl overflow-hidden"
      >
        <div className="p-4 border-b border-[var(--chart-grid)]">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-mono font-semibold text-[var(--text-primary)]">USD/COP - Historical Chart</h3>
              <p className="text-xs text-[var(--text-secondary)] mt-1">
                Showing {data.length.toLocaleString()} data points from 2020 to present • Use mouse wheel to zoom, drag to pan
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Badge className="bg-[var(--up-10)] text-[var(--market-up)] border-[var(--market-up)]">
                5MIN
              </Badge>
              <Badge className="bg-[var(--bg-interactive)] text-[var(--text-secondary)]">
                ALL HISTORY
              </Badge>
              {isAutoUpdating && (
                <Badge className="bg-[var(--market-up)] text-white animate-pulse">
                  LIVE
                </Badge>
              )}
            </div>
          </div>
        </div>
        <div className="h-[600px]">
          <InteractiveTradingChart 
            data={data} 
            isRealtime={isAutoUpdating}
            onRangeChange={(start, end) => {
              console.log(`[RealTimeChart] Date range changed: ${start.toISOString()} to ${end.toISOString()}`);
            }}
          />
        </div>
      </motion.div>

      {/* Professional Data Table - Last 20 Values */}
      {last20Values.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-[var(--bg-elevated)] backdrop-blur-md border border-[var(--chart-grid)] rounded-xl overflow-hidden"
        >
          <div className="p-4 border-b border-[var(--chart-grid)]">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-mono font-semibold text-[var(--text-primary)]">Recent Activity (Last 20)</h3>
              <div className="flex items-center space-x-2">
                <Badge className="bg-[var(--bg-interactive)] text-[var(--text-primary)] border border-[var(--chart-grid)]">
                  {last20Values.length} INTERVALS
                </Badge>
                {isAutoUpdating && (
                  <motion.div
                    className="px-2 py-1 bg-[var(--up-10)] border border-[var(--market-up)] rounded text-xs font-mono text-[var(--market-up)]"
                    animate={{ opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    LIVE
                  </motion.div>
                )}
              </div>
            </div>
          </div>
          <div className="overflow-x-auto max-h-64">
            <table className="min-w-full divide-y divide-[var(--chart-grid)]">
              <thead className="bg-[var(--bg-primary)] sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Date</th>
                  <th className="px-3 py-2 text-left text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Time</th>
                  <th className="px-3 py-2 text-right text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Open</th>
                  <th className="px-3 py-2 text-right text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">High</th>
                  <th className="px-3 py-2 text-right text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Low</th>
                  <th className="px-3 py-2 text-right text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Close</th>
                  <th className="px-3 py-2 text-right text-xs font-mono font-medium text-[var(--text-secondary)] uppercase tracking-wider">Change</th>
                </tr>
              </thead>
              <tbody className="bg-[var(--bg-elevated)] divide-y divide-[var(--chart-grid)]">
                {last20Values.slice().reverse().map((value, index) => {
                  const date = new Date(value.datetime);
                  const prevValue = last20Values[last20Values.length - index - 2];
                  const change = prevValue ? ((Number(value.close) - Number(prevValue.close)) / Number(prevValue.close)) * 100 : 0;
                  const isLatest = index === 0;
                  
                  return (
                    <motion.tr 
                      key={`${value.datetime}-${index}`} 
                      className={`hover:bg-[var(--bg-interactive)] transition-colors ${
                        isLatest ? 'bg-[var(--up-10)] border-l-2 border-l-[var(--market-up)]' : ''
                      }`}
                      initial={isLatest ? { opacity: 0, x: -10 } : {}}
                      animate={isLatest ? { opacity: 1, x: 0 } : {}}
                      transition={{ duration: 0.3 }}
                    >
                      <td className="px-3 py-2 text-xs font-mono text-[var(--text-secondary)]">
                        {date.toLocaleDateString('en-US', { month: 'short', day: '2-digit' })}
                      </td>
                      <td className="px-3 py-2 text-xs font-mono text-[var(--text-primary)]">
                        {date.toLocaleTimeString('en-US', { 
                          hour: '2-digit',
                          minute: '2-digit',
                          hour12: false
                        })}
                      </td>
                      <td className="px-3 py-2 text-xs font-mono text-right text-[var(--text-primary)]">
                        {Number(value.open).toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-xs font-mono text-right text-[var(--market-up)]">
                        {Number(value.high).toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-xs font-mono text-right text-[var(--market-down)]">
                        {Number(value.low).toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-xs font-mono text-right font-semibold text-[var(--text-primary)]">
                        {Number(value.close).toFixed(2)}
                      </td>
                      <td className={`px-3 py-2 text-xs font-mono text-right font-semibold ${
                        change >= 0 ? 'text-[var(--market-up)]' : 'text-[var(--market-down)]'
                      }`}>
                        {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {lastUpdateTime && (
            <div className="p-3 bg-[var(--bg-primary)] border-t border-[var(--chart-grid)] flex items-center justify-between text-xs">
              <div className="flex items-center space-x-2 text-[var(--text-secondary)]">
                <Clock className="h-3 w-3" />
                <span className="font-mono">Updated: {lastUpdateTime.toLocaleTimeString('en-US', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                  hour12: false
                })}</span>
              </div>
              {isAutoUpdating && (
                <div className="flex items-center space-x-2 text-[var(--market-up)]">
                  <Activity className="h-3 w-3 animate-pulse" />
                  <span className="font-mono">
                    Next: {Math.floor(secondsUntilNextUpdate / 60)}:{(secondsUntilNextUpdate % 60).toString().padStart(2, '0')}
                  </span>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}

    </div>
  );
}