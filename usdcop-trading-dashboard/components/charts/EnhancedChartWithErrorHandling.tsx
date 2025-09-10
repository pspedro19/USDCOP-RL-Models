/**
 * Enhanced Chart Component with Comprehensive Error Handling
 * Demonstrates best practices for chart error handling and graceful degradation
 */

'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { ChartErrorBoundary } from '@/components/common/ErrorBoundary';
import { ChartFallback, LoadingFallback, NetworkFallback } from '@/components/common/GracefulDegradation';
import { fetchWithRetry, networkErrorHandler } from '@/lib/services/network-error-handler';
import { reportChartError, reportApiError } from '@/lib/services/error-monitoring';
import { useNotifications } from '@/components/common/ErrorNotifications';
import { TrendingUp, TrendingDown, AlertTriangle, Wifi, WifiOff } from 'lucide-react';

interface ChartData {
  timestamp: string;
  price: number;
  volume?: number;
}

interface EnhancedChartProps {
  symbol: string;
  interval: string;
  height?: number;
  showVolume?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  fallbackData?: ChartData[];
}

const EnhancedChart: React.FC<EnhancedChartProps> = ({
  symbol = 'USD/COP',
  interval = '5min',
  height = 400,
  showVolume = false,
  autoRefresh = true,
  refreshInterval = 60000, // 1 minute
  fallbackData = []
}) => {
  const [data, setData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [isOffline, setIsOffline] = useState(false);
  
  const chartRef = useRef<HTMLDivElement>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout>();
  const refreshIntervalRef = useRef<NodeJS.Timeout>();
  
  const { showNotification } = useNotifications();

  // Memoized chart configuration to prevent unnecessary re-renders
  const chartConfig = useMemo(() => ({
    symbol,
    interval,
    height,
    showVolume
  }), [symbol, interval, height, showVolume]);

  // Network status monitoring
  useEffect(() => {
    const unsubscribe = networkErrorHandler.onNetworkStatusChange((status) => {
      setIsOffline(!status.online);
      if (status.online && error) {
        // Network came back online, retry loading
        loadChartData(true);
      }
    });

    // Set initial offline status
    setIsOffline(!networkErrorHandler.getNetworkStatus().online);

    return unsubscribe;
  }, [error]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const setupAutoRefresh = () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }

      refreshIntervalRef.current = setInterval(() => {
        if (!isOffline && !loading) {
          loadChartData(false); // Silent refresh
        }
      }, refreshInterval);
    };

    if (data.length > 0) {
      setupAutoRefresh();
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, isOffline, loading, data.length]);

  // Main data loading function with comprehensive error handling
  const loadChartData = async (showLoadingState = true) => {
    try {
      if (showLoadingState) {
        setLoading(true);
        setError(null);
      }

      console.log(`[EnhancedChart] Loading ${symbol} data (${interval})`);

      const response = await fetchWithRetry(`/api/data/realtime?symbol=${encodeURIComponent(symbol)}&interval=${interval}`, 
        {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Cache-Control': showLoadingState ? 'no-cache' : 'max-age=30'
          }
        },
        {
          maxAttempts: 3,
          baseDelay: 1000,
          maxDelay: 5000,
          onRetry: (error, attempt) => {
            console.warn(`[EnhancedChart] Retry ${attempt} for ${symbol}:`, error.message);
            setRetryCount(attempt);
          },
          onFailure: (error, attempts) => {
            reportApiError(error, `/api/data/realtime`, {
              symbol,
              interval,
              attempts,
              component: 'EnhancedChart'
            });
          }
        }
      );

      const result = await response.json();

      if (!result.success || !result.data || result.data.length === 0) {
        throw new Error(result.error || 'No data available');
      }

      // Transform data to chart format
      const chartData: ChartData[] = result.data.map((point: any) => ({
        timestamp: point.datetime || point.timestamp,
        price: parseFloat(point.close || point.price),
        volume: point.volume ? parseFloat(point.volume) : undefined
      }));

      setData(chartData);
      setLastUpdate(new Date());
      setError(null);
      setRetryCount(0);
      
      if (showLoadingState) {
        showNotification({
          type: 'success',
          title: 'Chart Updated',
          message: `${symbol} data loaded successfully`,
          duration: 3000
        });
      }

      console.log(`[EnhancedChart] Loaded ${chartData.length} data points for ${symbol}`);

    } catch (err) {
      const error = err as Error;
      console.error(`[EnhancedChart] Failed to load ${symbol} data:`, error);
      
      setError(error);
      setRetryCount(0);
      
      // Report error to monitoring system
      const errorId = reportChartError(error, 'EnhancedChart', {
        symbol,
        interval,
        height,
        showVolume,
        hasData: data.length > 0,
        isOffline
      });

      // Use fallback data if available and no existing data
      if (data.length === 0 && fallbackData.length > 0) {
        console.log(`[EnhancedChart] Using fallback data for ${symbol}`);
        setData(fallbackData);
        
        showNotification({
          type: 'warning',
          title: 'Using Cached Data',
          message: `Unable to fetch latest ${symbol} data. Showing cached information.`,
          duration: 6000,
          actions: [{
            label: 'Retry',
            action: () => loadChartData(true),
            style: 'primary'
          }]
        });
      } else if (showLoadingState) {
        showNotification({
          type: 'error',
          title: 'Chart Loading Failed',
          message: `Unable to load ${symbol} chart. ${isOffline ? 'Check your connection.' : 'Server may be busy.'}`,
          persistent: true,
          actions: [{
            label: 'Retry',
            action: () => loadChartData(true),
            style: 'primary'
          }]
        });
      }
    } finally {
      if (showLoadingState) {
        setLoading(false);
      }
    }
  };

  // Initial data load
  useEffect(() => {
    loadChartData(true);
  }, [symbol, interval]);

  // Cleanup timeouts
  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []);

  // Render loading state
  if (loading && data.length === 0) {
    return (
      <LoadingFallback
        title={`Loading ${symbol} Chart`}
        timeout={30000}
        onTimeout={() => {
          setError(new Error('Loading timeout'));
          setLoading(false);
        }}
      />
    );
  }

  // Render network error state
  if (isOffline && data.length === 0) {
    return (
      <NetworkFallback
        onRetry={() => loadChartData(true)}
        showOfflineMode={true}
      />
    );
  }

  // Render error state with fallback
  if (error && data.length === 0) {
    const staticData = fallbackData.slice(-5).map((point, index) => ({
      label: new Date(point.timestamp).toLocaleTimeString(),
      value: point.price,
      change: index > 0 ? ((point.price - fallbackData[fallbackData.length - 6 + index]?.price) / fallbackData[fallbackData.length - 6 + index]?.price * 100) : 0
    }));

    return (
      <ChartFallback
        title={`${symbol} Chart`}
        error={error}
        onRetry={() => loadChartData(true)}
        showStaticData={staticData.length > 0}
        staticData={staticData}
        height={height}
      />
    );
  }

  // Calculate price change for display
  const currentPrice = data.length > 0 ? data[data.length - 1].price : 0;
  const previousPrice = data.length > 1 ? data[data.length - 2].price : currentPrice;
  const priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;
  const isPositive = priceChange >= 0;

  return (
    <div className="relative">
      {/* Chart Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-terminal-text">{symbol}</h3>
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-bold text-terminal-text">
              {currentPrice.toFixed(4)}
            </span>
            <div className={`flex items-center space-x-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              <span className="text-sm font-medium">
                {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 text-xs text-terminal-text-dim">
          {isOffline && <WifiOff className="w-4 h-4 text-red-400" />}
          {error && <AlertTriangle className="w-4 h-4 text-yellow-400" />}
          {lastUpdate && (
            <span>
              Updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Chart Container */}
      <div 
        ref={chartRef}
        className="bg-terminal-surface border border-terminal-border rounded-lg p-4"
        style={{ height: `${height}px` }}
      >
        {/* Simple chart visualization */}
        {data.length > 0 ? (
          <div className="h-full flex items-end justify-between space-x-1">
            {data.slice(-50).map((point, index) => {
              const maxPrice = Math.max(...data.slice(-50).map(p => p.price));
              const minPrice = Math.min(...data.slice(-50).map(p => p.price));
              const heightPercent = ((point.price - minPrice) / (maxPrice - minPrice)) * 80 + 10;
              
              return (
                <div
                  key={index}
                  className={`flex-1 ${isPositive ? 'bg-green-400' : 'bg-red-400'} opacity-60 hover:opacity-100 transition-opacity rounded-sm`}
                  style={{ height: `${heightPercent}%` }}
                  title={`${point.timestamp}: ${point.price.toFixed(4)}`}
                />
              );
            })}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="text-terminal-text-dim mb-2">No data available</div>
              <button
                onClick={() => loadChartData(true)}
                className="terminal-button px-4 py-2 rounded"
              >
                Reload Chart
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Chart Footer */}
      {data.length > 0 && (
        <div className="flex justify-between items-center mt-2 text-xs text-terminal-text-dim">
          <span>{data.length} data points</span>
          <span>Interval: {interval}</span>
          {autoRefresh && <span>Auto-refresh: {refreshInterval / 1000}s</span>}
        </div>
      )}
    </div>
  );
};

// Wrapped component with error boundary
const EnhancedChartWithErrorHandling: React.FC<EnhancedChartProps> = (props) => {
  return (
    <ChartErrorBoundary>
      <EnhancedChart {...props} />
    </ChartErrorBoundary>
  );
};

export default EnhancedChartWithErrorHandling;