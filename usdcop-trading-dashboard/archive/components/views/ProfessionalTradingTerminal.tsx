/**
 * Professional Trading Terminal
 * Integrates the new Dynamic Navigation System with advanced charting
 * Bloomberg Terminal-class interface for USDCOP trading
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Wifi,
  WifiOff,
  Database,
  Settings,
  Maximize2,
  BarChart3,
  Zap
} from 'lucide-react';

import DynamicNavigationSystem from '../navigation/DynamicNavigationSystem';
import { TimeRange } from '../navigation/EnhancedTimeRangeSelector';
import { OHLCData, RealMarketMetricsCalculator } from '../../lib/services/real-market-metrics';
import { historicalDataManager } from '../../lib/services/historical-data-manager';
import { realTimeWebSocketManager } from '../../lib/services/realtime-websocket-manager';

interface MarketStats {
  price: number;
  change: number;
  changePercent: number;
  high24h: number;
  low24h: number;
  volume: number;
  spread: number;
  lastUpdate: Date;
}

interface ConnectionStatus {
  status: 'connected' | 'disconnected' | 'connecting';
  quality: 'excellent' | 'good' | 'poor' | 'disconnected';
  latency: number;
  source: string;
  lastUpdate: Date | null;
}

export default function ProfessionalTradingTerminal() {
  // State management
  const [currentRange, setCurrentRange] = useState<TimeRange>({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Last 30 days
    end: new Date(),
    timeframe: '1h',
    preset: '1m'
  });

  const [marketData, setMarketData] = useState<OHLCData[]>([]);
  const [currentTick, setCurrentTick] = useState<{
    price: number;
    bid: number;
    ask: number;
    timestamp: Date;
  } | null>(null);

  const [marketStats, setMarketStats] = useState<MarketStats>({
    price: 0,
    change: 0,
    changePercent: 0,
    high24h: 0,
    low24h: 0,
    volume: 0,
    spread: 0,
    lastUpdate: new Date()
  });

  const [connectionInfo, setConnectionInfo] = useState<ConnectionStatus>({
    status: 'disconnected',
    quality: 'disconnected',
    latency: 0,
    source: 'none',
    lastUpdate: null
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load historical data based on time range
  const loadDataForRange = useCallback(async (range: TimeRange): Promise<OHLCData[]> => {
    setIsLoading(true);
    setError(null);

    try {
      // Calculate optimal timeframe based on range duration
      const rangeDuration = range.end.getTime() - range.start.getTime();
      const days = rangeDuration / (24 * 60 * 60 * 1000);

      let optimalTimeframe = range.timeframe;
      if (days > 365) optimalTimeframe = '1d';
      else if (days > 90) optimalTimeframe = '4h';
      else if (days > 30) optimalTimeframe = '1h';
      else if (days > 7) optimalTimeframe = '15m';

      // Load data from our data manager
      const data = await historicalDataManager.getDataForRange(
        range.start,
        range.end,
        optimalTimeframe
      );

      // Transform to OHLCData format
      const ohlcData: OHLCData[] = data.map(point => ({
        timestamp: new Date(point.time),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        price: point.close,
        bid: point.close - 1, // Approximate bid
        ask: point.close + 1, // Approximate ask
        volume: point.volume
      }));

      setMarketData(ohlcData);
      return ohlcData;

    } catch (error) {
      console.error('Error loading data for range:', error);
      setError('Error cargando datos históricos');
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle time range changes
  const handleTimeRangeChange = useCallback(async (newRange: TimeRange) => {
    setCurrentRange(newRange);
    await loadDataForRange(newRange);
  }, [loadDataForRange]);

  // Initialize data and WebSocket connections
  useEffect(() => {
    let unsubscribeData: (() => void) | null = null;
    let unsubscribeStatus: (() => void) | null = null;

    const initializeSystem = async () => {
      try {
        // Load initial data
        await loadDataForRange(currentRange);

        // Set up real-time data subscriptions
        unsubscribeData = realTimeWebSocketManager.onData((data) => {
          const newTick = {
            price: data.price,
            bid: data.bid || data.price - 1,
            ask: data.ask || data.price + 1,
            timestamp: new Date(data.timestamp)
          };

          setCurrentTick(newTick);

          // Update market stats
          setMarketStats(prev => {
            const change = data.price - prev.price;
            const changePercent = prev.price > 0 ? (change / prev.price) * 100 : 0;

            return {
              price: data.price,
              change,
              changePercent,
              high24h: Math.max(prev.high24h, data.price),
              low24h: Math.min(prev.low24h || data.price, data.price),
              volume: data.volume || prev.volume,
              spread: (data.ask || data.price + 1) - (data.bid || data.price - 1),
              lastUpdate: new Date(data.timestamp)
            };
          });
        });

        // Set up connection status monitoring
        unsubscribeStatus = realTimeWebSocketManager.onStatusChange((status) => {
          setConnectionInfo({
            status: status.connected ? 'connected' : 'disconnected',
            quality: status.quality,
            latency: status.latency,
            source: status.connected ? 'websocket' : 'historical',
            lastUpdate: status.lastHeartbeat
          });
        });

        // Connect to WebSocket
        await realTimeWebSocketManager.connect();

        // Load latest market stats
        const latestData = await historicalDataManager.getLatestData('5m', 10);
        if (latestData.length > 0) {
          const latest = latestData[latestData.length - 1];
          const previous = latestData[latestData.length - 2];

          setMarketStats({
            price: latest.close,
            change: previous ? latest.close - previous.close : 0,
            changePercent: previous ? ((latest.close - previous.close) / previous.close) * 100 : 0,
            high24h: Math.max(...latestData.map(d => d.high)),
            low24h: Math.min(...latestData.map(d => d.low)),
            volume: latest.volume,
            spread: 2, // Default spread
            lastUpdate: new Date(latest.time)
          });

          setCurrentTick({
            price: latest.close,
            bid: latest.close - 1,
            ask: latest.close + 1,
            timestamp: new Date(latest.time)
          });
        }

      } catch (error) {
        console.error('Error initializing trading terminal:', error);
        setError('Error inicializando terminal');
      }
    };

    initializeSystem();

    // Cleanup
    return () => {
      unsubscribeData?.();
      unsubscribeStatus?.();
    };
  }, [currentRange, loadDataForRange]);

  // Calculate real-time metrics
  const realTimeMetrics = useMemo(() => {
    if (marketData.length === 0) return null;

    try {
      return RealMarketMetricsCalculator.calculateMetrics(marketData, currentTick || undefined);
    } catch (error) {
      console.error('Error calculating metrics:', error);
      return null;
    }
  }, [marketData, currentTick]);

  // Format price for display
  const formatPrice = useCallback((price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(price);
  }, []);

  // Format change for display
  const formatChange = useCallback((change: number, isPercent: boolean = false) => {
    const formatted = isPercent
      ? `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`
      : `${change >= 0 ? '+' : ''}${change.toFixed(4)}`;

    return { formatted, isPositive: change >= 0 };
  }, []);

  // Connection status component
  const ConnectionIndicator = useMemo(() => {
    const getStatusColor = () => {
      switch (connectionInfo.quality) {
        case 'excellent': return 'text-emerald-400';
        case 'good': return 'text-cyan-400';
        case 'poor': return 'text-yellow-400';
        default: return 'text-red-400';
      }
    };

    const getStatusIcon = () => {
      if (connectionInfo.status === 'connecting') {
        return <Activity className="h-4 w-4 animate-pulse" />;
      }
      return connectionInfo.status === 'connected' ?
        <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />;
    };

    return (
      <div className={`flex items-center space-x-2 ${getStatusColor()}`}>
        {getStatusIcon()}
        <span className="text-sm font-medium">
          {connectionInfo.status === 'connected' ? 'Conectado' :
           connectionInfo.status === 'connecting' ? 'Conectando...' : 'Desconectado'}
        </span>
        {connectionInfo.latency > 0 && (
          <span className="text-xs opacity-75">
            ({connectionInfo.latency}ms)
          </span>
        )}
      </div>
    );
  }, [connectionInfo]);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header Bar */}
      <div className="bg-slate-900/80 backdrop-blur border-b border-slate-600/30 p-4">
        <div className="flex items-center justify-between">
          {/* Title and Symbol */}
          <div className="flex items-center space-x-6">
            <div>
              <h1 className="text-2xl font-bold text-white">USD/COP</h1>
              <p className="text-sm text-slate-400">Professional Trading Terminal</p>
            </div>

            {/* Current Price Display */}
            <div className="flex items-center space-x-6">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-white">
                  {formatPrice(marketStats.price)}
                </div>
                <div className="text-xs text-slate-400">Precio Actual</div>
              </div>

              <div className="text-center">
                <div className={`text-lg font-mono font-bold flex items-center space-x-1 ${
                  marketStats.change >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {marketStats.change >= 0 ?
                    <TrendingUp className="h-4 w-4" /> :
                    <TrendingDown className="h-4 w-4" />
                  }
                  <span>{formatChange(marketStats.change).formatted}</span>
                  <span className="text-sm">
                    ({formatChange(marketStats.changePercent, true).formatted})
                  </span>
                </div>
                <div className="text-xs text-slate-400">Cambio 24h</div>
              </div>

              <div className="text-center">
                <div className="text-lg font-mono text-white">
                  {marketStats.spread.toFixed(2)} COP
                </div>
                <div className="text-xs text-slate-400">Spread</div>
              </div>
            </div>
          </div>

          {/* Connection and Controls */}
          <div className="flex items-center space-x-6">
            {ConnectionIndicator}

            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Database className="h-4 w-4" />
              <span>{marketData.length.toLocaleString()} registros</span>
            </div>

            <button className="p-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-colors">
              <Settings className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Market Status Bar */}
        <div className="flex items-center justify-between mt-4 text-sm text-slate-400">
          <div className="flex items-center space-x-6">
            <div>H: {formatPrice(marketStats.high24h)}</div>
            <div>L: {formatPrice(marketStats.low24h)}</div>
            <div>Vol: {marketStats.volume.toLocaleString()}</div>
          </div>

          <div className="flex items-center space-x-4">
            <div>
              {realTimeWebSocketManager.isMarketOpen() ? (
                <span className="text-emerald-400">● Mercado Abierto</span>
              ) : (
                <span className="text-yellow-400">● Mercado Cerrado</span>
              )}
            </div>
            <div>
              Última actualización: {marketStats.lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 m-4">
          <div className="flex items-center gap-2 text-red-400">
            <Zap className="h-4 w-4" />
            {error}
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex flex-1">
        {/* Left Panel - Navigation System */}
        <div className="w-96 bg-slate-900/30 border-r border-slate-600/30 p-6">
          <DynamicNavigationSystem
            data={marketData}
            currentTick={currentTick || undefined}
            onTimeRangeChange={handleTimeRangeChange}
            onDataRequest={loadDataForRange}
            layout="full"
            initialExpanded={true}
          />
        </div>

        {/* Center - Chart Area */}
        <div className="flex-1 p-6">
          <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl h-full">
            {/* Chart Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-600/30">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-cyan-400" />
                <span className="font-semibold">Gráfico Principal</span>
                <span className="text-sm text-slate-400">
                  {currentRange.timeframe} • {marketData.length.toLocaleString()} barras
                </span>
              </div>

              <div className="flex items-center gap-2">
                {isLoading && (
                  <div className="flex items-center gap-2 text-cyan-400">
                    <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                    <span className="text-sm">Cargando...</span>
                  </div>
                )}

                <button className="p-2 bg-slate-700/30 hover:bg-slate-600/50 rounded transition-colors">
                  <Maximize2 className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Chart Content */}
            <div className="p-4 h-full">
              <div className="w-full h-96 bg-slate-800/30 rounded-lg flex items-center justify-center">
                <div className="text-center text-slate-400">
                  <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>Gráfico TradingView se integrará aquí</p>
                  <p className="text-sm mt-1">
                    Rango: {currentRange.start.toLocaleDateString()} - {currentRange.end.toLocaleDateString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Additional Tools */}
        <div className="w-80 bg-slate-900/30 border-l border-slate-600/30 p-6">
          <div className="space-y-6">
            {/* Real-time Metrics Summary */}
            {realTimeMetrics && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Métricas en Tiempo Real
                </h3>

                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">ATR (14):</span>
                    <span className="font-mono">{realTimeMetrics.volatility.atr14.toFixed(2)}</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-slate-400">Spread (bps):</span>
                    <span className="font-mono">{realTimeMetrics.currentSpread.bps.toFixed(1)}</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-slate-400">Volatilidad:</span>
                    <span className="font-mono">{(realTimeMetrics.volatility.parkinson * 100).toFixed(1)}%</span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-slate-400">Drawdown:</span>
                    <span className="font-mono text-red-400">
                      {(realTimeMetrics.returns.drawdown * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Market Session Progress */}
            {realTimeMetrics && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Progreso de Sesión</h3>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Progreso:</span>
                    <span>{realTimeMetrics.session.progressPct.toFixed(1)}%</span>
                  </div>

                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div
                      className="bg-cyan-500 h-2 rounded-full transition-all duration-1000"
                      style={{ width: `${realTimeMetrics.session.progressPct}%` }}
                    />
                  </div>

                  <div className="text-xs text-slate-500 text-center">
                    {realTimeMetrics.session.remainingMinutes} minutos restantes
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}