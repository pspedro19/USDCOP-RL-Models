/**
 * Enhanced Trading Dashboard with Complete Historical Navigation
 * Integrates advanced chart with real-time WebSocket updates and full 2020-2025 data access
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Wifi,
  WifiOff,
  TrendingUp,
  TrendingDown,
  Database,
  RefreshCw,
  Clock,
  Signal,
  AlertCircle,
  CheckCircle,
  Calendar,
  BarChart3
} from 'lucide-react';

import AdvancedHistoricalChart from './AdvancedHistoricalChart';
import { realTimeWebSocketManager } from '../../lib/services/realtime-websocket-manager';
import { historicalDataManager } from '../../lib/services/historical-data-manager';

interface MarketStats {
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  lastUpdate: Date;
}

interface ConnectionInfo {
  status: 'connected' | 'disconnected' | 'connecting';
  quality: 'excellent' | 'good' | 'poor' | 'disconnected';
  latency: number;
  source: string;
  lastUpdate: Date | null;
}

export default function EnhancedTradingDashboard() {
  // State management
  const [marketStats, setMarketStats] = useState<MarketStats>({
    price: 0,
    change: 0,
    changePercent: 0,
    volume: 0,
    high24h: 0,
    low24h: 0,
    lastUpdate: new Date()
  });

  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    status: 'disconnected',
    quality: 'disconnected',
    latency: 0,
    source: 'none',
    lastUpdate: null
  });

  const [currentDate, setCurrentDate] = useState(new Date());
  const [dataStats, setDataStats] = useState({
    totalRecords: 0,
    cacheSize: 0,
    dateRange: { min: new Date(), max: new Date() }
  });

  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);
  const [showHistoricalOnly, setShowHistoricalOnly] = useState(false);

  /**
   * Initialize WebSocket connection and data manager
   */
  useEffect(() => {
    let unsubscribeData: (() => void) | null = null;
    let unsubscribeStatus: (() => void) | null = null;

    const initializeConnections = async () => {
      try {
        // Get data statistics
        const summary = await historicalDataManager.getDataSummary();
        setDataStats({
          totalRecords: summary.totalRecords,
          cacheSize: historicalDataManager.getCacheStats().size,
          dateRange: summary.dateRange
        });

        if (isRealTimeEnabled) {
          // Subscribe to real-time data
          unsubscribeData = realTimeWebSocketManager.onData((data) => {
            setMarketStats(prev => ({
              price: data.price,
              change: data.price - prev.price,
              changePercent: ((data.price - prev.price) / prev.price) * 100,
              volume: data.volume || prev.volume,
              high24h: Math.max(prev.high24h, data.price),
              low24h: Math.min(prev.low24h || data.price, data.price),
              lastUpdate: new Date(data.timestamp)
            }));
          });

          // Subscribe to connection status
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
        }

        // Load initial market data
        const latestData = await historicalDataManager.getLatestData('5m', 10);
        if (latestData.length > 0) {
          const latest = latestData[latestData.length - 1];
          const previous = latestData[latestData.length - 2];

          setMarketStats({
            price: latest.close,
            change: previous ? latest.close - previous.close : 0,
            changePercent: previous ? ((latest.close - previous.close) / previous.close) * 100 : 0,
            volume: latest.volume,
            high24h: Math.max(...latestData.map(d => d.high)),
            low24h: Math.min(...latestData.map(d => d.low)),
            lastUpdate: new Date(latest.time)
          });

          setCurrentDate(new Date(latest.time));
        }

      } catch (error) {
        console.error('Error initializing dashboard:', error);
      }
    };

    initializeConnections();

    // Cleanup on unmount
    return () => {
      unsubscribeData?.();
      unsubscribeStatus?.();
    };
  }, [isRealTimeEnabled]);

  /**
   * Update cache statistics periodically
   */
  useEffect(() => {
    const interval = setInterval(() => {
      setDataStats(prev => ({
        ...prev,
        cacheSize: historicalDataManager.getCacheStats().size
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  /**
   * Handle date change from chart navigator
   */
  const handleDateChange = useCallback((newDate: Date) => {
    setCurrentDate(newDate);
  }, []);

  /**
   * Toggle real-time mode
   */
  const toggleRealTime = useCallback(() => {
    setIsRealTimeEnabled(prev => !prev);

    if (isRealTimeEnabled) {
      realTimeWebSocketManager.disconnect();
      setConnectionInfo(prev => ({
        ...prev,
        status: 'disconnected',
        quality: 'disconnected'
      }));
    }
  }, [isRealTimeEnabled]);

  /**
   * Force reconnection
   */
  const forceReconnect = useCallback(() => {
    if (isRealTimeEnabled) {
      realTimeWebSocketManager.forceReconnect();
      setConnectionInfo(prev => ({
        ...prev,
        status: 'connecting'
      }));
    }
  }, [isRealTimeEnabled]);

  /**
   * Format numbers for display
   */
  const formatPrice = useCallback((price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(price);
  }, []);

  const formatChange = useCallback((change: number, isPercent: boolean = false) => {
    const formatted = isPercent
      ? `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`
      : `${change >= 0 ? '+' : ''}${change.toFixed(4)}`;

    return { formatted, isPositive: change >= 0 };
  }, []);

  /**
   * Connection status indicator
   */
  const ConnectionStatus = useMemo(() => {
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
        return <RefreshCw className="h-4 w-4 animate-spin" />;
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
    <div className="min-h-screen bg-slate-950 text-white p-4 space-y-6">
      {/* Header with market stats and controls */}
      <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-1">USD/COP</h1>
              <p className="text-slate-400 text-sm">Peso colombiano por dólar estadounidense</p>
            </div>

            <div className="flex items-center space-x-8">
              {/* Current price */}
              <div className="text-center">
                <div className="text-2xl font-mono font-bold text-white">
                  {formatPrice(marketStats.price)}
                </div>
                <div className="text-xs text-slate-400">Precio actual</div>
              </div>

              {/* Price change */}
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

              {/* Volume */}
              <div className="text-center">
                <div className="text-lg font-mono text-white">
                  {marketStats.volume.toLocaleString()}
                </div>
                <div className="text-xs text-slate-400">Volumen</div>
              </div>
            </div>
          </div>

          {/* Connection and controls */}
          <div className="flex items-center space-x-4">
            {/* Real-time toggle */}
            <div className="flex items-center space-x-2">
              <button
                onClick={toggleRealTime}
                className={`p-2 rounded-lg transition-colors ${
                  isRealTimeEnabled
                    ? 'bg-emerald-600 hover:bg-emerald-700'
                    : 'bg-slate-700 hover:bg-slate-600'
                }`}
                title={isRealTimeEnabled ? 'Desactivar tiempo real' : 'Activar tiempo real'}
              >
                <Activity className="h-4 w-4" />
              </button>
              <span className="text-sm text-slate-400">
                {isRealTimeEnabled ? 'Tiempo Real' : 'Solo Histórico'}
              </span>
            </div>

            {/* Connection status */}
            {ConnectionStatus}

            {/* Reconnect button */}
            {isRealTimeEnabled && connectionInfo.status === 'disconnected' && (
              <button
                onClick={forceReconnect}
                className="p-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors"
                title="Reconectar"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Data statistics */}
        <div className="flex items-center justify-between text-sm text-slate-400">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-1">
              <Database className="h-4 w-4" />
              <span>{dataStats.totalRecords.toLocaleString()} registros históricos</span>
            </div>
            <div className="flex items-center space-x-1">
              <Clock className="h-4 w-4" />
              <span>
                {dataStats.dateRange.min.getFullYear()} - {dataStats.dateRange.max.getFullYear()}
              </span>
            </div>
            <div className="flex items-center space-x-1">
              <BarChart3 className="h-4 w-4" />
              <span>{dataStats.cacheSize} chunks en caché</span>
            </div>
          </div>

          <div className="text-xs">
            Última actualización: {marketStats.lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Advanced Historical Chart */}
      <AdvancedHistoricalChart
        height={700}
        showVolume={true}
        showIndicators={true}
        enableRealTime={isRealTimeEnabled}
        onDateChange={handleDateChange}
      />

      {/* Additional stats row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* High/Low */}
        <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
          <div className="text-sm text-slate-400 mb-2">24h High/Low</div>
          <div className="space-y-1">
            <div className="text-emerald-400 font-mono">
              High: {formatPrice(marketStats.high24h)}
            </div>
            <div className="text-red-400 font-mono">
              Low: {formatPrice(marketStats.low24h)}
            </div>
          </div>
        </div>

        {/* Market Status */}
        <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
          <div className="text-sm text-slate-400 mb-2">Estado del Mercado</div>
          <div className="flex items-center space-x-2">
            {realTimeWebSocketManager.isMarketOpen() ? (
              <>
                <CheckCircle className="h-4 w-4 text-emerald-400" />
                <span className="text-emerald-400">Abierto</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-4 w-4 text-yellow-400" />
                <span className="text-yellow-400">Cerrado</span>
              </>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Horario: 8:00 - 12:55 COT
          </div>
        </div>

        {/* Data Source */}
        <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
          <div className="text-sm text-slate-400 mb-2">Fuente de Datos</div>
          <div className="text-white font-medium">
            {connectionInfo.source === 'websocket' ? 'WebSocket (Tiempo Real)' :
             connectionInfo.source === 'historical' ? 'Base de Datos (Histórico)' :
             'Sin conexión'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Calidad: {connectionInfo.quality}
          </div>
        </div>

        {/* Current View */}
        <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
          <div className="text-sm text-slate-400 mb-2">Vista Actual</div>
          <div className="text-white font-mono text-sm">
            {currentDate.toLocaleDateString('es-ES', {
              day: '2-digit',
              month: 'short',
              year: 'numeric'
            })}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {currentDate.toLocaleTimeString('es-ES')}
          </div>
        </div>
      </div>
    </div>
  );
}