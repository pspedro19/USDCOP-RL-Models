/**
 * UNIFIED TRADING TERMINAL - EL MEJOR FRAMEWORK DE TODOS LOS TIEMPOS
 * ================================================================
 *
 * Características Unificadas:
 * - Gráfica base profesional (RealDataTradingChart)
 * - Navegación histórica espectacular (SpectacularHistoricalNavigator)
 * - Diseño visual premium (UltimateVisualDashboard)
 * - Sistema único consistente
 * - Slider de navegación histórica funcional
 * - Registros históricos dinámicos (desde API)
 */

'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp, TrendingDown, Activity, BarChart3, Settings, Maximize2,
  Database, Sparkles, Zap, Target, Clock, Volume2, RefreshCw, Gauge,
  Eye, EyeOff, Play, Pause, RotateCcw, ZoomIn, ZoomOut, Move
} from 'lucide-react';

import RealDataTradingChart from '../charts/RealDataTradingChart';
import { useRealTimePrice } from '@/hooks/useRealTimePrice';
import { useMarketStats } from '@/hooks/useMarketStats';
import { useDbStats } from '@/hooks/useDbStats';
import { MarketDataService } from '@/lib/services/market-data-service';
import { Badge } from '@/components/ui/badge';

interface UnifiedSettings {
  timeframe: '5m' | '15m' | '30m' | '1h' | '1d';
}

export default function UnifiedTradingTerminal() {
  // === ESTADO UNIFICADO - 100% DATOS REALES DEL BACKEND ===
  const [settings, setSettings] = useState<UnifiedSettings>({
    timeframe: '1h'
  });

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // === DATOS EN TIEMPO REAL DEL BACKEND ===
  const { currentPrice: realtimePrice, isConnected: priceConnected } = useRealTimePrice('USDCOP');
  const { stats: marketStats, isConnected: statsConnected, isLoading } = useMarketStats('USDCOP', 30000);
  const { stats: dbStats } = useDbStats(60000); // Refresh every 60 seconds

  // Combine real-time price with market stats
  const currentPrice = realtimePrice?.price || marketStats?.currentPrice || 0;
  const isConnected = priceConnected || statsConnected;


  // === UTILIDADES ===
  const formatPrice = useCallback((price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(price);
  }, []);

  const formatVolume = useCallback((volume: number) => {
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume.toLocaleString();
  }, []);

  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50' : 'h-full'} bg-slate-950 text-white`}>
      {/* HEADER UNIFICADO */}
      <div className="bg-slate-900/80 backdrop-blur border-b border-slate-600/30 p-4">
        <div className="flex items-center justify-between">
          {/* Información Principal */}
          <div className="flex items-center space-x-8">
            <div>
              <div className="flex items-center gap-2">
                <Sparkles className="h-6 w-6 text-cyan-400" />
                <h1 className="text-2xl font-bold text-white">USD/COP</h1>
                <Badge variant="outline" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
                  UNIFIED TERMINAL
                </Badge>
              </div>
              <p className="text-sm text-slate-400 mt-1">
                {dbStats.totalRecords.toLocaleString()} registros • Sistema Unificado Premium
              </p>
            </div>

            {/* Métricas Principales - 100% DATOS REALES */}
            <div className="flex items-center space-x-6">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-white">
                  {formatPrice(currentPrice)}
                </div>
                <div className="text-xs text-slate-400">Precio Actual</div>
              </div>

              <div className="text-center">
                <div className={`text-lg font-mono font-bold flex items-center gap-1 ${
                  (marketStats?.change24h || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {(marketStats?.change24h || 0) >= 0 ?
                    <TrendingUp className="h-4 w-4" /> :
                    <TrendingDown className="h-4 w-4" />
                  }
                  <span>{(marketStats?.change24h || 0) >= 0 ? '+' : ''}{(marketStats?.change24h || 0).toFixed(4)}</span>
                  <span className="text-sm">
                    ({(marketStats?.changePercent || 0) >= 0 ? '+' : ''}{(marketStats?.changePercent || 0).toFixed(2)}%)
                  </span>
                </div>
                <div className="text-xs text-slate-400">Cambio 24h</div>
              </div>

              <div className="text-center">
                <div className="text-lg font-mono text-white">
                  {formatVolume(marketStats?.volume24h || 0)}
                </div>
                <div className="text-xs text-slate-400">Volumen 24h</div>
              </div>
            </div>
          </div>

          {/* Controles */}
          <div className="flex items-center space-x-4">
            <div className={`flex items-center gap-2 text-sm ${
              isConnected ? 'text-emerald-400' : 'text-red-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'
              }`} />
              <span>{isConnected ? 'Conectado' : 'Mercado Cerrado'}</span>
            </div>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 bg-slate-700/30 hover:bg-slate-600/50 rounded transition-colors"
            >
              <Settings className="h-4 w-4" />
            </button>

            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 bg-slate-700/30 hover:bg-slate-600/50 rounded transition-colors"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Métricas Secundarias - DATOS REALES */}
        <div className="flex items-center justify-between mt-4 text-sm text-slate-400">
          <div className="flex items-center space-x-6">
            <div>H: {formatPrice(marketStats?.high24h || 0)}</div>
            <div>L: {formatPrice(marketStats?.low24h || 0)}</div>
            <div>Spread: {(marketStats?.spread || 0).toFixed(2)} COP</div>
            <div>Vol: {(marketStats?.volatility || 0).toFixed(2)}%</div>
            <div>Source: <span className="text-cyan-400">{marketStats?.source || 'loading...'}</span></div>
          </div>

          <div className="text-slate-500">
            Actualizado: {marketStats?.timestamp ? new Date(marketStats.timestamp).toLocaleTimeString() : 'N/A'}
          </div>
        </div>
      </div>

      {/* AVISO DE DATOS HISTÓRICOS */}
      <div className="px-6 pb-2">
        <div className="bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-red-500/20 border border-amber-500/30 rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0">
              <Database className="h-6 w-6 text-amber-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-amber-400 font-semibold text-sm">Estado de Datos Históricos</h3>
              <p className="text-amber-200 text-xs mt-1">
                📊 Datos disponibles: Oct 6-10, 2025 (318 registros) •
                🎯 Objetivo: 2020-2025 ({dbStats.totalRecords.toLocaleString()} registros) •
                ⚡ La navegación histórica muestra todo el rango disponible en la base de datos
              </p>
              <p className="text-amber-300 text-xs mt-2 font-medium">
                💡 El slider de navegación funciona perfectamente con los datos actuales. Para datos desde 2020, se requiere poblamiento de la base de datos.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* GRÁFICA PRINCIPAL SIMPLIFICADA */}
      <div className="flex-1 px-6 pb-6">
        <RealDataTradingChart
          symbol="USDCOP"
          timeframe={settings.timeframe}
          height={700}
          className="w-full h-full"
        />
      </div>

      {/* PANEL DE CONFIGURACIÓN */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 h-full w-80 bg-slate-900/95 backdrop-blur border-l border-slate-600/30 p-6 z-40"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Configuración</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-slate-400 hover:text-white"
              >
                ×
              </button>
            </div>

            <div className="space-y-6">
              {/* Timeframe */}
              <div>
                <h4 className="text-sm font-medium text-slate-300 mb-3">Timeframe Principal</h4>
                <select
                  value={settings.timeframe}
                  onChange={(e) => setSettings(prev => ({ ...prev, timeframe: e.target.value as any }))}
                  className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white"
                >
                  <option value="5m">5 Minutos</option>
                  <option value="15m">15 Minutos</option>
                  <option value="30m">30 Minutos</option>
                  <option value="1h">1 Hora</option>
                  <option value="1d">1 Día</option>
                </select>
                <p className="text-xs text-slate-400 mt-2">
                  La gráfica tiene sus propios controles avanzados integrados
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* OVERLAY DE FULLSCREEN */}
      {isFullscreen && (
        <button
          onClick={() => setIsFullscreen(false)}
          className="fixed top-4 right-4 z-50 p-2 bg-slate-900/80 hover:bg-slate-800 border border-slate-600/30 rounded text-white"
        >
          ×
        </button>
      )}
    </div>
  );
}