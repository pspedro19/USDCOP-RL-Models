'use client'

/**
 * Chart Controls Component
 * ========================
 *
 * Professional toolbar controls for the trading chart.
 */

import React from 'react'
import {
  Activity,
  BarChart3,
  TrendingUp,
  Eye,
  EyeOff,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Volume2,
  Clock
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ChartType, TimeframePeriod, ChartSettings } from './types'

interface ChartControlsProps {
  isConnected: boolean
  chartType: ChartType
  setChartType: (type: ChartType) => void
  chartSettings: ChartSettings
  setChartSettings: React.Dispatch<React.SetStateAction<ChartSettings>>
  showIndicators: boolean
  setShowIndicators: (show: boolean) => void
  loading: boolean
  onRefresh: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  onResetView: () => void
}

export function ChartControls({
  isConnected,
  chartType,
  setChartType,
  chartSettings,
  setChartSettings,
  showIndicators,
  setShowIndicators,
  loading,
  onRefresh,
  onZoomIn,
  onZoomOut,
  onResetView
}: ChartControlsProps) {
  return (
    <div className="absolute top-4 right-4 z-30 flex flex-wrap items-center gap-2">
      {/* Connection Status */}
      <Badge
        variant={isConnected ? 'default' : 'secondary'}
        className={`${isConnected ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-200'} font-bold`}
      >
        <Activity className="w-3 h-3 mr-1" />
        {isConnected ? 'EN VIVO' : 'OFFLINE'}
      </Badge>

      {/* Chart Type Toggle */}
      <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
        <Button
          variant={chartType === 'candlestick' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setChartType('candlestick')}
          className={`text-xs px-3 py-1 ${chartType === 'candlestick' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
        >
          <BarChart3 className="w-3 h-3 mr-1" />
          Velas 3D
        </Button>
        <Button
          variant={chartType === 'line' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setChartType('line')}
          className={`text-xs px-3 py-1 ${chartType === 'line' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
        >
          <TrendingUp className="w-3 h-3 mr-1" />
          Linea Pro
        </Button>
      </div>

      {/* Zoom Controls */}
      <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
        <Button variant="ghost" size="sm" onClick={onZoomIn} className="text-xs px-2 py-1 text-gray-300 hover:text-white">
          <ZoomIn className="w-3 h-3" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onZoomOut} className="text-xs px-2 py-1 text-gray-300 hover:text-white">
          <ZoomOut className="w-3 h-3" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onResetView} className="text-xs px-2 py-1 text-gray-300 hover:text-white">
          <RotateCcw className="w-3 h-3" />
        </Button>
      </div>

      {/* Volume & Indicators */}
      <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
        <Button
          variant={chartSettings.showVolume ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setChartSettings((prev) => ({ ...prev, showVolume: !prev.showVolume }))}
          className={`text-xs px-3 py-1 ${chartSettings.showVolume ? 'bg-purple-600 text-white' : 'text-gray-300 hover:text-white'}`}
        >
          <Volume2 className="w-3 h-3 mr-1" />
          Vol
        </Button>
        <Button
          variant={showIndicators ? 'default' : 'ghost'}
          size="sm"
          onClick={() => setShowIndicators(!showIndicators)}
          className={`text-xs px-3 py-1 ${showIndicators ? 'bg-orange-600 text-white' : 'text-gray-300 hover:text-white'}`}
        >
          {showIndicators ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
          Tech
        </Button>
      </div>

      {/* Refresh */}
      <Button
        variant="outline"
        size="sm"
        onClick={onRefresh}
        disabled={loading}
        className="text-xs border-slate-600 text-gray-300 hover:text-white"
      >
        <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
      </Button>
    </div>
  )
}

interface TimeframeSelectorProps {
  selectedTimeframe: TimeframePeriod
  setSelectedTimeframe: (tf: TimeframePeriod) => void
}

export function TimeframeSelector({ selectedTimeframe, setSelectedTimeframe }: TimeframeSelectorProps) {
  const timeframes: TimeframePeriod[] = ['5m', '15m', '30m', '1h', '1d']

  return (
    <div className="absolute top-4 left-4 z-30">
      <div className="bg-slate-800/95 border border-slate-600 rounded-lg p-2">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-blue-400" />
          <div className="flex gap-1">
            {timeframes.map((tf) => (
              <Button
                key={tf}
                variant={selectedTimeframe === tf ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setSelectedTimeframe(tf)}
                className={`text-xs px-3 py-1 ${
                  selectedTimeframe === tf
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                {tf.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

interface ChartLegendProps {
  showIndicators: boolean
  showVolume: boolean
}

export function ChartLegend({ showIndicators, showVolume }: ChartLegendProps) {
  if (!showIndicators) return null

  return (
    <div className="absolute bottom-4 right-4 bg-slate-900/95 border border-slate-600 rounded-xl p-4 backdrop-blur-sm">
      <div className="text-white font-bold text-xs mb-2">Indicadores</div>
      <div className="grid grid-cols-1 gap-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-blue-400 rounded shadow-lg"></div>
          <span className="text-blue-400 font-bold">EMA 20</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-orange-400 rounded shadow-lg"></div>
          <span className="text-orange-400 font-bold">EMA 50</span>
        </div>
        {showVolume && (
          <div className="flex items-center gap-2">
            <div className="w-4 h-1 bg-purple-400 rounded shadow-lg"></div>
            <span className="text-purple-400 font-bold">Volumen</span>
          </div>
        )}
      </div>
    </div>
  )
}

interface ChartInfoPanelProps {
  totalDataPoints: number
  visibleDataPoints: number
  zoomLevel: number
  selectedTimeframe: TimeframePeriod
}

export function ChartInfoPanel({ totalDataPoints, visibleDataPoints, zoomLevel, selectedTimeframe }: ChartInfoPanelProps) {
  return (
    <div className="absolute top-20 right-4 bg-slate-900/95 border border-slate-600 rounded-xl p-3 backdrop-blur-sm">
      <div className="text-xs space-y-1">
        <div className="text-gray-300 font-bold">Estado del Grafico</div>
        <div className="text-green-400 font-mono">{totalDataPoints} puntos totales</div>
        <div className="text-blue-400 font-mono">{visibleDataPoints} visibles</div>
        <div className="text-purple-400 font-mono">Zoom: {zoomLevel.toFixed(1)}x</div>
        <div className="text-orange-400 font-mono">{selectedTimeframe.toUpperCase()}</div>
      </div>
    </div>
  )
}

export default ChartControls
