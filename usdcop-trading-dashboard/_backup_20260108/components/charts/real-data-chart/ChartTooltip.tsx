'use client'

/**
 * Chart Tooltip Component
 * =======================
 *
 * Professional hover tooltip for displaying candle data.
 */

import React from 'react'
import { HoverInfo, ChartType } from './types'

interface ChartTooltipProps {
  hoverInfo: HoverInfo
  chartType: ChartType
  showIndicators: boolean
}

export function ChartTooltip({ hoverInfo, chartType, showIndicators }: ChartTooltipProps) {
  if (!hoverInfo.visible) return null

  return (
    <div
      className="fixed bg-slate-900/95 border border-slate-600 rounded-xl p-4 pointer-events-none z-40 shadow-2xl backdrop-blur-sm"
      style={{
        left: hoverInfo.x + 15,
        top: hoverInfo.y - 15,
        transform: hoverInfo.x > window.innerWidth - 300 ? 'translateX(-100%)' : 'none'
      }}
    >
      <div className="text-white font-bold text-sm mb-3 border-b border-slate-600 pb-2">
        {new Date(hoverInfo.candle.time).toLocaleString('es-CO')}
      </div>

      {chartType === 'candlestick' ? (
        <CandlestickTooltipContent hoverInfo={hoverInfo} showIndicators={showIndicators} />
      ) : (
        <LineTooltipContent hoverInfo={hoverInfo} />
      )}
    </div>
  )
}

function CandlestickTooltipContent({
  hoverInfo,
  showIndicators
}: {
  hoverInfo: HoverInfo
  showIndicators: boolean
}) {
  return (
    <div className="space-y-2 text-xs">
      <div className="grid grid-cols-2 gap-4">
        <div className="flex justify-between">
          <span className="text-gray-400">Apertura:</span>
          <span className="text-cyan-400 font-mono font-bold">${hoverInfo.candle.open?.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Maximo:</span>
          <span className="text-green-400 font-mono font-bold">${hoverInfo.candle.high?.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Minimo:</span>
          <span className="text-red-400 font-mono font-bold">${hoverInfo.candle.low?.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Cierre:</span>
          <span className="text-yellow-400 font-mono font-bold">${hoverInfo.candle.close?.toFixed(2)}</span>
        </div>
      </div>

      <div className="flex justify-between pt-2 border-t border-slate-700">
        <span className="text-gray-400">Volumen:</span>
        <span className="text-purple-400 font-mono font-bold">{hoverInfo.candle.volume?.toLocaleString()}</span>
      </div>

      {showIndicators && hoverInfo.candle.indicators && (
        <IndicatorsSection indicators={hoverInfo.candle.indicators} />
      )}
    </div>
  )
}

function LineTooltipContent({ hoverInfo }: { hoverInfo: HoverInfo }) {
  return (
    <div className="space-y-2 text-xs">
      <div className="flex justify-between">
        <span className="text-gray-400">Precio:</span>
        <span className="text-cyan-400 font-mono font-bold">${hoverInfo.candle.close?.toFixed(2)}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-400">Volumen:</span>
        <span className="text-purple-400 font-mono font-bold">{hoverInfo.candle.volume?.toLocaleString()}</span>
      </div>
    </div>
  )
}

function IndicatorsSection({
  indicators
}: {
  indicators: { ema_20?: number; ema_50?: number; [key: string]: number | undefined }
}) {
  return (
    <div className="border-t border-slate-600 pt-2 mt-2">
      <div className="text-blue-300 font-bold text-xs mb-1">Indicadores Tecnicos</div>
      {indicators.ema_20 && (
        <div className="flex justify-between">
          <span className="text-blue-400">EMA 20:</span>
          <span className="text-blue-400 font-mono">${indicators.ema_20.toFixed(2)}</span>
        </div>
      )}
      {indicators.ema_50 && (
        <div className="flex justify-between">
          <span className="text-orange-400">EMA 50:</span>
          <span className="text-orange-400 font-mono">${indicators.ema_50.toFixed(2)}</span>
        </div>
      )}
    </div>
  )
}

export default ChartTooltip
