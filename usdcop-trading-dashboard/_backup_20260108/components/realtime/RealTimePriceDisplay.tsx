'use client'

/**
 * Real-Time Price Display Component
 * ================================
 *
 * Componente que muestra el precio USDCOP en tiempo real
 * Conecta vía WebSocket con la Trading API
 * Actualiza automáticamente cada 5 minutos cuando llegan nuevos datos
 */

import React from 'react'
import { useRealTimePrice } from '@/hooks/useRealTimePrice'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TrendingUp, TrendingDown, Wifi, WifiOff, Activity } from 'lucide-react'

interface RealTimePriceDisplayProps {
  symbol?: string
  className?: string
}

export default function RealTimePriceDisplay({
  symbol = 'USDCOP',
  className = ''
}: RealTimePriceDisplayProps) {
  const {
    currentPrice,
    isConnected,
    error,
    priceChange,
    priceChangePercent,
    isIncreasing,
    formattedPrice,
    formattedChange,
    formattedChangePercent,
    lastUpdated
  } = useRealTimePrice(symbol)

  const getChangeColor = () => {
    if (isIncreasing === null) return 'text-gray-500'
    return isIncreasing ? 'text-green-600' : 'text-red-600'
  }

  const getChangeIcon = () => {
    if (isIncreasing === null) return <Activity className="w-4 h-4" />
    return isIncreasing ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />
  }

  const getConnectionStatus = () => {
    if (error) {
      return (
        <Badge variant="destructive" className="flex items-center gap-1">
          <WifiOff className="w-3 h-3" />
          Error
        </Badge>
      )
    }

    if (isConnected) {
      return (
        <Badge variant="default" className="flex items-center gap-1 bg-green-600">
          <Wifi className="w-3 h-3" />
          En vivo
        </Badge>
      )
    }

    return (
      <Badge variant="secondary" className="flex items-center gap-1">
        <WifiOff className="w-3 h-3" />
        Conectando...
      </Badge>
    )
  }

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          {symbol} - Precio en Tiempo Real
        </CardTitle>
        {getConnectionStatus()}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Precio actual */}
          <div className="flex items-baseline space-x-2">
            <div className="text-2xl font-bold">
              {formattedPrice}
            </div>
            {currentPrice?.source && (
              <Badge variant="outline" className="text-xs">
                {currentPrice.source}
              </Badge>
            )}
          </div>

          {/* Cambio de precio */}
          {priceChange !== null && (
            <div className={`flex items-center space-x-2 ${getChangeColor()}`}>
              {getChangeIcon()}
              <div className="flex space-x-2">
                <span className="font-medium">
                  {formattedChange}
                </span>
                <span className="text-sm">
                  ({formattedChangePercent})
                </span>
              </div>
            </div>
          )}

          {/* Información adicional */}
          <div className="text-xs text-gray-500 space-y-1">
            {lastUpdated && (
              <div>Última actualización: {lastUpdated}</div>
            )}
            {currentPrice?.volume !== undefined && (
              <div>Volumen: {currentPrice.volume.toLocaleString()}</div>
            )}
            {error && (
              <div className="text-red-500">Error: {error}</div>
            )}
          </div>

          {/* Indicador de datos reales */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Datos 100% reales</span>
            <span className="text-gray-500">Actualización: cada 5min</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}