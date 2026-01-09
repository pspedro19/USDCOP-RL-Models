/**
 * Real-Time Price Hook
 * ===================
 *
 * Hook de React para obtener precios en tiempo real vía WebSocket
 * Conecta automáticamente con la Trading API
 */

import { useState, useEffect, useRef } from 'react'
import { MarketDataService, MarketDataPoint } from '@/lib/services/market-data-service'

interface RealTimePriceState {
  currentPrice: MarketDataPoint | null
  previousPrice: number | null
  isConnected: boolean
  error: string | null
  priceChange: number | null
  priceChangePercent: number | null
  isIncreasing: boolean | null
}

export function useRealTimePrice(symbol: string = 'USDCOP') {
  const [state, setState] = useState<RealTimePriceState>({
    currentPrice: null,
    previousPrice: null,
    isConnected: false,
    error: null,
    priceChange: null,
    priceChangePercent: null,
    isIncreasing: null
  })

  const unsubscribeRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    // Get initial data
    const fetchInitialData = async () => {
      try {
        const data = await MarketDataService.getRealTimeData()
        if (data.length > 0) {
          const currentPrice = data[0]
          setState(prev => ({
            ...prev,
            currentPrice,
            previousPrice: prev.currentPrice?.price || null,
            isConnected: true,
            error: null
          }))
        }
      } catch (error) {
        setState(prev => ({
          ...prev,
          error: 'Failed to fetch initial data',
          isConnected: false
        }))
      }
    }

    fetchInitialData()

    // Subscribe to WebSocket updates
    const unsubscribe = MarketDataService.subscribeToRealTimeUpdates((newPrice: MarketDataPoint) => {
      if (newPrice.symbol === symbol) {
        setState(prev => {
          const previousPrice = prev.currentPrice?.price || null
          const priceChange = previousPrice ? newPrice.price - previousPrice : null
          const priceChangePercent = previousPrice && previousPrice > 0
            ? (priceChange! / previousPrice) * 100
            : null

          return {
            ...prev,
            currentPrice: newPrice,
            previousPrice,
            isConnected: true,
            error: null,
            priceChange,
            priceChangePercent,
            isIncreasing: priceChange !== null ? priceChange > 0 : null
          }
        })
      }
    })

    unsubscribeRef.current = unsubscribe

    // Cleanup
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current()
      }
    }
  }, [symbol])

  // Format price for display
  const formatPrice = (price: number, decimals: number = 4): string => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(price)
  }

  // Format change for display
  const formatChange = (change: number | null, asPercentage: boolean = false): string => {
    if (change === null) return '--'

    const value = asPercentage ? change : change
    const sign = value >= 0 ? '+' : ''

    if (asPercentage) {
      return `${sign}${value.toFixed(2)}%`
    } else {
      return `${sign}${value.toFixed(4)}`
    }
  }

  return {
    ...state,
    formatPrice,
    formatChange,
    // Convenience getters
    get formattedPrice() {
      return state.currentPrice ? formatPrice(state.currentPrice.price) : '--'
    },
    get formattedChange() {
      return formatChange(state.priceChange)
    },
    get formattedChangePercent() {
      return formatChange(state.priceChangePercent, true)
    },
    get lastUpdated() {
      return state.currentPrice ? new Date(state.currentPrice.timestamp).toLocaleString() : null
    }
  }
}