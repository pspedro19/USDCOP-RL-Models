/**
 * useIntegratedChart Hook
 * ========================
 *
 * Manages both market data and trading signals for integrated chart views.
 * Synchronizes data streams, handles loading states, and provides unified state.
 *
 * Features:
 * - Market data fetching and caching
 * - Trading signals integration
 * - Position tracking
 * - WebSocket real-time updates
 * - Unified loading and error states
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { TradingSignal, Position, CandlestickData } from '@/types/trading'
import { createLogger } from '@/lib/utils/logger'

const logger = createLogger('useIntegratedChart')

interface UseIntegratedChartOptions {
  symbol?: string
  timeframe?: string
  startDate?: string
  endDate?: string
  limit?: number
  enableSignals?: boolean
  enablePositions?: boolean
  enableRealTime?: boolean
  refreshInterval?: number
  modelId?: string  // Model ID for filtering signals (ppo_v19_prod, ppo_v20_prod)
}

interface UseIntegratedChartReturn {
  // Market Data
  candlestickData: CandlestickData[]
  dataLoading: boolean
  dataError: string | null

  // Trading Signals
  signals: TradingSignal[]
  signalsLoading: boolean
  signalsError: string | null

  // Positions
  positions: Position[]
  positionsLoading: boolean
  positionsError: string | null

  // Combined State
  isLoading: boolean
  hasError: boolean
  errorMessage: string | null

  // Actions
  refresh: () => Promise<void>
  refreshMarketData: () => Promise<void>
  refreshSignals: () => Promise<void>
  refreshPositions: () => Promise<void>
}

export function useIntegratedChart({
  symbol = 'USDCOP',
  timeframe = '5m',
  startDate, // Optional - don't send to API if not specified
  endDate,   // Optional - don't send to API if not specified
  limit = 1000,
  enableSignals = true,
  enablePositions = true,
  enableRealTime = false,
  refreshInterval = 30000,
  modelId = 'ppo_v19_prod'  // Default to V19, can be ppo_v19_prod or ppo_v20_prod
}: UseIntegratedChartOptions = {}): UseIntegratedChartReturn {
  // Market Data State
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([])
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError, setDataError] = useState<string | null>(null)

  // Signals State
  const [signals, setSignals] = useState<TradingSignal[]>([])
  const [signalsLoading, setSignalsLoading] = useState(false)
  const [signalsError, setSignalsError] = useState<string | null>(null)

  // Positions State
  const [positions, setPositions] = useState<Position[]>([])
  const [positionsLoading, setPositionsLoading] = useState(false)
  const [positionsError, setPositionsError] = useState<string | null>(null)

  // Refs for cleanup
  const abortControllerRef = useRef<AbortController | null>(null)
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null)

  /**
   * Fetch market candlestick data using filtered endpoint
   * This returns only bars with price movement (filters out flat weekend/off-hours bars)
   */
  const refreshMarketData = useCallback(async () => {
    try {
      setDataLoading(true)
      setDataError(null)

      // Use filtered endpoint for better performance - excludes flat bars
      const start = startDate || '2025-01-01'
      const end = endDate || new Date().toISOString().split('T')[0]

      const response = await fetch(
        `/api/market/candlesticks-filtered?start_date=${start}&end_date=${end}&limit=${limit}`
      )

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`)
      }

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch candlestick data')
      }

      logger.debug(`Fetched ${result.count} candlesticks with price movement`)
      setCandlestickData(result.data)
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to fetch market data'
      logger.error('Error fetching market data:', err)
      setDataError(errorMsg)
    } finally {
      setDataLoading(false)
    }
  }, [startDate, endDate, limit])

  /**
   * Fetch trading signals for selected model
   * Maps frontend model IDs to database model IDs
   */
  const refreshSignals = useCallback(async () => {
    if (!enableSignals) return

    try {
      setSignalsLoading(true)
      setSignalsError(null)

      // Map frontend model ID to database model_id
      const dbModelId = modelId === 'ppo_v20_prod' ? 'ppo_v20_macro' : 'ppo_v1'

      // Fetch signals for the selected model
      const response = await fetch(`/api/trading/signals?action=recent&limit=500&model_id=${dbModelId}`)
      const data = await response.json()

      if (data.success && data.signals) {
        logger.debug(`Fetched ${data.signals.length} signals for model ${modelId} (db: ${dbModelId})`)
        setSignals(data.signals)
      } else {
        // Ensure error message is a string
        const errorMsg = typeof data.error === 'string'
          ? data.error
          : data.error?.message || data.message || 'Failed to fetch signals'
        throw new Error(errorMsg)
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to fetch signals'
      logger.error('Error fetching signals:', err)
      setSignalsError(errorMsg)
      setSignals([])
    } finally {
      setSignalsLoading(false)
    }
  }, [enableSignals, modelId])

  /**
   * Fetch active positions
   */
  const refreshPositions = useCallback(async () => {
    if (!enablePositions) return

    try {
      setPositionsLoading(true)
      setPositionsError(null)

      const response = await fetch('/api/agent/actions?action=today')
      const data = await response.json()

      if (data.success) {
        // Extract positions from agent actions
        const activePositions: Position[] = []

        if (data.data?.actions) {
          // Process actions to build position history
          let currentPosition: Position | null = null

          data.data.actions.forEach((action: any) => {
            if (action.action_type === 'ENTRY_LONG' || action.action_type === 'ENTRY_SHORT') {
              currentPosition = {
                id: `pos-${action.action_id}`,
                symbol,
                side: action.side === 'long' ? 'long' : 'short',
                entryPrice: action.price_at_action,
                currentPrice: action.price_at_action,
                quantity: Math.abs(action.position_after),
                notionalValue: action.price_at_action * Math.abs(action.position_after),
                unrealizedPnl: 0,
                realizedPnl: 0,
                openedAt: action.timestamp_cot,
                status: 'open'
              }
            } else if ((action.action_type === 'EXIT_LONG' || action.action_type === 'EXIT_SHORT') && currentPosition) {
              currentPosition.closedAt = action.timestamp_cot
              currentPosition.status = 'closed'
              currentPosition.currentPrice = action.price_at_action
              currentPosition.realizedPnl = (action.price_at_action - currentPosition.entryPrice) * currentPosition.quantity
              activePositions.push(currentPosition)
              currentPosition = null
            }
          })

          // Add current open position if exists
          if (currentPosition) {
            activePositions.push(currentPosition)
          }
        }

        logger.debug(`Loaded ${activePositions.length} positions`)
        setPositions(activePositions)
      } else {
        throw new Error(data.error || 'Failed to fetch positions')
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to fetch positions'
      logger.error('Error fetching positions:', err)
      setPositionsError(errorMsg)
      setPositions([])
    } finally {
      setPositionsLoading(false)
    }
  }, [enablePositions, symbol])

  /**
   * Refresh all data
   */
  const refresh = useCallback(async () => {
    await Promise.all([
      refreshMarketData(),
      enableSignals ? refreshSignals() : Promise.resolve(),
      enablePositions ? refreshPositions() : Promise.resolve()
    ])
  }, [refreshMarketData, refreshSignals, refreshPositions, enableSignals, enablePositions])

  /**
   * Initial data load
   */
  useEffect(() => {
    refresh()
  }, [refresh])

  /**
   * Auto-refresh interval
   */
  useEffect(() => {
    if (!enableRealTime || refreshInterval <= 0) return

    refreshIntervalRef.current = setInterval(() => {
      refresh()
    }, refreshInterval)

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
      }
    }
  }, [enableRealTime, refreshInterval, refresh])

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
      }
    }
  }, [])

  // Compute combined states
  const isLoading = dataLoading || signalsLoading || positionsLoading
  const hasError = !!(dataError || signalsError || positionsError)
  const errorMessage = dataError || signalsError || positionsError

  return {
    // Market Data
    candlestickData,
    dataLoading,
    dataError,

    // Trading Signals
    signals,
    signalsLoading,
    signalsError,

    // Positions
    positions,
    positionsLoading,
    positionsError,

    // Combined State
    isLoading,
    hasError,
    errorMessage,

    // Actions
    refresh,
    refreshMarketData,
    refreshSignals,
    refreshPositions
  }
}
