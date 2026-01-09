/**
 * useSignalOverlay Hook
 * ======================
 *
 * Custom hook for fetching and managing trading signals with real-time updates
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { Time } from 'lightweight-charts'
import useWebSocket, { ReadyState } from 'react-use-websocket'
import { SignalData, SignalFilterOptions, SignalStats, SignalUpdate } from '@/components/charts/signal-overlay/types'
import { OrderType } from '@/types/trading'

interface TradingSignalResponse {
  success: boolean
  signals: any[]
  performance?: {
    winRate: number
    avgWin: number
    avgLoss: number
    profitFactor: number
    sharpeRatio: number
    totalSignals: number
    activeSignals: number
  }
  lastUpdate: string
  dataSource?: string
}

interface UseSignalOverlayOptions {
  autoRefresh?: boolean
  refreshInterval?: number
  enableWebSocket?: boolean
  websocketUrl?: string
  filter?: SignalFilterOptions
}

export function useSignalOverlay(options: UseSignalOverlayOptions = {}) {
  const {
    autoRefresh = true,
    refreshInterval = 30000,
    enableWebSocket = false,
    websocketUrl = 'ws://localhost:3001',
    filter = {},
  } = options

  // State
  const [signals, setSignals] = useState<SignalData[]>([])
  const [filteredSignals, setFilteredSignals] = useState<SignalData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [stats, setStats] = useState<SignalStats>({
    total: 0,
    active: 0,
    closed: 0,
    winRate: 0,
    avgConfidence: 0,
    totalPnl: 0,
    avgPnl: 0,
  })

  // Refs
  const fetchAbortController = useRef<AbortController | null>(null)

  /**
   * Convert API signal to internal format
   */
  const convertApiSignal = useCallback((apiSignal: any): SignalData => {
    const timestamp = new Date(apiSignal.timestamp)
    const time = Math.floor(timestamp.getTime() / 1000) as Time

    // Calculate PnL if exitPrice is available
    let pnl: number | undefined
    if (apiSignal.exitPrice) {
      if (apiSignal.type === OrderType.BUY) {
        pnl = (apiSignal.exitPrice - apiSignal.price) * 100 // Assuming 100 units
      } else if (apiSignal.type === OrderType.SELL) {
        pnl = (apiSignal.price - apiSignal.exitPrice) * 100
      }
    }

    // Determine status
    let status: 'active' | 'closed' | 'cancelled' = 'active'
    if (apiSignal.exitTimestamp) {
      status = 'closed'
    }

    return {
      id: apiSignal.id || `sig_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: apiSignal.timestamp,
      time,
      type: apiSignal.type as OrderType,
      confidence: apiSignal.confidence,
      price: apiSignal.price,
      stopLoss: apiSignal.stopLoss,
      takeProfit: apiSignal.takeProfit,
      reasoning: apiSignal.reasoning || [],
      riskScore: apiSignal.riskScore || 5,
      expectedReturn: apiSignal.expectedReturn || 0,
      timeHorizon: apiSignal.timeHorizon || '15-30 min',
      modelSource: apiSignal.modelSource || 'Unknown',
      latency: apiSignal.latency || 0,
      exitPrice: apiSignal.exitPrice,
      exitTimestamp: apiSignal.exitTimestamp,
      exitTime: apiSignal.exitTimestamp
        ? (Math.floor(new Date(apiSignal.exitTimestamp).getTime() / 1000) as Time)
        : undefined,
      pnl,
      status,
    }
  }, [])

  /**
   * Fetch signals from API
   */
  const fetchSignals = useCallback(async () => {
    try {
      // Cancel any pending requests
      if (fetchAbortController.current) {
        fetchAbortController.current.abort()
      }

      fetchAbortController.current = new AbortController()

      const response = await fetch('/api/trading/signals', {
        signal: fetchAbortController.current.signal,
      })

      const data: TradingSignalResponse = await response.json()

      if (data.success && data.signals) {
        const convertedSignals = data.signals.map(convertApiSignal)
        setSignals(convertedSignals)
        setLastUpdate(new Date(data.lastUpdate))
        setError(null)

        // Update stats from performance data
        if (data.performance) {
          const activeSignals = convertedSignals.filter((s) => s.status === 'active')
          const closedSignals = convertedSignals.filter((s) => s.status === 'closed')
          const totalPnl = closedSignals.reduce((sum, s) => sum + (s.pnl || 0), 0)
          const avgConfidence =
            convertedSignals.reduce((sum, s) => sum + s.confidence, 0) / convertedSignals.length

          setStats({
            total: convertedSignals.length,
            active: activeSignals.length,
            closed: closedSignals.length,
            winRate: data.performance.winRate,
            avgConfidence,
            totalPnl,
            avgPnl: closedSignals.length > 0 ? totalPnl / closedSignals.length : 0,
            bestSignal: closedSignals.sort((a, b) => (b.pnl || 0) - (a.pnl || 0))[0],
            worstSignal: closedSignals.sort((a, b) => (a.pnl || 0) - (b.pnl || 0))[0],
          })
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('[useSignalOverlay] Error fetching signals:', err)
        setError('Failed to load trading signals')
      }
    } finally {
      setLoading(false)
    }
  }, [convertApiSignal])

  /**
   * WebSocket connection for real-time updates
   */
  const { lastMessage, readyState } = useWebSocket(
    enableWebSocket ? websocketUrl : null,
    {
      onOpen: () => {
        console.log('[useSignalOverlay] WebSocket connected')
        // Subscribe to signal updates
        // Note: Implementation depends on your WebSocket server
      },
      onClose: () => {
        console.log('[useSignalOverlay] WebSocket disconnected')
      },
      onError: () => {
        // Silent when backend unavailable
      },
      shouldReconnect: () => true,
      reconnectInterval: 3000,
    }
  )

  /**
   * Handle WebSocket messages
   */
  useEffect(() => {
    if (!lastMessage || !enableWebSocket) return

    try {
      const message: SignalUpdate = JSON.parse(lastMessage.data)

      if (message.type === 'new_signal') {
        // Add new signal
        const newSignal = convertApiSignal(message.signal)
        setSignals((prev) => [...prev, newSignal])
      } else if (message.type === 'update_signal') {
        // Update existing signal
        const updatedSignal = convertApiSignal(message.signal)
        setSignals((prev) =>
          prev.map((s) => (s.id === updatedSignal.id ? updatedSignal : s))
        )
      } else if (message.type === 'close_signal') {
        // Close signal
        const closedSignal = convertApiSignal(message.signal)
        setSignals((prev) =>
          prev.map((s) => (s.id === closedSignal.id ? closedSignal : s))
        )
      }

      setLastUpdate(new Date(message.timestamp))
    } catch (err) {
      console.error('[useSignalOverlay] Error parsing WebSocket message:', err)
    }
  }, [lastMessage, enableWebSocket, convertApiSignal])

  /**
   * Apply filters to signals
   */
  useEffect(() => {
    let filtered = [...signals]

    // Filter by date range
    if (filter.startDate || filter.endDate) {
      filtered = filtered.filter((signal) => {
        const signalDate = new Date(signal.timestamp)
        if (filter.startDate && signalDate < filter.startDate) return false
        if (filter.endDate && signalDate > filter.endDate) return false
        return true
      })
    }

    // Filter by action types
    if (filter.actionTypes && filter.actionTypes.length > 0) {
      filtered = filtered.filter((signal) =>
        filter.actionTypes!.includes(signal.type)
      )
    }

    // Filter by confidence
    if (filter.minConfidence !== undefined) {
      filtered = filtered.filter((signal) => signal.confidence >= filter.minConfidence!)
    }
    if (filter.maxConfidence !== undefined) {
      filtered = filtered.filter((signal) => signal.confidence <= filter.maxConfidence!)
    }

    // Filter HOLD signals
    if (!filter.showHold) {
      filtered = filtered.filter((signal) => signal.type !== OrderType.HOLD)
    }

    // Filter by status
    if (!filter.showActive) {
      filtered = filtered.filter((signal) => signal.status !== 'active')
    }
    if (!filter.showClosed) {
      filtered = filtered.filter((signal) => signal.status !== 'closed')
    }

    // Filter by model sources
    if (filter.modelSources && filter.modelSources.length > 0) {
      filtered = filtered.filter((signal) =>
        filter.modelSources!.includes(signal.modelSource)
      )
    }

    setFilteredSignals(filtered)
  }, [signals, filter])

  /**
   * Initial fetch and polling
   */
  useEffect(() => {
    fetchSignals()

    if (autoRefresh && !enableWebSocket) {
      const interval = setInterval(fetchSignals, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchSignals, autoRefresh, refreshInterval, enableWebSocket])

  /**
   * Cleanup
   */
  useEffect(() => {
    return () => {
      if (fetchAbortController.current) {
        fetchAbortController.current.abort()
      }
    }
  }, [])

  /**
   * Manual refresh
   */
  const refresh = useCallback(() => {
    setLoading(true)
    fetchSignals()
  }, [fetchSignals])

  /**
   * Update filter
   */
  const updateFilter = useCallback((newFilter: Partial<SignalFilterOptions>) => {
    // This is handled by passing the filter prop from parent
    // Could be extended to manage filter state internally if needed
  }, [])

  return {
    signals: filteredSignals,
    allSignals: signals,
    loading,
    error,
    lastUpdate,
    stats,
    refresh,
    updateFilter,
    isConnected: enableWebSocket ? readyState === ReadyState.OPEN : false,
    connectionStatus: enableWebSocket
      ? {
          [ReadyState.CONNECTING]: 'Connecting',
          [ReadyState.OPEN]: 'Connected',
          [ReadyState.CLOSING]: 'Closing',
          [ReadyState.CLOSED]: 'Closed',
          [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
        }[readyState]
      : 'Polling',
  }
}

export default useSignalOverlay
