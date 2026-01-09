/**
 * Database Statistics Hook
 * ========================
 *
 * Hook to fetch real-time database statistics from Trading API
 * NO HARDCODED VALUES - All data comes from /api/health endpoint
 */

import { useState, useEffect, useCallback } from 'react'

export interface DbStats {
  totalRecords: number
  latestData: string
  isConnected: boolean
  lastUpdated: Date | null
}

interface UseDbStatsReturn {
  stats: DbStats
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
}

const DEFAULT_STATS: DbStats = {
  totalRecords: 0,
  latestData: '',
  isConnected: false,
  lastUpdated: null
}

export function useDbStats(refreshInterval: number = 60000): UseDbStatsReturn {
  const [stats, setStats] = useState<DbStats>(DEFAULT_STATS)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/proxy/trading/health')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      setStats({
        totalRecords: data.total_records || 0,
        latestData: data.latest_data || '',
        isConnected: data.database === 'connected',
        lastUpdated: new Date()
      })

      setError(null)
      console.log('[useDbStats] Database stats updated:', {
        totalRecords: data.total_records,
        latestData: data.latest_data
      })

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch DB stats'
      console.error('[useDbStats] Error:', err)

      setError(errorMessage)

      // Keep previous stats if available
      if (stats.totalRecords === 0) {
        setStats(DEFAULT_STATS)
      }
    } finally {
      setIsLoading(false)
    }
  }, [stats.totalRecords])

  const refresh = useCallback(async () => {
    setIsLoading(true)
    await fetchStats()
  }, [fetchStats])

  useEffect(() => {
    // Initial fetch
    fetchStats()

    // Set up periodic refresh
    const intervalId = setInterval(() => {
      fetchStats()
    }, refreshInterval)

    return () => {
      clearInterval(intervalId)
    }
  }, [fetchStats, refreshInterval])

  return {
    stats,
    isLoading,
    error,
    refresh
  }
}
