import { api } from './api'
import type { Signal, SignalDetail, SignalListResponse, SignalStats } from '@/contracts/signal'
import { sleep } from '@/lib/utils'

const MOCK_MODE = true

const mockSignals: Signal[] = [
  { signal_id: 'sig-001', timestamp: '2026-01-18T14:30:00Z', symbol: 'USD/COP', action: 2, action_name: 'BUY', confidence: 0.85, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-002', timestamp: '2026-01-18T12:15:00Z', symbol: 'USD/COP', action: 0, action_name: 'SELL', confidence: 0.78, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-003', timestamp: '2026-01-18T10:00:00Z', symbol: 'USD/COP', action: 1, action_name: 'HOLD', confidence: 0.65, model_version: 'v21', execution_count: 0 },
  { signal_id: 'sig-004', timestamp: '2026-01-18T08:30:00Z', symbol: 'USD/COP', action: 2, action_name: 'BUY', confidence: 0.92, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-005', timestamp: '2026-01-17T16:45:00Z', symbol: 'USD/COP', action: 0, action_name: 'SELL', confidence: 0.71, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-006', timestamp: '2026-01-17T14:00:00Z', symbol: 'USD/COP', action: 1, action_name: 'HOLD', confidence: 0.55, model_version: 'v21', execution_count: 0 },
  { signal_id: 'sig-007', timestamp: '2026-01-17T11:30:00Z', symbol: 'USD/COP', action: 2, action_name: 'BUY', confidence: 0.88, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-008', timestamp: '2026-01-17T09:15:00Z', symbol: 'USD/COP', action: 0, action_name: 'SELL', confidence: 0.82, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-009', timestamp: '2026-01-16T15:00:00Z', symbol: 'USD/COP', action: 2, action_name: 'BUY', confidence: 0.75, model_version: 'v21', execution_count: 2 },
  { signal_id: 'sig-010', timestamp: '2026-01-16T12:30:00Z', symbol: 'USD/COP', action: 1, action_name: 'HOLD', confidence: 0.60, model_version: 'v21', execution_count: 0 },
]

export const signalService = {
  async getSignals(params?: {
    limit?: number
    page?: number
    action?: number
    since?: string
  }): Promise<SignalListResponse> {
    if (MOCK_MODE) {
      await sleep(300)
      let filtered = [...mockSignals]

      if (params?.action !== undefined) {
        filtered = filtered.filter(s => s.action === params.action)
      }

      if (params?.since) {
        const sinceDate = new Date(params.since)
        filtered = filtered.filter(s => new Date(s.timestamp) >= sinceDate)
      }

      const page = params?.page || 1
      const limit = params?.limit || 20
      const start = (page - 1) * limit
      const paginated = filtered.slice(start, start + limit)

      return {
        data: paginated,
        pagination: {
          page,
          limit,
          total: filtered.length,
        },
      }
    }

    const response = await api.get('/signals', { params })
    return response.data
  },

  async getSignal(signalId: string): Promise<SignalDetail> {
    if (MOCK_MODE) {
      await sleep(200)
      const signal = mockSignals.find(s => s.signal_id === signalId)
      if (!signal) throw new Error('Signal not found')
      return {
        ...signal,
        features: {
          rsi: 45.5,
          macd: 0.0012,
          volume_change: 1.25,
        },
        raw_prediction: [0.1, 0.15, 0.75],
        processing_time_ms: 45,
      }
    }

    const response = await api.get(`/signals/${signalId}`)
    return response.data
  },

  async getStats(days = 7): Promise<SignalStats> {
    if (MOCK_MODE) {
      await sleep(200)
      return {
        total: 45,
        buy_count: 18,
        sell_count: 15,
        hold_count: 12,
        executed_count: 38,
        skipped_count: 7,
        avg_confidence: 0.76,
      }
    }

    const response = await api.get('/signals/stats', { params: { days } })
    return response.data
  },

  async getRecentSignals(limit = 5): Promise<Signal[]> {
    if (MOCK_MODE) {
      await sleep(150)
      return mockSignals.slice(0, limit)
    }

    const response = await api.get('/signals/recent', { params: { limit } })
    return response.data
  },
}
