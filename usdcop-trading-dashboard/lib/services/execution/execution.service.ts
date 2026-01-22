import { api } from './api'
import type { Execution, ExecutionDetail, ExecutionListResponse, ExecutionStats } from '@/contracts/execution'
import { sleep } from '@/lib/utils'

const MOCK_MODE = true

const mockExecutions: Execution[] = [
  { request_id: 'exec-001', signal_id: 'sig-001', exchange: 'mexc', symbol: 'USDTCOP', side: 'BUY', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4250.50, fees: 0.10, fees_currency: 'USDT', created_at: '2026-01-18T14:30:01Z', executed_at: '2026-01-18T14:30:01Z' },
  { request_id: 'exec-002', signal_id: 'sig-001', exchange: 'binance', symbol: 'USDTCOP', side: 'BUY', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4250.80, fees: 0.10, fees_currency: 'USDT', created_at: '2026-01-18T14:30:02Z', executed_at: '2026-01-18T14:30:02Z' },
  { request_id: 'exec-003', signal_id: 'sig-002', exchange: 'mexc', symbol: 'USDTCOP', side: 'SELL', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4280.20, fees: 0.10, fees_currency: 'USDT', pnl: 7.20, created_at: '2026-01-18T12:15:01Z', executed_at: '2026-01-18T12:15:01Z' },
  { request_id: 'exec-004', signal_id: 'sig-002', exchange: 'binance', symbol: 'USDTCOP', side: 'SELL', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4280.00, fees: 0.10, fees_currency: 'USDT', pnl: 7.00, created_at: '2026-01-18T12:15:02Z', executed_at: '2026-01-18T12:15:02Z' },
  { request_id: 'exec-005', signal_id: 'sig-004', exchange: 'mexc', symbol: 'USDTCOP', side: 'BUY', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4200.00, fees: 0.10, fees_currency: 'USDT', created_at: '2026-01-18T08:30:01Z', executed_at: '2026-01-18T08:30:01Z' },
  { request_id: 'exec-006', signal_id: 'sig-004', exchange: 'binance', symbol: 'USDTCOP', side: 'BUY', status: 'rejected', requested_quantity: 50, filled_quantity: 0, filled_price: 0, fees: 0, fees_currency: 'USDT', error_message: 'Insufficient balance', created_at: '2026-01-18T08:30:02Z' },
  { request_id: 'exec-007', signal_id: 'sig-005', exchange: 'mexc', symbol: 'USDTCOP', side: 'SELL', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4310.50, fees: 0.10, fees_currency: 'USDT', pnl: 12.50, created_at: '2026-01-17T16:45:01Z', executed_at: '2026-01-17T16:45:01Z' },
  { request_id: 'exec-008', signal_id: 'sig-005', exchange: 'binance', symbol: 'USDTCOP', side: 'SELL', status: 'filled', requested_quantity: 100, filled_quantity: 100, filled_price: 4310.20, fees: 0.10, fees_currency: 'USDT', pnl: 12.20, created_at: '2026-01-17T16:45:02Z', executed_at: '2026-01-17T16:45:02Z' },
]

export const executionService = {
  async getExecutions(params?: {
    limit?: number
    page?: number
    exchange?: string
    status?: string
    since?: string
  }): Promise<ExecutionListResponse> {
    if (MOCK_MODE) {
      await sleep(300)
      let filtered = [...mockExecutions]

      if (params?.exchange) {
        filtered = filtered.filter(e => e.exchange === params.exchange)
      }

      if (params?.status) {
        filtered = filtered.filter(e => e.status === params.status)
      }

      if (params?.since) {
        const sinceDate = new Date(params.since)
        filtered = filtered.filter(e => new Date(e.created_at) >= sinceDate)
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

    const response = await api.get('/executions', { params })
    return response.data
  },

  async getExecution(executionId: string): Promise<ExecutionDetail> {
    if (MOCK_MODE) {
      await sleep(200)
      const exec = mockExecutions.find(e => e.request_id === executionId)
      if (!exec) throw new Error('Execution not found')
      return {
        ...exec,
        signal: {
          action: 2,
          action_name: 'BUY',
          confidence: 0.85,
        },
        order_id: 'ord-' + executionId,
        avg_price: exec.filled_price,
      }
    }

    const response = await api.get(`/executions/${executionId}`)
    return response.data
  },

  async getStats(days = 7): Promise<ExecutionStats> {
    if (MOCK_MODE) {
      await sleep(200)
      return {
        total: 89,
        filled: 85,
        rejected: 4,
        failed: 0,
        total_fees: 8.90,
        total_pnl: 145.20,
        win_rate: 0.68,
      }
    }

    const response = await api.get('/executions/stats', { params: { days } })
    return response.data
  },

  async getTodayStats(): Promise<{ count: number; pnl: number }> {
    if (MOCK_MODE) {
      await sleep(150)
      return {
        count: 7,
        pnl: 45.20,
      }
    }

    const response = await api.get('/executions/today')
    return response.data
  },
}
