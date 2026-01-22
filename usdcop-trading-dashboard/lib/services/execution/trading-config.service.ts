import { api } from './api'
import type { TradingConfig, TradingConfigUpdate, TradingStatus } from '@/contracts/trading'
import { sleep } from '@/lib/utils'

const MOCK_MODE = true

let mockConfig: TradingConfig = {
  symbol: 'USD/COP',
  is_enabled: true,
  stop_loss_pct: 0.02,
  take_profit_pct: 0.05,
  min_confidence: 0.70,
  execute_on_exchanges: ['mexc', 'binance'],
}

const mockStatus: TradingStatus = {
  is_enabled: true,
  last_signal_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 min ago
  today_trades_count: 7,
  max_daily_trades: 20,
  current_position: 'long',
}

export const tradingService = {
  async getConfig(): Promise<TradingConfig> {
    if (MOCK_MODE) {
      await sleep(200)
      return { ...mockConfig }
    }

    const response = await api.get('/trading/config')
    return response.data
  },

  async updateConfig(data: TradingConfigUpdate): Promise<TradingConfig> {
    if (MOCK_MODE) {
      await sleep(300)
      mockConfig = { ...mockConfig, ...data }
      return { ...mockConfig }
    }

    const response = await api.put('/trading/config', data)
    return response.data
  },

  async toggleTrading(enabled: boolean): Promise<{ trading_enabled: boolean }> {
    if (MOCK_MODE) {
      await sleep(200)
      mockConfig.is_enabled = enabled
      return { trading_enabled: enabled }
    }

    const response = await api.post('/trading/toggle', { enabled })
    return response.data
  },

  async getStatus(): Promise<TradingStatus> {
    if (MOCK_MODE) {
      await sleep(150)
      return {
        ...mockStatus,
        is_enabled: mockConfig.is_enabled,
      }
    }

    const response = await api.get('/trading/status')
    return response.data
  },
}
