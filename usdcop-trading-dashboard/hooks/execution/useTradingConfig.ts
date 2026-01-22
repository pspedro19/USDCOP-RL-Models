import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { tradingService } from '@/services/tradingService'
import { toast } from '@/stores/uiStore'
import type { TradingConfigUpdate } from '@/contracts/trading'

export function useTradingConfig() {
  return useQuery({
    queryKey: ['trading-config'],
    queryFn: tradingService.getConfig,
  })
}

export function useTradingStatus() {
  return useQuery({
    queryKey: ['trading-status'],
    queryFn: tradingService.getStatus,
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}

export function useUpdateTradingConfig() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: TradingConfigUpdate) => tradingService.updateConfig(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trading-config'] })
      queryClient.invalidateQueries({ queryKey: ['trading-status'] })
      toast.success('Configuration saved!')
    },
    onError: () => {
      toast.error('Failed to save configuration')
    },
  })
}

export function useToggleTrading() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (enabled: boolean) => tradingService.toggleTrading(enabled),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['trading-config'] })
      queryClient.invalidateQueries({ queryKey: ['trading-status'] })
      toast.success(data.trading_enabled ? 'Trading enabled' : 'Trading disabled')
    },
    onError: () => {
      toast.error('Failed to toggle trading')
    },
  })
}
