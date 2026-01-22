import { useQuery } from '@tanstack/react-query'
import { signalService } from '@/services/signalService'

interface SignalQueryParams {
  page?: number
  limit?: number
  action?: number
  since?: string
}

export function useSignals(params?: SignalQueryParams) {
  return useQuery({
    queryKey: ['signals', params],
    queryFn: () => signalService.getSignals(params),
  })
}

export function useSignal(signalId: string) {
  return useQuery({
    queryKey: ['signal', signalId],
    queryFn: () => signalService.getSignal(signalId),
    enabled: !!signalId,
  })
}

export function useSignalStats(days = 7) {
  return useQuery({
    queryKey: ['signal-stats', days],
    queryFn: () => signalService.getStats(days),
  })
}

export function useRecentSignals(limit = 5) {
  return useQuery({
    queryKey: ['recent-signals', limit],
    queryFn: () => signalService.getRecentSignals(limit),
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}
