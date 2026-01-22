import { useQuery } from '@tanstack/react-query'
import { executionService } from '@/services/executionService'

interface ExecutionQueryParams {
  page?: number
  limit?: number
  exchange?: string
  status?: string
  since?: string
}

export function useExecutions(params?: ExecutionQueryParams) {
  return useQuery({
    queryKey: ['executions', params],
    queryFn: () => executionService.getExecutions(params),
  })
}

export function useExecution(executionId: string) {
  return useQuery({
    queryKey: ['execution', executionId],
    queryFn: () => executionService.getExecution(executionId),
    enabled: !!executionId,
  })
}

export function useExecutionStats(days = 7) {
  return useQuery({
    queryKey: ['execution-stats', days],
    queryFn: () => executionService.getStats(days),
  })
}

export function useTodayStats() {
  return useQuery({
    queryKey: ['today-stats'],
    queryFn: executionService.getTodayStats,
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}
