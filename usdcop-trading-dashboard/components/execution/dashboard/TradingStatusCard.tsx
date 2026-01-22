import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Power, Clock } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { tradingService } from '@/services/tradingService'
import { toast } from '@/stores/uiStore'
import { formatRelativeTime } from '@/lib/utils'

export function TradingStatusCard() {
  const queryClient = useQueryClient()

  const { data: status, isLoading } = useQuery({
    queryKey: ['trading-status'],
    queryFn: tradingService.getStatus,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const toggleMutation = useMutation({
    mutationFn: (enabled: boolean) => tradingService.toggleTrading(enabled),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['trading-status'] })
      queryClient.invalidateQueries({ queryKey: ['trading-config'] })
      toast.success(data.trading_enabled ? 'Trading enabled' : 'Trading disabled')
    },
    onError: () => {
      toast.error('Failed to toggle trading')
    },
  })

  if (isLoading) {
    return (
      <Card variant="glass">
        <CardContent className="p-6">
          <Skeleton className="h-4 w-32 mb-4" />
          <Skeleton className="h-8 w-24 mb-2" />
          <Skeleton className="h-4 w-48" />
        </CardContent>
      </Card>
    )
  }

  const isActive = status?.is_enabled

  return (
    <Card variant={isActive ? 'success' : 'bordered'}>
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-text-secondary mb-2">
              TRADING STATUS
            </p>
            <div className="flex items-center gap-3">
              <Badge
                variant={isActive ? 'live' : 'offline'}
                dot
                pulse={isActive}
                size="lg"
              >
                {isActive ? 'ACTIVE' : 'INACTIVE'}
              </Badge>
            </div>
            {status?.last_signal_at && (
              <div className="flex items-center gap-2 mt-3 text-sm text-text-muted">
                <Clock className="h-4 w-4" />
                <span>Last signal: {formatRelativeTime(status.last_signal_at)}</span>
              </div>
            )}
          </div>
          <Button
            variant={isActive ? 'destructive' : 'success'}
            size="sm"
            onClick={() => toggleMutation.mutate(!isActive)}
            isLoading={toggleMutation.isPending}
            leftIcon={<Power className="h-4 w-4" />}
          >
            {isActive ? 'Disable' : 'Enable'}
          </Button>
        </div>
        {isActive && (
          <p className="mt-4 text-xs text-amber-400/80 bg-amber-500/10 rounded-lg p-2">
            Trades will execute automatically on incoming signals
          </p>
        )}
      </CardContent>
    </Card>
  )
}
