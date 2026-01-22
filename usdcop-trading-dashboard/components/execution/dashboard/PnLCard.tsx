import { useQuery } from '@tanstack/react-query'
import { TrendingUp, TrendingDown, Activity } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { executionService } from '@/services/executionService'
import { formatCurrency, formatPercent, cn } from '@/lib/utils'

export function PnLCard() {
  const { data: todayStats, isLoading } = useQuery({
    queryKey: ['today-stats'],
    queryFn: executionService.getTodayStats,
    refetchInterval: 30000,
  })

  if (isLoading) {
    return (
      <Card variant="glass">
        <CardHeader>
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-10 w-32 mb-2" />
          <Skeleton className="h-4 w-24" />
        </CardContent>
      </Card>
    )
  }

  const pnl = todayStats?.pnl || 0
  const isPositive = pnl >= 0
  const pnlPercent = 0.018 // Mock 1.8%

  return (
    <Card variant={isPositive ? 'success' : 'danger'} hover>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity className="h-4 w-4 text-terminal-accent" />
          TODAY'S P&L
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <span
            className={cn(
              'text-3xl font-bold',
              isPositive ? 'text-market-up' : 'text-market-down'
            )}
          >
            {isPositive ? '+' : ''}{formatCurrency(pnl)}
          </span>
          <span
            className={cn(
              'flex items-center gap-1 text-sm font-medium',
              isPositive ? 'text-market-up' : 'text-market-down'
            )}
          >
            {isPositive ? (
              <TrendingUp className="h-4 w-4" />
            ) : (
              <TrendingDown className="h-4 w-4" />
            )}
            {formatPercent(pnlPercent)}
          </span>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-muted">Trades Today</span>
            <span className="text-text-secondary font-medium">
              {todayStats?.count || 0}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
