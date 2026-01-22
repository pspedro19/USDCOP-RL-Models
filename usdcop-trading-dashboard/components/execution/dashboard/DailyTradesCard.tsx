import { useQuery } from '@tanstack/react-query'
import { BarChart3 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { tradingService } from '@/services/tradingService'
import { cn } from '@/lib/utils'

export function DailyTradesCard() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['trading-status'],
    queryFn: tradingService.getStatus,
  })

  if (isLoading) {
    return (
      <Card variant="glass">
        <CardHeader>
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-10 w-32 mb-2" />
          <Skeleton className="h-2 w-full" />
        </CardContent>
      </Card>
    )
  }

  const current = status?.today_trades_count || 0
  const max = status?.max_daily_trades || 20
  const percentage = (current / max) * 100
  const isNearLimit = percentage >= 80

  return (
    <Card variant="glass" hover>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <BarChart3 className="h-4 w-4 text-terminal-accent" />
          TODAY'S TRADES
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2 mb-3">
          <span className="text-3xl font-bold text-text-primary">{current}</span>
          <span className="text-text-muted">/ {max} limit</span>
        </div>

        {/* Progress bar */}
        <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500',
              isNearLimit
                ? 'bg-gradient-to-r from-amber-500 to-red-500'
                : 'bg-gradient-to-r from-terminal-accent to-terminal-emerald'
            )}
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>

        {isNearLimit && (
          <p className="mt-3 text-xs text-amber-400">
            Approaching daily trade limit
          </p>
        )}
      </CardContent>
    </Card>
  )
}
