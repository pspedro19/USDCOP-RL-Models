import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ArrowRight, Radio } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { signalService } from '@/services/signalService'
import { formatRelativeTime, formatPercent, cn } from '@/lib/utils'
import { ACTION_ICONS } from '@/lib/constants'

export function RecentSignalsTable() {
  const { data: signals, isLoading } = useQuery({
    queryKey: ['recent-signals'],
    queryFn: () => signalService.getRecentSignals(5),
    refetchInterval: 30000,
  })

  if (isLoading) {
    return (
      <Card variant="glass">
        <CardHeader>
          <Skeleton className="h-5 w-40" />
        </CardHeader>
        <CardContent>
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex items-center gap-4 py-3">
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-6 w-16" />
              <Skeleton className="h-4 w-12" />
              <Skeleton className="h-4 w-20" />
            </div>
          ))}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card variant="glass">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Radio className="h-4 w-4 text-terminal-accent" />
          Recent Signals
        </CardTitle>
        <Link to="/signals">
          <Button variant="ghost" size="sm" rightIcon={<ArrowRight className="h-4 w-4" />}>
            View All
          </Button>
        </Link>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="px-6 py-3 text-left text-xs font-medium text-text-tertiary uppercase">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-text-tertiary uppercase">
                  Action
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-text-tertiary uppercase">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-text-tertiary uppercase">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {signals?.map((signal) => (
                <tr
                  key={signal.signal_id}
                  className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                >
                  <td className="px-6 py-4 text-sm text-text-muted">
                    {formatRelativeTime(signal.timestamp)}
                  </td>
                  <td className="px-6 py-4">
                    <Badge
                      variant={
                        signal.action === 2 ? 'buy' :
                        signal.action === 0 ? 'sell' : 'hold'
                      }
                    >
                      {ACTION_ICONS[signal.action]} {signal.action_name}
                    </Badge>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={cn(
                        'text-sm font-medium',
                        signal.confidence >= 0.8 ? 'text-market-up' :
                        signal.confidence >= 0.6 ? 'text-status-delayed' :
                        'text-text-muted'
                      )}
                    >
                      {formatPercent(signal.confidence, 0)}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <Badge
                      variant={signal.execution_count > 0 ? 'success' : 'secondary'}
                      size="sm"
                    >
                      {signal.execution_count > 0 ? 'Executed' : 'Skipped'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {(!signals || signals.length === 0) && (
          <div className="p-8 text-center text-text-muted">
            No signals received yet
          </div>
        )}
      </CardContent>
    </Card>
  )
}
