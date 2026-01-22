import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Link2, Plus, CheckCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { exchangeService } from '@/services/exchangeService'
import { EXCHANGE_NAMES } from '@/lib/constants'

export function ExchangesCard() {
  const { data: exchanges, isLoading } = useQuery({
    queryKey: ['exchanges'],
    queryFn: exchangeService.getExchanges,
  })

  if (isLoading) {
    return (
      <Card variant="glass">
        <CardHeader>
          <Skeleton className="h-4 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-20 mb-2" />
          <Skeleton className="h-4 w-40" />
        </CardContent>
      </Card>
    )
  }

  const connectedCount = exchanges?.filter(e => e.is_valid).length || 0

  return (
    <Card variant="glass" hover>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Link2 className="h-4 w-4 text-terminal-accent" />
          CONNECTED EXCHANGES
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2 mb-3">
          <span className="text-3xl font-bold text-text-primary">
            {connectedCount}
          </span>
          <span className="text-text-muted">Exchange{connectedCount !== 1 ? 's' : ''}</span>
        </div>

        {exchanges && exchanges.length > 0 ? (
          <div className="flex flex-wrap gap-2 mb-4">
            {exchanges.map((exchange) => (
              <div
                key={exchange.id}
                className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-800/50 text-sm"
              >
                <CheckCircle className="h-3 w-3 text-emerald-400" />
                <span className="text-text-secondary">
                  {EXCHANGE_NAMES[exchange.exchange] || exchange.exchange}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-text-muted mb-4">
            No exchanges connected yet
          </p>
        )}

        <Link to="/exchanges">
          <Button variant="outline" size="sm" className="w-full" leftIcon={<Plus className="h-4 w-4" />}>
            {connectedCount > 0 ? 'Manage Exchanges' : 'Connect Exchange'}
          </Button>
        </Link>
      </CardContent>
    </Card>
  )
}
