import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Plus, Link2 } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { SkeletonCard } from '@/components/ui/skeleton'
import { ExchangeCard } from '@/components/exchanges/ExchangeCard'
import { exchangeService } from '@/services/exchangeService'
import { SUPPORTED_EXCHANGES, EXCHANGE_NAMES } from '@/lib/constants'

export function ExchangesPage() {
  const { data: exchanges, isLoading } = useQuery({
    queryKey: ['exchanges'],
    queryFn: exchangeService.getExchanges,
  })

  const connectedExchanges = new Set(exchanges?.map(e => e.exchange) || [])
  const availableExchanges = SUPPORTED_EXCHANGES.filter(e => !connectedExchanges.has(e))

  return (
    <div className="space-y-6">
      <PageHeader
        title="Exchanges"
        description="Connect and manage your exchange accounts"
        actions={
          availableExchanges.length > 0 && (
            <Link to={`/exchanges/connect/${availableExchanges[0]}`}>
              <Button variant="primary" leftIcon={<Plus className="h-4 w-4" />}>
                Connect Exchange
              </Button>
            </Link>
          )
        }
      />

      {/* Connected Exchanges */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-text-primary">Connected Exchanges</h2>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : exchanges && exchanges.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {exchanges.map((exchange) => (
              <ExchangeCard key={exchange.id} exchange={exchange} />
            ))}
          </div>
        ) : (
          <Card variant="bordered">
            <CardContent className="p-12 text-center">
              <div className="flex justify-center mb-4">
                <div className="h-16 w-16 rounded-2xl bg-slate-800 flex items-center justify-center">
                  <Link2 className="h-8 w-8 text-text-muted" />
                </div>
              </div>
              <h3 className="text-lg font-semibold text-text-primary mb-2">
                No Exchanges Connected
              </h3>
              <p className="text-text-secondary mb-6">
                Connect your first exchange to start receiving trading signals
              </p>
              <Link to={`/exchanges/connect/${SUPPORTED_EXCHANGES[0]}`}>
                <Button variant="primary" leftIcon={<Plus className="h-4 w-4" />}>
                  Connect Exchange
                </Button>
              </Link>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Available Exchanges */}
      {availableExchanges.length > 0 && exchanges && exchanges.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-text-primary">Available Exchanges</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {availableExchanges.map((exchange) => (
              <Card key={exchange} variant="bordered" hover>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="h-12 w-12 rounded-xl bg-slate-800 flex items-center justify-center">
                        <span className="text-2xl">🔗</span>
                      </div>
                      <div>
                        <h3 className="font-semibold text-text-primary">
                          {EXCHANGE_NAMES[exchange]}
                        </h3>
                        <p className="text-sm text-text-muted">Not connected</p>
                      </div>
                    </div>
                    <Link to={`/exchanges/connect/${exchange}`}>
                      <Button variant="outline" size="sm">
                        Connect
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Coming Soon */}
      <Card variant="bordered">
        <CardContent className="p-6">
          <h3 className="font-semibold text-text-primary mb-2">Coming Soon</h3>
          <p className="text-text-secondary text-sm">
            Support for Bybit, KuCoin, and more exchanges is on the way.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
