import { useQuery } from '@tanstack/react-query'
import { Wallet, TrendingUp, TrendingDown } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { exchangeService } from '@/services/exchangeService'
import { formatCurrency, cn } from '@/lib/utils'
import { EXCHANGE_NAMES } from '@/lib/constants'

export function PortfolioPage() {
  const { data: balances, isLoading } = useQuery({
    queryKey: ['all-balances'],
    queryFn: exchangeService.getAllBalances,
  })

  const totalBalance = balances?.reduce((sum, b) => sum + b.total_usd, 0) || 0
  const copRate = 4250
  const totalCOP = totalBalance * copRate

  if (isLoading) {
    return (
      <div className="space-y-6">
        <PageHeader title="Portfolio" />
        <div className="grid gap-6">
          <Card variant="glass">
            <CardContent className="p-6">
              <Skeleton className="h-8 w-40 mb-2" />
              <Skeleton className="h-12 w-60" />
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Portfolio"
        description="View your total balance across all exchanges"
      />

      {/* Total Balance */}
      <Card variant="gradient">
        <CardContent className="p-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-terminal-accent to-terminal-purple flex items-center justify-center">
              <Wallet className="h-7 w-7 text-white" />
            </div>
            <div>
              <p className="text-sm text-text-muted">Total Portfolio Value</p>
              <p className="text-4xl font-bold text-text-primary">
                {formatCurrency(totalBalance)}
              </p>
              <p className="text-lg text-text-secondary">
                ≈ {new Intl.NumberFormat('es-CO', {
                  style: 'currency',
                  currency: 'COP',
                  maximumFractionDigits: 0,
                }).format(totalCOP)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Exchange Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {balances?.map((exchangeBalance) => (
          <Card key={exchangeBalance.exchange} variant="glass" hover>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>{EXCHANGE_NAMES[exchangeBalance.exchange]}</span>
                <Badge variant="live" dot>Connected</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-text-primary mb-4">
                {formatCurrency(exchangeBalance.total_usd)}
              </p>

              <div className="space-y-2">
                {exchangeBalance.balances.map((balance) => (
                  <div
                    key={balance.asset}
                    className="flex items-center justify-between py-2 border-b border-slate-800/50 last:border-0"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-text-primary">
                        {balance.asset}
                      </span>
                      {balance.locked > 0 && (
                        <Badge variant="warning" size="sm">
                          {balance.locked} locked
                        </Badge>
                      )}
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-text-primary">
                        {balance.total.toFixed(balance.asset === 'USDT' ? 2 : 6)}
                      </p>
                      <p className="text-xs text-text-muted">
                        Free: {balance.free.toFixed(balance.asset === 'USDT' ? 2 : 6)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {(!balances || balances.length === 0) && (
        <Card variant="bordered">
          <CardContent className="p-12 text-center">
            <div className="flex justify-center mb-4">
              <div className="h-16 w-16 rounded-2xl bg-slate-800 flex items-center justify-center">
                <Wallet className="h-8 w-8 text-text-muted" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              No Balances Available
            </h3>
            <p className="text-text-secondary">
              Connect an exchange to view your portfolio
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
