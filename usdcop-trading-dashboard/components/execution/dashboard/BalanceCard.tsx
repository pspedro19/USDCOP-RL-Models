import { useQuery } from '@tanstack/react-query'
import { Wallet, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { exchangeService } from '@/services/exchangeService'
import { formatCurrency } from '@/lib/utils'

export function BalanceCard() {
  const { data: balances, isLoading } = useQuery({
    queryKey: ['all-balances'],
    queryFn: exchangeService.getAllBalances,
    refetchInterval: 60000, // Refresh every minute
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

  const totalBalance = balances?.reduce((sum, b) => sum + b.total_usd, 0) || 0
  const copRate = 4250 // Mock rate
  const totalCOP = totalBalance * copRate

  return (
    <Card variant="gradient" hover>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Wallet className="h-4 w-4 text-terminal-accent" />
          TOTAL BALANCE
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          <p className="text-3xl font-bold text-text-primary">
            {formatCurrency(totalBalance)}
          </p>
          <p className="text-sm text-text-muted">
            ≈ {new Intl.NumberFormat('es-CO', {
              style: 'currency',
              currency: 'COP',
              maximumFractionDigits: 0,
            }).format(totalCOP)}
          </p>
        </div>

        {balances && balances.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-2">
            {balances.map((b) => (
              <div key={b.exchange} className="flex items-center justify-between text-sm">
                <span className="text-text-muted capitalize">{b.exchange}</span>
                <span className="text-text-secondary font-medium">
                  {formatCurrency(b.total_usd)}
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
