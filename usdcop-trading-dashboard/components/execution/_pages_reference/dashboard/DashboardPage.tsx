import { useAuthStore } from '@/stores/authStore'
import { PageHeader } from '@/components/layout/PageHeader'
import {
  TradingStatusCard,
  ExchangesCard,
  BalanceCard,
  PnLCard,
  DailyTradesCard,
  RecentSignalsTable,
} from '@/components/dashboard'

export function DashboardPage() {
  const { user } = useAuthStore()
  const username = user?.email?.split('@')[0] || 'User'

  return (
    <div className="space-y-6">
      <PageHeader
        title={`Welcome back, ${username}`}
        description="Monitor your trading activity and performance"
      />

      {/* Trading Status - Full Width */}
      <TradingStatusCard />

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <ExchangesCard />
        <DailyTradesCard />
        <BalanceCard />
        <PnLCard />
      </div>

      {/* Recent Signals Table */}
      <RecentSignalsTable />
    </div>
  )
}
