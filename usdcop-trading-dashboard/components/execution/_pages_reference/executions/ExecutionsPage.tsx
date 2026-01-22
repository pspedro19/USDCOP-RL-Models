import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { TrendingUp } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { SkeletonTable } from '@/components/ui/skeleton'
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/table'
import { executionService } from '@/services/executionService'
import { formatDate, formatCurrency, cn } from '@/lib/utils'
import { EXCHANGE_NAMES } from '@/lib/constants'

export function ExecutionsPage() {
  const [filters, setFilters] = useState({
    exchange: '',
    status: '',
    page: 1,
  })

  const { data, isLoading } = useQuery({
    queryKey: ['executions', filters],
    queryFn: () =>
      executionService.getExecutions({
        page: filters.page,
        limit: 20,
        exchange: filters.exchange || undefined,
        status: filters.status || undefined,
      }),
  })

  const { data: stats } = useQuery({
    queryKey: ['execution-stats'],
    queryFn: () => executionService.getStats(7),
  })

  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'filled': return 'success'
      case 'rejected':
      case 'failed': return 'danger'
      case 'pending':
      case 'submitted': return 'pending'
      default: return 'secondary'
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Executions"
        description="View all trade executions and their results"
      />

      {/* Filters */}
      <Card variant="bordered">
        <CardContent className="p-4 flex flex-wrap gap-4 items-end">
          <Select
            value={filters.exchange}
            onChange={(value) => setFilters({ ...filters, exchange: value, page: 1 })}
            options={[
              { value: '', label: 'All Exchanges' },
              { value: 'mexc', label: 'MEXC' },
              { value: 'binance', label: 'Binance' },
            ]}
            label="Exchange"
            className="w-40"
          />
          <Select
            value={filters.status}
            onChange={(value) => setFilters({ ...filters, status: value, page: 1 })}
            options={[
              { value: '', label: 'All Status' },
              { value: 'filled', label: 'Filled' },
              { value: 'rejected', label: 'Rejected' },
              { value: 'pending', label: 'Pending' },
            ]}
            label="Status"
            className="w-40"
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFilters({ exchange: '', status: '', page: 1 })}
          >
            Clear
          </Button>
        </CardContent>
      </Card>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-text-primary">{stats.total}</p>
              <p className="text-xs text-text-muted">Total</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-market-up">{stats.filled}</p>
              <p className="text-xs text-text-muted">Filled</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-market-down">{stats.rejected}</p>
              <p className="text-xs text-text-muted">Rejected</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-status-delayed">
                ${stats.total_fees.toFixed(2)}
              </p>
              <p className="text-xs text-text-muted">Total Fees</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className={cn(
                'text-2xl font-bold',
                stats.total_pnl >= 0 ? 'text-market-up' : 'text-market-down'
              )}>
                {stats.total_pnl >= 0 ? '+' : ''}{formatCurrency(stats.total_pnl)}
              </p>
              <p className="text-xs text-text-muted">Total P&L</p>
            </CardContent>
          </Card>
          {stats.win_rate !== undefined && (
            <Card variant="glass">
              <CardContent className="p-4 text-center">
                <p className="text-2xl font-bold text-terminal-accent">
                  {(stats.win_rate * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-text-muted">Win Rate</p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Executions Table */}
      <Card variant="glass">
        <CardContent className="p-0">
          {isLoading ? (
            <SkeletonTable rows={10} />
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Exchange</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>P&L</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data?.data.map((exec) => (
                    <TableRow key={exec.request_id}>
                      <TableCell className="font-mono text-xs">
                        {formatDate(exec.created_at)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary" size="sm">
                          {EXCHANGE_NAMES[exec.exchange] || exec.exchange}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={exec.side === 'BUY' ? 'buy' : 'sell'}>
                          {exec.side}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {exec.filled_quantity > 0
                          ? `${exec.filled_quantity} USDT`
                          : `${exec.requested_quantity} USDT`}
                      </TableCell>
                      <TableCell>
                        {exec.filled_price > 0
                          ? formatCurrency(exec.filled_price)
                          : '-'}
                      </TableCell>
                      <TableCell>
                        <Badge variant={getStatusVariant(exec.status)} size="sm">
                          {exec.status.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {exec.pnl !== undefined ? (
                          <span className={cn(
                            'font-medium',
                            exec.pnl >= 0 ? 'text-market-up' : 'text-market-down'
                          )}>
                            {exec.pnl >= 0 ? '+' : ''}{formatCurrency(exec.pnl)}
                          </span>
                        ) : (
                          <span className="text-text-muted">-</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {data && data.pagination.total > 20 && (
                <div className="p-4 border-t border-slate-800 flex items-center justify-between">
                  <p className="text-sm text-text-muted">
                    Showing {((filters.page - 1) * 20) + 1}-{Math.min(filters.page * 20, data.pagination.total)} of {data.pagination.total}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={filters.page === 1}
                      onClick={() => setFilters({ ...filters, page: filters.page - 1 })}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={filters.page * 20 >= data.pagination.total}
                      onClick={() => setFilters({ ...filters, page: filters.page + 1 })}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
