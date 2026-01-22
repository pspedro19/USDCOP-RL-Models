import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Radio, Filter } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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
import { signalService } from '@/services/signalService'
import { formatDate, formatPercent, cn } from '@/lib/utils'
import { ACTION_ICONS } from '@/lib/constants'

export function SignalsPage() {
  const [filters, setFilters] = useState({
    action: '',
    page: 1,
  })

  const { data, isLoading } = useQuery({
    queryKey: ['signals', filters],
    queryFn: () =>
      signalService.getSignals({
        page: filters.page,
        limit: 20,
        action: filters.action ? parseInt(filters.action) : undefined,
      }),
  })

  const { data: stats } = useQuery({
    queryKey: ['signal-stats'],
    queryFn: () => signalService.getStats(7),
  })

  return (
    <div className="space-y-6">
      <PageHeader
        title="Signals"
        description="View all trading signals received from the ML model"
      />

      {/* Filters */}
      <Card variant="bordered">
        <CardContent className="p-4 flex flex-wrap gap-4 items-end">
          <Select
            value={filters.action}
            onChange={(value) => setFilters({ ...filters, action: value, page: 1 })}
            options={[
              { value: '', label: 'All Actions' },
              { value: '2', label: 'BUY' },
              { value: '0', label: 'SELL' },
              { value: '1', label: 'HOLD' },
            ]}
            label="Action"
            className="w-40"
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFilters({ action: '', page: 1 })}
          >
            Clear Filters
          </Button>
        </CardContent>
      </Card>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-text-primary">{stats.total}</p>
              <p className="text-xs text-text-muted">Total</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-market-up">{stats.buy_count}</p>
              <p className="text-xs text-text-muted">BUY</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-market-down">{stats.sell_count}</p>
              <p className="text-xs text-text-muted">SELL</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-text-secondary">{stats.hold_count}</p>
              <p className="text-xs text-text-muted">HOLD</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-terminal-accent">{stats.executed_count}</p>
              <p className="text-xs text-text-muted">Executed</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-text-muted">{stats.skipped_count}</p>
              <p className="text-xs text-text-muted">Skipped</p>
            </CardContent>
          </Card>
          <Card variant="glass">
            <CardContent className="p-4 text-center">
              <p className="text-2xl font-bold text-terminal-accent">
                {formatPercent(stats.avg_confidence, 0)}
              </p>
              <p className="text-xs text-text-muted">Avg Conf</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Signals Table */}
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
                    <TableHead>Symbol</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Confidence</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data?.data.map((signal) => (
                    <TableRow key={signal.signal_id}>
                      <TableCell className="font-mono text-xs">
                        {formatDate(signal.timestamp)}
                      </TableCell>
                      <TableCell>{signal.symbol}</TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            signal.action === 2 ? 'buy' :
                            signal.action === 0 ? 'sell' : 'hold'
                          }
                        >
                          {ACTION_ICONS[signal.action]} {signal.action_name}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <span
                          className={cn(
                            'font-medium',
                            signal.confidence >= 0.8 ? 'text-market-up' :
                            signal.confidence >= 0.6 ? 'text-status-delayed' :
                            'text-text-muted'
                          )}
                        >
                          {formatPercent(signal.confidence, 0)}
                        </span>
                      </TableCell>
                      <TableCell className="text-text-muted">
                        {signal.model_version}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={signal.execution_count > 0 ? 'success' : 'secondary'}
                          size="sm"
                        >
                          {signal.execution_count > 0
                            ? `Executed (${signal.execution_count})`
                            : 'Skipped'}
                        </Badge>
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
