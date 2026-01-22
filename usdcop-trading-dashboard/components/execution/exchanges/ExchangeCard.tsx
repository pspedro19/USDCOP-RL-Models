import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { CheckCircle, XCircle, RefreshCw, Settings, Trash2 } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Modal } from '@/components/ui/modal'
import { exchangeService } from '@/services/exchangeService'
import { toast } from '@/stores/uiStore'
import { formatRelativeTime, formatCurrency } from '@/lib/utils'
import { EXCHANGE_NAMES } from '@/lib/constants'
import type { ConnectedExchange } from '@/contracts/exchange'

interface ExchangeCardProps {
  exchange: ConnectedExchange
}

export function ExchangeCard({ exchange }: ExchangeCardProps) {
  const queryClient = useQueryClient()
  const [showDisconnect, setShowDisconnect] = useState(false)

  const testMutation = useMutation({
    mutationFn: () => exchangeService.testConnection(exchange.exchange),
    onSuccess: (result) => {
      if (result.is_valid) {
        toast.success(`${EXCHANGE_NAMES[exchange.exchange]} connection is valid`)
        queryClient.invalidateQueries({ queryKey: ['exchanges'] })
      } else {
        toast.error(result.error_message || 'Connection test failed')
      }
    },
    onError: () => {
      toast.error('Failed to test connection')
    },
  })

  const disconnectMutation = useMutation({
    mutationFn: () => exchangeService.disconnectExchange(exchange.exchange),
    onSuccess: () => {
      toast.success(`${EXCHANGE_NAMES[exchange.exchange]} disconnected`)
      queryClient.invalidateQueries({ queryKey: ['exchanges'] })
      setShowDisconnect(false)
    },
    onError: () => {
      toast.error('Failed to disconnect')
    },
  })

  return (
    <>
      <Card variant={exchange.is_valid ? 'success' : 'danger'}>
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-xl bg-slate-800 flex items-center justify-center">
                <span className="text-2xl">
                  {exchange.exchange === 'mexc' ? '🟡' : '🟡'}
                </span>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text-primary">
                  {EXCHANGE_NAMES[exchange.exchange]}
                </h3>
                <Badge variant={exchange.is_valid ? 'live' : 'offline'} dot size="sm">
                  {exchange.is_valid ? 'Connected' : 'Invalid'}
                </Badge>
              </div>
            </div>
          </div>

          <div className="space-y-3 mb-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">Status</span>
              <span className="flex items-center gap-1 text-text-secondary">
                {exchange.is_valid ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-emerald-400" />
                    Valid
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-red-400" />
                    Invalid
                  </>
                )}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">Last Used</span>
              <span className="text-text-secondary">
                {exchange.last_used_at
                  ? formatRelativeTime(exchange.last_used_at)
                  : 'Never'}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-muted">API Key</span>
              <span className="text-text-secondary font-mono text-xs">
                {exchange.key_fingerprint}
              </span>
            </div>
          </div>

          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={() => testMutation.mutate()}
              isLoading={testMutation.isPending}
              leftIcon={<RefreshCw className="h-4 w-4" />}
            >
              Test
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setShowDisconnect(true)}
              leftIcon={<Trash2 className="h-4 w-4" />}
            >
              Disconnect
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Disconnect Confirmation Modal */}
      <Modal
        open={showDisconnect}
        onClose={() => setShowDisconnect(false)}
        title="Disconnect Exchange"
        description={`Are you sure you want to disconnect ${EXCHANGE_NAMES[exchange.exchange]}?`}
      >
        <div className="space-y-4">
          <p className="text-text-secondary">
            This will remove your API credentials. You'll need to reconnect to trade on this exchange.
          </p>
          <div className="flex gap-3 justify-end">
            <Button variant="outline" onClick={() => setShowDisconnect(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => disconnectMutation.mutate()}
              isLoading={disconnectMutation.isPending}
            >
              Disconnect
            </Button>
          </div>
        </div>
      </Modal>
    </>
  )
}
