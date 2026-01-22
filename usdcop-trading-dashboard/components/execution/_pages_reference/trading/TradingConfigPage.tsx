import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Save, AlertTriangle } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Toggle } from '@/components/ui/toggle'
import { Slider } from '@/components/ui/slider'
import { Select } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { tradingService } from '@/services/tradingService'
import { exchangeService } from '@/services/exchangeService'
import { toast } from '@/stores/uiStore'
import { formatPercent } from '@/lib/utils'
import { EXCHANGE_NAMES } from '@/lib/constants'
import type { TradingConfig } from '@/contracts/trading'

export function TradingConfigPage() {
  const queryClient = useQueryClient()
  const [config, setConfig] = useState<TradingConfig | null>(null)
  const [hasChanges, setHasChanges] = useState(false)

  const { data: serverConfig, isLoading: configLoading } = useQuery({
    queryKey: ['trading-config'],
    queryFn: tradingService.getConfig,
  })

  const { data: exchanges } = useQuery({
    queryKey: ['exchanges'],
    queryFn: exchangeService.getExchanges,
  })

  useEffect(() => {
    if (serverConfig) {
      setConfig(serverConfig)
    }
  }, [serverConfig])

  const updateMutation = useMutation({
    mutationFn: (data: Partial<TradingConfig>) => tradingService.updateConfig(data),
    onSuccess: () => {
      toast.success('Configuration saved!')
      queryClient.invalidateQueries({ queryKey: ['trading-config'] })
      setHasChanges(false)
    },
    onError: () => {
      toast.error('Failed to save configuration')
    },
  })

  const handleChange = <K extends keyof TradingConfig>(
    key: K,
    value: TradingConfig[K]
  ) => {
    setConfig((prev) => prev ? { ...prev, [key]: value } : null)
    setHasChanges(true)
  }

  const handleExchangeToggle = (exchange: string, enabled: boolean) => {
    if (!config) return
    const newExchanges = enabled
      ? [...config.execute_on_exchanges, exchange as 'mexc' | 'binance']
      : config.execute_on_exchanges.filter((e) => e !== exchange)
    handleChange('execute_on_exchanges', newExchanges)
  }

  const handleSave = () => {
    if (!config) return
    updateMutation.mutate(config)
  }

  if (configLoading || !config) {
    return (
      <div className="space-y-6">
        <PageHeader title="Trading Configuration" />
        <div className="grid gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} variant="glass">
              <CardContent className="p-6">
                <Skeleton className="h-6 w-40 mb-4" />
                <Skeleton className="h-10 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  const connectedExchanges = exchanges?.filter((e) => e.is_valid) || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Trading Configuration"
        description="Configure your automated trading settings"
        actions={
          <Button
            variant="primary"
            onClick={handleSave}
            disabled={!hasChanges}
            isLoading={updateMutation.isPending}
            leftIcon={<Save className="h-4 w-4" />}
          >
            Save Configuration
          </Button>
        }
      />

      {/* Trading Status */}
      <Card variant={config.is_enabled ? 'success' : 'bordered'}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-1">
                Trading Status
              </h3>
              <p className="text-sm text-text-secondary">
                {config.is_enabled
                  ? 'Trades will execute automatically on signals'
                  : 'Trading is currently disabled'}
              </p>
            </div>
            <Toggle
              checked={config.is_enabled}
              onChange={(checked) => handleChange('is_enabled', checked)}
              size="lg"
            />
          </div>
          {config.is_enabled && (
            <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400 mt-0.5 shrink-0" />
              <p className="text-sm text-amber-400">
                Auto-trading is enabled. Trades will execute automatically based on incoming signals.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trading Pair */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Trading Pair</CardTitle>
        </CardHeader>
        <CardContent>
          <Select
            value={config.symbol}
            onChange={(value) => handleChange('symbol', value)}
            options={[
              { value: 'USD/COP', label: 'USD/COP' },
            ]}
            label="Symbol"
          />
        </CardContent>
      </Card>

      {/* Risk Management */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Risk Management</CardTitle>
        </CardHeader>
        <CardContent className="space-y-8">
          <Slider
            value={config.stop_loss_pct * 100}
            onChange={(value) => handleChange('stop_loss_pct', value / 100)}
            min={0.1}
            max={10}
            step={0.1}
            label="Stop Loss"
            formatValue={(v) => `${v.toFixed(1)}%`}
          />

          <Slider
            value={config.take_profit_pct * 100}
            onChange={(value) => handleChange('take_profit_pct', value / 100)}
            min={0.1}
            max={50}
            step={0.1}
            label="Take Profit"
            formatValue={(v) => `${v.toFixed(1)}%`}
          />

          <div>
            <Slider
              value={config.min_confidence * 100}
              onChange={(value) => handleChange('min_confidence', value / 100)}
              min={50}
              max={100}
              step={1}
              label="Minimum Confidence"
              formatValue={(v) => `${v.toFixed(0)}%`}
            />
            <p className="text-xs text-text-muted mt-2">
              Signals below this confidence level will be skipped
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Execute on Exchanges */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Execute on Exchanges</CardTitle>
        </CardHeader>
        <CardContent>
          {connectedExchanges.length > 0 ? (
            <div className="space-y-3">
              {connectedExchanges.map((exchange) => (
                <Checkbox
                  key={exchange.id}
                  checked={config.execute_on_exchanges.includes(exchange.exchange)}
                  onChange={(checked) => handleExchangeToggle(exchange.exchange, checked)}
                  label={EXCHANGE_NAMES[exchange.exchange]}
                  description={`Execute trades on ${EXCHANGE_NAMES[exchange.exchange]}`}
                />
              ))}
            </div>
          ) : (
            <p className="text-text-muted">
              No exchanges connected. Connect an exchange to enable trading.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Config Summary */}
      <Card variant="bordered">
        <CardHeader>
          <CardTitle>Configuration Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-text-muted">Status</p>
              <Badge variant={config.is_enabled ? 'live' : 'offline'} dot>
                {config.is_enabled ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-text-muted">Stop Loss</p>
              <p className="text-lg font-semibold text-market-down">
                {formatPercent(config.stop_loss_pct)}
              </p>
            </div>
            <div>
              <p className="text-sm text-text-muted">Take Profit</p>
              <p className="text-lg font-semibold text-market-up">
                {formatPercent(config.take_profit_pct)}
              </p>
            </div>
            <div>
              <p className="text-sm text-text-muted">Min Confidence</p>
              <p className="text-lg font-semibold text-terminal-accent">
                {formatPercent(config.min_confidence)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
