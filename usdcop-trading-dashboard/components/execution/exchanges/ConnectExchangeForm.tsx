import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Key, Shield, AlertTriangle, CheckCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { exchangeService } from '@/services/exchangeService'
import { toast } from '@/stores/uiStore'
import { EXCHANGE_NAMES, SUPPORTED_EXCHANGES } from '@/lib/constants'
import { ConnectExchangeRequestSchema } from '@/contracts/exchange'

export function ConnectExchangeForm() {
  const { exchange } = useParams<{ exchange: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [formData, setFormData] = useState({
    api_key: '',
    api_secret: '',
  })
  const [errors, setErrors] = useState<Record<string, string>>({})

  const exchangeName = EXCHANGE_NAMES[exchange as keyof typeof EXCHANGE_NAMES] || exchange

  const connectMutation = useMutation({
    mutationFn: () => exchangeService.connectExchange(exchange!, formData),
    onSuccess: (result) => {
      if (result.is_valid) {
        toast.success(`${exchangeName} connected successfully!`)
        queryClient.invalidateQueries({ queryKey: ['exchanges'] })
        navigate('/exchanges')
      } else {
        if (result.has_withdraw_permission) {
          setErrors({
            api_key: 'This API key has WITHDRAW permission. Please create a new key without withdrawal access.',
          })
        } else {
          setErrors({
            api_key: result.error_message || 'Invalid credentials',
          })
        }
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Connection failed')
    },
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    setErrors((prev) => ({ ...prev, [name]: '' }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const result = ConnectExchangeRequestSchema.safeParse(formData)
    if (!result.success) {
      const fieldErrors: Record<string, string> = {}
      result.error.errors.forEach((err) => {
        if (err.path[0]) {
          fieldErrors[err.path[0] as string] = err.message
        }
      })
      setErrors(fieldErrors)
      return
    }

    connectMutation.mutate()
  }

  if (!exchange || !SUPPORTED_EXCHANGES.includes(exchange as any)) {
    return (
      <div className="text-center py-12">
        <p className="text-text-secondary">Invalid exchange</p>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Security Warning */}
      <Card variant="warning">
        <CardContent className="p-6">
          <div className="flex gap-4">
            <AlertTriangle className="h-6 w-6 text-amber-400 shrink-0" />
            <div>
              <h3 className="font-semibold text-text-primary mb-2">
                Important Security Notes
              </h3>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-emerald-400" />
                  Only enable <strong>SPOT TRADING</strong> permission
                </li>
                <li className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-red-400" />
                  <strong>NEVER</strong> enable WITHDRAWAL permission
                </li>
                <li className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-terminal-accent" />
                  We recommend setting IP whitelist
                </li>
                <li className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-terminal-accent" />
                  Your keys are encrypted with AES-256
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Connect Form */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Connect {exchangeName}</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <Input
              name="api_key"
              label="API Key"
              placeholder="Enter your API key"
              value={formData.api_key}
              onChange={handleChange}
              error={errors.api_key}
              leftIcon={<Key className="h-4 w-4" />}
            />

            <Input
              name="api_secret"
              type="password"
              label="API Secret"
              placeholder="Enter your API secret"
              value={formData.api_secret}
              onChange={handleChange}
              error={errors.api_secret}
              leftIcon={<Shield className="h-4 w-4" />}
            />

            <Button
              type="submit"
              variant="primary"
              className="w-full"
              isLoading={connectMutation.isPending}
            >
              Validate & Connect
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Instructions */}
      <Card variant="bordered">
        <CardContent className="p-6">
          <h4 className="font-medium text-text-primary mb-3">
            How to get your API keys:
          </h4>
          <ol className="space-y-2 text-sm text-text-secondary">
            <li>1. Go to {exchangeName.toLowerCase()}.com → API Management</li>
            <li>2. Create a new API key</li>
            <li>3. Enable only SPOT trading permissions</li>
            <li>4. Copy and paste the keys here</li>
          </ol>
        </CardContent>
      </Card>
    </div>
  )
}
