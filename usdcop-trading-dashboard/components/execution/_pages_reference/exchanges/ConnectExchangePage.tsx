import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { ConnectExchangeForm } from '@/components/exchanges/ConnectExchangeForm'
import { EXCHANGE_NAMES } from '@/lib/constants'

export function ConnectExchangePage() {
  const { exchange } = useParams<{ exchange: string }>()
  const exchangeName = EXCHANGE_NAMES[exchange as keyof typeof EXCHANGE_NAMES] || exchange

  return (
    <div className="space-y-6">
      <PageHeader
        title={`Connect ${exchangeName}`}
        breadcrumbs={[
          { label: 'Exchanges', href: '/exchanges' },
          { label: `Connect ${exchangeName}` },
        ]}
        actions={
          <Link
            to="/exchanges"
            className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Exchanges
          </Link>
        }
      />

      <ConnectExchangeForm />
    </div>
  )
}
