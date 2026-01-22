import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`
}

export function formatNumber(value: number, decimals = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)
}

export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d)
}

export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins} min ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return formatDate(d)
}

export function truncateAddress(address: string, chars = 4): string {
  if (address.length <= chars * 2 + 3) return address
  return `${address.slice(0, chars)}...${address.slice(-chars)}`
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export function getActionColor(action: number | string): string {
  const actionNum = typeof action === 'string'
    ? { 'BUY': 2, 'SELL': 0, 'HOLD': 1 }[action] ?? 1
    : action

  switch (actionNum) {
    case 2: return 'text-market-up'
    case 0: return 'text-market-down'
    default: return 'text-market-neutral'
  }
}

export function getActionBgColor(action: number | string): string {
  const actionNum = typeof action === 'string'
    ? { 'BUY': 2, 'SELL': 0, 'HOLD': 1 }[action] ?? 1
    : action

  switch (actionNum) {
    case 2: return 'bg-up'
    case 0: return 'bg-down'
    default: return 'bg-slate-800/50'
  }
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'filled':
    case 'executed':
    case 'success':
      return 'text-market-up'
    case 'rejected':
    case 'failed':
    case 'error':
      return 'text-market-down'
    case 'pending':
    case 'submitted':
      return 'text-status-delayed'
    default:
      return 'text-text-secondary'
  }
}
