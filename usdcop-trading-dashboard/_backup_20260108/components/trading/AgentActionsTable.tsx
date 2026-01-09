'use client'

/**
 * AgentActionsTable Component
 *
 * Tabla interactiva que muestra las acciones tomadas por el agente RL
 * durante el día de trading actual o una fecha seleccionada.
 *
 * Features:
 * - Polling automático cada 30 segundos
 * - Indicadores visuales de tipo de acción
 * - Métricas de performance en tiempo real
 * - Filtros por tipo de acción
 * - Exportación a CSV
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { format, parseISO } from 'date-fns'

// Types
interface AgentAction {
  action_id: number
  timestamp_cot: string
  session_date: string
  bar_number: number
  action_type: string
  side: string | null
  price_at_action: number
  position_before: number
  position_after: number
  position_change: number
  pnl_action: number | null
  pnl_daily: number | null
  model_confidence: number
  marker_type: string
  marker_color: string
  reason_code: string | null
}

interface SessionPerformance {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number | null
  profit_factor: number | null
  daily_pnl: number | null
  daily_return_pct: number | null
  starting_equity: number | null
  ending_equity: number | null
  max_drawdown_intraday_pct: number | null
  intraday_sharpe: number | null
  total_long_bars: number
  total_short_bars: number
  total_flat_bars: number
  status: string
}

interface Alert {
  alert_id: number
  alert_type: string
  severity: string
  message: string
  timestamp_utc: string
}

interface AgentActionsTableProps {
  initialDate?: string
  autoRefresh?: boolean
  refreshInterval?: number
  showPerformance?: boolean
  showAlerts?: boolean
  maxRows?: number
  onActionClick?: (action: AgentAction) => void
}

// Action type configuration
const ACTION_CONFIG: Record<string, { icon: string; label: string; bgClass: string; textClass: string }> = {
  'ENTRY_LONG': { icon: '▲', label: 'Entry Long', bgClass: 'bg-green-900/30', textClass: 'text-green-400' },
  'ENTRY_SHORT': { icon: '▼', label: 'Entry Short', bgClass: 'bg-red-900/30', textClass: 'text-red-400' },
  'EXIT_LONG': { icon: '×', label: 'Exit Long', bgClass: 'bg-amber-900/30', textClass: 'text-amber-400' },
  'EXIT_SHORT': { icon: '×', label: 'Exit Short', bgClass: 'bg-amber-900/30', textClass: 'text-amber-400' },
  'INCREASE_LONG': { icon: '↑', label: '+Long', bgClass: 'bg-green-900/20', textClass: 'text-green-300' },
  'INCREASE_SHORT': { icon: '↓', label: '+Short', bgClass: 'bg-red-900/20', textClass: 'text-red-300' },
  'DECREASE_LONG': { icon: '↓', label: '-Long', bgClass: 'bg-gray-800', textClass: 'text-gray-400' },
  'DECREASE_SHORT': { icon: '↑', label: '-Short', bgClass: 'bg-gray-800', textClass: 'text-gray-400' },
  'FLIP_LONG': { icon: '⇅', label: 'Flip Long', bgClass: 'bg-green-900/40', textClass: 'text-green-500' },
  'FLIP_SHORT': { icon: '⇅', label: 'Flip Short', bgClass: 'bg-red-900/40', textClass: 'text-red-500' },
  'HOLD': { icon: '—', label: 'Hold', bgClass: 'bg-gray-900', textClass: 'text-gray-500' },
}

export default function AgentActionsTable({
  initialDate,
  autoRefresh = true,
  refreshInterval = 30000,
  showPerformance = true,
  showAlerts = true,
  maxRows = 100,
  onActionClick,
}: AgentActionsTableProps) {
  // State
  const [actions, setActions] = useState<AgentAction[]>([])
  const [performance, setPerformance] = useState<SessionPerformance | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedDate, setSelectedDate] = useState(
    initialDate || new Date().toISOString().split('T')[0]
  )
  const [filterType, setFilterType] = useState<string>('all')
  const [showHolds, setShowHolds] = useState(false)
  const [isLive, setIsLive] = useState(false)

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(`/api/agent/actions?date=${selectedDate}&limit=${maxRows}`)
      const data = await response.json()

      if (data.success) {
        setActions(data.data.actions || [])
        setPerformance(data.data.performance)
        setAlerts(data.data.alerts || [])
        setIsLive(data.data.isLive || false)
        setError(null)
      } else {
        setError(data.error || 'Error fetching data')
      }
    } catch (err) {
      setError('Network error')
      console.error('Error fetching agent actions:', err)
    } finally {
      setLoading(false)
    }
  }, [selectedDate, maxRows])

  // Initial fetch and polling
  useEffect(() => {
    fetchData()

    if (autoRefresh) {
      const interval = setInterval(fetchData, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchData, autoRefresh, refreshInterval])

  // Filtered actions
  const filteredActions = useMemo(() => {
    let filtered = actions

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(a => a.action_type === filterType)
    }

    // Filter out HOLD if not showing
    if (!showHolds) {
      filtered = filtered.filter(a => a.action_type !== 'HOLD')
    }

    return filtered
  }, [actions, filterType, showHolds])

  // Action types for filter
  const actionTypes = useMemo(() => {
    const types = new Set(actions.map(a => a.action_type))
    return Array.from(types)
  }, [actions])

  // Export to CSV
  const exportToCSV = () => {
    const headers = [
      'Hora', 'Barra', 'Acción', 'Lado', 'Precio',
      'Pos. Antes', 'Pos. Después', 'Cambio', 'PnL', 'PnL Día', 'Confianza'
    ]

    const rows = filteredActions.map(a => [
      format(parseISO(a.timestamp_cot), 'HH:mm'),
      a.bar_number,
      a.action_type,
      a.side || '',
      a.price_at_action?.toFixed(2),
      a.position_before?.toFixed(2),
      a.position_after?.toFixed(2),
      a.position_change?.toFixed(2),
      a.pnl_action?.toFixed(2) || '',
      a.pnl_daily?.toFixed(2) || '',
      a.model_confidence?.toFixed(2)
    ])

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `agent_actions_${selectedDate}.csv`
    a.click()
  }

  // Render performance metrics
  const renderPerformanceMetrics = () => {
    if (!showPerformance || !performance) return null

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 bg-gray-800/50 p-4 rounded-lg mb-4">
        <MetricCard
          label="Trades"
          value={performance.total_trades}
          color="white"
        />
        <MetricCard
          label="Win Rate"
          value={performance.win_rate != null ? `${(performance.win_rate * 100).toFixed(1)}%` : 'N/A'}
          color={performance.win_rate != null && performance.win_rate >= 0.5 ? 'green' : 'red'}
        />
        <MetricCard
          label="P&L Día"
          value={performance.daily_pnl != null ? `$${performance.daily_pnl.toFixed(2)}` : 'N/A'}
          color={performance.daily_pnl != null && performance.daily_pnl >= 0 ? 'green' : 'red'}
        />
        <MetricCard
          label="Retorno"
          value={performance.daily_return_pct != null ? `${performance.daily_return_pct.toFixed(2)}%` : 'N/A'}
          color={performance.daily_return_pct != null && performance.daily_return_pct >= 0 ? 'green' : 'red'}
        />
        <MetricCard
          label="Max DD"
          value={performance.max_drawdown_intraday_pct != null
            ? `${(performance.max_drawdown_intraday_pct * 100).toFixed(2)}%`
            : 'N/A'}
          color="yellow"
        />
        <MetricCard
          label="Profit Factor"
          value={performance.profit_factor != null ? performance.profit_factor.toFixed(2) : 'N/A'}
          color={performance.profit_factor != null && performance.profit_factor >= 1 ? 'green' : 'red'}
        />
      </div>
    )
  }

  // Render alerts
  const renderAlerts = () => {
    if (!showAlerts || alerts.length === 0) return null

    return (
      <div className="mb-4 space-y-2">
        {alerts.map(alert => (
          <div
            key={alert.alert_id}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              alert.severity === 'CRITICAL' ? 'bg-red-900/50 text-red-200' :
              alert.severity === 'ERROR' ? 'bg-red-800/30 text-red-300' :
              alert.severity === 'WARNING' ? 'bg-yellow-900/30 text-yellow-300' :
              'bg-blue-900/30 text-blue-300'
            }`}
          >
            <span className="font-bold">
              {alert.severity === 'CRITICAL' ? '🚨' :
               alert.severity === 'ERROR' ? '❌' :
               alert.severity === 'WARNING' ? '⚠️' : 'ℹ️'}
            </span>
            <span className="flex-1">{alert.message}</span>
            <span className="text-xs opacity-70">
              {format(parseISO(alert.timestamp_utc), 'HH:mm')}
            </span>
          </div>
        ))}
      </div>
    )
  }

  // Render position bar
  const renderPositionBar = (position: number) => {
    const absPosition = Math.abs(position)
    const isLong = position > 0
    const width = Math.min(absPosition * 100, 100)

    return (
      <div className="relative w-20 h-3 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`absolute h-full ${isLong ? 'bg-green-500 left-1/2' : 'bg-red-500 right-1/2'}`}
          style={{ width: `${width / 2}%` }}
        />
        <div className="absolute w-px h-full bg-gray-400 left-1/2 transform -translate-x-1/2" />
      </div>
    )
  }

  // Loading state
  if (loading && actions.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
          <span className="ml-3 text-gray-400">Cargando acciones...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 md:p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold text-white">Acciones del Agente</h2>
          {isLive && (
            <span className="flex items-center gap-1.5 px-2 py-1 bg-green-900/30 rounded text-xs text-green-400">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              LIVE
            </span>
          )}
          {loading && (
            <span className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Date picker */}
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="bg-gray-800 text-white text-sm px-3 py-1.5 rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
          />

          {/* Filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="bg-gray-800 text-white text-sm px-3 py-1.5 rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
          >
            <option value="all">Todas</option>
            {actionTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>

          {/* Show holds toggle */}
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showHolds}
              onChange={(e) => setShowHolds(e.target.checked)}
              className="rounded bg-gray-700 border-gray-600"
            />
            HOLDs
          </label>

          {/* Export button */}
          <button
            onClick={exportToCSV}
            className="px-3 py-1.5 bg-gray-800 text-gray-300 text-sm rounded border border-gray-700 hover:bg-gray-700 transition"
          >
            📥 CSV
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 text-red-400 px-4 py-2 rounded-lg">
          {error}
        </div>
      )}

      {/* Alerts */}
      {renderAlerts()}

      {/* Performance metrics */}
      {renderPerformanceMetrics()}

      {/* Position summary */}
      {performance && (
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="text-gray-400">Long: {performance.total_long_bars} barras</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span className="text-gray-400">Short: {performance.total_short_bars} barras</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-500 rounded-full" />
            <span className="text-gray-400">Flat: {performance.total_flat_bars} barras</span>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-400 border-b border-gray-800">
              <th className="px-3 py-2 font-medium">Hora</th>
              <th className="px-3 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Acción</th>
              <th className="px-3 py-2 font-medium">Precio</th>
              <th className="px-3 py-2 font-medium">Posición</th>
              <th className="px-3 py-2 font-medium text-right">P&L</th>
              <th className="px-3 py-2 font-medium text-right">P&L Día</th>
              <th className="px-3 py-2 font-medium text-center">Conf.</th>
            </tr>
          </thead>
          <tbody>
            {filteredActions.length === 0 ? (
              <tr>
                <td colSpan={8} className="text-center py-8 text-gray-500">
                  No hay acciones para mostrar
                </td>
              </tr>
            ) : (
              filteredActions.map((action, idx) => {
                const config = ACTION_CONFIG[action.action_type] || ACTION_CONFIG['HOLD']

                return (
                  <tr
                    key={action.action_id}
                    onClick={() => onActionClick?.(action)}
                    className={`border-b border-gray-800/50 hover:bg-gray-800/50 transition cursor-pointer ${config.bgClass}`}
                  >
                    <td className="px-3 py-2 font-mono text-gray-300">
                      {format(parseISO(action.timestamp_cot), 'HH:mm')}
                    </td>
                    <td className="px-3 py-2 text-gray-400">
                      {action.bar_number}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`inline-flex items-center gap-1.5 ${config.textClass}`}>
                        <span className="text-lg">{config.icon}</span>
                        <span className="font-medium">{config.label}</span>
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-200">
                      {action.price_at_action?.toFixed(2)}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        {renderPositionBar(action.position_after)}
                        <span className={`font-mono text-xs ${
                          action.position_after > 0 ? 'text-green-400' :
                          action.position_after < 0 ? 'text-red-400' :
                          'text-gray-400'
                        }`}>
                          {action.position_after > 0 ? '+' : ''}{action.position_after.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className={`px-3 py-2 font-mono text-right ${
                      (action.pnl_action || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {action.pnl_action != null ? (
                        <>
                          {action.pnl_action >= 0 ? '+' : ''}
                          ${action.pnl_action.toFixed(2)}
                        </>
                      ) : '—'}
                    </td>
                    <td className={`px-3 py-2 font-mono text-right ${
                      (action.pnl_daily || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {action.pnl_daily != null ? (
                        <>
                          {action.pnl_daily >= 0 ? '+' : ''}
                          ${action.pnl_daily.toFixed(2)}
                        </>
                      ) : '—'}
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex justify-center">
                        <div className="w-12 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500"
                            style={{ width: `${(action.model_confidence || 0) * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="flex justify-between items-center text-xs text-gray-500">
        <span>
          {filteredActions.length} acciones
          {filterType !== 'all' && ` (filtradas por ${filterType})`}
        </span>
        <span>
          Última actualización: {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  )
}

// Metric card component
function MetricCard({
  label,
  value,
  color
}: {
  label: string
  value: string | number
  color: 'white' | 'green' | 'red' | 'yellow' | 'blue'
}) {
  const colorClasses = {
    white: 'text-white',
    green: 'text-green-400',
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    blue: 'text-blue-400',
  }

  return (
    <div className="bg-gray-900/50 rounded-lg px-3 py-2">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-bold ${colorClasses[color]}`}>{value}</div>
    </div>
  )
}
