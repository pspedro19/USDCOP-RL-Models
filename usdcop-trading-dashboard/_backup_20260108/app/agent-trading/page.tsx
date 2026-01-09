'use client'

/**
 * Agent Trading Dashboard Page
 *
 * Vista completa del sistema de trading RL en tiempo real.
 * Incluye:
 * - Gráfica con posiciones del agente
 * - Tabla de acciones en tiempo real
 * - Métricas de performance
 * - Curva de equity
 * - Alertas del sistema
 */

import React, { useState, useEffect } from 'react'
import ChartWithPositions from '@/components/charts/ChartWithPositions'
import AgentActionsTable from '@/components/trading/AgentActionsTable'

// Types
interface SystemStatus {
  inference: 'online' | 'offline' | 'error'
  market: 'open' | 'closed' | 'pre-market' | 'after-hours'
  database: 'connected' | 'disconnected'
  lastUpdate: string
}

interface DailyStats {
  totalTrades: number
  winRate: number
  dailyPnL: number
  dailyReturn: number
  maxDrawdown: number
  currentPosition: number
  equity: number
}

export default function AgentTradingPage() {
  // State
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    inference: 'offline',
    market: 'closed',
    database: 'connected',
    lastUpdate: new Date().toISOString(),
  })
  const [dailyStats, setDailyStats] = useState<DailyStats | null>(null)
  const [selectedView, setSelectedView] = useState<'split' | 'chart' | 'table'>('split')
  const [refreshKey, setRefreshKey] = useState(0)

  // Check market status
  useEffect(() => {
    const checkMarketStatus = () => {
      const now = new Date()
      const cotOffset = -5 * 60 // UTC-5
      const utcMinutes = now.getUTCHours() * 60 + now.getUTCMinutes()
      const cotMinutes = ((utcMinutes + cotOffset) % 1440 + 1440) % 1440
      const cotHour = Math.floor(cotMinutes / 60)
      const dayOfWeek = now.getUTCDay()

      let market: SystemStatus['market'] = 'closed'
      if (dayOfWeek >= 1 && dayOfWeek <= 5) {
        if (cotHour >= 8 && cotHour < 13) {
          market = 'open'
        } else if (cotHour >= 7 && cotHour < 8) {
          market = 'pre-market'
        } else if (cotHour >= 13 && cotHour < 15) {
          market = 'after-hours'
        }
      }

      setSystemStatus(prev => ({
        ...prev,
        market,
        inference: market === 'open' ? 'online' : 'offline',
        lastUpdate: now.toISOString(),
      }))
    }

    checkMarketStatus()
    const interval = setInterval(checkMarketStatus, 60000)
    return () => clearInterval(interval)
  }, [])

  // Fetch daily stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/agent/actions?action=today')
        const data = await response.json()

        if (data.success && data.data.performance) {
          const perf = data.data.performance
          setDailyStats({
            totalTrades: perf.total_trades || 0,
            winRate: perf.win_rate || 0,
            dailyPnL: perf.daily_pnl || 0,
            dailyReturn: perf.daily_return_pct || 0,
            maxDrawdown: perf.max_drawdown_intraday_pct || 0,
            currentPosition: data.data.realtimeMetrics?.currentPosition || 0,
            equity: perf.ending_equity || 10000,
          })
        }
      } catch (error) {
        console.error('Error fetching stats:', error)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  // Force refresh
  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1)
  }

  // Status badge component
  const StatusBadge = ({
    status,
    label
  }: {
    status: 'online' | 'offline' | 'error' | 'open' | 'closed' | 'pre-market' | 'after-hours' | 'connected' | 'disconnected'
    label: string
  }) => {
    const colors = {
      online: 'bg-green-900/50 text-green-400 border-green-700',
      offline: 'bg-gray-800 text-gray-400 border-gray-700',
      error: 'bg-red-900/50 text-red-400 border-red-700',
      open: 'bg-green-900/50 text-green-400 border-green-700',
      closed: 'bg-gray-800 text-gray-400 border-gray-700',
      'pre-market': 'bg-yellow-900/50 text-yellow-400 border-yellow-700',
      'after-hours': 'bg-blue-900/50 text-blue-400 border-blue-700',
      connected: 'bg-green-900/50 text-green-400 border-green-700',
      disconnected: 'bg-red-900/50 text-red-400 border-red-700',
    }

    return (
      <div className={`px-3 py-1 rounded-lg border text-xs font-medium ${colors[status]}`}>
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${
            ['online', 'open', 'connected'].includes(status) ? 'bg-green-500 animate-pulse' :
            status === 'error' || status === 'disconnected' ? 'bg-red-500' :
            status === 'pre-market' ? 'bg-yellow-500' :
            status === 'after-hours' ? 'bg-blue-500' :
            'bg-gray-500'
          }`} />
          {label}: {status.toUpperCase()}
        </span>
      </div>
    )
  }

  // Quick stats component
  const QuickStats = () => {
    if (!dailyStats) return null

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <StatCard
          label="Equity"
          value={`$${dailyStats.equity.toFixed(2)}`}
          color="white"
        />
        <StatCard
          label="P&L Día"
          value={`${dailyStats.dailyPnL >= 0 ? '+' : ''}$${dailyStats.dailyPnL.toFixed(2)}`}
          color={dailyStats.dailyPnL >= 0 ? 'green' : 'red'}
        />
        <StatCard
          label="Retorno"
          value={`${dailyStats.dailyReturn >= 0 ? '+' : ''}${dailyStats.dailyReturn.toFixed(2)}%`}
          color={dailyStats.dailyReturn >= 0 ? 'green' : 'red'}
        />
        <StatCard
          label="Trades"
          value={dailyStats.totalTrades.toString()}
          color="blue"
        />
        <StatCard
          label="Win Rate"
          value={`${(dailyStats.winRate * 100).toFixed(1)}%`}
          color={dailyStats.winRate >= 0.5 ? 'green' : 'red'}
        />
        <StatCard
          label="Max DD"
          value={`${(dailyStats.maxDrawdown * 100).toFixed(2)}%`}
          color="yellow"
        />
        <StatCard
          label="Posición"
          value={dailyStats.currentPosition > 0.1 ? `LONG ${(dailyStats.currentPosition * 100).toFixed(0)}%` :
                 dailyStats.currentPosition < -0.1 ? `SHORT ${(Math.abs(dailyStats.currentPosition) * 100).toFixed(0)}%` :
                 'FLAT'}
          color={dailyStats.currentPosition > 0.1 ? 'green' :
                 dailyStats.currentPosition < -0.1 ? 'red' : 'gray'}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-4 py-3">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            {/* Title and status */}
            <div className="flex items-center gap-4">
              <h1 className="text-xl font-bold">
                USD/COP Agent Trading
              </h1>
              <div className="flex gap-2">
                <StatusBadge status={systemStatus.market} label="Mercado" />
                <StatusBadge status={systemStatus.inference} label="Inferencia" />
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3">
              {/* View toggle */}
              <div className="flex bg-gray-800 rounded-lg p-1">
                {(['split', 'chart', 'table'] as const).map(view => (
                  <button
                    key={view}
                    onClick={() => setSelectedView(view)}
                    className={`px-3 py-1 rounded text-sm transition ${
                      selectedView === view
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {view === 'split' ? 'Split' : view === 'chart' ? 'Gráfica' : 'Tabla'}
                  </button>
                ))}
              </div>

              {/* Refresh button */}
              <button
                onClick={handleRefresh}
                className="px-3 py-1.5 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition flex items-center gap-2"
              >
                <span className="text-lg">↻</span>
                Actualizar
              </button>

              {/* Time */}
              <div className="text-sm text-gray-400">
                {new Date().toLocaleTimeString('es-CO', { timeZone: 'America/Bogota' })} COT
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-[1920px] mx-auto p-4 space-y-4">
        {/* Quick stats */}
        <QuickStats />

        {/* Main panels */}
        <div className={`grid gap-4 ${
          selectedView === 'split' ? 'lg:grid-cols-2' : 'grid-cols-1'
        }`}>
          {/* Chart panel */}
          {(selectedView === 'split' || selectedView === 'chart') && (
            <div className={selectedView === 'chart' ? 'col-span-1' : ''}>
              <ChartWithPositions
                key={`chart-${refreshKey}`}
                symbol="USDCOP"
                timeframe="5m"
                height={selectedView === 'chart' ? 600 : 450}
                showEquityCurve={true}
                showPositionLine={true}
                autoRefresh={true}
                refreshInterval={30000}
              />
            </div>
          )}

          {/* Table panel */}
          {(selectedView === 'split' || selectedView === 'table') && (
            <div className={selectedView === 'table' ? 'col-span-1' : ''}>
              <AgentActionsTable
                key={`table-${refreshKey}`}
                autoRefresh={true}
                refreshInterval={30000}
                showPerformance={selectedView === 'table'}
                showAlerts={true}
                maxRows={selectedView === 'table' ? 200 : 50}
              />
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex flex-wrap justify-between items-center gap-4 text-sm text-gray-500">
            <div className="flex gap-6">
              <span>Modelo: PPO V11 (Fold 0)</span>
              <span>Horario: 8:00 - 12:55 COT</span>
              <span>Intervalo: 5 minutos</span>
            </div>
            <div>
              Última actualización: {new Date(systemStatus.lastUpdate).toLocaleTimeString()}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

// Stat card component
function StatCard({
  label,
  value,
  color
}: {
  label: string
  value: string
  color: 'white' | 'green' | 'red' | 'yellow' | 'blue' | 'gray'
}) {
  const colorClasses = {
    white: 'text-white',
    green: 'text-green-400',
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    blue: 'text-blue-400',
    gray: 'text-gray-400',
  }

  return (
    <div className="bg-gray-900 rounded-lg px-4 py-3 border border-gray-800">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-bold ${colorClasses[color]}`}>{value}</div>
    </div>
  )
}
