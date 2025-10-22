'use client'

/**
 * Página de Trading con Gráfico REAL
 * =================================
 *
 * Página dedicada para mostrar el gráfico TradingView con datos reales
 */

import RealDataTradingChart from '@/components/charts/RealDataTradingChart'
import RealTimePriceDisplay from '@/components/realtime/RealTimePriceDisplay'
import { useRealTimePrice } from '@/hooks/useRealTimePrice'
import { useDbStats } from '@/hooks/useDbStats'

export default function TradingPage() {
  const { formattedPrice, isConnected, currentPrice } = useRealTimePrice('USDCOP')
  const { stats: dbStats } = useDbStats(60000) // Refresh every 60 seconds

  return (
    <div className="min-h-screen bg-fintech-dark-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">
            USD/COP Real-Time Trading Chart
          </h1>
          <p className="text-fintech-dark-400">
            Datos históricos 100% reales • Disponible 24/7 • Tiempo real solo en horario de mercado (8AM-12:55PM COT)
          </p>
        </div>

        {/* Price Display */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <RealTimePriceDisplay symbol="USDCOP" />

          <div className="bg-fintech-dark-800 rounded-lg p-4 border border-fintech-dark-700">
            <h3 className="text-lg font-semibold text-white mb-2">Estado de Conexión</h3>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-white">
                {isConnected ? 'Conectado a WebSocket' : 'Desconectado'}
              </span>
            </div>
            {currentPrice && (
              <p className="text-sm text-fintech-dark-400 mt-2">
                Fuente: {currentPrice.source}
              </p>
            )}
          </div>

          <div className="bg-fintech-dark-800 rounded-lg p-4 border border-fintech-dark-700">
            <h3 className="text-lg font-semibold text-white mb-2">Datos Disponibles</h3>
            <div className="space-y-1">
              <p className="text-sm text-white">✅ {dbStats.totalRecords.toLocaleString()} registros históricos</p>
              <p className="text-sm text-white">✅ Indicadores técnicos</p>
              <p className="text-sm text-white">✅ WebSocket tiempo real</p>
            </div>
          </div>
        </div>

        {/* Main Chart */}
        <div className="bg-fintech-dark-800 rounded-lg p-6 border border-fintech-dark-700">
          <h2 className="text-2xl font-bold text-white mb-4">
            Gráfico USD/COP - Datos Históricos Reales ({dbStats.totalRecords.toLocaleString()} registros)
          </h2>
          <RealDataTradingChart
            symbol="USDCOP"
            timeframe="5m"
            height={600}
            className="w-full"
          />
        </div>

        {/* Instructions */}
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-400 mb-2">
            📊 Características del Gráfico
          </h3>
          <ul className="space-y-1 text-sm text-blue-300">
            <li>• <strong>Datos históricos</strong>: {dbStats.totalRecords.toLocaleString()} registros reales disponibles 24/7</li>
            <li>• <strong>Indicadores técnicos</strong>: EMA 20/50, Bollinger Bands, RSI</li>
            <li>• <strong>Tiempo real</strong>: Solo durante horario de mercado (8AM-12:55PM COT)</li>
            <li>• <strong>Interactivo</strong>: Canvas HTML5 para máxima performance</li>
            <li>• <strong>Fuente</strong>: TwelveData (datos del mercado real)</li>
          </ul>
        </div>

        {/* API Status */}
        <div className="bg-fintech-dark-800 rounded-lg p-4 border border-fintech-dark-700">
          <h3 className="text-lg font-semibold text-white mb-2">API Status</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-fintech-dark-400">REST API:</span>
              <span className="text-green-400 ml-2">http://localhost:8000</span>
            </div>
            <div>
              <span className="text-fintech-dark-400">WebSocket:</span>
              <span className="text-green-400 ml-2">ws://localhost:8000/ws</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}