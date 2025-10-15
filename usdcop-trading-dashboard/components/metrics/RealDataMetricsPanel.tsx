/**
 * Real Data Metrics Panel
 * Displays only metrics that are accurate with our actual data
 * Based on: price, bid, ask, OHLC (no synthetic volume)
 */

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Clock,
  Target,
  BarChart3,
  AlertCircle,
  CheckCircle,
  Percent
} from 'lucide-react';

import { RealMarketMetrics, RealMarketMetricsCalculator, OHLCData } from '../../lib/services/real-market-metrics';

interface RealDataMetricsPanelProps {
  data: OHLCData[];
  currentTick?: {
    price: number;
    bid: number;
    ask: number;
    timestamp: Date;
  };
  className?: string;
}

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'gray';
  change?: number;
  isGood?: boolean;
  tooltip?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color,
  change,
  isGood,
  tooltip
}) => {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30 text-blue-400',
    green: 'from-emerald-500/20 to-emerald-600/20 border-emerald-500/30 text-emerald-400',
    red: 'from-red-500/20 to-red-600/20 border-red-500/30 text-red-400',
    yellow: 'from-yellow-500/20 to-yellow-600/20 border-yellow-500/30 text-yellow-400',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30 text-purple-400',
    gray: 'from-slate-500/20 to-slate-600/20 border-slate-500/30 text-slate-400'
  };

  return (
    <motion.div
      className={`relative p-4 rounded-xl border bg-gradient-to-br ${colorClasses[color]} backdrop-blur`}
      whileHover={{ scale: 1.02 }}
      title={tooltip}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-white/90">{title}</span>
        </div>

        {change !== undefined && (
          <div className={`flex items-center gap-1 text-xs ${
            isGood ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {change > 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
            {Math.abs(change).toFixed(2)}%
          </div>
        )}
      </div>

      {/* Value */}
      <div className="mb-1">
        <span className="text-2xl font-mono font-bold text-white">
          {typeof value === 'number' ? value.toFixed(2) : value}
        </span>
      </div>

      {/* Subtitle */}
      {subtitle && (
        <div className="text-xs text-white/60">{subtitle}</div>
      )}
    </motion.div>
  );
};

export default function RealDataMetricsPanel({
  data,
  currentTick,
  className = ''
}: RealDataMetricsPanelProps) {
  // Calculate real metrics
  const metrics = useMemo(() => {
    if (data.length === 0) return null;

    try {
      return RealMarketMetricsCalculator.calculateMetrics(data, currentTick);
    } catch (error) {
      console.error('Error calculating metrics:', error);
      return null;
    }
  }, [data, currentTick]);

  const formattedMetrics = useMemo(() => {
    if (!metrics) return null;
    return RealMarketMetricsCalculator.formatMetrics(metrics);
  }, [metrics]);

  if (!metrics || !formattedMetrics) {
    return (
      <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6 ${className}`}>
        <div className="flex items-center gap-2 text-slate-400">
          <AlertCircle className="h-5 w-5" />
          <span>Calculando métricas...</span>
        </div>
      </div>
    );
  }

  // Determine spread quality
  const getSpreadQuality = (bps: number) => {
    if (bps < 5) return { color: 'green' as const, label: 'Excelente' };
    if (bps < 10) return { color: 'yellow' as const, label: 'Bueno' };
    return { color: 'red' as const, label: 'Alto' };
  };

  const spreadQuality = getSpreadQuality(metrics.currentSpread.bps);

  // Determine market activity level
  const getActivityLevel = (ticksPerHour: number) => {
    if (ticksPerHour > 100) return { color: 'green' as const, label: 'Alta' };
    if (ticksPerHour > 50) return { color: 'yellow' as const, label: 'Media' };
    return { color: 'red' as const, label: 'Baja' };
  };

  const activityLevel = getActivityLevel(metrics.activity.ticksPerHour);

  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Métricas de Mercado Reales
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Basadas en datos reales: Bid/Ask • OHLC • {data.length.toLocaleString()} registros
          </p>
        </div>

        {/* Data quality indicator */}
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1 text-sm ${
            metrics.activity.dataQuality > 0.95 ? 'text-emerald-400' : 'text-yellow-400'
          }`}>
            {metrics.activity.dataQuality > 0.95 ?
              <CheckCircle className="h-4 w-4" /> :
              <AlertCircle className="h-4 w-4" />
            }
            Calidad: {(metrics.activity.dataQuality * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Current Spread */}
        <MetricCard
          title="Spread Actual"
          value={formattedMetrics.spread.bps}
          subtitle={`${formattedMetrics.spread.absolute} COP • ${spreadQuality.label}`}
          icon={<Target className="h-4 w-4" />}
          color={spreadQuality.color}
          tooltip="Diferencia actual entre Ask y Bid en basis points"
        />

        {/* ATR (Volatilidad) */}
        <MetricCard
          title="ATR (14)"
          value={formattedMetrics.volatility.atr}
          subtitle="Average True Range"
          icon={<Activity className="h-4 w-4" />}
          color="purple"
          tooltip="Volatilidad promedio de las últimas 14 barras"
        />

        {/* Return Actual */}
        <MetricCard
          title="Return Actual"
          value={formattedMetrics.returns.current}
          subtitle="vs barra anterior"
          icon={metrics.returns.current >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
          color={metrics.returns.current >= 0 ? 'green' : 'red'}
          change={metrics.returns.current * 100}
          isGood={metrics.returns.current >= 0}
          tooltip="Cambio porcentual vs precio anterior"
        />

        {/* Session Progress */}
        <MetricCard
          title="Progreso Sesión"
          value={formattedMetrics.session.progress}
          subtitle={`${formattedMetrics.session.remaining} restantes`}
          icon={<Clock className="h-4 w-4" />}
          color={metrics.session.isMarketHours ? 'green' : 'gray'}
          tooltip="Progreso de la sesión de trading (8:00-12:55 COT)"
        />
      </div>

      {/* Advanced Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Volatility Analysis */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <h4 className="text-white font-medium mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Análisis de Volatilidad
          </h4>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Parkinson (anualizada)</span>
              <span className="text-sm font-mono text-white">{formattedMetrics.volatility.parkinson}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Garman-Klass (anualizada)</span>
              <span className="text-sm font-mono text-white">{formattedMetrics.volatility.garmanKlass}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Range Sesión</span>
              <span className="text-sm font-mono text-white">
                {metrics.priceAction.sessionRange.toFixed(2)} COP
              </span>
            </div>

            {/* Volatility quality indicator */}
            <div className="pt-2 border-t border-slate-600/30">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Nivel de volatilidad</span>
                <span className={`text-xs font-medium ${
                  metrics.volatility.parkinson > 0.25 ? 'text-red-400' :
                  metrics.volatility.parkinson > 0.15 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {metrics.volatility.parkinson > 0.25 ? 'Alta' :
                   metrics.volatility.parkinson > 0.15 ? 'Media' : 'Baja'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Market Activity */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <h4 className="text-white font-medium mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Actividad de Mercado
          </h4>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Ticks por Hora</span>
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono text-white">
                  {metrics.activity.ticksPerHour.toFixed(0)}
                </span>
                <span className={`text-xs px-2 py-1 rounded ${
                  activityLevel.color === 'green' ? 'bg-emerald-500/20 text-emerald-400' :
                  activityLevel.color === 'yellow' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  {activityLevel.label}
                </span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Spread Promedio</span>
              <span className="text-sm font-mono text-white">
                {metrics.activity.avgSpread.toFixed(2)} COP
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-400">Estabilidad Spread</span>
              <span className="text-sm font-mono text-white">
                {(metrics.activity.spreadStability * 100).toFixed(1)}%
              </span>
            </div>

            {/* Position in session range */}
            <div className="pt-2 border-t border-slate-600/30">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-500">Posición en Rango</span>
                <span className="text-xs text-slate-400">
                  {(metrics.priceAction.pricePosition * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="bg-cyan-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${metrics.priceAction.pricePosition * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Market Status Footer */}
      <div className="mt-6 pt-4 border-t border-slate-600/30">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                metrics.session.isMarketHours ? 'bg-emerald-500' : 'bg-slate-500'
              }`} />
              <span className="text-slate-400">
                {metrics.session.isMarketHours ? 'Mercado Abierto' : 'Mercado Cerrado'}
              </span>
            </div>

            <div className="text-slate-500">
              Horario: 8:00 - 12:55 COT
            </div>
          </div>

          <div className="text-slate-500">
            Última actualización: {currentTick?.timestamp.toLocaleTimeString() || 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
}