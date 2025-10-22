'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import useSWR from 'swr';
import {
  Shield, AlertTriangle, TrendingDown, TrendingUp, Activity,
  Target, BarChart3, Clock, Zap, Globe, Gauge, Timer,
  AlertCircle, CheckCircle, XCircle, Eye, Settings,
  ArrowUp, ArrowDown, Minus, DollarSign, Percent,
  Flame, Snowflake, Wind, Mountain, Waves, Loader2
} from 'lucide-react';

// Fetcher for SWR
const fetcher = (url: string) => fetch(url).then((res) => res.json());

// Risk Management Data from Real API
const useRiskManagement = () => {
  const { data, error, isLoading } = useSWR(
    '/api/analytics/risk-metrics?symbol=USDCOP',
    fetcher,
    {
      refreshInterval: 5000, // Refresh every 5 seconds
      revalidateOnFocus: true,
      dedupingInterval: 2000
    }
  );

  // Default fallback structure when API is loading or has no data
  const defaultRiskData = {
    var: {
      var95: 0,
      var99: 0,
      cvar95: 0,
      cvar99: 0,
      method: 'Historical + Monte Carlo'
    },
    stressTests: {
      copDevaluation: {
        scenario: 'COP -20% Crisis',
        impact: 0,
        probability: 0.05,
        status: 'pass' as const
      },
      wtiShock: {
        scenario: 'WTI ±30% Oil Shock',
        impact: 0,
        probability: 0.12,
        status: 'pass' as const
      },
      dxyStrength: {
        scenario: 'DXY +10% USD Rally',
        impact: 0,
        probability: 0.18,
        status: 'pass' as const
      },
      costStress: {
        scenario: 'Costs +25% Operational',
        impact: 0,
        probability: 0.25,
        status: 'pass' as const
      },
      banrepIntervention: {
        scenario: 'Surprise BanRep Action',
        impact: 0,
        probability: 0.08,
        status: 'pass' as const
      }
    },
    dynamicMonitoring: {
      usdcopWtiCorr: 0,
      corrBreakAlert: false,
      drawdownRecovery: 0,
      implementationShortfall: 0,
      attribution: {
        trend: 0,
        microstructure: 0,
        costs: 0
      }
    },
    exposures: {
      netExposure: 0,
      grossExposure: 0,
      leverage: 0,
      concentrationRisk: 0,
      sectorExposure: {
        currency: 0,
        commodities: 0,
        rates: 0
      }
    },
    limits: {
      maxDrawdown: { limit: 15.0, current: 0, utilization: 0 },
      var95: { limit: 3.5, current: 0, utilization: 0 },
      leverage: { limit: 2.0, current: 0, utilization: 0 },
      concentration: { limit: 0.30, current: 0, utilization: 0 }
    },
    alerts: []
  };

  return {
    riskData: data || defaultRiskData,
    isLoading,
    error
  };
};

interface RiskMetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'optimal' | 'warning' | 'critical';
  icon: React.ComponentType<any>;
  limit?: string;
  utilization?: number;
  format?: 'currency' | 'percentage' | 'ratio' | 'number' | 'bps';
}

const RiskMetricCard: React.FC<RiskMetricCardProps> = ({ 
  title, value, subtitle, status, icon: Icon, limit, utilization, format = 'number' 
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'currency':
        return `$${val.toLocaleString()}`;
      case 'percentage':
        return `${val.toFixed(2)}%`;
      case 'ratio':
        return val.toFixed(3);
      case 'bps':
        return `${val.toFixed(1)} bps`;
      default:
        return val.toLocaleString();
    }
  };

  const statusColors = {
    optimal: 'border-market-up text-market-up shadow-market-up',
    warning: 'border-fintech-purple-400 text-fintech-purple-400 shadow-glow-purple',
    critical: 'border-market-down text-market-down shadow-market-down'
  };

  const bgColors = {
    optimal: 'from-market-up/10 to-market-up/5',
    warning: 'from-fintech-purple-400/10 to-fintech-purple-400/5',
    critical: 'from-market-down/10 to-market-down/5'
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-6 rounded-xl border ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        <div className={`w-2 h-2 rounded-full ${
          status === 'optimal' ? 'bg-market-up animate-pulse' :
          status === 'warning' ? 'bg-fintech-purple-400 animate-pulse' :
          'bg-market-down animate-pulse'
        }`} />
      </div>
      
      <div className="space-y-2">
        <div className="text-2xl font-bold text-white">
          {formatValue(value)}
        </div>
        <div className="text-xs text-fintech-dark-300">{subtitle}</div>
        
        {limit && (
          <div className="text-xs text-fintech-dark-400">
            Limit: <span className="font-medium">{limit}</span>
          </div>
        )}
        
        {utilization !== undefined && (
          <div className="mt-2">
            <div className="flex justify-between text-xs text-fintech-dark-300 mb-1">
              <span>Utilization</span>
              <span>{utilization.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-fintech-dark-800 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-500 ${
                  utilization >= 90 ? 'bg-market-down' :
                  utilization >= 75 ? 'bg-fintech-purple-400' :
                  'bg-market-up'
                }`}
                style={{ width: `${Math.min(100, utilization)}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const StressTestCard: React.FC<{
  title: string;
  scenario: string;
  impact: number;
  probability: number;
  status: 'pass' | 'fail';
  icon: React.ComponentType<any>;
}> = ({ title, scenario, impact, probability, status, icon: Icon }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-surface p-6 rounded-xl border ${
        status === 'pass' 
          ? 'border-market-up shadow-market-up bg-gradient-to-r from-market-up/10 to-market-up/5'
          : 'border-market-down shadow-market-down bg-gradient-to-r from-market-down/10 to-market-down/5'
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className={`w-5 h-5 ${status === 'pass' ? 'text-market-up' : 'text-market-down'}`} />
          <span className="text-sm font-medium text-white">{scenario}</span>
        </div>
        {status === 'pass' ? (
          <CheckCircle className="w-5 h-5 text-market-up" />
        ) : (
          <XCircle className="w-5 h-5 text-market-down" />
        )}
      </div>
      
      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-xs text-fintech-dark-300">Impact:</span>
          <span className={`text-sm font-bold ${impact < 0 ? 'text-market-down' : 'text-market-up'}`}>
            {impact > 0 ? '+' : ''}{impact.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-xs text-fintech-dark-300">Probability:</span>
          <span className="text-sm font-medium text-white">{(probability * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-xs text-fintech-dark-300">Status:</span>
          <span className={`text-sm font-bold ${status === 'pass' ? 'text-market-up' : 'text-market-down'}`}>
            {status.toUpperCase()}
          </span>
        </div>
      </div>
    </motion.div>
  );
};

const ExposureBreakdown: React.FC<{ data: any }> = ({ data }) => {
  const totalExposure = data.sectorExposure.currency + data.sectorExposure.commodities + data.sectorExposure.rates;
  
  return (
    <div className="glass-surface p-6 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Target className="w-5 h-5 text-fintech-cyan-500" />
        Portfolio Exposure Analysis
      </h3>
      
      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="flex justify-between">
            <span className="text-fintech-dark-300">Net Exposure:</span>
            <span className="text-white font-bold">${data.netExposure.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fintech-dark-300">Gross Exposure:</span>
            <span className="text-white font-bold">${data.grossExposure.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-fintech-dark-300">Leverage Ratio:</span>
            <span className={`font-bold ${data.leverage < 1.8 ? 'text-market-up' : data.leverage < 1.95 ? 'text-fintech-purple-400' : 'text-market-down'}`}>
              {data.leverage.toFixed(2)}x
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-fintech-dark-300">Concentration:</span>
            <span className={`font-bold ${data.concentrationRisk < 0.25 ? 'text-market-up' : 'text-fintech-purple-400'}`}>
              {(data.concentrationRisk * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        
        <div className="space-y-3">
          <div className="text-sm text-fintech-dark-300 mb-2">Sector Breakdown:</div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">Currency (USD/COP):</span>
              <span className="text-fintech-cyan-400 font-medium">{data.sectorExposure.currency.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-fintech-dark-800 rounded-full h-2">
              <div 
                className="h-2 bg-fintech-cyan-400 rounded-full"
                style={{ width: `${(data.sectorExposure.currency / totalExposure) * 100}%` }}
              />
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">Commodities (WTI):</span>
              <span className="text-fintech-purple-400 font-medium">{data.sectorExposure.commodities.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-fintech-dark-800 rounded-full h-2">
              <div 
                className="h-2 bg-fintech-purple-400 rounded-full"
                style={{ width: `${(data.sectorExposure.commodities / totalExposure) * 100}%` }}
              />
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">Interest Rates:</span>
              <span className="text-market-up font-medium">{data.sectorExposure.rates.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-fintech-dark-800 rounded-full h-2">
              <div 
                className="h-2 bg-market-up rounded-full"
                style={{ width: `${(data.sectorExposure.rates / totalExposure) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const AlertsPanel: React.FC<{ alerts: any[] }> = ({ alerts }) => {
  const [filter, setFilter] = useState<'all' | 'critical' | 'warning' | 'info'>('all');

  const filteredAlerts = alerts.filter(alert => 
    filter === 'all' || alert.type === filter
  );

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical': return AlertTriangle;
      case 'warning': return AlertCircle;
      case 'info': return Eye;
      default: return AlertCircle;
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical': return 'text-market-down border-market-down';
      case 'warning': return 'text-fintech-purple-400 border-fintech-purple-400';
      case 'info': return 'text-fintech-cyan-400 border-fintech-cyan-400';
      default: return 'text-fintech-dark-400 border-fintech-dark-400';
    }
  };

  return (
    <div className="glass-surface p-6 rounded-xl">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-fintech-purple-400" />
          Risk Alerts Center
        </h3>
        
        <div className="flex items-center gap-2">
          {['all', 'critical', 'warning', 'info'].map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type as any)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                filter === type
                  ? 'bg-fintech-cyan-500 text-white'
                  : 'bg-fintech-dark-800 text-fintech-dark-400 hover:text-white'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>
      
      <div className="space-y-3 max-h-64 overflow-y-auto">
        {filteredAlerts.map((alert) => {
          const AlertIcon = getAlertIcon(alert.type);
          const colorClass = getAlertColor(alert.type);
          
          return (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`p-3 rounded-lg border ${colorClass} ${
                alert.acknowledged ? 'opacity-60' : ''
              } bg-gradient-to-r from-fintech-dark-900/40 to-fintech-dark-800/40`}
            >
              <div className="flex items-start gap-3">
                <AlertIcon className="w-4 h-4 mt-0.5" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white">{alert.title}</div>
                  <div className="text-xs text-fintech-dark-300 mt-1">{alert.message}</div>
                  <div className="text-xs text-fintech-dark-400 mt-2">
                    {alert.timestamp.toLocaleString()}
                  </div>
                </div>
                {!alert.acknowledged && (
                  <div className="w-2 h-2 bg-fintech-cyan-400 rounded-full animate-pulse" />
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default function RiskManagement() {
  const { riskData, isLoading, error } = useRiskManagement();
  const [selectedStressTest, setSelectedStressTest] = useState<string | null>(null);

  // Show loading state
  if (isLoading && !riskData) {
    return (
      <div className="w-full bg-fintech-dark-950 p-6 flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-12 h-12 text-fintech-cyan-400 animate-spin" />
          <p className="text-fintech-dark-300">Loading risk metrics...</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="w-full bg-fintech-dark-950 p-6 flex items-center justify-center min-h-screen">
        <div className="glass-surface p-8 rounded-xl border border-market-down max-w-md">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-8 h-8 text-market-down" />
            <h2 className="text-xl font-bold text-white">Failed to Load Risk Metrics</h2>
          </div>
          <p className="text-fintech-dark-300 mb-4">
            Unable to connect to the analytics API. Please check that the backend service is running.
          </p>
          <pre className="text-xs text-fintech-dark-400 bg-fintech-dark-900 p-3 rounded overflow-auto">
            {error.message || 'Unknown error'}
          </pre>
        </div>
      </div>
    );
  }

  const varMetrics = [
    {
      title: 'VaR 95%',
      value: riskData.var.var95,
      subtitle: 'Historical + Monte Carlo',
      status: riskData.var.var95 <= 3.0 ? 'optimal' as const : riskData.var.var95 <= 3.5 ? 'warning' as const : 'critical' as const,
      icon: Shield,
      limit: '≤3.5%',
      utilization: (riskData.var.var95 / 3.5) * 100,
      format: 'percentage' as const
    },
    {
      title: 'CVaR 95%',
      value: riskData.var.cvar95,
      subtitle: 'Expected Shortfall',
      status: riskData.var.cvar95 <= 4.0 ? 'optimal' as const : riskData.var.cvar95 <= 5.0 ? 'warning' as const : 'critical' as const,
      icon: TrendingDown,
      limit: '≤5.0%',
      utilization: (riskData.var.cvar95 / 5.0) * 100,
      format: 'percentage' as const
    },
    {
      title: 'VaR 99%',
      value: riskData.var.var99,
      subtitle: 'Tail Risk Measure',
      status: riskData.var.var99 <= 5.0 ? 'optimal' as const : riskData.var.var99 <= 6.0 ? 'warning' as const : 'critical' as const,
      icon: AlertTriangle,
      limit: '≤6.0%',
      utilization: (riskData.var.var99 / 6.0) * 100,
      format: 'percentage' as const
    },
    {
      title: 'CVaR 99%',
      value: riskData.var.cvar99,
      subtitle: 'Extreme Loss Scenario',
      status: riskData.var.cvar99 <= 6.0 ? 'optimal' as const : riskData.var.cvar99 <= 8.0 ? 'warning' as const : 'critical' as const,
      icon: Flame,
      limit: '≤8.0%',
      utilization: (riskData.var.cvar99 / 8.0) * 100,
      format: 'percentage' as const
    }
  ];

  const dynamicMetrics = [
    {
      title: 'USD/COP ↔ WTI Correlation',
      value: riskData.dynamicMonitoring.usdcopWtiCorr,
      subtitle: 'Critical relationship monitoring',
      status: Math.abs(riskData.dynamicMonitoring.usdcopWtiCorr) > 0.6 ? 'optimal' as const : 'warning' as const,
      icon: Globe,
      format: 'ratio' as const
    },
    {
      title: 'Drawdown Recovery',
      value: riskData.dynamicMonitoring.drawdownRecovery,
      subtitle: 'Days to recovery (target <30)',
      status: riskData.dynamicMonitoring.drawdownRecovery <= 30 ? 'optimal' as const : 'warning' as const,
      icon: Clock,
      format: 'number' as const
    },
    {
      title: 'Implementation Shortfall',
      value: riskData.dynamicMonitoring.implementationShortfall,
      subtitle: 'Execution cost (target <10bps)',
      status: riskData.dynamicMonitoring.implementationShortfall <= 10 ? 'optimal' as const : 'warning' as const,
      icon: Target,
      format: 'bps' as const
    }
  ];

  const limitMetrics = [
    {
      title: 'Max Drawdown',
      value: riskData.limits.maxDrawdown.current,
      subtitle: 'Peak-to-trough decline',
      status: riskData.limits.maxDrawdown.utilization <= 80 ? 'optimal' as const : riskData.limits.maxDrawdown.utilization <= 95 ? 'warning' as const : 'critical' as const,
      icon: TrendingDown,
      limit: `${riskData.limits.maxDrawdown.limit}%`,
      utilization: riskData.limits.maxDrawdown.utilization,
      format: 'percentage' as const
    },
    {
      title: 'Leverage Ratio',
      value: riskData.limits.leverage.current,
      subtitle: 'Position sizing multiplier',
      status: riskData.limits.leverage.utilization <= 80 ? 'optimal' as const : riskData.limits.leverage.utilization <= 95 ? 'warning' as const : 'critical' as const,
      icon: Gauge,
      limit: `${riskData.limits.leverage.limit}x`,
      utilization: riskData.limits.leverage.utilization,
      format: 'ratio' as const
    }
  ];

  const stressTests = [
    {
      title: 'COP Crisis',
      ...riskData.stressTests.copDevaluation,
      icon: TrendingDown
    },
    {
      title: 'Oil Shock',
      ...riskData.stressTests.wtiShock,
      icon: Flame
    },
    {
      title: 'USD Strength',
      ...riskData.stressTests.dxyStrength,
      icon: TrendingUp
    },
    {
      title: 'Cost Stress',
      ...riskData.stressTests.costStress,
      icon: DollarSign
    },
    {
      title: 'BanRep Action',
      ...riskData.stressTests.banrepIntervention,
      icon: AlertCircle
    }
  ];

  return (
    <div className="w-full bg-fintech-dark-950 p-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Risk Management</h1>
            <p className="text-fintech-dark-300">
              USD/COP Specific Risk Metrics • VaR/CVaR • Stress Testing • Dynamic Monitoring
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="glass-surface px-4 py-2 rounded-xl border border-fintech-cyan-500 shadow-glow-cyan">
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-fintech-cyan-400" />
                <span className="text-fintech-cyan-400 font-medium">Risk Monitoring Active</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* VaR/CVaR Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
          <Shield className="w-6 h-6 text-fintech-cyan-500" />
          VaR/CVaR Analysis (Fat Tails)
        </h2>
        
        {/* Critical VaR Metrics - Always Visible */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-6">
          {varMetrics.slice(0, 3).map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + index * 0.05 }}
            >
              <RiskMetricCard {...metric} />
            </motion.div>
          ))}
        </div>
        
        {/* Secondary Tail Risk - Expandable */}
        <details className="group">
          <summary className="cursor-pointer list-none mb-4">
            <div className="flex items-center gap-3 text-fintech-dark-300 hover:text-white transition-colors">
              <span className="text-sm font-medium">Tail Risk Analysis</span>
              <div className="group-open:rotate-90 transition-transform">▶</div>
            </div>
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {varMetrics.slice(3).map((metric, index) => (
              <motion.div
                key={metric.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
              >
                <RiskMetricCard {...metric} />
              </motion.div>
            ))}
          </div>
        </details>
      </motion.div>

      {/* Stress Tests */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mb-8"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-market-down" />
          USD/COP Specific Stress Tests
        </h2>
        
        {/* Critical Stress Tests - Always Visible */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-6">
          {stressTests.slice(0, 3).map((test, index) => (
            <motion.div
              key={test.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
            >
              <StressTestCard {...test} />
            </motion.div>
          ))}
        </div>
        
        {/* Additional Stress Scenarios - Expandable */}
        <details className="group">
          <summary className="cursor-pointer list-none mb-4">
            <div className="flex items-center gap-3 text-fintech-dark-300 hover:text-white transition-colors">
              <span className="text-sm font-medium">Additional Stress Scenarios</span>
              <div className="group-open:rotate-90 transition-transform">▶</div>
            </div>
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {stressTests.slice(3).map((test, index) => (
              <motion.div
                key={test.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
              >
                <StressTestCard {...test} />
              </motion.div>
            ))}
          </div>
        </details>
      </motion.div>

      {/* Dynamic Monitoring & Limits */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
      >
        {/* Dynamic Monitoring */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-fintech-purple-400" />
            Dynamic Monitoring
          </h3>
          
          {dynamicMetrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
            >
              <RiskMetricCard {...metric} />
            </motion.div>
          ))}
          
          {/* P&L Attribution */}
          <div className="glass-surface p-4 rounded-xl">
            <h4 className="text-md font-semibold text-white mb-3">P&L Attribution</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Trend Capture:</span>
                <span className="text-market-up font-medium">{riskData.dynamicMonitoring.attribution.trend.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Microstructure:</span>
                <span className="text-fintech-cyan-400 font-medium">{riskData.dynamicMonitoring.attribution.microstructure.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Transaction Costs:</span>
                <span className="text-market-down font-medium">-{riskData.dynamicMonitoring.attribution.costs.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Limits */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Target className="w-5 h-5 text-fintech-cyan-400" />
            Risk Limits
          </h3>
          
          {limitMetrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
            >
              <RiskMetricCard {...metric} />
            </motion.div>
          ))}
          
          {/* Additional Risk Metrics */}
          <div className="glass-surface p-4 rounded-xl">
            <h4 className="text-md font-semibold text-white mb-3">Additional Metrics</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Correlation Break Alert:</span>
                <span className={`font-medium ${
                  riskData.dynamicMonitoring.corrBreakAlert ? 'text-market-down' : 'text-market-up'
                }`}>
                  {riskData.dynamicMonitoring.corrBreakAlert ? 'ACTIVE' : 'NORMAL'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Method:</span>
                <span className="text-white font-medium">{riskData.var.method}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-fintech-dark-300">Update Frequency:</span>
                <span className="text-fintech-cyan-400 font-medium">Real-time</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Exposure Analysis & Alerts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <ExposureBreakdown data={riskData.exposures} />
        <AlertsPanel alerts={riskData.alerts} />
      </motion.div>
    </div>
  );
}