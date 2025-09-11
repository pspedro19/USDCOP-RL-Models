'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, TrendingDown, Shield, AlertTriangle, CheckCircle, 
  XCircle, Clock, Target, BarChart3, Activity, Zap, Database,
  DollarSign, Percent, TrendingUp as TrendUp, LineChart
} from 'lucide-react';

// KPI Calculation Functions
const calculateSortinoRatio = (returns: number[], targetReturn: number = 0) => {
  const excessReturns = returns.map(r => r - targetReturn);
  const meanExcessReturn = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
  const downwardDeviations = excessReturns.filter(r => r < 0);
  const downsideDeviation = Math.sqrt(downwardDeviations.reduce((sum, r) => sum + r * r, 0) / downwardDeviations.length);
  return downsideDeviation > 0 ? (meanExcessReturn * Math.sqrt(252)) / (downsideDeviation * Math.sqrt(252)) : 0;
};

const calculateCalmarRatio = (cagr: number, maxDrawdown: number) => {
  return maxDrawdown > 0 ? Math.abs(cagr / maxDrawdown) : 0;
};

const calculateProfitFactor = (grossProfit: number, grossLoss: number) => {
  return grossLoss > 0 ? grossProfit / grossLoss : 0;
};

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'optimal' | 'warning' | 'critical';
  change?: number;
  target?: string;
  icon: React.ComponentType<any>;
  format?: 'currency' | 'percentage' | 'ratio' | 'number';
}

const KPICard: React.FC<KPICardProps> = ({ 
  title, value, subtitle, status, change, target, icon: Icon, format = 'number' 
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'currency':
        return `$${val.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
      case 'percentage':
        return `${val.toFixed(2)}%`;
      case 'ratio':
        return val.toFixed(3);
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card p-6 border-2 ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-gradient-to-r ${bgColors[status]} border ${statusColors[status]}`}>
            <Icon className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-fintech-dark-200">{title}</h3>
            <p className="text-xs text-fintech-dark-400">{subtitle}</p>
          </div>
        </div>
        {change !== undefined && (
          <div className={`flex items-center gap-1 ${change >= 0 ? 'text-market-up' : 'text-market-down'}`}>
            {change >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            <span className="text-sm font-medium">{change >= 0 ? '+' : ''}{change.toFixed(2)}%</span>
          </div>
        )}
      </div>
      
      <div className="space-y-2">
        <div className="text-3xl font-bold text-white">
          {formatValue(value)}
        </div>
        {target && (
          <div className="text-sm text-fintech-dark-300">
            Target: <span className="font-medium">{target}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const ProductionGateCard: React.FC<{ 
  title: string; 
  status: boolean; 
  value: string; 
  threshold: string;
  description: string;
}> = ({ title, status, value, threshold, description }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-4 rounded-xl border-2 ${
        status 
          ? 'border-market-up shadow-market-up bg-gradient-to-r from-market-up/10 to-market-up/5' 
          : 'border-market-down shadow-market-down bg-gradient-to-r from-market-down/10 to-market-down/5'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-white">{title}</h4>
        {status ? (
          <CheckCircle className="w-5 h-5 text-market-up" />
        ) : (
          <XCircle className="w-5 h-5 text-market-down" />
        )}
      </div>
      
      <div className="space-y-1">
        <div className="flex justify-between items-end">
          <span className="text-lg font-bold text-white">{value}</span>
          <span className="text-xs text-fintech-dark-300">vs {threshold}</span>
        </div>
        <p className="text-xs text-fintech-dark-400">{description}</p>
      </div>
    </motion.div>
  );
};

export default function ExecutiveOverview() {
  const [kpiData, setKpiData] = useState({
    sortinoRatio: 1.47,
    calmarRatio: 0.89,
    maxDrawdown: 12.3,
    profitFactor: 1.52,
    benchmarkSpread: 8.7,
    cagr: 18.4,
    sharpeRatio: 1.33,
    volatility: 11.8
  });

  const [productionGates, setProductionGates] = useState([
    { 
      title: 'Sortino Test', 
      status: true, 
      value: '1.47', 
      threshold: '≥1.3',
      description: 'Risk-adjusted returns vs downside deviation'
    },
    { 
      title: 'Max Drawdown', 
      status: true, 
      value: '12.3%', 
      threshold: '≤15%',
      description: 'Maximum peak-to-trough decline'
    },
    { 
      title: 'Calmar Ratio', 
      status: true, 
      value: '0.89', 
      threshold: '≥0.8',
      description: 'CAGR to Max Drawdown ratio'
    },
    { 
      title: 'Stress Test', 
      status: true, 
      value: '16.2%', 
      threshold: '≤20%',
      description: 'CAGR drop under +25% cost stress'
    },
    { 
      title: 'ONNX Latency', 
      status: true, 
      value: '15ms', 
      threshold: '<20ms',
      description: 'P99 inference latency'
    },
    { 
      title: 'E2E Latency', 
      status: true, 
      value: '87ms', 
      threshold: '<100ms',
      description: 'End-to-end execution latency'
    }
  ]);

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setKpiData(prev => ({
        ...prev,
        sortinoRatio: prev.sortinoRatio + (Math.random() - 0.5) * 0.02,
        maxDrawdown: prev.maxDrawdown + (Math.random() - 0.5) * 0.1,
        profitFactor: prev.profitFactor + (Math.random() - 0.5) * 0.01,
        benchmarkSpread: prev.benchmarkSpread + (Math.random() - 0.5) * 0.2,
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const kpiCards = [
    {
      title: 'Sortino Ratio',
      value: kpiData.sortinoRatio,
      subtitle: '30/90d Test Annualized',
      status: kpiData.sortinoRatio >= 1.5 ? 'optimal' as const : kpiData.sortinoRatio >= 1.3 ? 'warning' as const : 'critical' as const,
      change: 2.3,
      target: '≥1.3-1.5',
      icon: TrendUp,
      format: 'ratio' as const
    },
    {
      title: 'Calmar Ratio',
      value: kpiData.calmarRatio,
      subtitle: 'CAGR/|MaxDD|',
      status: kpiData.calmarRatio >= 0.9 ? 'optimal' as const : kpiData.calmarRatio >= 0.8 ? 'warning' as const : 'critical' as const,
      change: 1.8,
      target: '≥0.8',
      icon: Shield,
      format: 'ratio' as const
    },
    {
      title: 'Max Drawdown',
      value: kpiData.maxDrawdown,
      subtitle: 'Critical Gate',
      status: kpiData.maxDrawdown <= 12 ? 'optimal' as const : kpiData.maxDrawdown <= 15 ? 'warning' as const : 'critical' as const,
      change: -0.7,
      target: '≤15%',
      icon: AlertTriangle,
      format: 'percentage' as const
    },
    {
      title: 'Profit Factor',
      value: kpiData.profitFactor,
      subtitle: 'Gross P&L Ratio (Net)',
      status: kpiData.profitFactor >= 1.6 ? 'optimal' as const : kpiData.profitFactor >= 1.3 ? 'warning' as const : 'critical' as const,
      change: 3.1,
      target: '≥1.3-1.6',
      icon: DollarSign,
      format: 'ratio' as const
    },
    {
      title: 'Benchmark Spread',
      value: kpiData.benchmarkSpread,
      subtitle: 'vs CDT 12% E.A.',
      status: kpiData.benchmarkSpread > 5 ? 'optimal' as const : kpiData.benchmarkSpread > 0 ? 'warning' as const : 'critical' as const,
      change: 0.9,
      target: '>0%',
      icon: Target,
      format: 'percentage' as const
    },
    {
      title: 'CAGR',
      value: kpiData.cagr,
      subtitle: 'Compound Annual Growth',
      status: kpiData.cagr >= 20 ? 'optimal' as const : kpiData.cagr >= 15 ? 'warning' as const : 'critical' as const,
      change: 1.2,
      target: '>12%',
      icon: LineChart,
      format: 'percentage' as const
    }
  ];

  const allGatesPassing = productionGates.every(gate => gate.status);

  return (
    <div className="min-h-screen bg-fintech-dark-950 p-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Executive Overview</h1>
            <p className="text-fintech-dark-300">USD/COP Professional Trading Terminal - Performance Dashboard</p>
          </div>
          
          <div className={`glass-surface px-6 py-3 rounded-xl border-2 ${
            allGatesPassing 
              ? 'border-market-up text-market-up shadow-market-up' 
              : 'border-market-down text-market-down shadow-market-down'
          }`}>
            <div className="flex items-center gap-2">
              {allGatesPassing ? <CheckCircle className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
              <div>
                <div className="text-lg font-bold">
                  {allGatesPassing ? 'PRODUCTION READY' : 'REVIEW REQUIRED'}
                </div>
                <div className="text-sm opacity-80">
                  {productionGates.filter(g => g.status).length}/{productionGates.length} Gates Passing
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {kpiCards.map((kpi, index) => (
          <motion.div
            key={kpi.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <KPICard {...kpi} />
          </motion.div>
        ))}
      </div>

      {/* Production Gates */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mb-8"
      >
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Shield className="w-7 h-7 text-fintech-cyan-500" />
          Production Gates (✅/❌)
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {productionGates.map((gate, index) => (
            <motion.div
              key={gate.title}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + index * 0.1 }}
            >
              <ProductionGateCard {...gate} />
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Performance Summary Chart Placeholder */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass-card p-6"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
          <BarChart3 className="w-6 h-6 text-fintech-cyan-500" />
          Performance Overview (M5 → 72,576 periods/year)
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Metrics */}
          <div className="space-y-4">
            <div className="glass-surface p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-white mb-3">Key Metrics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-fintech-dark-300">Sharpe Ratio:</span>
                  <span className="text-white font-medium">{kpiData.sharpeRatio.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-fintech-dark-300">Volatility:</span>
                  <span className="text-white font-medium">{kpiData.volatility.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-fintech-dark-300">Max Periods:</span>
                  <span className="text-white font-medium">72,576/year</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-fintech-dark-300">Data Quality:</span>
                  <span className="text-market-up font-medium">Premium</span>
                </div>
              </div>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="space-y-4">
            <div className="glass-surface p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-white mb-3">System Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-fintech-dark-300">Model Training:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                    <span className="text-market-up font-medium">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-fintech-dark-300">Data Pipeline:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                    <span className="text-market-up font-medium">L0-L5 Healthy</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-fintech-dark-300">Risk Controls:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                    <span className="text-market-up font-medium">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-fintech-dark-300">Market Session:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-market-up rounded-full animate-pulse"></div>
                    <span className="text-market-up font-medium">Premium Hours</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}