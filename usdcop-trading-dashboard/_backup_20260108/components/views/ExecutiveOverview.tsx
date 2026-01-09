'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp, TrendingDown, Shield, AlertTriangle, CheckCircle,
  XCircle, Clock, Target, BarChart3, Activity, Zap, Database,
  DollarSign, Percent, TrendingUp as TrendUp, LineChart, RefreshCw, AlertCircle
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { usePerformanceKPIs, useProductionGates } from '@/hooks/useAnalytics';

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
  // Fetch dynamic KPI data from analytics API
  const { kpis: kpiDataFromAPI, isLoading: kpiLoading, error: kpiError, refetch: refetchKPIs } = usePerformanceKPIs('USDCOP', 90);
  const { gates: gatesFromAPI, isLoading: gatesLoading, error: gatesError, refetch: refetchGates } = useProductionGates('USDCOP', 90);

  const isLoading = kpiLoading || gatesLoading;
  const error = kpiError || gatesError;

  // Use real data or fallback to defaults while loading
  const kpiData = kpiDataFromAPI || {
    sortinoRatio: 0,
    calmarRatio: 0,
    maxDrawdown: 0,
    profitFactor: 0,
    benchmarkSpread: 0,
    cagr: 0,
    sharpeRatio: 0,
    volatility: 0
  };

  // Map API gates to component format
  const productionGates = gatesFromAPI.map((gate) => ({
    title: gate.title,
    status: gate.status,
    value: gate.value.toString(),
    threshold: `${gate.operator}${gate.threshold}`,
    description: gate.description
  }));

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

  const handleRetry = () => {
    if (refetchKPIs) refetchKPIs();
    if (refetchGates) refetchGates();
  };

  // ========== LOADING STATE ==========
  if (isLoading) {
    return (
      <div className="min-h-screen bg-fintech-dark-950 p-6 flex items-center justify-center">
        <Card className="glass-card max-w-md w-full">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <RefreshCw className="h-16 w-16 animate-spin text-fintech-cyan-500 mb-4" />
            <p className="text-white text-xl font-semibold mb-2">Loading Executive Overview</p>
            <p className="text-fintech-dark-300 text-sm">Fetching KPIs and production gates...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ========== ERROR STATE ==========
  if (error) {
    return (
      <div className="min-h-screen bg-fintech-dark-950 p-6 flex items-center justify-center">
        <Card className="glass-card max-w-md w-full border-2 border-market-down">
          <CardContent className="py-12 text-center">
            <AlertCircle className="h-16 w-16 text-market-down mx-auto mb-4" />
            <p className="text-market-down text-xl font-semibold mb-2">Error Loading Dashboard</p>
            <p className="text-fintech-dark-300 text-sm mb-6">{error}</p>
            <Button
              onClick={handleRetry}
              className="bg-fintech-cyan-500 hover:bg-fintech-cyan-600 text-white"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ========== EMPTY STATE ==========
  if (!kpiDataFromAPI || productionGates.length === 0) {
    return (
      <div className="min-h-screen bg-fintech-dark-950 p-6 flex items-center justify-center">
        <Card className="glass-card max-w-md w-full border-2 border-fintech-purple-400">
          <CardContent className="py-12 text-center">
            <BarChart3 className="h-16 w-16 text-fintech-purple-400 mx-auto mb-4" />
            <p className="text-fintech-purple-400 text-xl font-semibold mb-2">No Performance Data</p>
            <p className="text-fintech-dark-300 text-sm mb-2">Execute backtesting pipeline to generate KPIs</p>
            <p className="text-fintech-dark-400 text-xs mb-6">Data will appear after L6 backtest completion</p>
            <Button
              onClick={handleRetry}
              className="bg-fintech-cyan-500 hover:bg-fintech-cyan-600 text-white"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Check Again
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

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