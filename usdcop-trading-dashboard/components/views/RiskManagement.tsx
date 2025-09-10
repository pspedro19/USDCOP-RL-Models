'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, BarChart, Bar, ComposedChart, Cell, Heatmap, Legend
} from 'recharts';
import { metricsCalculator, HedgeFundMetrics } from '@/lib/services/hedge-fund-metrics';
import { minioClient } from '@/lib/services/minio-client';
import { realTimeRiskEngine, RealTimeRiskMetrics } from '@/lib/services/real-time-risk-engine';
import { 
  Shield, AlertTriangle, TrendingDown, DollarSign, Activity, 
  Zap, Target, BarChart3, Users, Clock, Gauge, AlertOctagon,
  Download, Settings, RefreshCw, Radio
} from 'lucide-react';

interface RiskMetrics {
  // VaR Metrics
  var95_1d: number;
  var99_1d: number;
  cvar95_1d: number;
  cvar99_1d: number;
  var95_10d: number;
  
  // Portfolio Risk
  portfolioValue: number;
  maxDrawdown: number;
  currentDrawdown: number;
  volatility: number;
  beta: number;
  correlation: number;
  
  // Position Risk
  totalExposure: number;
  netExposure: number;
  grossExposure: number;
  leverage: number;
  concentration: number;
  
  // Liquidity Risk
  liquidityScore: number;
  timeToLiquidate: number;
  marginUsed: number;
  marginAvailable: number;
  
  // Market Risk
  deltaEquivalent: number;
  gamma: number;
  vega: number;
  theta: number;
  
  // Credit Risk
  counterpartyRisk: number;
  settlementRisk: number;
}

interface StressScenario {
  name: string;
  description: string;
  shockMagnitude: number;
  portfolioImpact: number;
  probability: number;
  category: 'market' | 'credit' | 'liquidity' | 'operational';
}

interface CorrelationData {
  asset1: string;
  asset2: string;
  correlation: number;
  period: string;
}

interface RiskAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  recommendation?: string;
}

export default function RiskManagement() {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [stressScenarios, setStressScenarios] = useState<StressScenario[]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationData[]>([]);
  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([]);
  const [varHistory, setVarHistory] = useState<any[]>([]);
  const [drawdownData, setDrawdownData] = useState<any[]>([]);
  const [monteCarloResults, setMonteCarloResults] = useState<any[]>([]);
  const [positionSizing, setPositionSizing] = useState<any[]>([]);
  const [selectedTimeHorizon, setSelectedTimeHorizon] = useState<'1D' | '1W' | '1M' | '1Y'>('1D');
  const [confidenceLevel, setConfidenceLevel] = useState<95 | 99>(95);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Generate realistic risk data
  const generateRiskData = useCallback(() => {
    const basePortfolioValue = 10000000; // $10M portfolio
    
    // Generate VaR metrics using proper calculations
    const returns = Array.from({ length: 252 }, () => (Math.random() - 0.5) * 0.04); // Daily returns
    const var95 = Math.abs(metricsCalculator.calculateVaR(returns, 0.95)) * basePortfolioValue;
    const var99 = Math.abs(metricsCalculator.calculateVaR(returns, 0.99)) * basePortfolioValue;
    const cvar95 = Math.abs(metricsCalculator.calculateCVaR(returns, 0.95)) * basePortfolioValue;
    const cvar99 = Math.abs(metricsCalculator.calculateCVaR(returns, 0.99)) * basePortfolioValue;
    
    const mockRiskMetrics: RiskMetrics = {
      // VaR Metrics
      var95_1d: var95,
      var99_1d: var99,
      cvar95_1d: cvar95,
      cvar99_1d: cvar99,
      var95_10d: var95 * Math.sqrt(10), // Scaling rule
      
      // Portfolio Risk
      portfolioValue: basePortfolioValue,
      maxDrawdown: 0.087, // 8.7%
      currentDrawdown: 0.023, // 2.3%
      volatility: 0.145, // 14.5% annualized
      beta: 0.78,
      correlation: 0.65,
      
      // Position Risk
      totalExposure: basePortfolioValue * 1.25,
      netExposure: basePortfolioValue * 0.85,
      grossExposure: basePortfolioValue * 2.1,
      leverage: 2.1,
      concentration: 0.15, // 15% in largest position
      
      // Liquidity Risk
      liquidityScore: 0.82, // 0-1 scale
      timeToLiquidate: 2.5, // days
      marginUsed: 0.35, // 35%
      marginAvailable: 0.65, // 65%
      
      // Market Risk
      deltaEquivalent: basePortfolioValue * 0.92,
      gamma: 12500,
      vega: 8750,
      theta: -125,
      
      // Credit Risk
      counterpartyRisk: 0.02, // 2%
      settlementRisk: 0.005, // 0.5%
    };
    
    // Generate stress scenarios
    const scenarios: StressScenario[] = [
      {
        name: 'Central Bank Policy Shift',
        description: '200 bps interest rate increase by Banco de la República',
        shockMagnitude: 0.02,
        portfolioImpact: -0.065,
        probability: 0.15,
        category: 'market'
      },
      {
        name: 'Fed Hawkish Surprise',
        description: 'Unexpected 100 bps Fed rate hike',
        shockMagnitude: 0.01,
        portfolioImpact: -0.042,
        probability: 0.08,
        category: 'market'
      },
      {
        name: 'Commodity Price Shock',
        description: 'Oil price drops 40% affecting COP',
        shockMagnitude: -0.4,
        portfolioImpact: -0.078,
        probability: 0.12,
        category: 'market'
      },
      {
        name: 'Political Uncertainty',
        description: 'Colombian election volatility surge',
        shockMagnitude: 0.25,
        portfolioImpact: -0.055,
        probability: 0.20,
        category: 'market'
      },
      {
        name: 'Counterparty Default',
        description: 'Major broker insolvency',
        shockMagnitude: 0.5,
        portfolioImpact: -0.015,
        probability: 0.02,
        category: 'credit'
      },
      {
        name: 'Liquidity Crisis',
        description: 'Market freeze - no USDCOP trading',
        shockMagnitude: 0.8,
        portfolioImpact: -0.095,
        probability: 0.01,
        category: 'liquidity'
      }
    ];
    
    // Generate correlation matrix
    const assets = ['USDCOP', 'Oil WTI', 'Colombian Bonds', 'USD Index', 'EM Currencies', 'S&P 500'];
    const correlationData: CorrelationData[] = [];
    assets.forEach((asset1, i) => {
      assets.forEach((asset2, j) => {
        if (i !== j) {
          correlationData.push({
            asset1,
            asset2,
            correlation: asset1 === 'USDCOP' && asset2 === 'Oil WTI' ? -0.72 :
                        asset1 === 'USDCOP' && asset2 === 'Colombian Bonds' ? 0.85 :
                        asset1 === 'USDCOP' && asset2 === 'USD Index' ? 0.58 :
                        asset1 === 'USDCOP' && asset2 === 'EM Currencies' ? 0.67 :
                        asset1 === 'USDCOP' && asset2 === 'S&P 500' ? -0.31 :
                        Math.random() * 2 - 1, // Random for others
            period: '252D'
          });
        }
      });
    });
    
    // Generate risk alerts
    const alerts: RiskAlert[] = [
      {
        id: 'var-breach-001',
        severity: 'high',
        category: 'Market Risk',
        message: 'VaR 95% exceeded by 12% in last trading session',
        timestamp: subMinutes(new Date(), 15),
        acknowledged: false,
        recommendation: 'Reduce position size by 15% or implement hedging strategy'
      },
      {
        id: 'concentration-002',
        severity: 'medium',
        category: 'Concentration',
        message: 'Single position exceeds 15% portfolio limit',
        timestamp: subMinutes(new Date(), 45),
        acknowledged: false,
        recommendation: 'Divest 3% of position to maintain compliance'
      },
      {
        id: 'correlation-003',
        severity: 'medium',
        category: 'Correlation',
        message: 'Asset correlation spiked to 0.89 - diversification breakdown',
        timestamp: subMinutes(new Date(), 120),
        acknowledged: true,
        recommendation: 'Review portfolio diversification strategy'
      },
      {
        id: 'liquidity-004',
        severity: 'low',
        category: 'Liquidity',
        message: 'Market liquidity below normal - increased bid-ask spreads',
        timestamp: subMinutes(new Date(), 200),
        acknowledged: true
      }
    ];
    
    return { mockRiskMetrics, scenarios, correlationData, alerts };
  }, []);
  
  useEffect(() => {
    setLoading(true);
    
    const { mockRiskMetrics, scenarios, correlationData, alerts } = generateRiskData();
    setRiskMetrics(mockRiskMetrics);
    setStressScenarios(scenarios);
    setCorrelationMatrix(correlationData);
    setRiskAlerts(alerts);
    
    // Generate historical VaR data
    const varHistoryData = Array.from({ length: 30 }, (_, i) => {
      const date = subDays(new Date(), 29 - i);
      return {
        date: formatDate(date, 'MM/dd'),
        var95: mockRiskMetrics.var95_1d * (0.8 + Math.random() * 0.4),
        var99: mockRiskMetrics.var99_1d * (0.8 + Math.random() * 0.4),
        actual: mockRiskMetrics.var95_1d * (0.5 + Math.random() * 0.6),
        limit: mockRiskMetrics.var95_1d * 1.2
      };
    });
    setVarHistory(varHistoryData);
    
    // Generate drawdown data
    const drawdownData = Array.from({ length: 90 }, (_, i) => {
      const date = subDays(new Date(), 89 - i);
      return {
        date: formatDate(date, 'MM/dd'),
        drawdown: Math.random() * mockRiskMetrics.maxDrawdown * 100,
        underwater: Math.random() < 0.3 ? Math.random() * 5 : 0
      };
    });
    setDrawdownData(drawdownData);
    
    // Generate Monte Carlo simulation results
    const monteCarloData = Array.from({ length: 1000 }, (_, i) => ({
      scenario: i + 1,
      pnl: (Math.random() - 0.5) * mockRiskMetrics.portfolioValue * 0.1,
      probability: Math.random()
    })).sort((a, b) => a.pnl - b.pnl);
    setMonteCarloResults(monteCarloData);
    
    // Generate position sizing recommendations
    const positions = [
      { symbol: 'USDCOP Long', currentSize: 0.15, optimalSize: 0.12, kellySize: 0.08, risk: 'medium' },
      { symbol: 'Oil Hedge', currentSize: 0.05, optimalSize: 0.07, kellySize: 0.06, risk: 'low' },
      { symbol: 'Rate Spread', currentSize: 0.08, optimalSize: 0.06, kellySize: 0.04, risk: 'high' },
      { symbol: 'Volatility Play', currentSize: 0.03, optimalSize: 0.04, kellySize: 0.02, risk: 'medium' }
    ];
    setPositionSizing(positions);
    
    setLastUpdate(new Date());
    setLoading(false);
  }, [generateRiskData, selectedTimeHorizon, confidenceLevel]);
  
  const formatCurrency = useCallback((value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  }, []);
  
  const formatPercent = useCallback((value: number, decimals = 2) => {
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(decimals)}%`;
  }, []);
  
  const getRiskLevel = useCallback((value: number, thresholds: { low: number; medium: number; high: number }) => {
    if (value < thresholds.low) return { level: 'Low', color: 'text-green-400', bg: 'bg-green-950' };
    if (value < thresholds.medium) return { level: 'Medium', color: 'text-yellow-400', bg: 'bg-yellow-950' };
    if (value < thresholds.high) return { level: 'High', color: 'text-orange-400', bg: 'bg-orange-950' };
    return { level: 'Critical', color: 'text-red-400', bg: 'bg-red-950' };
  }, []);
  
  const getSeverityColor = useCallback((severity: string) => {
    switch (severity) {
      case 'low': return 'border-blue-400 bg-blue-950/30';
      case 'medium': return 'border-yellow-400 bg-yellow-950/30';
      case 'high': return 'border-orange-400 bg-orange-950/30';
      case 'critical': return 'border-red-400 bg-red-950/30';
      default: return 'border-slate-400 bg-slate-950/30';
    }
  }, []);
  
  // Custom date formatting function
  const formatDate = (date: Date, pattern: string): string => {
    const months = [
      'ene', 'feb', 'mar', 'abr', 'may', 'jun',
      'jul', 'ago', 'sep', 'oct', 'nov', 'dic'
    ];
    
    const monthsFull = [
      'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
      'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
    ];
    
    const day = date.getDate();
    const month = date.getMonth();
    const year = date.getFullYear();
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();
    const pad = (num: number) => num.toString().padStart(2, '0');
    
    switch (pattern) {
      case 'MM/dd':
        return `${pad(month + 1)}/${pad(day)}`;
      case 'HH:mm:ss':
        return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
      case 'HH:mm':
        return `${pad(hours)}:${pad(minutes)}`;
      case 'dd/MM':
        return `${pad(day)}/${pad(month + 1)}`;
      case 'yyyy-MM-dd HH:mm:ss':
        return `${year}-${pad(month + 1)}-${pad(day)} ${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
      case 'PPpp':
        return `${day} de ${monthsFull[month]} de ${year} a las ${pad(hours)}:${pad(minutes)}`;
      default:
        return date.toLocaleDateString('es-ES');
    }
  };
  
  // Custom date manipulation functions
  const subDays = (date: Date, days: number): Date => {
    return new Date(date.getTime() - days * 24 * 60 * 60 * 1000);
  };
  
  function subMinutes(date: Date, minutes: number): Date {
    return new Date(date.getTime() - minutes * 60000);
  }
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Professional Risk Analytics...</p>
        </div>
      </div>
    );
  }
  
  if (!riskMetrics) return null;
  
  const leverageRisk = getRiskLevel(riskMetrics.leverage, { low: 1.5, medium: 2.5, high: 4 });
  const concentrationRisk = getRiskLevel(riskMetrics.concentration, { low: 0.1, medium: 0.2, high: 0.3 });
  const liquidityRisk = getRiskLevel(1 - riskMetrics.liquidityScore, { low: 0.2, medium: 0.4, high: 0.6 });
  
  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono">PROFESSIONAL RISK MANAGEMENT</h1>
          <p className="text-slate-400 text-sm mt-1">
            Institutional Risk Analytics • Portfolio: {formatCurrency(riskMetrics.portfolioValue)} • 
            Last Update: {formatDate(lastUpdate, 'HH:mm:ss')}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {(['1D', '1W', '1M', '1Y'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setSelectedTimeHorizon(period)}
                className={`px-3 py-2 text-sm font-mono transition-colors ${
                  selectedTimeHorizon === period
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-2 bg-slate-900 border border-amber-500/20 text-amber-500 rounded-lg hover:bg-amber-500/10 transition-colors flex items-center gap-2 text-sm">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button className="px-3 py-2 bg-slate-900 border border-amber-500/20 text-amber-500 rounded-lg hover:bg-amber-500/10 transition-colors flex items-center gap-2 text-sm">
              <Download className="h-4 w-4" />
              Export
            </button>
          </div>
        </div>
      </div>
      
      {/* Key Risk Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              VaR {confidenceLevel}%
              <Shield className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono">
              {formatCurrency(confidenceLevel === 95 ? riskMetrics.var95_1d : riskMetrics.var99_1d)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              CVaR: {formatCurrency(confidenceLevel === 95 ? riskMetrics.cvar95_1d : riskMetrics.cvar99_1d)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Leverage
              <Activity className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 mb-2">
              <div className="text-xl font-bold text-white font-mono">{riskMetrics.leverage.toFixed(1)}x</div>
              <Badge className={`${leverageRisk.bg} ${leverageRisk.color} text-xs px-1`}>
                {leverageRisk.level}
              </Badge>
            </div>
            <Progress value={Math.min((riskMetrics.leverage / 5) * 100, 100)} className="h-2" />
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Max Drawdown
              <TrendingDown className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono">
              {formatPercent(riskMetrics.maxDrawdown)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Current: {formatPercent(riskMetrics.currentDrawdown)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Volatility
              <BarChart3 className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-amber-400 font-mono">
              {formatPercent(riskMetrics.volatility)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Beta: {riskMetrics.beta.toFixed(2)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Concentration
              <Target className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 mb-2">
              <div className="text-xl font-bold text-white font-mono">
                {formatPercent(riskMetrics.concentration, 1)}
              </div>
              <Badge className={`${concentrationRisk.bg} ${concentrationRisk.color} text-xs px-1`}>
                {concentrationRisk.level}
              </Badge>
            </div>
            <Progress value={riskMetrics.concentration * 100} className="h-2" />
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Liquidity
              <Zap className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 mb-2">
              <div className="text-xl font-bold text-green-400 font-mono">
                {(riskMetrics.liquidityScore * 100).toFixed(0)}%
              </div>
              <Badge className={`${liquidityRisk.bg} ${liquidityRisk.color} text-xs px-1`}>
                {liquidityRisk.level}
              </Badge>
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {riskMetrics.timeToLiquidate.toFixed(1)}d to exit
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Risk Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* VaR History */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">VaR Backtesting & Model Performance</CardTitle>
            <p className="text-slate-400 text-sm">Historical VaR vs Actual P&L • Model Validation</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={varHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="date" stroke="#64748B" fontSize={10} />
                <YAxis stroke="#64748B" fontSize={10} tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any, name: string) => [formatCurrency(value), name]}
                />
                <Bar dataKey="actual" fill="#10B981" name="Actual P&L" opacity={0.7} />
                <Line type="monotone" dataKey="var95" stroke="#EF4444" strokeWidth={2} dot={false} name="VaR 95%" />
                <Line type="monotone" dataKey="var99" stroke="#DC2626" strokeWidth={2} strokeDasharray="5 5" dot={false} name="VaR 99%" />
                <Line type="monotone" dataKey="limit" stroke="#F59E0B" strokeWidth={1} strokeDasharray="2 2" dot={false} name="Risk Limit" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* Monte Carlo Simulation */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Monte Carlo Risk Simulation</CardTitle>
            <p className="text-slate-400 text-sm">1,000 Scenarios • Tail Risk Distribution</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={monteCarloResults.slice(0, 100)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="scenario" stroke="#64748B" fontSize={10} />
                <YAxis stroke="#64748B" fontSize={10} tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any) => [formatCurrency(value), 'P&L']}
                />
                <Area 
                  type="monotone" 
                  dataKey="pnl" 
                  stroke="#8B5CF6" 
                  fill="#8B5CF6" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Stress Testing */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono">Advanced Stress Testing</CardTitle>
          <p className="text-slate-400 text-sm">Scenario Analysis • Tail Risk Assessment • Crisis Simulation</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {stressScenarios.map((scenario, index) => (
              <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-start mb-3">
                  <h4 className="font-semibold text-white text-sm">{scenario.name}</h4>
                  <Badge className={`${scenario.category === 'market' ? 'bg-blue-950 text-blue-400' :
                                     scenario.category === 'credit' ? 'bg-red-950 text-red-400' :
                                     scenario.category === 'liquidity' ? 'bg-yellow-950 text-yellow-400' :
                                     'bg-purple-950 text-purple-400'}`}>
                    {scenario.category}
                  </Badge>
                </div>
                <p className="text-slate-400 text-xs mb-3">{scenario.description}</p>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Portfolio Impact:</span>
                    <span className={`font-mono ${scenario.portfolioImpact < 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {formatPercent(scenario.portfolioImpact)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Probability:</span>
                    <span className="font-mono text-amber-400">{formatPercent(scenario.probability, 1)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Expected Loss:</span>
                    <span className="font-mono text-red-400">
                      {formatCurrency(riskMetrics.portfolioValue * scenario.portfolioImpact)}
                    </span>
                  </div>
                </div>
                <Progress 
                  value={Math.abs(scenario.portfolioImpact) * 1000} 
                  className="mt-3 h-2" 
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      
      {/* Position Sizing & Greeks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Position Sizing Recommendations */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Optimal Position Sizing</CardTitle>
            <p className="text-slate-400 text-sm">Kelly Criterion • Risk Parity • Capital Allocation</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {positionSizing.map((position, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      position.risk === 'low' ? 'bg-green-400' :
                      position.risk === 'medium' ? 'bg-yellow-400' : 'bg-red-400'
                    }`}></div>
                    <div>
                      <div className="text-white font-semibold text-sm">{position.symbol}</div>
                      <div className="text-slate-400 text-xs">Current: {formatPercent(position.currentSize)}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-amber-400 font-mono text-sm">
                      Optimal: {formatPercent(position.optimalSize)}
                    </div>
                    <div className="text-slate-400 font-mono text-xs">
                      Kelly: {formatPercent(position.kellySize)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Greeks Dashboard */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Risk Sensitivities (Greeks)</CardTitle>
            <p className="text-slate-400 text-sm">Portfolio Greeks • Risk Factor Sensitivities</p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-slate-800 rounded-lg">
                <div className="text-slate-400 text-xs mb-1">Delta Equivalent</div>
                <div className="text-white font-bold font-mono text-lg">
                  {formatCurrency(riskMetrics.deltaEquivalent)}
                </div>
                <div className="text-green-400 text-xs mt-1">92% of NAV</div>
              </div>
              <div className="text-center p-3 bg-slate-800 rounded-lg">
                <div className="text-slate-400 text-xs mb-1">Gamma</div>
                <div className="text-white font-bold font-mono text-lg">
                  {riskMetrics.gamma.toLocaleString()}
                </div>
                <div className="text-amber-400 text-xs mt-1">Per 1% move</div>
              </div>
              <div className="text-center p-3 bg-slate-800 rounded-lg">
                <div className="text-slate-400 text-xs mb-1">Vega</div>
                <div className="text-white font-bold font-mono text-lg">
                  {riskMetrics.vega.toLocaleString()}
                </div>
                <div className="text-blue-400 text-xs mt-1">Per vol point</div>
              </div>
              <div className="text-center p-3 bg-slate-800 rounded-lg">
                <div className="text-slate-400 text-xs mb-1">Theta</div>
                <div className="text-red-400 font-bold font-mono text-lg">
                  {riskMetrics.theta.toLocaleString()}
                </div>
                <div className="text-red-400 text-xs mt-1">Daily decay</div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-slate-800 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Credit & Settlement Risk</span>
                <div className="text-right">
                  <div className="text-amber-400 font-mono text-sm">
                    {formatPercent(riskMetrics.counterpartyRisk + riskMetrics.settlementRisk)}
                  </div>
                  <div className="text-slate-500 text-xs">
                    Counterparty: {formatPercent(riskMetrics.counterpartyRisk)} | Settlement: {formatPercent(riskMetrics.settlementRisk)}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Risk Alerts */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" />
            Real-Time Risk Alerts & Compliance Monitoring
          </CardTitle>
          <p className="text-slate-400 text-sm">Active Alerts: {riskAlerts.filter(a => !a.acknowledged).length} | Critical: {riskAlerts.filter(a => a.severity === 'critical').length}</p>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {riskAlerts.map((alert) => (
              <Alert key={alert.id} className={`${getSeverityColor(alert.severity)} ${alert.acknowledged ? 'opacity-60' : ''}`}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <AlertOctagon className={`h-4 w-4 mt-0.5 ${
                      alert.severity === 'critical' ? 'text-red-400' :
                      alert.severity === 'high' ? 'text-orange-400' :
                      alert.severity === 'medium' ? 'text-yellow-400' : 'text-blue-400'
                    }`} />
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-white text-sm">{alert.category}</span>
                        <Badge className={`text-xs ${
                          alert.severity === 'critical' ? 'bg-red-950 text-red-400' :
                          alert.severity === 'high' ? 'bg-orange-950 text-orange-400' :
                          alert.severity === 'medium' ? 'bg-yellow-950 text-yellow-400' : 'bg-blue-950 text-blue-400'
                        }`}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                        {alert.acknowledged && <Badge className="bg-green-950 text-green-400 text-xs">ACK</Badge>}
                      </div>
                      <AlertDescription className="text-slate-300 text-sm mb-2">
                        {alert.message}
                      </AlertDescription>
                      {alert.recommendation && (
                        <div className="text-amber-400 text-xs">
                          <strong>Recommendation:</strong> {alert.recommendation}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right text-slate-500">
                    <div className="text-xs">{formatDate(alert.timestamp, 'HH:mm')}</div>
                    <div className="text-xs">{formatDate(alert.timestamp, 'dd/MM')}</div>
                  </div>
                </div>
              </Alert>
            ))}
          </div>
        </CardContent>
      </Card>
      
      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Professional Risk Management • Generated {formatDate(new Date(), 'PPpp')} • 
          VaR Model: Historical Simulation (1Y) • Confidence Level: {confidenceLevel}% • 
          Compliance: Basel III Standards
        </p>
      </div>
    </div>
  );
}