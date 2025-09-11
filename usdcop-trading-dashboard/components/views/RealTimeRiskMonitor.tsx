'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell
} from 'recharts';
import { realTimeRiskEngine, RealTimeRiskMetrics, Position } from '@/lib/services/real-time-risk-engine';
import {
  Activity, AlertTriangle, Shield, TrendingUp, TrendingDown, DollarSign,
  Target, Zap, BarChart3, PieChart as PieChartIcon, Gauge, Thermometer,
  ArrowUpRight, ArrowDownRight, Clock, RefreshCw, Settings, Maximize2,
  Radio, WifiOff, Wifi, AlertOctagon, CheckCircle, Bell
} from 'lucide-react';

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
    case 'HH:mm:ss':
      return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    case 'HH:mm':
      return `${pad(hours)}:${pad(minutes)}`;
    case 'PPpp':
      return `${day} de ${monthsFull[month]} de ${year} a las ${pad(hours)}:${pad(minutes)}`;
    default:
      return date.toLocaleDateString('es-ES');
  }
};

// Custom date manipulation functions
const subMinutes = (date: Date, minutes: number): Date => {
  return new Date(date.getTime() - minutes * 60000);
};

interface PortfolioSnapshot {
  timestamp: Date;
  portfolioValue: number;
  var95: number;
  leverage: number;
  drawdown: number;
  volatility: number;
  exposures: Record<string, number>;
}

interface RiskHeatmapData {
  position: string;
  var95: number;
  leverage: number;
  liquidity: number;
  concentration: number;
  riskScore: number;
  color: string;
}

interface MarketCondition {
  indicator: string;
  value: number;
  status: 'normal' | 'warning' | 'critical';
  change: number;
  description: string;
}

export default function RealTimeRiskMonitor() {
  const [riskMetrics, setRiskMetrics] = useState<RealTimeRiskMetrics | null>(null);
  const [portfolioHistory, setPortfolioHistory] = useState<PortfolioSnapshot[]>([]);
  const [riskHeatmap, setRiskHeatmap] = useState<RiskHeatmapData[]>([]);
  const [marketConditions, setMarketConditions] = useState<MarketCondition[]>([]);
  const [isConnected, setIsConnected] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [updateFrequency, setUpdateFrequency] = useState<5 | 10 | 30>(10); // seconds
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1H' | '4H' | '1D' | '1W'>('4H');
  const [alertsCount, setAlertsCount] = useState(0);
  const [loading, setLoading] = useState(true);
  
  // Refs for real-time updates
  const intervalRef = useRef<NodeJS.Timeout>();
  const metricsRef = useRef<RealTimeRiskMetrics | null>(null);

  // Mock positions for demonstration
  const mockPositions = useCallback((): Position[] => {
    return [
      {
        symbol: 'USDCOP_SPOT',
        quantity: 2000000,
        marketValue: 8500000,
        avgPrice: 4200,
        currentPrice: 4250,
        pnl: 100000,
        weight: 0.85,
        sector: 'FX',
        country: 'Colombia',
        currency: 'COP'
      },
      {
        symbol: 'COP_BONDS',
        quantity: 500000,
        marketValue: 1200000,
        avgPrice: 98.5,
        currentPrice: 99.2,
        pnl: 3500,
        weight: 0.12,
        sector: 'Fixed Income',
        country: 'Colombia',
        currency: 'COP'
      },
      {
        symbol: 'OIL_HEDGE',
        quantity: -50000,
        marketValue: -300000,
        avgPrice: 85.2,
        currentPrice: 84.8,
        pnl: 2000,
        weight: 0.03,
        sector: 'Commodities',
        country: 'Global',
        currency: 'USD'
      }
    ];
  }, []);

  const generateRiskHeatmap = useCallback((metrics: RealTimeRiskMetrics): RiskHeatmapData[] => {
    const positions = mockPositions();
    
    return positions.map(position => {
      // Calculate risk scores for each dimension
      const varScore = Math.random() * 100;
      const leverageScore = position.weight * 100;
      const liquidityScore = position.sector === 'FX' ? 90 : position.sector === 'Commodities' ? 70 : 80;
      const concentrationScore = position.weight * 100;
      
      // Overall risk score (weighted average)
      const riskScore = (varScore * 0.3 + leverageScore * 0.25 + (100 - liquidityScore) * 0.25 + concentrationScore * 0.20);
      
      // Color coding based on risk score
      let color = '#10B981'; // Green
      if (riskScore > 70) color = '#EF4444'; // Red
      else if (riskScore > 50) color = '#F59E0B'; // Amber
      else if (riskScore > 30) color = '#8B5CF6'; // Purple
      
      return {
        position: position.symbol,
        var95: varScore,
        leverage: leverageScore,
        liquidity: liquidityScore,
        concentration: concentrationScore,
        riskScore,
        color
      };
    });
  }, [mockPositions]);

  const generateMarketConditions = useCallback((): MarketCondition[] => {
    return [
      {
        indicator: 'VIX Index',
        value: 18.5,
        status: 'normal',
        change: -2.3,
        description: 'Market volatility within normal range'
      },
      {
        indicator: 'USD/COP Volatility',
        value: 24.2,
        status: 'warning',
        change: 5.8,
        description: 'Above average volatility in USDCOP'
      },
      {
        indicator: 'Credit Spreads',
        value: 145,
        status: 'normal',
        change: -3.2,
        description: 'Colombian spreads tightening'
      },
      {
        indicator: 'Oil Price',
        value: 84.7,
        status: 'critical',
        change: -12.4,
        description: 'Significant oil price decline affecting COP'
      },
      {
        indicator: 'Fed Policy',
        value: 5.25,
        status: 'normal',
        change: 0,
        description: 'Fed funds rate unchanged'
      },
      {
        indicator: 'EM Sentiment',
        value: 42.1,
        status: 'warning',
        change: -8.7,
        description: 'EM risk-off sentiment building'
      }
    ];
  }, []);

  const updateRiskMetrics = useCallback(async () => {
    try {
      // Simulate real-time data update
      const positions = mockPositions();
      
      // Update positions in risk engine
      positions.forEach(position => {
        realTimeRiskEngine.updatePosition(position);
      });
      
      // Get updated metrics
      const metrics = realTimeRiskEngine.getRiskMetrics();
      if (metrics) {
        setRiskMetrics(metrics);
        metricsRef.current = metrics;
        
        // Update portfolio history
        setPortfolioHistory(prev => {
          const newSnapshot: PortfolioSnapshot = {
            timestamp: new Date(),
            portfolioValue: metrics.portfolioValue,
            var95: metrics.portfolioVaR95,
            leverage: metrics.leverage,
            drawdown: metrics.currentDrawdown,
            volatility: metrics.portfolioVolatility,
            exposures: {
              'FX': 8500000,
              'Bonds': 1200000,
              'Commodities': 300000
            }
          };
          
          // Keep last 100 snapshots
          const updated = [...prev, newSnapshot];
          return updated.slice(-100);
        });
        
        // Update risk heatmap
        setRiskHeatmap(generateRiskHeatmap(metrics));
        
        // Update alerts count
        const alerts = realTimeRiskEngine.getAlerts(true); // unacknowledged only
        setAlertsCount(alerts.length);
      }
      
      // Update market conditions
      setMarketConditions(generateMarketConditions());
      
      setLastUpdate(new Date());
      setIsConnected(true);
      
    } catch (error) {
      console.error('Failed to update risk metrics:', error);
      setIsConnected(false);
    }
  }, [mockPositions, generateRiskHeatmap, generateMarketConditions]);

  useEffect(() => {
    const initialize = async () => {
      setLoading(true);
      await updateRiskMetrics();
      setLoading(false);
    };
    
    initialize();
    
    // Set up real-time updates
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    intervalRef.current = setInterval(updateRiskMetrics, updateFrequency * 1000);
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [updateRiskMetrics, updateFrequency]);

  // Subscribe to risk engine updates
  useEffect(() => {
    const handleRiskUpdate = (metrics: RealTimeRiskMetrics) => {
      setRiskMetrics(metrics);
      setLastUpdate(new Date());
    };
    
    realTimeRiskEngine.subscribeToUpdates(handleRiskUpdate);
    
    return () => {
      realTimeRiskEngine.unsubscribeFromUpdates(handleRiskUpdate);
    };
  }, []);

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

  const getRiskLevel = useCallback((value: number, thresholds: [number, number, number]) => {
    if (value <= thresholds[0]) return { level: 'Low', color: 'text-green-400', bg: 'bg-green-950' };
    if (value <= thresholds[1]) return { level: 'Medium', color: 'text-yellow-400', bg: 'bg-yellow-950' };
    if (value <= thresholds[2]) return { level: 'High', color: 'text-orange-400', bg: 'bg-orange-950' };
    return { level: 'Critical', color: 'text-red-400', bg: 'bg-red-950' };
  }, []);

  const getMarketConditionIcon = (status: string) => {
    switch (status) {
      case 'normal': return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-400" />;
      case 'critical': return <AlertOctagon className="h-4 w-4 text-red-400" />;
      default: return <Activity className="h-4 w-4 text-slate-400" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Initializing Real-Time Risk Monitor...</p>
        </div>
      </div>
    );
  }

  if (!riskMetrics) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-amber-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-bold text-amber-500 mb-2">Cargando Risk Monitor</h2>
          <p className="text-slate-400">Inicializando métricas de riesgo en tiempo real...</p>
        </div>
      </div>
    );
  }

  const leverageRisk = riskMetrics ? getRiskLevel(riskMetrics.leverage, [2, 3, 4]) : 'loading';
  const varRisk = riskMetrics ? getRiskLevel(riskMetrics.portfolioVaR95 / riskMetrics.portfolioValue, [0.02, 0.05, 0.08]) : 'loading';

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header with Connection Status */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono flex items-center gap-2">
            <Activity className="h-6 w-6" />
            REAL-TIME RISK MONITOR
          </h1>
          <div className="flex items-center gap-4 mt-1">
            <p className="text-slate-400 text-sm">
              Portfolio: {formatCurrency(riskMetrics.portfolioValue)} • 
              Last Update: {formatDate(lastUpdate, 'HH:mm:ss')}
            </p>
            <div className="flex items-center gap-2">
              {isConnected ? (
                <div className="flex items-center gap-1 text-green-400">
                  <Wifi className="h-4 w-4" />
                  <span className="text-sm">Live</span>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-red-400">
                  <WifiOff className="h-4 w-4" />
                  <span className="text-sm">Disconnected</span>
                </div>
              )}
              {alertsCount > 0 && (
                <div className="flex items-center gap-1 text-red-400">
                  <Bell className="h-4 w-4" />
                  <span className="text-sm">{alertsCount} alerts</span>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Update Frequency Selector */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-sm">Update:</span>
            <select
              value={updateFrequency}
              onChange={(e) => setUpdateFrequency(Number(e.target.value) as 5 | 10 | 30)}
              className="px-2 py-1 bg-slate-900 border border-amber-500/20 text-amber-500 rounded text-sm focus:outline-none"
            >
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={30}>30s</option>
            </select>
          </div>
          
          {/* Time Range Selector */}
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {(['1H', '4H', '1D', '1W'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setSelectedTimeRange(range)}
                className={`px-3 py-2 text-sm font-mono transition-colors ${
                  selectedTimeRange === range
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          
          <Button
            onClick={updateRiskMetrics}
            className="bg-slate-900 hover:bg-slate-800 text-amber-500 border-amber-500/20"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Critical Risk Alerts Banner */}
      {alertsCount > 0 && (
        <Alert className="border-red-500/30 bg-red-950/20">
          <AlertOctagon className="h-5 w-5 text-red-500" />
          <AlertDescription className="flex items-center justify-between">
            <div className="text-red-200">
              <strong>Risk Alert:</strong> {alertsCount} unacknowledged alert{alertsCount > 1 ? 's' : ''} require immediate attention.
            </div>
            <Button className="bg-red-900 hover:bg-red-800 text-red-100 text-sm">
              View Alerts
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Real-Time Risk Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Portfolio Overview */}
        <Card className="lg:col-span-2 bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Gauge className="h-5 w-5" />
              Portfolio Risk Dashboard
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              {/* Main Risk Metrics */}
              <div className="space-y-4">
                <div className="text-center p-4 bg-slate-800 rounded-lg">
                  <div className="text-slate-400 text-sm mb-1">Value at Risk (95%)</div>
                  <div className="text-2xl font-bold text-red-400 font-mono">
                    {formatCurrency(riskMetrics.portfolioVaR95)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {formatPercent(riskMetrics.portfolioVaR95 / riskMetrics.portfolioValue)} of portfolio
                  </div>
                  <Badge className={`${varRisk.bg} ${varRisk.color} text-xs mt-2`}>
                    {varRisk.level} Risk
                  </Badge>
                </div>
                
                <div className="text-center p-4 bg-slate-800 rounded-lg">
                  <div className="text-slate-400 text-sm mb-1">Portfolio Leverage</div>
                  <div className="text-2xl font-bold text-amber-400 font-mono">
                    {riskMetrics?.leverage?.toFixed(2) || '0.00'}x
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Gross: {formatCurrency(riskMetrics.grossExposure)}
                  </div>
                  <Badge className={`${leverageRisk.bg} ${leverageRisk.color} text-xs mt-2`}>
                    {leverageRisk.level} Risk
                  </Badge>
                </div>
              </div>
              
              {/* Risk Gauges */}
              <div className="space-y-4">
                <div className="bg-slate-800 p-4 rounded-lg">
                  <div className="text-slate-400 text-sm mb-2">Maximum Drawdown</div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <Progress value={Math.abs(riskMetrics.maximumDrawdown) * 100} className="h-3" />
                    </div>
                    <span className="text-red-400 font-mono text-sm">
                      {formatPercent(riskMetrics.maximumDrawdown)}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Current: {formatPercent(riskMetrics.currentDrawdown)}
                  </div>
                </div>
                
                <div className="bg-slate-800 p-4 rounded-lg">
                  <div className="text-slate-400 text-sm mb-2">Liquidity Score</div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <Progress value={riskMetrics.liquidityScore * 100} className="h-3" />
                    </div>
                    <span className="text-green-400 font-mono text-sm">
                      {((riskMetrics?.liquidityScore || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Liquidation: {riskMetrics?.timeToLiquidate?.toFixed(1) || '0.0'}d
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Market Conditions */}
        <Card className="lg:col-span-2 bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Thermometer className="h-5 w-5" />
              Market Conditions Monitor
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {marketConditions.map((condition, index) => (
                <div key={index} className="bg-slate-800 p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white text-sm font-semibold">{condition.indicator}</span>
                    {getMarketConditionIcon(condition.status)}
                  </div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xl font-mono text-white">{condition.value}</span>
                    <span className={`text-sm font-mono ${
                      condition.change >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {condition.change >= 0 ? '+' : ''}{condition?.change?.toFixed(1) || '0.0'}%
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">{condition.description}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Visualization Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Risk History */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Real-Time Risk Evolution</CardTitle>
            <p className="text-slate-400 text-sm">VaR 95% and Portfolio Leverage Over Time</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={portfolioHistory.slice(-50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#64748B" 
                  fontSize={10}
                  tickFormatter={(value) => formatDate(new Date(value), 'HH:mm')}
                />
                <YAxis yAxisId="left" stroke="#64748B" fontSize={10} />
                <YAxis yAxisId="right" orientation="right" stroke="#64748B" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  labelFormatter={(value) => formatDate(new Date(value), 'HH:mm:ss')}
                  formatter={(value: any, name) => [
                    name === 'var95' ? formatCurrency(value) : `${(value || 0).toFixed(2)}x`,
                    name === 'var95' ? 'VaR 95%' : 'Leverage'
                  ]}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="var95"
                  stroke="#EF4444"
                  fill="#EF4444"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="leverage"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Heatmap */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Position Risk Heatmap</CardTitle>
            <p className="text-slate-400 text-sm">Multi-Dimensional Risk Analysis</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {riskHeatmap.map((item, index) => (
                <div key={index} className="bg-slate-800 p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-semibold text-sm">{item.position}</span>
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: item.color }}
                    ></div>
                  </div>
                  
                  <div className="grid grid-cols-4 gap-2 text-xs">
                    <div className="text-center">
                      <div className="text-slate-400">VaR</div>
                      <div className="text-white font-mono">{item?.var95?.toFixed(0) || '0'}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400">Leverage</div>
                      <div className="text-white font-mono">{item?.leverage?.toFixed(0) || '0'}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400">Liquidity</div>
                      <div className="text-white font-mono">{item?.liquidity?.toFixed(0) || '0'}</div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400">Conc.</div>
                      <div className="text-white font-mono">{item?.concentration?.toFixed(0) || '0'}</div>
                    </div>
                  </div>
                  
                  <div className="mt-2">
                    <Progress value={item.riskScore} className="h-2" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stress Test & Scenario Analysis */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Real-Time Stress Testing
          </CardTitle>
          <p className="text-slate-400 text-sm">
            Live Scenario Impact Analysis • Tail Risk Assessment
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
            {Object.entries(riskMetrics?.stressTestResults || {}).slice(0, 4).map(([scenario, impact], index) => (
              <div key={index} className="bg-slate-800 p-4 rounded-lg text-center">
                <div className="text-slate-400 text-sm mb-2">{scenario}</div>
                <div className={`text-xl font-bold font-mono mb-1 ${
                  impact < 0 ? 'text-red-400' : 'text-green-400'
                }`}>
                  {formatCurrency(impact)}
                </div>
                <div className="text-xs text-slate-500">
                  {formatPercent(impact / riskMetrics.portfolioValue)}
                </div>
                <Progress 
                  value={Math.min(Math.abs(impact / riskMetrics.portfolioValue) * 1000, 100)} 
                  className="h-2 mt-2"
                />
              </div>
            ))}
          </div>
          
          {/* Monte Carlo Results */}
          <div className="mt-6">
            <h4 className="text-white font-semibold mb-3">Monte Carlo Simulation (1000 scenarios)</h4>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-slate-400 text-sm">Best Case</div>
                <div className="text-green-400 font-mono text-lg">
                  {formatCurrency(riskMetrics.bestCaseScenario)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-slate-400 text-sm">Expected Shortfall</div>
                <div className="text-orange-400 font-mono text-lg">
                  {formatCurrency(riskMetrics.expectedShortfall95)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-slate-400 text-sm">Worst Case</div>
                <div className="text-red-400 font-mono text-lg">
                  {formatCurrency(riskMetrics.worstCaseScenario)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-slate-400 text-sm">Max Loss Estimate</div>
                <div className="text-red-600 font-mono text-lg">
                  {formatCurrency(Math.min(riskMetrics.worstCaseScenario * 1.2, -riskMetrics.portfolioValue * 0.15))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Real-Time Risk Monitor • Update Frequency: {updateFrequency}s • 
          Professional Risk Management • Generated {formatDate(new Date(), 'PPpp')} • 
          Institutional Grade Risk Analytics
        </p>
      </div>
    </div>
  );
}