'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { realTimeRiskEngine, RealTimeRiskMetrics } from '@/lib/services/real-time-risk-engine';
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

interface MarketCondition {
  indicator: string;
  value: number;
  status: 'normal' | 'warning' | 'critical';
  change: number;
  description: string;
}

export default function RealTimeRiskMonitor() {
  const [riskMetrics, setRiskMetrics] = useState<RealTimeRiskMetrics | null>(null);
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

  const fetchMarketConditions = useCallback(async (): Promise<MarketCondition[]> => {
    try {
      // ✅ Use Next.js API proxy (no direct backend calls) to avoid CORS and caching issues
      const response = await fetch(`/api/analytics/market-conditions?symbol=USDCOP&days=30`);

      if (!response.ok) {
        throw new Error('Failed to fetch market conditions');
      }

      const data = await response.json();

      // Parse all numeric values to ensure they're numbers, not strings
      const conditions = (data.conditions || []).map((condition: any) => ({
        ...condition,
        value: typeof condition.value === 'number' ? condition.value : parseFloat(condition.value) || 0,
        change: typeof condition.change === 'number' ? condition.change : parseFloat(condition.change) || 0
      }));

      return conditions;
    } catch (error) {
      console.error('Error fetching market conditions:', error);
      // Return empty array on error - component will show "No data" message
      return [];
    }
  }, []);

  const updateRiskMetrics = useCallback(async () => {
    try {
      // Check if risk engine is available
      if (!realTimeRiskEngine) {
        console.warn('RealTimeRiskEngine not available');
        setIsConnected(false);
        return;
      }

      // Get updated metrics with error handling
      let metrics = null;
      if (typeof realTimeRiskEngine.getRiskMetrics === 'function') {
        metrics = realTimeRiskEngine.getRiskMetrics();
      }

      if (metrics) {
        setRiskMetrics(metrics);
        metricsRef.current = metrics;

        // Update alerts count with error handling
        try {
          if (typeof realTimeRiskEngine.getAlerts === 'function') {
            const alerts = realTimeRiskEngine.getAlerts(true); // unacknowledged only
            setAlertsCount(alerts.length);
          }
        } catch (alertsError) {
          console.error('Error getting alerts:', alertsError);
          setAlertsCount(0);
        }
      } else {
        console.warn('No risk metrics available from engine');
        setIsConnected(false);
      }

      // Fetch market conditions from API
      const conditions = await fetchMarketConditions();
      setMarketConditions(conditions);

      setLastUpdate(new Date());
      setIsConnected(true);

    } catch (error) {
      console.error('Failed to update risk metrics:', error);
      setIsConnected(false);
    }
  }, [fetchMarketConditions]);

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

    try {
      // Attempt to subscribe to updates with error handling
      if (realTimeRiskEngine && typeof realTimeRiskEngine.subscribeToUpdates === 'function') {
        realTimeRiskEngine.subscribeToUpdates(handleRiskUpdate);
        setIsConnected(true);
      } else {
        console.warn('RealTimeRiskEngine not available or subscribeToUpdates method missing');
        setIsConnected(false);
      }
    } catch (error) {
      console.error('Failed to subscribe to risk engine updates:', error);
      setIsConnected(false);
    }

    return () => {
      try {
        if (realTimeRiskEngine && typeof realTimeRiskEngine.unsubscribeFromUpdates === 'function') {
          realTimeRiskEngine.unsubscribeFromUpdates(handleRiskUpdate);
        }
      } catch (error) {
        console.error('Error unsubscribing from risk engine updates:', error);
      }
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
        <div className="text-center max-w-md">
          <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-red-500 mb-2">❌ No Risk Data Available</h2>
          <p className="text-slate-400 mb-4">Risk metrics cannot be displayed because Analytics API is not available.</p>
          <div className="bg-slate-900 border border-red-500/20 rounded-lg p-4 text-left">
            <p className="text-sm text-slate-300 mb-2"><strong>Required:</strong></p>
            <ul className="text-sm text-slate-400 space-y-1 list-disc list-inside">
              <li>Analytics API must be running on http://localhost:8001</li>
              <li>PostgreSQL database must have market data</li>
              <li>Check API logs for errors</li>
            </ul>
          </div>
          <div className="mt-4">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg"
            >
              Retry Connection
            </button>
          </div>
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
            {marketConditions.length === 0 ? (
              <div className="text-center py-8 text-slate-400">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                <p>No market conditions data available</p>
                <p className="text-xs mt-1">Check Analytics API connection</p>
              </div>
            ) : (
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
            )}
          </CardContent>
        </Card>
      </div>

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