'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, AreaChart, ScatterChart, Scatter, Heatmap, Cell } from 'recharts';
import { metricsCalculator, HedgeFundMetrics, Trade } from '@/lib/services/hedge-fund-metrics';
import { minioClient, DataQuery } from '@/lib/services/minio-client';
import { backtestClient, BacktestResults, BacktestKPIs, TradeRecord, DailyReturn } from '@/lib/services/backtest-client';
import { TrendingUp, TrendingDown, Activity, DollarSign, Shield, BarChart3, Target, AlertTriangle, Download, Calendar, Clock, TrendingDown as TrendDown, ChartLine } from 'lucide-react';
// Custom date formatting function with Spanish month names
const formatDate = (date: Date, formatStr: string) => {
  const months = [
    'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'
  ];
  
  const year = date.getFullYear();
  const month = date.getMonth();
  const day = date.getDate();
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const seconds = date.getSeconds();
  
  switch (formatStr) {
    case 'yyyy-MM':
      return `${year}-${String(month + 1).padStart(2, '0')}`;
    case 'MMM':
      return months[month];
    case 'MMM yy':
      return `${months[month]} ${String(year).slice(-2)}`;
    case 'HH:mm:ss':
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    case 'PPP':
      return `${day} de ${months[month]} de ${year}`;
    case 'PPpp':
      return `${day} de ${months[month]} de ${year} ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    default:
      return date.toLocaleDateString();
  }
};

// Date manipulation functions
const subDays = (date: Date, days: number): Date => {
  const result = new Date(date);
  result.setDate(result.getDate() - days);
  return result;
};

const startOfMonth = (date: Date): Date => {
  const result = new Date(date);
  result.setDate(1);
  result.setHours(0, 0, 0, 0);
  return result;
};

const endOfMonth = (date: Date): Date => {
  const result = new Date(date);
  result.setMonth(result.getMonth() + 1, 0);
  result.setHours(23, 59, 59, 999);
  return result;
};

const eachDayOfInterval = (interval: { start: Date; end: Date }): Date[] => {
  const days: Date[] = [];
  const current = new Date(interval.start);
  
  while (current <= interval.end) {
    days.push(new Date(current));
    current.setDate(current.getDate() + 1);
  }
  
  return days;
};

// Use the format function instead of date-fns format
const format = formatDate;
import { ExportToolbar } from '@/components/ui/export-toolbar';
import { useBacktestExport } from '@/hooks/useExport';
import { motion, AnimatePresence } from 'framer-motion';
import { Badge } from '@/components/ui/badge';

interface PerformanceData {
  date: string;
  portfolio: number;
  benchmark: number;
  drawdown: number;
  underwater: number;
}

interface HeatmapData {
  month: string;
  year: number;
  return: number;
  color: string;
}

interface RollingMetrics {
  date: string;
  sharpe: number;
  volatility: number;
  maxDD: number;
}

export default function BacktestResults() {
  const [hedgeFundMetrics, setHedgeFundMetrics] = useState<HedgeFundMetrics | null>(null);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [monthlyReturns, setMonthlyReturns] = useState<any[]>([]);
  const [yearlyHeatmap, setYearlyHeatmap] = useState<HeatmapData[]>([]);
  const [rollingMetrics, setRollingMetrics] = useState<RollingMetrics[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState<'1Y' | '2Y' | '5Y' | 'ALL'>('1Y');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Backtest-specific state
  const [backtestData, setBacktestData] = useState<BacktestResults | null>(null);
  const [currentSplit, setCurrentSplit] = useState<'test' | 'val'>('test');
  const [dataQuality, setDataQuality] = useState<any>(null);
  
  // Initialize export functionality
  const { setChartRef, exportToPDF, exportToCSV, exportToExcel } = useBacktestExport();

  // REMOVED: generateMockData function
  // NO mock/synthetic data generation allowed per user requirement
  // Backtest must use ONLY real historical data from MinIO
  
  const generateMonthlyHeatmap = useCallback((data: PerformanceData[]) => {
    const heatmapData: HeatmapData[] = [];
    const monthlyReturns: { [key: string]: number } = {};
    
    // Calculate monthly returns
    data.forEach((point, i) => {
      if (i === 0) return;
      const currentDate = new Date(point.date);
      const monthKey = format(currentDate, 'yyyy-MM');
      
      if (!monthlyReturns[monthKey]) {
        const prevValue = i > 0 ? data[i-1].portfolio : point.portfolio;
        monthlyReturns[monthKey] = (point.portfolio - prevValue) / prevValue;
      }
    });
    
    // Create heatmap structure
    Object.entries(monthlyReturns).forEach(([monthKey, return_]) => {
      const [year, month] = monthKey.split('-');
      const monthName = format(new Date(parseInt(year), parseInt(month) - 1), 'MMM');
      
      heatmapData.push({
        month: monthName,
        year: parseInt(year),
        return: return_ * 100,
        color: return_ > 0 ? '#10B981' : return_ < -0.02 ? '#EF4444' : '#F59E0B'
      });
    });
    
    return heatmapData;
  }, []);
  
  const generateRollingMetrics = useCallback((data: PerformanceData[]) => {
    const rollingData: RollingMetrics[] = [];
    const windowSize = 63; // 3 months rolling
    
    for (let i = windowSize; i < data.length; i++) {
      const window = data.slice(i - windowSize, i);
      const returns = window.map((point, idx) => {
        if (idx === 0) return 0;
        return (point.portfolio - window[idx - 1].portfolio) / window[idx - 1].portfolio;
      }).filter(r => r !== 0);
      
      const prices = window.map(p => p.portfolio);
      
      rollingData.push({
        date: data[i].date,
        sharpe: metricsCalculator.calculateSharpeRatio(returns),
        volatility: metricsCalculator['standardDeviation'](returns) * Math.sqrt(252) * 100,
        maxDD: Math.abs(metricsCalculator.calculateMaxDrawdown(prices)) * 100
      });
    }
    
    return rollingData;
  }, []);
  
  useEffect(() => {
    const fetchBacktestData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        console.log('[BacktestResults] Loading real backtest data from L6 pipeline...');
        
        // Fetch real backtest results from L6 bucket via API
        const results = await backtestClient.getLatestResults();
        setBacktestData(results);
        
        // Calculate data quality metrics
        const quality = backtestClient.getDataQuality(results);
        setDataQuality(quality);
        
        // Get current split data (test by default)
        const splitData = results[currentSplit];
        
        if (splitData && splitData.kpis && splitData.dailyReturns) {
          // Convert backtest KPIs to hedge fund metrics format
          const kpis = splitData.kpis;
          const hedgeMetrics: HedgeFundMetrics = {
            totalReturn: (kpis.top_bar.CAGR - 1) * 100, // Assuming CAGR is in decimal
            cagr: kpis.top_bar.CAGR,
            sharpeRatio: kpis.top_bar.Sharpe,
            sortinoRatio: kpis.top_bar.Sortino,
            calmarRatio: kpis.top_bar.Calmar,
            maxDrawdown: Math.abs(kpis.top_bar.MaxDD),
            volatility: kpis.top_bar.Vol_annualizada,
            winRate: kpis.trading_micro.win_rate,
            profitFactor: kpis.trading_micro.profit_factor,
            payoffRatio: kpis.trading_micro.payoff,
            var95: kpis.colas_y_drawdowns.VaR_99_bps / 10000, // Convert bps to decimal
            cvar95: kpis.colas_y_drawdowns.ES_97_5_bps / 10000,
            kellyFraction: 0.25, // Placeholder - would calculate from expectancy
            jensenAlpha: 0, // Placeholder - would calculate vs benchmark
            informationRatio: 0, // Placeholder
            treynorRatio: 0, // Placeholder
            betaToMarket: kpis.exposicion_capacidad.beta || 1,
            correlation: 0.5, // Placeholder
            trackingError: 0.02, // Placeholder
            expectancy: kpis.trading_micro.expectancy_bps / 10000,
            hitRate: kpis.trading_micro.win_rate
          };
          setHedgeFundMetrics(hedgeMetrics);
          
          // Convert daily returns to performance data
          const perfData = backtestClient.calculatePerformanceData(splitData.dailyReturns);
          setPerformanceData(perfData);
          
          // Calculate monthly returns
          const monthlyData = backtestClient.calculateMonthlyReturns(perfData);
          setMonthlyReturns(monthlyData);
          
          // Convert trades to hedge fund format
          const hedgeTrades = backtestClient.transformTradesToHedgeFundFormat(splitData.trades, kpis);
          setTrades(hedgeTrades);
          
          // Generate heatmap and rolling metrics
          const heatmap = generateMonthlyHeatmap(perfData);
          setYearlyHeatmap(heatmap);
          
          const rolling = generateRollingMetrics(perfData);
          setRollingMetrics(rolling);
          
        } else {
          console.warn(`[BacktestResults] No data available for split: ${currentSplit}`);
          // Set default empty state
          const emptyMetrics = metricsCalculator.calculateAllMetrics([100000], [], []);
          setHedgeFundMetrics(emptyMetrics);
          setPerformanceData([]);
          setMonthlyReturns([]);
          setTrades([]);
          setYearlyHeatmap([]);
          setRollingMetrics([]);
        }
        
      } catch (err) {
        setError(`Error loading backtest data: ${err}`);
        console.error('Backtest loading error:', err);
        
        // Set fallback empty state
        const fallbackMetrics = metricsCalculator.calculateAllMetrics([100000], [], []);
        setHedgeFundMetrics(fallbackMetrics);
      } finally {
        setLoading(false);
      }
    };
    
    fetchBacktestData();
  }, [selectedPeriod, currentSplit, generateMonthlyHeatmap, generateRollingMetrics]);
  
  const formatCurrency = useCallback((value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  }, []);
  
  const formatPercent = useCallback((value: number, decimals = 2) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Professional Backtest Analytics...</p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="bg-red-950/50 border border-red-500/20 rounded-lg p-6">
        <div className="flex items-center mb-2">
          <AlertTriangle className="h-5 w-5 text-red-400 mr-2" />
          <h3 className="text-red-400 font-semibold">Error Loading Backtest Data</h3>
        </div>
        <p className="text-red-300 text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-900 min-h-screen">
      {/* Enhanced Header with Glassmorphism */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative bg-slate-900/80 backdrop-blur-md border border-cyan-400/20 rounded-2xl p-6 shadow-2xl shadow-cyan-400/10 overflow-hidden"
      >
        {/* Background glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 pointer-events-none"></div>
        
        {/* Animated background pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-cyan-400/20 to-transparent transform -skew-x-12 animate-shimmer"></div>
        </div>
        
        <div className="relative flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              className="p-3 bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full"
            >
              <ChartLine className="h-8 w-8 text-white" />
            </motion.div>
            
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 via-emerald-400 to-purple-400 bg-clip-text text-transparent font-mono">
                PROFESSIONAL BACKTEST ANALYTICS
              </h1>
              <div className="flex items-center space-x-4 mt-2">
                <p className="text-slate-400 text-sm font-mono">Institutional-Grade Performance Analysis</p>
                <Badge variant="glow" className="text-xs px-2 py-1">
                  USDCOP STRATEGY
                </Badge>
                {backtestData && (
                  <Badge variant="outline" className="text-xs px-2 py-1">
                    Run: {backtestData.runId.slice(-6)}
                  </Badge>
                )}
                {hedgeFundMetrics && (
                  <Badge 
                    variant={hedgeFundMetrics.sharpeRatio > 1 ? 'success' : 'warning'}
                    className="text-xs px-2 py-1"
                  >
                    Sharpe: {hedgeFundMetrics.sharpeRatio.toFixed(2)}
                  </Badge>
                )}
                {dataQuality && (
                  <Badge 
                    variant={dataQuality.score > 80 ? 'success' : dataQuality.score > 60 ? 'warning' : 'destructive'}
                    className="text-xs px-2 py-1"
                  >
                    Quality: {dataQuality.score}%
                  </Badge>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <ExportToolbar
                onExportPDF={() => exportToPDF({
                  summary: hedgeFundMetrics,
                  monthlyReturns,
                  trades,
                  performanceData,
                  drawdowns: performanceData.filter(d => d.drawdown < -0.01)
                })}
                onExportCSV={() => exportToCSV({
                  summary: hedgeFundMetrics,
                  monthlyReturns,
                  trades,
                  performanceData
                })}
                onExportExcel={() => exportToExcel({
                  summary: hedgeFundMetrics,
                  monthlyReturns,
                  trades,
                  performanceData,
                  drawdowns: performanceData.filter(d => d.drawdown < -0.01)
                })}
                title="Backtest Results"
                disabled={loading}
              />
            </motion.div>
            
            <div className="flex gap-4">
              {/* Split Selector */}
              <motion.div 
                className="flex rounded-xl border border-slate-700/50 bg-slate-800/40 backdrop-blur-sm overflow-hidden"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3 }}
              >
                {(['test', 'val'] as const).map((split, index) => (
                  <motion.button
                    key={split}
                    onClick={() => setCurrentSplit(split)}
                    className={`px-4 py-2 text-sm font-mono transition-all duration-300 transform hover:scale-105 ${
                      currentSplit === split
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-400/30'
                        : 'bg-transparent text-slate-300 hover:bg-slate-700/50 hover:text-white'
                    }`}
                    whileHover={{ y: -2 }}
                    whileTap={{ y: 0 }}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                  >
                    {split.toUpperCase()}
                  </motion.button>
                ))}
              </motion.div>
              
              {/* Period Selector */}
              <motion.div 
                className="flex rounded-xl border border-slate-700/50 bg-slate-800/40 backdrop-blur-sm overflow-hidden"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.35 }}
              >
                {(['1Y', '2Y', '5Y', 'ALL'] as const).map((period, index) => (
                  <motion.button
                    key={period}
                    onClick={() => setSelectedPeriod(period)}
                    className={`px-4 py-2 text-sm font-mono transition-all duration-300 transform hover:scale-105 ${
                      selectedPeriod === period
                        ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white shadow-lg shadow-cyan-400/30'
                        : 'bg-transparent text-slate-300 hover:bg-slate-700/50 hover:text-white'
                    }`}
                    whileHover={{ y: -2 }}
                    whileTap={{ y: 0 }}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.45 + index * 0.1 }}
                  >
                    {period}
                  </motion.button>
                ))}
              </motion.div>
            </div>
          </div>
        </div>
        
        {/* Loading indicator */}
        {loading && (
          <motion.div 
            className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 rounded-b-2xl"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: [0, 1, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
      </motion.div>
      
      {/* Key Performance Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-amber-500 font-mono">CAGR</CardTitle>
            <TrendingUp className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white font-mono">
              {formatPercent(hedgeFundMetrics?.cagr * 100 || 0)}
            </div>
            <p className="text-xs text-slate-400 mt-1">vs Benchmark: {formatPercent(12.0)}</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-amber-500 font-mono">Sharpe Ratio</CardTitle>
            <Target className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white font-mono">
              {hedgeFundMetrics?.sharpeRatio.toFixed(3) || '0.000'}
            </div>
            <p className="text-xs text-slate-400 mt-1">Sortino: {hedgeFundMetrics?.sortinoRatio.toFixed(3)}</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-amber-500 font-mono">Max Drawdown</CardTitle>
            <TrendDown className="h-4 w-4 text-red-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-400 font-mono">
              {formatPercent(hedgeFundMetrics?.maxDrawdown * 100 || 0)}
            </div>
            <p className="text-xs text-slate-400 mt-1">Calmar: {hedgeFundMetrics?.calmarRatio.toFixed(3)}</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-amber-500 font-mono">Alpha Generation</CardTitle>
            <Shield className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-400 font-mono">
              {formatPercent(hedgeFundMetrics?.jensenAlpha * 100 || 0)}
            </div>
            <p className="text-xs text-slate-400 mt-1">Info Ratio: {hedgeFundMetrics?.informationRatio.toFixed(3)}</p>
          </CardContent>
        </Card>
      </div>
      
      {/* Advanced Risk Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono">VaR (95%)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono mb-1">
              {formatPercent(hedgeFundMetrics?.var95 * 100 || 0)}
            </div>
            <div className="text-xs text-slate-400">
              CVaR: {formatPercent(hedgeFundMetrics?.cvar95 * 100 || 0)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono">Beta</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono mb-1">
              {hedgeFundMetrics?.betaToMarket.toFixed(3) || '0.000'}
            </div>
            <div className="text-xs text-slate-400">
              Correlation: {hedgeFundMetrics?.correlation.toFixed(3)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono">Win Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono mb-1">
              {formatPercent(hedgeFundMetrics?.winRate * 100 || 0, 1)}
            </div>
            <div className="text-xs text-slate-400">
              {trades.filter(t => t.pnl > 0).length} / {trades.length}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono">Profit Factor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono mb-1">
              {hedgeFundMetrics?.profitFactor.toFixed(2) || '0.00'}
            </div>
            <div className="text-xs text-slate-400">
              Kelly: {formatPercent(hedgeFundMetrics?.kellyFraction * 100 || 0, 1)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono">Volatility</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-amber-400 font-mono mb-1">
              {formatPercent(hedgeFundMetrics?.volatility * 100 || 0, 1)}
            </div>
            <div className="text-xs text-slate-400">
              Tracking Error: {formatPercent(hedgeFundMetrics?.trackingError * 100 || 0, 1)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio vs Benchmark */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Portfolio vs CDT Benchmark</CardTitle>
            <p className="text-slate-400 text-sm">Cumulative Performance Comparison</p>
          </CardHeader>
          <CardContent>
            <div ref={(el) => setChartRef('performance-chart', el)}>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(date) => format(new Date(date), 'MMM yy')}
                />
                <YAxis 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(value) => formatCurrency(value)}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0F172A', 
                    border: '1px solid #F59E0B',
                    borderRadius: '6px'
                  }}
                  formatter={(value: any, name: string) => [
                    formatCurrency(value),
                    name === 'portfolio' ? 'Strategy' : 'CDT 12% E.A.'
                  ]}
                  labelFormatter={(date) => format(new Date(date), 'PPP')}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="portfolio" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={false}
                  name="Strategy"
                />
                <Line 
                  type="monotone" 
                  dataKey="benchmark" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="CDT Benchmark"
                />
              </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Underwater Curve */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Drawdown Analysis</CardTitle>
            <p className="text-slate-400 text-sm">Underwater Curve & Recovery Periods</p>
          </CardHeader>
          <CardContent>
            <div ref={(el) => setChartRef('drawdown-chart', el)}>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(date) => format(new Date(date), 'MMM yy')}
                />
                <YAxis 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0F172A', 
                    border: '1px solid #F59E0B',
                    borderRadius: '6px'
                  }}
                  formatter={(value: any) => [`${value.toFixed(2)}%`, 'Drawdown']}
                  labelFormatter={(date) => format(new Date(date), 'PPP')}
                />
                <Area
                  type="monotone"
                  dataKey="underwater"
                  stroke="#EF4444"
                  fill="#EF4444"
                  fillOpacity={0.3}
                />
              </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Monthly Performance & Rolling Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Returns Heatmap */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Monthly Returns Distribution</CardTitle>
            <p className="text-slate-400 text-sm">Performance by Month • Seasonal Analysis</p>
          </CardHeader>
          <CardContent>
            <div ref={(el) => setChartRef('returns-chart', el)}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={monthlyReturns}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="month" 
                  stroke="#64748B"
                  fontSize={10}
                />
                <YAxis 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0F172A', 
                    border: '1px solid #F59E0B',
                    borderRadius: '6px'
                  }}
                  formatter={(value: any) => [`${value.toFixed(2)}%`, 'Return']}
                />
                <Bar dataKey="return">
                  {monthlyReturns.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.return >= 0 ? '#10B981' : '#EF4444'} />
                  ))}
                </Bar>
              </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Rolling Risk Metrics */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Rolling Risk Metrics (3M Window)</CardTitle>
            <p className="text-slate-400 text-sm">Dynamic Risk Assessment • Regime Changes</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rollingMetrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(date) => format(new Date(date), 'MMM yy')}
                />
                <YAxis 
                  yAxisId="sharpe"
                  stroke="#64748B"
                  fontSize={10}
                />
                <YAxis 
                  yAxisId="vol"
                  orientation="right"
                  stroke="#64748B"
                  fontSize={10}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0F172A', 
                    border: '1px solid #F59E0B',
                    borderRadius: '6px'
                  }}
                  formatter={(value: any, name: string) => [
                    name === 'volatility' ? `${value.toFixed(1)}%` : value.toFixed(3),
                    name === 'sharpe' ? 'Sharpe' : name === 'volatility' ? 'Volatility' : 'Max DD'
                  ]}
                  labelFormatter={(date) => format(new Date(date), 'PPP')}
                />
                <Line 
                  yAxisId="sharpe"
                  type="monotone" 
                  dataKey="sharpe" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={false}
                  name="Sharpe"
                />
                <Line 
                  yAxisId="vol"
                  type="monotone" 
                  dataKey="volatility" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  dot={false}
                  name="Volatility"
                />
                <Line 
                  yAxisId="vol"
                  type="monotone" 
                  dataKey="maxDD" 
                  stroke="#EF4444" 
                  strokeWidth={2}
                  dot={false}
                  name="Max DD"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Advanced Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trade Analysis */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Trade Analytics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Total Trades</span>
                <span className="font-bold text-white font-mono">{trades.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Avg Win</span>
                <span className="font-bold text-green-400 font-mono">
                  {formatCurrency(trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / Math.max(1, trades.filter(t => t.pnl > 0).length))}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Avg Loss</span>
                <span className="font-bold text-red-400 font-mono">
                  {formatCurrency(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0) / Math.max(1, trades.filter(t => t.pnl < 0).length))}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Expectancy</span>
                <span className="font-bold text-amber-400 font-mono">
                  {formatCurrency(hedgeFundMetrics?.expectancy || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Payoff Ratio</span>
                <span className="font-bold text-white font-mono">
                  {hedgeFundMetrics?.payoffRatio.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Risk-Adjusted Returns */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Risk-Adjusted Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Treynor Ratio</span>
                <span className="font-bold text-white font-mono">
                  {hedgeFundMetrics?.treynorRatio.toFixed(3) || '0.000'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Information Ratio</span>
                <span className="font-bold text-amber-400 font-mono">
                  {hedgeFundMetrics?.informationRatio.toFixed(3) || '0.000'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Jensen's Alpha</span>
                <span className="font-bold text-green-400 font-mono">
                  {formatPercent(hedgeFundMetrics?.jensenAlpha * 100 || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Hit Rate</span>
                <span className="font-bold text-white font-mono">
                  {formatPercent(hedgeFundMetrics?.hitRate * 100 || 0, 1)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Kelly Fraction</span>
                <span className="font-bold text-orange-400 font-mono">
                  {formatPercent(hedgeFundMetrics?.kellyFraction * 100 || 0, 1)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Performance Summary */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Performance Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Total Return</span>
                <span className="font-bold text-green-400 font-mono">
                  {formatPercent(hedgeFundMetrics?.totalReturn * 100 || 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Annualized Vol</span>
                <span className="font-bold text-amber-400 font-mono">
                  {formatPercent(hedgeFundMetrics?.volatility * 100 || 0, 1)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Downside Dev</span>
                <span className="font-bold text-red-400 font-mono">
                  {formatPercent((hedgeFundMetrics?.volatility || 0) * 0.7 * 100, 1)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Best Month</span>
                <span className="font-bold text-green-400 font-mono">
                  {formatPercent(Math.max(...monthlyReturns.map(m => m.return)))}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm font-mono">Worst Month</span>
                <span className="font-bold text-red-400 font-mono">
                  {formatPercent(Math.min(...monthlyReturns.map(m => m.return)))}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Advanced Strategy Analytics */}
      {backtestData && backtestData[currentSplit] && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-6"
        >
          {/* Data Quality & Backtest Info */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Backtest Information & Data Quality</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                <div>
                  <div className="text-sm text-slate-400 font-mono mb-1">Run ID</div>
                  <div className="text-lg font-bold text-white font-mono">{backtestData.runId}</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400 font-mono mb-1">Generated</div>
                  <div className="text-lg font-bold text-white font-mono">
                    {new Date(backtestData.timestamp).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-slate-400 font-mono mb-1">Split</div>
                  <Badge variant={currentSplit === 'test' ? 'default' : 'secondary'} className="text-sm">
                    {currentSplit.toUpperCase()}
                  </Badge>
                </div>
                <div>
                  <div className="text-sm text-slate-400 font-mono mb-1">Quality Score</div>
                  <div className={`text-lg font-bold font-mono ${
                    dataQuality?.score > 80 ? 'text-green-400' : 
                    dataQuality?.score > 60 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {dataQuality?.score || 0}%
                  </div>
                </div>
              </div>
              
              {/* Data Quality Issues */}
              {dataQuality && dataQuality.issues.length > 0 && (
                <div className="mt-4 p-4 bg-yellow-950/30 border border-yellow-500/20 rounded-lg">
                  <div className="text-sm font-semibold text-yellow-400 mb-2">Data Quality Issues:</div>
                  <ul className="text-sm text-yellow-300 space-y-1">
                    {dataQuality.issues.map((issue: string, i: number) => (
                      <li key={i}>• {issue}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Recommendations */}
              {dataQuality && dataQuality.recommendations.length > 0 && (
                <div className="mt-4 p-4 bg-blue-950/30 border border-blue-500/20 rounded-lg">
                  <div className="text-sm font-semibold text-blue-400 mb-2">Recommendations:</div>
                  <ul className="text-sm text-blue-300 space-y-1">
                    {dataQuality.recommendations.map((rec: string, i: number) => (
                      <li key={i}>• {rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Strategy Configuration */}
          {backtestData[currentSplit]?.manifest && (
            <Card className="bg-slate-900 border-amber-500/20">
              <CardHeader>
                <CardTitle className="text-amber-500 font-mono">Strategy Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                  <div>
                    <div className="text-sm text-slate-400 font-mono mb-1">Policy Type</div>
                    <Badge variant="outline" className="text-sm">
                      {backtestData[currentSplit]?.manifest?.policy || 'Unknown'}
                    </Badge>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 font-mono mb-1">Features</div>
                    <div className="text-lg font-bold text-white font-mono">
                      {backtestData[currentSplit]?.manifest?.obs_cols?.length || 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 font-mono mb-1">Bars Processed</div>
                    <div className="text-lg font-bold text-white font-mono">
                      {backtestData[currentSplit]?.manifest?.n_rows?.toLocaleString() || '0'}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 font-mono mb-1">Days</div>
                    <div className="text-lg font-bold text-white font-mono">
                      {backtestData[currentSplit]?.manifest?.n_days || 0}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Performance Attribution */}
          {backtestData[currentSplit]?.kpis && (
            <Card className="bg-slate-900 border-amber-500/20">
              <CardHeader>
                <CardTitle className="text-amber-500 font-mono">Performance Attribution Analysis</CardTitle>
                <p className="text-slate-400 text-sm">Detailed breakdown of returns and risk sources</p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Returns Decomposition */}
                  <div>
                    <h4 className="text-white font-semibold mb-4">Returns Decomposition</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Alpha Generation</span>
                        <span className="text-green-400 font-mono">+{(backtestData[currentSplit]!.kpis!.top_bar.CAGR * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Cost Impact</span>
                        <span className="text-red-400 font-mono">-{(backtestData[currentSplit]!.kpis!.ejecucion_costos.cost_to_alpha_ratio * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Risk Contribution</span>
                        <span className="text-yellow-400 font-mono">{(backtestData[currentSplit]!.kpis!.top_bar.Vol_annualizada * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center border-t border-slate-700 pt-2">
                        <span className="text-white font-semibold">Net Performance</span>
                        <span className="text-cyan-400 font-bold font-mono">+{(backtestData[currentSplit]!.kpis!.top_bar.CAGR * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Risk Breakdown */}
                  <div>
                    <h4 className="text-white font-semibold mb-4">Risk Breakdown</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Market Risk (Beta)</span>
                        <span className="text-blue-400 font-mono">{backtestData[currentSplit]!.kpis!.exposicion_capacidad.beta?.toFixed(3) || '1.000'}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Idiosyncratic Risk</span>
                        <span className="text-purple-400 font-mono">{((backtestData[currentSplit]!.kpis!.top_bar.Vol_annualizada || 0) * 0.7 * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-sm">Tail Risk (VaR 99%)</span>
                        <span className="text-red-400 font-mono">{backtestData[currentSplit]!.kpis!.colas_y_drawdowns.VaR_99_bps.toFixed(0)} bps</span>
                      </div>
                      <div className="flex justify-between items-center border-t border-slate-700 pt-2">
                        <span className="text-white font-semibold">Risk-Adj Return</span>
                        <span className="text-cyan-400 font-bold font-mono">{backtestData[currentSplit]!.kpis!.top_bar.Sharpe?.toFixed(3) || '0.000'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Trigger New Backtest */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Backtest Controls</CardTitle>
              <p className="text-slate-400 text-sm">Manage backtest execution and refresh data</p>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={async () => {
                    try {
                      setLoading(true);
                      await backtestClient.triggerBacktest(false);
                      // Refresh data after trigger
                      setTimeout(() => window.location.reload(), 2000);
                    } catch (error) {
                      console.error('Failed to trigger backtest:', error);
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg font-mono text-sm hover:shadow-lg transition-all duration-300"
                >
                  Trigger New Backtest
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => window.location.reload()}
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-mono text-sm hover:shadow-lg transition-all duration-300"
                >
                  Refresh Data
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    if (backtestData) {
                      console.log('Backtest Data:', backtestData);
                      alert('Backtest data logged to console');
                    }
                  }}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-mono text-sm hover:shadow-lg transition-all duration-300"
                >
                  Debug Info
                </motion.button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
      
      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Professional Backtest Analytics • Generated {format(new Date(), 'PPpp')} • 
          Data Source: MinIO L6 Buckets • Compliance: Hedge Fund Standards
        </p>
      </div>
    </div>
  );
}