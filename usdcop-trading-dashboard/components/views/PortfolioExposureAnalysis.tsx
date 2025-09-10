'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, ComposedChart, ScatterChart, Scatter, TreemapChart, Treemap
} from 'recharts';
import { realTimeRiskEngine, Position, RealTimeRiskMetrics } from '@/lib/services/real-time-risk-engine';
import {
  Target, TrendingUp, TrendingDown, AlertTriangle, BarChart3, PieChart as PieChartIcon,
  Activity, DollarSign, Percent, Globe, Building, MapPin, Calendar,
  Shield, Zap, Users, ArrowUpRight, ArrowDownRight, Info, Settings
} from 'lucide-react';
// Custom date formatting function with Spanish month names
const formatDate = (date: Date, formatStr: string) => {
  const months = [
    'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'
  ];
  
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const seconds = date.getSeconds();
  
  switch (formatStr) {
    case 'HH:mm:ss':
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    case 'PPpp':
      return `${date.getDate()} de ${months[date.getMonth()]} de ${date.getFullYear()} ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
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

// Use the format function instead of date-fns format
const format = formatDate;

interface ExposureBreakdown {
  // Geographic Exposure
  countryExposure: Array<{ country: string; exposure: number; percentage: number; risk: 'low' | 'medium' | 'high' }>;
  
  // Currency Exposure
  currencyExposure: Array<{ currency: string; exposure: number; percentage: number; hedged: boolean }>;
  
  // Sector/Asset Class Exposure
  sectorExposure: Array<{ sector: string; exposure: number; percentage: number; beta: number }>;
  
  // Time-based Exposure
  maturityBuckets: Array<{ bucket: string; exposure: number; avgMaturity: number }>;
  
  // Risk Factor Exposure
  riskFactors: Array<{ factor: string; exposure: number; sensitivity: number; contribution: number }>;
  
  // Concentration Analysis
  concentrationMetrics: {
    herfindahlIndex: number;
    top5Concentration: number;
    top10Concentration: number;
    effectiveNumPositions: number;
    diversificationRatio: number;
  };
  
  // Correlation Analysis
  correlationStructure: {
    avgCorrelation: number;
    maxCorrelation: number;
    minCorrelation: number;
    correlationClusters: Array<{ cluster: string; positions: string[]; avgCorrelation: number }>;
  };
}

interface RiskAttribution {
  totalRisk: number;
  components: Array<{
    name: string;
    contribution: number;
    percentage: number;
    marginalContribution: number;
  }>;
  factorBreakdown: {
    systematic: number;
    specific: number;
    interaction: number;
  };
}

interface LiquidityAnalysis {
  liquidityTiers: Array<{
    tier: string;
    exposure: number;
    averageDays: number;
    positions: number;
  }>;
  liquidityConcentration: number;
  worstCaseLiquidation: number;
  liquidityBuffer: number;
  marketImpactCost: number;
}

interface StressTestResults {
  scenarios: Array<{
    name: string;
    impact: number;
    probability: number;
    timeHorizon: string;
    contributors: Array<{ position: string; impact: number }>;
  }>;
  tailRisk: {
    var95: number;
    var99: number;
    expectedShortfall: number;
    maxLoss: number;
  };
}

export default function PortfolioExposureAnalysis() {
  const [exposureData, setExposureData] = useState<ExposureBreakdown | null>(null);
  const [riskAttribution, setRiskAttribution] = useState<RiskAttribution | null>(null);
  const [liquidityAnalysis, setLiquidityAnalysis] = useState<LiquidityAnalysis | null>(null);
  const [stressResults, setStressResults] = useState<StressTestResults | null>(null);
  const [riskMetrics, setRiskMetrics] = useState<RealTimeRiskMetrics | null>(null);
  const [selectedView, setSelectedView] = useState<'geographic' | 'currency' | 'sector' | 'risk-factors'>('geographic');
  const [timeHorizon, setTimeHorizon] = useState<'1D' | '1W' | '1M' | '3M'>('1M');
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Colors for charts
  const exposureColors = ['#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#EC4899', '#14B8A6', '#F97316'];
  const riskColors = ['#DC2626', '#EA580C', '#D97706', '#CA8A04', '#65A30D'];

  const generateMockExposureData = useCallback((): ExposureBreakdown => {
    return {
      countryExposure: [
        { country: 'Colombia', exposure: 8500000, percentage: 85, risk: 'medium' },
        { country: 'United States', exposure: 1000000, percentage: 10, risk: 'low' },
        { country: 'Brazil', exposure: 300000, percentage: 3, risk: 'medium' },
        { country: 'Mexico', exposure: 200000, percentage: 2, risk: 'medium' }
      ],
      currencyExposure: [
        { currency: 'COP', exposure: 8500000, percentage: 85, hedged: false },
        { currency: 'USD', exposure: 1200000, percentage: 12, hedged: true },
        { currency: 'BRL', exposure: 300000, percentage: 3, hedged: false }
      ],
      sectorExposure: [
        { sector: 'FX Spot', exposure: 6000000, percentage: 60, beta: 1.0 },
        { sector: 'Interest Rates', exposure: 2000000, percentage: 20, beta: 0.8 },
        { sector: 'Commodities', exposure: 1200000, percentage: 12, beta: 1.2 },
        { sector: 'Volatility', exposure: 800000, percentage: 8, beta: 1.5 }
      ],
      maturityBuckets: [
        { bucket: '0-1M', exposure: 4000000, avgMaturity: 15 },
        { bucket: '1-3M', exposure: 3000000, avgMaturity: 60 },
        { bucket: '3-6M', exposure: 2000000, avgMaturity: 135 },
        { bucket: '6M+', exposure: 1000000, avgMaturity: 270 }
      ],
      riskFactors: [
        { factor: 'USD/COP Rate', exposure: 8500000, sensitivity: 1.0, contribution: 0.65 },
        { factor: 'COP Vol', exposure: 2000000, sensitivity: 0.3, contribution: 0.15 },
        { factor: 'Oil Prices', exposure: 1500000, sensitivity: -0.4, contribution: 0.12 },
        { factor: 'EM Risk', exposure: 1000000, sensitivity: 0.6, contribution: 0.08 }
      ],
      concentrationMetrics: {
        herfindahlIndex: 0.385,
        top5Concentration: 0.87,
        top10Concentration: 0.95,
        effectiveNumPositions: 2.6,
        diversificationRatio: 0.73
      },
      correlationStructure: {
        avgCorrelation: 0.67,
        maxCorrelation: 0.89,
        minCorrelation: -0.23,
        correlationClusters: [
          { cluster: 'FX Cluster', positions: ['USDCOP', 'COPUSD'], avgCorrelation: 0.95 },
          { cluster: 'Rate Cluster', positions: ['COP Bonds', 'USD Rates'], avgCorrelation: 0.72 },
          { cluster: 'Commodity Cluster', positions: ['Oil', 'Commodities'], avgCorrelation: 0.81 }
        ]
      }
    };
  }, []);

  const generateRiskAttribution = useCallback((): RiskAttribution => {
    return {
      totalRisk: 0.0247, // 2.47% daily vol
      components: [
        { name: 'USDCOP Spot', contribution: 0.0165, percentage: 66.8, marginalContribution: 0.0143 },
        { name: 'Rate Positions', contribution: 0.0048, percentage: 19.4, marginalContribution: 0.0039 },
        { name: 'Vol Positions', contribution: 0.0024, percentage: 9.7, marginalContribution: 0.0031 },
        { name: 'Commodity Hedge', contribution: 0.0010, percentage: 4.1, marginalContribution: 0.0008 }
      ],
      factorBreakdown: {
        systematic: 0.0185, // 75%
        specific: 0.0049,   // 20%
        interaction: 0.0013  // 5%
      }
    };
  }, []);

  const generateLiquidityAnalysis = useCallback((): LiquidityAnalysis => {
    return {
      liquidityTiers: [
        { tier: 'Tier 1 (Same Day)', exposure: 6000000, averageDays: 0.5, positions: 3 },
        { tier: 'Tier 2 (1-3 Days)', exposure: 2500000, averageDays: 2, positions: 4 },
        { tier: 'Tier 3 (1 Week)', exposure: 1000000, averageDays: 5, positions: 2 },
        { tier: 'Tier 4 (1 Month)', exposure: 500000, averageDays: 20, positions: 1 }
      ],
      liquidityConcentration: 0.60,
      worstCaseLiquidation: 7.5, // days
      liquidityBuffer: 0.85,
      marketImpactCost: 0.0025 // 25bps
    };
  }, []);

  const generateStressResults = useCallback((): StressTestResults => {
    return {
      scenarios: [
        {
          name: 'Banco República 200bp Hike',
          impact: -650000,
          probability: 0.15,
          timeHorizon: '1M',
          contributors: [
            { position: 'USDCOP Long', impact: -520000 },
            { position: 'COP Bonds', impact: -130000 }
          ]
        },
        {
          name: 'Oil Price Collapse -40%',
          impact: -780000,
          probability: 0.12,
          timeHorizon: '3M',
          contributors: [
            { position: 'USDCOP Long', impact: -650000 },
            { position: 'Commodity Positions', impact: -130000 }
          ]
        },
        {
          name: 'EM Crisis Contagion',
          impact: -890000,
          probability: 0.08,
          timeHorizon: '1M',
          contributors: [
            { position: 'USDCOP Long', impact: -712000 },
            { position: 'EM Positions', impact: -178000 }
          ]
        },
        {
          name: 'Liquidity Crisis',
          impact: -1200000,
          probability: 0.02,
          timeHorizon: '1W',
          contributors: [
            { position: 'All Positions', impact: -1200000 }
          ]
        }
      ],
      tailRisk: {
        var95: -450000,
        var99: -720000,
        expectedShortfall: -890000,
        maxLoss: -1500000
      }
    };
  }, []);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      
      // Simulate loading delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Generate mock data
      setExposureData(generateMockExposureData());
      setRiskAttribution(generateRiskAttribution());
      setLiquidityAnalysis(generateLiquidityAnalysis());
      setStressResults(generateStressResults());
      
      // Get current risk metrics
      const metrics = realTimeRiskEngine.getRiskMetrics();
      setRiskMetrics(metrics);
      
      setLastUpdate(new Date());
      setLoading(false);
    };
    
    loadData();
    
    // Set up periodic updates
    const interval = setInterval(loadData, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, [generateMockExposureData, generateRiskAttribution, generateLiquidityAnalysis, generateStressResults]);

  const formatCurrency = useCallback((value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  }, []);

  const formatPercent = useCallback((value: number, decimals = 1) => {
    return `${(value * 100).toFixed(decimals)}%`;
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Portfolio Exposure Analysis...</p>
        </div>
      </div>
    );
  }

  if (!exposureData || !riskAttribution || !liquidityAnalysis || !stressResults) return null;

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono">PORTFOLIO EXPOSURE ANALYSIS</h1>
          <p className="text-slate-400 text-sm mt-1">
            Multi-Dimensional Risk Analytics • Last Update: {format(lastUpdate, 'HH:mm:ss')}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {(['geographic', 'currency', 'sector', 'risk-factors'] as const).map((view) => (
              <button
                key={view}
                onClick={() => setSelectedView(view)}
                className={`px-3 py-2 text-sm font-mono transition-colors capitalize ${
                  selectedView === view
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                }`}
              >
                {view.replace('-', ' ')}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-2 bg-slate-900 border border-amber-500/20 text-amber-500 rounded-lg hover:bg-amber-500/10 transition-colors flex items-center gap-2 text-sm">
              <Settings className="h-4 w-4" />
              Settings
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Portfolio Value
              <DollarSign className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono">
              {formatCurrency(riskMetrics?.portfolioValue || 10000000)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Net Exposure: {formatCurrency(riskMetrics?.netExposure || 8500000)}
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
            <div className="text-xl font-bold text-amber-400 font-mono">
              {formatPercent(exposureData.concentrationMetrics.top5Concentration)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              HHI: {exposureData.concentrationMetrics.herfindahlIndex.toFixed(3)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Diversification
              <BarChart3 className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(exposureData.concentrationMetrics.diversificationRatio)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Eff. Positions: {exposureData.concentrationMetrics.effectiveNumPositions.toFixed(1)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Avg Correlation
              <Activity className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-orange-400 font-mono">
              {formatPercent(exposureData.correlationStructure.avgCorrelation)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Max: {formatPercent(exposureData.correlationStructure.maxCorrelation)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Liquidity Score
              <Zap className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-blue-400 font-mono">
              {formatPercent(liquidityAnalysis.liquidityBuffer)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Liquidation: {liquidityAnalysis.worstCaseLiquidation}d
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Market Impact
              <AlertTriangle className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono">
              {(liquidityAnalysis.marketImpactCost * 10000).toFixed(0)}bp
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Est. Trading Cost
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Exposure Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Primary Exposure Breakdown */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              {selectedView === 'geographic' && 'Geographic Exposure'}
              {selectedView === 'currency' && 'Currency Exposure'}
              {selectedView === 'sector' && 'Sector Exposure'}
              {selectedView === 'risk-factors' && 'Risk Factor Exposure'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={
                    selectedView === 'geographic' ? exposureData.countryExposure :
                    selectedView === 'currency' ? exposureData.currencyExposure :
                    selectedView === 'sector' ? exposureData.sectorExposure :
                    exposureData.riskFactors
                  }
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name}: ${percentage}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="exposure"
                >
                  {(selectedView === 'geographic' ? exposureData.countryExposure :
                    selectedView === 'currency' ? exposureData.currencyExposure :
                    selectedView === 'sector' ? exposureData.sectorExposure :
                    exposureData.riskFactors
                  ).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={exposureColors[index % exposureColors.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: any) => [formatCurrency(value), 'Exposure']} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Attribution */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Risk Attribution Analysis
            </CardTitle>
            <p className="text-slate-400 text-sm">
              Total Portfolio Risk: {formatPercent(riskAttribution.totalRisk)} Daily Vol
            </p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={riskAttribution.components}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#64748B" fontSize={10} angle={-45} textAnchor="end" height={60} />
                <YAxis stroke="#64748B" fontSize={10} tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any, name) => [
                    name === 'contribution' ? formatPercent(value) : value,
                    name === 'contribution' ? 'Risk Contribution' : 'Marginal Contribution'
                  ]}
                />
                <Bar dataKey="contribution" fill="#8B5CF6" />
                <Bar dataKey="marginalContribution" fill="#10B981" opacity={0.7} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis Tabs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Concentration Metrics */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Target className="h-5 w-5" />
              Concentration Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Herfindahl Index</span>
                <span className="font-mono text-white">{exposureData.concentrationMetrics.herfindahlIndex.toFixed(3)}</span>
              </div>
              <Progress value={exposureData.concentrationMetrics.herfindahlIndex * 100} className="h-2" />
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Top 5 Concentration</span>
                <span className="font-mono text-amber-400">{formatPercent(exposureData.concentrationMetrics.top5Concentration)}</span>
              </div>
              <Progress value={exposureData.concentrationMetrics.top5Concentration * 100} className="h-2" />
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Effective Positions</span>
                <span className="font-mono text-green-400">{exposureData.concentrationMetrics.effectiveNumPositions.toFixed(1)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Diversification Ratio</span>
                <span className="font-mono text-blue-400">{formatPercent(exposureData.concentrationMetrics.diversificationRatio)}</span>
              </div>
              
              <Alert className="border-yellow-500/30 bg-yellow-950/20 mt-4">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <AlertDescription className="text-yellow-200 text-sm">
                  High concentration in USDCOP positions. Consider diversification.
                </AlertDescription>
              </Alert>
            </div>
          </CardContent>
        </Card>

        {/* Liquidity Breakdown */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Liquidity Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {liquidityAnalysis.liquidityTiers.map((tier, index) => (
                <div key={tier.tier} className="bg-slate-800 p-3 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-white font-semibold text-sm">{tier.tier}</span>
                    <Badge className="bg-blue-950 text-blue-400">{tier.positions} positions</Badge>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Exposure:</span>
                    <span className="text-amber-400 font-mono">{formatCurrency(tier.exposure)}</span>
                  </div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-400">Avg Days:</span>
                    <span className="text-slate-300 font-mono">{tier.averageDays}</span>
                  </div>
                  <Progress 
                    value={(tier.exposure / 10000000) * 100} 
                    className="h-2 mt-2" 
                  />
                </div>
              ))}
              
              <div className="mt-4 p-3 bg-slate-800 rounded-lg">
                <div className="text-sm text-slate-400 mb-2">Summary Metrics</div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Liquidity Buffer:</span>
                    <span className="text-green-400 font-mono">{formatPercent(liquidityAnalysis.liquidityBuffer)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Market Impact:</span>
                    <span className="text-red-400 font-mono">{(liquidityAnalysis.marketImpactCost * 10000).toFixed(0)}bp</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Factor Breakdown */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Risk Factor Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Systematic', value: riskAttribution.factorBreakdown.systematic, color: '#EF4444' },
                    { name: 'Specific', value: riskAttribution.factorBreakdown.specific, color: '#10B981' },
                    { name: 'Interaction', value: riskAttribution.factorBreakdown.interaction, color: '#8B5CF6' }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                >
                  {[
                    { color: '#EF4444' },
                    { color: '#10B981' },
                    { color: '#8B5CF6' }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: any) => [formatPercent(value), 'Risk Contribution']} />
              </PieChart>
            </ResponsiveContainer>
            
            <div className="mt-4 space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Systematic Risk:</span>
                <span className="text-red-400 font-mono">{formatPercent(riskAttribution.factorBreakdown.systematic)}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Specific Risk:</span>
                <span className="text-green-400 font-mono">{formatPercent(riskAttribution.factorBreakdown.specific)}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Interaction:</span>
                <span className="text-purple-400 font-mono">{formatPercent(riskAttribution.factorBreakdown.interaction)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stress Test Results */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Portfolio Stress Testing & Scenario Analysis
          </CardTitle>
          <p className="text-slate-400 text-sm">
            Impact Analysis • Tail Risk Assessment • Crisis Simulation
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-white font-semibold mb-3">Scenario Impacts</h4>
              <div className="space-y-3">
                {stressResults.scenarios.map((scenario, index) => (
                  <div key={index} className="bg-slate-800 p-4 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <h5 className="text-white font-semibold text-sm">{scenario.name}</h5>
                      <Badge className="bg-orange-950 text-orange-400">
                        {formatPercent(scenario.probability, 0)} prob
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-slate-400 text-sm">Portfolio Impact:</span>
                      <span className={`font-mono text-sm ${scenario.impact < 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {formatCurrency(scenario.impact)}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500">
                      Horizon: {scenario.timeHorizon} • 
                      Top Impact: {scenario.contributors[0]?.position} ({formatCurrency(scenario.contributors[0]?.impact || 0)})
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-3">Tail Risk Metrics</h4>
              <div className="space-y-3">
                <div className="bg-slate-800 p-4 rounded-lg">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-1">VaR 95%</div>
                      <div className="text-red-400 font-bold font-mono text-lg">
                        {formatCurrency(stressResults.tailRisk.var95)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-1">VaR 99%</div>
                      <div className="text-red-500 font-bold font-mono text-lg">
                        {formatCurrency(stressResults.tailRisk.var99)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-1">Expected Shortfall</div>
                      <div className="text-orange-400 font-bold font-mono text-lg">
                        {formatCurrency(stressResults.tailRisk.expectedShortfall)}
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-1">Maximum Loss</div>
                      <div className="text-red-600 font-bold font-mono text-lg">
                        {formatCurrency(stressResults.tailRisk.maxLoss)}
                      </div>
                    </div>
                  </div>
                </div>
                
                <Alert className="border-red-500/30 bg-red-950/20">
                  <AlertTriangle className="h-4 w-4 text-red-500" />
                  <AlertDescription className="text-red-200 text-sm">
                    Significant tail risk concentration in USDCOP exposure. 
                    Consider hedging strategies for extreme scenarios.
                  </AlertDescription>
                </Alert>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Portfolio Exposure Analysis • Generated {format(new Date(), 'PPpp')} • 
          Multi-Factor Risk Model • Real-Time Stress Testing • 
          Institutional Risk Management Framework
        </p>
      </div>
    </div>
  );
}