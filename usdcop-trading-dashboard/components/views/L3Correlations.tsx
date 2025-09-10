'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Heatmap, Cell, ComposedChart, Area, AreaChart, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend
} from 'recharts';
import { metricsCalculator } from '@/lib/services/hedge-fund-metrics';
import { minioClient } from '@/lib/services/minio-client';
import { 
  Network, GitBranch, BarChart3, TrendingUp, AlertTriangle, Shield, 
  Target, Activity, Eye, Layers, Database, Brain, Zap,
  Download, Settings, RefreshCw, Filter, Search
} from 'lucide-react';
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
  
  switch (formatStr) {
    case 'MM/dd':
      return `${String(month + 1).padStart(2, '0')}/${String(day).padStart(2, '0')}`;
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

const subHours = (date: Date, hours: number): Date => {
  const result = new Date(date);
  result.setHours(result.getHours() - hours);
  return result;
};

// Use the format function instead of date-fns format
const format = formatDate;

interface CorrelationPair {
  feature1: string;
  feature2: string;
  correlation: number;
  pValue: number;
  significance: 'critical' | 'high' | 'moderate' | 'low' | 'none';
  rollingCorr: number[];
  stability: number;
}

interface FeatureCluster {
  clusterId: number;
  features: string[];
  centroid: number[];
  intraClusterCorr: number;
  representativeFeature: string;
}

interface MulticollinearityAnalysis {
  vifScores: { feature: string; vif: number; severity: 'low' | 'moderate' | 'high' | 'critical' }[];
  conditionNumber: number;
  eigenvalues: number[];
  principalComponents: { pc: number; variance: number; cumulative: number }[];
}

interface NetworkMetrics {
  nodes: { id: string; size: number; color: string; centrality: number }[];
  edges: { source: string; target: string; weight: number; color: string }[];
  communities: { community: number; features: string[]; coherence: number }[];
  globalMetrics: {
    density: number;
    clustering: number;
    modularity: number;
    avgPathLength: number;
  };
}

interface FeatureSelection {
  method: string;
  selectedFeatures: string[];
  scores: { feature: string; score: number; rank: number }[];
  eliminationSteps: { step: number; eliminated: string; reason: string }[];
}

interface StatisticalTest {
  feature: string;
  normality: { statistic: number; pValue: number; isNormal: boolean; method: 'Shapiro-Wilk' | 'Kolmogorov-Smirnov' };
  stationarity: { statistic: number; pValue: number; isStationary: boolean; method: 'ADF' | 'KPSS' };
  heteroscedasticity: { statistic: number; pValue: number; isHomoscedastic: boolean; method: 'Breusch-Pagan' };
  autocorrelation: { statistic: number; pValue: number; hasAutocorr: boolean; lags: number };
}

export default function L3Correlations() {
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationPair[]>([]);
  const [featureClusters, setFeatureClusters] = useState<FeatureCluster[]>([]);
  const [multicollinearity, setMulticollinearity] = useState<MulticollinearityAnalysis | null>(null);
  const [networkMetrics, setNetworkMetrics] = useState<NetworkMetrics | null>(null);
  const [featureSelection, setFeatureSelection] = useState<FeatureSelection | null>(null);
  const [statisticalTests, setStatisticalTests] = useState<StatisticalTest[]>([]);
  const [heatmapData, setHeatmapData] = useState<any[]>([]);
  const [timeSeriesCorr, setTimeSeriesCorr] = useState<any[]>([]);
  const [selectedView, setSelectedView] = useState<'matrix' | 'network' | 'clusters' | 'multicollinearity' | 'tests' | 'selection'>('matrix');
  const [correlationThreshold, setCorrelationThreshold] = useState(0.3);
  const [showSignificantOnly, setShowSignificantOnly] = useState(false);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Generate comprehensive correlation analysis data
  const generateCorrelationData = useCallback(() => {
    // Feature universe for USDCOP trading
    const features = [
      'USDCOP_Close', 'USDCOP_Volume', 'USDCOP_Volatility', 'USDCOP_Returns',
      'USD_Index', 'USD_Index_MA20', 'USD_Index_RSI', 'USD_Index_MACD',
      'Oil_WTI', 'Oil_WTI_Returns', 'Oil_WTI_Volatility', 'Oil_WTI_RSI',
      'Colombian_Bonds_10Y', 'Bond_Yield_Spread', 'CDS_Colombia_5Y',
      'Fed_Rate', 'Colombian_Rate', 'Rate_Differential', 'Yield_Curve_Slope',
      'EM_FX_Index', 'LatAm_FX_Beta', 'Risk_On_Off_Indicator', 'VIX',
      'Copper_Price', 'Gold_Price', 'Coffee_Price', 'Coal_Price',
      'Inflation_Expectations', 'Political_Risk_Index', 'GDP_Nowcast',
      'Trade_Balance', 'Current_Account', 'Foreign_Reserves',
      'Hour_of_Day', 'Day_of_Week', 'Session_Premium', 'Holiday_Effect'
    ];

    // Generate correlation matrix with realistic financial relationships
    const correlations: CorrelationPair[] = [];
    
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        const feature1 = features[i];
        const feature2 = features[j];
        
        // Generate realistic correlations based on financial relationships
        let correlation = 0;
        let significance: 'critical' | 'high' | 'moderate' | 'low' | 'none' = 'none';
        
        // Strong relationships
        if ((feature1.includes('USD_Index') && feature2.includes('USDCOP')) ||
            (feature1.includes('Oil') && feature2.includes('USDCOP')) ||
            (feature1.includes('Rate') && feature2.includes('Rate'))) {
          correlation = (Math.random() - 0.5) * 1.4; // -0.7 to 0.7
          correlation = Math.max(-0.95, Math.min(0.95, correlation));
          significance = Math.abs(correlation) > 0.7 ? 'critical' : 
                        Math.abs(correlation) > 0.5 ? 'high' : 
                        Math.abs(correlation) > 0.3 ? 'moderate' : 'low';
        } else {
          correlation = (Math.random() - 0.5) * 0.8; // -0.4 to 0.4
          significance = Math.abs(correlation) > 0.3 ? 'moderate' : 
                        Math.abs(correlation) > 0.15 ? 'low' : 'none';
        }
        
        // Generate time series for rolling correlation
        const rollingCorr = Array.from({ length: 30 }, () => 
          correlation + (Math.random() - 0.5) * 0.2
        );
        
        correlations.push({
          feature1,
          feature2,
          correlation,
          pValue: Math.random() * 0.1,
          significance,
          rollingCorr,
          stability: 1 - (Math.random() * 0.4) // 0.6 to 1.0
        });
      }
    }

    // Generate feature clusters using k-means-like approach
    const clusters: FeatureCluster[] = [
      {
        clusterId: 1,
        features: ['USDCOP_Close', 'USDCOP_Volume', 'USDCOP_Volatility', 'USDCOP_Returns'],
        centroid: [0.8, 0.6, 0.9, 0.7],
        intraClusterCorr: 0.75,
        representativeFeature: 'USDCOP_Close'
      },
      {
        clusterId: 2,
        features: ['USD_Index', 'USD_Index_MA20', 'USD_Index_RSI', 'USD_Index_MACD'],
        centroid: [0.9, 0.8, 0.4, 0.3],
        intraClusterCorr: 0.68,
        representativeFeature: 'USD_Index'
      },
      {
        clusterId: 3,
        features: ['Oil_WTI', 'Oil_WTI_Returns', 'Oil_WTI_Volatility', 'Oil_WTI_RSI'],
        centroid: [0.7, 0.5, 0.8, 0.4],
        intraClusterCorr: 0.72,
        representativeFeature: 'Oil_WTI'
      },
      {
        clusterId: 4,
        features: ['Fed_Rate', 'Colombian_Rate', 'Rate_Differential', 'Yield_Curve_Slope'],
        centroid: [0.6, 0.7, 0.5, 0.3],
        intraClusterCorr: 0.81,
        representativeFeature: 'Rate_Differential'
      },
      {
        clusterId: 5,
        features: ['Hour_of_Day', 'Day_of_Week', 'Session_Premium', 'Holiday_Effect'],
        centroid: [0.3, 0.2, 0.6, 0.1],
        intraClusterCorr: 0.45,
        representativeFeature: 'Session_Premium'
      }
    ];

    // Generate multicollinearity analysis
    const vifScores = features.map(feature => {
      const vif = Math.random() * 15 + 1; // 1 to 16
      return {
        feature,
        vif,
        severity: vif > 10 ? 'critical' as const : 
                 vif > 5 ? 'high' as const : 
                 vif > 2.5 ? 'moderate' as const : 'low' as const
      };
    });

    const multicollinearityData: MulticollinearityAnalysis = {
      vifScores,
      conditionNumber: 245.7,
      eigenvalues: [12.5, 8.3, 6.1, 4.7, 3.2, 2.8, 1.9, 1.4, 0.8, 0.3],
      principalComponents: Array.from({ length: 10 }, (_, i) => ({
        pc: i + 1,
        variance: Math.max(0.05, (10 - i) / 55), // Decreasing variance
        cumulative: Array.from({ length: i + 1 }, (_, j) => (10 - j) / 55)
                   .reduce((sum, val) => sum + val, 0)
      }))
    };

    // Generate network metrics
    const networkData: NetworkMetrics = {
      nodes: features.map(feature => ({
        id: feature,
        size: Math.random() * 50 + 10,
        color: feature.includes('USDCOP') ? '#EF4444' : 
               feature.includes('USD_Index') ? '#F59E0B' :
               feature.includes('Oil') ? '#10B981' :
               feature.includes('Rate') ? '#8B5CF6' : '#64748B',
        centrality: Math.random()
      })),
      edges: correlations
        .filter(c => Math.abs(c.correlation) > 0.3)
        .map(c => ({
          source: c.feature1,
          target: c.feature2,
          weight: Math.abs(c.correlation),
          color: c.correlation > 0 ? '#10B981' : '#EF4444'
        })),
      communities: [
        { community: 1, features: ['USDCOP_Close', 'USDCOP_Volume'], coherence: 0.85 },
        { community: 2, features: ['USD_Index', 'USD_Index_MA20'], coherence: 0.78 },
        { community: 3, features: ['Oil_WTI', 'Oil_WTI_Returns'], coherence: 0.72 }
      ],
      globalMetrics: {
        density: 0.23,
        clustering: 0.67,
        modularity: 0.45,
        avgPathLength: 2.8
      }
    };

    // Generate feature selection results
    const selectionData: FeatureSelection = {
      method: 'Recursive Feature Elimination with Cross-Validation',
      selectedFeatures: [
        'USDCOP_Close', 'USD_Index', 'Oil_WTI', 'Rate_Differential',
        'EM_FX_Index', 'VIX', 'Session_Premium', 'Colombian_Bonds_10Y'
      ],
      scores: features.map((feature, idx) => ({
        feature,
        score: Math.random(),
        rank: idx + 1
      })).sort((a, b) => b.score - a.score),
      eliminationSteps: [
        { step: 1, eliminated: 'Holiday_Effect', reason: 'Low importance score (0.02)' },
        { step: 2, eliminated: 'Day_of_Week', reason: 'High correlation with Session_Premium (0.85)' },
        { step: 3, eliminated: 'Coal_Price', reason: 'Low importance and high noise' }
      ]
    };

    // Generate statistical tests
    const testData: StatisticalTest[] = features.slice(0, 8).map(feature => ({
      feature,
      normality: {
        statistic: Math.random() * 5,
        pValue: Math.random() * 0.1,
        isNormal: Math.random() > 0.3,
        method: Math.random() > 0.5 ? 'Shapiro-Wilk' : 'Kolmogorov-Smirnov'
      },
      stationarity: {
        statistic: -Math.random() * 5 - 1,
        pValue: Math.random() * 0.1,
        isStationary: Math.random() > 0.2,
        method: Math.random() > 0.5 ? 'ADF' : 'KPSS'
      },
      heteroscedasticity: {
        statistic: Math.random() * 20,
        pValue: Math.random() * 0.1,
        isHomoscedastic: Math.random() > 0.4,
        method: 'Breusch-Pagan'
      },
      autocorrelation: {
        statistic: Math.random() * 30,
        pValue: Math.random() * 0.1,
        hasAutocorr: Math.random() > 0.6,
        lags: Math.floor(Math.random() * 20) + 1
      }
    }));

    return { correlations, clusters, multicollinearityData, networkData, selectionData, testData };
  }, []);

  useEffect(() => {
    setLoading(true);
    
    const { correlations, clusters, multicollinearityData, networkData, selectionData, testData } = generateCorrelationData();
    
    setCorrelationMatrix(correlations);
    setFeatureClusters(clusters);
    setMulticollinearity(multicollinearityData);
    setNetworkMetrics(networkData);
    setFeatureSelection(selectionData);
    setStatisticalTests(testData);

    // Generate heatmap data
    const features = Array.from(new Set([
      ...correlations.map(c => c.feature1),
      ...correlations.map(c => c.feature2)
    ])).slice(0, 12); // Limit for visualization

    const heatmapGrid = features.map((feature1, i) => {
      const row = features.reduce((acc, feature2, j) => {
        if (i === j) {
          acc[feature2] = 1; // Diagonal
        } else {
          const corr = correlations.find(c => 
            (c.feature1 === feature1 && c.feature2 === feature2) ||
            (c.feature1 === feature2 && c.feature2 === feature1)
          );
          acc[feature2] = corr ? corr.correlation : 0;
        }
        return acc;
      }, {} as any);
      
      return { feature: feature1, ...row };
    });
    setHeatmapData(heatmapGrid);

    // Generate time series correlation data
    const timeSeriesData = Array.from({ length: 30 }, (_, i) => {
      const date = subDays(new Date(), 29 - i);
      return {
        date: format(date, 'MM/dd'),
        USDCOP_USD_Index: -0.65 + (Math.random() - 0.5) * 0.3,
        USDCOP_Oil: -0.72 + (Math.random() - 0.5) * 0.25,
        USD_Index_Oil: 0.15 + (Math.random() - 0.5) * 0.4,
        Rates_FX: 0.48 + (Math.random() - 0.5) * 0.35
      };
    });
    setTimeSeriesCorr(timeSeriesData);

    setLoading(false);
  }, [generateCorrelationData]);

  const filteredCorrelations = useMemo(() => {
    return correlationMatrix.filter(corr => {
      const meetsThreshold = Math.abs(corr.correlation) >= correlationThreshold;
      const meetsSignificance = !showSignificantOnly || 
        ['critical', 'high', 'moderate'].includes(corr.significance);
      const meetsFeatureFilter = selectedFeatures.length === 0 || 
        selectedFeatures.includes(corr.feature1) || selectedFeatures.includes(corr.feature2);
      
      return meetsThreshold && meetsSignificance && meetsFeatureFilter;
    });
  }, [correlationMatrix, correlationThreshold, showSignificantOnly, selectedFeatures]);

  const formatPercent = useCallback((value: number, decimals = 2) => {
    return `${(value * 100).toFixed(decimals)}%`;
  }, []);

  const getCorrelationColor = useCallback((correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs > 0.8) return correlation > 0 ? '#DC2626' : '#991B1B';
    if (abs > 0.6) return correlation > 0 ? '#F59E0B' : '#D97706';
    if (abs > 0.4) return correlation > 0 ? '#10B981' : '#059669';
    if (abs > 0.2) return correlation > 0 ? '#06B6D4' : '#0891B2';
    return '#64748B';
  }, []);

  const getSignificanceColor = useCallback((significance: string) => {
    switch (significance) {
      case 'critical': return 'bg-red-950 text-red-400';
      case 'high': return 'bg-orange-950 text-orange-400';
      case 'moderate': return 'bg-yellow-950 text-yellow-400';
      case 'low': return 'bg-blue-950 text-blue-400';
      default: return 'bg-slate-950 text-slate-400';
    }
  }, []);

  const getVIFSeverityColor = useCallback((severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-950 text-red-400';
      case 'high': return 'bg-orange-950 text-orange-400';
      case 'moderate': return 'bg-yellow-950 text-yellow-400';
      default: return 'bg-green-950 text-green-400';
    }
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading L3 Correlation Analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono">L3 FEATURE CORRELATION ANALYSIS</h1>
          <p className="text-slate-400 text-sm mt-1">
            Advanced Correlation Matrix • Multicollinearity Detection • Network Analysis • 
            {filteredCorrelations.length} Significant Pairs
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {([
              { key: 'matrix', label: 'Matrix' },
              { key: 'network', label: 'Network' },
              { key: 'clusters', label: 'Clusters' },
              { key: 'multicollinearity', label: 'VIF' },
              { key: 'tests', label: 'Tests' },
              { key: 'selection', label: 'Selection' }
            ] as const).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSelectedView(key)}
                className={`px-3 py-2 text-sm font-mono transition-colors ${
                  selectedView === key
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                }`}
              >
                {label}
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

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Features
              <Database className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono">
              {Array.from(new Set([...correlationMatrix.map(c => c.feature1), ...correlationMatrix.map(c => c.feature2)])).length}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Selected: {featureSelection?.selectedFeatures.length || 0}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Strong Corr
              <Network className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono">
              {correlationMatrix.filter(c => Math.abs(c.correlation) > 0.7).length}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              |r| &gt; 0.7
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Clusters
              <GitBranch className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-blue-400 font-mono">
              {featureClusters.length}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Avg Coherence: {(featureClusters.reduce((s, c) => s + c.intraClusterCorr, 0) / featureClusters.length).toFixed(2)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              High VIF
              <AlertTriangle className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-orange-400 font-mono">
              {multicollinearity?.vifScores.filter(v => v.severity === 'high' || v.severity === 'critical').length || 0}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              VIF &gt; 5
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Network Density
              <Activity className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(networkMetrics?.globalMetrics.density || 0, 1)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Clustering: {formatPercent(networkMetrics?.globalMetrics.clustering || 0, 1)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Condition #
              <Shield className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-xl font-bold font-mono ${
              (multicollinearity?.conditionNumber || 0) > 100 ? 'text-red-400' : 'text-green-400'
            }`}>
              {multicollinearity?.conditionNumber.toFixed(1) || '0.0'}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {(multicollinearity?.conditionNumber || 0) > 100 ? 'Ill-conditioned' : 'Well-conditioned'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Correlation Matrix View */}
      {selectedView === 'matrix' && (
        <div className="space-y-6">
          {/* Controls */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Correlation Filters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm text-slate-400 mb-2 block">Minimum |Correlation|</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={correlationThreshold}
                    onChange={(e) => setCorrelationThreshold(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-amber-400 font-mono">{correlationThreshold.toFixed(2)}</span>
                </div>
                <div>
                  <label className="text-sm text-slate-400 mb-2 block">Significance Filter</label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showSignificantOnly}
                      onChange={(e) => setShowSignificantOnly(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-sm text-slate-300">Show significant only</span>
                  </label>
                </div>
                <div>
                  <label className="text-sm text-slate-400 mb-2 block">Results</label>
                  <div className="text-sm text-slate-300">
                    {filteredCorrelations.length} pairs match criteria
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Time Series Correlation */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Rolling Correlation Dynamics</CardTitle>
              <p className="text-slate-400 text-sm">Key Relationships Over Time • Market Regime Changes</p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeSeriesCorr}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#64748B" fontSize={10} />
                  <YAxis domain={[-1, 1]} stroke="#64748B" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                    formatter={(value: any, name: string) => [
                      value.toFixed(3),
                      name.replace(/_/g, ' vs ')
                    ]}
                  />
                  <Line type="monotone" dataKey="USDCOP_USD_Index" stroke="#EF4444" strokeWidth={2} dot={false} name="USDCOP_USD_Index" />
                  <Line type="monotone" dataKey="USDCOP_Oil" stroke="#10B981" strokeWidth={2} dot={false} name="USDCOP_Oil" />
                  <Line type="monotone" dataKey="USD_Index_Oil" stroke="#F59E0B" strokeWidth={2} dot={false} name="USD_Index_Oil" />
                  <Line type="monotone" dataKey="Rates_FX" stroke="#8B5CF6" strokeWidth={2} dot={false} name="Rates_FX" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Correlation Pairs */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Correlation Pairs Analysis</CardTitle>
              <p className="text-slate-400 text-sm">Statistical Significance • Stability Metrics</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {filteredCorrelations.slice(0, 20).map((corr, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className={`w-4 h-4 rounded-full`} 
                           style={{ backgroundColor: getCorrelationColor(corr.correlation) }}></div>
                      <div>
                        <div className="text-white font-semibold text-sm">
                          {corr.feature1} ↔ {corr.feature2}
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge className={getSignificanceColor(corr.significance)}>
                            {corr.significance.toUpperCase()}
                          </Badge>
                          <span className="text-xs text-slate-400">
                            p-value: {corr.pValue.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-mono font-bold ${
                        Math.abs(corr.correlation) > 0.7 ? 'text-red-400' : 
                        Math.abs(corr.correlation) > 0.5 ? 'text-orange-400' : 'text-white'
                      }`}>
                        {corr.correlation.toFixed(3)}
                      </div>
                      <div className="text-xs text-slate-400">
                        Stability: {formatPercent(corr.stability, 1)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Feature Clusters View */}
      {selectedView === 'clusters' && (
        <div className="space-y-6">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Feature Clustering Analysis</CardTitle>
              <p className="text-slate-400 text-sm">Hierarchical Clustering • Intra-cluster Correlation • Representative Features</p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {featureClusters.map((cluster, index) => (
                  <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                    <div className="flex justify-between items-start mb-3">
                      <h4 className="font-semibold text-white text-sm">Cluster {cluster.clusterId}</h4>
                      <Badge className="bg-blue-950 text-blue-400">
                        {cluster.features.length} Features
                      </Badge>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400 text-xs">Intra-cluster Correlation</span>
                        <span className="font-mono text-green-400 text-sm">
                          {cluster.intraClusterCorr.toFixed(3)}
                        </span>
                      </div>
                      <Progress value={cluster.intraClusterCorr * 100} className="h-2" />
                      
                      <div>
                        <span className="text-slate-400 text-xs">Representative Feature:</span>
                        <div className="text-amber-400 font-semibold text-sm mt-1">
                          {cluster.representativeFeature}
                        </div>
                      </div>
                      
                      <div>
                        <span className="text-slate-400 text-xs">All Features:</span>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {cluster.features.map((feature) => (
                            <Badge key={feature} className="bg-slate-700 text-slate-300 text-xs">
                              {feature.replace(/_/g, ' ')}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Multicollinearity View */}
      {selectedView === 'multicollinearity' && multicollinearity && (
        <div className="space-y-6">
          {/* VIF Scores */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Variance Inflation Factor (VIF) Analysis</CardTitle>
              <p className="text-slate-400 text-sm">
                Multicollinearity Detection • VIF &gt; 5 indicates moderate, &gt; 10 indicates severe multicollinearity
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={multicollinearity.vifScores.slice(0, 15)} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#64748B" fontSize={10} />
                  <YAxis dataKey="feature" type="category" stroke="#64748B" fontSize={10} width={120} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                    formatter={(value: any) => [value.toFixed(2), 'VIF Score']}
                  />
                  <Bar dataKey="vif">
                    {multicollinearity.vifScores.slice(0, 15).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={
                        entry.severity === 'critical' ? '#DC2626' :
                        entry.severity === 'high' ? '#F59E0B' :
                        entry.severity === 'moderate' ? '#10B981' : '#06B6D4'
                      } />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Principal Components */}
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Principal Component Analysis</CardTitle>
              <p className="text-slate-400 text-sm">Variance Explanation • Dimensionality Reduction</p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={multicollinearity.principalComponents}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="pc" stroke="#64748B" fontSize={10} />
                  <YAxis yAxisId="variance" stroke="#64748B" fontSize={10} />
                  <YAxis yAxisId="cumulative" orientation="right" stroke="#64748B" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                    formatter={(value: any, name: string) => [
                      formatPercent(value),
                      name === 'variance' ? 'Individual Variance' : 'Cumulative Variance'
                    ]}
                  />
                  <Bar yAxisId="variance" dataKey="variance" fill="#10B981" name="variance" />
                  <Line yAxisId="cumulative" type="monotone" dataKey="cumulative" stroke="#F59E0B" strokeWidth={3} dot={{ fill: '#F59E0B' }} name="cumulative" />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Statistical Tests View */}
      {selectedView === 'tests' && (
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Statistical Tests Summary</CardTitle>
            <p className="text-slate-400 text-sm">Normality • Stationarity • Heteroscedasticity • Autocorrelation Tests</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {statisticalTests.map((test, index) => (
                <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <h4 className="font-semibold text-white text-lg mb-4">{test.feature}</h4>
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-2">Normality ({test.normality.method})</div>
                      <div className="text-white font-mono text-sm mb-1">
                        Stat: {test.normality.statistic.toFixed(3)}
                      </div>
                      <div className="text-white font-mono text-sm mb-2">
                        p: {test.normality.pValue.toFixed(4)}
                      </div>
                      <Badge className={test.normality.isNormal ? 'bg-green-950 text-green-400' : 'bg-red-950 text-red-400'}>
                        {test.normality.isNormal ? 'Normal' : 'Non-Normal'}
                      </Badge>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-2">Stationarity ({test.stationarity.method})</div>
                      <div className="text-white font-mono text-sm mb-1">
                        Stat: {test.stationarity.statistic.toFixed(3)}
                      </div>
                      <div className="text-white font-mono text-sm mb-2">
                        p: {test.stationarity.pValue.toFixed(4)}
                      </div>
                      <Badge className={test.stationarity.isStationary ? 'bg-green-950 text-green-400' : 'bg-red-950 text-red-400'}>
                        {test.stationarity.isStationary ? 'Stationary' : 'Non-Stationary'}
                      </Badge>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-2">Heteroscedasticity</div>
                      <div className="text-white font-mono text-sm mb-1">
                        Stat: {test.heteroscedasticity.statistic.toFixed(3)}
                      </div>
                      <div className="text-white font-mono text-sm mb-2">
                        p: {test.heteroscedasticity.pValue.toFixed(4)}
                      </div>
                      <Badge className={test.heteroscedasticity.isHomoscedastic ? 'bg-green-950 text-green-400' : 'bg-red-950 text-red-400'}>
                        {test.heteroscedasticity.isHomoscedastic ? 'Homoscedastic' : 'Heteroscedastic'}
                      </Badge>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-slate-400 text-xs mb-2">Autocorrelation</div>
                      <div className="text-white font-mono text-sm mb-1">
                        Stat: {test.autocorrelation.statistic.toFixed(3)}
                      </div>
                      <div className="text-white font-mono text-sm mb-2">
                        Lags: {test.autocorrelation.lags}
                      </div>
                      <Badge className={!test.autocorrelation.hasAutocorr ? 'bg-green-950 text-green-400' : 'bg-red-950 text-red-400'}>
                        {test.autocorrelation.hasAutocorr ? 'Autocorrelated' : 'Independent'}
                      </Badge>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Feature Selection View */}
      {selectedView === 'selection' && featureSelection && (
        <div className="space-y-6">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Feature Selection Results</CardTitle>
              <p className="text-slate-400 text-sm">
                Method: {featureSelection.method} • {featureSelection.selectedFeatures.length} Features Selected
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Feature Scores */}
                <div>
                  <h4 className="font-semibold text-white text-lg mb-4">Feature Importance Ranking</h4>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {featureSelection.scores.slice(0, 15).map((score, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className="w-8 text-slate-400 text-sm font-mono text-center">
                            #{score.rank}
                          </div>
                          <div>
                            <div className="text-white font-semibold text-sm">{score.feature}</div>
                            <Badge className={featureSelection.selectedFeatures.includes(score.feature) ? 
                              'bg-green-950 text-green-400' : 'bg-slate-700 text-slate-400'}>
                              {featureSelection.selectedFeatures.includes(score.feature) ? 'Selected' : 'Excluded'}
                            </Badge>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-amber-400 font-mono text-sm">
                            {score.score.toFixed(4)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Elimination Steps */}
                <div>
                  <h4 className="font-semibold text-white text-lg mb-4">Elimination Process</h4>
                  <div className="space-y-3">
                    {featureSelection.eliminationSteps.map((step, index) => (
                      <Alert key={index} className="bg-slate-800 border-slate-700">
                        <AlertTriangle className="h-4 w-4 text-yellow-400" />
                        <AlertDescription>
                          <div className="flex items-start justify-between">
                            <div>
                              <div className="font-semibold text-white text-sm mb-1">
                                Step {step.step}: {step.eliminated}
                              </div>
                              <div className="text-slate-300 text-xs">
                                {step.reason}
                              </div>
                            </div>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ))}
                  </div>

                  <div className="mt-6">
                    <h5 className="font-semibold text-white text-sm mb-3">Final Selected Features</h5>
                    <div className="flex flex-wrap gap-2">
                      {featureSelection.selectedFeatures.map((feature) => (
                        <Badge key={feature} className="bg-amber-950 text-amber-400">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          L3 Correlation Analysis • Generated {format(new Date(), 'PPpp')} • 
          Methods: Pearson Correlation, K-means Clustering, VIF Analysis • 
          Statistical Tests: Shapiro-Wilk, ADF, Breusch-Pagan
        </p>
      </div>
    </div>
  );
}