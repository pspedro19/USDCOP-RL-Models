'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter,
  AreaChart, Area, ComposedChart, Heatmap, Cell
} from 'recharts';
import { metricsCalculator } from '@/lib/services/hedge-fund-metrics';
import { minioClient } from '@/lib/services/minio-client';
import { 
  Brain, Zap, Target, TrendingUp, AlertTriangle, Shield, Activity,
  BarChart3, Clock, Cpu, Database, GitBranch, Eye, Settings,
  Download, RefreshCw, Play, Pause, RotateCcw, Layers, Gauge
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
    case 'HH:mm':
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    case 'HH:mm:ss':
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
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

interface ModelMetrics {
  // Core Performance
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  
  // Trading Performance
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  
  // Model Health
  driftScore: number;
  stabilityIndex: number;
  confidenceInterval: number;
  calibrationError: number;
  
  // Operational
  inferenceTime: number;
  memoryUsage: number;
  throughput: number;
  lastUpdate: Date;
}

interface DriftDetection {
  timestamp: Date;
  feature: string;
  driftMagnitude: number;
  pValue: number;
  threshold: number;
  status: 'stable' | 'warning' | 'drift';
  recommendation?: string;
}

interface SHAPValue {
  feature: string;
  shapValue: number;
  baseValue: number;
  featureValue: number;
  contribution: number;
}

interface EnsembleModel {
  name: string;
  type: 'PPO' | 'SAC' | 'A3C' | 'DQN' | 'TD3' | 'XGBoost' | 'LightGBM';
  weight: number;
  performance: number;
  reliability: number;
  latency: number;
  status: 'active' | 'inactive' | 'degraded';
}

interface ModelPrediction {
  timestamp: Date;
  prediction: number;
  confidence: number;
  actualOutcome?: number;
  modelContributions: { [key: string]: number };
}

export default function ModelPerformance() {
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [ensembleModels, setEnsembleModels] = useState<EnsembleModel[]>([]);
  const [driftDetections, setDriftDetections] = useState<DriftDetection[]>([]);
  const [shapValues, setShapValues] = useState<SHAPValue[]>([]);
  const [recentPredictions, setRecentPredictions] = useState<ModelPrediction[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<any[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [calibrationData, setCalibrationData] = useState<any[]>([]);
  const [learningCurves, setLearningCurves] = useState<any[]>([]);
  const [selectedTimeWindow, setSelectedTimeWindow] = useState<'1H' | '6H' | '1D' | '1W'>('1D');
  const [isRetraining, setIsRetraining] = useState(false);
  const [loading, setLoading] = useState(true);

  // Generate comprehensive model performance data
  const generateModelData = useCallback(() => {
    // Core model metrics
    const mockMetrics: ModelMetrics = {
      accuracy: 0.847,
      precision: 0.862,
      recall: 0.835,
      f1Score: 0.848,
      auc: 0.923,
      
      sharpeRatio: 2.34,
      maxDrawdown: 0.087,
      winRate: 0.643,
      profitFactor: 1.76,
      
      driftScore: 0.23, // 0-1 scale, <0.3 is good
      stabilityIndex: 0.89, // 0-1 scale, >0.8 is good
      confidenceInterval: 0.95,
      calibrationError: 0.034,
      
      inferenceTime: 12.5, // ms
      memoryUsage: 2.3, // GB
      throughput: 1250, // predictions/sec
      lastUpdate: new Date()
    };
    
    // Ensemble models configuration
    const ensemble: EnsembleModel[] = [
      {
        name: 'Primary PPO',
        type: 'PPO',
        weight: 0.35,
        performance: 0.847,
        reliability: 0.92,
        latency: 11.2,
        status: 'active'
      },
      {
        name: 'SAC Hedge',
        type: 'SAC',
        weight: 0.25,
        performance: 0.824,
        reliability: 0.88,
        latency: 14.8,
        status: 'active'
      },
      {
        name: 'XGBoost Signal',
        type: 'XGBoost',
        weight: 0.20,
        performance: 0.809,
        reliability: 0.95,
        latency: 8.3,
        status: 'active'
      },
      {
        name: 'LightGBM Fast',
        type: 'LightGBM',
        weight: 0.15,
        performance: 0.798,
        reliability: 0.94,
        latency: 6.1,
        status: 'active'
      },
      {
        name: 'TD3 Backup',
        type: 'TD3',
        weight: 0.05,
        performance: 0.789,
        reliability: 0.85,
        latency: 18.2,
        status: 'inactive'
      }
    ];
    
    // Drift detection alerts
    const drifts: DriftDetection[] = [
      {
        timestamp: subHours(new Date(), 2),
        feature: 'USD_Index',
        driftMagnitude: 0.34,
        pValue: 0.023,
        threshold: 0.05,
        status: 'drift',
        recommendation: 'Retrain with recent Fed policy data'
      },
      {
        timestamp: subHours(new Date(), 6),
        feature: 'Oil_WTI',
        driftMagnitude: 0.28,
        pValue: 0.081,
        threshold: 0.05,
        status: 'warning',
        recommendation: 'Monitor commodity correlation shifts'
      },
      {
        timestamp: subHours(new Date(), 12),
        feature: 'COP_Volatility',
        driftMagnitude: 0.15,
        pValue: 0.234,
        threshold: 0.05,
        status: 'stable'
      },
      {
        timestamp: subHours(new Date(), 18),
        feature: 'Interest_Rate_Spread',
        driftMagnitude: 0.42,
        pValue: 0.012,
        threshold: 0.05,
        status: 'drift',
        recommendation: 'Update yield curve features immediately'
      }
    ];
    
    // SHAP values for latest prediction
    const shapData: SHAPValue[] = [
      {
        feature: 'USD_Index_MA20',
        shapValue: 0.085,
        baseValue: 0.52,
        featureValue: 103.45,
        contribution: 0.163
      },
      {
        feature: 'Oil_WTI_RSI',
        shapValue: -0.062,
        baseValue: 0.52,
        featureValue: 32.1,
        contribution: -0.119
      },
      {
        feature: 'COP_Volatility_Z',
        shapValue: 0.047,
        baseValue: 0.52,
        featureValue: 1.85,
        contribution: 0.090
      },
      {
        feature: 'Rate_Spread_5Y',
        shapValue: 0.038,
        baseValue: 0.52,
        featureValue: 2.34,
        contribution: 0.073
      },
      {
        feature: 'EM_FX_Beta',
        shapValue: -0.029,
        baseValue: 0.52,
        featureValue: 0.78,
        contribution: -0.056
      },
      {
        feature: 'Sentiment_Index',
        shapValue: 0.024,
        baseValue: 0.52,
        featureValue: 0.62,
        contribution: 0.046
      }
    ].sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
    
    return { mockMetrics, ensemble, drifts, shapData };
  }, []);
  
  useEffect(() => {
    setLoading(true);
    
    const { mockMetrics, ensemble, drifts, shapData } = generateModelData();
    setModelMetrics(mockMetrics);
    setEnsembleModels(ensemble);
    setDriftDetections(drifts);
    setShapValues(shapData);
    
    // Generate performance history
    const hours = selectedTimeWindow === '1H' ? 1 : selectedTimeWindow === '6H' ? 6 : selectedTimeWindow === '1D' ? 24 : 168;
    const performanceData = Array.from({ length: hours }, (_, i) => {
      const timestamp = subHours(new Date(), hours - i);
      return {
        timestamp: format(timestamp, 'HH:mm'),
        fullTimestamp: timestamp,
        accuracy: mockMetrics.accuracy + (Math.random() - 0.5) * 0.1,
        latency: mockMetrics.inferenceTime + (Math.random() - 0.5) * 5,
        throughput: mockMetrics.throughput + (Math.random() - 0.5) * 200,
        driftScore: Math.max(0, mockMetrics.driftScore + (Math.random() - 0.5) * 0.2),
        memoryUsage: mockMetrics.memoryUsage + (Math.random() - 0.5) * 0.5
      };
    });
    setPerformanceHistory(performanceData);
    
    // Generate feature importance over time
    const features = ['USD_Index', 'Oil_WTI', 'COP_Vol', 'Rate_Spread', 'EM_FX', 'Sentiment'];
    const importanceData = features.map(feature => ({
      feature,
      importance: Math.random() * 0.8 + 0.1,
      trend: (Math.random() - 0.5) * 0.1,
      stability: Math.random() * 0.3 + 0.7
    })).sort((a, b) => b.importance - a.importance);
    setFeatureImportance(importanceData);
    
    // Generate model predictions
    const predictions = Array.from({ length: 24 }, (_, i) => {
      const timestamp = subHours(new Date(), 24 - i);
      const prediction = Math.random();
      return {
        timestamp,
        prediction,
        confidence: 0.7 + Math.random() * 0.25,
        actualOutcome: i < 20 ? Math.random() : undefined, // Some don't have outcomes yet
        modelContributions: {
          'PPO': prediction * 0.35 + (Math.random() - 0.5) * 0.1,
          'SAC': prediction * 0.25 + (Math.random() - 0.5) * 0.1,
          'XGBoost': prediction * 0.20 + (Math.random() - 0.5) * 0.1,
          'LightGBM': prediction * 0.15 + (Math.random() - 0.5) * 0.1,
          'TD3': prediction * 0.05 + (Math.random() - 0.5) * 0.1
        }
      };
    });
    setRecentPredictions(predictions);
    
    // Generate calibration data
    const calibrationBins = Array.from({ length: 10 }, (_, i) => {
      const binStart = i * 0.1;
      const binEnd = (i + 1) * 0.1;
      const predicted = binStart + 0.05;
      const actual = predicted + (Math.random() - 0.5) * 0.08; // Some calibration error
      return {
        bin: `${(binStart * 100).toFixed(0)}-${(binEnd * 100).toFixed(0)}%`,
        predicted: predicted * 100,
        actual: Math.max(0, Math.min(100, actual * 100)),
        count: Math.floor(Math.random() * 500 + 100)
      };
    });
    setCalibrationData(calibrationBins);
    
    // Generate learning curves
    const epochs = Array.from({ length: 100 }, (_, i) => ({
      epoch: i + 1,
      trainLoss: Math.exp(-i / 30) * 0.8 + 0.1 + (Math.random() - 0.5) * 0.02,
      valLoss: Math.exp(-i / 25) * 0.9 + 0.12 + (Math.random() - 0.5) * 0.03,
      trainAcc: 1 - Math.exp(-i / 20) * 0.6 + (Math.random() - 0.5) * 0.02,
      valAcc: 1 - Math.exp(-i / 18) * 0.65 + (Math.random() - 0.5) * 0.03
    }));
    setLearningCurves(epochs);
    
    setLoading(false);
  }, [generateModelData, selectedTimeWindow]);
  
  const formatPercent = useCallback((value: number, decimals = 2) => {
    return `${(value * 100).toFixed(decimals)}%`;
  }, []);
  
  const getDriftStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'stable': return 'text-green-400 border-green-400 bg-green-950/30';
      case 'warning': return 'text-yellow-400 border-yellow-400 bg-yellow-950/30';
      case 'drift': return 'text-red-400 border-red-400 bg-red-950/30';
      default: return 'text-slate-400 border-slate-400 bg-slate-950/30';
    }
  }, []);
  
  const getModelStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'active': return 'bg-green-950 text-green-400';
      case 'inactive': return 'bg-slate-950 text-slate-400';
      case 'degraded': return 'bg-orange-950 text-orange-400';
      default: return 'bg-slate-950 text-slate-400';
    }
  }, []);
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Model Performance Analytics...</p>
        </div>
      </div>
    );
  }
  
  if (!modelMetrics) return null;
  
  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono">MODEL PERFORMANCE & MONITORING</h1>
          <p className="text-slate-400 text-sm mt-1">
            Ensemble AI System • {ensembleModels.filter(m => m.status === 'active').length}/{ensembleModels.length} Models Active • 
            Last Update: {format(modelMetrics.lastUpdate, 'HH:mm:ss')}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {(['1H', '6H', '1D', '1W'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setSelectedTimeWindow(period)}
                className={`px-3 py-2 text-sm font-mono transition-colors ${
                  selectedTimeWindow === period
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                }`}
              >
                {period}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button 
              className="px-3 py-2 bg-slate-900 border border-amber-500/20 text-amber-500 rounded-lg hover:bg-amber-500/10 transition-colors flex items-center gap-2 text-sm"
              onClick={() => setIsRetraining(!isRetraining)}
              disabled={isRetraining}
            >
              {isRetraining ? <Pause className="h-4 w-4 animate-pulse" /> : <Play className="h-4 w-4" />}
              {isRetraining ? 'Training...' : 'Retrain'}
            </button>
            <button className="px-3 py-2 bg-slate-900 border border-amber-500/20 text-amber-500 rounded-lg hover:bg-amber-500/10 transition-colors flex items-center gap-2 text-sm">
              <Download className="h-4 w-4" />
              Export
            </button>
          </div>
        </div>
      </div>
      
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Accuracy
              <Target className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(modelMetrics.accuracy)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              F1: {modelMetrics.f1Score.toFixed(3)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Sharpe Ratio
              <BarChart3 className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {modelMetrics.sharpeRatio.toFixed(2)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Win Rate: {formatPercent(modelMetrics.winRate, 1)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Drift Score
              <Activity className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-xl font-bold font-mono ${
              modelMetrics.driftScore < 0.3 ? 'text-green-400' :
              modelMetrics.driftScore < 0.6 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {modelMetrics.driftScore.toFixed(3)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {driftDetections.filter(d => d.status === 'drift').length} features drifted
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Latency
              <Clock className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono">
              {modelMetrics.inferenceTime.toFixed(1)}ms
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {modelMetrics.throughput.toLocaleString()} pred/sec
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Stability
              <Shield className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-xl font-bold font-mono ${
              modelMetrics.stabilityIndex > 0.8 ? 'text-green-400' :
              modelMetrics.stabilityIndex > 0.6 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {formatPercent(modelMetrics.stabilityIndex)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Cal. Error: {formatPercent(modelMetrics.calibrationError, 1)}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Memory
              <Cpu className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono">
              {modelMetrics.memoryUsage.toFixed(1)}GB
            </div>
            <div className="text-xs text-slate-400 mt-1">
              AUC: {modelMetrics.auc.toFixed(3)}
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Performance Monitoring Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Performance */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Real-Time Model Performance</CardTitle>
            <p className="text-slate-400 text-sm">Accuracy, Latency & Drift Monitoring</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={performanceHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="timestamp" stroke="#64748B" fontSize={10} />
                <YAxis yAxisId="acc" stroke="#64748B" fontSize={10} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                <YAxis yAxisId="latency" orientation="right" stroke="#64748B" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any, name: string) => [
                    name === 'accuracy' ? formatPercent(value) : 
                    name === 'latency' ? `${value.toFixed(1)}ms` :
                    name === 'driftScore' ? value.toFixed(3) : value.toFixed(0),
                    name === 'accuracy' ? 'Accuracy' :
                    name === 'latency' ? 'Latency' :
                    name === 'driftScore' ? 'Drift Score' : name
                  ]}
                />
                <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#10B981" strokeWidth={2} dot={false} name="accuracy" />
                <Line yAxisId="latency" type="monotone" dataKey="latency" stroke="#F59E0B" strokeWidth={2} dot={false} name="latency" />
                <Line yAxisId="acc" type="monotone" dataKey="driftScore" stroke="#EF4444" strokeWidth={2} dot={false} name="driftScore" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        {/* SHAP Values */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">SHAP Feature Attribution</CardTitle>
            <p className="text-slate-400 text-sm">Explainable AI • Latest Prediction Breakdown</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shapValues} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#64748B" fontSize={10} />
                <YAxis dataKey="feature" type="category" stroke="#64748B" fontSize={10} width={100} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any) => [value.toFixed(4), 'SHAP Value']}
                />
                <Bar dataKey="shapValue">
                  {shapValues.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.shapValue >= 0 ? '#10B981' : '#EF4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Ensemble Models Dashboard */}
      <Card className="bg-slate-900 border-amber-500/20">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Ensemble Model Performance
          </CardTitle>
          <p className="text-slate-400 text-sm">Multi-Model Architecture • Weighted Predictions • Performance Tracking</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            {ensembleModels.map((model, index) => (
              <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h4 className="font-semibold text-white text-sm">{model.name}</h4>
                    <Badge className={getModelStatusColor(model.status)}>
                      {model.status.toUpperCase()}
                    </Badge>
                  </div>
                  <Badge className="bg-blue-950 text-blue-400 text-xs">
                    {model.type}
                  </Badge>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Weight:</span>
                    <span className="font-mono text-amber-400">{formatPercent(model.weight, 1)}</span>
                  </div>
                  <Progress value={model.weight * 100} className="h-2" />
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Performance:</span>
                    <span className="font-mono text-green-400">{formatPercent(model.performance)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Reliability:</span>
                    <span className="font-mono text-white">{formatPercent(model.reliability)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Latency:</span>
                    <span className="font-mono text-slate-400">{model.latency.toFixed(1)}ms</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      
      {/* Drift Detection & Model Calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Drift Detection */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Feature Drift Detection</CardTitle>
            <p className="text-slate-400 text-sm">
              Statistical Tests • Distribution Shifts • {driftDetections.filter(d => d.status === 'drift').length} Critical Alerts
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {driftDetections.map((drift, index) => (
                <Alert key={index} className={getDriftStatusColor(drift.status)}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className={`h-4 w-4 mt-0.5 ${
                        drift.status === 'drift' ? 'text-red-400' :
                        drift.status === 'warning' ? 'text-yellow-400' : 'text-green-400'
                      }`} />
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold text-white text-sm">{drift.feature}</span>
                          <Badge className={`text-xs ${
                            drift.status === 'drift' ? 'bg-red-950 text-red-400' :
                            drift.status === 'warning' ? 'bg-yellow-950 text-yellow-400' : 'bg-green-950 text-green-400'
                          }`}>
                            {drift.status.toUpperCase()}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-xs">
                          <div>
                            <span className="text-slate-400">Drift Magnitude: </span>
                            <span className="font-mono text-white">{drift.driftMagnitude.toFixed(3)}</span>
                          </div>
                          <div>
                            <span className="text-slate-400">P-Value: </span>
                            <span className="font-mono text-white">{drift.pValue.toFixed(4)}</span>
                          </div>
                        </div>
                        {drift.recommendation && (
                          <div className="text-amber-400 text-xs mt-2">
                            <strong>Action:</strong> {drift.recommendation}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-slate-500 text-xs">
                      {format(drift.timestamp, 'HH:mm')}
                    </div>
                  </div>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Model Calibration */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Model Calibration Analysis</CardTitle>
            <p className="text-slate-400 text-sm">Predicted vs Actual Probabilities • Reliability Curve</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={calibrationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="bin" stroke="#64748B" fontSize={10} />
                <YAxis stroke="#64748B" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any, name: string) => [`${value.toFixed(1)}%`, name === 'predicted' ? 'Predicted' : 'Actual']}
                />
                <Line type="monotone" dataKey="predicted" stroke="#F59E0B" strokeWidth={2} dot={{ fill: '#F59E0B' }} name="predicted" />
                <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} dot={{ fill: '#10B981' }} name="actual" />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#64748B" 
                  strokeDasharray="5 5" 
                  dot={false} 
                  name="Perfect Calibration"
                  data={calibrationData.map(d => ({ ...d, predicted: d.actual }))}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Feature Importance & Learning Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dynamic Feature Importance */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Dynamic Feature Importance</CardTitle>
            <p className="text-slate-400 text-sm">Ranked by Model Impact • Stability Tracking</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {featureImportance.map((feature, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-3 flex-1">
                    <div className="w-8 text-slate-400 text-sm font-mono text-center">
                      #{index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-white font-semibold text-sm">{feature.feature}</span>
                        <Badge className={`text-xs ${
                          feature.stability > 0.8 ? 'bg-green-950 text-green-400' :
                          feature.stability > 0.6 ? 'bg-yellow-950 text-yellow-400' : 'bg-red-950 text-red-400'
                        }`}>
                          {feature.stability > 0.8 ? 'Stable' : feature.stability > 0.6 ? 'Moderate' : 'Unstable'}
                        </Badge>
                      </div>
                      <Progress value={feature.importance * 100} className="h-2" />
                    </div>
                  </div>
                  <div className="text-right ml-4">
                    <div className="text-amber-400 font-mono text-sm">
                      {formatPercent(feature.importance, 1)}
                    </div>
                    <div className={`text-xs font-mono ${
                      feature.trend > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {feature.trend > 0 ? '↗' : '↘'} {formatPercent(Math.abs(feature.trend), 2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Learning Curves */}
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Training Progress & Learning Curves</CardTitle>
            <p className="text-slate-400 text-sm">Loss & Accuracy Evolution • Convergence Analysis</p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={learningCurves.slice(-50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#64748B" fontSize={10} />
                <YAxis yAxisId="loss" stroke="#64748B" fontSize={10} />
                <YAxis yAxisId="acc" orientation="right" stroke="#64748B" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                  formatter={(value: any, name: string) => [
                    name.includes('Loss') ? value.toFixed(4) : formatPercent(value),
                    name.replace(/([A-Z])/g, ' $1').trim()
                  ]}
                />
                <Line yAxisId="loss" type="monotone" dataKey="trainLoss" stroke="#EF4444" strokeWidth={2} dot={false} name="trainLoss" />
                <Line yAxisId="loss" type="monotone" dataKey="valLoss" stroke="#F59E0B" strokeWidth={2} dot={false} name="valLoss" />
                <Line yAxisId="acc" type="monotone" dataKey="trainAcc" stroke="#10B981" strokeWidth={2} dot={false} name="trainAcc" />
                <Line yAxisId="acc" type="monotone" dataKey="valAcc" stroke="#06B6D4" strokeWidth={2} dot={false} name="valAcc" />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      
      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Model Performance Analytics • Generated {format(new Date(), 'PPpp')} • 
          Framework: TensorFlow + Stable Baselines3 • Environment: Gymnasium • 
          Monitoring: MLflow + Evidently AI
        </p>
      </div>
    </div>
  );
}