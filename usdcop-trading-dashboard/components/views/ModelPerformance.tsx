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
  // ⚠️ COMPONENT DISABLED - ALL DATA WAS SIMULATED WITH Math.random()
  // This component previously used 50+ Math.random() calls and hardcoded values
  // To enable: Connect to ML Analytics API (port 8005) endpoints

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
  const [loading, setLoading] = useState(false);

  // ❌ ALL MOCK DATA GENERATION REMOVED - WAS 100% SIMULATED
  // This component previously generated all data with Math.random() and hardcoded values
  // Including: mockMetrics, ensemble models, drift detection, SHAP values,
  // performance history, feature importance, predictions, calibration, learning curves

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
      <Card className="bg-slate-900 border-red-500/40 max-w-3xl w-full">
        <CardHeader>
          <CardTitle className="text-red-500 font-mono text-2xl flex items-center gap-3">
            <AlertTriangle className="h-8 w-8" />
            MODEL PERFORMANCE COMPONENT DISABLED
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <Alert className="bg-red-950/30 border-red-500/40">
            <AlertDescription className="text-white">
              <div className="space-y-4">
                <p className="font-semibold text-lg">
                  This component has been disabled because it was displaying 100% simulated data.
                </p>

                <div className="bg-slate-800 rounded-lg p-4 space-y-2">
                  <p className="text-amber-400 font-mono text-sm font-semibold">Issues Found:</p>
                  <ul className="list-disc list-inside space-y-1 text-slate-300 text-sm">
                    <li>50+ calls to Math.random() for generating fake metrics</li>
                    <li>Hardcoded model performance values (accuracy: 0.847, sharpe: 2.34, etc.)</li>
                    <li>Simulated ensemble models (PPO, SAC, XGBoost, LightGBM, TD3)</li>
                    <li>Mock drift detection alerts with fabricated p-values</li>
                    <li>Fake SHAP values and feature importance</li>
                    <li>Randomly generated learning curves and calibration data</li>
                  </ul>
                </div>

                <div className="bg-slate-800 rounded-lg p-4 space-y-2">
                  <p className="text-green-400 font-mono text-sm font-semibold">To Enable This Component:</p>
                  <ol className="list-decimal list-inside space-y-2 text-slate-300 text-sm">
                    <li>Connect to <span className="text-amber-400 font-mono">ML Analytics API</span> (port 8005)</li>
                    <li>Implement endpoints for:
                      <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                        <li><code className="text-amber-400">/api/models/performance</code> - Core metrics</li>
                        <li><code className="text-amber-400">/api/models/ensemble</code> - Ensemble status</li>
                        <li><code className="text-amber-400">/api/monitoring/drift</code> - Drift detection</li>
                        <li><code className="text-amber-400">/api/monitoring/shap</code> - SHAP values</li>
                        <li><code className="text-amber-400">/api/models/history</code> - Performance history</li>
                      </ul>
                    </li>
                    <li>Replace mock data generation with real API calls</li>
                    <li>Verify all metrics come from actual model training/inference</li>
                  </ol>
                </div>

                <div className="bg-blue-950/30 border border-blue-500/40 rounded-lg p-4">
                  <p className="text-blue-400 text-sm">
                    <strong>Note:</strong> This component has been disabled as part of the system-wide cleanup
                    to eliminate all hardcoded and simulated values. The system now enforces
                    <span className="text-amber-400 font-mono"> 100% real data from backend APIs</span>.
                  </p>
                </div>
              </div>
            </AlertDescription>
          </Alert>

          <div className="text-center text-slate-500 text-sm font-mono">
            Component disabled on {format(new Date(), 'PPpp')}<br />
            See <code className="text-amber-400">components/views/ModelPerformance.tsx</code> for implementation details
          </div>
        </CardContent>
      </Card>
    </div>
  );
}