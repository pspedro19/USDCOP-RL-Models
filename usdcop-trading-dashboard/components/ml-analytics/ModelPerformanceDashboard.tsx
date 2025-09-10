'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  Cell,
  PieChart,
  Pie
} from 'recharts';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Brain,
  Zap,
  Database,
  RefreshCw,
  Download,
  Settings
} from 'lucide-react';
import PredictionsVsActualsChart from './PredictionsVsActualsChart';
import FeatureImportanceChart from './FeatureImportanceChart';
import ModelHealthMonitoring from './ModelHealthMonitoring';

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  mse: number;
  mae: number;
  r2_score: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  total_returns?: number;
  win_rate?: number;
}

interface MLModel {
  run_id: string;
  name: string;
  version: string;
  stage: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  description?: string;
  status: string;
}

interface HealthStatus {
  model_id: string;
  model_name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  health_score: number;
  alerts: any[];
}

const ModelPerformanceDashboard: React.FC = () => {
  const [models, setModels] = useState<MLModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [healthData, setHealthData] = useState<any>(null);
  const [predictionData, setPredictionData] = useState<any[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<any[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelDetails(selectedModel.run_id);
    }
  }, [selectedModel]);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load models
      const modelsResponse = await fetch('/api/ml-analytics/models?action=list&limit=10');
      const modelsData = await modelsResponse.json();
      
      if (modelsData.success && modelsData.data.length > 0) {
        setModels(modelsData.data);
        setSelectedModel(modelsData.data[0]);
      }
      
      // Load system health
      const healthResponse = await fetch('/api/ml-analytics/health?action=summary');
      const healthData = await healthResponse.json();
      
      if (healthData.success) {
        setHealthData(healthData.data);
      }
      
    } catch (error) {
      console.error('Failed to load initial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModelDetails = async (runId: string) => {
    try {
      // Load model metrics
      const metricsResponse = await fetch(`/api/ml-analytics/models?action=metrics&runId=${runId}`);
      const metricsData = await metricsResponse.json();
      
      if (metricsData.success) {
        setModelMetrics(metricsData.data.metrics);
      }
      
      // Load prediction data
      const predictionsResponse = await fetch(`/api/ml-analytics/predictions?action=data&runId=${runId}&limit=50`);
      const predictionsData = await predictionsResponse.json();
      
      if (predictionsData.success) {
        setPredictionData(predictionsData.data);
      }
      
      // Load accuracy history
      const accuracyResponse = await fetch(`/api/ml-analytics/predictions?action=accuracy-over-time&runId=${runId}&limit=100`);
      const accuracyData = await accuracyResponse.json();
      
      if (accuracyData.success) {
        setAccuracyHistory(accuracyData.data);
      }
      
      // Load feature importance
      const featuresResponse = await fetch(`/api/ml-analytics/predictions?action=feature-impact&runId=${runId}`);
      const featuresData = await featuresResponse.json();
      
      if (featuresData.success) {
        setFeatureImportance(featuresData.data);
      }
      
    } catch (error) {
      console.error('Failed to load model details:', error);
    }
  };

  const getStatusBadge = (status: string) => {
    const statusConfig = {
      healthy: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
      warning: { color: 'bg-yellow-100 text-yellow-800', icon: AlertTriangle },
      critical: { color: 'bg-red-100 text-red-800', icon: AlertTriangle },
      offline: { color: 'bg-gray-100 text-gray-800', icon: Clock }
    };
    
    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.offline;
    const Icon = config.icon;
    
    return (
      <Badge className={`${config.color} flex items-center gap-1`}>
        <Icon size={12} />
        {status}
      </Badge>
    );
  };

  const formatMetric = (value: number | undefined, decimals = 2, suffix = '') => {
    if (value === undefined || value === null) return 'N/A';
    return `${value.toFixed(decimals)}${suffix}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">ML Model Analytics</h1>
          <p className="text-muted-foreground">
            Monitor model performance, accuracy, and health metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* System Health Overview */}
      {healthData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Models</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{healthData.summary.total_models}</div>
              <p className="text-xs text-muted-foreground">
                {healthData.summary.healthy_models} healthy
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">System Health</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {getStatusBadge(healthData.summary.overall_status)}
              </div>
              <Progress 
                value={
                  healthData.summary.overall_status === 'healthy' ? 95 :
                  healthData.summary.overall_status === 'warning' ? 70 : 40
                } 
                className="mt-2" 
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{healthData.summary.total_alerts}</div>
              <p className="text-xs text-muted-foreground">
                {healthData.summary.critical_alerts} critical
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Last Updated</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-sm">
                {new Date(healthData.summary.last_updated).toLocaleTimeString()}
              </div>
              <p className="text-xs text-muted-foreground">
                {new Date(healthData.summary.last_updated).toLocaleDateString()}
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Model Selection</CardTitle>
          <CardDescription>Choose a model to analyze performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {models.map((model) => (
              <Button
                key={model.run_id}
                variant={selectedModel?.run_id === model.run_id ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedModel(model)}
                className="flex items-center gap-2"
              >
                <Target className="h-4 w-4" />
                {model.name}
                <Badge variant="secondary" className="text-xs">
                  {model.version}
                </Badge>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Main Analytics Tabs */}
      {selectedModel && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="predictions">Predictions</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="health">Health</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Model Metrics Cards */}
            {modelMetrics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
                    <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{formatMetric(modelMetrics.accuracy * 100, 1, '%')}</div>
                    <Progress value={modelMetrics.accuracy * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Precision</CardTitle>
                    <Target className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{formatMetric(modelMetrics.precision * 100, 1, '%')}</div>
                    <Progress value={modelMetrics.precision * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">F1 Score</CardTitle>
                    <Zap className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{formatMetric(modelMetrics.f1_score * 100, 1, '%')}</div>
                    <Progress value={modelMetrics.f1_score * 100} className="mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                    <TrendingUp className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{formatMetric(modelMetrics.sharpe_ratio, 2)}</div>
                    <p className="text-xs text-muted-foreground mt-2">Risk-adjusted returns</p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Model Information */}
            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Name:</span>
                      <span className="text-sm">{selectedModel.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Version:</span>
                      <Badge variant="outline">{selectedModel.version}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Stage:</span>
                      <Badge>{selectedModel.stage}</Badge>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Created:</span>
                      <span className="text-sm">
                        {new Date(selectedModel.creation_timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Updated:</span>
                      <span className="text-sm">
                        {new Date(selectedModel.last_updated_timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Status:</span>
                      {getStatusBadge(selectedModel.status.toLowerCase())}
                    </div>
                  </div>
                </div>
                {selectedModel.description && (
                  <div className="mt-4">
                    <span className="text-sm font-medium">Description:</span>
                    <p className="text-sm text-muted-foreground mt-1">
                      {selectedModel.description}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="accuracy" className="space-y-4">
            {/* Accuracy over time chart will be implemented next */}
            <Card>
              <CardHeader>
                <CardTitle>Accuracy Over Time</CardTitle>
                <CardDescription>Model accuracy trends and performance windows</CardDescription>
              </CardHeader>
              <CardContent>
                {accuracyHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={accuracyHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="window_start" 
                        tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <YAxis domain={[0, 100]} />
                      <Tooltip 
                        labelFormatter={(value) => `Time: ${new Date(value).toLocaleString()}`}
                        formatter={(value, name) => [`${value}%`, name]}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                        name="Accuracy"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="direction_accuracy" 
                        stroke="#82ca9d" 
                        strokeWidth={2}
                        name="Direction Accuracy"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No accuracy data available
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="predictions">
            <PredictionsVsActualsChart 
              runId={selectedModel?.run_id}
              timeRange="24h"
              limit={100}
            />
          </TabsContent>

          <TabsContent value="features">
            <FeatureImportanceChart 
              runId={selectedModel?.run_id}
              modelType={selectedModel?.name.includes('PPO') ? 'PPO' : 'Unknown'}
            />
          </TabsContent>

          <TabsContent value="health">
            <ModelHealthMonitoring />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default ModelPerformanceDashboard;