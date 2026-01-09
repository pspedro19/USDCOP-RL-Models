'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
  AreaChart,
  Area
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Cpu,
  HardDrive,
  Zap,
  Wifi,
  WifiOff,
  RefreshCw,
  Bell,
  BellOff,
  Eye,
  Gauge
} from 'lucide-react';

interface ModelAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'drift' | 'performance' | 'resource' | 'availability';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  model_id?: string;
  model_name?: string;
  threshold_value?: number;
  current_value?: number;
}

interface ModelHealthMetrics {
  model_id: string;
  model_name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  health_score: number;
  last_prediction_time: string;
  metrics: {
    prediction_latency: number;
    throughput: number;
    error_rate: number;
    drift_score: number;
    confidence_avg: number;
  };
  resource_usage: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
  alerts: ModelAlert[];
}

interface SystemHealthSummary {
  overall_status: 'healthy' | 'warning' | 'critical';
  total_models: number;
  healthy_models: number;
  models_with_warnings: number;
  critical_models: number;
  offline_models: number;
  total_alerts: number;
  critical_alerts: number;
}

const ModelHealthMonitoring: React.FC = () => {
  const [healthData, setHealthData] = useState<{
    summary: SystemHealthSummary;
    models: ModelHealthMetrics[];
  } | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelHealthMetrics | null>(null);
  const [alerts, setAlerts] = useState<ModelAlert[]>([]);
  const [healthHistory, setHealthHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadHealthData();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadModelHealthHistory(selectedModel.model_id);
    }
  }, [selectedModel]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(loadHealthData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const loadHealthData = async () => {
    try {
      setLoading(true);
      
      // Load system health summary
      const summaryResponse = await fetch('/api/ml-analytics/health?action=summary');
      const summaryData = await summaryResponse.json();
      
      if (summaryData.success) {
        setHealthData(summaryData.data);
        if (summaryData.data.models && summaryData.data.models.length > 0) {
          // Load detailed health for each model
          const detailedModels = await Promise.all(
            summaryData.data.models.slice(0, 5).map(async (model: any) => {
              const detailResponse = await fetch(
                `/api/ml-analytics/health?action=detail&modelId=${model.model_id}`
              );
              const detailData = await detailResponse.json();
              return detailData.success ? detailData.data : model;
            })
          );
          
          setHealthData(prev => prev ? {
            ...prev,
            models: detailedModels
          } : null);
          
          if (!selectedModel && detailedModels.length > 0) {
            setSelectedModel(detailedModels[0]);
          }
        }
      }
      
      // Load all alerts
      const alertsResponse = await fetch('/api/ml-analytics/health?action=alerts');
      const alertsData = await alertsResponse.json();
      
      if (alertsData.success) {
        setAlerts(alertsData.data.alerts);
      }
      
    } catch (error) {
      console.error('Failed to load health data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModelHealthHistory = async (modelId: string) => {
    try {
      const response = await fetch(
        `/api/ml-analytics/health?action=metrics-history&modelId=${modelId}`
      );
      const data = await response.json();
      
      if (data.success) {
        setHealthHistory(data.data.metrics);
      }
    } catch (error) {
      console.error('Failed to load health history:', error);
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch('/api/ml-analytics/health', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'acknowledge-alert',
          alert_id: alertId
        })
      });
      
      if (response.ok) {
        setAlerts(prev => 
          prev.map(alert => 
            alert.id === alertId ? { ...alert, acknowledged: true } : alert
          )
        );
      }
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    const icons = {
      healthy: <CheckCircle className="h-4 w-4 text-green-500" />,
      warning: <AlertTriangle className="h-4 w-4 text-yellow-500" />,
      critical: <AlertTriangle className="h-4 w-4 text-red-500" />,
      offline: <WifiOff className="h-4 w-4 text-gray-500" />
    };
    return icons[status as keyof typeof icons] || icons.offline;
  };

  const getStatusBadge = (status: string) => {
    const configs = {
      healthy: { color: 'bg-green-100 text-green-800 border-green-200' },
      warning: { color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
      critical: { color: 'bg-red-100 text-red-800 border-red-200' },
      offline: { color: 'bg-gray-100 text-gray-800 border-gray-200' }
    };
    
    const config = configs[status as keyof typeof configs] || configs.offline;
    
    return (
      <Badge className={`${config.color} flex items-center gap-1`}>
        {getStatusIcon(status)}
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const getAlertIcon = (type: string) => {
    const icons = {
      drift: <TrendingUp className="h-4 w-4" />,
      performance: <Zap className="h-4 w-4" />,
      resource: <Cpu className="h-4 w-4" />,
      availability: <Wifi className="h-4 w-4" />
    };
    return icons[type as keyof typeof icons] || <AlertTriangle className="h-4 w-4" />;
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
          <h2 className="text-2xl font-bold tracking-tight">Model Health Monitoring</h2>
          <p className="text-muted-foreground">
            Real-time monitoring of model performance and system health
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={autoRefresh ? 'default' : 'outline'}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? <Bell className="h-4 w-4 mr-2" /> : <BellOff className="h-4 w-4 mr-2" />}
            Auto Refresh
          </Button>
          <Button onClick={loadHealthData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Health Overview */}
      {healthData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">System Status</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {getStatusBadge(healthData.summary.overall_status)}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {healthData.summary.healthy_models}/{healthData.summary.total_models} models healthy
              </p>
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
              <CardTitle className="text-sm font-medium">Models Online</CardTitle>
              <Wifi className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {healthData.summary.total_models - healthData.summary.offline_models}
              </div>
              <Progress 
                value={((healthData.summary.total_models - healthData.summary.offline_models) / healthData.summary.total_models) * 100}
                className="mt-2"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Issues</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {healthData.summary.models_with_warnings + healthData.summary.critical_models}
              </div>
              <p className="text-xs text-muted-foreground">
                {healthData.summary.critical_models} critical issues
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Model Selection */}
      {healthData && (
        <Card>
          <CardHeader>
            <CardTitle>Model Health Status</CardTitle>
            <CardDescription>Select a model to view detailed health metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {healthData.models.map((model) => (
                <Card 
                  key={model.model_id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel?.model_id === model.model_id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedModel(model)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-sm">{model.model_name}</h3>
                      {getStatusBadge(model.status)}
                    </div>
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Health Score</span>
                      <span className="font-mono">{model.health_score}%</span>
                    </div>
                    <Progress value={model.health_score} className="mt-1 h-1" />
                    {model.alerts.length > 0 && (
                      <div className="mt-2 text-xs text-red-600">
                        {model.alerts.length} alert{model.alerts.length !== 1 ? 's' : ''}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Selected Model Details */}
      {selectedModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>{selectedModel.model_name} - Current Status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Prediction Latency</div>
                    <div className="text-2xl font-bold">{selectedModel.metrics.prediction_latency}ms</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Throughput</div>
                    <div className="text-2xl font-bold">{selectedModel.metrics.throughput}/min</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Error Rate</div>
                    <div className="text-2xl font-bold">{selectedModel.metrics.error_rate.toFixed(1)}%</div>
                    <Progress value={selectedModel.metrics.error_rate} className="mt-1" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-muted-foreground">Avg Confidence</div>
                    <div className="text-2xl font-bold">{(selectedModel.metrics.confidence_avg * 100).toFixed(1)}%</div>
                    <Progress value={selectedModel.metrics.confidence_avg * 100} className="mt-1" />
                  </div>
                </div>
                
                <div>
                  <div className="text-sm font-medium text-muted-foreground">Data Drift Score</div>
                  <div className="text-2xl font-bold">{(selectedModel.metrics.drift_score * 100).toFixed(1)}%</div>
                  <Progress 
                    value={selectedModel.metrics.drift_score * 100} 
                    className="mt-1"
                  />
                  {selectedModel.metrics.drift_score > 0.2 && (
                    <p className="text-xs text-red-600 mt-1">
                      High drift detected - model may need retraining
                    </p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Resource Usage */}
          <Card>
            <CardHeader>
              <CardTitle>Resource Usage</CardTitle>
              <CardDescription>System resource consumption</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4" />
                      <span>CPU Usage</span>
                    </div>
                    <span className="font-mono">{selectedModel.resource_usage.cpu_usage}%</span>
                  </div>
                  <Progress value={selectedModel.resource_usage.cpu_usage} className="mt-2" />
                </div>
                
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      <span>Memory Usage</span>
                    </div>
                    <span className="font-mono">{selectedModel.resource_usage.memory_usage} MB</span>
                  </div>
                  <Progress 
                    value={Math.min((selectedModel.resource_usage.memory_usage / 2000) * 100, 100)} 
                    className="mt-2" 
                  />
                </div>
                
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4" />
                      <span>Disk Usage</span>
                    </div>
                    <span className="font-mono">{selectedModel.resource_usage.disk_usage} MB</span>
                  </div>
                  <Progress 
                    value={Math.min((selectedModel.resource_usage.disk_usage / 1000) * 100, 100)} 
                    className="mt-2" 
                  />
                </div>
                
                <div className="text-xs text-muted-foreground">
                  Last prediction: {new Date(selectedModel.last_prediction_time).toLocaleString()}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Health History Chart */}
      {healthHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Health Score History</CardTitle>
            <CardDescription>24-hour health score trend for {selectedModel?.model_name}</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={healthHistory}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis 
                  dataKey="timestamp"
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(value) => `Time: ${new Date(value).toLocaleString()}`}
                  formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, 'Health Score']}
                />
                <Area 
                  type="monotone" 
                  dataKey="health_score" 
                  stroke="#8884d8" 
                  fill="#8884d8" 
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Active Alerts */}
      <Card>
        <CardHeader>
          <CardTitle>Active Alerts</CardTitle>
          <CardDescription>Recent alerts across all models</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alerts.filter(alert => !alert.acknowledged).slice(0, 10).map((alert) => (
              <Alert 
                key={alert.id} 
                className={`${
                  alert.severity === 'critical' ? 'border-red-200 bg-red-50' :
                  alert.severity === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                  'border-blue-200 bg-blue-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    {getAlertIcon(alert.type)}
                    <div>
                      <AlertTitle className="text-sm">{alert.title}</AlertTitle>
                      <AlertDescription className="text-xs">
                        {alert.message}
                        {alert.model_name && (
                          <div className="mt-1 text-muted-foreground">
                            Model: {alert.model_name}
                          </div>
                        )}
                        <div className="mt-1 text-muted-foreground">
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                      </AlertDescription>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => acknowledgeAlert(alert.id)}
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    Acknowledge
                  </Button>
                </div>
              </Alert>
            ))}
            {alerts.filter(alert => !alert.acknowledged).length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500 opacity-50" />
                <p>No active alerts</p>
                <p className="text-xs">All systems are running normally</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelHealthMonitoring;