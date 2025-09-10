'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  RefreshCw, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock,
  Activity,
  Database,
  Zap,
  TrendingUp,
  Server,
  AlertCircle
} from 'lucide-react';

// Types
interface PipelineStageHealth {
  stage: string;
  status: 'HEALTHY' | 'WARNING' | 'ERROR' | 'UNKNOWN';
  last_run: string | null;
  processing_time: number | null;
  records_processed: number;
  error_count: number;
  error_rate: number;
  data_completeness: number;
  quality_score: number;
  next_scheduled: string | null;
  dag_status: string;
}

interface SystemHealth {
  overall_status: string;
  pipeline_availability: number;
  average_processing_time: number;
  total_error_rate: number;
  data_freshness: number;
  storage_usage: Record<string, number>;
  active_dags: number;
  failed_dags: number;
}

interface DataFlowMetrics {
  source_stage: string;
  target_stage: string;
  records_transferred: number;
  transfer_time: number;
  data_quality_delta: number;
  last_transfer: string;
}

interface Alert {
  id: string;
  stage: string;
  severity: 'ERROR' | 'WARNING' | 'INFO';
  message: string;
  metric: string;
  value: number;
  timestamp: string;
}

interface MetricsSummary {
  timestamp: string;
  overall_status: string;
  pipeline_summary: {
    total_stages: number;
    healthy_stages: number;
    warning_stages: number;
    error_stages: number;
    availability_percentage: number;
  };
  processing_summary: {
    total_records_processed: number;
    total_errors: number;
    overall_error_rate: number;
    average_processing_time: number;
  };
  quality_summary: {
    average_quality_score: number;
    average_completeness: number;
    data_freshness_hours: number;
  };
  data_flow_summary: {
    total_records_transferred: number;
    average_transfer_time: number;
    active_flows: number;
  };
  infrastructure_summary: {
    active_dags: number;
    failed_dags: number;
    storage_usage_mb: Record<string, number>;
  };
}

const API_BASE_URL = process.env.NEXT_PUBLIC_PIPELINE_HEALTH_API || 'http://localhost:8002';

const PipelineHealthDashboard: React.FC = () => {
  const [pipelineHealth, setPipelineHealth] = useState<PipelineStageHealth[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [dataFlowMetrics, setDataFlowMetrics] = useState<DataFlowMetrics[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<Alert[]>([]);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);

  // Fetch all health data
  const fetchHealthData = useCallback(async () => {
    try {
      setError(null);
      
      const [
        pipelineRes,
        systemRes, 
        dataFlowRes,
        alertsRes,
        summaryRes
      ] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/pipeline/health`),
        fetch(`${API_BASE_URL}/api/v1/system/health`),
        fetch(`${API_BASE_URL}/api/v1/pipeline/dataflow`),
        fetch(`${API_BASE_URL}/api/v1/alerts/active`),
        fetch(`${API_BASE_URL}/api/v1/metrics/summary`)
      ]);

      if (!pipelineRes.ok || !systemRes.ok) {
        throw new Error('Failed to fetch health data');
      }

      const [pipeline, system, dataFlow, alerts, summary] = await Promise.all([
        pipelineRes.json(),
        systemRes.json(),
        dataFlowRes.ok ? dataFlowRes.json() : [],
        alertsRes.ok ? alertsRes.json() : { alerts: [] },
        summaryRes.ok ? summaryRes.json() : null
      ]);

      setPipelineHealth(pipeline);
      setSystemHealth(system);
      setDataFlowMetrics(dataFlow);
      setActiveAlerts(alerts.alerts || []);
      setMetricsSummary(summary);
      setLastUpdate(new Date());
      
    } catch (err) {
      console.error('Error fetching health data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Auto refresh effect
  useEffect(() => {
    fetchHealthData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchHealthData, 30000); // 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh, fetchHealthData]);

  // Manual refresh
  const handleRefresh = () => {
    setIsLoading(true);
    fetchHealthData();
  };

  // Status color mapping
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'HEALTHY': return 'bg-green-500';
      case 'WARNING': return 'bg-yellow-500';
      case 'ERROR': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'HEALTHY': return <CheckCircle className="w-4 h-4" />;
      case 'WARNING': return <AlertTriangle className="w-4 h-4" />;
      case 'ERROR': return <XCircle className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };

  // Format helpers
  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  const formatNumber = (value: number) => new Intl.NumberFormat().format(value);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Pipeline Health Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Real-time monitoring of L0-L6 pipeline stages
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge 
            className={systemHealth ? getStatusColor(systemHealth.overall_status) : 'bg-gray-500'}
          >
            {systemHealth?.overall_status || 'UNKNOWN'}
          </Badge>
          <Button 
            onClick={handleRefresh}
            disabled={isLoading}
            size="sm"
            variant="outline"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Auto refresh</span>
          </label>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <div>
            <strong>Error:</strong> {error}
          </div>
        </Alert>
      )}

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <Card className="p-4 border-l-4 border-l-red-500">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-red-600 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              Active Alerts ({activeAlerts.length})
            </h3>
          </div>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {activeAlerts.map((alert) => (
              <div key={alert.id} className="flex items-center justify-between p-2 bg-red-50 rounded">
                <div>
                  <span className={`font-semibold ${alert.severity === 'ERROR' ? 'text-red-600' : 'text-yellow-600'}`}>
                    [{alert.severity}] {alert.stage}
                  </span>
                  <p className="text-sm text-gray-600">{alert.message}</p>
                </div>
                <span className="text-xs text-gray-500">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* System Overview Cards */}
      {metricsSummary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Pipeline Availability</p>
                <p className="text-2xl font-bold">
                  {metricsSummary.pipeline_summary.availability_percentage.toFixed(1)}%
                </p>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>
            <div className="mt-2">
              <div className="flex space-x-1 text-xs">
                <span className="text-green-600">
                  {metricsSummary.pipeline_summary.healthy_stages} Healthy
                </span>
                <span className="text-yellow-600">
                  {metricsSummary.pipeline_summary.warning_stages} Warning
                </span>
                <span className="text-red-600">
                  {metricsSummary.pipeline_summary.error_stages} Error
                </span>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Data Processed</p>
                <p className="text-2xl font-bold">
                  {formatNumber(metricsSummary.processing_summary.total_records_processed)}
                </p>
              </div>
              <Database className="w-8 h-8 text-green-500" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-600">
                Error Rate: {metricsSummary.processing_summary.overall_error_rate.toFixed(2)}%
              </p>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Processing Time</p>
                <p className="text-2xl font-bold">
                  {formatDuration(metricsSummary.processing_summary.average_processing_time)}
                </p>
              </div>
              <Clock className="w-8 h-8 text-purple-500" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-600">
                Quality Score: {metricsSummary.quality_summary.average_quality_score.toFixed(1)}/100
              </p>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Data Freshness</p>
                <p className="text-2xl font-bold">
                  {metricsSummary.quality_summary.data_freshness_hours.toFixed(1)}h
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-orange-500" />
            </div>
            <div className="mt-2">
              <p className="text-xs text-gray-600">
                Completeness: {metricsSummary.quality_summary.average_completeness.toFixed(1)}%
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Pipeline Stages Overview */}
      <Card className="p-6">
        <h3 className="text-xl font-semibold mb-4">Pipeline Stages Health</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {pipelineHealth.map((stage) => (
            <Card 
              key={stage.stage} 
              className={`p-4 cursor-pointer transition-all hover:shadow-lg ${
                selectedStage === stage.stage ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedStage(selectedStage === stage.stage ? null : stage.stage)}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-sm">{stage.stage}</h4>
                <Badge className={getStatusColor(stage.status)}>
                  {getStatusIcon(stage.status)}
                </Badge>
              </div>
              
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Quality Score:</span>
                  <span className="font-medium">{stage.quality_score.toFixed(1)}/100</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Records:</span>
                  <span className="font-medium">{formatNumber(stage.records_processed)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Completeness:</span>
                  <span className="font-medium">{formatPercentage(stage.data_completeness)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Processing Time:</span>
                  <span className="font-medium">{formatDuration(stage.processing_time)}</span>
                </div>
                {stage.error_count > 0 && (
                  <div className="flex justify-between text-red-600">
                    <span>Errors:</span>
                    <span className="font-medium">{stage.error_count}</span>
                  </div>
                )}
              </div>
              
              {stage.last_run && (
                <div className="mt-2 pt-2 border-t border-gray-200">
                  <p className="text-xs text-gray-500">
                    Last run: {new Date(stage.last_run).toLocaleString()}
                  </p>
                </div>
              )}
            </Card>
          ))}
        </div>
      </Card>

      {/* Data Flow Visualization */}
      {dataFlowMetrics.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Data Flow Transfer Times</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dataFlowMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="target_stage"
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value: any) => [formatDuration(value), 'Transfer Time']}
                    labelFormatter={(label) => `To: ${label}`}
                  />
                  <Bar dataKey="transfer_time" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Records Transferred</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={dataFlowMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="target_stage"
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value: any) => [formatNumber(value), 'Records']}
                    labelFormatter={(label) => `To: ${label}`}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="records_transferred" 
                    stroke="#10b981" 
                    fill="#10b981" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      )}

      {/* Quality and Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Quality Scores by Stage</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pipelineHealth}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="stage"
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value: any) => [`${value}/100`, 'Quality Score']}
                />
                <Bar 
                  dataKey="quality_score" 
                  fill="#8b5cf6"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Processing Times</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pipelineHealth.filter(s => s.processing_time)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="stage"
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value: any) => [formatDuration(value), 'Processing Time']}
                />
                <Line 
                  type="monotone" 
                  dataKey="processing_time" 
                  stroke="#f59e0b" 
                  strokeWidth={3}
                  dot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Storage Usage */}
      {systemHealth?.storage_usage && (
        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Storage Usage by Stage</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {Object.entries(systemHealth.storage_usage).map(([stage, usage]) => (
              <div key={stage} className="text-center p-3 bg-gray-50 rounded">
                <div className="flex items-center justify-center mb-2">
                  <Server className="w-6 h-6 text-gray-600" />
                </div>
                <p className="text-sm font-medium">{stage}</p>
                <p className="text-lg font-bold">{usage.toFixed(1)} MB</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Footer */}
      <div className="text-center text-sm text-gray-500">
        Last updated: {lastUpdate.toLocaleString()} | 
        Auto refresh: {autoRefresh ? 'ON' : 'OFF'} | 
        Update interval: 30 seconds
      </div>
    </div>
  );
};

export default PipelineHealthDashboard;