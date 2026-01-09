'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart, ComposedChart
} from 'recharts';
import { 
  enhancedApiMonitor, 
  APIKeyUsage, 
  APIUsageStats, 
  APIAlert,
  APIHealthMetrics,
  MonitoringStatus
} from '@/lib/services/enhanced-api-monitor';
import { RefreshCw, AlertTriangle, CheckCircle, XCircle, Clock, DollarSign, Activity } from 'lucide-react';

export default function EnhancedAPIUsageDashboard() {
  const [monitoringStatus, setMonitoringStatus] = useState<MonitoringStatus | null>(null);
  const [usageStats, setUsageStats] = useState<APIUsageStats | null>(null);
  const [alerts, setAlerts] = useState<APIAlert[]>([]);
  const [callHistory, setCallHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<6 | 24 | 168>(24);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  const fetchAPIData = async () => {
    try {
      setIsLoading(true);
      const [statusData, statsData, alertsData, historyData] = await Promise.all([
        enhancedApiMonitor.getMonitoringStatus(),
        enhancedApiMonitor.getUsageStats(),
        enhancedApiMonitor.getAlerts(),
        enhancedApiMonitor.getCallHistoryForChart(selectedTimeframe),
      ]);
      
      setMonitoringStatus(statusData);
      setUsageStats(statsData);
      setAlerts(alertsData.alerts);
      setCallHistory(historyData);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching API monitoring data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAPIData();
    
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(fetchAPIData, 60000); // Update every minute
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedTimeframe, autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ACTIVE': return 'bg-green-500 text-white';
      case 'WARNING': return 'bg-yellow-500 text-white';
      case 'RATE_LIMITED': return 'bg-orange-500 text-white';
      case 'EXPIRED': return 'bg-red-500 text-white';
      case 'ERROR': return 'bg-red-600 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACTIVE': return <CheckCircle className="w-4 h-4" />;
      case 'WARNING': return <AlertTriangle className="w-4 h-4" />;
      case 'RATE_LIMITED': return <Clock className="w-4 h-4" />;
      case 'ERROR': return <XCircle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 3,
    }).format(amount);
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'bg-red-100 border-red-500 text-red-800';
      case 'WARNING': return 'bg-yellow-100 border-yellow-500 text-yellow-800';
      default: return 'bg-blue-100 border-blue-500 text-blue-800';
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

  if (isLoading && !monitoringStatus) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading API monitoring data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">API Usage Monitoring</h2>
          <p className="text-sm text-gray-500">
            Last updated: {lastUpdated.toLocaleString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium">Timeframe:</label>
            <select 
              value={selectedTimeframe} 
              onChange={(e) => setSelectedTimeframe(Number(e.target.value) as any)}
              className="px-3 py-1 border rounded-md text-sm"
            >
              <option value={6}>Last 6 Hours</option>
              <option value={24}>Last 24 Hours</option>
              <option value={168}>Last Week</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="autoRefresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="autoRefresh" className="text-sm font-medium">Auto Refresh</label>
          </div>
          
          <Button
            onClick={fetchAPIData}
            disabled={isLoading}
            variant="outline"
            size="sm"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Critical Alerts */}
      {alerts.length > 0 && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4 text-red-500" />
          <div className="ml-2">
            <h4 className="font-semibold text-red-800">Active Alerts ({alerts.length})</h4>
            <div className="mt-2 space-y-1">
              {alerts.slice(0, 3).map((alert, index) => (
                <div key={index} className={`p-2 rounded text-sm ${getSeverityColor(alert.severity)}`}>
                  <strong>{alert.severity}:</strong> {alert.message}
                </div>
              ))}
              {alerts.length > 3 && (
                <p className="text-sm text-red-600">
                  + {alerts.length - 3} more alerts
                </p>
              )}
            </div>
          </div>
        </Alert>
      )}

      {/* Summary Statistics */}
      {usageStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">API Calls Today</p>
                <p className="text-2xl font-bold">{usageStats.total_calls_today.toLocaleString()}</p>
                <p className="text-xs text-gray-500">
                  Month: {usageStats.total_calls_month.toLocaleString()}
                </p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Cost Today</p>
                <p className="text-2xl font-bold">{formatCurrency(usageStats.total_cost_today)}</p>
                <p className="text-xs text-gray-500">
                  Month: {formatCurrency(usageStats.total_cost_month)}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Keys</p>
                <p className="text-2xl font-bold text-green-600">{usageStats.active_keys}</p>
                <p className="text-xs text-gray-500">
                  Rate Limited: {usageStats.rate_limited_keys}
                </p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold">{usageStats.success_rate.toFixed(1)}%</p>
                <p className="text-xs text-gray-500">
                  Avg Latency: {usageStats.average_latency_ms}ms
                </p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </Card>
        </div>
      )}

      {/* API Key Status Grid */}
      {monitoringStatus && (
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">API Key Status</h3>
            <div className="flex space-x-2 text-sm">
              <span className="flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
                Active: {monitoringStatus.summary.active_keys}
              </span>
              <span className="flex items-center">
                <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
                Warning: {monitoringStatus.summary.warning_keys}
              </span>
              <span className="flex items-center">
                <div className="w-3 h-3 bg-orange-500 rounded-full mr-1"></div>
                Rate Limited: {monitoringStatus.summary.rate_limited_keys}
              </span>
              <span className="flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
                Error: {monitoringStatus.summary.error_keys}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {monitoringStatus.key_statuses.map((usage) => (
              <div key={usage.key_id} className="border rounded-lg p-4 bg-gray-50">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <Badge className={getStatusColor(usage.status)}>
                      {getStatusIcon(usage.status)}
                      <span className="ml-1">{usage.status}</span>
                    </Badge>
                    <div>
                      <p className="font-medium">{usage.key_id}</p>
                      <p className="text-sm text-gray-600">{usage.api_name} • {usage.plan_type}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-bold">{formatCurrency(usage.daily_cost)}</p>
                    <p className="text-xs text-gray-500">daily cost</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Daily Usage:</span>
                    <span>{usage.daily_calls}/{usage.daily_limit}</span>
                  </div>
                  <Progress 
                    value={(usage.daily_calls / usage.daily_limit) * 100} 
                    className="h-2"
                  />

                  <div className="grid grid-cols-2 gap-4 mt-3 text-sm">
                    <div>
                      <p className="text-gray-600">Success Rate</p>
                      <p className="font-medium">{usage.success_rate.toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Avg Response</p>
                      <p className="font-medium">{usage.avg_response_time.toFixed(0)}ms</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Errors</p>
                      <p className="font-medium">{usage.error_count}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Last Used</p>
                      <p className="font-medium text-xs">
                        {usage.last_used ? new Date(usage.last_used).toLocaleTimeString() : 'Never'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* API Calls Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">API Calls Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={callHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="hour" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any, name) => [
                    name === 'cost' ? formatCurrency(value) : value,
                    name === 'calls' ? 'API Calls' : name === 'cost' ? 'Cost' : name
                  ]}
                />
                <Bar yAxisId="left" dataKey="calls" fill="#3b82f6" />
                <Line yAxisId="right" type="monotone" dataKey="cost" stroke="#10b981" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Success Rate Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Success Rate & Response Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={callHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="hour" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                />
                <YAxis yAxisId="left" domain={[90, 100]} />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any, name) => [
                    name === 'successRate' ? `${value.toFixed(1)}%` : `${value}ms`,
                    name === 'successRate' ? 'Success Rate' : 'Avg Response Time'
                  ]}
                />
                <Area 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="successRate" 
                  fill="#10b981" 
                  fillOpacity={0.6}
                  stroke="#10b981"
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="avgResponseTime" 
                  stroke="#f59e0b" 
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Health Metrics per API */}
      {monitoringStatus && Object.keys(monitoringStatus.health_metrics).length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">API Health Metrics</h3>
          <div className="space-y-4">
            {Object.entries(monitoringStatus.health_metrics).map(([apiName, metrics]) => (
              <div key={apiName} className="border rounded-lg p-4 bg-gradient-to-r from-blue-50 to-indigo-50">
                <h4 className="font-semibold text-lg mb-3 capitalize">{apiName} API</h4>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Uptime</p>
                    <p className="text-lg font-bold text-green-600">{metrics.uptime_percentage.toFixed(2)}%</p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Current RPS</p>
                    <p className="text-lg font-bold">{metrics.current_rps.toFixed(2)}</p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Rate Limit Hits</p>
                    <p className="text-lg font-bold text-orange-600">{metrics.rate_limit_hits}</p>
                  </div>
                  
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Est. Monthly Cost</p>
                    <p className="text-lg font-bold">{formatCurrency(metrics.estimated_monthly_cost)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Footer */}
      <div className="text-center text-sm text-gray-500">
        <p>Real-time API monitoring • Auto-refreshes every minute when enabled</p>
      </div>
    </div>
  );
}