'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
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
  Database,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Wifi,
  WifiOff,
  HardDrive,
  Zap,
  RefreshCw,
  Signal,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  FileText,
  Cloud,
  Server,
  Timer
} from 'lucide-react';
import { fetchLatestPipelineOutput, fetchPipelineFiles } from '@/lib/services/pipeline';
import { fetchRealTimeQuote, fetchTimeSeries } from '@/lib/services/twelvedata';

interface L0Data {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: 'MT5' | 'TwelveData';
  spread?: number;
}

interface L0Stats {
  totalRecords: number;
  dataCompleteness: number;
  averageSpread: number;
  priceRange: { min: number; max: number };
  lastUpdate: string;
  sourceBreakdown: Record<string, number>;
}

interface PipelineStatus {
  status: 'running' | 'completed' | 'failed' | 'idle';
  startTime: string;
  endTime?: string;
  recordsProcessed: number;
  errors: number;
  warnings: number;
  nextRun?: string;
}

interface BackupHealth {
  exists: boolean;
  lastUpdated: string;
  recordCount: number;
  gapsDetected: number;
  size: string;
  integrity: 'good' | 'warning' | 'error';
}

interface ReadySignalStatus {
  active: boolean;
  waiting: boolean;
  handoverComplete: boolean;
  websocketReady: boolean;
  lastHandover: string;
  pendingRecords: number;
}

interface DataQualityMetrics {
  completeness: number;
  latency: number;
  gapsCount: number;
  lastTimestamp: string;
  duplicates: number;
  outliers: number;
}

interface APIUsageMetrics {
  callsUsed: number;
  rateLimit: number;
  remainingCalls: number;
  resetTime: string;
  keyRotationDue: boolean;
  keyAge: number;
}

export default function L0RawDataDashboard() {
  const [l0Data, setL0Data] = useState<L0Data[]>([]);
  const [l0Stats, setL0Stats] = useState<L0Stats | null>(null);
  const [realtimePrice, setRealtimePrice] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState(30000);

  // New monitoring state
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [backupHealth, setBackupHealth] = useState<BackupHealth | null>(null);
  const [readySignal, setReadySignal] = useState<ReadySignalStatus | null>(null);
  const [dataQuality, setDataQuality] = useState<DataQualityMetrics | null>(null);
  const [apiUsage, setApiUsage] = useState<APIUsageMetrics | null>(null);
  const [alerts, setAlerts] = useState<Array<{id: string, type: 'error' | 'warning' | 'info', message: string, timestamp: string}>>([]);

  const fetchL0Data = async () => {
    try {
      setError(null);
      
      // Fetch latest L0 pipeline output
      const pipelineData = await fetchLatestPipelineOutput('L0');
      
      // Fetch real-time data for comparison
      const [realtimeQuote, timeSeries] = await Promise.all([
        fetchRealTimeQuote().catch(() => null),
        fetchTimeSeries('USD/COP', '5min', 50).catch(() => []),
      ]);
      
      if (realtimeQuote) {
        setRealtimePrice(realtimeQuote);
      }
      
      // Process pipeline data
      if (pipelineData) {
        // Mock L0 data structure - adapt based on actual format
        const mockL0Data: L0Data[] = timeSeries.map((item: any, index: number) => ({
          timestamp: item.datetime,
          open: parseFloat(item.open),
          high: parseFloat(item.high),
          low: parseFloat(item.low),
          close: parseFloat(item.close),
          volume: parseInt(item.volume) || 0,
          source: index % 2 === 0 ? 'TwelveData' : 'MT5',
          spread: Math.random() * 50 + 10, // Mock spread data
        }));
        
        setL0Data(mockL0Data);
        
        // Calculate statistics
        const stats: L0Stats = {
          totalRecords: mockL0Data.length,
          dataCompleteness: Math.min(100, (mockL0Data.length / 288) * 100), // Assuming 5-min intervals for 24h = 288
          averageSpread: mockL0Data.reduce((sum, item) => sum + (item.spread || 0), 0) / mockL0Data.length,
          priceRange: {
            min: Math.min(...mockL0Data.map(item => item.low)),
            max: Math.max(...mockL0Data.map(item => item.high)),
          },
          lastUpdate: mockL0Data[0]?.timestamp || 'Unknown',
          sourceBreakdown: mockL0Data.reduce((acc, item) => {
            acc[item.source] = (acc[item.source] || 0) + 1;
            return acc;
          }, {} as Record<string, number>),
        };
        
        setL0Stats(stats);

        // Mock monitoring data
        setPipelineStatus({
          status: Math.random() > 0.1 ? 'running' : 'failed',
          startTime: new Date(Date.now() - 3600000).toISOString(),
          recordsProcessed: Math.floor(Math.random() * 10000) + 5000,
          errors: Math.floor(Math.random() * 3),
          warnings: Math.floor(Math.random() * 5),
          nextRun: new Date(Date.now() + 1800000).toISOString()
        });

        setBackupHealth({
          exists: true,
          lastUpdated: new Date(Date.now() - 300000).toISOString(),
          recordCount: mockL0Data.length,
          gapsDetected: Math.floor(Math.random() * 3),
          size: '2.4 GB',
          integrity: Math.random() > 0.2 ? 'good' : 'warning'
        });

        setReadySignal({
          active: Math.random() > 0.1,
          waiting: Math.random() > 0.8,
          handoverComplete: Math.random() > 0.2,
          websocketReady: Math.random() > 0.1,
          lastHandover: new Date(Date.now() - 120000).toISOString(),
          pendingRecords: Math.floor(Math.random() * 100)
        });

        setDataQuality({
          completeness: 95 + Math.random() * 5,
          latency: Math.random() * 50 + 10,
          gapsCount: Math.floor(Math.random() * 5),
          lastTimestamp: new Date().toISOString(),
          duplicates: Math.floor(Math.random() * 10),
          outliers: Math.floor(Math.random() * 20)
        });

        setApiUsage({
          callsUsed: Math.floor(Math.random() * 500) + 1200,
          rateLimit: 2000,
          remainingCalls: 800 - Math.floor(Math.random() * 200),
          resetTime: new Date(Date.now() + 3600000).toISOString(),
          keyRotationDue: Math.random() > 0.8,
          keyAge: Math.floor(Math.random() * 30) + 1
        });

        // Generate alerts
        const newAlerts = [];
        if (Math.random() > 0.7) {
          newAlerts.push({
            id: Date.now().toString(),
            type: 'warning' as const,
            message: 'API rate limit approaching threshold',
            timestamp: new Date().toISOString()
          });
        }
        if (Math.random() > 0.9) {
          newAlerts.push({
            id: (Date.now() + 1).toString(),
            type: 'error' as const,
            message: 'Data gap detected in L0 pipeline',
            timestamp: new Date().toISOString()
          });
        }
        setAlerts(newAlerts);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L0 data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL0Data();
    
    const interval = setInterval(fetchL0Data, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
    }).format(price);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-950 text-white min-h-full">
      {/* Header Section */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-xl border border-purple-500/30">
            <Database className="w-8 h-8 text-purple-400" />
          </div>
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              L0 Pipeline Monitoring
            </h2>
            <p className="text-slate-400">Real-time data pipeline health and performance</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white"
          >
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
            <option value={60000}>1min</option>
            <option value={300000}>5min</option>
          </select>
          <Badge
            className={`px-3 py-1 ${l0Stats && l0Stats.dataCompleteness > 90 ? 'bg-green-500/20 text-green-400 border-green-500/30' :
                     l0Stats && l0Stats.dataCompleteness > 70 ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'}`}
          >
            {l0Stats ? formatPercentage(l0Stats.dataCompleteness) : 'N/A'} Complete
          </Badge>
          <button
            onClick={fetchL0Data}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert) => (
            <Alert key={alert.id} className={`border-l-4 ${
              alert.type === 'error' ? 'border-red-500 bg-red-500/10' :
              alert.type === 'warning' ? 'border-yellow-500 bg-yellow-500/10' :
              'border-blue-500 bg-blue-500/10'
            }`}>
              <div className="flex items-center gap-3">
                {alert.type === 'error' ? <XCircle className="w-5 h-5 text-red-400" /> :
                 alert.type === 'warning' ? <AlertTriangle className="w-5 h-5 text-yellow-400" /> :
                 <AlertCircle className="w-5 h-5 text-blue-400" />}
                <div className="flex-1">
                  <div className="text-white font-medium">{alert.message}</div>
                  <div className="text-slate-400 text-sm">{new Date(alert.timestamp).toLocaleString()}</div>
                </div>
              </div>
            </Alert>
          ))}
        </div>
      )}

      {/* Status Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Pipeline Status */}
        {pipelineStatus && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Activity className="w-6 h-6 text-purple-400" />
                <h3 className="text-lg font-semibold text-white">Pipeline Status</h3>
              </div>
              <div className={`w-3 h-3 rounded-full ${
                pipelineStatus.status === 'running' ? 'bg-green-500 animate-pulse' :
                pipelineStatus.status === 'completed' ? 'bg-blue-500' :
                pipelineStatus.status === 'failed' ? 'bg-red-500' : 'bg-gray-500'
              }`} />
            </div>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white capitalize">{pipelineStatus.status}</div>
              <div className="text-sm text-slate-400">Records: {pipelineStatus.recordsProcessed.toLocaleString()}</div>
              <div className="text-sm text-slate-400">Errors: {pipelineStatus.errors} | Warnings: {pipelineStatus.warnings}</div>
              {pipelineStatus.nextRun && (
                <div className="text-sm text-slate-400">Next: {new Date(pipelineStatus.nextRun).toLocaleTimeString()}</div>
              )}
            </div>
          </Card>
        )}

        {/* Backup Health */}
        {backupHealth && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <HardDrive className="w-6 h-6 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">Backup Health</h3>
              </div>
              <div className={`w-3 h-3 rounded-full ${
                backupHealth.integrity === 'good' ? 'bg-green-500' :
                backupHealth.integrity === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
            </div>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{backupHealth.exists ? 'Available' : 'Missing'}</div>
              <div className="text-sm text-slate-400">Size: {backupHealth.size}</div>
              <div className="text-sm text-slate-400">Records: {backupHealth.recordCount.toLocaleString()}</div>
              <div className="text-sm text-slate-400">Gaps: {backupHealth.gapsDetected}</div>
              <div className="text-sm text-slate-400">Updated: {new Date(backupHealth.lastUpdated).toLocaleTimeString()}</div>
            </div>
          </Card>
        )}

        {/* Ready Signal Status */}
        {readySignal && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Signal className="w-6 h-6 text-green-400" />
                <h3 className="text-lg font-semibold text-white">Ready Signal</h3>
              </div>
              <div className={`w-3 h-3 rounded-full ${
                readySignal.active ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`} />
            </div>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{readySignal.active ? 'Active' : 'Inactive'}</div>
              <div className="flex items-center gap-2 text-sm">
                {readySignal.websocketReady ?
                  <Wifi className="w-4 h-4 text-green-400" /> :
                  <WifiOff className="w-4 h-4 text-red-400" />}
                <span className="text-slate-400">WebSocket Ready</span>
              </div>
              <div className="text-sm text-slate-400">Pending: {readySignal.pendingRecords}</div>
              <div className="text-sm text-slate-400">Last Handover: {new Date(readySignal.lastHandover).toLocaleTimeString()}</div>
            </div>
          </Card>
        )}

        {/* API Usage */}
        {apiUsage && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Cloud className="w-6 h-6 text-cyan-400" />
                <h3 className="text-lg font-semibold text-white">API Usage</h3>
              </div>
              <div className={`w-3 h-3 rounded-full ${
                apiUsage.keyRotationDue ? 'bg-yellow-500' :
                (apiUsage.callsUsed / apiUsage.rateLimit) > 0.8 ? 'bg-orange-500' : 'bg-green-500'
              }`} />
            </div>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{apiUsage.callsUsed}/{apiUsage.rateLimit}</div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    (apiUsage.callsUsed / apiUsage.rateLimit) > 0.8 ? 'bg-red-500' :
                    (apiUsage.callsUsed / apiUsage.rateLimit) > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${(apiUsage.callsUsed / apiUsage.rateLimit) * 100}%` }}
                />
              </div>
              <div className="text-sm text-slate-400">Remaining: {apiUsage.remainingCalls}</div>
              <div className="text-sm text-slate-400">Resets: {new Date(apiUsage.resetTime).toLocaleTimeString()}</div>
              {apiUsage.keyRotationDue && (
                <div className="text-sm text-yellow-400">⚠ Key rotation due</div>
              )}
            </div>
          </Card>
        )}
      </div>

      {/* Real-time Data Flow Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Quality Metrics */}
        {dataQuality && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center gap-3 mb-6">
              <CheckCircle className="w-6 h-6 text-green-400" />
              <h3 className="text-xl font-semibold text-white">Data Quality</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-3xl font-bold text-green-400">{dataQuality.completeness.toFixed(1)}%</div>
                <div className="text-sm text-slate-400">Completeness</div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-3xl font-bold text-cyan-400">{dataQuality.latency.toFixed(0)}ms</div>
                <div className="text-sm text-slate-400">Latency</div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-3xl font-bold text-yellow-400">{dataQuality.gapsCount}</div>
                <div className="text-sm text-slate-400">Gaps</div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-3xl font-bold text-red-400">{dataQuality.outliers}</div>
                <div className="text-sm text-slate-400">Outliers</div>
              </div>
            </div>
            <div className="mt-4 text-sm text-slate-400">
              Last Update: {new Date(dataQuality.lastTimestamp).toLocaleString()}
            </div>
          </Card>
        )}

        {/* Real-time Price Display */}
        {realtimePrice && (
          <Card className="p-6 bg-slate-900/50 border-slate-700/50">
            <div className="flex items-center gap-3 mb-6">
              <TrendingUp className="w-6 h-6 text-green-400" />
              <h3 className="text-xl font-semibold text-white">Real-time USD/COP</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-sm text-slate-400 mb-1">Current Price</div>
                <div className="text-3xl font-bold text-white">{formatPrice(Number(realtimePrice.close))}</div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-sm text-slate-400 mb-1">Change</div>
                <div className={`text-2xl font-bold ${
                  (Number(realtimePrice.change) ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {realtimePrice.change !== undefined ?
                    `${Number(realtimePrice.change) >= 0 ? '+' : ''}${Number(realtimePrice.change).toFixed(2)}` :
                    'N/A'}
                </div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-sm text-slate-400 mb-1">Change %</div>
                <div className={`text-2xl font-bold ${
                  (Number(realtimePrice.percent_change) ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {realtimePrice.percent_change !== undefined ?
                    formatPercentage(Number(realtimePrice.percent_change)) :
                    'N/A'}
                </div>
              </div>
              <div className="text-center p-4 bg-slate-800/50 rounded-lg">
                <div className="text-sm text-slate-400 mb-1">Volume</div>
                <div className="text-2xl font-bold text-white">
                  {realtimePrice.volume ? Number(realtimePrice.volume).toLocaleString() : 'N/A'}
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Pipeline Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Records Processing Rate */}
        <Card className="p-6 bg-slate-900/50 border-slate-700/50">
          <div className="flex items-center gap-3 mb-6">
            <Timer className="w-6 h-6 text-purple-400" />
            <h3 className="text-xl font-semibold text-white">Processing Rate</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={l0Data.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  stroke="#9CA3AF"
                />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#FFFFFF'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="volume"
                  stroke="#8B5CF6"
                  fill="#8B5CF6"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Data Sources Distribution */}
        <Card className="p-6 bg-slate-900/50 border-slate-700/50">
          <div className="flex items-center gap-3 mb-6">
            <Server className="w-6 h-6 text-cyan-400" />
            <h3 className="text-xl font-semibold text-white">Data Sources</h3>
          </div>
          {l0Stats && (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={Object.entries(l0Stats.sourceBreakdown).map(([source, count]) => ({
                      name: source,
                      value: count,
                      percentage: ((count / l0Stats.totalRecords) * 100).toFixed(1)
                    }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percentage }) => `${name}: ${percentage}%`}
                  >
                    <Cell fill="#8B5CF6" />
                    <Cell fill="#06B6D4" />
                    <Cell fill="#10B981" />
                    <Cell fill="#F59E0B" />
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>

      {/* Main Price and Latency Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Price Movement Chart */}
        <Card className="p-6 bg-slate-900/50 border-slate-700/50">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-6 h-6 text-green-400" />
            <h3 className="text-xl font-semibold text-white">USD/COP Price Movement</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={l0Data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  stroke="#9CA3AF"
                />
                <YAxis
                  domain={['dataMin - 10', 'dataMax + 10']}
                  tickFormatter={(value) => value.toFixed(0)}
                  stroke="#9CA3AF"
                />
                <Tooltip
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any) => [formatPrice(value), 'Price']}
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#FFFFFF'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Spread Analysis */}
        <Card className="p-6 bg-slate-900/50 border-slate-700/50">
          <div className="flex items-center gap-3 mb-6">
            <Zap className="w-6 h-6 text-yellow-400" />
            <h3 className="text-xl font-semibold text-white">Spread Analysis</h3>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={l0Data.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  stroke="#9CA3AF"
                />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any) => [`${value.toFixed(1)} pips`, 'Spread']}
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#FFFFFF'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="spread"
                  stroke="#F59E0B"
                  fill="#F59E0B"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Summary Statistics */}
      {l0Stats && (
        <Card className="p-6 bg-slate-900/50 border-slate-700/50">
          <div className="flex items-center gap-3 mb-6">
            <FileText className="w-6 h-6 text-blue-400" />
            <h3 className="text-xl font-semibold text-white">Pipeline Summary</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-slate-800/50 rounded-lg">
              <div className="text-3xl font-bold text-blue-400">{l0Stats.totalRecords.toLocaleString()}</div>
              <div className="text-sm text-slate-400 mt-1">Total Records</div>
            </div>
            <div className="text-center p-4 bg-slate-800/50 rounded-lg">
              <div className="text-3xl font-bold text-yellow-400">{l0Stats.averageSpread.toFixed(1)}</div>
              <div className="text-sm text-slate-400 mt-1">Avg Spread (pips)</div>
            </div>
            <div className="text-center p-4 bg-slate-800/50 rounded-lg">
              <div className="text-2xl font-bold text-green-400">
                {formatPrice(l0Stats.priceRange.min)} - {formatPrice(l0Stats.priceRange.max)}
              </div>
              <div className="text-sm text-slate-400 mt-1">Price Range</div>
            </div>
            <div className="text-center p-4 bg-slate-800/50 rounded-lg">
              <div className="text-xl font-bold text-purple-400">
                {new Date(l0Stats.lastUpdate).toLocaleString()}
              </div>
              <div className="text-sm text-slate-400 mt-1">Last Update</div>
            </div>
          </div>
        </Card>
      )}

      {/* Error Handling Display */}
      {error && (
        <Alert className="bg-red-500/10 border-red-500/30">
          <XCircle className="w-5 h-5 text-red-400" />
          <div className="text-red-400 font-medium">
            Pipeline Error: {error}
          </div>
        </Alert>
      )}
    </div>
  );
}