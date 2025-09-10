'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ComposedChart, Area, AreaChart, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, Cell
} from 'recharts';
import { minioClient } from '@/lib/services/minio-client';
import { 
  Activity, AlertTriangle, CheckCircle, Clock, Cpu, Database, 
  GitBranch, HardDrive, MemoryStick, Network, Play, Pause,
  RefreshCw, Shield, Zap, TrendingUp, TrendingDown, 
  Download, Settings, Bell, Eye, Server, Layers
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
    case 'dd/MM':
      return `${String(day).padStart(2, '0')}/${String(month + 1).padStart(2, '0')}`;
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

const subMinutes = (date: Date, minutes: number): Date => {
  const result = new Date(date);
  result.setMinutes(result.getMinutes() - minutes);
  return result;
};

// Use the format function instead of date-fns format
const format = formatDate;

interface DAGNode {
  id: string;
  name: string;
  layer: 'L0' | 'L1' | 'L2' | 'L3' | 'L4' | 'L5';
  status: 'running' | 'success' | 'failed' | 'pending' | 'skipped' | 'retry';
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  dependencies: string[];
  retryCount: number;
  maxRetries: number;
  nextRun?: Date;
  slaTarget: number; // minutes
  slaStatus: 'within' | 'warning' | 'breach';
}

interface ResourceMetrics {
  timestamp: Date;
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkIO: number;
  taskCount: number;
  queueDepth: number;
}

interface SLAMetrics {
  metric: string;
  current: number;
  target: number;
  threshold: number;
  trend: 'improving' | 'stable' | 'degrading';
  breaches24h: number;
}

interface DataQualityCheck {
  layer: string;
  check: string;
  status: 'pass' | 'warning' | 'fail';
  score: number;
  threshold: number;
  lastCheck: Date;
  details: string;
}

interface PipelineAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  category: 'performance' | 'quality' | 'sla' | 'resource' | 'dependency';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  node?: string;
  autoResolved: boolean;
}

export default function PipelineHealth() {
  const [dagNodes, setDagNodes] = useState<DAGNode[]>([]);
  const [resourceMetrics, setResourceMetrics] = useState<ResourceMetrics[]>([]);
  const [slaMetrics, setSlaMetrics] = useState<SLAMetrics[]>([]);
  const [qualityChecks, setQualityChecks] = useState<DataQualityCheck[]>([]);
  const [alerts, setAlerts] = useState<PipelineAlert[]>([]);
  const [selectedView, setSelectedView] = useState<'overview' | 'dag' | 'sla' | 'resources' | 'quality' | 'alerts'>('overview');
  const [timeRange, setTimeRange] = useState<'1H' | '6H' | '24H' | '7D'>('24H');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);

  // Generate comprehensive pipeline health data
  const generatePipelineData = useCallback(() => {
    // Generate DAG nodes for the complete USDCOP pipeline
    const nodes: DAGNode[] = [
      // L0 - Data Acquisition
      {
        id: 'l0-mt5-acquire',
        name: 'MT5 Data Acquisition',
        layer: 'L0',
        status: 'success',
        startTime: subMinutes(new Date(), 15),
        endTime: subMinutes(new Date(), 12),
        duration: 3,
        dependencies: [],
        retryCount: 0,
        maxRetries: 3,
        nextRun: new Date(Date.now() + 5 * 60 * 1000), // 5 min
        slaTarget: 10,
        slaStatus: 'within'
      },
      {
        id: 'l0-twelvedata-acquire',
        name: 'TwelveData API Ingestion',
        layer: 'L0',
        status: 'success',
        startTime: subMinutes(new Date(), 18),
        endTime: subMinutes(new Date(), 15),
        duration: 3,
        dependencies: [],
        retryCount: 1,
        maxRetries: 3,
        nextRun: new Date(Date.now() + 3 * 60 * 1000),
        slaTarget: 15,
        slaStatus: 'within'
      },
      // L1 - Standardization
      {
        id: 'l1-standardize',
        name: 'Data Standardization',
        layer: 'L1',
        status: 'running',
        startTime: subMinutes(new Date(), 8),
        dependencies: ['l0-mt5-acquire', 'l0-twelvedata-acquire'],
        retryCount: 0,
        maxRetries: 2,
        nextRun: new Date(Date.now() + 12 * 60 * 1000),
        slaTarget: 20,
        slaStatus: 'within'
      },
      // L2 - Preparation
      {
        id: 'l2-prepare',
        name: 'Feature Preparation',
        layer: 'L2',
        status: 'pending',
        dependencies: ['l1-standardize'],
        retryCount: 0,
        maxRetries: 2,
        nextRun: new Date(Date.now() + 15 * 60 * 1000),
        slaTarget: 30,
        slaStatus: 'within'
      },
      // L3 - Feature Engineering
      {
        id: 'l3-feature',
        name: 'Advanced Feature Engineering',
        layer: 'L3',
        status: 'failed',
        startTime: subMinutes(new Date(), 45),
        endTime: subMinutes(new Date(), 25),
        duration: 20,
        dependencies: ['l2-prepare'],
        retryCount: 2,
        maxRetries: 3,
        nextRun: new Date(Date.now() + 30 * 60 * 1000),
        slaTarget: 25,
        slaStatus: 'breach'
      },
      // L4 - ML Ready
      {
        id: 'l4-rlready',
        name: 'RL Model Preparation',
        layer: 'L4',
        status: 'skipped',
        dependencies: ['l3-feature'],
        retryCount: 0,
        maxRetries: 2,
        nextRun: new Date(Date.now() + 45 * 60 * 1000),
        slaTarget: 35,
        slaStatus: 'warning'
      },
      // L5 - Serving
      {
        id: 'l5-serving',
        name: 'Model Serving Pipeline',
        layer: 'L5',
        status: 'retry',
        startTime: subMinutes(new Date(), 5),
        dependencies: ['l4-rlready'],
        retryCount: 1,
        maxRetries: 2,
        nextRun: new Date(Date.now() + 60 * 60 * 1000),
        slaTarget: 15,
        slaStatus: 'warning'
      }
    ];

    // Generate resource metrics over time
    const hours = timeRange === '1H' ? 1 : timeRange === '6H' ? 6 : timeRange === '24H' ? 24 : 168;
    const resources: ResourceMetrics[] = Array.from({ length: hours * 4 }, (_, i) => {
      const timestamp = subMinutes(new Date(), (hours * 4 - i) * 15);
      return {
        timestamp,
        cpuUsage: 40 + Math.sin(i / 10) * 20 + Math.random() * 10,
        memoryUsage: 60 + Math.sin(i / 15) * 15 + Math.random() * 8,
        diskUsage: 75 + Math.random() * 5,
        networkIO: Math.max(0, 50 + Math.sin(i / 8) * 30 + Math.random() * 20),
        taskCount: Math.floor(Math.random() * 8) + 2,
        queueDepth: Math.floor(Math.random() * 5)
      };
    });

    // Generate SLA metrics
    const slaData: SLAMetrics[] = [
      {
        metric: 'Pipeline Completion Time',
        current: 85.4,
        target: 90.0,
        threshold: 75.0,
        trend: 'stable',
        breaches24h: 2
      },
      {
        metric: 'Data Quality Score',
        current: 94.7,
        target: 95.0,
        threshold: 90.0,
        trend: 'improving',
        breaches24h: 0
      },
      {
        metric: 'System Availability',
        current: 99.2,
        target: 99.5,
        threshold: 99.0,
        trend: 'stable',
        breaches24h: 1
      },
      {
        metric: 'Error Rate',
        current: 0.8,
        target: 1.0,
        threshold: 2.0,
        trend: 'improving',
        breaches24h: 0
      },
      {
        metric: 'Resource Utilization',
        current: 72.3,
        target: 80.0,
        threshold: 90.0,
        trend: 'degrading',
        breaches24h: 3
      }
    ];

    // Generate data quality checks
    const qualityData: DataQualityCheck[] = [
      {
        layer: 'L0',
        check: 'Data Completeness',
        status: 'pass',
        score: 98.5,
        threshold: 95.0,
        lastCheck: subMinutes(new Date(), 10),
        details: 'Missing 1.5% of expected records from weekend gaps'
      },
      {
        layer: 'L0',
        check: 'Price Consistency',
        status: 'pass',
        score: 99.8,
        threshold: 98.0,
        lastCheck: subMinutes(new Date(), 10),
        details: 'All price ranges within expected bounds'
      },
      {
        layer: 'L1',
        check: 'Schema Validation',
        status: 'warning',
        score: 92.3,
        threshold: 95.0,
        lastCheck: subMinutes(new Date(), 25),
        details: '3 fields missing in 7.7% of records'
      },
      {
        layer: 'L2',
        check: 'Feature Distribution',
        status: 'pass',
        score: 96.1,
        threshold: 90.0,
        lastCheck: subMinutes(new Date(), 40),
        details: 'All feature distributions within 2 sigma'
      },
      {
        layer: 'L3',
        check: 'Correlation Stability',
        status: 'fail',
        score: 78.4,
        threshold: 85.0,
        lastCheck: subMinutes(new Date(), 50),
        details: 'USD_Index correlation degraded by 15%'
      },
      {
        layer: 'L4',
        check: 'Target Leakage',
        status: 'pass',
        score: 100.0,
        threshold: 95.0,
        lastCheck: subMinutes(new Date(), 60),
        details: 'No future information detected in features'
      }
    ];

    // Generate pipeline alerts
    const alertsData: PipelineAlert[] = [
      {
        id: 'alert-001',
        severity: 'critical',
        category: 'performance',
        message: 'L3 feature engineering exceeded SLA by 15 minutes',
        timestamp: subMinutes(new Date(), 25),
        acknowledged: false,
        node: 'l3-feature',
        autoResolved: false
      },
      {
        id: 'alert-002',
        severity: 'warning',
        category: 'quality',
        message: 'L1 schema validation score below threshold',
        timestamp: subMinutes(new Date(), 45),
        acknowledged: false,
        node: 'l1-standardize',
        autoResolved: false
      },
      {
        id: 'alert-003',
        severity: 'warning',
        category: 'resource',
        message: 'Memory usage approaching 80% threshold',
        timestamp: subMinutes(new Date(), 60),
        acknowledged: true,
        autoResolved: false
      },
      {
        id: 'alert-004',
        severity: 'info',
        category: 'sla',
        message: 'Pipeline completion time improved by 5%',
        timestamp: subMinutes(new Date(), 120),
        acknowledged: true,
        autoResolved: true
      },
      {
        id: 'alert-005',
        severity: 'critical',
        category: 'dependency',
        message: 'External API timeout: TwelveData service unavailable',
        timestamp: subMinutes(new Date(), 180),
        acknowledged: false,
        node: 'l0-twelvedata-acquire',
        autoResolved: false
      }
    ];

    return { nodes, resources, slaData, qualityData, alertsData };
  }, [timeRange]);

  useEffect(() => {
    setLoading(true);
    
    const { nodes, resources, slaData, qualityData, alertsData } = generatePipelineData();
    
    setDagNodes(nodes);
    setResourceMetrics(resources);
    setSlaMetrics(slaData);
    setQualityChecks(qualityData);
    setAlerts(alertsData);
    
    setLoading(false);
  }, [generatePipelineData]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      const { nodes, resources, slaData, qualityData, alertsData } = generatePipelineData();
      setDagNodes(nodes);
      setResourceMetrics(resources);
      setSlaMetrics(slaData);
      setQualityChecks(qualityData);
      setAlerts(alertsData);
    }, 30000); // 30 seconds
    
    return () => clearInterval(interval);
  }, [autoRefresh, generatePipelineData]);

  const formatPercent = useCallback((value: number, decimals = 1) => {
    return `${value.toFixed(decimals)}%`;
  }, []);

  const getNodeStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'success': return 'bg-green-950 text-green-400';
      case 'running': return 'bg-blue-950 text-blue-400';
      case 'failed': return 'bg-red-950 text-red-400';
      case 'pending': return 'bg-slate-950 text-slate-400';
      case 'skipped': return 'bg-yellow-950 text-yellow-400';
      case 'retry': return 'bg-orange-950 text-orange-400';
      default: return 'bg-slate-950 text-slate-400';
    }
  }, []);

  const getSLAStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'within': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'breach': return 'text-red-400';
      default: return 'text-slate-400';
    }
  }, []);

  const getAlertSeverityColor = useCallback((severity: string) => {
    switch (severity) {
      case 'critical': return 'border-red-400 bg-red-950/30';
      case 'warning': return 'border-yellow-400 bg-yellow-950/30';
      case 'info': return 'border-blue-400 bg-blue-950/30';
      default: return 'border-slate-400 bg-slate-950/30';
    }
  }, []);

  const getQualityStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'pass': return 'bg-green-950 text-green-400';
      case 'warning': return 'bg-yellow-950 text-yellow-400';
      case 'fail': return 'bg-red-950 text-red-400';
      default: return 'bg-slate-950 text-slate-400';
    }
  }, []);

  const pipelineOverallStatus = useMemo(() => {
    const failedNodes = dagNodes.filter(n => n.status === 'failed').length;
    const runningNodes = dagNodes.filter(n => n.status === 'running').length;
    const warningNodes = dagNodes.filter(n => n.status === 'retry').length;
    
    if (failedNodes > 0) return 'critical';
    if (warningNodes > 0 || runningNodes > 3) return 'warning';
    return 'healthy';
  }, [dagNodes]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-950 rounded-lg border border-amber-500/20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-amber-500/20 border-t-amber-500 mx-auto mb-4"></div>
          <p className="text-amber-500 font-mono text-sm">Loading Pipeline Health Analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-amber-500/20 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-amber-500 font-mono">PIPELINE HEALTH MONITORING</h1>
          <p className="text-slate-400 text-sm mt-1">
            Data Pipeline Status • SLA Compliance • Resource Utilization • 
            Overall Status: 
            <span className={`ml-1 font-semibold ${
              pipelineOverallStatus === 'critical' ? 'text-red-400' :
              pipelineOverallStatus === 'warning' ? 'text-yellow-400' : 'text-green-400'
            }`}>
              {pipelineOverallStatus.toUpperCase()}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-slate-400 text-sm">Auto Refresh</label>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-2 rounded-lg transition-colors ${
                autoRefresh 
                  ? 'bg-amber-500 text-slate-950' 
                  : 'bg-slate-900 border border-amber-500/20 text-amber-500 hover:bg-amber-500/10'
              }`}
            >
              <RefreshCw className={`h-4 w-4 ${autoRefresh ? 'animate-spin' : ''}`} />
            </button>
          </div>
          <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
            {([
              { key: 'overview', label: 'Overview' },
              { key: 'dag', label: 'DAG' },
              { key: 'sla', label: 'SLA' },
              { key: 'resources', label: 'Resources' },
              { key: 'quality', label: 'Quality' },
              { key: 'alerts', label: 'Alerts' }
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
              <Download className="h-4 w-4" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics Dashboard */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Pipeline Status
              <Activity className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-xl font-bold font-mono ${
              pipelineOverallStatus === 'critical' ? 'text-red-400' :
              pipelineOverallStatus === 'warning' ? 'text-yellow-400' : 'text-green-400'
            }`}>
              {pipelineOverallStatus.toUpperCase()}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {dagNodes.filter(n => n.status === 'success').length}/{dagNodes.length} healthy
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              SLA Compliance
              <Shield className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(slaMetrics.find(m => m.metric === 'Pipeline Completion Time')?.current || 0)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Target: {formatPercent(slaMetrics.find(m => m.metric === 'Pipeline Completion Time')?.target || 0)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Data Quality
              <Database className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(qualityChecks.reduce((sum, q) => sum + q.score, 0) / qualityChecks.length)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {qualityChecks.filter(q => q.status === 'pass').length}/{qualityChecks.length} checks pass
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Resource Usage
              <Cpu className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-white font-mono">
              {formatPercent(resourceMetrics[resourceMetrics.length - 1]?.cpuUsage || 0)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Memory: {formatPercent(resourceMetrics[resourceMetrics.length - 1]?.memoryUsage || 0)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Active Alerts
              <Bell className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-red-400 font-mono">
              {alerts.filter(a => !a.acknowledged && a.severity === 'critical').length}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {alerts.filter(a => !a.acknowledged).length} unacknowledged
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-amber-500 font-mono flex items-center justify-between">
              Uptime
              <Clock className="h-4 w-4" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-400 font-mono">
              {formatPercent(slaMetrics.find(m => m.metric === 'System Availability')?.current || 0)}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Last 24h
            </div>
          </CardContent>
        </Card>
      </div>

      {/* DAG View */}
      {selectedView === 'dag' && (
        <div className="space-y-6">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Data Pipeline DAG Status</CardTitle>
              <p className="text-slate-400 text-sm">L0-L5 Pipeline Flow • Dependencies • Execution Status</p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {dagNodes.map((node, index) => (
                  <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h4 className="font-semibold text-white text-sm">{node.name}</h4>
                        <Badge className={getNodeStatusColor(node.status)}>
                          {node.status.toUpperCase()}
                        </Badge>
                      </div>
                      <Badge className="bg-blue-950 text-blue-400 text-xs">
                        {node.layer}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2 text-xs">
                      {node.duration && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Duration:</span>
                          <span className="font-mono text-white">{node.duration}m</span>
                        </div>
                      )}
                      
                      <div className="flex justify-between">
                        <span className="text-slate-400">SLA Target:</span>
                        <span className={`font-mono ${getSLAStatusColor(node.slaStatus)}`}>
                          {node.slaTarget}m
                        </span>
                      </div>
                      
                      {node.retryCount > 0 && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Retries:</span>
                          <span className="font-mono text-orange-400">
                            {node.retryCount}/{node.maxRetries}
                          </span>
                        </div>
                      )}
                      
                      {node.nextRun && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Next Run:</span>
                          <span className="font-mono text-slate-300">
                            {format(node.nextRun, 'HH:mm')}
                          </span>
                        </div>
                      )}
                      
                      {node.dependencies.length > 0 && (
                        <div>
                          <span className="text-slate-400">Dependencies:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {node.dependencies.map((dep) => (
                              <Badge key={dep} className="bg-slate-700 text-slate-300 text-xs">
                                {dep.split('-')[0].toUpperCase()}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <Progress 
                      value={node.status === 'success' ? 100 : 
                             node.status === 'running' ? 65 :
                             node.status === 'failed' ? 25 : 0} 
                      className="mt-3 h-2" 
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* SLA View */}
      {selectedView === 'sla' && (
        <div className="space-y-6">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">SLA Compliance Dashboard</CardTitle>
              <p className="text-slate-400 text-sm">Service Level Agreements • Performance Targets • Trend Analysis</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {slaMetrics.map((sla, index) => (
                  <div key={index} className="bg-slate-800 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-3">
                      <h4 className="font-semibold text-white text-sm">{sla.metric}</h4>
                      <div className="flex items-center gap-2">
                        <Badge className={`${
                          sla.current >= sla.target ? 'bg-green-950 text-green-400' :
                          sla.current >= sla.threshold ? 'bg-yellow-950 text-yellow-400' : 'bg-red-950 text-red-400'
                        }`}>
                          {sla.trend === 'improving' ? '↗' : sla.trend === 'degrading' ? '↘' : '→'} 
                          {sla.trend.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-slate-400 text-xs mb-1">Current</div>
                        <div className={`text-lg font-bold font-mono ${
                          sla.current >= sla.target ? 'text-green-400' :
                          sla.current >= sla.threshold ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {sla.metric.includes('Rate') ? sla.current.toFixed(2) + '%' : 
                           sla.metric.includes('Time') ? sla.current.toFixed(1) + '%' :
                           sla.current.toFixed(1) + '%'}
                        </div>
                      </div>
                      
                      <div className="text-center">
                        <div className="text-slate-400 text-xs mb-1">Target</div>
                        <div className="text-white font-mono text-lg">
                          {sla.metric.includes('Rate') ? sla.target.toFixed(2) + '%' : 
                           sla.metric.includes('Time') ? sla.target.toFixed(1) + '%' :
                           sla.target.toFixed(1) + '%'}
                        </div>
                      </div>
                      
                      <div className="text-center">
                        <div className="text-slate-400 text-xs mb-1">Threshold</div>
                        <div className="text-slate-300 font-mono text-lg">
                          {sla.metric.includes('Rate') ? sla.threshold.toFixed(2) + '%' : 
                           sla.metric.includes('Time') ? sla.threshold.toFixed(1) + '%' :
                           sla.threshold.toFixed(1) + '%'}
                        </div>
                      </div>
                      
                      <div className="text-center">
                        <div className="text-slate-400 text-xs mb-1">Breaches (24h)</div>
                        <div className={`text-lg font-bold font-mono ${
                          sla.breaches24h === 0 ? 'text-green-400' :
                          sla.breaches24h <= 2 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {sla.breaches24h}
                        </div>
                      </div>
                    </div>
                    
                    <Progress 
                      value={(sla.current / sla.target) * 100} 
                      className="mt-4 h-3" 
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Resources View */}
      {selectedView === 'resources' && (
        <div className="space-y-6">
          <Card className="bg-slate-900 border-amber-500/20">
            <CardHeader>
              <CardTitle className="text-amber-500 font-mono">Resource Utilization Monitoring</CardTitle>
              <p className="text-slate-400 text-sm">CPU, Memory, Disk & Network Usage • Capacity Planning</p>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <div className="flex rounded-lg border border-amber-500/20 overflow-hidden">
                  {(['1H', '6H', '24H', '7D'] as const).map((period) => (
                    <button
                      key={period}
                      onClick={() => setTimeRange(period)}
                      className={`px-3 py-2 text-sm font-mono transition-colors ${
                        timeRange === period
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-slate-900 text-amber-500 hover:bg-amber-500/10'
                      }`}
                    >
                      {period}
                    </button>
                  ))}
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={resourceMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="timestamp" 
                    stroke="#64748B" 
                    fontSize={10}
                    tickFormatter={(timestamp) => format(new Date(timestamp), 'HH:mm')}
                  />
                  <YAxis yAxisId="percentage" stroke="#64748B" fontSize={10} domain={[0, 100]} />
                  <YAxis yAxisId="tasks" orientation="right" stroke="#64748B" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #F59E0B', borderRadius: '6px' }}
                    formatter={(value: any, name: string) => [
                      name.includes('Usage') ? `${value.toFixed(1)}%` : value.toFixed(0),
                      name.replace(/([A-Z])/g, ' $1').trim()
                    ]}
                    labelFormatter={(timestamp) => format(new Date(timestamp), 'PPpp')}
                  />
                  <Area yAxisId="percentage" type="monotone" dataKey="cpuUsage" stackId="1" stroke="#EF4444" fill="#EF4444" fillOpacity={0.3} name="cpuUsage" />
                  <Area yAxisId="percentage" type="monotone" dataKey="memoryUsage" stackId="2" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.3} name="memoryUsage" />
                  <Line yAxisId="percentage" type="monotone" dataKey="diskUsage" stroke="#8B5CF6" strokeWidth={2} dot={false} name="diskUsage" />
                  <Line yAxisId="tasks" type="monotone" dataKey="taskCount" stroke="#10B981" strokeWidth={2} dot={false} name="taskCount" />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Quality View */}
      {selectedView === 'quality' && (
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono">Data Quality Monitoring</CardTitle>
            <p className="text-slate-400 text-sm">Quality Checks by Layer • Score Trends • Anomaly Detection</p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {qualityChecks.map((check, index) => (
                <div key={index} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-semibold text-white text-sm">{check.check}</h4>
                      <Badge className={getQualityStatusColor(check.status)}>
                        {check.status.toUpperCase()}
                      </Badge>
                    </div>
                    <Badge className="bg-blue-950 text-blue-400 text-xs">
                      {check.layer}
                    </Badge>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-sm">Score:</span>
                      <span className={`font-mono font-bold ${
                        check.score >= check.threshold ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatPercent(check.score)}
                      </span>
                    </div>
                    
                    <Progress 
                      value={check.score} 
                      className="h-3"
                    />
                    
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-400">Threshold: {formatPercent(check.threshold)}</span>
                      <span className="text-slate-400">
                        Last Check: {format(check.lastCheck, 'HH:mm')}
                      </span>
                    </div>
                    
                    <div className="text-xs text-slate-300 bg-slate-900 rounded p-2">
                      {check.details}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Alerts View */}
      {selectedView === 'alerts' && (
        <Card className="bg-slate-900 border-amber-500/20">
          <CardHeader>
            <CardTitle className="text-amber-500 font-mono flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Pipeline Alerts & Notifications
            </CardTitle>
            <p className="text-slate-400 text-sm">
              Active: {alerts.filter(a => !a.acknowledged).length} • 
              Critical: {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length} • 
              Auto-resolved: {alerts.filter(a => a.autoResolved).length}
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.map((alert) => (
                <Alert key={alert.id} className={`${getAlertSeverityColor(alert.severity)} ${alert.acknowledged ? 'opacity-60' : ''}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className={`h-4 w-4 mt-0.5 ${
                        alert.severity === 'critical' ? 'text-red-400' :
                        alert.severity === 'warning' ? 'text-yellow-400' : 'text-blue-400'
                      }`} />
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className={`text-xs ${
                            alert.severity === 'critical' ? 'bg-red-950 text-red-400' :
                            alert.severity === 'warning' ? 'bg-yellow-950 text-yellow-400' : 'bg-blue-950 text-blue-400'
                          }`}>
                            {alert.severity.toUpperCase()}
                          </Badge>
                          <Badge className="bg-slate-700 text-slate-300 text-xs">
                            {alert.category}
                          </Badge>
                          {alert.acknowledged && <Badge className="bg-green-950 text-green-400 text-xs">ACK</Badge>}
                          {alert.autoResolved && <Badge className="bg-blue-950 text-blue-400 text-xs">AUTO</Badge>}
                        </div>
                        <AlertDescription className="text-slate-300 text-sm">
                          {alert.message}
                        </AlertDescription>
                        {alert.node && (
                          <div className="text-amber-400 text-xs mt-1">
                            Node: {alert.node}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-right text-slate-500">
                      <div className="text-xs">{format(alert.timestamp, 'HH:mm')}</div>
                      <div className="text-xs">{format(alert.timestamp, 'dd/MM')}</div>
                    </div>
                  </div>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Footer */}
      <div className="text-center py-6 border-t border-amber-500/20">
        <p className="text-slate-500 text-xs font-mono">
          Pipeline Health Monitoring • Generated {format(new Date(), 'PPpp')} • 
          Orchestrator: Apache Airflow • Monitoring: Custom Health Checks • 
          Auto-refresh: {autoRefresh ? 'Enabled (30s)' : 'Disabled'}
        </p>
      </div>
    </div>
  );
}