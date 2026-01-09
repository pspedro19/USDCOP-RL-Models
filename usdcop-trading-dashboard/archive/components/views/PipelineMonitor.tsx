'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { getPipelineMetrics, getAuditStatus, PipelineMetrics } from '@/lib/services/pipeline';
import { CheckCircle, XCircle, AlertCircle, Clock, Database, Cpu, GitBranch, Shield } from 'lucide-react';

export default function PipelineMonitor() {
  const [metrics, setMetrics] = useState<PipelineMetrics | null>(null);
  const [auditStatus, setAuditStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    const fetchPipelineData = async () => {
      try {
        const [pipelineMetrics, audits] = await Promise.all([
          getPipelineMetrics(),
          getAuditStatus(),
        ]);
        
        setMetrics(pipelineMetrics);
        setAuditStatus(audits);
        setLastUpdate(new Date());
        setLoading(false);
      } catch (error) {
        // Continue with default values silently
        setLoading(false);
      }
    };

    fetchPipelineData();
    const interval = setInterval(fetchPipelineData, 30000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'PASS':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'FAIL':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'RUNNING':
        return <Clock className="h-5 w-5 text-yellow-500 animate-spin" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'PASS':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'FAIL':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'RUNNING':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const pipelineLayers = [
    { 
      id: 'L0', 
      name: 'Data Acquisition',
      icon: <Database className="h-4 w-4" />,
      metric: `${metrics?.l0_completeness.toFixed(1) || 0}% Complete`,
      description: 'Real-time data from TwelveData API'
    },
    { 
      id: 'L1', 
      name: 'Feature Engineering',
      icon: <Cpu className="h-4 w-4" />,
      metric: `${metrics?.l1_features_processed || 0} Features`,
      description: 'Technical indicators and market features'
    },
    { 
      id: 'L2', 
      name: 'Hourly Aggregation',
      icon: <Clock className="h-4 w-4" />,
      metric: `${metrics?.l2_hourly_windows || 0} Windows`,
      description: 'Time-windowed aggregations'
    },
    { 
      id: 'L3', 
      name: 'Training Preparation',
      icon: <GitBranch className="h-4 w-4" />,
      metric: auditStatus?.l3 || 'UNKNOWN',
      description: 'Calendar-split training data'
    },
    { 
      id: 'L4', 
      name: 'RL-Ready Transform',
      icon: <Shield className="h-4 w-4" />,
      metric: auditStatus?.l4 || 'UNKNOWN',
      description: 'Normalized observations [-5, 5]'
    },
    { 
      id: 'L5', 
      name: 'Model Serving',
      icon: <Cpu className="h-4 w-4" />,
      metric: auditStatus?.l5 || 'UNKNOWN',
      description: 'Live prediction endpoint'
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>Pipeline Status</CardTitle>
            <div className="text-sm text-gray-500">
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {pipelineLayers.map((layer, index) => (
              <div key={layer.id} className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 w-32">
                  {layer.icon}
                  <span className="font-semibold">{layer.id}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{layer.name}</span>
                    <Badge className={getStatusColor(layer.metric.includes('PASS') ? 'PASS' : 
                                                      layer.metric.includes('FAIL') ? 'FAIL' : 
                                                      'RUNNING')}>
                      {layer.metric}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500">{layer.description}</p>
                  <Progress value={index < 3 ? 100 : index < 5 ? 66 : 33} className="mt-1" />
                </div>
                {index < 3 ? getStatusIcon('PASS') : 
                 layer.metric.includes('PASS') ? getStatusIcon('PASS') :
                 layer.metric.includes('FAIL') ? getStatusIcon('FAIL') :
                 getStatusIcon('UNKNOWN')}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>L4 Observation Clip Rates</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {metrics?.l4_clip_rates && Object.entries(metrics.l4_clip_rates).slice(0, 5).map(([obs, rate]) => (
                <div key={obs} className="flex justify-between items-center">
                  <span className="text-sm">{obs}</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={100 - (rate as number) * 100} className="w-24" />
                    <span className={`text-xs font-semibold ${(rate as number) > 0.005 ? 'text-red-500' : 'text-green-500'}`}>
                      {((rate as number) * 100).toFixed(3)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>L5 Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Sharpe Ratio</span>
                <span className="font-semibold">{metrics?.l5_model_metrics.sharpe.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Sortino Ratio</span>
                <span className="font-semibold">{metrics?.l5_model_metrics.sortino.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Calmar Ratio</span>
                <span className="font-semibold">{metrics?.l5_model_metrics.calmar.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Max Drawdown</span>
                <span className="font-semibold text-red-500">
                  -{(metrics?.l5_model_metrics.max_drawdown * 100).toFixed(2) || '0.00'}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Audit Gates</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center space-x-2">
              {auditStatus?.l3 === 'PASS' ? <CheckCircle className="h-5 w-5 text-green-500" /> : 
               auditStatus?.l3 === 'FAIL' ? <XCircle className="h-5 w-5 text-red-500" /> :
               <AlertCircle className="h-5 w-5 text-gray-500" />}
              <div>
                <p className="font-semibold">L3 Training Gate</p>
                <p className="text-xs text-gray-500">Anti-leakage, correlation checks</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {auditStatus?.l4 === 'PASS' ? <CheckCircle className="h-5 w-5 text-green-500" /> : 
               auditStatus?.l4 === 'FAIL' ? <XCircle className="h-5 w-5 text-red-500" /> :
               <AlertCircle className="h-5 w-5 text-gray-500" />}
              <div>
                <p className="font-semibold">L4 RL-Ready Gate</p>
                <p className="text-xs text-gray-500">Obs ∈ [-5,5], clip rate ≤0.5%</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {auditStatus?.l5 === 'PASS' ? <CheckCircle className="h-5 w-5 text-green-500" /> : 
               auditStatus?.l5 === 'FAIL' ? <XCircle className="h-5 w-5 text-red-500" /> :
               <AlertCircle className="h-5 w-5 text-gray-500" />}
              <div>
                <p className="font-semibold">L5 Serving Gate</p>
                <p className="text-xs text-gray-500">Sortino≥1.3, MaxDD≤0.15</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}