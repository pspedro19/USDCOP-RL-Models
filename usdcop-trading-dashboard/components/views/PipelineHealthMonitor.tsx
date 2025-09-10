'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getPipelineHealth, PipelineHealthStatus } from '@/lib/services/pipeline';

export default function PipelineHealthMonitor() {
  const [healthData, setHealthData] = useState<PipelineHealthStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealthData = async () => {
    try {
      setError(null);
      const health = await getPipelineHealth();
      setHealthData(health);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch health data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
    const interval = setInterval(fetchHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'HEALTHY': return 'bg-green-500';
      case 'WARNING': return 'bg-yellow-500';
      case 'ERROR': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getOverallHealth = () => {
    if (healthData.length === 0) return 'UNKNOWN';
    const errorCount = healthData.filter(h => h.status === 'ERROR').length;
    const warningCount = healthData.filter(h => h.status === 'WARNING').length;
    
    if (errorCount > 0) return 'ERROR';
    if (warningCount > 0) return 'WARNING';
    return 'HEALTHY';
  };

  if (isLoading) {
    return <div className="animate-pulse h-96 bg-gray-200 rounded"></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Pipeline Health Monitor</h2>
        <Badge className={getStatusColor(getOverallHealth())}>
          Overall: {getOverallHealth()}
        </Badge>
      </div>

      {error && <Alert><div className="text-red-600">Error: {error}</div></Alert>}

      {/* Health Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {healthData.map((health) => (
          <Card key={health.layer} className="p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">{health.layer}</h3>
              <Badge className={getStatusColor(health.status)}>
                {health.status}
              </Badge>
            </div>
            <div className="text-sm text-gray-600">
              <p>Last Update: {health.last_update === 'Never' ? 'Never' : new Date(health.last_update).toLocaleString()}</p>
              <p>Data Age: {health.data_freshness_hours.toFixed(1)}h</p>
              {health.error_message && (
                <p className="text-red-600 mt-2">{health.error_message}</p>
              )}
            </div>
            <Progress 
              value={Math.max(0, Math.min(100, (24 - health.data_freshness_hours) / 24 * 100))} 
              className="mt-2" 
            />
          </Card>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4 text-center">
          <p className="text-sm text-gray-600">Healthy Layers</p>
          <p className="text-2xl font-bold text-green-600">
            {healthData.filter(h => h.status === 'HEALTHY').length}
          </p>
        </Card>
        <Card className="p-4 text-center">
          <p className="text-sm text-gray-600">Warnings</p>
          <p className="text-2xl font-bold text-yellow-600">
            {healthData.filter(h => h.status === 'WARNING').length}
          </p>
        </Card>
        <Card className="p-4 text-center">
          <p className="text-sm text-gray-600">Errors</p>
          <p className="text-2xl font-bold text-red-600">
            {healthData.filter(h => h.status === 'ERROR').length}
          </p>
        </Card>
        <Card className="p-4 text-center">
          <p className="text-sm text-gray-600">Total Layers</p>
          <p className="text-2xl font-bold">{healthData.length}</p>
        </Card>
      </div>
    </div>
  );
}