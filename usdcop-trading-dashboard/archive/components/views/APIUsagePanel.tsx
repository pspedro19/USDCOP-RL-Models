'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { apiMonitor, APIKeyUsage, APIUsageStats } from '@/lib/services/api-monitor';

export default function APIUsagePanel() {
  const [keyUsages, setKeyUsages] = useState<APIKeyUsage[]>([]);
  const [usageStats, setUsageStats] = useState<APIUsageStats | null>(null);
  const [callHistory, setCallHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState<6 | 24 | 168>(24);

  const fetchAPIData = async () => {
    try {
      const [keyData, statsData, historyData] = await Promise.all([
        Promise.resolve(apiMonitor.getKeyUsage()),
        Promise.resolve(apiMonitor.getUsageStats()),
        Promise.resolve(apiMonitor.getCallHistoryForChart(selectedTimeframe)),
      ]);
      
      setKeyUsages(keyData);
      setUsageStats(statsData);
      setCallHistory(historyData);
    } catch (error) {
      console.error('Error fetching API data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAPIData();
    const interval = setInterval(fetchAPIData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [selectedTimeframe]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ACTIVE': return 'bg-green-500';
      case 'RATE_LIMITED': return 'bg-yellow-500';
      case 'EXPIRED': return 'bg-red-500';
      case 'ERROR': return 'bg-red-600';
      default: return 'bg-gray-500';
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 3,
    }).format(amount);
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  if (isLoading) {
    return <div className="animate-pulse h-96 bg-gray-200 rounded"></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">API Usage Statistics</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={selectedTimeframe} 
            onChange={(e) => setSelectedTimeframe(Number(e.target.value) as any)}
            className="px-3 py-1 border rounded"
          >
            <option value={6}>Last 6 Hours</option>
            <option value={24}>Last 24 Hours</option>
            <option value={168}>Last Week</option>
          </select>
        </div>
      </div>

      {/* Usage Overview */}
      {usageStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4 text-center">
            <p className="text-sm text-gray-600">Calls Today</p>
            <p className="text-2xl font-bold">{usageStats.total_calls_today.toLocaleString()}</p>
          </Card>
          <Card className="p-4 text-center">
            <p className="text-sm text-gray-600">Cost Today</p>
            <p className="text-2xl font-bold">{formatCurrency(usageStats.total_cost_today)}</p>
          </Card>
          <Card className="p-4 text-center">
            <p className="text-sm text-gray-600">Active Keys</p>
            <p className="text-2xl font-bold text-green-600">{usageStats.active_keys}</p>
          </Card>
          <Card className="p-4 text-center">
            <p className="text-sm text-gray-600">Success Rate</p>
            <p className="text-2xl font-bold">{usageStats.success_rate.toFixed(1)}%</p>
          </Card>
        </div>
      )}

      {/* API Key Status */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">API Key Status</h3>
        <div className="space-y-3">
          {keyUsages.map((usage, index) => (
            <div key={usage.key_id} className="flex items-center justify-between p-3 bg-gray-50 rounded">
              <div className="flex items-center space-x-3">
                <Badge className={getStatusColor(usage.status)}>
                  {usage.status}
                </Badge>
                <div>
                  <p className="font-medium">{usage.key_id}</p>
                  <p className="text-sm text-gray-600">
                    {usage.daily_calls}/{usage.daily_limit} calls today
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-bold">{formatCurrency(usage.daily_cost)}</p>
                <p className="text-xs text-gray-500">
                  Last used: {usage.last_used === 'Never' ? 'Never' : new Date(usage.last_used).toLocaleTimeString()}
                </p>
              </div>
              <div className="w-32">
                <Progress value={(usage.daily_calls / usage.daily_limit) * 100} />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Usage Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Call History Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">API Calls Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={callHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="hour" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any, name) => [
                    name === 'cost' ? formatCurrency(value) : value,
                    name === 'calls' ? 'API Calls' : name === 'cost' ? 'Cost' : name
                  ]}
                />
                <Bar dataKey="calls" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Success Rate Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Success Rate Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={callHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="hour" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any) => [`${value.toFixed(1)}%`, 'Success Rate']}
                />
                <Line 
                  type="monotone" 
                  dataKey="successRate" 
                  stroke="#10b981" 
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Cost Breakdown */}
      {usageStats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Cost Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <p className="text-sm text-gray-600">Daily Cost</p>
              <p className="text-2xl font-bold">{formatCurrency(usageStats.total_cost_today)}</p>
              <p className="text-xs text-gray-500">
                Avg per call: {formatCurrency(usageStats.total_calls_today > 0 ? usageStats.total_cost_today / usageStats.total_calls_today : 0)}
              </p>
            </div>
            
            <div className="text-center">
              <p className="text-sm text-gray-600">Monthly Cost</p>
              <p className="text-2xl font-bold">{formatCurrency(usageStats.total_cost_month)}</p>
              <p className="text-xs text-gray-500">
                Projected: {formatCurrency(usageStats.total_cost_month * 30 / new Date().getDate())}
              </p>
            </div>
            
            <div className="text-center">
              <p className="text-sm text-gray-600">Peak Usage Hour</p>
              <p className="text-2xl font-bold">{usageStats.peak_usage_hour}:00</p>
              <p className="text-xs text-gray-500">
                Avg Latency: {usageStats.average_latency_ms}ms
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Rate Limiting Alerts */}
      {keyUsages.filter(k => k.status === 'RATE_LIMITED' || k.status === 'ERROR').length > 0 && (
        <Alert>
          <div className="text-yellow-600">
            <p className="font-semibold">API Key Issues Detected:</p>
            <ul className="mt-2 space-y-1">
              {keyUsages
                .filter(k => k.status === 'RATE_LIMITED' || k.status === 'ERROR')
                .map(k => (
                  <li key={k.key_id}>
                    {k.key_id}: {k.status === 'RATE_LIMITED' ? 'Rate limited' : `Error (${k.error_count} errors)`}
                  </li>
                ))
              }
            </ul>
          </div>
        </Alert>
      )}
    </div>
  );
}