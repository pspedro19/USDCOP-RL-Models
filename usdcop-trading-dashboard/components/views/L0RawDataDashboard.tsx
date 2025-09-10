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
  Bar
} from 'recharts';
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

export default function L0RawDataDashboard() {
  const [l0Data, setL0Data] = useState<L0Data[]>([]);
  const [l0Stats, setL0Stats] = useState<L0Stats | null>(null);
  const [realtimePrice, setRealtimePrice] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState(30000);

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
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">L0 Raw Data Dashboard</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={refreshInterval} 
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="px-3 py-1 border rounded"
          >
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
            <option value={60000}>1min</option>
            <option value={300000}>5min</option>
          </select>
          <Badge 
            className={l0Stats && l0Stats.dataCompleteness > 90 ? 'bg-green-500' : 
                     l0Stats && l0Stats.dataCompleteness > 70 ? 'bg-yellow-500' : 'bg-red-500'}
          >
            {l0Stats ? formatPercentage(l0Stats.dataCompleteness) : 'N/A'} Complete
          </Badge>
        </div>
      </div>

      {error && (
        <Alert>
          <div className="text-red-600">
            Error: {error}
          </div>
        </Alert>
      )}

      {/* Real-time Price Display */}
      {realtimePrice && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Real-time USD/COP</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Current Price</p>
              <p className="text-2xl font-bold">{formatPrice(Number(realtimePrice.close))}</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">Change</p>
              <p className={`text-xl font-semibold ${
                (Number(realtimePrice.change) ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {realtimePrice.change !== undefined ? 
                  `${Number(realtimePrice.change) >= 0 ? '+' : ''}${Number(realtimePrice.change).toFixed(2)}` : 
                  'N/A'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">Change %</p>
              <p className={`text-xl font-semibold ${
                (Number(realtimePrice.percent_change) ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {realtimePrice.percent_change !== undefined ? 
                  formatPercentage(Number(realtimePrice.percent_change)) : 
                  'N/A'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-600">Volume</p>
              <p className="text-xl font-semibold">{realtimePrice.volume ? Number(realtimePrice.volume).toLocaleString() : 'N/A'}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Statistics Cards */}
      {l0Stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Total Records</p>
              <p className="text-2xl font-bold">{l0Stats.totalRecords.toLocaleString()}</p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Avg Spread</p>
              <p className="text-2xl font-bold">{l0Stats.averageSpread.toFixed(1)} pips</p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Price Range</p>
              <p className="text-lg font-semibold">
                {formatPrice(l0Stats.priceRange.min)} - {formatPrice(l0Stats.priceRange.max)}
              </p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Last Update</p>
              <p className="text-sm font-semibold">
                {new Date(l0Stats.lastUpdate).toLocaleString()}
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Price Chart */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">USD/COP Price Movement</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={l0Data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis 
                domain={['dataMin - 10', 'dataMax + 10']}
                tickFormatter={(value) => value.toFixed(0)}
              />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: any) => [formatPrice(value), 'Price']}
              />
              <Line 
                type="monotone" 
                dataKey="close" 
                stroke="#2563eb" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Volume and Spread Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Volume Profile</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={l0Data.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                />
                <Bar dataKey="volume" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Spread Analysis</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={l0Data.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: any) => [`${value.toFixed(1)} pips`, 'Spread']}
                />
                <Area 
                  type="monotone" 
                  dataKey="spread" 
                  stroke="#f59e0b" 
                  fill="#fbbf24"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Data Source Breakdown */}
      {l0Stats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Data Sources</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(l0Stats.sourceBreakdown).map(([source, count]) => (
              <div key={source} className="text-center p-4 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">{source}</p>
                <p className="text-xl font-bold">{count}</p>
                <p className="text-xs text-gray-500">
                  {formatPercentage((count / l0Stats.totalRecords) * 100)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}