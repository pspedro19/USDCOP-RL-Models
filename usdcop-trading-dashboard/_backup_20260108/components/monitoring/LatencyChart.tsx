'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface LatencyChartProps {
  refreshInterval?: number;
  minutes?: number;
}

interface LatencyData {
  timestamp: string;
  trading_api?: number;
  ml_analytics?: number;
  pipeline_api?: number;
  websocket?: number;
  postgres?: number;
}

export function LatencyChart({
  refreshInterval = 30000,
  minutes = 5,
}: LatencyChartProps) {
  const [data, setData] = useState<LatencyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLatencyData = async () => {
    try {
      const response = await fetch(
        `/api/health/latency?type=end-to-end&minutes=${minutes}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch latency data');
      }

      const result = await response.json();

      // Add new data point
      const newPoint: LatencyData = {
        timestamp: new Date().toLocaleTimeString(),
        ...result.latency,
      };

      setData((prev) => {
        const updated = [...prev, newPoint];
        // Keep last 20 data points
        return updated.slice(-20);
      });

      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLatencyData();

    const interval = setInterval(fetchLatencyData, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval, minutes]);

  if (loading && data.length === 0) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-red-500">
        <CardContent className="p-6">
          <p className="text-red-500 text-center">{error}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>System Latency Over Time</span>
          <span className="text-sm font-normal text-gray-600">
            Last {minutes} minutes
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis
              label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="avg_ms"
              stroke="#8884d8"
              name="Average"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="p95_ms"
              stroke="#82ca9d"
              name="P95"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="max_ms"
              stroke="#ffc658"
              name="Max"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Current Stats */}
        {data.length > 0 && (
          <div className="mt-4 grid grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-sm text-gray-600">Min</p>
              <p className="text-lg font-semibold">
                {data[data.length - 1].min_ms?.toFixed(0) || 0} ms
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Avg</p>
              <p className="text-lg font-semibold">
                {data[data.length - 1].avg_ms?.toFixed(0) || 0} ms
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">P95</p>
              <p className="text-lg font-semibold">
                {data[data.length - 1].p95_ms?.toFixed(0) || 0} ms
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Max</p>
              <p className="text-lg font-semibold">
                {data[data.length - 1].max_ms?.toFixed(0) || 0} ms
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
