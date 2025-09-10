'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Legend
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  RefreshCw,
  BarChart3
} from 'lucide-react';

interface PredictionPoint {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence?: number;
  error?: number;
  absolute_error?: number;
  percentage_error?: number;
}

interface PredictionMetrics {
  mse: number;
  mae: number;
  rmse: number;
  mape: number;
  accuracy: number;
  correlation: number;
  direction_accuracy: number;
}

interface PredictionsVsActualsChartProps {
  runId?: string;
  timeRange?: string;
  limit?: number;
}

const PredictionsVsActualsChart: React.FC<PredictionsVsActualsChartProps> = ({
  runId,
  timeRange = '24h',
  limit = 50
}) => {
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [metrics, setMetrics] = useState<PredictionMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'timeseries' | 'scatter'>('timeseries');
  const [errorStats, setErrorStats] = useState<any>(null);

  useEffect(() => {
    loadPredictionData();
  }, [runId, timeRange, limit]);

  const loadPredictionData = async () => {
    try {
      setLoading(true);
      
      // Load prediction data
      const dataResponse = await fetch(
        `/api/ml-analytics/predictions?action=data&runId=${runId || 'latest'}&limit=${limit}&timeRange=${timeRange}`
      );
      const dataResult = await dataResponse.json();
      
      if (dataResult.success) {
        setPredictions(dataResult.data);
      }
      
      // Load metrics
      const metricsResponse = await fetch(
        `/api/ml-analytics/predictions?action=metrics&runId=${runId || 'latest'}&limit=${limit}`
      );
      const metricsResult = await metricsResponse.json();
      
      if (metricsResult.success) {
        setMetrics(metricsResult.data.metrics);
        
        // Calculate error statistics
        const errors = dataResult.data?.map((p: PredictionPoint) => p.error || 0) || [];
        const absErrors = dataResult.data?.map((p: PredictionPoint) => p.absolute_error || 0) || [];
        
        if (errors.length > 0) {
          const sortedErrors = [...errors].sort((a, b) => a - b);
          const sortedAbsErrors = [...absErrors].sort((a, b) => a - b);
          
          setErrorStats({
            mean_error: errors.reduce((a, b) => a + b, 0) / errors.length,
            median_error: sortedErrors[Math.floor(sortedErrors.length / 2)],
            p95_abs_error: sortedAbsErrors[Math.floor(sortedAbsErrors.length * 0.95)],
            error_std: Math.sqrt(errors.reduce((sum, err) => sum + Math.pow(err - (errors.reduce((a, b) => a + b, 0) / errors.length), 2), 0) / errors.length)
          });
        }
      }
      
    } catch (error) {
      console.error('Failed to load prediction data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatValue = (value: number, decimals = 4) => {
    return value.toFixed(decimals);
  };

  const formatPercentage = (value: number, decimals = 1) => {
    return `${value.toFixed(decimals)}%`;
  };

  const getErrorColor = (error: number | undefined) => {
    if (!error) return '#8884d8';
    const absError = Math.abs(error);
    if (absError < 10) return '#00C851'; // Green for low error
    if (absError < 50) return '#ffbb33'; // Yellow for medium error
    return '#ff4444'; // Red for high error
  };

  const calculateIdealLine = () => {
    if (predictions.length === 0) return [];
    
    const minVal = Math.min(...predictions.map(p => Math.min(p.actual, p.predicted)));
    const maxVal = Math.max(...predictions.map(p => Math.max(p.actual, p.predicted)));
    
    return [
      { actual: minVal, predicted: minVal },
      { actual: maxVal, predicted: maxVal }
    ];
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Metrics Summary */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">MAPE</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatPercentage(metrics.mape)}</div>
              <p className="text-xs text-muted-foreground">Mean Absolute Percentage Error</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Correlation</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatValue(metrics.correlation, 3)}</div>
              <p className="text-xs text-muted-foreground">Prediction-Actual Correlation</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Direction Accuracy</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatPercentage(metrics.direction_accuracy)}</div>
              <p className="text-xs text-muted-foreground">Correct trend prediction</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">RMSE</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatValue(metrics.rmse, 2)}</div>
              <p className="text-xs text-muted-foreground">Root Mean Square Error</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Predictions vs Actuals</CardTitle>
              <CardDescription>
                Model prediction accuracy over time ({predictions.length} data points)
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === 'timeseries' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('timeseries')}
              >
                Time Series
              </Button>
              <Button
                variant={viewMode === 'scatter' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('scatter')}
              >
                Scatter Plot
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {predictions.length > 0 ? (
            <div className="h-96">
              {viewMode === 'timeseries' ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={predictions}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                      dataKey="timestamp"
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis 
                      tickFormatter={(value) => value.toFixed(0)}
                    />
                    <Tooltip 
                      labelFormatter={(value) => `Time: ${new Date(value).toLocaleString()}`}
                      formatter={(value: number, name: string) => [
                        formatValue(value, 2), 
                        name === 'actual' ? 'Actual' : 'Predicted'
                      ]}
                      contentStyle={{
                        backgroundColor: '#f8f9fa',
                        border: '1px solid #dee2e6',
                        borderRadius: '4px'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="actual" 
                      stroke="#2563eb" 
                      strokeWidth={2}
                      dot={{ fill: '#2563eb', strokeWidth: 0, r: 3 }}
                      name="Actual Values"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#dc2626" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ fill: '#dc2626', strokeWidth: 0, r: 3 }}
                      name="Predictions"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={predictions} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis 
                      type="number" 
                      dataKey="actual" 
                      name="Actual"
                      tickFormatter={(value) => value.toFixed(0)}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="predicted" 
                      name="Predicted"
                      tickFormatter={(value) => value.toFixed(0)}
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value: number, name: string) => [formatValue(value, 2), name]}
                      labelFormatter={() => 'Prediction Point'}
                    />
                    <Scatter 
                      name="Predictions"
                      data={predictions}
                      fill="#8884d8"
                    />
                    {/* Perfect prediction line */}
                    <ReferenceLine 
                      segment={calculateIdealLine().map(point => ({ x: point.actual, y: point.predicted }))} 
                      stroke="#ff7300" 
                      strokeDasharray="5 5"
                      strokeWidth={2}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No prediction data available</p>
              <Button onClick={loadPredictionData} variant="outline" className="mt-2">
                <RefreshCw className="h-4 w-4 mr-2" />
                Reload Data
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error Analysis */}
      {errorStats && (
        <Card>
          <CardHeader>
            <CardTitle>Error Analysis</CardTitle>
            <CardDescription>Statistical analysis of prediction errors</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {formatValue(errorStats.mean_error, 3)}
                </div>
                <p className="text-sm text-muted-foreground">Mean Error</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {formatValue(errorStats.median_error, 3)}
                </div>
                <p className="text-sm text-muted-foreground">Median Error</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {formatValue(errorStats.p95_abs_error, 3)}
                </div>
                <p className="text-sm text-muted-foreground">95th Percentile |Error|</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {formatValue(errorStats.error_std, 3)}
                </div>
                <p className="text-sm text-muted-foreground">Error Std Dev</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Predictions</CardTitle>
          <CardDescription>Latest model predictions with accuracy details</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Time</th>
                  <th className="text-right py-2">Actual</th>
                  <th className="text-right py-2">Predicted</th>
                  <th className="text-right py-2">Error</th>
                  <th className="text-right py-2">Error %</th>
                  <th className="text-right py-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(-10).reverse().map((point, index) => (
                  <tr key={index} className="border-b last:border-0 hover:bg-muted/50">
                    <td className="py-2 text-xs">
                      {new Date(point.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="text-right py-2 font-mono">
                      {formatValue(point.actual, 2)}
                    </td>
                    <td className="text-right py-2 font-mono">
                      {formatValue(point.predicted, 2)}
                    </td>
                    <td className={`text-right py-2 font-mono ${
                      (point.error || 0) > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {point.error ? (point.error > 0 ? '+' : '') + formatValue(point.error, 2) : 'N/A'}
                    </td>
                    <td className="text-right py-2 font-mono">
                      {point.percentage_error ? formatPercentage(point.percentage_error, 1) : 'N/A'}
                    </td>
                    <td className="text-right py-2">
                      {point.confidence ? (
                        <Badge 
                          variant={point.confidence > 0.8 ? 'default' : point.confidence > 0.6 ? 'secondary' : 'outline'}
                          className="text-xs"
                        >
                          {formatPercentage(point.confidence * 100, 0)}
                        </Badge>
                      ) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictionsVsActualsChart;