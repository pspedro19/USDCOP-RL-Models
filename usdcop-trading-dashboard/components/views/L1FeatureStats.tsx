'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter
} from 'recharts';
import { fetchLatestPipelineOutput } from '@/lib/services/pipeline';

interface L1FeatureStats {
  feature_count: number;
  quality_score: number;
  completeness: number;
  standardization_rate: number;
  timezone_conversions: number;
  session_breakdown: Record<string, number>;
  anomaly_detection: {
    outliers_detected: number;
    outlier_rate: number;
    anomaly_types: Record<string, number>;
  };
  processing_metrics: {
    avg_processing_time: number;
    records_per_second: number;
    memory_usage: number;
  };
  data_quality_checks: {
    missing_values: number;
    duplicate_records: number;
    invalid_timestamps: number;
    ohlc_violations: number;
  };
  feature_distributions: {
    name: string;
    mean: number;
    std: number;
    min: number;
    max: number;
    skewness: number;
    kurtosis: number;
  }[];
}

interface QualityReport {
  overall_score: number;
  checks: {
    name: string;
    status: 'PASS' | 'WARN' | 'FAIL';
    score: number;
    message: string;
    threshold?: number;
    actual?: number;
  }[];
  recommendations: string[];
}

export default function L1FeatureStats() {
  const [featureStats, setFeatureStats] = useState<L1FeatureStats | null>(null);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  const fetchL1Data = async () => {
    try {
      setError(null);
      
      const pipelineData = await fetchLatestPipelineOutput('L1');
      
      if (pipelineData) {
        // Mock L1 feature statistics - replace with actual data structure
        const mockFeatureStats: L1FeatureStats = {
          feature_count: 45,
          quality_score: 87.5,
          completeness: 94.2,
          standardization_rate: 98.7,
          timezone_conversions: 1247,
          session_breakdown: {
            'Premium (08:00-14:00 COT)': 756,
            'London (03:00-08:00 COT)': 312,
            'Afternoon (14:00-17:00 COT)': 179,
          },
          anomaly_detection: {
            outliers_detected: 23,
            outlier_rate: 1.8,
            anomaly_types: {
              'Price Spike': 12,
              'Volume Anomaly': 7,
              'Gap Detection': 4,
            },
          },
          processing_metrics: {
            avg_processing_time: 2.3,
            records_per_second: 450,
            memory_usage: 78.5,
          },
          data_quality_checks: {
            missing_values: 5,
            duplicate_records: 2,
            invalid_timestamps: 1,
            ohlc_violations: 0,
          },
          feature_distributions: [
            { name: 'Close Price', mean: 4250.5, std: 85.2, min: 4100, max: 4400, skewness: 0.15, kurtosis: -0.8 },
            { name: 'Volume', mean: 12500, std: 3200, min: 500, max: 25000, skewness: 1.2, kurtosis: 2.1 },
            { name: 'Spread', mean: 25.3, std: 8.1, min: 10, max: 65, skewness: 0.9, kurtosis: 1.5 },
          ],
        };
        
        setFeatureStats(mockFeatureStats);
        
        // Mock quality report
        const mockQualityReport: QualityReport = {
          overall_score: 87.5,
          checks: [
            {
              name: 'Data Completeness',
              status: 'PASS',
              score: 94.2,
              message: 'Data completeness is above threshold',
              threshold: 90,
              actual: 94.2,
            },
            {
              name: 'Timezone Standardization',
              status: 'PASS',
              score: 98.7,
              message: 'All timestamps successfully converted to UTC',
              threshold: 95,
              actual: 98.7,
            },
            {
              name: 'OHLC Consistency',
              status: 'PASS',
              score: 100,
              message: 'No OHLC violations detected',
              threshold: 99,
              actual: 100,
            },
            {
              name: 'Missing Value Rate',
              status: 'WARN',
              score: 85.3,
              message: 'Missing values slightly elevated',
              threshold: 95,
              actual: 85.3,
            },
          ],
          recommendations: [
            'Investigate source of missing values in afternoon session',
            'Monitor outlier detection in high volatility periods',
            'Consider implementing additional validation for weekend data',
          ],
        };
        
        setQualityReport(mockQualityReport);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L1 data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL1Data();
    
    const interval = setInterval(fetchL1Data, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [selectedTimeframe]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'PASS': return 'bg-green-500';
      case 'WARN': return 'bg-yellow-500';
      case 'FAIL': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const formatNumber = (num: number, decimals: number = 1) => {
    return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded mb-4"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">L1 Feature Statistics & Quality</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={selectedTimeframe} 
            onChange={(e) => setSelectedTimeframe(e.target.value as any)}
            className="px-3 py-1 border rounded"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24h</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <Badge 
            className={qualityReport && qualityReport.overall_score > 85 ? 'bg-green-500' : 
                     qualityReport && qualityReport.overall_score > 70 ? 'bg-yellow-500' : 'bg-red-500'}
          >
            Quality: {qualityReport?.overall_score.toFixed(1)}%
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

      {/* Key Metrics Cards */}
      {featureStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Features Processed</p>
              <p className="text-2xl font-bold">{featureStats.feature_count}</p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Completeness</p>
              <p className="text-2xl font-bold">{featureStats.completeness.toFixed(1)}%</p>
              <Progress value={featureStats.completeness} className="mt-2" />
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Standardization Rate</p>
              <p className="text-2xl font-bold">{featureStats.standardization_rate.toFixed(1)}%</p>
              <Progress value={featureStats.standardization_rate} className="mt-2" />
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Processing Speed</p>
              <p className="text-2xl font-bold">{formatNumber(featureStats.processing_metrics.records_per_second, 0)}</p>
              <p className="text-xs text-gray-500">records/sec</p>
            </div>
          </Card>
        </div>
      )}

      {/* Quality Checks */}
      {qualityReport && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Data Quality Checks</h3>
          <div className="space-y-3">
            {qualityReport.checks.map((check, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <div className="flex items-center space-x-3">
                  <Badge className={getStatusColor(check.status)}>
                    {check.status}
                  </Badge>
                  <div>
                    <p className="font-medium">{check.name}</p>
                    <p className="text-sm text-gray-600">{check.message}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold">{check.score.toFixed(1)}%</p>
                  {check.threshold && (
                    <p className="text-xs text-gray-500">Target: {check.threshold}%</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Session Breakdown and Anomaly Detection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {featureStats && (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Trading Session Breakdown</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={Object.entries(featureStats.session_breakdown).map(([session, count]) => ({
                      name: session,
                      value: count,
                    }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label
                  >
                    {Object.entries(featureStats.session_breakdown).map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}

        {featureStats && (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Anomaly Detection</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span>Outliers Detected:</span>
                <span className="font-bold">{featureStats.anomaly_detection.outliers_detected}</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Outlier Rate:</span>
                <span className="font-bold">{featureStats.anomaly_detection.outlier_rate}%</span>
              </div>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={Object.entries(featureStats.anomaly_detection.anomaly_types).map(([type, count]) => ({
                    type,
                    count,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Feature Distributions */}
      {featureStats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Feature Distribution Analysis</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={featureStats.feature_distributions}>
                <PolarGrid />
                <PolarAngleAxis dataKey="name" />
                <PolarRadiusAxis />
                <Radar
                  name="Mean"
                  dataKey="mean"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            {featureStats.feature_distributions.map((feature, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded">
                <h4 className="font-medium">{feature.name}</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Mean: <span className="font-medium">{formatNumber(feature.mean)}</span></div>
                  <div>Std: <span className="font-medium">{formatNumber(feature.std)}</span></div>
                  <div>Min: <span className="font-medium">{formatNumber(feature.min)}</span></div>
                  <div>Max: <span className="font-medium">{formatNumber(feature.max)}</span></div>
                  <div>Skewness: <span className="font-medium">{formatNumber(feature.skewness, 2)}</span></div>
                  <div>Kurtosis: <span className="font-medium">{formatNumber(feature.kurtosis, 2)}</span></div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Processing Metrics */}
      {featureStats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Processing Performance</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <p className="text-sm text-gray-600">Avg Processing Time</p>
              <p className="text-2xl font-bold">{featureStats.processing_metrics.avg_processing_time}s</p>
              <Progress value={(5 - featureStats.processing_metrics.avg_processing_time) * 20} className="mt-2" />
            </div>
            
            <div className="text-center">
              <p className="text-sm text-gray-600">Records/Second</p>
              <p className="text-2xl font-bold">{formatNumber(featureStats.processing_metrics.records_per_second, 0)}</p>
              <Progress value={Math.min(100, (featureStats.processing_metrics.records_per_second / 1000) * 100)} className="mt-2" />
            </div>
            
            <div className="text-center">
              <p className="text-sm text-gray-600">Memory Usage</p>
              <p className="text-2xl font-bold">{featureStats.processing_metrics.memory_usage.toFixed(1)}%</p>
              <Progress value={featureStats.processing_metrics.memory_usage} className="mt-2" />
            </div>
          </div>
        </Card>
      )}

      {/* Recommendations */}
      {qualityReport && qualityReport.recommendations.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
          <ul className="space-y-2">
            {qualityReport.recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-yellow-500 mt-1">⚠</span>
                <span>{recommendation}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}