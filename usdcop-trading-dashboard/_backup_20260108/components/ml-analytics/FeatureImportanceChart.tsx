'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  RadialBarChart,
  RadialBar,
  Legend
} from 'recharts';
import {
  BarChart3,
  TrendingUp,
  Zap,
  Target,
  RefreshCw,
  PieChart as PieChartIcon,
  Activity
} from 'lucide-react';

interface FeatureImportance {
  feature_name: string;
  impact_score: number;
  correlation: number;
  importance_rank: number;
}

interface FeatureImportanceChartProps {
  runId?: string;
  modelType?: string;
}

const FEATURE_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1',
  '#d084d0', '#ffb347', '#87d068', '#ffa39e', '#b37feb'
];

const FEATURE_DESCRIPTIONS = {
  rsi: 'Relative Strength Index - Momentum oscillator measuring overbought/oversold conditions',
  macd: 'Moving Average Convergence Divergence - Trend-following momentum indicator',
  bollinger_position: 'Position relative to Bollinger Bands - Price volatility indicator',
  volume_ratio: 'Current volume relative to average - Trading activity measure',
  volatility: 'Price volatility measure - Market uncertainty indicator',
  sma_5: '5-period Simple Moving Average - Short-term trend',
  sma_20: '20-period Simple Moving Average - Medium-term trend',
  ema_12: '12-period Exponential Moving Average - Weighted recent prices',
  atr: 'Average True Range - Volatility measure',
  stochastic: 'Stochastic Oscillator - Momentum indicator'
};

const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  runId,
  modelType = 'PPO'
}) => {
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'bar' | 'radial' | 'pie'>('bar');
  const [selectedFeature, setSelectedFeature] = useState<FeatureImportance | null>(null);

  useEffect(() => {
    loadFeatureImportance();
  }, [runId, modelType]);

  const loadFeatureImportance = async () => {
    try {
      setLoading(true);
      
      const response = await fetch(
        `/api/ml-analytics/predictions?action=feature-impact&runId=${runId || 'latest'}`
      );
      const result = await response.json();
      
      if (result.success) {
        setFeatureImportance(result.data);
        if (result.data.length > 0) {
          setSelectedFeature(result.data[0]);
        }
      }
      
    } catch (error) {
      console.error('Failed to load feature importance:', error);
    } finally {
      setLoading(false);
    }
  };

  const getFeatureDescription = (featureName: string): string => {
    return FEATURE_DESCRIPTIONS[featureName as keyof typeof FEATURE_DESCRIPTIONS] || 
           `${featureName.toUpperCase()} - Technical analysis indicator`;
  };

  const formatFeatureName = (name: string): string => {
    return name.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const getImportanceLevel = (score: number): { label: string; color: string } => {
    if (score >= 80) return { label: 'Critical', color: 'bg-red-100 text-red-800' };
    if (score >= 60) return { label: 'High', color: 'bg-orange-100 text-orange-800' };
    if (score >= 40) return { label: 'Medium', color: 'bg-yellow-100 text-yellow-800' };
    if (score >= 20) return { label: 'Low', color: 'bg-blue-100 text-blue-800' };
    return { label: 'Minimal', color: 'bg-gray-100 text-gray-800' };
  };

  const renderBarChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart
        data={featureImportance}
        margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        layout="vertical"
      >
        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
        <XAxis type="number" domain={[0, 100]} />
        <YAxis 
          type="category" 
          dataKey="feature_name" 
          width={120}
          tickFormatter={formatFeatureName}
        />
        <Tooltip 
          formatter={(value: number) => [`${value.toFixed(1)}%`, 'Impact Score']}
          labelFormatter={(label) => formatFeatureName(label)}
        />
        <Bar dataKey="impact_score" radius={[0, 4, 4, 0]}>
          {featureImportance.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={FEATURE_COLORS[index % FEATURE_COLORS.length]}
              style={{ cursor: 'pointer' }}
              onClick={() => setSelectedFeature(entry)}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );

  const renderRadialChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <RadialBarChart
        cx="50%"
        cy="50%"
        innerRadius="20%"
        outerRadius="90%"
        data={featureImportance}
      >
        <RadialBar
          dataKey="impact_score"
          cornerRadius={4}
          label={{ position: 'insideStart', fill: '#fff', fontSize: 12 }}
        >
          {featureImportance.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={FEATURE_COLORS[index % FEATURE_COLORS.length]}
            />
          ))}
        </RadialBar>
        <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Impact Score']} />
        <Legend 
          iconSize={8}
          layout="vertical"
          verticalAlign="middle"
          align="right"
          formatter={(value) => formatFeatureName(value)}
        />
      </RadialBarChart>
    </ResponsiveContainer>
  );

  const renderPieChart = () => (
    <ResponsiveContainer width="100%" height={400}>
      <PieChart>
        <Pie
          data={featureImportance}
          cx="50%"
          cy="50%"
          outerRadius={120}
          dataKey="impact_score"
          label={({ feature_name, impact_score }) => 
            `${formatFeatureName(feature_name)}: ${impact_score.toFixed(1)}%`
          }
        >
          {featureImportance.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={FEATURE_COLORS[index % FEATURE_COLORS.length]}
            />
          ))}
        </Pie>
        <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Impact Score']} />
      </PieChart>
    </ResponsiveContainer>
  );

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
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Feature</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {featureImportance[0] ? formatFeatureName(featureImportance[0].feature_name) : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              {featureImportance[0] ? `${featureImportance[0].impact_score.toFixed(1)}% impact` : ''}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Features</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{featureImportance.length}</div>
            <p className="text-xs text-muted-foreground">
              {featureImportance.filter(f => f.impact_score >= 20).length} significant
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Impact</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {featureImportance.length > 0 
                ? (featureImportance.reduce((sum, f) => sum + f.impact_score, 0) / featureImportance.length).toFixed(1)
                : 0
              }%
            </div>
            <p className="text-xs text-muted-foreground">Average feature impact</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Feature Importance Analysis</CardTitle>
              <CardDescription>
                Impact of each feature on model predictions ({modelType} model)
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === 'bar' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('bar')}
              >
                <BarChart3 className="h-4 w-4 mr-1" />
                Bar
              </Button>
              <Button
                variant={viewMode === 'radial' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('radial')}
              >
                <Activity className="h-4 w-4 mr-1" />
                Radial
              </Button>
              <Button
                variant={viewMode === 'pie' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('pie')}
              >
                <PieChartIcon className="h-4 w-4 mr-1" />
                Pie
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {featureImportance.length > 0 ? (
            <div>
              {viewMode === 'bar' && renderBarChart()}
              {viewMode === 'radial' && renderRadialChart()}
              {viewMode === 'pie' && renderPieChart()}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No feature importance data available</p>
              <Button onClick={loadFeatureImportance} variant="outline" className="mt-2">
                <RefreshCw className="h-4 w-4 mr-2" />
                Reload Data
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Feature Details */}
      {selectedFeature && (
        <Card>
          <CardHeader>
            <CardTitle>Feature Details</CardTitle>
            <CardDescription>Detailed information about the selected feature</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">
                  {formatFeatureName(selectedFeature.feature_name)}
                </h3>
                <div className="flex items-center gap-2">
                  <Badge className={getImportanceLevel(selectedFeature.impact_score).color}>
                    {getImportanceLevel(selectedFeature.impact_score).label}
                  </Badge>
                  <Badge variant="outline">
                    Rank #{selectedFeature.importance_rank}
                  </Badge>
                </div>
              </div>
              
              <p className="text-muted-foreground text-sm">
                {getFeatureDescription(selectedFeature.feature_name)}
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium mb-2">Impact Score</div>
                  <div className="text-2xl font-bold mb-2">
                    {selectedFeature.impact_score.toFixed(1)}%
                  </div>
                  <Progress value={selectedFeature.impact_score} className="w-full" />
                </div>
                
                <div>
                  <div className="text-sm font-medium mb-2">Correlation</div>
                  <div className="text-2xl font-bold mb-2">
                    {selectedFeature.correlation.toFixed(3)}
                  </div>
                  <Progress 
                    value={Math.abs(selectedFeature.correlation) * 100} 
                    className="w-full" 
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Feature Ranking Table */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Ranking</CardTitle>
          <CardDescription>Complete ranking of all features by importance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2">Rank</th>
                  <th className="text-left py-2">Feature</th>
                  <th className="text-right py-2">Impact Score</th>
                  <th className="text-right py-2">Correlation</th>
                  <th className="text-center py-2">Level</th>
                </tr>
              </thead>
              <tbody>
                {featureImportance.map((feature, index) => (
                  <tr 
                    key={index} 
                    className={`border-b last:border-0 hover:bg-muted/50 cursor-pointer ${
                      selectedFeature?.feature_name === feature.feature_name ? 'bg-muted' : ''
                    }`}
                    onClick={() => setSelectedFeature(feature)}
                  >
                    <td className="py-2 font-bold">
                      #{feature.importance_rank}
                    </td>
                    <td className="py-2">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: FEATURE_COLORS[index % FEATURE_COLORS.length] }}
                        />
                        {formatFeatureName(feature.feature_name)}
                      </div>
                    </td>
                    <td className="text-right py-2 font-mono">
                      {feature.impact_score.toFixed(1)}%
                    </td>
                    <td className={`text-right py-2 font-mono ${
                      feature.correlation >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {feature.correlation >= 0 ? '+' : ''}{feature.correlation.toFixed(3)}
                    </td>
                    <td className="text-center py-2">
                      <Badge 
                        variant="outline" 
                        className={getImportanceLevel(feature.impact_score).color}
                      >
                        {getImportanceLevel(feature.impact_score).label}
                      </Badge>
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

export default FeatureImportanceChart;