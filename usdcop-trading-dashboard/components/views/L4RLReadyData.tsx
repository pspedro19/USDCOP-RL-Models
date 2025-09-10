'use client';

import React, { useState, useEffect, useMemo } from 'react';
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
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  Cell,
  ComposedChart
} from 'recharts';
import { fetchLatestPipelineOutput } from '@/lib/services/pipeline';

interface ClipRates {
  feature_name: string;
  clip_rate_percent: number;
  lower_clip_count: number;
  upper_clip_count: number;
  total_values: number;
  clip_thresholds: {
    lower: number;
    upper: number;
  };
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

interface RLDataMetrics {
  total_episodes: number;
  episode_length_stats: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
  };
  reward_distribution: {
    mean_reward: number;
    std_reward: number;
    min_reward: number;
    max_reward: number;
    reward_histogram: { range: string; count: number }[];
  };
  action_space_analysis: {
    action_type: string;
    action_count: number;
    action_distribution: { action: string; probability: number }[];
  };
  state_space_stats: {
    feature_count: number;
    observation_shape: number[];
    normalization_stats: {
      feature: string;
      mean: number;
      std: number;
      min_val: number;
      max_val: number;
    }[];
  };
}

interface L4RLData {
  clip_rates: ClipRates[];
  rl_metrics: RLDataMetrics;
  data_readiness_score: number;
  validation_results: {
    episode_consistency: boolean;
    reward_stability: boolean;
    action_space_valid: boolean;
    state_space_valid: boolean;
    temporal_consistency: boolean;
  };
  preprocessing_stats: {
    scaling_method: string;
    outlier_handling: string;
    missing_value_strategy: string;
    feature_engineering_steps: string[];
  };
  quality_metrics: {
    signal_to_noise_ratio: number;
    feature_stability: number;
    temporal_consistency: number;
    reward_signal_quality: number;
  };
}

export default function L4RLReadyData() {
  const [rlData, setRLData] = useState<L4RLData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'overview' | 'clipping' | 'episodes' | 'rewards' | 'validation'>('overview');
  const [clipSeverityFilter, setClipSeverityFilter] = useState<string>('ALL');

  const fetchL4Data = async () => {
    try {
      setError(null);
      
      const pipelineData = await fetchLatestPipelineOutput('L4');
      
      if (pipelineData) {
        // Mock L4 RL-ready data - replace with actual data structure
        const mockRLData: L4RLData = {
          clip_rates: [
            { feature_name: 'close_price_norm', clip_rate_percent: 2.3, lower_clip_count: 45, upper_clip_count: 67, total_values: 4800, clip_thresholds: { lower: -3.0, upper: 3.0 }, severity: 'LOW' },
            { feature_name: 'volume_norm', clip_rate_percent: 8.7, lower_clip_count: 123, upper_clip_count: 295, total_values: 4800, clip_thresholds: { lower: -2.5, upper: 2.5 }, severity: 'MEDIUM' },
            { feature_name: 'rsi_14', clip_rate_percent: 0.8, lower_clip_count: 12, upper_clip_count: 26, total_values: 4800, clip_thresholds: { lower: 0, upper: 100 }, severity: 'LOW' },
            { feature_name: 'macd_signal', clip_rate_percent: 15.2, lower_clip_count: 234, upper_clip_count: 496, total_values: 4800, clip_thresholds: { lower: -2.0, upper: 2.0 }, severity: 'HIGH' },
            { feature_name: 'spread_norm', clip_rate_percent: 22.1, lower_clip_count: 67, upper_clip_count: 994, total_values: 4800, clip_thresholds: { lower: -1.5, upper: 1.5 }, severity: 'CRITICAL' },
          ],
          rl_metrics: {
            total_episodes: 1247,
            episode_length_stats: {
              mean: 288.5,
              median: 287,
              std: 45.2,
              min: 180,
              max: 360,
            },
            reward_distribution: {
              mean_reward: 0.0023,
              std_reward: 0.0156,
              min_reward: -0.0450,
              max_reward: 0.0380,
              reward_histogram: [
                { range: '[-0.045, -0.030)', count: 45 },
                { range: '[-0.030, -0.015)', count: 156 },
                { range: '[-0.015, 0.000)', count: 387 },
                { range: '[0.000, 0.015)', count: 421 },
                { range: '[0.015, 0.030)', count: 194 },
                { range: '[0.030, 0.045)', count: 44 },
              ],
            },
            action_space_analysis: {
              action_type: 'Discrete',
              action_count: 3,
              action_distribution: [
                { action: 'HOLD', probability: 0.52 },
                { action: 'BUY', probability: 0.24 },
                { action: 'SELL', probability: 0.24 },
              ],
            },
            state_space_stats: {
              feature_count: 42,
              observation_shape: [42],
              normalization_stats: [
                { feature: 'close_price_norm', mean: 0.0001, std: 0.9998, min_val: -2.99, max_val: 2.98 },
                { feature: 'volume_norm', mean: 0.0034, std: 1.0245, min_val: -2.45, max_val: 2.47 },
                { feature: 'rsi_14', mean: 49.87, std: 28.43, min_val: 0.12, max_val: 99.88 },
              ],
            },
          },
          data_readiness_score: 87.3,
          validation_results: {
            episode_consistency: true,
            reward_stability: true,
            action_space_valid: true,
            state_space_valid: true,
            temporal_consistency: false, // Some issue with temporal consistency
          },
          preprocessing_stats: {
            scaling_method: 'StandardScaler',
            outlier_handling: 'Winsorization (1%, 99%)',
            missing_value_strategy: 'Forward Fill + Interpolation',
            feature_engineering_steps: [
              'Technical Indicators Calculation',
              'Temporal Feature Extraction',
              'Normalization & Scaling',
              'Outlier Clipping',
              'Episode Segmentation',
            ],
          },
          quality_metrics: {
            signal_to_noise_ratio: 1.34,
            feature_stability: 0.89,
            temporal_consistency: 0.75, // Lower due to some gaps
            reward_signal_quality: 0.91,
          },
        };
        
        setRLData(mockRLData);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L4 RL data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL4Data();
    
    const interval = setInterval(fetchL4Data, 180000); // Refresh every 3 minutes
    return () => clearInterval(interval);
  }, []);

  const filteredClipRates = useMemo(() => {
    if (!rlData) return [];
    if (clipSeverityFilter === 'ALL') return rlData.clip_rates;
    return rlData.clip_rates.filter(clip => clip.severity === clipSeverityFilter);
  }, [rlData, clipSeverityFilter]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'LOW': return 'bg-green-500';
      case 'MEDIUM': return 'bg-yellow-500';
      case 'HIGH': return 'bg-orange-500';
      case 'CRITICAL': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getValidationIcon = (isValid: boolean) => {
    return isValid ? '✅' : '❌';
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

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
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">L4 RL-Ready Data Analysis</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={selectedView} 
            onChange={(e) => setSelectedView(e.target.value as any)}
            className="px-3 py-1 border rounded"
          >
            <option value="overview">Overview</option>
            <option value="clipping">Clip Analysis</option>
            <option value="episodes">Episodes</option>
            <option value="rewards">Rewards</option>
            <option value="validation">Validation</option>
          </select>
          <Badge 
            className={rlData && rlData.data_readiness_score > 85 ? 'bg-green-500' : 
                     rlData && rlData.data_readiness_score > 70 ? 'bg-yellow-500' : 'bg-red-500'}
          >
            Readiness: {rlData?.data_readiness_score.toFixed(1)}%
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

      {/* Overview */}
      {selectedView === 'overview' && rlData && (
        <>
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="text-center">
                <p className="text-sm text-gray-600">Total Episodes</p>
                <p className="text-2xl font-bold">{rlData.rl_metrics.total_episodes.toLocaleString()}</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-sm text-gray-600">Avg Episode Length</p>
                <p className="text-2xl font-bold">{rlData.rl_metrics.episode_length_stats.mean.toFixed(0)}</p>
                <p className="text-xs text-gray-500">±{rlData.rl_metrics.episode_length_stats.std.toFixed(1)}</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-sm text-gray-600">Feature Count</p>
                <p className="text-2xl font-bold">{rlData.rl_metrics.state_space_stats.feature_count}</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-sm text-gray-600">Mean Reward</p>
                <p className="text-2xl font-bold">{formatNumber(rlData.rl_metrics.reward_distribution.mean_reward, 4)}</p>
                <p className="text-xs text-gray-500">±{formatNumber(rlData.rl_metrics.reward_distribution.std_reward, 4)}</p>
              </div>
            </Card>
          </div>

          {/* Quality Metrics */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Data Quality Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="text-center">
                <p className="text-sm text-gray-600">Signal-to-Noise Ratio</p>
                <p className="text-xl font-bold">{rlData.quality_metrics.signal_to_noise_ratio.toFixed(2)}</p>
                <Progress value={Math.min(100, rlData.quality_metrics.signal_to_noise_ratio * 50)} className="mt-2" />
              </div>
              
              <div className="text-center">
                <p className="text-sm text-gray-600">Feature Stability</p>
                <p className="text-xl font-bold">{formatNumber(rlData.quality_metrics.feature_stability * 100, 1)}%</p>
                <Progress value={rlData.quality_metrics.feature_stability * 100} className="mt-2" />
              </div>
              
              <div className="text-center">
                <p className="text-sm text-gray-600">Temporal Consistency</p>
                <p className="text-xl font-bold">{formatNumber(rlData.quality_metrics.temporal_consistency * 100, 1)}%</p>
                <Progress value={rlData.quality_metrics.temporal_consistency * 100} className="mt-2" />
              </div>
              
              <div className="text-center">
                <p className="text-sm text-gray-600">Reward Quality</p>
                <p className="text-xl font-bold">{formatNumber(rlData.quality_metrics.reward_signal_quality * 100, 1)}%</p>
                <Progress value={rlData.quality_metrics.reward_signal_quality * 100} className="mt-2" />
              </div>
            </div>
          </Card>
        </>
      )}

      {/* Clipping Analysis */}
      {selectedView === 'clipping' && rlData && (
        <>
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Feature Clipping Analysis</h3>
              <div className="flex items-center space-x-2">
                <label className="text-sm">Severity:</label>
                <select 
                  value={clipSeverityFilter} 
                  onChange={(e) => setClipSeverityFilter(e.target.value)}
                  className="px-3 py-1 border rounded"
                >
                  <option value="ALL">All Levels</option>
                  <option value="LOW">Low</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="HIGH">High</option>
                  <option value="CRITICAL">Critical</option>
                </select>
              </div>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={filteredClipRates}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature_name" angle={-45} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip 
                    formatter={(value: any, name) => [
                      name === 'clip_rate_percent' ? `${value.toFixed(2)}%` : value,
                      name === 'clip_rate_percent' ? 'Clip Rate' : name === 'total_values' ? 'Total Values' : name
                    ]}
                  />
                  <Bar yAxisId="left" dataKey="clip_rate_percent" fill="#f59e0b" name="Clip Rate %" />
                  <Line yAxisId="right" type="monotone" dataKey="total_values" stroke="#3b82f6" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Clipping Details</h3>
              <div className="space-y-3">
                {rlData.clip_rates.map((clip, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                    <div className="flex items-center space-x-3">
                      <Badge className={getSeverityColor(clip.severity)}>
                        {clip.severity}
                      </Badge>
                      <div>
                        <p className="font-medium">{clip.feature_name}</p>
                        <p className="text-sm text-gray-600">
                          {clip.lower_clip_count + clip.upper_clip_count} / {clip.total_values} clipped
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold">{clip.clip_rate_percent.toFixed(2)}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Clipping Severity Distribution</h3>
              <div className="space-y-4">
                {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map(severity => {
                  const count = rlData.clip_rates.filter(clip => clip.severity === severity).length;
                  const percentage = (count / rlData.clip_rates.length) * 100;
                  
                  return (
                    <div key={severity} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Badge className={getSeverityColor(severity)}>
                          {severity}
                        </Badge>
                        <span>{count} features</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={percentage} className="w-20" />
                        <span className="text-sm font-medium">{percentage.toFixed(0)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </>
      )}

      {/* Episodes Analysis */}
      {selectedView === 'episodes' && rlData && (
        <>
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Episode Length Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={[
                  { length: 'Min', value: rlData.rl_metrics.episode_length_stats.min },
                  { length: 'Mean', value: rlData.rl_metrics.episode_length_stats.mean },
                  { length: 'Median', value: rlData.rl_metrics.episode_length_stats.median },
                  { length: 'Max', value: rlData.rl_metrics.episode_length_stats.max },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="length" />
                  <YAxis />
                  <Tooltip formatter={(value: any) => [value.toFixed(0), 'Episode Length']} />
                  <Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Action Space Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="font-medium mb-2">Action Type: {rlData.rl_metrics.action_space_analysis.action_type}</p>
                <p className="text-sm text-gray-600 mb-4">Actions Available: {rlData.rl_metrics.action_space_analysis.action_count}</p>
                
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rlData.rl_metrics.action_space_analysis.action_distribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="action" />
                      <YAxis />
                      <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Probability']} />
                      <Bar dataKey="probability" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              <div>
                <p className="font-medium mb-2">Action Distribution</p>
                <div className="space-y-3">
                  {rlData.rl_metrics.action_space_analysis.action_distribution.map((action, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="font-medium">{action.action}</span>
                      <div className="flex items-center space-x-2">
                        <Progress value={action.probability * 100} className="w-24" />
                        <span className="text-sm font-medium">{(action.probability * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </>
      )}

      {/* Rewards Analysis */}
      {selectedView === 'rewards' && rlData && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Reward Distribution</h3>
          <div className="h-64 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rlData.rl_metrics.reward_distribution.reward_histogram}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Mean Reward</p>
              <p className="text-lg font-bold">{formatNumber(rlData.rl_metrics.reward_distribution.mean_reward, 4)}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Std Deviation</p>
              <p className="text-lg font-bold">{formatNumber(rlData.rl_metrics.reward_distribution.std_reward, 4)}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Min Reward</p>
              <p className="text-lg font-bold">{formatNumber(rlData.rl_metrics.reward_distribution.min_reward, 4)}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Max Reward</p>
              <p className="text-lg font-bold">{formatNumber(rlData.rl_metrics.reward_distribution.max_reward, 4)}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Validation Results */}
      {selectedView === 'validation' && rlData && (
        <>
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Data Validation Results</h3>
            <div className="space-y-4">
              {Object.entries(rlData.validation_results).map(([check, isValid]) => (
                <div key={check} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getValidationIcon(isValid)}</span>
                    <div>
                      <p className="font-medium capitalize">{check.replace('_', ' ')}</p>
                      <p className="text-sm text-gray-600">
                        {isValid ? 'Passed validation' : 'Failed validation - requires attention'}
                      </p>
                    </div>
                  </div>
                  <Badge className={isValid ? 'bg-green-500' : 'bg-red-500'}>
                    {isValid ? 'PASS' : 'FAIL'}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Preprocessing Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="space-y-3">
                  <div>
                    <p className="font-medium">Scaling Method</p>
                    <p className="text-sm text-gray-600">{rlData.preprocessing_stats.scaling_method}</p>
                  </div>
                  <div>
                    <p className="font-medium">Outlier Handling</p>
                    <p className="text-sm text-gray-600">{rlData.preprocessing_stats.outlier_handling}</p>
                  </div>
                  <div>
                    <p className="font-medium">Missing Values</p>
                    <p className="text-sm text-gray-600">{rlData.preprocessing_stats.missing_value_strategy}</p>
                  </div>
                </div>
              </div>
              
              <div>
                <p className="font-medium mb-2">Processing Steps</p>
                <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                  {rlData.preprocessing_stats.feature_engineering_steps.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}