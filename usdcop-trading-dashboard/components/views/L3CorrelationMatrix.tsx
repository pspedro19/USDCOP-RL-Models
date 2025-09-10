'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
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
  ScatterChart,
  Scatter,
  Cell
} from 'recharts';
import { fetchLatestPipelineOutput } from '@/lib/services/pipeline';

interface CorrelationData {
  feature1: string;
  feature2: string;
  correlation: number;
  p_value: number;
  significance: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';
}

interface FeatureImportance {
  feature_name: string;
  importance_score: number;
  feature_type: 'technical' | 'microstructure' | 'temporal';
  description: string;
  stability_score: number;
}

interface L3AnalysisData {
  correlation_matrix: CorrelationData[];
  feature_importance: FeatureImportance[];
  feature_groups: {
    technical_indicators: string[];
    microstructure: string[];
    temporal: string[];
  };
  multicollinearity: {
    vif_scores: { feature: string; vif: number }[];
    high_correlation_pairs: { features: string[]; correlation: number }[];
  };
  feature_selection: {
    selected_features: string[];
    selection_method: string;
    selection_criteria: {
      importance_threshold: number;
      correlation_threshold: number;
    };
  };
  statistical_tests: {
    feature: string;
    normality_test: { statistic: number; p_value: number; is_normal: boolean };
    stationarity_test: { statistic: number; p_value: number; is_stationary: boolean };
  }[];
}

export default function L3CorrelationMatrix() {
  const [analysisData, setAnalysisData] = useState<L3AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'correlation' | 'importance' | 'multicollinearity' | 'tests'>('correlation');
  const [correlationFilter, setCorrelationFilter] = useState<number>(0.5);
  const [importanceFilter, setImportanceFilter] = useState<number>(0.1);

  const fetchL3Data = async () => {
    try {
      setError(null);
      
      const pipelineData = await fetchLatestPipelineOutput('L3');
      
      if (pipelineData) {
        // Mock L3 analysis data - replace with actual data structure
        const mockAnalysisData: L3AnalysisData = {
          correlation_matrix: [
            { feature1: 'RSI_14', feature2: 'Close_Price', correlation: 0.85, p_value: 0.001, significance: 'HIGH' },
            { feature1: 'MACD', feature2: 'Volume', correlation: 0.72, p_value: 0.003, significance: 'HIGH' },
            { feature1: 'BB_Upper', feature2: 'BB_Lower', correlation: 0.95, p_value: 0.000, significance: 'HIGH' },
            { feature1: 'SMA_20', feature2: 'SMA_50', correlation: 0.88, p_value: 0.001, significance: 'HIGH' },
            { feature1: 'ATR', feature2: 'Volatility', correlation: 0.79, p_value: 0.002, significance: 'HIGH' },
            { feature1: 'Hour_Of_Day', feature2: 'Session_Type', correlation: 0.65, p_value: 0.008, significance: 'MEDIUM' },
            { feature1: 'Spread', feature2: 'Volume', correlation: -0.55, p_value: 0.012, significance: 'MEDIUM' },
            { feature1: 'Day_Of_Week', feature2: 'Volatility', correlation: 0.42, p_value: 0.025, significance: 'MEDIUM' },
          ],
          feature_importance: [
            { feature_name: 'Close_Price', importance_score: 0.92, feature_type: 'technical', description: 'Current closing price', stability_score: 0.88 },
            { feature_name: 'RSI_14', importance_score: 0.78, feature_type: 'technical', description: 'Relative Strength Index (14 periods)', stability_score: 0.85 },
            { feature_name: 'MACD', importance_score: 0.73, feature_type: 'technical', description: 'Moving Average Convergence Divergence', stability_score: 0.82 },
            { feature_name: 'Volume', importance_score: 0.68, feature_type: 'microstructure', description: 'Trading volume', stability_score: 0.79 },
            { feature_name: 'ATR', importance_score: 0.65, feature_type: 'technical', description: 'Average True Range', stability_score: 0.76 },
            { feature_name: 'BB_Width', importance_score: 0.61, feature_type: 'technical', description: 'Bollinger Band Width', stability_score: 0.73 },
            { feature_name: 'Spread', importance_score: 0.58, feature_type: 'microstructure', description: 'Bid-ask spread', stability_score: 0.71 },
            { feature_name: 'Hour_Of_Day', importance_score: 0.52, feature_type: 'temporal', description: 'Hour of trading day', stability_score: 0.89 },
            { feature_name: 'SMA_20', importance_score: 0.48, feature_type: 'technical', description: 'Simple Moving Average (20 periods)', stability_score: 0.67 },
            { feature_name: 'Volatility', importance_score: 0.45, feature_type: 'technical', description: 'Price volatility', stability_score: 0.64 },
          ],
          feature_groups: {
            technical_indicators: ['Close_Price', 'RSI_14', 'MACD', 'ATR', 'BB_Width', 'SMA_20', 'Volatility'],
            microstructure: ['Volume', 'Spread', 'Order_Flow'],
            temporal: ['Hour_Of_Day', 'Day_Of_Week', 'Session_Type'],
          },
          multicollinearity: {
            vif_scores: [
              { feature: 'Close_Price', vif: 1.2 },
              { feature: 'RSI_14', vif: 2.1 },
              { feature: 'MACD', vif: 1.8 },
              { feature: 'Volume', vif: 1.5 },
              { feature: 'BB_Upper', vif: 8.5 }, // High VIF indicating multicollinearity
              { feature: 'BB_Lower', vif: 8.3 }, // High VIF indicating multicollinearity
              { feature: 'SMA_20', vif: 3.2 },
              { feature: 'SMA_50', vif: 3.8 },
            ],
            high_correlation_pairs: [
              { features: ['BB_Upper', 'BB_Lower'], correlation: 0.95 },
              { features: ['SMA_20', 'SMA_50'], correlation: 0.88 },
              { features: ['RSI_14', 'Close_Price'], correlation: 0.85 },
            ],
          },
          feature_selection: {
            selected_features: ['Close_Price', 'RSI_14', 'MACD', 'Volume', 'ATR', 'Hour_Of_Day'],
            selection_method: 'Recursive Feature Elimination',
            selection_criteria: {
              importance_threshold: 0.5,
              correlation_threshold: 0.8,
            },
          },
          statistical_tests: [
            {
              feature: 'Close_Price',
              normality_test: { statistic: 2.45, p_value: 0.014, is_normal: false },
              stationarity_test: { statistic: -3.78, p_value: 0.003, is_stationary: true },
            },
            {
              feature: 'RSI_14',
              normality_test: { statistic: 1.23, p_value: 0.218, is_normal: true },
              stationarity_test: { statistic: -4.12, p_value: 0.001, is_stationary: true },
            },
            {
              feature: 'Volume',
              normality_test: { statistic: 5.67, p_value: 0.000, is_normal: false },
              stationarity_test: { statistic: -2.89, p_value: 0.045, is_stationary: true },
            },
          ],
        };
        
        setAnalysisData(mockAnalysisData);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L3 data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL3Data();
    
    const interval = setInterval(fetchL3Data, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const filteredCorrelations = useMemo(() => {
    if (!analysisData) return [];
    return analysisData.correlation_matrix.filter(
      correlation => Math.abs(correlation.correlation) >= correlationFilter
    );
  }, [analysisData, correlationFilter]);

  const filteredImportance = useMemo(() => {
    if (!analysisData) return [];
    return analysisData.feature_importance.filter(
      feature => feature.importance_score >= importanceFilter
    );
  }, [analysisData, importanceFilter]);

  const getCorrelationColor = (correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs >= 0.8) return '#dc2626'; // Strong correlation - red
    if (abs >= 0.6) return '#f59e0b'; // Moderate correlation - amber
    if (abs >= 0.4) return '#10b981'; // Weak correlation - green
    return '#6b7280'; // Very weak - gray
  };

  const getVIFColor = (vif: number) => {
    if (vif >= 10) return '#dc2626'; // High multicollinearity - red
    if (vif >= 5) return '#f59e0b';  // Moderate multicollinearity - amber
    return '#10b981'; // Low multicollinearity - green
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {[1, 2, 3].map(i => (
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
        <h2 className="text-2xl font-bold">L3 Feature Analysis & Correlation</h2>
        <div className="flex items-center space-x-2">
          <select 
            value={selectedView} 
            onChange={(e) => setSelectedView(e.target.value as any)}
            className="px-3 py-1 border rounded"
          >
            <option value="correlation">Correlation Matrix</option>
            <option value="importance">Feature Importance</option>
            <option value="multicollinearity">Multicollinearity</option>
            <option value="tests">Statistical Tests</option>
          </select>
        </div>
      </div>

      {error && (
        <Alert>
          <div className="text-red-600">
            Error: {error}
          </div>
        </Alert>
      )}

      {/* Summary Cards */}
      {analysisData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Total Features</p>
              <p className="text-2xl font-bold">{analysisData.feature_importance.length}</p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Selected Features</p>
              <p className="text-2xl font-bold">{analysisData.feature_selection.selected_features.length}</p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">High Correlations</p>
              <p className="text-2xl font-bold">
                {analysisData.correlation_matrix.filter(c => Math.abs(c.correlation) > 0.7).length}
              </p>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="text-center">
              <p className="text-sm text-gray-600">Multicollinear</p>
              <p className="text-2xl font-bold">
                {analysisData.multicollinearity.vif_scores.filter(v => v.vif > 5).length}
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Correlation Matrix View */}
      {selectedView === 'correlation' && analysisData && (
        <div className="space-y-4">
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Feature Correlation Matrix</h3>
              <div className="flex items-center space-x-2">
                <label className="text-sm">Min Correlation:</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={correlationFilter}
                  onChange={(e) => setCorrelationFilter(parseFloat(e.target.value))}
                  className="w-20"
                />
                <span className="text-sm font-mono">{correlationFilter.toFixed(1)}</span>
              </div>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={filteredCorrelations} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[-1, 1]} />
                  <YAxis 
                    type="category" 
                    dataKey={(entry) => `${entry.feature1} vs ${entry.feature2}`}
                    width={150}
                  />
                  <Tooltip 
                    formatter={(value: any, name, props) => [
                      `${(value as number).toFixed(3)}`,
                      'Correlation'
                    ]}
                    labelFormatter={(label) => `Features: ${label}`}
                  />
                  <Bar dataKey="correlation">
                    {filteredCorrelations.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getCorrelationColor(entry.correlation)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">High Correlation Pairs</h3>
            <div className="space-y-3">
              {analysisData.multicollinearity.high_correlation_pairs.map((pair, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <p className="font-medium">{pair.features.join(' ↔ ')}</p>
                    <Badge className={getCorrelationColor(pair.correlation).replace('#', 'bg-')}>
                      {pair.correlation > 0 ? 'Positive' : 'Negative'}
                    </Badge>
                  </div>
                  <div className="text-right">
                    <p className="font-bold text-lg">{pair.correlation.toFixed(3)}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Feature Importance View */}
      {selectedView === 'importance' && analysisData && (
        <div className="space-y-4">
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Feature Importance Ranking</h3>
              <div className="flex items-center space-x-2">
                <label className="text-sm">Min Importance:</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={importanceFilter}
                  onChange={(e) => setImportanceFilter(parseFloat(e.target.value))}
                  className="w-20"
                />
                <span className="text-sm font-mono">{importanceFilter.toFixed(1)}</span>
              </div>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={filteredImportance} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} />
                  <YAxis type="category" dataKey="feature_name" width={100} />
                  <Tooltip 
                    formatter={(value: any) => [value.toFixed(3), 'Importance']}
                  />
                  <Bar dataKey="importance_score" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(analysisData.feature_groups).map(([groupName, features]) => (
              <Card key={groupName} className="p-4">
                <h4 className="font-semibold mb-3 capitalize">
                  {groupName.replace('_', ' ')} Features
                </h4>
                <div className="space-y-2">
                  {features.map((feature) => {
                    const featureData = analysisData.feature_importance.find(f => f.feature_name === feature);
                    return (
                      <div key={feature} className="flex justify-between items-center text-sm">
                        <span>{feature}</span>
                        {featureData && (
                          <span className="font-medium">
                            {featureData.importance_score.toFixed(2)}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Multicollinearity View */}
      {selectedView === 'multicollinearity' && analysisData && (
        <div className="space-y-4">
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Variance Inflation Factor (VIF) Scores</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analysisData.multicollinearity.vif_scores} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={100} />
                  <Tooltip 
                    formatter={(value: any) => [value.toFixed(2), 'VIF Score']}
                  />
                  <Bar dataKey="vif">
                    {analysisData.multicollinearity.vif_scores.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getVIFColor(entry.vif)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div className="mt-4 text-sm text-gray-600">
              <p><span className="w-3 h-3 bg-green-500 inline-block rounded mr-2"></span>VIF &lt; 5: Low multicollinearity</p>
              <p><span className="w-3 h-3 bg-yellow-500 inline-block rounded mr-2"></span>5 ≤ VIF &lt; 10: Moderate multicollinearity</p>
              <p><span className="w-3 h-3 bg-red-500 inline-block rounded mr-2"></span>VIF ≥ 10: High multicollinearity</p>
            </div>
          </Card>
        </div>
      )}

      {/* Statistical Tests View */}
      {selectedView === 'tests' && analysisData && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Statistical Tests</h3>
          <div className="space-y-4">
            {analysisData.statistical_tests.map((test, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded">
                <h4 className="font-medium mb-3">{test.feature}</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium">Normality Test (Shapiro-Wilk)</p>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Statistic:</span>
                      <span className="font-mono">{test.normality_test.statistic.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">P-value:</span>
                      <span className="font-mono">{test.normality_test.p_value.toFixed(3)}</span>
                    </div>
                    <Badge className={test.normality_test.is_normal ? 'bg-green-500' : 'bg-red-500'}>
                      {test.normality_test.is_normal ? 'Normal' : 'Non-normal'}
                    </Badge>
                  </div>
                  
                  <div>
                    <p className="text-sm font-medium">Stationarity Test (ADF)</p>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Statistic:</span>
                      <span className="font-mono">{test.stationarity_test.statistic.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">P-value:</span>
                      <span className="font-mono">{test.stationarity_test.p_value.toFixed(3)}</span>
                    </div>
                    <Badge className={test.stationarity_test.is_stationary ? 'bg-green-500' : 'bg-red-500'}>
                      {test.stationarity_test.is_stationary ? 'Stationary' : 'Non-stationary'}
                    </Badge>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Feature Selection Summary */}
      {analysisData && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Feature Selection Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <p className="font-medium mb-2">Selection Method:</p>
              <p className="text-gray-600">{analysisData.feature_selection.selection_method}</p>
              
              <p className="font-medium mt-4 mb-2">Criteria:</p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>Importance threshold: {analysisData.feature_selection.selection_criteria.importance_threshold}</li>
                <li>Correlation threshold: {analysisData.feature_selection.selection_criteria.correlation_threshold}</li>
              </ul>
            </div>
            
            <div>
              <p className="font-medium mb-2">Selected Features ({analysisData.feature_selection.selected_features.length}):</p>
              <div className="flex flex-wrap gap-2">
                {analysisData.feature_selection.selected_features.map((feature) => (
                  <Badge key={feature} className="bg-blue-100 text-blue-800">
                    {feature}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}