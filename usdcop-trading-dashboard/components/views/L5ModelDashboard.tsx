'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, ScatterChart, Scatter, Cell,
  RadialBarChart, RadialBar, PieChart, Pie
} from 'recharts';
import { fetchLatestPipelineOutput } from '@/lib/services/pipeline';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target,
  Clock,
  Cpu,
  BarChart3,
  Shield,
  CheckCircle2,
  AlertCircle,
  Gauge,
  Network
} from 'lucide-react';

interface ModelPrediction {
  timestamp: string;
  predicted_action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  expected_return: number;
  risk_score: number;
  features_used: string[];
}

interface LatencyMetrics {
  avg_inference_time: number;
  p95_inference_time: number;
  p99_inference_time: number;
  total_predictions: number;
  predictions_per_second: number;
  model_load_time: number;
}

interface ModelPerformance {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
}

interface L5ServingData {
  latest_predictions: ModelPrediction[];
  latency_metrics: LatencyMetrics;
  model_performance: ModelPerformance;
  model_info: {
    model_name: string;
    version: string;
    last_trained: string;
    training_samples: number;
    feature_count: number;
  };
  serving_status: {
    is_healthy: boolean;
    last_update: string;
    error_rate: number;
    uptime_minutes: number;
  };
}

export default function L5ModelDashboard() {
  const [servingData, setServingData] = useState<L5ServingData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchL5Data = async () => {
    try {
      setError(null);
      const pipelineData = await fetchLatestPipelineOutput('L5');
      
      // Mock L5 serving data
      const mockServingData: L5ServingData = {
        latest_predictions: [
          { timestamp: '2025-09-01T10:30:00Z', predicted_action: 'BUY', confidence: 0.87, expected_return: 0.0045, risk_score: 0.23, features_used: ['RSI_14', 'MACD', 'Volume'] },
          { timestamp: '2025-09-01T10:25:00Z', predicted_action: 'HOLD', confidence: 0.92, expected_return: 0.0012, risk_score: 0.15, features_used: ['RSI_14', 'ATR', 'Hour'] },
          { timestamp: '2025-09-01T10:20:00Z', predicted_action: 'SELL', confidence: 0.78, expected_return: -0.0023, risk_score: 0.31, features_used: ['MACD', 'BB_Width', 'Spread'] },
        ],
        latency_metrics: {
          avg_inference_time: 23.4,
          p95_inference_time: 45.2,
          p99_inference_time: 67.8,
          total_predictions: 15847,
          predictions_per_second: 42.3,
          model_load_time: 2340,
        },
        model_performance: {
          accuracy: 0.734,
          precision: 0.712,
          recall: 0.689,
          f1_score: 0.698,
          sharpe_ratio: 1.87,
          sortino_ratio: 2.34,
          max_drawdown: 0.087,
          win_rate: 0.623,
          profit_factor: 1.45,
        },
        model_info: {
          model_name: 'USDCOP_RL_Agent_v2.1',
          version: '2.1.0',
          last_trained: '2025-08-29T14:30:00Z',
          training_samples: 485920,
          feature_count: 42,
        },
        serving_status: {
          is_healthy: true,
          last_update: '2025-09-01T10:30:15Z',
          error_rate: 0.023,
          uptime_minutes: 14523,
        },
      };
      
      setServingData(mockServingData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L5 data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL5Data();
    const interval = setInterval(fetchL5Data, 30000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-800/60 rounded mb-4"></div>
          <div className="grid grid-cols-4 gap-4 mb-6">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-24 bg-slate-800/60 rounded-xl"></div>
            ))}
          </div>
          <div className="h-64 bg-slate-800/60 rounded-xl"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 bg-slate-900 min-h-screen p-6">
      {/* Enhanced Header with Glassmorphism */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-between items-center bg-slate-900/80 backdrop-blur-md border border-cyan-400/20 rounded-xl p-6 shadow-2xl shadow-cyan-400/10"
      >
        <div className="flex items-center space-x-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
          >
            <Brain className="h-8 w-8 text-white" />
          </motion.div>
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
              L5 Model Serving Dashboard
            </h2>
            <p className="text-slate-400 font-mono text-sm">Real-time AI predictions & performance monitoring</p>
          </div>
        </div>
        
        <motion.div
          animate={servingData?.serving_status.is_healthy ? { scale: [1, 1.05, 1] } : {}}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Badge 
            variant={servingData?.serving_status.is_healthy ? 'success' : 'destructive'}
            className="text-sm px-4 py-2 font-mono"
          >
            <div className="flex items-center space-x-2">
              {servingData?.serving_status.is_healthy ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
              <span>{servingData?.serving_status.is_healthy ? 'HEALTHY' : 'ERROR'}</span>
            </div>
          </Badge>
        </motion.div>
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Alert className="bg-red-900/20 border-red-500/50 text-red-400">
            <AlertCircle className="h-4 w-4" />
            <div>Error: {error}</div>
          </Alert>
        </motion.div>
      )}

      {servingData && (
        <>
          {/* Enhanced Key Metrics with Gradients and Animations */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-4 gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            {/* Avg Latency Card */}
            <motion.div
              whileHover={{ scale: 1.02, rotateY: 5 }}
              className="group"
            >
              <Card variant="glass" className="p-6 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/10 to-blue-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-center mb-3">
                    <Clock className="h-5 w-5 text-cyan-400 mr-2" />
                    <p className="text-sm text-slate-400 font-mono uppercase tracking-wide">Avg Latency</p>
                  </div>
                  <motion.p 
                    className="text-3xl font-bold text-cyan-400 font-mono"
                    animate={{ opacity: [0.7, 1, 0.7] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    {servingData.latency_metrics.avg_inference_time.toFixed(1)}ms
                  </motion.p>
                  <div className="mt-2 w-full bg-slate-800 rounded-full h-1">
                    <motion.div 
                      className="bg-gradient-to-r from-cyan-400 to-blue-400 h-1 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, servingData.latency_metrics.avg_inference_time)}%` }}
                      transition={{ duration: 1, delay: 0.5 }}
                    ></motion.div>
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Accuracy Card */}
            <motion.div
              whileHover={{ scale: 1.02, rotateY: 5 }}
              className="group"
            >
              <Card variant="glass" className="p-6 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-400/10 to-green-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-center mb-3">
                    <Target className="h-5 w-5 text-emerald-400 mr-2" />
                    <p className="text-sm text-slate-400 font-mono uppercase tracking-wide">Accuracy</p>
                  </div>
                  <motion.p 
                    className="text-3xl font-bold text-emerald-400 font-mono"
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    {(servingData.model_performance.accuracy * 100).toFixed(1)}%
                  </motion.p>
                  <div className="mt-2">
                    <Progress 
                      variant="shimmer" 
                      value={servingData.model_performance.accuracy * 100} 
                      className="h-1"
                    />
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Sharpe Ratio Card */}
            <motion.div
              whileHover={{ scale: 1.02, rotateY: 5 }}
              className="group"
            >
              <Card variant="glass" className="p-6 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-400/10 to-pink-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-center mb-3">
                    <BarChart3 className="h-5 w-5 text-purple-400 mr-2" />
                    <p className="text-sm text-slate-400 font-mono uppercase tracking-wide">Sharpe Ratio</p>
                  </div>
                  <motion.p 
                    className="text-3xl font-bold text-purple-400 font-mono"
                    animate={{ opacity: [0.8, 1, 0.8] }}
                    transition={{ duration: 2.5, repeat: Infinity }}
                  >
                    {servingData.model_performance.sharpe_ratio.toFixed(2)}
                  </motion.p>
                  <div className="mt-2 text-xs text-slate-500">
                    {servingData.model_performance.sharpe_ratio > 2 ? 'Excellent' : 
                     servingData.model_performance.sharpe_ratio > 1 ? 'Good' : 'Fair'}
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Predictions per Second Card */}
            <motion.div
              whileHover={{ scale: 1.02, rotateY: 5 }}
              className="group"
            >
              <Card variant="glass" className="p-6 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/10 to-orange-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-center mb-3">
                    <Zap className="h-5 w-5 text-yellow-400 mr-2 animate-pulse" />
                    <p className="text-sm text-slate-400 font-mono uppercase tracking-wide">Predictions/sec</p>
                  </div>
                  <motion.p 
                    className="text-3xl font-bold text-yellow-400 font-mono"
                    animate={{ scale: [0.95, 1.05, 0.95] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    {servingData.latency_metrics.predictions_per_second.toFixed(1)}
                  </motion.p>
                  <div className="mt-2">
                    <motion.div 
                      className="flex space-x-1 justify-center"
                      animate={{ opacity: [0.4, 1, 0.4] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    >
                      {[1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="w-1 h-4 bg-yellow-400 rounded-full"></div>
                      ))}
                    </motion.div>
                  </div>
                </div>
              </Card>
            </motion.div>
          </motion.div>

          {/* Enhanced Latest Predictions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card variant="glass" className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <Network className="h-6 w-6 text-cyan-400" />
                  <h3 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                    Recent AI Predictions
                  </h3>
                </div>
                <Badge variant="glow" className="px-3 py-1">
                  LIVE FEED
                </Badge>
              </div>
              
              <div className="space-y-4">
                <AnimatePresence>
                  {servingData.latest_predictions.map((pred, index) => (
                    <motion.div 
                      key={index}
                      initial={{ opacity: 0, x: -20, scale: 0.9 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, x: 20, scale: 0.9 }}
                      transition={{ delay: index * 0.1 }}
                      className="group relative bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4 hover:border-cyan-400/50 transition-all duration-300"
                    >
                      {/* Prediction Action and Confidence */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-4">
                          <motion.div
                            animate={{ 
                              scale: pred.predicted_action === 'BUY' ? [1, 1.1, 1] : 
                                     pred.predicted_action === 'SELL' ? [1, 0.9, 1] : 1 
                            }}
                            transition={{ duration: 2, repeat: Infinity }}
                          >
                            <Badge 
                              variant={
                                pred.predicted_action === 'BUY' ? 'success' :
                                pred.predicted_action === 'SELL' ? 'destructive' : 'secondary'
                              }
                              className="text-sm font-mono px-3 py-1"
                            >
                              {pred.predicted_action === 'BUY' && <TrendingUp className="h-3 w-3 mr-1" />}
                              {pred.predicted_action === 'SELL' && <TrendingDown className="h-3 w-3 mr-1" />}
                              {pred.predicted_action === 'HOLD' && <Activity className="h-3 w-3 mr-1" />}
                              {pred.predicted_action}
                            </Badge>
                          </motion.div>
                          
                          <div>
                            <div className="flex items-center space-x-2">
                              <span className="text-sm text-slate-400 font-mono">Confidence:</span>
                              <span className="font-bold text-white">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                            <p className="text-xs text-slate-500 font-mono">
                              {new Date(pred.timestamp).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        
                        {/* Return and Risk */}
                        <div className="text-right">
                          <div className={`font-bold text-lg font-mono ${
                            pred.expected_return >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {pred.expected_return >= 0 ? '+' : ''}
                            {(pred.expected_return * 100).toFixed(3)}%
                          </div>
                          <div className="flex items-center space-x-2 text-xs">
                            <Shield className="h-3 w-3 text-slate-400" />
                            <span className="text-slate-400">Risk: {(pred.risk_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Confidence Meter */}
                      <div className="mb-3">
                        <div className="flex justify-between text-xs text-slate-400 mb-1">
                          <span>Confidence Level</span>
                          <span>{(pred.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-700/50 rounded-full h-2 overflow-hidden">
                          <motion.div 
                            className={`h-full rounded-full ${
                              pred.confidence > 0.8 ? 'bg-gradient-to-r from-emerald-500 to-green-400' :
                              pred.confidence > 0.6 ? 'bg-gradient-to-r from-yellow-500 to-amber-400' :
                              'bg-gradient-to-r from-red-500 to-orange-400'
                            }`}
                            initial={{ width: 0 }}
                            animate={{ width: `${pred.confidence * 100}%` }}
                            transition={{ duration: 1, delay: index * 0.2 }}
                          />
                        </div>
                      </div>
                      
                      {/* Features Used */}
                      <div className="flex flex-wrap gap-1">
                        <span className="text-xs text-slate-500">Features:</span>
                        {pred.features_used.map((feature, i) => (
                          <motion.div
                            key={feature}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.1 + i * 0.1 }}
                          >
                            <Badge 
                              variant="outline" 
                              className="text-xs px-2 py-0 font-mono bg-slate-800/60 text-cyan-400 border-cyan-400/30"
                            >
                              {feature}
                            </Badge>
                          </motion.div>
                        ))}
                      </div>
                      
                      {/* Hover Glow Effect */}
                      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-400/5 to-purple-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </Card>
          </motion.div>

          {/* Enhanced Performance Metrics */}
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            {/* Model Performance */}
            <Card variant="glass" className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <Gauge className="h-6 w-6 text-purple-400" />
                <h3 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Model Performance
                </h3>
              </div>
              
              <div className="space-y-4">
                {Object.entries(servingData.model_performance).map(([metric, value], index) => {
                  const percentage = metric.includes('ratio') || metric === 'profit_factor' ? null : value * 100;
                  const displayValue = percentage !== null ? `${percentage.toFixed(1)}%` : value.toFixed(2);
                  
                  return (
                    <motion.div 
                      key={metric}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.8 + index * 0.1 }}
                      className="group"
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="capitalize text-slate-300 font-mono text-sm">
                          {metric.replace('_', ' ')}
                        </span>
                        <span className={`font-bold text-lg font-mono ${
                          metric === 'accuracy' ? 'text-emerald-400' :
                          metric === 'sharpe_ratio' ? 'text-purple-400' :
                          metric === 'win_rate' ? 'text-cyan-400' :
                          'text-white'
                        }`}>
                          {displayValue}
                        </span>
                      </div>
                      
                      {/* Performance Bar */}
                      <div className="w-full bg-slate-800/60 rounded-full h-2 overflow-hidden">
                        <motion.div 
                          className={`h-full rounded-full ${
                            metric === 'accuracy' ? 'bg-gradient-to-r from-emerald-500 to-green-400' :
                            metric === 'sharpe_ratio' ? 'bg-gradient-to-r from-purple-500 to-pink-400' :
                            metric === 'win_rate' ? 'bg-gradient-to-r from-cyan-500 to-blue-400' :
                            'bg-gradient-to-r from-slate-500 to-slate-400'
                          }`}
                          initial={{ width: 0 }}
                          animate={{ 
                            width: percentage !== null ? `${percentage}%` : 
                                  metric.includes('ratio') ? `${Math.min(100, (value / 3) * 100)}%` :
                                  `${Math.min(100, value * 100)}%`
                          }}
                          transition={{ duration: 1.5, delay: 0.8 + index * 0.1 }}
                        />
                      </div>
                      
                      {/* Hover indicator */}
                      <div className="h-0.5 bg-gradient-to-r from-purple-400/20 to-pink-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-full mt-1" />
                    </motion.div>
                  );
                })}
              </div>
            </Card>

            {/* Enhanced Latency Distribution */}
            <Card variant="glass" className="p-6">
              <div className="flex items-center space-x-3 mb-6">
                <Cpu className="h-6 w-6 text-cyan-400" />
                <h3 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Latency Distribution
                </h3>
              </div>
              
              <div className="space-y-6">
                {/* Average Latency */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1 }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-300 font-mono text-sm">Average</span>
                    <motion.span 
                      className="font-bold text-cyan-400 font-mono text-lg"
                      animate={{ opacity: [0.7, 1, 0.7] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      {servingData.latency_metrics.avg_inference_time.toFixed(1)}ms
                    </motion.span>
                  </div>
                  <Progress 
                    variant="glow" 
                    value={Math.min(100, servingData.latency_metrics.avg_inference_time)} 
                    className="h-2"
                  />
                </motion.div>
                
                {/* 95th Percentile */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.2 }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-300 font-mono text-sm">95th percentile</span>
                    <span className="font-bold text-yellow-400 font-mono text-lg">
                      {servingData.latency_metrics.p95_inference_time.toFixed(1)}ms
                    </span>
                  </div>
                  <Progress 
                    variant="gradient" 
                    value={Math.min(100, servingData.latency_metrics.p95_inference_time)} 
                    className="h-2"
                  />
                </motion.div>
                
                {/* 99th Percentile */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.4 }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-300 font-mono text-sm">99th percentile</span>
                    <span className="font-bold text-red-400 font-mono text-lg">
                      {servingData.latency_metrics.p99_inference_time.toFixed(1)}ms
                    </span>
                  </div>
                  <Progress 
                    variant="shimmer" 
                    value={Math.min(100, servingData.latency_metrics.p99_inference_time)} 
                    className="h-2"
                  />
                </motion.div>
                
                {/* Additional Metrics */}
                <div className="pt-4 border-t border-slate-700/50">
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                      <div className="text-xs text-slate-400 font-mono mb-1">Total Predictions</div>
                      <div className="text-lg font-bold text-white font-mono">
                        {servingData.latency_metrics.total_predictions.toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-slate-400 font-mono mb-1">Model Load Time</div>
                      <div className="text-lg font-bold text-purple-400 font-mono">
                        {(servingData.latency_metrics.model_load_time / 1000).toFixed(1)}s
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        </>
      )}
    </div>
  );
}