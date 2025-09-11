'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Activity, Target, BarChart3, AlertTriangle, CheckCircle, 
  XCircle, Clock, Zap, Database, GitBranch, Layers, Settings,
  TrendingUp, TrendingDown, Eye, EyeOff, RotateCcw, Play, Pause,
  Cpu, MemoryStick, HardDrive, Network, Timer, Gauge
} from 'lucide-react';

// RL Model Health Data Simulation
const useRLModelHealth = () => {
  const [modelHealth, setModelHealth] = useState({
    production: {
      model: 'PPO-LSTM',
      version: 'v2.1.5',
      tradesPerEpisode: 6,
      fullEpisodes: 45,
      shortEpisodes: 23, // t≤35 window
      actionDistribution: {
        sell: 18.5,
        hold: 63.2,
        buy: 18.3
      },
      policyEntropy: 0.34,
      klDivergence: 0.019,
      policyCollapse: false,
      lastUpdate: new Date(Date.now() - 300000) // 5 minutes ago
    },
    ppo: {
      policyLoss: 0.0023,
      valueLoss: 0.045,
      explainedVariance: 0.87,
      clipFraction: 0.12,
      timesteps: 2450000,
      learningRate: 0.0003
    },
    lstm: {
      resetRate: 0.08,
      avgSequenceLength: 42.3,
      truncationRate: 0.15,
      hiddenStateNorm: 1.23,
      cellStateNorm: 0.98
    },
    qrDqn: {
      quantileLoss: 0.0156,
      bufferFillRate: 0.78,
      perAlpha: 0.6,
      perBeta: 0.4,
      explorationRate: 0.05,
      targetUpdate: 1000
    },
    reward: {
      rmse: 0.0008,
      definedRate: 0.60, // 36/60 for t≤35
      costCurriculum: 0.75, // 75% progress
      rewardRange: [-0.045, 0.038],
      meanReward: 0.0025
    },
    performance: {
      cpu: 45.2,
      memory: 68.7,
      gpu: 82.1,
      inference: 17.3,
      training: false
    }
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setModelHealth(prev => ({
        ...prev,
        production: {
          ...prev.production,
          tradesPerEpisode: Math.max(1, Math.min(12, prev.production.tradesPerEpisode + (Math.random() - 0.5) * 0.5)),
          policyEntropy: Math.max(0.1, Math.min(0.8, prev.production.policyEntropy + (Math.random() - 0.5) * 0.02)),
          klDivergence: Math.max(0.005, Math.min(0.05, prev.production.klDivergence + (Math.random() - 0.5) * 0.001)),
          lastUpdate: new Date()
        },
        ppo: {
          ...prev.ppo,
          policyLoss: Math.max(0.001, prev.ppo.policyLoss + (Math.random() - 0.5) * 0.0002),
          valueLoss: Math.max(0.01, prev.ppo.valueLoss + (Math.random() - 0.5) * 0.005),
          explainedVariance: Math.max(0.7, Math.min(0.95, prev.ppo.explainedVariance + (Math.random() - 0.5) * 0.01))
        },
        performance: {
          ...prev.performance,
          cpu: Math.max(20, Math.min(90, prev.performance.cpu + (Math.random() - 0.5) * 5)),
          memory: Math.max(40, Math.min(95, prev.performance.memory + (Math.random() - 0.5) * 3)),
          gpu: Math.max(60, Math.min(98, prev.performance.gpu + (Math.random() - 0.5) * 4)),
          inference: Math.max(10, Math.min(25, prev.performance.inference + (Math.random() - 0.5) * 1))
        }
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return modelHealth;
};

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  status: 'optimal' | 'warning' | 'critical';
  icon: React.ComponentType<any>;
  target?: string;
  format?: 'number' | 'percentage' | 'ratio' | 'time';
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, value, subtitle, status, icon: Icon, target, format = 'number' 
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'percentage':
        return `${val.toFixed(2)}%`;
      case 'ratio':
        return val.toFixed(4);
      case 'time':
        return `${val.toFixed(1)}ms`;
      default:
        return val.toLocaleString();
    }
  };

  const statusColors = {
    optimal: 'border-market-up text-market-up shadow-market-up',
    warning: 'border-fintech-purple-400 text-fintech-purple-400 shadow-glow-purple',
    critical: 'border-market-down text-market-down shadow-market-down'
  };

  const bgColors = {
    optimal: 'from-market-up/10 to-market-up/5',
    warning: 'from-fintech-purple-400/10 to-fintech-purple-400/5',
    critical: 'from-market-down/10 to-market-down/5'
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-surface p-4 rounded-xl border ${statusColors[status]} bg-gradient-to-br ${bgColors[status]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5" />
          <span className="text-sm font-medium text-white">{title}</span>
        </div>
        <div className={`w-2 h-2 rounded-full ${
          status === 'optimal' ? 'bg-market-up animate-pulse' :
          status === 'warning' ? 'bg-fintech-purple-400 animate-pulse' :
          'bg-market-down animate-pulse'
        }`} />
      </div>
      
      <div className="space-y-1">
        <div className="text-2xl font-bold text-white">
          {formatValue(value)}
        </div>
        <div className="text-xs text-fintech-dark-300">{subtitle}</div>
        {target && (
          <div className="text-xs text-fintech-dark-400">
            Target: <span className="font-medium">{target}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const ActionHeatmap: React.FC<{ 
  data: { sell: number; hold: number; buy: number };
  episodeCount: number;
}> = ({ data, episodeCount }) => {
  const maxAction = Math.max(data.sell, data.hold, data.buy);
  
  return (
    <div className="glass-surface p-4 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Target className="w-5 h-5 text-fintech-cyan-500" />
        Action Heatmap (60×3 Matrix)
      </h3>
      
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-sm text-fintech-dark-300 mb-2">SELL</div>
            <div 
              className="h-20 bg-gradient-to-t from-market-down/20 to-market-down rounded-lg flex items-end justify-center p-2"
              style={{ 
                background: `linear-gradient(to top, rgba(220, 38, 38, ${data.sell / maxAction * 0.5}) 0%, rgba(220, 38, 38, ${data.sell / maxAction * 0.2}) 100%)`
              }}
            >
              <span className="text-white font-bold">{data.sell.toFixed(1)}%</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-sm text-fintech-dark-300 mb-2">HOLD</div>
            <div 
              className="h-20 bg-gradient-to-t from-fintech-dark-400/20 to-fintech-dark-400 rounded-lg flex items-end justify-center p-2"
              style={{ 
                background: `linear-gradient(to top, rgba(100, 116, 139, ${data.hold / maxAction * 0.5}) 0%, rgba(100, 116, 139, ${data.hold / maxAction * 0.2}) 100%)`
              }}
            >
              <span className="text-white font-bold">{data.hold.toFixed(1)}%</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-sm text-fintech-dark-300 mb-2">BUY</div>
            <div 
              className="h-20 bg-gradient-to-t from-market-up/20 to-market-up rounded-lg flex items-end justify-center p-2"
              style={{ 
                background: `linear-gradient(to top, rgba(13, 158, 117, ${data.buy / maxAction * 0.5}) 0%, rgba(13, 158, 117, ${data.buy / maxAction * 0.2}) 100%)`
              }}
            >
              <span className="text-white font-bold">{data.buy.toFixed(1)}%</span>
            </div>
          </div>
        </div>
        
        <div className="text-sm text-fintech-dark-400 text-center">
          Episodes: {episodeCount} • Balance Target: ≥5% each action
        </div>
      </div>
    </div>
  );
};

const ModelArchitecture: React.FC<{ model: string; version: string }> = ({ model, version }) => {
  const architectures = {
    'PPO-LSTM': {
      layers: ['Input(17)', 'LSTM(64)', 'Dense(32)', 'Policy(3)', 'Value(1)'],
      parameters: '45.2K',
      memory: '2.1MB'
    },
    'QR-DQN': {
      layers: ['Input(17)', 'Dense(128)', 'Dense(64)', 'Quantile(3×51)'],
      parameters: '78.5K', 
      memory: '3.8MB'
    }
  };

  const arch = architectures[model as keyof typeof architectures] || architectures['PPO-LSTM'];

  return (
    <div className="glass-surface p-4 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5 text-fintech-purple-400" />
        Model Architecture - {model} {version}
      </h3>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-fintech-dark-300">Parameters:</span>
          <span className="text-white font-medium">{arch.parameters}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-fintech-dark-300">Memory:</span>
          <span className="text-white font-medium">{arch.memory}</span>
        </div>
        
        <div className="mt-4">
          <div className="text-sm text-fintech-dark-300 mb-2">Network Layers:</div>
          <div className="space-y-2">
            {arch.layers.map((layer, index) => (
              <div 
                key={index}
                className="flex items-center gap-3"
              >
                <div className="w-2 h-2 bg-fintech-cyan-400 rounded-full" />
                <span className="text-sm text-white font-mono">{layer}</span>
                {index < arch.layers.length - 1 && (
                  <div className="flex-1 border-b border-fintech-dark-700 border-dashed" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function RLModelHealth() {
  const modelHealth = useRLModelHealth();
  const [selectedModel, setSelectedModel] = useState<'ppo' | 'qrdqn'>('ppo');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const getStatus = (metric: string, value: number): 'optimal' | 'warning' | 'critical' => {
    const thresholds = {
      tradesPerEpisode: { optimal: [2, 10], warning: [1, 12] },
      policyEntropy: { optimal: [0.2, 0.4], warning: [0.15, 0.5] },
      klDivergence: { optimal: [0.015, 0.025], warning: [0.01, 0.03] },
      rewardRmse: { optimal: [0, 0.001], warning: [0, 0.002] },
      explainedVariance: { optimal: [0.8, 1.0], warning: [0.7, 0.8] }
    };

    const threshold = thresholds[metric as keyof typeof thresholds];
    if (!threshold) return 'optimal';

    if (value >= threshold.optimal[0] && value <= threshold.optimal[1]) return 'optimal';
    if (value >= threshold.warning[0] && value <= threshold.warning[1]) return 'warning';
    return 'critical';
  };

  const productionMetrics = [
    {
      title: 'Trades/Episode',
      value: modelHealth.production.tradesPerEpisode,
      subtitle: 'Full & t≤35 episodes',
      status: getStatus('tradesPerEpisode', modelHealth.production.tradesPerEpisode),
      icon: Activity,
      target: '2-10',
      format: 'number' as const
    },
    {
      title: 'Policy Entropy',
      value: modelHealth.production.policyEntropy,
      subtitle: 'Exploration measure',
      status: getStatus('policyEntropy', modelHealth.production.policyEntropy),
      icon: Brain,
      target: '0.2-0.4',
      format: 'ratio' as const
    },
    {
      title: 'KL Divergence',
      value: modelHealth.production.klDivergence,
      subtitle: 'Policy stability',
      status: getStatus('klDivergence', modelHealth.production.klDivergence),
      icon: Target,
      target: '0.02±0.005',
      format: 'ratio' as const
    },
    {
      title: 'Reward RMSE',
      value: modelHealth.reward.rmse,
      subtitle: 'vs L4 validation',
      status: getStatus('rewardRmse', modelHealth.reward.rmse),
      icon: CheckCircle,
      target: '<1e-3',
      format: 'ratio' as const
    }
  ];

  const performanceMetrics = [
    {
      title: 'CPU Usage',
      value: modelHealth.performance.cpu,
      subtitle: 'Training workload',
      status: modelHealth.performance.cpu < 70 ? 'optimal' as const : modelHealth.performance.cpu < 85 ? 'warning' as const : 'critical' as const,
      icon: Cpu,
      format: 'percentage' as const
    },
    {
      title: 'Memory',
      value: modelHealth.performance.memory,
      subtitle: 'RAM utilization',
      status: modelHealth.performance.memory < 80 ? 'optimal' as const : modelHealth.performance.memory < 90 ? 'warning' as const : 'critical' as const,
      icon: MemoryStick,
      format: 'percentage' as const
    },
    {
      title: 'GPU Usage',
      value: modelHealth.performance.gpu,
      subtitle: 'Training acceleration',
      status: modelHealth.performance.gpu < 90 ? 'optimal' as const : modelHealth.performance.gpu < 95 ? 'warning' as const : 'critical' as const,
      icon: Gauge,
      format: 'percentage' as const
    },
    {
      title: 'Inference',
      value: modelHealth.performance.inference,
      subtitle: 'ONNX latency',
      status: modelHealth.performance.inference < 20 ? 'optimal' as const : modelHealth.performance.inference < 30 ? 'warning' as const : 'critical' as const,
      icon: Timer,
      format: 'time' as const
    }
  ];

  return (
    <div className="min-h-screen bg-fintech-dark-950 p-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">RL Model Health</h1>
            <p className="text-fintech-dark-300">
              Policy Performance • Training Metrics • System Health
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Model Selector */}
            <div className="glass-surface px-4 py-2 rounded-xl">
              <div className="flex items-center gap-3">
                <span className="text-sm text-fintech-dark-300">Active Model:</span>
                <span className="text-fintech-cyan-400 font-bold">
                  {modelHealth.production.model} {modelHealth.production.version}
                </span>
                <div className={`w-2 h-2 rounded-full ${
                  modelHealth.production.policyCollapse ? 'bg-market-down' : 'bg-market-up'
                } animate-pulse`} />
              </div>
            </div>

            {/* Controls */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`p-2 rounded-lg transition-all ${
                showAdvanced 
                  ? 'bg-fintech-cyan-500/20 text-fintech-cyan-400 border border-fintech-cyan-500/30'
                  : 'bg-fintech-dark-800/50 text-fintech-dark-400 hover:text-white'
              }`}
            >
              {showAdvanced ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Policy Production Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-8"
      >
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-3">
          <Brain className="w-6 h-6 text-fintech-cyan-500" />
          Policy in Production
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {productionMetrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + index * 0.05 }}
            >
              <MetricCard {...metric} />
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Action Distribution & Architecture */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
      >
        <ActionHeatmap 
          data={modelHealth.production.actionDistribution}
          episodeCount={modelHealth.production.fullEpisodes + modelHealth.production.shortEpisodes}
        />
        
        <ModelArchitecture 
          model={modelHealth.production.model}
          version={modelHealth.production.version}
        />
      </motion.div>

      {/* PPO/QR-DQN Specific Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-3">
            <GitBranch className="w-6 h-6 text-fintech-purple-400" />
            Training Metrics
          </h2>
          
          <div className="flex items-center gap-2 bg-fintech-dark-800 rounded-lg p-1">
            <button
              onClick={() => setSelectedModel('ppo')}
              className={`px-3 py-1 rounded text-sm font-medium transition-all ${
                selectedModel === 'ppo'
                  ? 'bg-fintech-purple-400 text-white'
                  : 'text-fintech-dark-400 hover:text-white'
              }`}
            >
              PPO-LSTM
            </button>
            <button
              onClick={() => setSelectedModel('qrdqn')}
              className={`px-3 py-1 rounded text-sm font-medium transition-all ${
                selectedModel === 'qrdqn'
                  ? 'bg-fintech-purple-400 text-white'
                  : 'text-fintech-dark-400 hover:text-white'
              }`}
            >
              QR-DQN
            </button>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {selectedModel === 'ppo' && (
            <motion.div
              key="ppo"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">Policy Training</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Policy Loss:</span>
                    <span className="text-white font-medium">{modelHealth.ppo.policyLoss.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Value Loss:</span>
                    <span className="text-white font-medium">{modelHealth.ppo.valueLoss.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Explained Var:</span>
                    <span className={`font-medium ${
                      modelHealth.ppo.explainedVariance > 0.8 ? 'text-market-up' : 'text-market-down'
                    }`}>
                      {modelHealth.ppo.explainedVariance.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">LSTM Statistics</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Reset Rate:</span>
                    <span className="text-white font-medium">{modelHealth.lstm.resetRate.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Avg Sequence:</span>
                    <span className="text-white font-medium">{modelHealth.lstm.avgSequenceLength.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Truncations:</span>
                    <span className="text-white font-medium">{modelHealth.lstm.truncationRate.toFixed(2)}%</span>
                  </div>
                </div>
              </div>

              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">Training Progress</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Timesteps:</span>
                    <span className="text-white font-medium">{(modelHealth.ppo.timesteps / 1000000).toFixed(2)}M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Learning Rate:</span>
                    <span className="text-white font-medium">{modelHealth.ppo.learningRate.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Clip Fraction:</span>
                    <span className="text-white font-medium">{modelHealth.ppo.clipFraction.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {selectedModel === 'qrdqn' && (
            <motion.div
              key="qrdqn"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">Quantile Learning</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Quantile Loss:</span>
                    <span className="text-white font-medium">{modelHealth.qrDqn.quantileLoss.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Target Update:</span>
                    <span className="text-white font-medium">{modelHealth.qrDqn.targetUpdate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Exploration:</span>
                    <span className="text-white font-medium">{(modelHealth.qrDqn.explorationRate * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">Experience Replay</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Buffer Fill:</span>
                    <span className={`font-medium ${
                      modelHealth.qrDqn.bufferFillRate > 0.7 ? 'text-market-up' : 'text-market-down'
                    }`}>
                      {(modelHealth.qrDqn.bufferFillRate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">PER α:</span>
                    <span className="text-white font-medium">{modelHealth.qrDqn.perAlpha.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">PER β:</span>
                    <span className="text-white font-medium">{modelHealth.qrDqn.perBeta.toFixed(1)}</span>
                  </div>
                </div>
              </div>

              <div className="glass-surface p-4 rounded-xl">
                <h3 className="text-lg font-semibold text-white mb-3">Network Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Q-Network:</span>
                    <span className="text-market-up font-medium">Online</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Target Net:</span>
                    <span className="text-market-up font-medium">Synced</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-fintech-dark-300">Quantiles:</span>
                    <span className="text-white font-medium">51</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Reward Consistency & Performance */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
      >
        {/* Reward Consistency */}
        <div className="glass-surface p-6 rounded-xl">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-fintech-cyan-500" />
            Reward Consistency (Window [12,24])
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">RMSE vs L4:</span>
              <span className={`font-bold ${
                modelHealth.reward.rmse < 0.001 ? 'text-market-up' : 'text-market-down'
              }`}>
                {modelHealth.reward.rmse.toFixed(4)}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">% Rows Defined (t≤35):</span>
              <span className="text-white font-bold">
                {(modelHealth.reward.definedRate * 100).toFixed(1)}% (36/60)
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-fintech-dark-300">Cost Curriculum:</span>
              <span className="text-fintech-cyan-400 font-bold">
                {(modelHealth.reward.costCurriculum * 100).toFixed(0)}% → 100%
              </span>
            </div>
            
            <div className="mt-4">
              <div className="text-sm text-fintech-dark-300 mb-2">Reward Range:</div>
              <div className="flex items-center gap-2">
                <span className="text-market-down font-mono text-sm">
                  {modelHealth.reward.rewardRange[0].toFixed(3)}
                </span>
                <div className="flex-1 h-2 bg-fintech-dark-800 rounded-full relative">
                  <div 
                    className="absolute top-0 left-1/2 w-1 h-2 bg-white rounded-full transform -translate-x-1/2"
                    style={{ 
                      left: `${((modelHealth.reward.meanReward - modelHealth.reward.rewardRange[0]) / 
                        (modelHealth.reward.rewardRange[1] - modelHealth.reward.rewardRange[0])) * 100}%` 
                    }}
                  />
                </div>
                <span className="text-market-up font-mono text-sm">
                  {modelHealth.reward.rewardRange[1].toFixed(3)}
                </span>
              </div>
              <div className="text-center text-xs text-fintech-dark-400 mt-1">
                Mean: {modelHealth.reward.meanReward.toFixed(4)}
              </div>
            </div>
          </div>
        </div>

        {/* System Performance */}
        <div className="space-y-4">
          {performanceMetrics.map((metric, index) => (
            <motion.div
              key={metric.title}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
            >
              <MetricCard {...metric} />
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Policy Collapse Detection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className={`glass-surface p-6 rounded-xl border-2 ${
          modelHealth.production.policyCollapse 
            ? 'border-market-down shadow-market-down' 
            : 'border-market-up shadow-market-up'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              modelHealth.production.policyCollapse 
                ? 'bg-market-down/20 text-market-down' 
                : 'bg-market-up/20 text-market-up'
            }`}>
              {modelHealth.production.policyCollapse ? (
                <AlertTriangle className="w-6 h-6" />
              ) : (
                <CheckCircle className="w-6 h-6" />
              )}
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">
                {modelHealth.production.policyCollapse ? 'Policy Collapse Detected' : 'Policy Health: Optimal'}
              </h3>
              <p className="text-fintech-dark-300">
                {modelHealth.production.policyCollapse 
                  ? 'Single action >95% - Requires immediate retraining'
                  : 'Action diversity maintained - All systems nominal'
                }
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-fintech-dark-400">Last Check:</div>
            <div className="text-white font-medium">
              {modelHealth.production.lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}