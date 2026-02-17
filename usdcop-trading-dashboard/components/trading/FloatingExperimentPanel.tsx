'use client';

/**
 * FloatingExperimentPanel
 * =======================
 * Floating panel for experiment approval in Dashboard.
 * Appears when selected model has a PENDING_APPROVAL proposal.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  Activity,
  AlertCircle,
  Loader2,
  Check,
  X,
  Shield,
} from 'lucide-react';
import {
  Experiment,
  PromotionRecommendation,
  formatExpiryTime,
} from '@/lib/contracts/experiments.contract';

interface FloatingExperimentPanelProps {
  experiment: Experiment | null;
  modelId: string;
  onApprove: (notes: string) => Promise<void>;
  onReject: (reason: string) => Promise<void>;
  isVisible: boolean;
  /** Override metrics with real backtest results when available */
  backtestSummary?: {
    sharpe_ratio?: number;
    max_drawdown_pct?: number;
    win_rate?: number;
    total_trades?: number;
    total_return_pct?: number;
  };
}

export function FloatingExperimentPanel({
  experiment,
  modelId,
  onApprove,
  onReject,
  isVisible,
  backtestSummary,
}: FloatingExperimentPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [notes, setNotes] = useState('');
  const [isApproving, setIsApproving] = useState(false);
  const [isRejecting, setIsRejecting] = useState(false);

  if (!isVisible || !experiment) return null;

  const handleApprove = async () => {
    if (!confirm('Aprobar este experimento promovera el modelo a produccion. Continuar?')) {
      return;
    }
    setIsApproving(true);
    try {
      await onApprove(notes);
      setNotes('');
    } finally {
      setIsApproving(false);
    }
  };

  const handleReject = async () => {
    if (!confirm('Rechazar este experimento? El modelo no sera promovido.')) {
      return;
    }
    setIsRejecting(true);
    try {
      await onReject(notes);
      setNotes('');
    } finally {
      setIsRejecting(false);
    }
  };

  // Border color based on recommendation
  const getBorderGradient = (rec: PromotionRecommendation) => {
    switch (rec) {
      case 'PROMOTE':
        return 'from-green-500 via-green-400 to-green-500';
      case 'REJECT':
        return 'from-red-500 via-red-400 to-red-500';
      case 'REVIEW':
        return 'from-amber-500 via-amber-400 to-amber-500';
    }
  };

  const getRecommendationBadge = (rec: PromotionRecommendation) => {
    switch (rec) {
      case 'PROMOTE':
        return { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/50', icon: CheckCircle2 };
      case 'REJECT':
        return { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', icon: XCircle };
      case 'REVIEW':
        return { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/50', icon: AlertCircle };
    }
  };

  const recBadge = getRecommendationBadge(experiment.recommendation);
  const RecIcon = recBadge.icon;

  // Count passed gates
  const passedGates = experiment.criteriaResults?.filter(c => c.passed).length || 0;
  const totalGates = experiment.criteriaResults?.length || 0;

  // Metrics - Use backtest results when available, fallback to original proposal metrics
  // This ensures both FloatingExperimentPanel and BacktestMetricsPanel show the SAME values
  const metrics = backtestSummary ? {
    totalReturn: (backtestSummary.total_return_pct || 0) / 100,
    sharpeRatio: backtestSummary.sharpe_ratio || 0,
    maxDrawdown: (backtestSummary.max_drawdown_pct || 0) / 100,
    winRate: (backtestSummary.win_rate || 0) / 100,
    totalTrades: backtestSummary.total_trades || 0,
    profitFactor: experiment.metrics?.profitFactor || 0,
    avgTradeReturn: experiment.metrics?.avgTradeReturn || 0,
  } : experiment.metrics;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 100, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="fixed bottom-0 left-0 right-0 z-50"
      >
        {/* Gradient border at top */}
        <div className={`h-1 bg-gradient-to-r ${getBorderGradient(experiment.recommendation)}`} />

        <div className="bg-gray-900/95 backdrop-blur-xl border-t border-gray-800">
          {/* Header - Always visible */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="relative">
                <Bell className="w-5 h-5 text-amber-400" />
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
              </div>
              <div className="text-left">
                <span className="text-gray-400 text-sm">PROPUESTA PENDIENTE:</span>
                <span className="ml-2 text-white font-mono text-sm">{experiment.experimentName || experiment.modelId}</span>
              </div>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${recBadge.bg} ${recBadge.text} border ${recBadge.border}`}>
                {experiment.recommendation}
              </span>
            </div>
            <div className="flex items-center gap-3">
              {experiment.hoursUntilExpiry !== null && (
                <span className="text-xs text-gray-500">
                  {formatExpiryTime(experiment.hoursUntilExpiry)}
                </span>
              )}
              {isExpanded ? (
                <ChevronDown className="w-5 h-5 text-gray-400" />
              ) : (
                <ChevronUp className="w-5 h-5 text-gray-400" />
              )}
            </div>
          </button>

          {/* Expanded Content */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 space-y-4">
                  {/* Metrics Grid */}
                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                    <MetricCard
                      label="Return"
                      value={`${((metrics?.totalReturn || 0) * 100).toFixed(2)}%`}
                      icon={TrendingUp}
                      positive={(metrics?.totalReturn || 0) >= 0}
                    />
                    <MetricCard
                      label="Sharpe"
                      value={(metrics?.sharpeRatio || 0).toFixed(2)}
                      icon={BarChart3}
                      positive={(metrics?.sharpeRatio || 0) >= 1}
                    />
                    <MetricCard
                      label="Win Rate"
                      value={`${((metrics?.winRate || 0) * 100).toFixed(1)}%`}
                      icon={Target}
                      positive={(metrics?.winRate || 0) >= 0.5}
                    />
                    <MetricCard
                      label="Max DD"
                      value={`${((metrics?.maxDrawdown || 0) * 100).toFixed(2)}%`}
                      icon={TrendingDown}
                      positive={false}
                    />
                    <MetricCard
                      label="Trades"
                      value={metrics?.totalTrades?.toString() || '0'}
                      icon={Activity}
                    />
                    <MetricCard
                      label="Confidence"
                      value={`${(experiment.confidence * 100).toFixed(0)}%`}
                      icon={Shield}
                      positive={experiment.confidence >= 0.7}
                    />
                  </div>

                  {/* Gates Status */}
                  <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
                    <span className="text-gray-400 text-sm">Gates:</span>
                    <div className="flex items-center gap-1">
                      {experiment.criteriaResults?.map((criteria, i) => (
                        <span
                          key={i}
                          className={`w-5 h-5 rounded flex items-center justify-center ${
                            criteria.passed
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}
                          title={`${criteria.criterion}: ${criteria.passed ? 'PASSED' : 'FAILED'}`}
                        >
                          {criteria.passed ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                        </span>
                      ))}
                    </div>
                    <span className={`text-sm font-medium ${
                      passedGates === totalGates ? 'text-green-400' : 'text-amber-400'
                    }`}>
                      ({passedGates}/{totalGates} PASSED)
                    </span>
                  </div>

                  {/* Reason */}
                  {experiment.reason && (
                    <p className="text-sm text-gray-400 italic">
                      "{experiment.reason}"
                    </p>
                  )}

                  {/* Notes Input */}
                  <div>
                    <input
                      type="text"
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Notas opcionales..."
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm placeholder:text-gray-500 focus:outline-none focus:border-gray-600"
                    />
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleReject}
                      disabled={isRejecting || isApproving}
                      className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isRejecting ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <XCircle className="w-4 h-4" />
                      )}
                      <span className="font-medium">RECHAZAR</span>
                    </button>
                    <button
                      onClick={handleApprove}
                      disabled={isApproving || isRejecting}
                      className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-green-500/10 border border-green-500/30 text-green-400 rounded-xl hover:bg-green-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isApproving ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4" />
                      )}
                      <span className="font-medium">APROBAR Y PROMOVER</span>
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// Helper component for metric cards
function MetricCard({
  label,
  value,
  icon: Icon,
  positive,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  positive?: boolean;
}) {
  const colorClass = positive === true
    ? 'text-green-400'
    : positive === false
    ? 'text-red-400'
    : 'text-gray-300';

  return (
    <div className="flex flex-col items-center p-2 bg-gray-800/50 rounded-lg">
      <Icon className={`w-4 h-4 mb-1 ${colorClass}`} />
      <span className="text-[10px] text-gray-500 uppercase">{label}</span>
      <span className={`font-mono text-sm font-bold ${colorClass}`}>{value}</span>
    </div>
  );
}

export default FloatingExperimentPanel;
