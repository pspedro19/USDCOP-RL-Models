'use client';

/**
 * Floating Approval Panel
 * =======================
 * Sticky panel that follows scroll for model approval workflow.
 * Part of the Two-Vote Promotion System (L4 first vote, Human second vote).
 *
 * MLOps Best Practices:
 * - Clear model identification
 * - Metrics summary before approval
 * - Audit trail context
 * - Confirmation dialogs for critical actions
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronUp,
  ChevronDown,
  Loader2,
  Shield,
  TrendingUp,
  Target,
  Activity,
  Clock,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ApprovalMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
}

export interface FloatingApprovalPanelProps {
  /** Model ID being evaluated */
  modelId: string;
  /** Experiment name */
  experimentName?: string;
  /** L4 recommendation */
  recommendation: 'PROMOTE' | 'REJECT' | 'REVIEW';
  /** L4 confidence score */
  confidence: number;
  /** Current backtest metrics */
  metrics?: ApprovalMetrics;
  /** Whether backtest is still running */
  isBacktestRunning: boolean;
  /** Progress percentage (0-100) */
  backestProgress: number;
  /** Callback when approved */
  onApprove: (notes: string) => Promise<void>;
  /** Callback when rejected */
  onReject: (reason: string) => Promise<void>;
  /** Whether to show the panel */
  visible: boolean;
  /** Position: 'top' or 'bottom' */
  position?: 'top' | 'bottom';
  /** Whether the model is from an experiment proposal */
  isExperimentProposal?: boolean;
  /** Proposal ID if from experiment */
  proposalId?: string;
}

export function FloatingApprovalPanel({
  modelId,
  experimentName,
  recommendation,
  confidence,
  metrics,
  isBacktestRunning,
  backestProgress,
  onApprove,
  onReject,
  visible,
  position = 'bottom',
  isExperimentProposal = false,
  proposalId,
}: FloatingApprovalPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isApproving, setIsApproving] = useState(false);
  const [isRejecting, setIsRejecting] = useState(false);
  const [notes, setNotes] = useState('');
  const [showConfirmDialog, setShowConfirmDialog] = useState<'approve' | 'reject' | null>(null);

  if (!visible) return null;

  const handleApprove = async () => {
    setIsApproving(true);
    try {
      await onApprove(notes);
      setNotes('');
      setShowConfirmDialog(null);
    } finally {
      setIsApproving(false);
    }
  };

  const handleReject = async () => {
    setIsRejecting(true);
    try {
      await onReject(notes);
      setNotes('');
      setShowConfirmDialog(null);
    } finally {
      setIsRejecting(false);
    }
  };

  const recommendationColors = {
    PROMOTE: 'bg-green-500/20 text-green-400 border-green-500/50',
    REJECT: 'bg-red-500/20 text-red-400 border-red-500/50',
    REVIEW: 'bg-amber-500/20 text-amber-400 border-amber-500/50',
  };

  const positionClasses = position === 'top'
    ? 'top-20 left-0 right-0'
    : 'bottom-0 left-0 right-0';

  return (
    <>
      {/* Floating Panel */}
      <motion.div
        initial={{ y: position === 'top' ? -100 : 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: position === 'top' ? -100 : 100, opacity: 0 }}
        className={cn(
          'fixed z-50',
          positionClasses,
          'px-4 pb-4 pt-2'
        )}
      >
        <div className="max-w-6xl mx-auto">
          <div className={cn(
            'bg-gray-900/95 backdrop-blur-xl border rounded-2xl shadow-2xl overflow-hidden',
            isBacktestRunning ? 'border-cyan-500/50' : 'border-gray-700'
          )}>
            {/* Progress Bar (when running) */}
            {isBacktestRunning && (
              <div className="h-1 bg-gray-800">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${backestProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            )}

            {/* Header - Always Visible */}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="w-full px-6 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
            >
              <div className="flex items-center gap-4">
                {/* Mode Badge */}
                <div className={cn(
                  'px-3 py-1.5 rounded-lg border text-sm font-bold',
                  isBacktestRunning
                    ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50'
                    : 'bg-purple-500/20 text-purple-400 border-purple-500/50'
                )}>
                  {isBacktestRunning ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      BACKTEST {backestProgress}%
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Shield className="w-4 h-4" />
                      REVISION
                    </span>
                  )}
                </div>

                {/* Model Info */}
                <div className="text-left">
                  <p className="text-white font-semibold">{experimentName || modelId}</p>
                  <p className="text-gray-500 text-xs font-mono">{modelId.slice(0, 24)}...</p>
                </div>

                {/* L4 Recommendation */}
                <div className={cn(
                  'px-3 py-1 rounded-lg border text-sm font-bold',
                  recommendationColors[recommendation]
                )}>
                  L4: {recommendation}
                  <span className="ml-2 opacity-70">{(confidence * 100).toFixed(0)}%</span>
                </div>
              </div>

              <div className="flex items-center gap-4">
                {/* Quick Metrics */}
                {metrics && !isBacktestRunning && (
                  <div className="hidden md:flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      <TrendingUp className="w-4 h-4 text-gray-500" />
                      <span className={metrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
                        {(metrics.totalReturn * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Target className="w-4 h-4 text-gray-500" />
                      <span className="text-cyan-400">{metrics.sharpeRatio.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Activity className="w-4 h-4 text-gray-500" />
                      <span className="text-amber-400">{(metrics.winRate * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )}

                {/* Expand Toggle */}
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
                  className="border-t border-gray-800"
                >
                  <div className="px-6 py-4">
                    {/* Metrics Grid */}
                    {metrics && (
                      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-4">
                        {[
                          { label: 'Return', value: `${(metrics.totalReturn * 100).toFixed(2)}%`, positive: metrics.totalReturn >= 0 },
                          { label: 'Sharpe', value: metrics.sharpeRatio.toFixed(2), positive: metrics.sharpeRatio >= 1 },
                          { label: 'Max DD', value: `${(metrics.maxDrawdown * 100).toFixed(2)}%`, positive: false },
                          { label: 'Win Rate', value: `${(metrics.winRate * 100).toFixed(1)}%`, positive: metrics.winRate >= 0.5 },
                          { label: 'Trades', value: String(metrics.totalTrades), positive: true },
                          { label: 'PF', value: metrics.profitFactor.toFixed(2), positive: metrics.profitFactor >= 1 },
                        ].map((m, i) => (
                          <div key={i} className="bg-gray-800/50 rounded-lg p-2 text-center">
                            <p className="text-gray-500 text-xs">{m.label}</p>
                            <p className={cn(
                              'text-lg font-bold',
                              m.positive ? 'text-green-400' : 'text-red-400'
                            )}>
                              {m.value}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Notes Input */}
                    <div className="mb-4">
                      <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        placeholder="Notas de revision (opcional)..."
                        className="w-full bg-gray-800/50 border border-gray-700 rounded-lg p-3 text-white text-sm resize-none focus:border-purple-500 focus:outline-none"
                        rows={2}
                        disabled={isBacktestRunning}
                      />
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-4">
                      <button
                        onClick={() => setShowConfirmDialog('approve')}
                        disabled={isBacktestRunning || isApproving || isRejecting}
                        className={cn(
                          'flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all',
                          'bg-green-500/20 border border-green-500/50 text-green-400',
                          'hover:bg-green-500/30 disabled:opacity-50 disabled:cursor-not-allowed'
                        )}
                      >
                        <CheckCircle2 className="w-5 h-5" />
                        Aprobar y Promover a Produccion
                      </button>
                      <button
                        onClick={() => setShowConfirmDialog('reject')}
                        disabled={isBacktestRunning || isApproving || isRejecting}
                        className={cn(
                          'flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all',
                          'bg-red-500/20 border border-red-500/50 text-red-400',
                          'hover:bg-red-500/30 disabled:opacity-50 disabled:cursor-not-allowed'
                        )}
                      >
                        <XCircle className="w-5 h-5" />
                        Rechazar
                      </button>
                    </div>

                    {/* Warning for running backtest */}
                    {isBacktestRunning && (
                      <div className="mt-4 flex items-center gap-2 text-amber-400 text-sm">
                        <Clock className="w-4 h-4" />
                        Espera a que termine el backtest para tomar una decision
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>

      {/* Confirmation Dialog */}
      <AnimatePresence>
        {showConfirmDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setShowConfirmDialog(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {showConfirmDialog === 'approve' ? (
                <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-green-500/20 rounded-xl">
                      <Zap className="w-6 h-6 text-green-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">Confirmar Promocion</h3>
                      <p className="text-gray-400 text-sm">Segundo voto - Promocion a Produccion</p>
                    </div>
                  </div>

                  <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 mb-6">
                    <p className="text-green-400 text-sm">
                      Al aprobar, el modelo <strong>{experimentName || modelId.slice(0, 16)}</strong> sera
                      promovido a produccion. El modelo actual sera archivado automaticamente.
                    </p>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => setShowConfirmDialog(null)}
                      className="flex-1 px-4 py-3 bg-gray-800 text-gray-300 rounded-xl hover:bg-gray-700 transition-colors"
                    >
                      Cancelar
                    </button>
                    <button
                      onClick={handleApprove}
                      disabled={isApproving}
                      className="flex-1 px-4 py-3 bg-green-500 text-white rounded-xl hover:bg-green-600 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {isApproving ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <CheckCircle2 className="w-5 h-5" />
                      )}
                      Confirmar
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-red-500/20 rounded-xl">
                      <AlertTriangle className="w-6 h-6 text-red-400" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">Confirmar Rechazo</h3>
                      <p className="text-gray-400 text-sm">El modelo no sera promovido</p>
                    </div>
                  </div>

                  <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6">
                    <p className="text-red-400 text-sm">
                      Al rechazar, el modelo <strong>{experimentName || modelId.slice(0, 16)}</strong> sera
                      marcado como rechazado y no sera promovido a produccion.
                    </p>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => setShowConfirmDialog(null)}
                      className="flex-1 px-4 py-3 bg-gray-800 text-gray-300 rounded-xl hover:bg-gray-700 transition-colors"
                    >
                      Cancelar
                    </button>
                    <button
                      onClick={handleReject}
                      disabled={isRejecting}
                      className="flex-1 px-4 py-3 bg-red-500 text-white rounded-xl hover:bg-red-600 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {isRejecting ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <XCircle className="w-5 h-5" />
                      )}
                      Rechazar
                    </button>
                  </div>
                </>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default FloatingApprovalPanel;
