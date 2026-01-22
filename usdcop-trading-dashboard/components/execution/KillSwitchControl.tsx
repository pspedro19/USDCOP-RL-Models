'use client';

/**
 * Kill Switch Control Component
 * =============================
 *
 * Emergency stop control for trading operations.
 * Provides visual feedback and confirmation dialogs.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Power, AlertTriangle, Shield, RefreshCw, X } from 'lucide-react';
import { signalBridgeService } from '@/lib/services/execution';
import { type TradingMode, TRADING_MODE_LABELS, TRADING_MODE_COLORS } from '@/lib/contracts/execution/signal-bridge.contract';

interface KillSwitchControlProps {
  /** Compact mode for toolbar integration */
  compact?: boolean;
  /** Callback when kill switch state changes */
  onStateChange?: (active: boolean) => void;
}

export function KillSwitchControl({
  compact = false,
  onStateChange,
}: KillSwitchControlProps) {
  const [isActive, setIsActive] = useState(false);
  const [reason, setReason] = useState<string | null>(null);
  const [tradingMode, setTradingMode] = useState<TradingMode>('PAPER');
  const [isLoading, setIsLoading] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmReason, setConfirmReason] = useState('');

  // Fetch initial state
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await signalBridgeService.getKillSwitchStatus();
        setIsActive(status.active);
        setReason(status.reason);
        setTradingMode(status.trading_mode);
      } catch (err) {
        console.error('Failed to fetch kill switch status:', err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleToggle = () => {
    if (isActive) {
      // Deactivating - show confirmation
      setShowConfirmDialog(true);
    } else {
      // Activating - show reason dialog
      setShowConfirmDialog(true);
    }
  };

  const handleConfirm = async () => {
    setIsLoading(true);
    try {
      if (isActive) {
        await signalBridgeService.deactivateKillSwitch();
        setIsActive(false);
        setReason(null);
      } else {
        await signalBridgeService.activateKillSwitch(confirmReason || 'Manual activation');
        setIsActive(true);
        setReason(confirmReason || 'Manual activation');
      }
      onStateChange?.(!isActive);
    } catch (err) {
      console.error('Failed to toggle kill switch:', err);
      alert('Failed to toggle kill switch');
    } finally {
      setIsLoading(false);
      setShowConfirmDialog(false);
      setConfirmReason('');
    }
  };

  if (compact) {
    return (
      <button
        onClick={handleToggle}
        disabled={isLoading}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
          isActive
            ? 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30'
            : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:text-white hover:border-gray-600'
        }`}
        title={isActive ? 'Kill switch active - click to deactivate' : 'Click to activate kill switch'}
      >
        <Power className={`w-4 h-4 ${isActive ? 'animate-pulse' : ''}`} />
        <span className="text-sm font-medium">
          {isActive ? 'KILL' : 'Safe'}
        </span>
      </button>
    );
  }

  return (
    <>
      <div className={`rounded-xl p-4 border ${
        isActive
          ? 'bg-red-500/10 border-red-500/30'
          : 'bg-gray-900/50 border-gray-800/50'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              isActive ? 'bg-red-500/20' : 'bg-gray-800'
            }`}>
              <Shield className={`w-5 h-5 ${isActive ? 'text-red-400' : 'text-gray-400'}`} />
            </div>
            <div>
              <h3 className="font-bold text-white">Emergency Kill Switch</h3>
              <p className="text-sm text-gray-400">
                {isActive ? 'All trading is halted' : 'Trading is active'}
              </p>
            </div>
          </div>

          {/* Trading Mode Badge */}
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            TRADING_MODE_COLORS[tradingMode] || 'bg-gray-500/20 text-gray-400'
          }`}>
            {TRADING_MODE_LABELS[tradingMode]}
          </div>
        </div>

        {/* Active Reason */}
        {isActive && reason && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Reason:</p>
                <p className="text-sm text-red-300">{reason}</p>
              </div>
            </div>
          </div>
        )}

        {/* Toggle Button */}
        <button
          onClick={handleToggle}
          disabled={isLoading}
          className={`w-full py-3 rounded-lg font-bold transition-all flex items-center justify-center gap-2 ${
            isActive
              ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30 border border-green-500/30'
              : 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30'
          } disabled:opacity-50`}
        >
          {isLoading ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <>
              <Power className="w-5 h-5" />
              {isActive ? 'Resume Trading' : 'Stop All Trading'}
            </>
          )}
        </button>
      </div>

      {/* Confirmation Dialog */}
      <AnimatePresence>
        {showConfirmDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
            onClick={() => setShowConfirmDialog(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className={`w-full max-w-md rounded-2xl p-6 border ${
                isActive
                  ? 'bg-gray-900 border-green-500/30'
                  : 'bg-gray-900 border-red-500/30'
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {isActive ? (
                    <div className="p-2 bg-green-500/20 rounded-lg">
                      <Power className="w-5 h-5 text-green-400" />
                    </div>
                  ) : (
                    <div className="p-2 bg-red-500/20 rounded-lg">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    </div>
                  )}
                  <h2 className="text-lg font-bold text-white">
                    {isActive ? 'Resume Trading?' : 'Activate Kill Switch?'}
                  </h2>
                </div>
                <button
                  onClick={() => setShowConfirmDialog(false)}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Content */}
              <div className="space-y-4">
                {isActive ? (
                  <p className="text-gray-300">
                    Are you sure you want to resume trading? Make sure all systems are
                    functioning correctly before continuing.
                  </p>
                ) : (
                  <>
                    <p className="text-gray-300">
                      This will immediately halt all trading operations across all exchanges.
                      Please provide a reason:
                    </p>
                    <input
                      type="text"
                      value={confirmReason}
                      onChange={(e) => setConfirmReason(e.target.value)}
                      placeholder="Reason for activation..."
                      className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-red-500"
                    />
                  </>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowConfirmDialog(false)}
                  className="flex-1 py-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirm}
                  disabled={isLoading || (!isActive && !confirmReason)}
                  className={`flex-1 py-3 rounded-lg font-bold transition-colors disabled:opacity-50 ${
                    isActive
                      ? 'bg-green-500 text-white hover:bg-green-600'
                      : 'bg-red-500 text-white hover:bg-red-600'
                  }`}
                >
                  {isLoading ? (
                    <RefreshCw className="w-5 h-5 animate-spin mx-auto" />
                  ) : (
                    'Confirm'
                  )}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default KillSwitchControl;
