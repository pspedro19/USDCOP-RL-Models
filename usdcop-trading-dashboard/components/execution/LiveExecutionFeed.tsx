'use client';

/**
 * Live Execution Feed Component
 * =============================
 *
 * Displays real-time execution events from SignalBridge.
 * Uses WebSocket for live updates.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Bell,
  X,
} from 'lucide-react';
import { useSignalBridgeWebSocket } from '@/hooks/execution';
import {
  type BridgeExecutionResult,
  type TradingMode,
  TRADING_MODE_LABELS,
  TRADING_MODE_COLORS,
  INFERENCE_ACTION_LABELS,
} from '@/lib/contracts/execution/signal-bridge.contract';
import { ORDER_STATUS_NAMES } from '@/lib/contracts/execution/execution.contract';

interface LiveExecutionFeedProps {
  /** Maximum number of executions to display */
  maxItems?: number;
  /** Whether to show notifications */
  showNotifications?: boolean;
  /** User ID for filtering */
  userId?: string;
  /** Compact mode for smaller displays */
  compact?: boolean;
}

interface Notification {
  id: string;
  type: 'execution' | 'kill_switch' | 'risk_alert' | 'mode_change';
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
}

export function LiveExecutionFeed({
  maxItems = 10,
  showNotifications = true,
  userId,
  compact = false,
}: LiveExecutionFeedProps) {
  const [executions, setExecutions] = useState<BridgeExecutionResult[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    if (!showNotifications) return;

    const newNotification: Notification = {
      ...notification,
      id: Math.random().toString(36).slice(2),
      timestamp: new Date(),
    };

    setNotifications(prev => [newNotification, ...prev].slice(0, 5));

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== newNotification.id));
    }, 5000);
  }, [showNotifications]);

  const { isConnected, error } = useSignalBridgeWebSocket({
    userId,
    autoConnect: true,
    onExecution: (execution) => {
      setExecutions(prev => [execution, ...prev].slice(0, maxItems));

      // Add notification
      const isSuccess = execution.status === 'filled';
      addNotification({
        type: 'execution',
        message: `${execution.side?.toUpperCase() || 'TRADE'} ${execution.symbol || 'USD/COP'} - ${ORDER_STATUS_NAMES[execution.status] || execution.status}`,
        severity: isSuccess ? 'success' : 'warning',
      });
    },
    onKillSwitch: (active, reason) => {
      addNotification({
        type: 'kill_switch',
        message: active ? `Kill switch activated: ${reason}` : 'Kill switch deactivated',
        severity: active ? 'error' : 'info',
      });
    },
    onTradingModeChange: (mode) => {
      addNotification({
        type: 'mode_change',
        message: `Trading mode changed to ${TRADING_MODE_LABELS[mode]}`,
        severity: 'info',
      });
    },
    onRiskAlert: (alert) => {
      addNotification({
        type: 'risk_alert',
        message: alert.message,
        severity: alert.severity === 'high' ? 'error' : 'warning',
      });
    },
  });

  const dismissNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const getExecutionIcon = (execution: BridgeExecutionResult) => {
    if (execution.status === 'filled') {
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    }
    if (execution.status === 'failed' || execution.status === 'rejected') {
      return <XCircle className="w-4 h-4 text-red-400" />;
    }
    return <Clock className="w-4 h-4 text-yellow-400" />;
  };

  const getSideIcon = (side?: string) => {
    if (side === 'buy') {
      return <TrendingUp className="w-4 h-4 text-green-400" />;
    }
    if (side === 'sell') {
      return <TrendingDown className="w-4 h-4 text-red-400" />;
    }
    return null;
  };

  const getNotificationIcon = (type: Notification['type'], severity: Notification['severity']) => {
    switch (type) {
      case 'kill_switch':
        return <AlertTriangle className={`w-4 h-4 ${severity === 'error' ? 'text-red-400' : 'text-green-400'}`} />;
      case 'risk_alert':
        return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      case 'mode_change':
        return <Zap className="w-4 h-4 text-cyan-400" />;
      default:
        return <Bell className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className={`w-4 h-4 ${isConnected ? 'text-green-400' : 'text-red-400'}`} />
          <span className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>
        {error && (
          <span className="text-xs text-red-400">{error}</span>
        )}
      </div>

      {/* Notifications */}
      <AnimatePresence>
        {notifications.length > 0 && (
          <div className="space-y-2">
            {notifications.map((notification) => (
              <motion.div
                key={notification.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className={`flex items-center justify-between p-3 rounded-lg border ${
                  notification.severity === 'error'
                    ? 'bg-red-500/10 border-red-500/30'
                    : notification.severity === 'success'
                      ? 'bg-green-500/10 border-green-500/30'
                      : notification.severity === 'warning'
                        ? 'bg-amber-500/10 border-amber-500/30'
                        : 'bg-cyan-500/10 border-cyan-500/30'
                }`}
              >
                <div className="flex items-center gap-3">
                  {getNotificationIcon(notification.type, notification.severity)}
                  <span className="text-sm text-white">{notification.message}</span>
                </div>
                <button
                  onClick={() => dismissNotification(notification.id)}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            ))}
          </div>
        )}
      </AnimatePresence>

      {/* Execution Feed */}
      <div className={`space-y-2 ${compact ? 'max-h-[200px]' : 'max-h-[400px]'} overflow-y-auto`}>
        {executions.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No recent executions</p>
          </div>
        ) : (
          <AnimatePresence mode="popLayout">
            {executions.map((execution, idx) => (
              <motion.div
                key={execution.execution_id || idx}
                layout
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className={`bg-gray-800/50 rounded-lg ${compact ? 'p-2' : 'p-3'}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getExecutionIcon(execution)}
                    <div>
                      <div className="flex items-center gap-2">
                        {getSideIcon(execution.side)}
                        <span className="font-medium text-white">
                          {execution.symbol || 'USD/COP'}
                        </span>
                        <span className={`text-sm ${execution.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                          {execution.side?.toUpperCase() || ''}
                        </span>
                      </div>
                      {!compact && (
                        <div className="text-xs text-gray-400 mt-1">
                          {execution.filled_quantity > 0 && (
                            <span>Qty: {execution.filled_quantity.toFixed(4)}</span>
                          )}
                          {execution.filled_price > 0 && (
                            <span className="ml-2">@ ${execution.filled_price.toFixed(2)}</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      execution.status === 'filled'
                        ? 'bg-green-500/20 text-green-400'
                        : execution.status === 'failed'
                          ? 'bg-red-500/20 text-red-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {ORDER_STATUS_NAMES[execution.status] || execution.status}
                    </span>
                    {!compact && execution.created_at && (
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(execution.created_at).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

export default LiveExecutionFeed;
