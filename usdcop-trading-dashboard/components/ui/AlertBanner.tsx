"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  Info,
  X,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Shield,
  Clock,
  Bell,
  AlertCircle,
  Target,
  DollarSign
} from "lucide-react";

export interface AlertData {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info' | 'trading' | 'risk' | 'system';
  title: string;
  message: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  priority?: 'low' | 'normal' | 'high' | 'urgent';
  timestamp?: number;
  duration?: number; // Auto-dismiss duration in ms
  persistent?: boolean;
  actions?: {
    label: string;
    action: () => void;
    variant?: 'primary' | 'secondary' | 'destructive';
  }[];
  data?: Record<string, any>;
}

export interface AlertBannerProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  alerts: AlertData[];
  variant?: 'toast' | 'banner' | 'sidebar' | 'modal' | 'floating';
  position?: 'top' | 'bottom' | 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
  maxVisible?: number;
  showTimestamp?: boolean;
  showPriority?: boolean;
  groupByType?: boolean;
  animated?: boolean;
  sound?: boolean;
  onDismiss?: (alertId: string) => void;
  onDismissAll?: () => void;
  onAction?: (alertId: string, actionIndex: number) => void;
  className?: string;
}

const AlertBanner = React.forwardRef<HTMLDivElement, AlertBannerProps>(
  ({
    alerts,
    variant = 'toast',
    position = 'top-right',
    maxVisible = 5,
    showTimestamp = true,
    showPriority = false,
    groupByType = false,
    animated = true,
    sound = false,
    onDismiss,
    onDismissAll,
    onAction,
    className,
    ...props
  }, ref) => {
    const [dismissedAlerts, setDismissedAlerts] = React.useState<Set<string>>(new Set());

    React.useEffect(() => {
      // Auto-dismiss non-persistent alerts
      alerts.forEach(alert => {
        if (!alert.persistent && alert.duration && alert.duration > 0) {
          const timer = setTimeout(() => {
            handleDismiss(alert.id);
          }, alert.duration);

          return () => clearTimeout(timer);
        }
      });
    }, [alerts]);

    React.useEffect(() => {
      // Play sound for new critical alerts
      if (sound) {
        const criticalAlerts = alerts.filter(
          alert => alert.severity === 'critical' && !dismissedAlerts.has(alert.id)
        );
        if (criticalAlerts.length > 0) {
          // Would play sound here in a real implementation
          console.log('🔊 Critical alert sound');
        }
      }
    }, [alerts, dismissedAlerts, sound]);

    const handleDismiss = (alertId: string) => {
      setDismissedAlerts(prev => new Set([...prev, alertId]));
      onDismiss?.(alertId);
    };

    const handleDismissAll = () => {
      const allIds = alerts.map(alert => alert.id);
      setDismissedAlerts(new Set(allIds));
      onDismissAll?.();
    };

    const formatTimestamp = (timestamp: number) => {
      const now = Date.now();
      const diff = now - timestamp;
      const minutes = Math.floor(diff / (1000 * 60));
      const hours = Math.floor(diff / (1000 * 60 * 60));

      if (minutes < 1) return 'now';
      if (minutes < 60) return `${minutes}m ago`;
      if (hours < 24) return `${hours}h ago`;
      return new Date(timestamp).toLocaleDateString();
    };

    const getAlertIcon = (type: AlertData['type'], severity?: AlertData['severity']) => {
      switch (type) {
        case 'success':
          return <CheckCircle className="w-5 h-5" />;
        case 'warning':
          return <AlertTriangle className="w-5 h-5" />;
        case 'error':
          return <XCircle className="w-5 h-5" />;
        case 'info':
          return <Info className="w-5 h-5" />;
        case 'trading':
          return severity === 'critical' ? <TrendingDown className="w-5 h-5" /> : <TrendingUp className="w-5 h-5" />;
        case 'risk':
          return <Shield className="w-5 h-5" />;
        case 'system':
          return <Activity className="w-5 h-5" />;
        default:
          return <Bell className="w-5 h-5" />;
      }
    };

    const getAlertColor = (type: AlertData['type'], severity?: AlertData['severity']) => {
      if (severity === 'critical') {
        return 'border-red-500/50 bg-red-500/10 text-red-400';
      }

      switch (type) {
        case 'success':
          return 'border-market-up/50 bg-market-up/10 text-market-up';
        case 'warning':
          return 'border-amber-500/50 bg-amber-500/10 text-amber-400';
        case 'error':
          return 'border-market-down/50 bg-market-down/10 text-market-down';
        case 'info':
          return 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400';
        case 'trading':
          return severity === 'high'
            ? 'border-amber-500/50 bg-amber-500/10 text-amber-400'
            : 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400';
        case 'risk':
          return 'border-purple-500/50 bg-purple-500/10 text-purple-400';
        case 'system':
          return 'border-slate-500/50 bg-slate-500/10 text-slate-400';
        default:
          return 'border-slate-600/50 bg-slate-700/10 text-slate-300';
      }
    };

    const getPriorityIndicator = (priority?: AlertData['priority']) => {
      switch (priority) {
        case 'urgent':
          return <motion.div
            className="w-2 h-2 rounded-full bg-red-500"
            animate={{ scale: [1, 1.2, 1], opacity: [0.8, 1, 0.8] }}
            transition={{ duration: 1, repeat: Infinity }}
          />;
        case 'high':
          return <div className="w-2 h-2 rounded-full bg-amber-500" />;
        case 'normal':
          return <div className="w-2 h-2 rounded-full bg-cyan-500" />;
        case 'low':
          return <div className="w-2 h-2 rounded-full bg-slate-500" />;
        default:
          return null;
      }
    };

    const AlertItem = ({ alert, index }: { alert: AlertData; index: number }) => {
      const colorClasses = getAlertColor(alert.type, alert.severity);
      const icon = getAlertIcon(alert.type, alert.severity);

      const alertVariants = {
        initial: {
          opacity: 0,
          y: variant === 'toast' && position.includes('top') ? -50 :
             variant === 'toast' && position.includes('bottom') ? 50 : 0,
          x: variant === 'toast' && position.includes('left') ? -50 :
             variant === 'toast' && position.includes('right') ? 50 : 0,
          scale: 0.9
        },
        animate: {
          opacity: 1,
          y: 0,
          x: 0,
          scale: 1,
          transition: {
            duration: 0.4,
            delay: animated ? index * 0.1 : 0,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        exit: {
          opacity: 0,
          y: variant === 'toast' && position.includes('top') ? -20 :
             variant === 'toast' && position.includes('bottom') ? 20 : 0,
          x: variant === 'toast' && position.includes('right') ? 50 :
             variant === 'toast' && position.includes('left') ? -50 : 0,
          scale: 0.95,
          transition: {
            duration: 0.3,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.02,
          y: -2,
          transition: { duration: 0.2 }
        }
      };

      return (
        <motion.div
          variants={alertVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? "animate" : undefined}
          exit={animated ? "exit" : undefined}
          whileHover={animated ? "hover" : undefined}
          className={cn(
            "relative group backdrop-blur-lg border rounded-2xl overflow-hidden",
            "bg-gradient-to-br from-slate-900/90 via-slate-800/70 to-slate-900/95",
            colorClasses,
            variant === 'toast' && "min-w-[320px] max-w-[400px] shadow-glass-lg",
            variant === 'banner' && "w-full",
            variant === 'sidebar' && "w-full",
            variant === 'modal' && "max-w-md mx-auto",
            variant === 'floating' && "min-w-[280px] shadow-glass-xl"
          )}
        >
          {/* Progress bar for auto-dismiss */}
          {alert.duration && !alert.persistent && (
            <motion.div
              className="absolute top-0 left-0 h-1 bg-current opacity-50"
              initial={{ width: "100%" }}
              animate={{ width: "0%" }}
              transition={{ duration: alert.duration / 1000, ease: "linear" }}
            />
          )}

          {/* Severity pulse effect */}
          {alert.severity === 'critical' && (
            <motion.div
              className="absolute inset-0 border-2 border-red-500/30 rounded-2xl pointer-events-none"
              animate={{
                borderColor: ['rgba(239, 68, 68, 0.3)', 'rgba(239, 68, 68, 0.8)', 'rgba(239, 68, 68, 0.3)']
              }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
          )}

          <div className="p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center space-x-3">
                <motion.div
                  className="flex-shrink-0"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  {icon}
                </motion.div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    {showPriority && getPriorityIndicator(alert.priority)}
                    <motion.h4
                      className="font-bold text-sm leading-tight"
                      whileHover={{ scale: 1.02 }}
                    >
                      {alert.title}
                    </motion.h4>
                  </div>

                  <motion.p
                    className="text-xs opacity-90 leading-relaxed"
                    initial={{ opacity: 0.7 }}
                    animate={{ opacity: 0.9 }}
                  >
                    {alert.message}
                  </motion.p>
                </div>
              </div>

              {!alert.persistent && (
                <motion.button
                  onClick={() => handleDismiss(alert.id)}
                  className="flex-shrink-0 p-1 rounded-lg hover:bg-current/20 transition-colors opacity-50 hover:opacity-100"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <X className="w-4 h-4" />
                </motion.button>
              )}
            </div>

            {/* Timestamp */}
            {showTimestamp && alert.timestamp && (
              <motion.div
                className="flex items-center space-x-1 text-xs opacity-60 mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.6 }}
                transition={{ delay: 0.2 }}
              >
                <Clock className="w-3 h-3" />
                <span>{formatTimestamp(alert.timestamp)}</span>
              </motion.div>
            )}

            {/* Additional Data */}
            {alert.data && Object.keys(alert.data).length > 0 && (
              <motion.div
                className="text-xs space-y-1 mb-3 p-2 rounded-lg bg-current/10"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                transition={{ delay: 0.3 }}
              >
                {Object.entries(alert.data).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="opacity-70 capitalize">{key}:</span>
                    <span className="font-medium">{value?.toString()}</span>
                  </div>
                ))}
              </motion.div>
            )}

            {/* Actions */}
            {alert.actions && alert.actions.length > 0 && (
              <motion.div
                className="flex items-center space-x-2 mt-3"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                {alert.actions.map((action, actionIndex) => (
                  <motion.button
                    key={actionIndex}
                    onClick={() => {
                      action.action();
                      onAction?.(alert.id, actionIndex);
                    }}
                    className={cn(
                      "px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
                      action.variant === 'primary' && "bg-current/20 hover:bg-current/30",
                      action.variant === 'secondary' && "bg-slate-600/20 hover:bg-slate-600/30 text-slate-300",
                      action.variant === 'destructive' && "bg-red-500/20 hover:bg-red-500/30 text-red-400",
                      !action.variant && "bg-current/20 hover:bg-current/30"
                    )}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {action.label}
                  </motion.button>
                ))}
              </motion.div>
            )}
          </div>

          {/* Shimmer effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 pointer-events-none"
            animate={{
              x: ['-100%', '100%']
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "linear"
            }}
          />
        </motion.div>
      );
    };

    const visibleAlerts = alerts
      .filter(alert => !dismissedAlerts.has(alert.id))
      .slice(0, maxVisible);

    const getPositionClasses = () => {
      switch (position) {
        case 'top':
          return 'top-4 left-1/2 transform -translate-x-1/2';
        case 'bottom':
          return 'bottom-4 left-1/2 transform -translate-x-1/2';
        case 'top-left':
          return 'top-4 left-4';
        case 'top-right':
          return 'top-4 right-4';
        case 'bottom-left':
          return 'bottom-4 left-4';
        case 'bottom-right':
          return 'bottom-4 right-4';
        case 'center':
          return 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2';
        default:
          return 'top-4 right-4';
      }
    };

    if (visibleAlerts.length === 0) {
      return null;
    }

    return (
      <motion.div
        ref={ref}
        className={cn(
          variant === 'toast' || variant === 'floating' ? "fixed z-50" : "relative",
          variant === 'toast' || variant === 'floating' ? getPositionClasses() : "",
          variant === 'modal' && "fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm",
          className
        )}
        {...props}
      >
        {/* Dismiss All Button */}
        {visibleAlerts.length > 1 && onDismissAll && (
          <motion.button
            onClick={handleDismissAll}
            className={cn(
              "mb-2 px-3 py-1 rounded-lg text-xs font-medium",
              "bg-slate-600/20 hover:bg-slate-600/30 text-slate-300 border border-slate-600/30",
              "transition-colors duration-200"
            )}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Dismiss All ({visibleAlerts.length})
          </motion.button>
        )}

        {/* Alert Container */}
        <div className={cn(
          "space-y-3",
          variant === 'sidebar' && "space-y-2",
          variant === 'banner' && "space-y-1"
        )}>
          <AnimatePresence mode="popLayout">
            {visibleAlerts.map((alert, index) => (
              <AlertItem key={alert.id} alert={alert} index={index} />
            ))}
          </AnimatePresence>
        </div>

        {/* Overflow Indicator */}
        {alerts.length - dismissedAlerts.size > maxVisible && (
          <motion.div
            className="mt-2 p-2 rounded-lg bg-slate-600/20 border border-slate-600/30 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <p className="text-xs text-slate-400">
              +{alerts.length - dismissedAlerts.size - maxVisible} more alerts
            </p>
          </motion.div>
        )}
      </motion.div>
    );
  }
);

AlertBanner.displayName = "AlertBanner";

export { AlertBanner };