"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Activity,
  Wifi,
  WifiOff,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Pause,
  Play,
  RotateCcw,
  Zap,
  Shield,
  Globe,
  Server,
  Database,
  TrendingUp,
  TrendingDown,
  Minus,
  Circle,
  Square,
  Triangle
} from "lucide-react";

export interface StatusData {
  id: string;
  label: string;
  status: 'online' | 'offline' | 'connecting' | 'error' | 'warning' | 'maintenance' | 'unknown';
  value?: string | number;
  unit?: string;
  description?: string;
  lastUpdate?: number;
  uptime?: number;
  latency?: number;
  health?: number; // 0-100 percentage
  trend?: 'up' | 'down' | 'stable';
  metadata?: Record<string, any>;
}

export interface StatusIndicatorProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  statuses: StatusData[];
  variant?: 'dot' | 'card' | 'badge' | 'detailed' | 'minimal' | 'professional';
  layout?: 'horizontal' | 'vertical' | 'grid';
  animated?: boolean;
  realTime?: boolean;
  showLabels?: boolean;
  showValues?: boolean;
  showTrends?: boolean;
  showTimestamp?: boolean;
  showHealth?: boolean;
  groupByStatus?: boolean;
  size?: 'sm' | 'md' | 'lg';
  glowEffect?: boolean;
  pulseAnimation?: boolean;
  onStatusClick?: (status: StatusData) => void;
  className?: string;
}

const StatusIndicator = React.forwardRef<HTMLDivElement, StatusIndicatorProps>(
  ({
    statuses,
    variant = 'card',
    layout = 'horizontal',
    animated = true,
    realTime = true,
    showLabels = true,
    showValues = false,
    showTrends = false,
    showTimestamp = false,
    showHealth = false,
    groupByStatus = false,
    size = 'md',
    glowEffect = true,
    pulseAnimation = true,
    onStatusClick,
    className,
    ...props
  }, ref) => {
    const [previousStatuses, setPreviousStatuses] = React.useState<StatusData[]>([]);
    const [updatedStatuses, setUpdatedStatuses] = React.useState<Set<string>>(new Set());

    React.useEffect(() => {
      if (realTime) {
        // Track status changes for animations
        const updated = new Set<string>();
        statuses.forEach(status => {
          const previous = previousStatuses.find(prev => prev.id === status.id);
          if (previous && previous.status !== status.status) {
            updated.add(status.id);
          }
        });
        setUpdatedStatuses(updated);
        setPreviousStatuses(statuses);

        // Clear update flags after animation
        const timer = setTimeout(() => setUpdatedStatuses(new Set()), 1000);
        return () => clearTimeout(timer);
      }
    }, [statuses, previousStatuses, realTime]);

    const getStatusColor = (status: StatusData['status']) => {
      switch (status) {
        case 'online':
          return {
            color: 'text-status-live',
            bg: 'bg-status-live/20',
            border: 'border-status-live/50',
            shadow: 'shadow-status-live',
            ring: 'ring-status-live/30'
          };
        case 'offline':
          return {
            color: 'text-status-offline',
            bg: 'bg-status-offline/20',
            border: 'border-status-offline/50',
            shadow: 'shadow-status-offline',
            ring: 'ring-status-offline/30'
          };
        case 'connecting':
          return {
            color: 'text-status-delayed',
            bg: 'bg-status-delayed/20',
            border: 'border-status-delayed/50',
            shadow: 'shadow-status-delayed',
            ring: 'ring-status-delayed/30'
          };
        case 'error':
          return {
            color: 'text-market-down',
            bg: 'bg-market-down/20',
            border: 'border-market-down/50',
            shadow: 'shadow-market-down',
            ring: 'ring-market-down/30'
          };
        case 'warning':
          return {
            color: 'text-amber-400',
            bg: 'bg-amber-400/20',
            border: 'border-amber-400/50',
            shadow: 'shadow-amber-400/50',
            ring: 'ring-amber-400/30'
          };
        case 'maintenance':
          return {
            color: 'text-purple-400',
            bg: 'bg-purple-400/20',
            border: 'border-purple-400/50',
            shadow: 'shadow-purple-400/50',
            ring: 'ring-purple-400/30'
          };
        default:
          return {
            color: 'text-slate-400',
            bg: 'bg-slate-400/20',
            border: 'border-slate-400/50',
            shadow: 'shadow-slate-400/50',
            ring: 'ring-slate-400/30'
          };
      }
    };

    const getStatusIcon = (status: StatusData) => {
      switch (status.status) {
        case 'online':
          return <CheckCircle className="w-full h-full" />;
        case 'offline':
          return <XCircle className="w-full h-full" />;
        case 'connecting':
          return <RotateCcw className="w-full h-full" />;
        case 'error':
          return <AlertTriangle className="w-full h-full" />;
        case 'warning':
          return <AlertTriangle className="w-full h-full" />;
        case 'maintenance':
          return <Pause className="w-full h-full" />;
        default:
          return <Circle className="w-full h-full" />;
      }
    };

    const getTrendIcon = (trend?: StatusData['trend']) => {
      switch (trend) {
        case 'up':
          return <TrendingUp className="w-3 h-3 text-market-up" />;
        case 'down':
          return <TrendingDown className="w-3 h-3 text-market-down" />;
        case 'stable':
          return <Minus className="w-3 h-3 text-slate-400" />;
        default:
          return null;
      }
    };

    const formatUptime = (uptime?: number) => {
      if (!uptime) return '';
      const hours = Math.floor(uptime / 3600);
      const minutes = Math.floor((uptime % 3600) / 60);
      if (hours > 0) return `${hours}h ${minutes}m`;
      return `${minutes}m`;
    };

    const formatTimestamp = (timestamp?: number) => {
      if (!timestamp) return '';
      const now = Date.now();
      const diff = now - timestamp;
      const minutes = Math.floor(diff / (1000 * 60));
      if (minutes < 1) return 'now';
      if (minutes < 60) return `${minutes}m ago`;
      const hours = Math.floor(minutes / 60);
      return `${hours}h ago`;
    };

    const getSizeClasses = () => {
      switch (size) {
        case 'sm':
          return {
            dot: 'w-2 h-2',
            icon: 'w-3 h-3',
            text: 'text-xs',
            padding: 'p-2'
          };
        case 'lg':
          return {
            dot: 'w-4 h-4',
            icon: 'w-6 h-6',
            text: 'text-base',
            padding: 'p-4'
          };
        default:
          return {
            dot: 'w-3 h-3',
            icon: 'w-4 h-4',
            text: 'text-sm',
            padding: 'p-3'
          };
      }
    };

    const sizeClasses = getSizeClasses();

    const DotIndicator = ({ status, index }: { status: StatusData; index: number }) => {
      const colors = getStatusColor(status.status);
      const isUpdated = updatedStatuses.has(status.id);

      const dotVariants = {
        initial: { scale: 0, opacity: 0 },
        animate: {
          scale: 1,
          opacity: 1,
          transition: {
            duration: 0.3,
            delay: animated ? index * 0.1 : 0,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        pulse: {
          scale: [1, 1.2, 1],
          opacity: [0.8, 1, 0.8],
          transition: {
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }
        },
        updated: {
          scale: [1, 1.3, 1],
          boxShadow: [
            '0 0 0px rgba(255, 255, 255, 0)',
            '0 0 16px rgba(255, 255, 255, 0.6)',
            '0 0 8px rgba(255, 255, 255, 0.3)'
          ],
          transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
        }
      };

      return (
        <motion.div
          variants={dotVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? (isUpdated ? "updated" : pulseAnimation ? "pulse" : "animate") : undefined}
          onClick={() => onStatusClick?.(status)}
          className={cn(
            "relative rounded-full cursor-pointer",
            sizeClasses.dot,
            colors.bg,
            glowEffect && colors.shadow,
            onStatusClick && "hover:scale-110 transition-transform duration-200"
          )}
          title={`${status.label}: ${status.status}`}
        >
          <div className={cn("absolute inset-0 rounded-full", colors.bg)} />
          {glowEffect && (
            <div className={cn("absolute inset-0 rounded-full blur-sm", colors.bg, "opacity-60")} />
          )}
        </motion.div>
      );
    };

    const BadgeIndicator = ({ status, index }: { status: StatusData; index: number }) => {
      const colors = getStatusColor(status.status);
      const isUpdated = updatedStatuses.has(status.id);

      const badgeVariants = {
        initial: { opacity: 0, x: -20 },
        animate: {
          opacity: 1,
          x: 0,
          transition: {
            duration: 0.4,
            delay: animated ? index * 0.05 : 0,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.05,
          y: -1,
          transition: { duration: 0.2 }
        },
        updated: {
          scale: [1, 1.05, 1],
          borderColor: ['rgba(255, 255, 255, 0.3)', 'rgba(255, 255, 255, 0.8)', 'rgba(255, 255, 255, 0.3)'],
          transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] }
        }
      };

      return (
        <motion.div
          variants={badgeVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? (isUpdated ? "updated" : "animate") : undefined}
          whileHover={animated ? "hover" : undefined}
          onClick={() => onStatusClick?.(status)}
          className={cn(
            "flex items-center space-x-2 px-3 py-1.5 rounded-full border cursor-pointer",
            colors.bg,
            colors.border,
            colors.color,
            sizeClasses.text,
            "transition-all duration-200"
          )}
        >
          <div className={cn("rounded-full", sizeClasses.dot, colors.bg)} />
          {showLabels && <span className="font-medium">{status.label}</span>}
          {showValues && status.value && (
            <span className="opacity-80">{status.value}{status.unit}</span>
          )}
        </motion.div>
      );
    };

    const CardIndicator = ({ status, index }: { status: StatusData; index: number }) => {
      const colors = getStatusColor(status.status);
      const isUpdated = updatedStatuses.has(status.id);

      const cardVariants = {
        initial: { opacity: 0, y: 20, scale: 0.95 },
        animate: {
          opacity: 1,
          y: 0,
          scale: 1,
          transition: {
            duration: 0.5,
            delay: animated ? index * 0.1 : 0,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.02,
          y: -2,
          transition: { duration: 0.2 }
        },
        updated: {
          scale: [1, 1.02, 1],
          borderColor: ['rgba(255, 255, 255, 0.3)', 'rgba(255, 255, 255, 0.6)', 'rgba(255, 255, 255, 0.3)'],
          transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
        }
      };

      return (
        <motion.div
          variants={cardVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? (isUpdated ? "updated" : "animate") : undefined}
          whileHover={animated ? "hover" : undefined}
          onClick={() => onStatusClick?.(status)}
          className={cn(
            "relative cursor-pointer backdrop-blur-lg border rounded-xl overflow-hidden",
            "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
            "border-slate-600/30 hover:border-cyan-400/40",
            sizeClasses.padding,
            variant === 'professional' && "shadow-glass-md hover:shadow-glass-lg",
            "transition-all duration-300"
          )}
        >
          <div className="flex items-center space-x-3">
            {/* Status Icon */}
            <motion.div
              className={cn(
                "relative flex items-center justify-center rounded-xl",
                colors.bg,
                colors.color,
                sizeClasses.icon === 'w-3 h-3' ? 'w-8 h-8' :
                sizeClasses.icon === 'w-6 h-6' ? 'w-12 h-12' : 'w-10 h-10'
              )}
              animate={status.status === 'connecting' ? {
                rotate: [0, 360]
              } : {}}
              transition={{
                duration: 2,
                repeat: status.status === 'connecting' ? Infinity : 0,
                ease: "linear"
              }}
            >
              <div className={sizeClasses.icon}>
                {getStatusIcon(status)}
              </div>

              {/* Pulse ring for active status */}
              {(status.status === 'online' || status.status === 'connecting') && pulseAnimation && (
                <motion.div
                  className={cn("absolute inset-0 rounded-xl border-2", colors.border)}
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0, 0.5]
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              )}
            </motion.div>

            {/* Status Info */}
            <div className="flex-1 min-w-0">
              {showLabels && (
                <div className="flex items-center space-x-2 mb-1">
                  <motion.h4
                    className={cn(
                      "font-medium truncate",
                      colors.color,
                      sizeClasses.text
                    )}
                    whileHover={{ scale: 1.02 }}
                  >
                    {status.label}
                  </motion.h4>

                  {showTrends && getTrendIcon(status.trend)}
                </div>
              )}

              <div className="flex items-center justify-between">
                <motion.span
                  className={cn(
                    "text-xs uppercase tracking-wide font-medium",
                    colors.color,
                    "opacity-90"
                  )}
                  whileHover={{ opacity: 1 }}
                >
                  {status.status}
                </motion.span>

                {showValues && status.value && (
                  <motion.span
                    className={cn("text-xs tabular-nums", colors.color)}
                    whileHover={{ scale: 1.05 }}
                  >
                    {status.value}{status.unit}
                  </motion.span>
                )}
              </div>

              {/* Additional Info */}
              {variant === 'detailed' && (
                <motion.div
                  className="mt-2 space-y-1 text-xs text-slate-400"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ delay: 0.2 }}
                >
                  {status.latency && (
                    <div className="flex justify-between">
                      <span>Latency:</span>
                      <span className="tabular-nums">{status.latency}ms</span>
                    </div>
                  )}

                  {status.uptime && (
                    <div className="flex justify-between">
                      <span>Uptime:</span>
                      <span>{formatUptime(status.uptime)}</span>
                    </div>
                  )}

                  {showTimestamp && status.lastUpdate && (
                    <div className="flex justify-between">
                      <span>Updated:</span>
                      <span>{formatTimestamp(status.lastUpdate)}</span>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Health Bar */}
              {showHealth && status.health !== undefined && (
                <motion.div
                  className="mt-2"
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: "100%" }}
                  transition={{ delay: 0.3 }}
                >
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Health</span>
                    <span>{status.health}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                    <motion.div
                      className={cn(
                        "h-full rounded-full",
                        status.health >= 80 ? "bg-status-live" :
                        status.health >= 60 ? "bg-status-delayed" :
                        "bg-status-offline"
                      )}
                      initial={{ width: 0 }}
                      animate={{ width: `${status.health}%` }}
                      transition={{ duration: 1, ease: [0.4, 0, 0.2, 1] }}
                    />
                  </div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Background glow effect */}
          {glowEffect && (
            <motion.div
              className={cn(
                "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none",
                "bg-gradient-to-br from-cyan-400/5 via-transparent to-purple-400/5"
              )}
            />
          )}
        </motion.div>
      );
    };

    const groupedStatuses = React.useMemo(() => {
      if (!groupByStatus) return { all: statuses };

      return statuses.reduce((groups, status) => {
        const key = status.status;
        if (!groups[key]) groups[key] = [];
        groups[key].push(status);
        return groups;
      }, {} as Record<string, StatusData[]>);
    }, [statuses, groupByStatus]);

    const containerVariants = {
      initial: { opacity: 0 },
      animate: {
        opacity: 1,
        transition: {
          duration: 0.5,
          staggerChildren: animated ? 0.1 : 0,
          delayChildren: 0.1
        }
      }
    };

    return (
      <motion.div
        ref={ref}
        variants={containerVariants}
        initial={animated ? "initial" : undefined}
        animate={animated ? "animate" : undefined}
        className={cn(
          layout === 'horizontal' && "flex items-center flex-wrap gap-3",
          layout === 'vertical' && "flex flex-col space-y-3",
          layout === 'grid' && "grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3",
          className
        )}
        {...props}
      >
        <AnimatePresence mode="popLayout">
          {groupByStatus ? (
            Object.entries(groupedStatuses).map(([statusType, statusList]) => (
              <div key={statusType} className="space-y-2">
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                  {statusType} ({statusList.length})
                </h3>
                <div className={cn(
                  layout === 'horizontal' && "flex items-center flex-wrap gap-2",
                  layout === 'vertical' && "flex flex-col space-y-2",
                  layout === 'grid' && "grid grid-cols-1 gap-2"
                )}>
                  {statusList.map((status, index) => (
                    <div key={status.id}>
                      {variant === 'dot' && <DotIndicator status={status} index={index} />}
                      {variant === 'badge' && <BadgeIndicator status={status} index={index} />}
                      {(variant === 'card' || variant === 'detailed' || variant === 'professional') && (
                        <CardIndicator status={status} index={index} />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))
          ) : (
            statuses.map((status, index) => (
              <div key={status.id}>
                {variant === 'dot' && <DotIndicator status={status} index={index} />}
                {variant === 'badge' && <BadgeIndicator status={status} index={index} />}
                {(variant === 'card' || variant === 'detailed' || variant === 'professional') && (
                  <CardIndicator status={status} index={index} />
                )}
              </div>
            ))
          )}
        </AnimatePresence>
      </motion.div>
    );
  }
);

StatusIndicator.displayName = "StatusIndicator";

export { StatusIndicator };