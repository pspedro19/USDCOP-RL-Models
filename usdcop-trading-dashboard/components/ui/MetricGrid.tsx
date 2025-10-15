"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  Percent,
  Target,
  BarChart3,
  PieChart,
  LineChart,
  Zap,
  Clock,
  Users,
  Globe,
  Shield,
  Award,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Minus,
  ArrowUpRight,
  ArrowDownRight,
  ArrowRight
} from "lucide-react";

export interface MetricData {
  id: string;
  title: string;
  value: number | string;
  previousValue?: number | string;
  unit?: string;
  prefix?: string;
  suffix?: string;
  change?: number;
  changePercent?: number;
  trend?: 'up' | 'down' | 'neutral';
  status?: 'positive' | 'negative' | 'neutral' | 'warning' | 'critical';
  icon?: React.ReactNode;
  description?: string;
  target?: number;
  min?: number;
  max?: number;
  format?: 'currency' | 'percentage' | 'number' | 'decimal';
  precision?: number;
  sparkline?: number[];
  timestamp?: number;
}

export interface MetricGridProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  metrics: MetricData[];
  variant?: 'default' | 'compact' | 'detailed' | 'professional' | 'analytics';
  columns?: 1 | 2 | 3 | 4 | 5 | 6;
  animated?: boolean;
  realTime?: boolean;
  showTrends?: boolean;
  showSparklines?: boolean;
  showTargets?: boolean;
  staggerChildren?: boolean;
  onMetricClick?: (metric: MetricData) => void;
  className?: string;
}

const MetricGrid = React.forwardRef<HTMLDivElement, MetricGridProps>(
  ({
    metrics,
    variant = 'default',
    columns = 3,
    animated = true,
    realTime = true,
    showTrends = true,
    showSparklines = false,
    showTargets = false,
    staggerChildren = true,
    onMetricClick,
    className,
    ...props
  }, ref) => {
    const [updatedMetrics, setUpdatedMetrics] = React.useState<Set<string>>(new Set());

    React.useEffect(() => {
      if (realTime) {
        // Mark metrics as updated for animation
        const updated = new Set<string>();
        metrics.forEach(metric => {
          if (metric.previousValue !== undefined && metric.value !== metric.previousValue) {
            updated.add(metric.id);
          }
        });
        setUpdatedMetrics(updated);

        // Clear update flags after animation
        const timer = setTimeout(() => setUpdatedMetrics(new Set()), 1000);
        return () => clearTimeout(timer);
      }
    }, [metrics, realTime]);

    const formatValue = (metric: MetricData) => {
      const { value, format, precision = 2, prefix = '', suffix = '', unit = '' } = metric;

      if (typeof value === 'string') return `${prefix}${value}${suffix}${unit}`;

      let formatted: string;

      switch (format) {
        case 'currency':
          formatted = `$${value.toFixed(precision)}`;
          break;
        case 'percentage':
          formatted = `${value.toFixed(precision)}%`;
          break;
        case 'decimal':
          formatted = value.toFixed(precision);
          break;
        default:
          if (value >= 1e9) formatted = `${(value / 1e9).toFixed(1)}B`;
          else if (value >= 1e6) formatted = `${(value / 1e6).toFixed(1)}M`;
          else if (value >= 1e3) formatted = `${(value / 1e3).toFixed(1)}K`;
          else formatted = value.toFixed(precision);
      }

      return `${prefix}${formatted}${suffix}${unit}`;
    };

    const formatChange = (change: number, isPercent = false) => {
      const formatted = isPercent ? `${change.toFixed(2)}%` : change.toFixed(2);
      return `${change >= 0 ? '+' : ''}${formatted}`;
    };

    const getStatusColor = (status?: string, trend?: string) => {
      if (status) {
        switch (status) {
          case 'positive': return 'text-market-up border-market-up/30 bg-market-up/10';
          case 'negative': return 'text-market-down border-market-down/30 bg-market-down/10';
          case 'warning': return 'text-amber-400 border-amber-400/30 bg-amber-400/10';
          case 'critical': return 'text-red-500 border-red-500/30 bg-red-500/10';
          default: return 'text-slate-300 border-slate-600/30 bg-slate-700/10';
        }
      }

      if (trend) {
        switch (trend) {
          case 'up': return 'text-market-up border-market-up/30 bg-market-up/10';
          case 'down': return 'text-market-down border-market-down/30 bg-market-down/10';
          default: return 'text-slate-300 border-slate-600/30 bg-slate-700/10';
        }
      }

      return 'text-slate-300 border-slate-600/30 bg-slate-700/10';
    };

    const getTrendIcon = (trend?: string, status?: string) => {
      if (status === 'positive') return <CheckCircle className="w-4 h-4" />;
      if (status === 'negative') return <XCircle className="w-4 h-4" />;
      if (status === 'warning') return <AlertTriangle className="w-4 h-4" />;
      if (status === 'critical') return <AlertTriangle className="w-4 h-4" />;

      if (trend === 'up') return <ArrowUpRight className="w-4 h-4" />;
      if (trend === 'down') return <ArrowDownRight className="w-4 h-4" />;
      return <ArrowRight className="w-4 h-4" />;
    };

    const getDefaultIcon = (title: string) => {
      const titleLower = title.toLowerCase();
      if (titleLower.includes('revenue') || titleLower.includes('profit') || titleLower.includes('pnl')) {
        return <DollarSign className="w-5 h-5" />;
      }
      if (titleLower.includes('volume') || titleLower.includes('trades')) {
        return <BarChart3 className="w-5 h-5" />;
      }
      if (titleLower.includes('percent') || titleLower.includes('%')) {
        return <Percent className="w-5 h-5" />;
      }
      if (titleLower.includes('active') || titleLower.includes('live')) {
        return <Activity className="w-5 h-5" />;
      }
      if (titleLower.includes('user') || titleLower.includes('client')) {
        return <Users className="w-5 h-5" />;
      }
      if (titleLower.includes('time') || titleLower.includes('duration')) {
        return <Clock className="w-5 h-5" />;
      }
      if (titleLower.includes('risk') || titleLower.includes('exposure')) {
        return <Shield className="w-5 h-5" />;
      }
      if (titleLower.includes('target') || titleLower.includes('goal')) {
        return <Target className="w-5 h-5" />;
      }
      return <BarChart3 className="w-5 h-5" />;
    };

    const Sparkline = ({ data }: { data: number[] }) => {
      if (data.length < 2) return null;

      const max = Math.max(...data);
      const min = Math.min(...data);
      const range = max - min;

      if (range === 0) return null;

      const points = data.map((value, index) => {
        const x = (index / (data.length - 1)) * 100;
        const y = 100 - ((value - min) / range) * 100;
        return `${x},${y}`;
      }).join(' ');

      const trend = data[data.length - 1] > data[0] ? 'up' : 'down';

      return (
        <motion.div
          className="h-8 w-16"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <svg width="100%" height="100%" viewBox="0 0 100 100" className="overflow-visible">
            <motion.polyline
              fill="none"
              stroke={trend === 'up' ? '#00D395' : '#FF3B69'}
              strokeWidth="2"
              points={points}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
            />
          </svg>
        </motion.div>
      );
    };

    const ProgressBar = ({ value, target, max }: { value: number; target?: number; max?: number }) => {
      const maxValue = max || target || 100;
      const percentage = Math.min((value / maxValue) * 100, 100);
      const targetPercentage = target ? Math.min((target / maxValue) * 100, 100) : undefined;

      return (
        <div className="w-full h-2 bg-slate-700/50 rounded-full overflow-hidden">
          <motion.div
            className={cn(
              "h-full rounded-full",
              percentage >= (targetPercentage || 80) ? "bg-market-up" : "bg-amber-400"
            )}
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
          />
          {targetPercentage && (
            <div
              className="absolute top-0 w-0.5 h-full bg-cyan-400"
              style={{ left: `${targetPercentage}%` }}
            />
          )}
        </div>
      );
    };

    const MetricCard = ({ metric, index }: { metric: MetricData; index: number }) => {
      const isUpdated = updatedMetrics.has(metric.id);
      const statusColor = getStatusColor(metric.status, metric.trend);
      const icon = metric.icon || getDefaultIcon(metric.title);

      const cardVariants = {
        initial: {
          opacity: 0,
          scale: 0.9,
          y: 20
        },
        animate: {
          opacity: 1,
          scale: 1,
          y: 0,
          transition: {
            duration: 0.5,
            delay: staggerChildren ? index * 0.1 : 0,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.02,
          y: -4,
          transition: { duration: 0.2 }
        },
        tap: {
          scale: 0.98,
          transition: { duration: 0.1 }
        },
        updated: {
          scale: [1, 1.05, 1],
          borderColor: [
            'rgba(100, 116, 139, 0.3)',
            'rgba(6, 182, 212, 0.6)',
            'rgba(100, 116, 139, 0.3)'
          ],
          transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
        }
      };

      return (
        <motion.div
          variants={cardVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? (isUpdated ? "updated" : "animate") : undefined}
          whileHover={animated ? "hover" : undefined}
          whileTap={animated ? "tap" : undefined}
          onClick={() => onMetricClick?.(metric)}
          className={cn(
            "relative group cursor-pointer",
            "backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
            "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
            "transition-all duration-300 ease-in-out",
            variant === 'professional' && "shadow-glass-md hover:shadow-glass-lg",
            variant === 'compact' && "p-3",
            variant === 'detailed' && "p-6",
            variant !== 'compact' && variant !== 'detailed' && "p-4",
            onMetricClick && "hover:border-cyan-400/40",
            statusColor.includes('market-up') && "hover:shadow-market-up/20",
            statusColor.includes('market-down') && "hover:shadow-market-down/20"
          )}
        >
          {/* Status Indicator */}
          {(metric.status || metric.trend) && (
            <motion.div
              className={cn(
                "absolute top-2 right-2 p-1 rounded-lg",
                statusColor
              )}
              whileHover={{ scale: 1.1 }}
            >
              {getTrendIcon(metric.trend, metric.status)}
            </motion.div>
          )}

          {/* Real-time Indicator */}
          {realTime && isUpdated && (
            <motion.div
              className="absolute top-2 left-2 w-2 h-2 rounded-full bg-status-live shadow-status-live"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.8, 1, 0.8]
              }}
              transition={{
                duration: 1,
                repeat: 3,
                ease: "easeInOut"
              }}
            />
          )}

          <div className="relative z-10">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <motion.div
                className={cn(
                  "flex items-center justify-center w-10 h-10 rounded-xl",
                  "bg-gradient-to-br from-cyan-500/20 to-purple-500/20",
                  "text-cyan-400"
                )}
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                {icon}
              </motion.div>

              {showSparklines && metric.sparkline && (
                <Sparkline data={metric.sparkline} />
              )}
            </div>

            {/* Title */}
            <motion.h3
              className={cn(
                "text-sm font-medium text-slate-300 mb-2 leading-tight",
                variant === 'professional' && "font-mono uppercase tracking-wide"
              )}
              whileHover={{ scale: 1.02 }}
            >
              {metric.title}
            </motion.h3>

            {/* Value */}
            <motion.div
              className={cn(
                "text-2xl font-bold tabular-nums mb-2",
                variant === 'professional' && "font-mono text-glow text-cyan-100",
                variant === 'compact' && "text-xl",
                variant === 'detailed' && "text-3xl"
              )}
              animate={isUpdated ? {
                scale: [1, 1.1, 1],
                color: ['#CBD5E1', '#06B6D4', '#CBD5E1']
              } : {}}
              transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
            >
              {formatValue(metric)}
            </motion.div>

            {/* Change Indicators */}
            {showTrends && (metric.change !== undefined || metric.changePercent !== undefined) && (
              <motion.div
                className="flex items-center space-x-2 mb-2"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                {metric.trend && (
                  <motion.div
                    className={cn(
                      "flex items-center",
                      metric.trend === 'up' && "text-market-up",
                      metric.trend === 'down' && "text-market-down",
                      metric.trend === 'neutral' && "text-slate-400"
                    )}
                    whileHover={{ scale: 1.05 }}
                  >
                    {metric.trend === 'up' && <TrendingUp className="w-3 h-3" />}
                    {metric.trend === 'down' && <TrendingDown className="w-3 h-3" />}
                    {metric.trend === 'neutral' && <Minus className="w-3 h-3" />}
                  </motion.div>
                )}

                {metric.change !== undefined && (
                  <motion.span
                    className={cn(
                      "text-xs font-medium",
                      metric.change >= 0 ? "text-market-up" : "text-market-down"
                    )}
                    whileHover={{ scale: 1.05 }}
                  >
                    {formatChange(metric.change)}
                  </motion.span>
                )}

                {metric.changePercent !== undefined && (
                  <motion.span
                    className={cn(
                      "text-xs font-medium px-1.5 py-0.5 rounded",
                      metric.changePercent >= 0
                        ? "bg-market-up/20 text-market-up border border-market-up/30"
                        : "bg-market-down/20 text-market-down border border-market-down/30"
                    )}
                    whileHover={{ scale: 1.05 }}
                  >
                    {formatChange(metric.changePercent, true)}
                  </motion.span>
                )}
              </motion.div>
            )}

            {/* Progress Bar */}
            {showTargets && (metric.target || metric.max) && typeof metric.value === 'number' && (
              <motion.div
                className="space-y-1"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Progress</span>
                  {metric.target && <span>Target: {formatValue({ ...metric, value: metric.target })}</span>}
                </div>
                <ProgressBar
                  value={metric.value}
                  target={metric.target}
                  max={metric.max}
                />
              </motion.div>
            )}

            {/* Description */}
            {variant === 'detailed' && metric.description && (
              <motion.p
                className="text-xs text-slate-400 mt-2 leading-relaxed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                {metric.description}
              </motion.p>
            )}

            {/* Timestamp */}
            {metric.timestamp && (
              <motion.div
                className="flex items-center space-x-1 mt-2 text-xs text-slate-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
              >
                <Clock className="w-3 h-3" />
                <span>
                  {new Date(metric.timestamp).toLocaleTimeString()}
                </span>
              </motion.div>
            )}
          </div>

          {/* Background Gradient Effect */}
          <motion.div
            className={cn(
              "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none",
              "bg-gradient-to-br from-cyan-400/5 via-transparent to-purple-400/5"
            )}
          />

          {/* Shimmer Effect */}
          {isUpdated && (
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent pointer-events-none"
              initial={{ x: '-100%' }}
              animate={{ x: '100%' }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
            />
          )}
        </motion.div>
      );
    };

    const containerVariants = {
      initial: { opacity: 0 },
      animate: {
        opacity: 1,
        transition: {
          duration: 0.5,
          staggerChildren: staggerChildren ? 0.1 : 0,
          delayChildren: 0.2
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
          "grid gap-4",
          columns === 1 && "grid-cols-1",
          columns === 2 && "grid-cols-1 sm:grid-cols-2",
          columns === 3 && "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
          columns === 4 && "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
          columns === 5 && "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5",
          columns === 6 && "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 3xl:grid-cols-6",
          className
        )}
        {...props}
      >
        <AnimatePresence mode="popLayout">
          {metrics.map((metric, index) => (
            <MetricCard key={metric.id} metric={metric} index={index} />
          ))}
        </AnimatePresence>
      </motion.div>
    );
  }
);

MetricGrid.displayName = "MetricGrid";

export { MetricGrid };