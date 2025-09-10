'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { motionLibrary } from '@/lib/motion';

/**
 * Enhanced AnimatedList component for professional stagger animations
 * Provides smooth, performant list animations for trading dashboard
 */

interface AnimatedListProps {
  children: React.ReactNode[];
  className?: string;
  staggerDelay?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
  variant?: 'fade' | 'slide' | 'scale' | 'professional';
  once?: boolean;
  loading?: boolean;
  gridColumns?: number;
}

export function AnimatedList({
  children,
  className = '',
  staggerDelay = 0.1,
  direction = 'up',
  variant = 'professional',
  once = false,
  loading = false,
  gridColumns,
}: AnimatedListProps) {
  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: 0.1,
      }
    },
    exit: {
      opacity: 0,
      transition: {
        staggerChildren: staggerDelay / 2,
        staggerDirection: -1,
      }
    }
  };

  const itemVariants = {
    fade: {
      initial: { opacity: 0 },
      animate: { 
        opacity: 1,
        transition: {
          duration: 0.4,
          ease: motionLibrary.presets.easing.smooth,
        }
      },
      exit: { 
        opacity: 0,
        transition: {
          duration: 0.2,
        }
      }
    },

    slide: motionLibrary.utils.createSlideAnimation(direction, 20),

    scale: {
      initial: { opacity: 0, scale: 0.8 },
      animate: {
        opacity: 1,
        scale: 1,
        transition: {
          duration: 0.4,
          ease: motionLibrary.presets.easing.bounce,
        }
      },
      exit: {
        opacity: 0,
        scale: 0.8,
        transition: {
          duration: 0.2,
        }
      }
    },

    professional: {
      initial: { 
        opacity: 0, 
        y: direction === 'up' ? 30 : direction === 'down' ? -30 : 0,
        x: direction === 'left' ? 30 : direction === 'right' ? -30 : 0,
        scale: 0.95,
      },
      animate: {
        opacity: 1,
        y: 0,
        x: 0,
        scale: 1,
        transition: {
          duration: 0.5,
          ease: motionLibrary.presets.easing.smooth,
        }
      },
      exit: {
        opacity: 0,
        y: direction === 'up' ? -20 : direction === 'down' ? 20 : 0,
        x: direction === 'left' ? -20 : direction === 'right' ? 20 : 0,
        scale: 0.95,
        transition: {
          duration: 0.3,
          ease: motionLibrary.presets.easing.professional,
        }
      }
    }
  };

  const currentItemVariant = itemVariants[variant];

  if (loading) {
    return (
      <motion.div 
        className={`space-y-4 ${gridColumns ? `grid grid-cols-${gridColumns} gap-4` : ''} ${className}`}
        variants={containerVariants}
        initial="initial"
        animate="animate"
      >
        {children.map((_, index) => (
          <motion.div
            key={index}
            className="glass-surface-secondary rounded-xl p-6 backdrop-blur-md"
            variants={motionLibrary.loading.skeleton}
            animate="animate"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="space-y-3">
              <div className="h-6 bg-slate-700/50 rounded animate-pulse" />
              <div className="h-4 bg-slate-700/30 rounded animate-pulse" />
              <div className="h-4 bg-slate-700/20 rounded animate-pulse w-3/4" />
            </div>
          </motion.div>
        ))}
      </motion.div>
    );
  }

  return (
    <motion.div
      className={`${gridColumns ? `grid grid-cols-${gridColumns} gap-4` : 'space-y-4'} ${className}`}
      variants={containerVariants}
      initial="initial"
      animate="animate"
      exit="exit"
    >
      <AnimatePresence mode="popLayout">
        {children.map((child, index) => (
          <motion.div
            key={index}
            variants={currentItemVariant}
            initial="initial"
            animate="animate"
            exit="exit"
            layout
            className="relative"
            style={{ willChange: 'transform, opacity' }}
            whileHover={{
              scale: 1.02,
              transition: { duration: 0.2, ease: 'easeOut' }
            }}
          >
            {child}
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  );
}

/**
 * AnimatedGrid - Specialized component for grid layouts with enhanced animations
 */
interface AnimatedGridProps extends AnimatedListProps {
  columns: number;
  gap?: 'sm' | 'md' | 'lg';
  responsive?: {
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
  };
}

export function AnimatedGrid({
  children,
  columns,
  gap = 'md',
  responsive,
  className = '',
  staggerDelay = 0.05,
  variant = 'professional',
  direction = 'up',
  loading = false,
  ...props
}: AnimatedGridProps) {
  const gapClasses = {
    sm: 'gap-3',
    md: 'gap-4',
    lg: 'gap-6',
  };

  const responsiveClasses = responsive ? 
    `grid-cols-1 ${responsive.sm ? `sm:grid-cols-${responsive.sm}` : ''} ${responsive.md ? `md:grid-cols-${responsive.md}` : ''} ${responsive.lg ? `lg:grid-cols-${responsive.lg}` : ''} ${responsive.xl ? `xl:grid-cols-${responsive.xl}` : ''}` :
    `grid-cols-${columns}`;

  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: 0.1,
      }
    }
  };

  const itemVariants = {
    initial: { 
      opacity: 0, 
      y: direction === 'up' ? 40 : direction === 'down' ? -40 : 0,
      x: direction === 'left' ? 40 : direction === 'right' ? -40 : 0,
      scale: 0.9
    },
    animate: (index: number) => ({
      opacity: 1,
      y: 0,
      x: 0,
      scale: 1,
      transition: {
        duration: 0.6,
        ease: motionLibrary.presets.easing.smooth,
        delay: index * staggerDelay,
      }
    })
  };

  if (loading) {
    return (
      <div className={`grid ${responsiveClasses} ${gapClasses[gap]} ${className}`}>
        {Array.from({ length: children.length || 6 }).map((_, index) => (
          <motion.div
            key={index}
            className="glass-surface-secondary rounded-xl p-6 backdrop-blur-md"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ 
              delay: index * 0.1,
              duration: 0.5,
              ease: motionLibrary.presets.easing.smooth
            }}
          >
            <div className="space-y-4">
              <div className="h-6 bg-slate-700/50 rounded loading-shimmer-glass" />
              <div className="h-16 bg-slate-700/30 rounded loading-shimmer-glass" />
              <div className="flex justify-between">
                <div className="h-4 bg-slate-700/40 rounded w-1/3 loading-shimmer-glass" />
                <div className="h-4 bg-slate-700/40 rounded w-1/4 loading-shimmer-glass" />
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    );
  }

  return (
    <motion.div
      className={`grid ${responsiveClasses} ${gapClasses[gap]} ${className}`}
      variants={containerVariants}
      initial="initial"
      animate="animate"
      {...props}
    >
      {children.map((child, index) => (
        <motion.div
          key={index}
          custom={index}
          variants={itemVariants}
          initial="initial"
          animate="animate"
          whileHover={{
            scale: 1.02,
            y: -4,
            transition: { 
              duration: 0.2, 
              ease: motionLibrary.presets.easing.smooth 
            }
          }}
          whileTap={{
            scale: 0.98,
            transition: { duration: 0.1 }
          }}
          className="relative"
          style={{ willChange: 'transform' }}
        >
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}

/**
 * AnimatedMetricsList - Specialized for financial metrics with number animations
 */
interface Metric {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  prefix?: string;
  suffix?: string;
}

interface AnimatedMetricsListProps {
  metrics: Metric[];
  className?: string;
  variant?: 'cards' | 'list' | 'compact';
  showTrends?: boolean;
  loading?: boolean;
}

export function AnimatedMetricsList({
  metrics,
  className = '',
  variant = 'cards',
  showTrends = true,
  loading = false,
}: AnimatedMetricsListProps) {
  const containerVariants = {
    initial: { opacity: 0 },
    animate: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      }
    }
  };

  const itemVariants = {
    initial: { opacity: 0, x: -20, scale: 0.95 },
    animate: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: {
        duration: 0.4,
        ease: motionLibrary.presets.easing.smooth,
      }
    }
  };

  const getTrendColor = (trend?: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up': return 'text-emerald-400';
      case 'down': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  const getTrendIcon = (trend?: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  };

  if (loading) {
    return (
      <div className={`space-y-4 ${className}`}>
        {Array.from({ length: metrics.length || 4 }).map((_, index) => (
          <motion.div
            key={index}
            className="glass-surface-secondary p-4 rounded-xl backdrop-blur-md"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1, duration: 0.3 }}
          >
            <div className="flex justify-between items-center">
              <div className="h-4 bg-slate-700/50 rounded w-1/3 loading-shimmer-glass" />
              <div className="h-6 bg-slate-700/40 rounded w-1/4 loading-shimmer-glass" />
            </div>
          </motion.div>
        ))}
      </div>
    );
  }

  return (
    <motion.div
      className={`${variant === 'cards' ? 'grid grid-cols-1 md:grid-cols-2 gap-4' : 'space-y-3'} ${className}`}
      variants={containerVariants}
      initial="initial"
      animate="animate"
    >
      {metrics.map((metric, index) => (
        <motion.div
          key={`${metric.label}-${index}`}
          variants={itemVariants}
          className={`
            glass-surface-secondary backdrop-blur-md rounded-xl p-4 
            hover:shadow-glass-md transition-all duration-300
            ${variant === 'compact' ? 'py-2' : ''}
          `}
          whileHover={{
            scale: 1.02,
            boxShadow: '0 8px 32px rgba(6, 182, 212, 0.15)',
          }}
        >
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-400 font-mono">
              {metric.label}
            </span>
            <div className="flex items-center gap-2">
              <motion.span
                className="text-lg font-bold text-white"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 + 0.3 }}
              >
                {metric.prefix}{metric.value}{metric.suffix}
              </motion.span>
              {showTrends && metric.trend && (
                <motion.span
                  className={`text-sm font-mono ${getTrendColor(metric.trend)}`}
                  initial={{ opacity: 0, rotate: -180 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  transition={{ delay: index * 0.1 + 0.4 }}
                >
                  {getTrendIcon(metric.trend)}
                </motion.span>
              )}
            </div>
          </div>
          {metric.change !== undefined && (
            <motion.div
              className="mt-2 text-xs font-mono"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 + 0.5 }}
            >
              <span className={getTrendColor(metric.trend)}>
                {metric.change > 0 ? '+' : ''}{metric.change.toFixed(2)}%
              </span>
            </motion.div>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
}

export default AnimatedList;