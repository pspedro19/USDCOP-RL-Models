"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus, Activity } from "lucide-react";

export interface PriceData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: number;
  bid?: number;
  ask?: number;
  high?: number;
  low?: number;
  timestamp?: number;
  status?: 'live' | 'delayed' | 'offline';
}

export interface PriceTickerProProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  data: PriceData[];
  variant?: 'horizontal' | 'vertical' | 'grid' | 'compact' | 'detailed';
  speed?: 'slow' | 'normal' | 'fast';
  direction?: 'left' | 'right' | 'up' | 'down';
  pauseOnHover?: boolean;
  showVolume?: boolean;
  showBidAsk?: boolean;
  showHighLow?: boolean;
  showChart?: boolean;
  animated?: boolean;
  glowEffect?: boolean;
  terminalStyle?: boolean;
  onSymbolClick?: (symbol: string) => void;
  className?: string;
}

const PriceTickerPro = React.forwardRef<HTMLDivElement, PriceTickerProProps>(
  ({
    data,
    variant = 'horizontal',
    speed = 'normal',
    direction = 'left',
    pauseOnHover = true,
    showVolume = false,
    showBidAsk = false,
    showHighLow = false,
    showChart = false,
    animated = true,
    glowEffect = true,
    terminalStyle = true,
    onSymbolClick,
    className,
    ...props
  }, ref) => {
    const [hoveredSymbol, setHoveredSymbol] = React.useState<string | null>(null);
    const [isPlaying, setIsPlaying] = React.useState(true);

    const speedMap = {
      slow: 60,
      normal: 40,
      fast: 20,
    };

    const directionMap = {
      left: { x: [0, -100] },
      right: { x: [0, 100] },
      up: { y: [0, -100] },
      down: { y: [0, 100] },
    };

    const formatPrice = (price: number, precision = 4) => {
      return price.toFixed(precision);
    };

    const formatChange = (change: number) => {
      return `${change >= 0 ? '+' : ''}${change.toFixed(4)}`;
    };

    const formatPercent = (percent: number) => {
      return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
    };

    const formatVolume = (volume: number) => {
      if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
      if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
      if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
      return volume.toString();
    };

    const getChangeDirection = (change: number) => {
      if (change > 0) return 'up';
      if (change < 0) return 'down';
      return 'neutral';
    };

    const getStatusColor = (status?: string) => {
      switch (status) {
        case 'live': return 'text-status-live shadow-status-live';
        case 'delayed': return 'text-status-delayed shadow-status-delayed';
        case 'offline': return 'text-status-offline shadow-status-offline';
        default: return 'text-cyan-400';
      }
    };

    const TickerItem = ({ item, index }: { item: PriceData; index: number }) => {
      const direction = getChangeDirection(item.change);
      const isHovered = hoveredSymbol === item.symbol;

      const itemVariants = {
        initial: {
          opacity: 0,
          scale: 0.95,
          y: variant === 'vertical' ? 20 : 0,
          x: variant === 'horizontal' ? 20 : 0
        },
        animate: {
          opacity: 1,
          scale: 1,
          y: 0,
          x: 0,
          transition: {
            duration: 0.5,
            delay: index * 0.1,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.05,
          y: -2,
          transition: { duration: 0.2, ease: [0.4, 0, 0.2, 1] }
        },
        tap: {
          scale: 0.98,
          transition: { duration: 0.1 }
        }
      };

      const priceUpdateVariants = {
        update: {
          scale: [1, 1.1, 1],
          boxShadow: direction === 'up'
            ? ['0 0 0px rgba(0, 211, 149, 0)', '0 0 20px rgba(0, 211, 149, 0.6)', '0 0 8px rgba(0, 211, 149, 0.3)']
            : direction === 'down'
            ? ['0 0 0px rgba(255, 59, 105, 0)', '0 0 20px rgba(255, 59, 105, 0.6)', '0 0 8px rgba(255, 59, 105, 0.3)']
            : ['0 0 0px rgba(139, 146, 168, 0)', '0 0 16px rgba(139, 146, 168, 0.4)', '0 0 6px rgba(139, 146, 168, 0.2)'],
          transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
        }
      };

      return (
        <motion.div
          variants={itemVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? "animate" : undefined}
          whileHover={animated && pauseOnHover ? "hover" : undefined}
          whileTap={animated ? "tap" : undefined}
          onHoverStart={() => {
            setHoveredSymbol(item.symbol);
            if (pauseOnHover) setIsPlaying(false);
          }}
          onHoverEnd={() => {
            setHoveredSymbol(null);
            if (pauseOnHover) setIsPlaying(true);
          }}
          onClick={() => onSymbolClick?.(item.symbol)}
          className={cn(
            "relative flex-shrink-0 cursor-pointer group",
            "backdrop-blur-lg border border-slate-600/30",
            "transition-all duration-300 ease-in-out",
            terminalStyle ? "font-mono bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90" : "bg-slate-900/80",
            glowEffect && isHovered && "shadow-glass-lg",
            variant === 'compact' ? "p-3 rounded-xl min-w-[200px]" :
            variant === 'detailed' ? "p-4 rounded-2xl min-w-[280px]" :
            "p-4 rounded-xl min-w-[240px]",
            direction === 'up' && glowEffect && "hover:shadow-market-up",
            direction === 'down' && glowEffect && "hover:shadow-market-down",
            onSymbolClick && "hover:border-cyan-400/50"
          )}
        >
          {/* Status Indicator */}
          {item.status && (
            <motion.div
              className={cn(
                "absolute top-2 right-2 w-2 h-2 rounded-full",
                getStatusColor(item.status)
              )}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.8, 1, 0.8]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          )}

          {/* Gradient Border Effect */}
          <motion.div
            className={cn(
              "absolute inset-0 rounded-inherit bg-gradient-to-r opacity-0 group-hover:opacity-100 transition-opacity duration-500",
              direction === 'up' && "from-market-up/20 via-transparent to-market-up/20",
              direction === 'down' && "from-market-down/20 via-transparent to-market-down/20",
              direction === 'neutral' && "from-cyan-400/20 via-transparent to-cyan-400/20"
            )}
          />

          <div className="relative z-10 space-y-2">
            {/* Symbol and Trend Icon */}
            <div className="flex items-center justify-between">
              <motion.h3
                className={cn(
                  "text-sm font-bold tracking-wider",
                  terminalStyle ? "text-cyan-300" : "text-slate-200",
                  isHovered && "text-cyan-100"
                )}
                whileHover={{ scale: 1.05 }}
              >
                {item.symbol}
              </motion.h3>

              <motion.div
                className={cn(
                  "flex items-center",
                  direction === 'up' && "text-market-up",
                  direction === 'down' && "text-market-down",
                  direction === 'neutral' && "text-market-neutral"
                )}
                animate={{
                  rotate: direction === 'up' ? [0, 5, 0] : direction === 'down' ? [0, -5, 0] : 0
                }}
                transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
              >
                {direction === 'up' && <TrendingUp className="w-4 h-4" />}
                {direction === 'down' && <TrendingDown className="w-4 h-4" />}
                {direction === 'neutral' && <Minus className="w-4 h-4" />}
                {showChart && <Activity className="w-3 h-3 ml-1 opacity-60" />}
              </motion.div>
            </div>

            {/* Price */}
            <motion.div
              className={cn(
                "text-xl font-bold tracking-tight",
                terminalStyle ? "font-mono" : "",
                direction === 'up' && "text-market-up",
                direction === 'down' && "text-market-down",
                direction === 'neutral' && "text-slate-100"
              )}
              variants={priceUpdateVariants}
              animate={animated ? "update" : undefined}
            >
              {formatPrice(item.price)}
            </motion.div>

            {/* Change and Percent */}
            <div className="flex items-center justify-between text-xs space-x-2">
              <motion.span
                className={cn(
                  "font-medium",
                  direction === 'up' && "text-market-up",
                  direction === 'down' && "text-market-down",
                  direction === 'neutral' && "text-market-neutral"
                )}
                animate={{
                  opacity: [0.8, 1, 0.8]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                {formatChange(item.change)}
              </motion.span>

              <motion.span
                className={cn(
                  "font-medium px-1.5 py-0.5 rounded text-xs",
                  direction === 'up' && "bg-market-up/20 text-market-up border border-market-up/30",
                  direction === 'down' && "bg-market-down/20 text-market-down border border-market-down/30",
                  direction === 'neutral' && "bg-market-neutral/20 text-market-neutral border border-market-neutral/30"
                )}
                whileHover={{ scale: 1.05 }}
              >
                {formatPercent(item.changePercent)}
              </motion.span>
            </div>

            {/* Additional Data */}
            {(showBidAsk || showHighLow || showVolume) && (
              <motion.div
                className="space-y-1 text-xs text-slate-400 pt-1 border-t border-slate-600/30"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: isHovered ? 1 : 0.7, height: "auto" }}
                transition={{ duration: 0.3 }}
              >
                {showBidAsk && item.bid && item.ask && (
                  <div className="flex justify-between">
                    <span>Bid: <span className="text-red-400">{formatPrice(item.bid)}</span></span>
                    <span>Ask: <span className="text-green-400">{formatPrice(item.ask)}</span></span>
                  </div>
                )}

                {showHighLow && item.high && item.low && (
                  <div className="flex justify-between">
                    <span>H: <span className="text-green-400">{formatPrice(item.high)}</span></span>
                    <span>L: <span className="text-red-400">{formatPrice(item.low)}</span></span>
                  </div>
                )}

                {showVolume && item.volume && (
                  <div className="text-center">
                    <span>Vol: <span className="text-cyan-400">{formatVolume(item.volume)}</span></span>
                  </div>
                )}
              </motion.div>
            )}
          </div>

          {/* Shimmer Effect */}
          {glowEffect && (
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100"
              animate={{
                x: ['-100%', '100%']
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "linear"
              }}
            />
          )}
        </motion.div>
      );
    };

    const containerVariants = {
      animate: {
        ...(variant === 'horizontal' && {
          x: direction === 'left' ? ['0%', '-100%'] : ['0%', '100%']
        }),
        ...(variant === 'vertical' && {
          y: direction === 'up' ? ['0%', '-100%'] : ['0%', '100%']
        }),
        transition: {
          duration: speedMap[speed],
          repeat: Infinity,
          ease: "linear"
        }
      }
    };

    if (variant === 'grid') {
      return (
        <motion.div
          ref={ref}
          className={cn(
            "grid gap-4 p-4",
            "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
            className
          )}
          {...props}
        >
          <AnimatePresence mode="popLayout">
            {data.map((item, index) => (
              <TickerItem key={item.symbol} item={item} index={index} />
            ))}
          </AnimatePresence>
        </motion.div>
      );
    }

    if (variant === 'vertical') {
      return (
        <motion.div
          ref={ref}
          className={cn(
            "relative overflow-hidden h-full",
            "bg-gradient-to-b from-slate-900/50 via-slate-800/30 to-slate-900/50",
            "backdrop-blur-sm border border-slate-600/30 rounded-2xl",
            className
          )}
          {...props}
        >
          <motion.div
            className="flex flex-col gap-4 p-4"
            variants={containerVariants}
            animate={isPlaying && animated ? "animate" : undefined}
          >
            {[...data, ...data].map((item, index) => (
              <TickerItem key={`${item.symbol}-${index}`} item={item} index={index} />
            ))}
          </motion.div>
        </motion.div>
      );
    }

    return (
      <motion.div
        ref={ref}
        className={cn(
          "relative overflow-hidden",
          "bg-gradient-to-r from-slate-900/50 via-slate-800/30 to-slate-900/50",
          "backdrop-blur-sm border border-slate-600/30 rounded-2xl p-4",
          className
        )}
        {...props}
      >
        <motion.div
          className="flex gap-4 whitespace-nowrap"
          variants={containerVariants}
          animate={isPlaying && animated ? "animate" : undefined}
        >
          {[...data, ...data].map((item, index) => (
            <TickerItem key={`${item.symbol}-${index}`} item={item} index={index} />
          ))}
        </motion.div>
      </motion.div>
    );
  }
);

PriceTickerPro.displayName = "PriceTickerPro";

export { PriceTickerPro };