"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Volume2, Zap, BarChart3 } from "lucide-react";

export interface OrderbookEntry {
  price: number;
  size: number;
  total: number;
  orders?: number;
  side: 'bid' | 'ask';
}

export interface OrderbookData {
  bids: OrderbookEntry[];
  asks: OrderbookEntry[];
  spread: number;
  spreadPercent: number;
  lastPrice?: number;
  timestamp?: number;
}

export interface OrderbookVisualProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  data: OrderbookData;
  variant?: 'standard' | 'depth' | 'heatmap' | 'professional' | 'compact';
  maxLevels?: number;
  showDepthChart?: boolean;
  showSpread?: boolean;
  showTotals?: boolean;
  showOrderCount?: boolean;
  precision?: number;
  sizeFormat?: 'decimal' | 'abbreviated' | 'percentage';
  animated?: boolean;
  realTime?: boolean;
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void;
  className?: string;
}

const OrderbookVisual = React.forwardRef<HTMLDivElement, OrderbookVisualProps>(
  ({
    data,
    variant = 'standard',
    maxLevels = 10,
    showDepthChart = true,
    showSpread = true,
    showTotals = true,
    showOrderCount = false,
    precision = 4,
    sizeFormat = 'abbreviated',
    animated = true,
    realTime = true,
    onPriceClick,
    className,
    ...props
  }, ref) => {
    const [hoveredLevel, setHoveredLevel] = React.useState<number | null>(null);
    const [maxBidSize, setMaxBidSize] = React.useState(0);
    const [maxAskSize, setMaxAskSize] = React.useState(0);
    const [previousData, setPreviousData] = React.useState<OrderbookData | null>(null);

    React.useEffect(() => {
      if (data.bids.length > 0) {
        setMaxBidSize(Math.max(...data.bids.slice(0, maxLevels).map(b => b.total)));
      }
      if (data.asks.length > 0) {
        setMaxAskSize(Math.max(...data.asks.slice(0, maxLevels).map(a => a.total)));
      }

      if (realTime) {
        setPreviousData(data);
      }
    }, [data, maxLevels, realTime]);

    const formatPrice = (price: number) => {
      return price.toFixed(precision);
    };

    const formatSize = (size: number) => {
      switch (sizeFormat) {
        case 'abbreviated':
          if (size >= 1e9) return `${(size / 1e9).toFixed(1)}B`;
          if (size >= 1e6) return `${(size / 1e6).toFixed(1)}M`;
          if (size >= 1e3) return `${(size / 1e3).toFixed(1)}K`;
          return size.toFixed(2);
        case 'percentage':
          const total = Math.max(maxBidSize, maxAskSize);
          return `${((size / total) * 100).toFixed(1)}%`;
        default:
          return size.toFixed(2);
      }
    };

    const formatSpread = (spread: number, spreadPercent: number) => {
      return `${spread.toFixed(precision)} (${spreadPercent.toFixed(2)}%)`;
    };

    const getDepthPercentage = (total: number, side: 'bid' | 'ask') => {
      const maxSize = side === 'bid' ? maxBidSize : maxAskSize;
      return maxSize > 0 ? (total / maxSize) * 100 : 0;
    };

    const hasChanged = (current: OrderbookEntry, previous?: OrderbookEntry) => {
      if (!previous || !realTime) return false;
      return current.price !== previous.price || current.size !== previous.size;
    };

    const OrderbookLevel = ({
      entry,
      index,
      side,
      previousEntry
    }: {
      entry: OrderbookEntry;
      index: number;
      side: 'bid' | 'ask';
      previousEntry?: OrderbookEntry;
    }) => {
      const depthPercentage = getDepthPercentage(entry.total, side);
      const isHovered = hoveredLevel === index;
      const changed = hasChanged(entry, previousEntry);

      const levelVariants = {
        initial: {
          opacity: 0,
          x: side === 'bid' ? -20 : 20,
          scale: 0.95
        },
        animate: {
          opacity: 1,
          x: 0,
          scale: 1,
          transition: {
            duration: 0.3,
            delay: index * 0.02,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          scale: 1.02,
          x: side === 'bid' ? 4 : -4,
          transition: { duration: 0.2 }
        },
        updated: {
          scale: [1, 1.05, 1],
          backgroundColor: side === 'bid'
            ? ['rgba(0, 211, 149, 0.1)', 'rgba(0, 211, 149, 0.3)', 'rgba(0, 211, 149, 0.1)']
            : ['rgba(255, 59, 105, 0.1)', 'rgba(255, 59, 105, 0.3)', 'rgba(255, 59, 105, 0.1)'],
          transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] }
        }
      };

      const depthBarVariants = {
        initial: { width: 0, opacity: 0 },
        animate: {
          width: `${depthPercentage}%`,
          opacity: variant === 'heatmap' ? 0.8 : 0.3,
          transition: {
            duration: 0.5,
            delay: index * 0.05,
            ease: [0.4, 0, 0.2, 1]
          }
        },
        hover: {
          opacity: variant === 'heatmap' ? 1 : 0.5,
          transition: { duration: 0.2 }
        }
      };

      return (
        <motion.div
          variants={levelVariants}
          initial={animated ? "initial" : undefined}
          animate={animated ? (changed ? "updated" : "animate") : undefined}
          whileHover={animated ? "hover" : undefined}
          onHoverStart={() => setHoveredLevel(index)}
          onHoverEnd={() => setHoveredLevel(null)}
          onClick={() => onPriceClick?.(entry.price, side)}
          className={cn(
            "relative flex items-center justify-between p-2 cursor-pointer group",
            "hover:bg-slate-800/50 transition-colors duration-200",
            "border-l-2 border-transparent",
            side === 'bid' && "hover:border-l-market-up",
            side === 'ask' && "hover:border-l-market-down",
            variant === 'professional' && "font-mono text-sm",
            variant === 'compact' && "p-1 text-xs"
          )}
        >
          {/* Depth Bar Background */}
          <motion.div
            variants={depthBarVariants}
            initial={animated ? "initial" : undefined}
            animate={animated ? "animate" : undefined}
            whileHover={animated ? "hover" : undefined}
            className={cn(
              "absolute inset-y-0 pointer-events-none",
              side === 'bid' ? "left-0 bg-market-up/20" : "right-0 bg-market-down/20",
              variant === 'heatmap' && side === 'bid' && "bg-gradient-to-r from-market-up/30 to-transparent",
              variant === 'heatmap' && side === 'ask' && "bg-gradient-to-l from-market-down/30 to-transparent"
            )}
          />

          {/* Level Content */}
          <div className={cn(
            "relative z-10 flex items-center justify-between w-full",
            side === 'ask' && "flex-row-reverse"
          )}>
            {/* Price */}
            <motion.span
              className={cn(
                "font-medium tabular-nums",
                side === 'bid' ? "text-market-up" : "text-market-down",
                isHovered && "font-bold",
                variant === 'professional' && "text-glow"
              )}
              whileHover={{ scale: 1.05 }}
            >
              {formatPrice(entry.price)}
            </motion.span>

            {/* Size */}
            <motion.span
              className={cn(
                "text-slate-300 tabular-nums",
                isHovered && "text-white font-medium"
              )}
              whileHover={{ scale: 1.02 }}
            >
              {formatSize(entry.size)}
            </motion.span>

            {/* Total (if enabled) */}
            {showTotals && (
              <motion.span
                className={cn(
                  "text-slate-400 text-xs tabular-nums",
                  isHovered && "text-slate-200"
                )}
                whileHover={{ scale: 1.02 }}
              >
                {formatSize(entry.total)}
              </motion.span>
            )}

            {/* Order Count (if enabled) */}
            {showOrderCount && entry.orders && (
              <motion.span
                className={cn(
                  "text-slate-500 text-xs",
                  isHovered && "text-slate-300"
                )}
                whileHover={{ scale: 1.02 }}
              >
                ({entry.orders})
              </motion.span>
            )}
          </div>

          {/* Hover Glow Effect */}
          {isHovered && (
            <motion.div
              className={cn(
                "absolute inset-0 pointer-events-none rounded",
                side === 'bid' ? "shadow-market-up/20" : "shadow-market-down/20"
              )}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
          )}
        </motion.div>
      );
    };

    const SpreadIndicator = () => {
      if (!showSpread) return null;

      return (
        <motion.div
          className={cn(
            "flex items-center justify-center p-3 mx-2 my-2",
            "bg-gradient-to-r from-slate-800/50 via-slate-700/50 to-slate-800/50",
            "border border-slate-600/30 rounded-xl backdrop-blur-sm",
            variant === 'professional' && "font-mono"
          )}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          <div className="flex items-center space-x-2">
            <motion.div
              className="flex items-center text-slate-400"
              whileHover={{ scale: 1.05 }}
            >
              <BarChart3 className="w-4 h-4 mr-1" />
              <span className="text-xs font-medium">SPREAD</span>
            </motion.div>

            <motion.span
              className={cn(
                "text-amber-400 font-bold tabular-nums",
                variant === 'professional' && "text-glow"
              )}
              animate={{
                scale: [1, 1.05, 1],
                opacity: [0.8, 1, 0.8]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              {formatSpread(data.spread, data.spreadPercent)}
            </motion.span>

            {data.lastPrice && (
              <motion.div
                className="flex items-center text-cyan-400 text-xs"
                whileHover={{ scale: 1.05 }}
              >
                <Zap className="w-3 h-3 mr-1" />
                <span className="tabular-nums">{formatPrice(data.lastPrice)}</span>
              </motion.div>
            )}
          </div>
        </motion.div>
      );
    };

    const DepthChart = () => {
      if (!showDepthChart) return null;

      const maxDepth = Math.max(maxBidSize, maxAskSize);
      const bidLevels = data.bids.slice(0, maxLevels);
      const askLevels = data.asks.slice(0, maxLevels);

      return (
        <motion.div
          className="h-32 p-4 border-t border-slate-600/30"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 128 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className="flex items-end justify-center h-full space-x-1">
            {/* Bids */}
            {bidLevels.reverse().map((bid, index) => (
              <motion.div
                key={`bid-${index}`}
                className="bg-market-up/40 border-t border-market-up/60 min-w-[4px] flex-1"
                style={{ height: `${(bid.total / maxDepth) * 100}%` }}
                initial={{ height: 0 }}
                animate={{ height: `${(bid.total / maxDepth) * 100}%` }}
                transition={{ duration: 0.5, delay: index * 0.02 }}
                whileHover={{
                  backgroundColor: 'rgba(0, 211, 149, 0.6)',
                  scale: 1.05
                }}
              />
            ))}

            {/* Spread Gap */}
            <div className="w-2 bg-gradient-to-t from-amber-500/20 to-transparent" />

            {/* Asks */}
            {askLevels.map((ask, index) => (
              <motion.div
                key={`ask-${index}`}
                className="bg-market-down/40 border-t border-market-down/60 min-w-[4px] flex-1"
                style={{ height: `${(ask.total / maxDepth) * 100}%` }}
                initial={{ height: 0 }}
                animate={{ height: `${(ask.total / maxDepth) * 100}%` }}
                transition={{ duration: 0.5, delay: index * 0.02 }}
                whileHover={{
                  backgroundColor: 'rgba(255, 59, 105, 0.6)',
                  scale: 1.05
                }}
              />
            ))}
          </div>
        </motion.div>
      );
    };

    const containerVariants = {
      initial: { opacity: 0, y: 20 },
      animate: {
        opacity: 1,
        y: 0,
        transition: {
          duration: 0.5,
          staggerChildren: 0.02,
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
          "relative",
          "bg-gradient-to-b from-slate-900/80 via-slate-800/60 to-slate-900/90",
          "backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
          variant === 'professional' && "shadow-glass-lg",
          variant === 'compact' && "text-sm",
          className
        )}
        {...props}
      >
        {/* Header */}
        <motion.div
          className={cn(
            "flex items-center justify-between p-4 border-b border-slate-600/30",
            "bg-gradient-to-r from-slate-800/50 to-transparent"
          )}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="flex items-center space-x-2">
            <Volume2 className="w-5 h-5 text-cyan-400" />
            <h3 className={cn(
              "font-bold text-cyan-100",
              variant === 'professional' && "font-mono text-glow"
            )}>
              Order Book
            </h3>
          </div>

          <motion.div
            className="flex items-center space-x-4 text-xs text-slate-400"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center space-x-1">
              <TrendingUp className="w-3 h-3 text-market-up" />
              <span>Bids</span>
            </div>
            <div className="flex items-center space-x-1">
              <TrendingDown className="w-3 h-3 text-market-down" />
              <span>Asks</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Column Headers */}
        <motion.div
          className={cn(
            "grid grid-cols-3 gap-4 p-3 text-xs text-slate-400 font-medium border-b border-slate-600/20",
            showTotals && "grid-cols-4",
            variant === 'professional' && "font-mono"
          )}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <div>Price</div>
          <div className="text-center">Size</div>
          {showTotals && <div className="text-center">Total</div>}
          <div className="text-right">Depth</div>
        </motion.div>

        {/* Order Book Content */}
        <div className="relative max-h-96 overflow-y-auto">
          {/* Asks (reversed order - highest first) */}
          <div className="space-y-0.5">
            <AnimatePresence mode="popLayout">
              {data.asks.slice(0, maxLevels).reverse().map((ask, index) => (
                <OrderbookLevel
                  key={`ask-${ask.price}`}
                  entry={ask}
                  index={index}
                  side="ask"
                  previousEntry={previousData?.asks.find(a => a.price === ask.price)}
                />
              ))}
            </AnimatePresence>
          </div>

          {/* Spread Indicator */}
          <SpreadIndicator />

          {/* Bids */}
          <div className="space-y-0.5">
            <AnimatePresence mode="popLayout">
              {data.bids.slice(0, maxLevels).map((bid, index) => (
                <OrderbookLevel
                  key={`bid-${bid.price}`}
                  entry={bid}
                  index={index}
                  side="bid"
                  previousEntry={previousData?.bids.find(b => b.price === bid.price)}
                />
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Depth Chart */}
        <DepthChart />

        {/* Real-time indicator */}
        {realTime && (
          <motion.div
            className="absolute top-4 right-4 flex items-center space-x-1"
            animate={{
              opacity: [0.5, 1, 0.5],
              scale: [1, 1.1, 1]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <div className="w-2 h-2 rounded-full bg-status-live shadow-status-live" />
            <span className="text-xs text-status-live font-medium">LIVE</span>
          </motion.div>
        )}
      </motion.div>
    );
  }
);

OrderbookVisual.displayName = "OrderbookVisual";

export { OrderbookVisual };