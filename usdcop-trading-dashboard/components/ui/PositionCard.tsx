"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  Clock,
  Target,
  AlertTriangle,
  Activity,
  BarChart3,
  Zap,
  Shield,
  X
} from "lucide-react";

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL?: number;
  unrealizedPnLPercent: number;
  leverage?: number;
  marginUsed?: number;
  liquidationPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  openTime: number;
  volume?: number;
  fees?: number;
  status?: 'open' | 'closed' | 'partial';
}

export interface PositionCardProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  position: Position;
  variant?: 'default' | 'compact' | 'detailed' | 'professional' | 'minimal';
  animated?: boolean;
  showChart?: boolean;
  showControls?: boolean;
  showRisk?: boolean;
  realTime?: boolean;
  precision?: number;
  onClose?: (positionId: string) => void;
  onModify?: (positionId: string) => void;
  onAnalyze?: (positionId: string) => void;
  className?: string;
}

const PositionCard = React.forwardRef<HTMLDivElement, PositionCardProps>(
  ({
    position,
    variant = 'default',
    animated = true,
    showChart = false,
    showControls = true,
    showRisk = true,
    realTime = true,
    precision = 4,
    onClose,
    onModify,
    onAnalyze,
    className,
    ...props
  }, ref) => {
    const [isExpanded, setIsExpanded] = React.useState(false);
    const [previousPnL, setPreviousPnL] = React.useState(position.unrealizedPnL);
    const [pnlChanged, setPnlChanged] = React.useState(false);

    React.useEffect(() => {
      if (realTime && position.unrealizedPnL !== previousPnL) {
        setPnlChanged(true);
        const timer = setTimeout(() => setPnlChanged(false), 1000);
        setPreviousPnL(position.unrealizedPnL);
        return () => clearTimeout(timer);
      }
    }, [position.unrealizedPnL, previousPnL, realTime]);

    const formatPrice = (price: number) => {
      return price.toFixed(precision);
    };

    const formatPnL = (pnl: number) => {
      return `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`;
    };

    const formatPercent = (percent: number) => {
      return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
    };

    const formatDuration = (timestamp: number) => {
      const now = Date.now();
      const diff = now - timestamp;
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      return `${hours}h ${minutes}m`;
    };

    const getPnLDirection = () => {
      if (position.unrealizedPnL > 0) return 'profit';
      if (position.unrealizedPnL < 0) return 'loss';
      return 'neutral';
    };

    const getRiskLevel = () => {
      if (!position.liquidationPrice) return 'low';
      const currentPrice = position.currentPrice;
      const liquidationPrice = position.liquidationPrice;
      const entryPrice = position.entryPrice;

      const distanceToLiquidation = position.side === 'long'
        ? (currentPrice - liquidationPrice) / entryPrice
        : (liquidationPrice - currentPrice) / entryPrice;

      if (distanceToLiquidation < 0.05) return 'critical';
      if (distanceToLiquidation < 0.15) return 'high';
      if (distanceToLiquidation < 0.30) return 'medium';
      return 'low';
    };

    const cardVariants = {
      initial: {
        opacity: 0,
        scale: 0.95,
        y: 20
      },
      animate: {
        opacity: 1,
        scale: 1,
        y: 0,
        transition: {
          duration: 0.5,
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
      profit: {
        boxShadow: [
          '0 4px 16px rgba(0, 0, 0, 0.1)',
          '0 8px 32px rgba(0, 211, 149, 0.3)',
          '0 4px 16px rgba(0, 0, 0, 0.1)'
        ],
        transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
      },
      loss: {
        boxShadow: [
          '0 4px 16px rgba(0, 0, 0, 0.1)',
          '0 8px 32px rgba(255, 59, 105, 0.3)',
          '0 4px 16px rgba(0, 0, 0, 0.1)'
        ],
        transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
      }
    };

    const pnlDirection = getPnLDirection();
    const riskLevel = getRiskLevel();

    return (
      <motion.div
        ref={ref}
        variants={cardVariants}
        initial={animated ? "initial" : undefined}
        animate={animated ? (pnlChanged ? pnlDirection : "animate") : undefined}
        whileHover={animated ? "hover" : undefined}
        whileTap={animated ? "tap" : undefined}
        className={cn(
          "relative group cursor-pointer",
          "backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
          "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
          "transition-all duration-300 ease-in-out",
          variant === 'professional' && "shadow-glass-lg",
          variant === 'compact' && "p-3",
          variant === 'detailed' && "p-6",
          variant !== 'compact' && variant !== 'detailed' && "p-4",
          pnlDirection === 'profit' && "hover:border-market-up/40",
          pnlDirection === 'loss' && "hover:border-market-down/40",
          pnlDirection === 'neutral' && "hover:border-cyan-400/40",
          className
        )}
        onClick={() => setIsExpanded(!isExpanded)}
        {...props}
      >
        {/* Risk Level Indicator */}
        {showRisk && position.liquidationPrice && (
          <motion.div
            className={cn(
              "absolute top-2 right-2 w-3 h-3 rounded-full",
              riskLevel === 'critical' && "bg-red-500 shadow-red-500/50",
              riskLevel === 'high' && "bg-orange-500 shadow-orange-500/50",
              riskLevel === 'medium' && "bg-yellow-500 shadow-yellow-500/50",
              riskLevel === 'low' && "bg-green-500 shadow-green-500/50"
            )}
            animate={{
              scale: riskLevel === 'critical' ? [1, 1.2, 1] : 1,
              opacity: riskLevel === 'critical' ? [0.8, 1, 0.8] : 1
            }}
            transition={{
              duration: 1,
              repeat: riskLevel === 'critical' ? Infinity : 0,
              ease: "easeInOut"
            }}
          />
        )}

        {/* Real-time indicator */}
        {realTime && (
          <motion.div
            className="absolute top-2 left-2 flex items-center space-x-1"
            animate={{
              opacity: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <div className="w-2 h-2 rounded-full bg-status-live shadow-status-live" />
          </motion.div>
        )}

        {/* Main Content */}
        <div className="relative z-10">
          {/* Header */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              <motion.div
                className={cn(
                  "flex items-center justify-center w-8 h-8 rounded-xl",
                  position.side === 'long' ? "bg-market-up/20 text-market-up" : "bg-market-down/20 text-market-down"
                )}
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                {position.side === 'long' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              </motion.div>

              <div>
                <motion.h3
                  className={cn(
                    "font-bold text-lg",
                    variant === 'professional' && "font-mono text-glow text-cyan-100"
                  )}
                  whileHover={{ scale: 1.02 }}
                >
                  {position.symbol}
                </motion.h3>

                <motion.p
                  className="text-xs text-slate-400 uppercase tracking-wide"
                  whileHover={{ scale: 1.02 }}
                >
                  {position.side} • {formatDuration(position.openTime)}
                </motion.p>
              </div>
            </div>

            {showControls && (
              <motion.div
                className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 0, x: 0 }}
                whileHover={{ opacity: 1, x: 0 }}
              >
                {onAnalyze && (
                  <motion.button
                    onClick={(e) => { e.stopPropagation(); onAnalyze(position.id); }}
                    className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 transition-colors"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <BarChart3 className="w-3 h-3" />
                  </motion.button>
                )}

                {onModify && (
                  <motion.button
                    onClick={(e) => { e.stopPropagation(); onModify(position.id); }}
                    className="p-1.5 rounded-lg bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Target className="w-3 h-3" />
                  </motion.button>
                )}

                {onClose && (
                  <motion.button
                    onClick={(e) => { e.stopPropagation(); onClose(position.id); }}
                    className="p-1.5 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <X className="w-3 h-3" />
                  </motion.button>
                )}
              </motion.div>
            )}
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            {/* Entry Price */}
            <div className="space-y-1">
              <p className="text-xs text-slate-400 uppercase tracking-wide">Entry</p>
              <motion.p
                className={cn(
                  "font-bold tabular-nums",
                  variant === 'professional' && "font-mono text-cyan-300"
                )}
                whileHover={{ scale: 1.02 }}
              >
                {formatPrice(position.entryPrice)}
              </motion.p>
            </div>

            {/* Current Price */}
            <div className="space-y-1">
              <p className="text-xs text-slate-400 uppercase tracking-wide">Current</p>
              <motion.p
                className={cn(
                  "font-bold tabular-nums",
                  variant === 'professional' && "font-mono text-cyan-300"
                )}
                animate={pnlChanged ? {
                  scale: [1, 1.05, 1],
                  color: pnlDirection === 'profit' ? '#00D395' : pnlDirection === 'loss' ? '#FF3B69' : '#CBD5E1'
                } : {}}
                transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
              >
                {formatPrice(position.currentPrice)}
              </motion.p>
            </div>

            {/* Size */}
            <div className="space-y-1">
              <p className="text-xs text-slate-400 uppercase tracking-wide">Size</p>
              <motion.p
                className={cn(
                  "font-bold tabular-nums",
                  variant === 'professional' && "font-mono"
                )}
                whileHover={{ scale: 1.02 }}
              >
                {position.size.toFixed(4)}
              </motion.p>
            </div>

            {/* Leverage */}
            {position.leverage && (
              <div className="space-y-1">
                <p className="text-xs text-slate-400 uppercase tracking-wide">Leverage</p>
                <motion.p
                  className={cn(
                    "font-bold tabular-nums",
                    variant === 'professional' && "font-mono"
                  )}
                  whileHover={{ scale: 1.02 }}
                >
                  {position.leverage}x
                </motion.p>
              </div>
            )}
          </div>

          {/* P&L Section */}
          <motion.div
            className={cn(
              "p-4 rounded-xl border",
              pnlDirection === 'profit' && "bg-market-up/10 border-market-up/30",
              pnlDirection === 'loss' && "bg-market-down/10 border-market-down/30",
              pnlDirection === 'neutral' && "bg-slate-700/20 border-slate-600/30"
            )}
            animate={pnlChanged ? {
              scale: [1, 1.02, 1],
              borderColor: pnlDirection === 'profit'
                ? ['rgba(0, 211, 149, 0.3)', 'rgba(0, 211, 149, 0.6)', 'rgba(0, 211, 149, 0.3)']
                : pnlDirection === 'loss'
                ? ['rgba(255, 59, 105, 0.3)', 'rgba(255, 59, 105, 0.6)', 'rgba(255, 59, 105, 0.3)']
                : ['rgba(139, 146, 168, 0.3)', 'rgba(139, 146, 168, 0.6)', 'rgba(139, 146, 168, 0.3)']
            } : {}}
            transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <motion.div
                  className={cn(
                    "p-1.5 rounded-lg",
                    pnlDirection === 'profit' && "bg-market-up/20 text-market-up",
                    pnlDirection === 'loss' && "bg-market-down/20 text-market-down",
                    pnlDirection === 'neutral' && "bg-slate-600/20 text-slate-400"
                  )}
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  <DollarSign className="w-4 h-4" />
                </motion.div>

                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-wide">Unrealized P&L</p>
                  <motion.p
                    className={cn(
                      "text-xl font-bold tabular-nums",
                      pnlDirection === 'profit' && "text-market-up",
                      pnlDirection === 'loss' && "text-market-down",
                      pnlDirection === 'neutral' && "text-slate-300",
                      variant === 'professional' && "text-glow"
                    )}
                    animate={pnlChanged ? {
                      scale: [1, 1.1, 1],
                      textShadow: pnlDirection === 'profit'
                        ? ['0 0 0px rgba(0, 211, 149, 0)', '0 0 16px rgba(0, 211, 149, 0.8)', '0 0 8px rgba(0, 211, 149, 0.4)']
                        : pnlDirection === 'loss'
                        ? ['0 0 0px rgba(255, 59, 105, 0)', '0 0 16px rgba(255, 59, 105, 0.8)', '0 0 8px rgba(255, 59, 105, 0.4)']
                        : ['0 0 0px rgba(139, 146, 168, 0)', '0 0 12px rgba(139, 146, 168, 0.6)', '0 0 6px rgba(139, 146, 168, 0.3)']
                    } : {}}
                    transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
                  >
                    ${formatPnL(position.unrealizedPnL)}
                  </motion.p>
                </div>
              </div>

              <motion.div
                className={cn(
                  "px-3 py-1.5 rounded-lg text-sm font-bold",
                  pnlDirection === 'profit' && "bg-market-up/20 text-market-up border border-market-up/30",
                  pnlDirection === 'loss' && "bg-market-down/20 text-market-down border border-market-down/30",
                  pnlDirection === 'neutral' && "bg-slate-600/20 text-slate-400 border border-slate-600/30"
                )}
                whileHover={{ scale: 1.05 }}
              >
                {formatPercent(position.unrealizedPnLPercent)}
              </motion.div>
            </div>
          </motion.div>

          {/* Expanded Details */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginTop: 0 }}
                animate={{ opacity: 1, height: "auto", marginTop: 16 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
                className="space-y-4"
              >
                {/* Risk Metrics */}
                {showRisk && position.liquidationPrice && (
                  <motion.div
                    className="p-4 rounded-xl bg-slate-800/50 border border-slate-600/30"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                  >
                    <div className="flex items-center space-x-2 mb-3">
                      <Shield className="w-4 h-4 text-amber-400" />
                      <h4 className="text-sm font-bold text-amber-400 uppercase tracking-wide">Risk Management</h4>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-slate-400">Liquidation Price</p>
                        <p className="font-bold text-red-400 tabular-nums">{formatPrice(position.liquidationPrice)}</p>
                      </div>

                      {position.marginUsed && (
                        <div>
                          <p className="text-slate-400">Margin Used</p>
                          <p className="font-bold tabular-nums">${position.marginUsed.toFixed(2)}</p>
                        </div>
                      )}

                      {position.stopLoss && (
                        <div>
                          <p className="text-slate-400">Stop Loss</p>
                          <p className="font-bold text-red-400 tabular-nums">{formatPrice(position.stopLoss)}</p>
                        </div>
                      )}

                      {position.takeProfit && (
                        <div>
                          <p className="text-slate-400">Take Profit</p>
                          <p className="font-bold text-green-400 tabular-nums">{formatPrice(position.takeProfit)}</p>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}

                {/* Additional Metrics */}
                <motion.div
                  className="grid grid-cols-2 gap-4 text-sm"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  {position.volume && (
                    <div>
                      <p className="text-slate-400">Volume</p>
                      <p className="font-bold tabular-nums">{position.volume.toFixed(2)}</p>
                    </div>
                  )}

                  {position.fees && (
                    <div>
                      <p className="text-slate-400">Fees</p>
                      <p className="font-bold tabular-nums">${position.fees.toFixed(2)}</p>
                    </div>
                  )}

                  {position.realizedPnL !== undefined && (
                    <div>
                      <p className="text-slate-400">Realized P&L</p>
                      <p className={cn(
                        "font-bold tabular-nums",
                        position.realizedPnL >= 0 ? "text-market-up" : "text-market-down"
                      )}>
                        ${formatPnL(position.realizedPnL)}
                      </p>
                    </div>
                  )}

                  <div>
                    <p className="text-slate-400">Status</p>
                    <motion.p
                      className={cn(
                        "font-bold uppercase text-xs px-2 py-1 rounded inline-block",
                        position.status === 'open' && "bg-green-500/20 text-green-400",
                        position.status === 'closed' && "bg-slate-500/20 text-slate-400",
                        position.status === 'partial' && "bg-amber-500/20 text-amber-400"
                      )}
                      whileHover={{ scale: 1.05 }}
                    >
                      {position.status || 'open'}
                    </motion.p>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Background Gradient Effect */}
        <motion.div
          className={cn(
            "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none",
            "bg-gradient-to-br",
            pnlDirection === 'profit' && "from-market-up/5 via-transparent to-market-up/10",
            pnlDirection === 'loss' && "from-market-down/5 via-transparent to-market-down/10",
            pnlDirection === 'neutral' && "from-cyan-400/5 via-transparent to-cyan-400/10"
          )}
        />

        {/* Shimmer Effect */}
        {pnlChanged && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent pointer-events-none"
            initial={{ x: '-100%' }}
            animate={{ x: '100%' }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
        )}
      </motion.div>
    );
  }
);

PositionCard.displayName = "PositionCard";

export { PositionCard };