'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, MinusCircle, Clock, Brain, Cpu, Sparkles, Users, RefreshCw, AlertCircle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/info-tooltip';

interface ModelSignalCardProps {
  strategyCode: string;
  strategyName: string;
  signal: {
    signal: 'long' | 'short' | 'flat' | 'close';
    side: 'buy' | 'sell' | 'hold';
    size: number;
    confidence: number;
    entry_price: number;
    stop_loss: number | null;
    take_profit: number | null;
    risk_usd: number;
    reasoning: string;
    timestamp_utc: string;
  } | undefined;
  color: 'blue' | 'purple' | 'orange' | 'amber' | 'slate';
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export default function ModelSignalCard({
  strategyCode,
  strategyName,
  signal,
  color,
  isLoading = false,
  error = null,
  onRetry
}: ModelSignalCardProps) {

  const colorMap = {
    blue: {
      bg: 'bg-blue-500/10',
      border: 'border-blue-500/30',
      text: 'text-blue-400',
      badge: 'bg-blue-500',
      icon: <Brain className="h-6 w-6" />
    },
    purple: {
      bg: 'bg-purple-500/10',
      border: 'border-purple-500/30',
      text: 'text-purple-400',
      badge: 'bg-purple-500',
      icon: <Cpu className="h-6 w-6" />
    },
    orange: {
      bg: 'bg-orange-500/10',
      border: 'border-orange-500/30',
      text: 'text-orange-400',
      badge: 'bg-orange-500',
      icon: <Cpu className="h-6 w-6" />
    },
    amber: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
      badge: 'bg-amber-500',
      icon: <Sparkles className="h-6 w-6" />
    },
    slate: {
      bg: 'bg-slate-700/10',
      border: 'border-slate-500/30',
      text: 'text-slate-300',
      badge: 'bg-slate-500',
      icon: <Users className="h-6 w-6" />
    }
  };

  const colors = colorMap[color];

  // ========== LOADING STATE ==========
  if (isLoading) {
    return (
      <motion.div
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.2 }}
      >
        <Card className={`${colors.bg} border-2 ${colors.border} h-full`}>
          <CardContent className="flex items-center justify-center py-16">
            <div className="text-center">
              <RefreshCw className={`h-8 w-8 animate-spin ${colors.text} mx-auto mb-3`} />
              <p className="text-slate-400 text-sm font-mono">Loading {strategyName}...</p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  // ========== ERROR STATE ==========
  if (error) {
    return (
      <motion.div
        whileHover={{ scale: 1.02 }}
        transition={{ duration: 0.2 }}
      >
        <Card className="bg-red-950/20 border-2 border-red-500/30 h-full">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="text-red-400">
                  {colors.icon}
                </div>
                <div>
                  <CardTitle className="text-red-400 font-mono text-sm">
                    {strategyName}
                  </CardTitle>
                  <p className="text-slate-500 text-xs font-mono mt-1">
                    {strategyCode}
                  </p>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent className="text-center py-8">
            <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
            <p className="text-red-400 text-sm font-mono mb-1">Error Loading Signal</p>
            <p className="text-slate-500 text-xs mb-4">{error}</p>
            {onRetry && (
              <Button
                onClick={onRetry}
                size="sm"
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                <RefreshCw className="h-3 w-3 mr-2" />
                Retry
              </Button>
            )}
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  const getSignalIcon = () => {
    if (!signal) return <MinusCircle className="h-8 w-8 text-slate-500" />;

    switch (signal.signal) {
      case 'long':
        return <TrendingUp className="h-8 w-8 text-green-400" />;
      case 'short':
        return <TrendingDown className="h-8 w-8 text-red-400" />;
      default:
        return <MinusCircle className="h-8 w-8 text-yellow-400" />;
    }
  };

  const getSignalBadgeColor = () => {
    if (!signal) return 'bg-slate-800 text-slate-400';

    switch (signal.signal) {
      case 'long':
        return 'bg-green-900/40 text-green-400 border-green-500/50';
      case 'short':
        return 'bg-red-900/40 text-red-400 border-red-500/50';
      default:
        return 'bg-yellow-900/40 text-yellow-400 border-yellow-500/50';
    }
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      <Card className={`${colors.bg} border-2 ${colors.border} h-full`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={colors.text}>
                {colors.icon}
              </div>
              <div>
                <CardTitle className={`${colors.text} font-mono text-sm`}>
                  {strategyName}
                </CardTitle>
                <p className="text-slate-500 text-xs font-mono mt-1">
                  {strategyCode}
                </p>
              </div>
            </div>

            <div className={`w-3 h-3 rounded-full ${colors.badge} ${signal ? 'animate-pulse' : ''}`} />
          </div>
        </CardHeader>

        <CardContent className="space-y-4">

          {/* Current Signal */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getSignalIcon()}
              <div>
                <Badge className={`${getSignalBadgeColor()} font-mono border`}>
                  {signal?.signal.toUpperCase() || 'NO SIGNAL'}
                </Badge>
                {signal && (
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    Size: {(signal.size * 100).toFixed(0)}%
                  </p>
                )}
              </div>
            </div>

            {signal && (
              <div className="text-right">
                <div className="text-2xl font-bold text-white font-mono">
                  {(signal.confidence * 100).toFixed(0)}%
                </div>
                <p className="text-xs text-slate-400">Confidence</p>
              </div>
            )}
          </div>

          {/* Price Levels */}
          {signal && signal.entry_price > 0 && (
            <div className="space-y-2 border-t border-slate-700/50 pt-4">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400 font-mono">Entry:</span>
                <span className="text-white font-mono font-bold">${signal.entry_price.toFixed(2)}</span>
              </div>

              {signal.stop_loss && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400 font-mono">Stop Loss:</span>
                  <span className="text-red-400 font-mono">${signal.stop_loss.toFixed(2)}</span>
                </div>
              )}

              {signal.take_profit && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400 font-mono">Take Profit:</span>
                  <span className="text-green-400 font-mono">${signal.take_profit.toFixed(2)}</span>
                </div>
              )}

              <div className="flex justify-between text-sm">
                <span className="text-slate-400 font-mono">Risk:</span>
                <span className="text-white font-mono">${signal.risk_usd.toFixed(2)}</span>
              </div>
            </div>
          )}

          {/* Reasoning */}
          {signal && signal.reasoning && (
            <div className="bg-slate-950/50 rounded-lg p-3 border border-slate-700/30">
              <p className="text-xs text-slate-300 font-mono line-clamp-3">
                {signal.reasoning}
              </p>
            </div>
          )}

          {/* Timestamp */}
          {signal && (
            <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
              <Clock className="h-3 w-3" />
              {new Date(signal.timestamp_utc).toLocaleString()}
            </div>
          )}

          {/* No signal state */}
          {!signal && (
            <div className="text-center py-8">
              <MinusCircle className="h-10 w-10 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500 text-xs font-mono mb-1">
                No Active Signal
              </p>
              <p className="text-slate-600 text-xs">
                Awaiting data from pipeline
              </p>
              {onRetry && (
                <Button
                  onClick={onRetry}
                  size="sm"
                  variant="outline"
                  className="mt-3 border-slate-600 text-slate-400 hover:bg-slate-800"
                >
                  <RefreshCw className="h-3 w-3 mr-2" />
                  Check Now
                </Button>
              )}
            </div>
          )}

        </CardContent>
      </Card>
    </motion.div>
  );
}
