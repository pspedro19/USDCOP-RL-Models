'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, TrendingUp, TrendingDown, Clock, DollarSign, Target } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getWebSocketManager } from '@/lib/services/websocket-manager';

interface Trade {
  trade_id: number;
  timestamp: string;
  strategy_code: string;
  strategy_name: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price: number | null;
  size: number;
  pnl: number;
  pnl_percent: number;
  status: 'open' | 'closed' | 'pending';
  duration_minutes: number | null;
  commission: number;
}

export default function TradeHistory() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'open' | 'closed'>('all');

  // Fetch initial trade history
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch('/api/trading/trades/history?limit=50', {
          cache: 'no-store'
        });

        if (response.ok) {
          const data = await response.json();
          setTrades(data.trades || []);
        }
      } catch (error) {
        console.error('[TradeHistory] Error fetching trades:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();

    // Setup WebSocket for real-time trade updates
    const wsManager = getWebSocketManager();

    const handleTradeUpdate = (trade: Trade) => {
      console.log('[TradeHistory] New trade update:', trade);
      setTrades(prev => {
        const existing = prev.find(t => t.trade_id === trade.trade_id);
        if (existing) {
          // Update existing trade
          return prev.map(t => t.trade_id === trade.trade_id ? trade : t);
        } else {
          // Add new trade at the beginning
          return [trade, ...prev].slice(0, 50); // Keep only last 50
        }
      });
    };

    wsManager.on('trades', handleTradeUpdate);
    wsManager.connect();

    return () => {
      wsManager.off('trades', handleTradeUpdate);
    };
  }, []);

  const filteredTrades = trades.filter(trade => {
    if (filter === 'all') return true;
    return trade.status === filter;
  });

  const stats = {
    total: trades.length,
    open: trades.filter(t => t.status === 'open').length,
    closed: trades.filter(t => t.status === 'closed').length,
    totalPnL: trades.reduce((sum, t) => sum + (t.pnl || 0), 0),
    winRate: trades.filter(t => t.status === 'closed').length > 0
      ? (trades.filter(t => t.status === 'closed' && t.pnl > 0).length /
         trades.filter(t => t.status === 'closed').length) * 100
      : 0
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDuration = (minutes: number | null) => {
    if (!minutes) return '-';
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  if (loading) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardContent className="p-12 text-center">
          <Activity className="h-8 w-8 mx-auto mb-4 animate-pulse text-cyan-500" />
          <p className="text-slate-500 font-mono">Loading Trade History...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-slate-900 border-cyan-500/20 shadow-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-cyan-500 font-mono flex items-center gap-2">
              <Activity className="h-5 w-5 text-cyan-400" />
              TRADE HISTORY
            </CardTitle>
            <p className="text-slate-400 text-sm mt-1">
              Executed operations • Real-time updates
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Badge className="bg-green-500/20 text-green-400 border-green-500/50 animate-pulse">
              LIVE
            </Badge>
          </div>
        </div>

        {/* Stats Summary */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4 bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">
          <div>
            <p className="text-slate-500 text-xs font-mono">Total Trades</p>
            <p className="text-white text-lg font-bold font-mono mt-1">{stats.total}</p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Open</p>
            <p className="text-yellow-400 text-lg font-bold font-mono mt-1">{stats.open}</p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Closed</p>
            <p className="text-slate-300 text-lg font-bold font-mono mt-1">{stats.closed}</p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Total P&L</p>
            <p className={`text-lg font-bold font-mono mt-1 ${stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${stats.totalPnL.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Win Rate</p>
            <p className="text-cyan-400 text-lg font-bold font-mono mt-1">
              {stats.winRate.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Filter Buttons */}
        <div className="flex items-center gap-2 mt-4">
          {(['all', 'open', 'closed'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-1.5 rounded-md text-xs font-mono font-bold transition-all ${
                filter === f
                  ? 'bg-cyan-500 text-slate-950'
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'
              }`}
            >
              {f.toUpperCase()} ({f === 'all' ? stats.total : f === 'open' ? stats.open : stats.closed})
            </button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="p-4">
        {filteredTrades.length === 0 ? (
          <div className="text-center py-12">
            <Target className="h-12 w-12 mx-auto mb-4 text-slate-600" />
            <p className="text-slate-500 font-mono">No trades found</p>
            <p className="text-slate-600 font-mono text-sm mt-1">
              {filter === 'all'
                ? 'Execute strategies to generate trades'
                : `No ${filter} trades available`}
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[500px] overflow-y-auto">
            <AnimatePresence>
              {filteredTrades.map((trade, idx) => (
                <motion.div
                  key={trade.trade_id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ delay: idx * 0.02 }}
                  className={`bg-slate-800/50 rounded-lg p-4 border transition-all hover:bg-slate-800/70 ${
                    trade.status === 'open'
                      ? 'border-yellow-500/30'
                      : trade.pnl >= 0
                        ? 'border-green-500/30'
                        : 'border-red-500/30'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    {/* Left: Trade Info */}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        {/* Side Icon */}
                        {trade.side === 'buy' ? (
                          <TrendingUp className="h-5 w-5 text-green-400" />
                        ) : (
                          <TrendingDown className="h-5 w-5 text-red-400" />
                        )}

                        {/* Strategy Badge */}
                        <Badge className={`font-mono text-xs ${
                          trade.strategy_code.startsWith('RL') ? 'bg-blue-500/20 text-blue-400' :
                          trade.strategy_code.startsWith('ML') ? 'bg-purple-500/20 text-purple-400' :
                          trade.strategy_code.startsWith('LLM') ? 'bg-amber-500/20 text-amber-400' :
                          'bg-slate-500/20 text-slate-400'
                        }`}>
                          {trade.strategy_code}
                        </Badge>

                        {/* Side Badge */}
                        <Badge className={`font-mono text-xs ${
                          trade.side === 'buy'
                            ? 'bg-green-900/40 text-green-400 border-green-500/50'
                            : 'bg-red-900/40 text-red-400 border-red-500/50'
                        }`}>
                          {trade.side.toUpperCase()}
                        </Badge>

                        {/* Status Badge */}
                        <Badge className={`font-mono text-xs ${
                          trade.status === 'open'
                            ? 'bg-yellow-900/40 text-yellow-400 border-yellow-500/50'
                            : 'bg-slate-700/40 text-slate-400 border-slate-600/50'
                        }`}>
                          {trade.status.toUpperCase()}
                        </Badge>
                      </div>

                      {/* Price & Size Info */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3 text-xs font-mono">
                        <div>
                          <p className="text-slate-500">Entry Price</p>
                          <p className="text-white font-bold">${trade.entry_price.toFixed(2)}</p>
                        </div>
                        {trade.exit_price && (
                          <div>
                            <p className="text-slate-500">Exit Price</p>
                            <p className="text-white font-bold">${trade.exit_price.toFixed(2)}</p>
                          </div>
                        )}
                        <div>
                          <p className="text-slate-500">Size</p>
                          <p className="text-white font-bold">{(trade.size * 100).toFixed(0)}%</p>
                        </div>
                        {trade.duration_minutes && (
                          <div>
                            <p className="text-slate-500">Duration</p>
                            <p className="text-white font-bold">{formatDuration(trade.duration_minutes)}</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Right: P&L */}
                    <div className="text-right ml-4">
                      <div className={`text-2xl font-bold font-mono ${
                        trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                      </div>
                      <div className={`text-sm font-mono ${
                        trade.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {trade.pnl_percent >= 0 ? '+' : ''}{trade.pnl_percent.toFixed(2)}%
                      </div>
                      <div className="flex items-center gap-1 text-xs text-slate-500 mt-2">
                        <Clock className="h-3 w-3" />
                        {formatTime(trade.timestamp)}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
