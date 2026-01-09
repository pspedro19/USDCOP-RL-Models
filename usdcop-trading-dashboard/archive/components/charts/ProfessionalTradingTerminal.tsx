'use client';

/**
 * Professional Trading Terminal
 * Complete trading interface with order flow visualization, market depth, and real-time execution
 * Features: Level II quotes, order book, trade execution, position management
 */

import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  Zap,
  DollarSign,
  BarChart3,
  Users,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Settings,
  Maximize2,
  RefreshCw,
  Play,
  Pause,
  Volume2,
  VolumeX,
  ArrowUpCircle,
  ArrowDownCircle,
  Circle,
  Square
} from 'lucide-react';
import dynamic from 'next/dynamic';

// Import the working InteractiveTradingChart component
const InteractiveTradingChart = dynamic(
  () => import('@/components/charts/InteractiveTradingChart').then(mod => ({ default: mod.InteractiveTradingChart })),
  { ssr: false }
);

interface OrderBookEntry {
  price: number;
  size: number;
  orders: number;
  total: number;
}

interface Trade {
  id: string;
  timestamp: number;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  aggressive: boolean;
}

interface Order {
  id: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  price?: number;
  size: number;
  filled: number;
  status: 'pending' | 'partial' | 'filled' | 'cancelled';
  timestamp: number;
}

interface Position {
  symbol: string;
  size: number;
  avgPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  spread: number;
  high24h: number;
  low24h: number;
}

interface TradingTerminalProps {
  marketData: MarketData;
  chartData?: any[]; // Add chart data prop
  onOrderSubmit?: (order: Partial<Order>) => void;
  onOrderCancel?: (orderId: string) => void;
  isConnected?: boolean;
  enableSound?: boolean;
}

export const ProfessionalTradingTerminal: React.FC<TradingTerminalProps> = ({
  marketData,
  chartData = [],
  onOrderSubmit,
  onOrderCancel,
  isConnected = true,
  enableSound = true
}) => {
  // State management
  const [orderBook, setOrderBook] = useState<{ bids: OrderBookEntry[], asks: OrderBookEntry[] }>({
    bids: [],
    asks: []
  });
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [activeOrders, setActiveOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [selectedOrderType, setSelectedOrderType] = useState<'market' | 'limit' | 'stop'>('limit');
  const [orderSize, setOrderSize] = useState(100);
  const [orderPrice, setOrderPrice] = useState(marketData.price);
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'orders' | 'positions' | 'trades'>('orders');
  const [soundEnabled, setSoundEnabled] = useState(enableSound);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Generate mock order book data
  const generateOrderBook = useCallback(() => {
    const spread = marketData.spread || 0.02;
    const midPrice = (marketData.bid + marketData.ask) / 2;
    
    const bids: OrderBookEntry[] = [];
    const asks: OrderBookEntry[] = [];
    
    // Generate bids (below mid price)
    let totalBidSize = 0;
    for (let i = 0; i < 15; i++) {
      const price = midPrice - (i + 1) * spread * 0.1;
      const size = Math.floor(Math.random() * 1000 + 100);
      const orders = Math.floor(Math.random() * 10 + 1);
      totalBidSize += size;
      
      bids.push({
        price,
        size,
        orders,
        total: totalBidSize
      });
    }
    
    // Generate asks (above mid price)
    let totalAskSize = 0;
    for (let i = 0; i < 15; i++) {
      const price = midPrice + (i + 1) * spread * 0.1;
      const size = Math.floor(Math.random() * 1000 + 100);
      const orders = Math.floor(Math.random() * 10 + 1);
      totalAskSize += size;
      
      asks.push({
        price,
        size,
        orders,
        total: totalAskSize
      });
    }
    
    setOrderBook({ bids, asks: asks.reverse() });
  }, [marketData.bid, marketData.ask, marketData.spread]);

  // Generate mock trades
  const generateTrades = useCallback(() => {
    const newTrades: Trade[] = [];
    const now = Date.now();
    
    for (let i = 0; i < 50; i++) {
      const timestamp = now - i * 1000 * Math.random() * 60;
      const side = Math.random() > 0.5 ? 'buy' : 'sell';
      const price = marketData.price + (Math.random() - 0.5) * 0.5;
      const size = Math.floor(Math.random() * 500 + 10);
      
      newTrades.push({
        id: `trade_${i}`,
        timestamp,
        price,
        size,
        side,
        aggressive: Math.random() > 0.3
      });
    }
    
    setRecentTrades(newTrades.sort((a, b) => b.timestamp - a.timestamp));
  }, [marketData.price]);

  // Calculate order book metrics
  const orderBookMetrics = useMemo(() => {
    if (orderBook.bids.length === 0 || orderBook.asks.length === 0) {
      return { bidVolume: 0, askVolume: 0, imbalance: 0, maxSize: 0 };
    }
    
    const bidVolume = orderBook.bids.reduce((sum, bid) => sum + bid.size, 0);
    const askVolume = orderBook.asks.reduce((sum, ask) => sum + ask.size, 0);
    const totalVolume = bidVolume + askVolume;
    const imbalance = totalVolume > 0 ? (bidVolume - askVolume) / totalVolume : 0;
    const maxSize = Math.max(
      ...orderBook.bids.map(b => b.size),
      ...orderBook.asks.map(a => a.size)
    );
    
    return { bidVolume, askVolume, imbalance, maxSize };
  }, [orderBook]);

  // Handle order submission
  const handleOrderSubmit = (side: 'buy' | 'sell') => {
    const order: Partial<Order> = {
      side,
      type: selectedOrderType,
      size: orderSize,
      price: selectedOrderType === 'market' ? undefined : orderPrice,
      timestamp: Date.now()
    };
    
    onOrderSubmit?.(order);
    
    // Add to active orders (mock)
    const newOrder: Order = {
      id: `order_${Date.now()}`,
      ...order,
      filled: 0,
      status: 'pending'
    } as Order;
    
    setActiveOrders(prev => [...prev, newOrder]);
    setShowOrderForm(false);
    
    // Play sound if enabled
    if (soundEnabled) {
      const audio = new Audio('/sounds/order-submitted.mp3');
      audio.play().catch(() => {}); // Ignore errors
    }
  };

  // Initialize data
  useEffect(() => {
    generateOrderBook();
    generateTrades();
    
    if (autoRefresh) {
      const interval = setInterval(() => {
        generateOrderBook();
        generateTrades();
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [generateOrderBook, generateTrades, autoRefresh]);

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatSize = (size: number) => size.toLocaleString();
  const formatTime = (timestamp: number) => new Date(timestamp).toLocaleTimeString();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="grid grid-cols-1 xl:grid-cols-3 gap-6 h-full"
    >
      {/* Left Panel - Market Data & Order Book */}
      <div className="xl:col-span-2 space-y-6">
        {/* Market Data Header */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <motion.h2
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="text-2xl font-bold text-white"
              >
                {marketData.symbol}
              </motion.h2>
              <Badge className={`${isConnected ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSoundEnabled(!soundEnabled)}
                className={`${soundEnabled ? 'text-cyan-400' : 'text-slate-400'} hover:text-white`}
              >
                {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`${autoRefresh ? 'text-emerald-400' : 'text-slate-400'} hover:text-white`}
              >
                {autoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-mono font-bold text-white mb-1">
                {formatPrice(marketData.price)}
              </div>
              <div className={`text-sm font-medium ${marketData.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {marketData.change >= 0 ? '+' : ''}{formatPrice(marketData.change)} ({marketData.changePercent.toFixed(2)}%)
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-emerald-400 font-mono font-bold text-xl">
                {formatPrice(marketData.bid)}
              </div>
              <div className="text-xs text-slate-400">BID</div>
            </div>
            
            <div className="text-center">
              <div className="text-red-400 font-mono font-bold text-xl">
                {formatPrice(marketData.ask)}
              </div>
              <div className="text-xs text-slate-400">ASK</div>
            </div>
            
            <div className="text-center">
              <div className="text-cyan-400 font-mono font-bold text-xl">
                {formatSize(marketData.volume)}
              </div>
              <div className="text-xs text-slate-400">VOLUME</div>
            </div>
          </div>
        </Card>

        {/* Order Book */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              Order Book
            </h3>
            <div className="flex items-center gap-4 text-sm">
              <div className="text-emerald-400">
                Bids: {formatSize(orderBookMetrics.bidVolume)}
              </div>
              <div className="text-red-400">
                Asks: {formatSize(orderBookMetrics.askVolume)}
              </div>
              <div className={`font-mono ${orderBookMetrics.imbalance > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(orderBookMetrics.imbalance * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 h-96 overflow-hidden">
            {/* Asks (Sell Orders) */}
            <div className="space-y-1">
              <div className="grid grid-cols-3 gap-2 text-xs text-slate-400 font-semibold border-b border-slate-700 pb-2">
                <span>Price</span>
                <span className="text-right">Size</span>
                <span className="text-right">Total</span>
              </div>
              <div className="overflow-y-auto h-full space-y-px">
                {orderBook.asks.map((ask, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.01 }}
                    className="grid grid-cols-3 gap-2 text-xs p-1 rounded hover:bg-red-500/10 relative"
                  >
                    <div 
                      className="absolute inset-0 bg-red-500/20 rounded"
                      style={{ width: `${(ask.size / orderBookMetrics.maxSize) * 100}%` }}
                    />
                    <span className="text-red-400 font-mono relative z-10">{formatPrice(ask.price)}</span>
                    <span className="text-white text-right font-mono relative z-10">{formatSize(ask.size)}</span>
                    <span className="text-slate-400 text-right font-mono relative z-10">{formatSize(ask.total)}</span>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Bids (Buy Orders) */}
            <div className="space-y-1">
              <div className="grid grid-cols-3 gap-2 text-xs text-slate-400 font-semibold border-b border-slate-700 pb-2">
                <span>Price</span>
                <span className="text-right">Size</span>
                <span className="text-right">Total</span>
              </div>
              <div className="overflow-y-auto h-full space-y-px">
                {orderBook.bids.map((bid, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.01 }}
                    className="grid grid-cols-3 gap-2 text-xs p-1 rounded hover:bg-emerald-500/10 relative"
                  >
                    <div 
                      className="absolute inset-0 bg-emerald-500/20 rounded"
                      style={{ width: `${(bid.size / orderBookMetrics.maxSize) * 100}%` }}
                    />
                    <span className="text-emerald-400 font-mono relative z-10">{formatPrice(bid.price)}</span>
                    <span className="text-white text-right font-mono relative z-10">{formatSize(bid.size)}</span>
                    <span className="text-slate-400 text-right font-mono relative z-10">{formatSize(bid.total)}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* Professional Trading Chart */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-purple-400" />
            Live Trading Chart
          </h3>
          
          <div className="h-96">
            {chartData && chartData.length > 0 ? (
              <InteractiveTradingChart 
                data={chartData} 
                isRealtime={isConnected}
              />
            ) : (
              <div className="h-full flex items-center justify-center bg-slate-800/30 rounded-lg">
                <div className="text-center text-slate-400">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Loading chart data...</p>
                  <p className="text-xs mt-1">Waiting for market data</p>
                </div>
              </div>
            )}
          </div>
        </Card>
        
        {/* Recent Trades */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-purple-400" />
            Recent Trades
          </h3>
          
          <div className="h-64 overflow-y-auto">
            <div className="grid grid-cols-4 gap-2 text-xs text-slate-400 font-semibold border-b border-slate-700 pb-2 mb-2">
              <span>Time</span>
              <span>Price</span>
              <span className="text-right">Size</span>
              <span className="text-right">Side</span>
            </div>
            
            {recentTrades.slice(0, 50).map((trade) => (
              <motion.div
                key={trade.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-4 gap-2 text-xs p-1 rounded hover:bg-slate-800/30"
              >
                <span className="text-slate-400 font-mono">{formatTime(trade.timestamp)}</span>
                <span className={`font-mono ${trade.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatPrice(trade.price)}
                </span>
                <span className="text-white text-right font-mono">{formatSize(trade.size)}</span>
                <div className="text-right">
                  {trade.side === 'buy' ? (
                    <ArrowUpCircle className="w-3 h-3 text-emerald-400 inline" />
                  ) : (
                    <ArrowDownCircle className="w-3 h-3 text-red-400 inline" />
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </div>

      {/* Right Panel - Trading Interface */}
      <div className="space-y-6">
        {/* Quick Trade */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-yellow-400" />
            Quick Trade
          </h3>
          
          <div className="space-y-4">
            {/* Order Type */}
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Order Type</label>
              <div className="grid grid-cols-3 gap-2">
                {(['market', 'limit', 'stop'] as const).map(type => (
                  <button
                    key={type}
                    onClick={() => setSelectedOrderType(type)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      selectedOrderType === type
                        ? 'bg-cyan-500 text-white'
                        : 'bg-slate-800/50 text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                  >
                    {type.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Price Input */}
            {selectedOrderType !== 'market' && (
              <div>
                <label className="text-sm text-slate-400 mb-2 block">Price</label>
                <input
                  type="number"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(parseFloat(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono"
                  step="0.01"
                />
              </div>
            )}

            {/* Size Input */}
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Size</label>
              <input
                type="number"
                value={orderSize}
                onChange={(e) => setOrderSize(parseInt(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono"
                min="1"
              />
            </div>

            {/* Buy/Sell Buttons */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                onClick={() => handleOrderSubmit('buy')}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-3"
              >
                <ArrowUpCircle className="w-4 h-4 mr-2" />
                BUY {formatPrice(marketData.ask)}
              </Button>
              <Button
                onClick={() => handleOrderSubmit('sell')}
                className="bg-red-600 hover:bg-red-500 text-white font-semibold py-3"
              >
                <ArrowDownCircle className="w-4 h-4 mr-2" />
                SELL {formatPrice(marketData.bid)}
              </Button>
            </div>

            {/* Order Value */}
            <div className="text-center text-sm text-slate-400">
              Order Value: {formatPrice(orderSize * (selectedOrderType === 'market' ? marketData.price : orderPrice))}
            </div>
          </div>
        </Card>

        {/* Account Summary */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <DollarSign className="w-5 h-5 text-emerald-400" />
            Account
          </h3>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-slate-400">Balance</div>
              <div className="text-xl font-mono font-bold text-white">$25,420.50</div>
            </div>
            <div>
              <div className="text-slate-400">Available</div>
              <div className="text-xl font-mono font-bold text-emerald-400">$18,250.30</div>
            </div>
            <div>
              <div className="text-slate-400">P&L Today</div>
              <div className="text-lg font-mono font-bold text-emerald-400">+$2,184.70</div>
            </div>
            <div>
              <div className="text-slate-400">Open Orders</div>
              <div className="text-lg font-mono font-bold text-cyan-400">{activeOrders.length}</div>
            </div>
          </div>
        </Card>

        {/* Orders/Positions/Trades Tabs */}
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50">
          <div className="flex border-b border-slate-700/50">
            {(['orders', 'positions', 'trades'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setSelectedTab(tab)}
                className={`flex-1 px-4 py-3 text-sm font-medium transition-all duration-200 ${
                  selectedTab === tab
                    ? 'text-white border-b-2 border-cyan-500 bg-slate-800/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/20'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          
          <div className="p-6 h-64 overflow-y-auto">
            {selectedTab === 'orders' && (
              <div className="space-y-2">
                {activeOrders.length === 0 ? (
                  <div className="text-center text-slate-400 py-8">
                    <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    No active orders
                  </div>
                ) : (
                  activeOrders.map(order => (
                    <div
                      key={order.id}
                      className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <Badge className={`${order.side === 'buy' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                            {order.side.toUpperCase()}
                          </Badge>
                          <span className="text-white font-mono text-sm">
                            {formatSize(order.size)} @ {order.price ? formatPrice(order.price) : 'Market'}
                          </span>
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          {formatTime(order.timestamp)} • {order.type} • {order.status}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onOrderCancel?.(order.id)}
                        className="text-red-400 hover:text-red-300"
                      >
                        <XCircle className="w-4 h-4" />
                      </Button>
                    </div>
                  ))
                )}
              </div>
            )}
            
            {selectedTab === 'positions' && (
              <div className="text-center text-slate-400 py-8">
                <Circle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                No open positions
              </div>
            )}
            
            {selectedTab === 'trades' && (
              <div className="text-center text-slate-400 py-8">
                <Square className="w-8 h-8 mx-auto mb-2 opacity-50" />
                No recent trades
              </div>
            )}
          </div>
        </Card>
      </div>
    </motion.div>
  );
};

export default ProfessionalTradingTerminal;