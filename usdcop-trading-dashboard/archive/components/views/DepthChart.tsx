'use client';

import { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Layers } from 'lucide-react';

interface DepthChartProps {
  bids: Array<{ price: number; size: number; total: number }>;
  asks: Array<{ price: number; size: number; total: number }>;
  lastPrice: number;
}

export default function DepthChart({ bids, asks, lastPrice }: DepthChartProps) {

  // Transform data for Recharts
  const chartData = useMemo(() => {
    // Reverse bids to go from lowest to highest
    const bidData = [...bids].reverse().map(bid => ({
      price: bid.price,
      bidDepth: bid.total,
      askDepth: 0,
      type: 'bid'
    }));

    const askData = asks.map(ask => ({
      price: ask.price,
      bidDepth: 0,
      askDepth: ask.total,
      type: 'ask'
    }));

    // Combine and sort by price
    return [...bidData, ...askData].sort((a, b) => a.price - b.price);
  }, [bids, asks]);

  const formatPrice = (value: number) => `$${value.toFixed(2)}`;
  const formatSize = (value: number) => `${(value / 1000).toFixed(1)}k`;

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const isBid = payload[0]?.payload?.type === 'bid';
      const depth = isBid ? payload[0]?.payload?.bidDepth : payload[1]?.payload?.askDepth;

      return (
        <div className="bg-slate-900/95 backdrop-blur border border-cyan-500/30 rounded-lg p-3 shadow-xl">
          <p className="text-slate-400 text-xs font-mono mb-2">
            Price: {formatPrice(label)}
          </p>
          <div className={`text-sm font-mono font-bold ${isBid ? 'text-green-400' : 'text-red-400'}`}>
            {isBid ? 'Bid' : 'Ask'} Depth: {formatSize(depth)}
          </div>
        </div>
      );
    }
    return null;
  };

  if (bids.length === 0 && asks.length === 0) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardContent className="p-12 text-center">
          <Layers className="h-12 w-12 mx-auto mb-4 text-slate-600" />
          <p className="text-slate-500 font-mono">No depth data available</p>
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
              <Layers className="h-5 w-5 text-cyan-400" />
              MARKET DEPTH CHART
            </CardTitle>
            <p className="text-slate-400 text-sm mt-1">
              Cumulative order book visualization
            </p>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full" />
              <span className="text-slate-400">Bids (Buy)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full" />
              <span className="text-slate-400">Asks (Sell)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-1 bg-yellow-500" />
              <span className="text-slate-400">Last: ${lastPrice.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <defs>
              {/* Gradient for Bids (Green) */}
              <linearGradient id="bidGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22C55E" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#22C55E" stopOpacity={0.1} />
              </linearGradient>

              {/* Gradient for Asks (Red) */}
              <linearGradient id="askGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#EF4444" stopOpacity={0.1} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />

            <XAxis
              dataKey="price"
              stroke="#94a3b8"
              fontSize={10}
              tickFormatter={formatPrice}
              tick={{ fill: '#94a3b8' }}
              domain={['dataMin', 'dataMax']}
            />

            <YAxis
              stroke="#94a3b8"
              fontSize={10}
              tickFormatter={formatSize}
              tick={{ fill: '#94a3b8' }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Vertical line at last price */}
            <ReferenceLine
              x={lastPrice}
              stroke="#EAB308"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{
                value: `Last: $${lastPrice.toFixed(2)}`,
                fill: '#EAB308',
                fontSize: 11,
                fontFamily: 'monospace',
                position: 'top'
              }}
            />

            {/* Bid Area (Green) */}
            <Area
              type="stepAfter"
              dataKey="bidDepth"
              stroke="#22C55E"
              strokeWidth={2}
              fill="url(#bidGradient)"
              animationDuration={500}
            />

            {/* Ask Area (Red) */}
            <Area
              type="stepBefore"
              dataKey="askDepth"
              stroke="#EF4444"
              strokeWidth={2}
              fill="url(#askGradient)"
              animationDuration={500}
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* Market Depth Stats */}
        <div className="grid grid-cols-3 gap-3 mt-4 bg-slate-800/30 rounded-lg p-3 border border-slate-700/50">
          <div>
            <p className="text-slate-500 text-xs font-mono">Total Bid Volume</p>
            <p className="text-green-400 text-sm font-bold font-mono mt-1">
              {formatSize(bids[bids.length - 1]?.total || 0)}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Total Ask Volume</p>
            <p className="text-red-400 text-sm font-bold font-mono mt-1">
              {formatSize(asks[asks.length - 1]?.total || 0)}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs font-mono">Bid/Ask Ratio</p>
            <p className="text-cyan-400 text-sm font-bold font-mono mt-1">
              {((bids[bids.length - 1]?.total || 0) / (asks[asks.length - 1]?.total || 1)).toFixed(2)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
