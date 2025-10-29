'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, BarChart3, TrendingUp, RefreshCw, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { motion } from 'framer-motion';

interface VolumeLevel {
  priceLevel: number;
  totalVolume: number;
  barCount: number;
  percentage: number;
}

interface VolumeProfileData {
  levels: VolumeLevel[];
  currentPrice: number;
  pocPrice: number; // Point of Control (highest volume)
  totalVolume: number;
  timestamp: string;
}

export default function VolumeProfileChart() {
  const [volumeData, setVolumeData] = useState<VolumeProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('24h');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchVolumeProfile = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Map time range to hours
      const hoursMap = { '1h': 1, '6h': 6, '24h': 24, '7d': 168 };
      const hours = hoursMap[timeRange];

      const response = await fetch(`/api/market/volume-profile?hours=${hours}`, {
        cache: 'no-store'
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch volume profile');
      }

      setVolumeData(data.data);
      setLastUpdate(new Date());
    } catch (err: any) {
      console.error('[VolumeProfile] Error:', err);
      setError(err.message || 'Failed to load volume profile');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    fetchVolumeProfile();

    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchVolumeProfile, 300000);

    return () => clearInterval(interval);
  }, [fetchVolumeProfile]);

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatVolume = (volume: number) => {
    if (volume === 0) return '0';
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume.toFixed(0);
  };

  // Show loading state
  if (loading && !volumeData) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20 shadow-xl">
        <CardContent className="p-12 text-center">
          <RefreshCw className="h-8 w-8 mx-auto mb-4 animate-spin text-cyan-500" />
          <p className="text-slate-400 font-mono">Loading Volume Profile...</p>
          <p className="text-slate-500 text-sm mt-2">Analyzing market liquidity from OHLCV data</p>
        </CardContent>
      </Card>
    );
  }

  // Show error state (but still display axes)
  const hasData = volumeData && volumeData.levels && volumeData.levels.length > 0;

  // Generate empty data for display if no data
  const displayData = hasData ? volumeData.levels : [
    { priceLevel: 3800, totalVolume: 0, barCount: 0, percentage: 0 },
    { priceLevel: 3850, totalVolume: 0, barCount: 0, percentage: 0 },
    { priceLevel: 3900, totalVolume: 0, barCount: 0, percentage: 0 },
    { priceLevel: 3950, totalVolume: 0, barCount: 0, percentage: 0 },
    { priceLevel: 4000, totalVolume: 0, barCount: 0, percentage: 0 },
  ];

  const maxVolume = hasData
    ? Math.max(...volumeData.levels.map(l => l.totalVolume))
    : 100;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length && hasData) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-900/95 backdrop-blur border border-cyan-500/30 rounded-lg p-3 shadow-xl">
          <p className="text-cyan-400 font-bold font-mono text-sm mb-2">
            {formatPrice(data.priceLevel)}
          </p>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Volume:</span>
              <span className="text-white font-mono">{formatVolume(data.totalVolume)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Bars:</span>
              <span className="text-white font-mono">{data.barCount}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">% of Total:</span>
              <span className="text-green-400 font-mono">{data.percentage.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-slate-900 border-cyan-500/20 shadow-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-cyan-500 font-mono flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-cyan-400" />
              VOLUME PROFILE - USD/COP
            </CardTitle>
            <p className="text-slate-400 text-sm mt-1">
              {hasData
                ? 'Market liquidity distribution by price level'
                : 'No data available - Run pipeline or check date range'}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Time Range Selector */}
            <div className="flex rounded-lg border border-slate-600/50 overflow-hidden">
              {(['1h', '6h', '24h', '7d'] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1.5 text-xs font-mono transition-all ${
                    timeRange === range
                      ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50'
                      : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>

            <button
              onClick={fetchVolumeProfile}
              disabled={loading}
              className="p-2 rounded-lg bg-slate-800/50 border border-slate-600/50 text-slate-400 hover:bg-slate-700/50 hover:text-cyan-400 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>

            {hasData && (
              <Badge className="bg-green-500/20 text-green-400 border-green-500/50">
                LIVE
              </Badge>
            )}
          </div>
        </div>

        {/* Stats Bar */}
        {hasData && (
          <div className="grid grid-cols-4 gap-3 mt-4 bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">
            <div>
              <p className="text-slate-500 text-xs font-mono">Current Price</p>
              <p className="text-white text-lg font-bold font-mono mt-1">
                {formatPrice(volumeData.currentPrice)}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs font-mono">POC Price</p>
              <p className="text-cyan-400 text-lg font-bold font-mono mt-1">
                {formatPrice(volumeData.pocPrice)}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs font-mono">Total Volume</p>
              <p className="text-green-400 text-lg font-bold font-mono mt-1">
                {formatVolume(volumeData.totalVolume)}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs font-mono">Last Update</p>
              <p className="text-slate-400 text-sm font-mono mt-1">
                {lastUpdate.toLocaleTimeString()}
              </p>
            </div>
          </div>
        )}

        {/* Info Banner for No Data */}
        {!hasData && error && (
          <div className="mt-4 bg-yellow-950/30 border border-yellow-500/40 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Info className="h-5 w-5 text-yellow-400 mt-0.5" />
              <div>
                <p className="text-yellow-400 font-semibold text-sm mb-1">No Volume Data Available</p>
                <p className="text-yellow-200 text-xs">
                  {error}
                </p>
                <p className="text-slate-400 text-xs mt-2">
                  <strong>Possible causes:</strong>
                </p>
                <ul className="text-slate-400 text-xs list-disc list-inside ml-2 mt-1">
                  <li>No OHLCV data in selected time range</li>
                  <li>Pipeline not executed (run L0 DAG)</li>
                  <li>Database table empty</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={displayData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis
              type="number"
              stroke="#64748B"
              fontSize={11}
              domain={[0, maxVolume * 1.1]}
              tickFormatter={formatVolume}
            />
            <YAxis
              type="category"
              dataKey="priceLevel"
              stroke="#64748B"
              fontSize={11}
              tickFormatter={(value) => formatPrice(value)}
              width={70}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(34, 211, 238, 0.1)' }} />

            {/* POC Line */}
            {hasData && volumeData.pocPrice && (
              <ReferenceLine
                y={volumeData.pocPrice}
                stroke="#22D3EE"
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{
                  value: 'POC',
                  position: 'right',
                  fill: '#22D3EE',
                  fontSize: 10,
                  fontFamily: 'monospace'
                }}
              />
            )}

            {/* Current Price Line */}
            {hasData && volumeData.currentPrice && (
              <ReferenceLine
                y={volumeData.currentPrice}
                stroke="#F59E0B"
                strokeWidth={2}
                label={{
                  value: 'Current',
                  position: 'right',
                  fill: '#F59E0B',
                  fontSize: 10,
                  fontFamily: 'monospace'
                }}
              />
            )}

            <Bar
              dataKey="totalVolume"
              radius={[0, 4, 4, 0]}
            >
              {displayData.map((entry, index) => {
                const isPOC = hasData && entry.priceLevel === volumeData.pocPrice;
                const isNearCurrent = hasData && Math.abs(entry.priceLevel - volumeData.currentPrice) < 10;

                let fillColor = '#1E40AF'; // Default blue
                if (isPOC) fillColor = '#22D3EE'; // Cyan for POC
                else if (isNearCurrent) fillColor = '#F59E0B'; // Amber near current
                else if (!hasData) fillColor = '#475569'; // Gray for no data

                return <Cell key={`cell-${index}`} fill={fillColor} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-800 rounded" />
            <span className="text-slate-400">Volume</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-cyan-500 rounded" />
            <span className="text-slate-400">POC (Highest Volume)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-amber-500" />
            <span className="text-slate-400">Current Price</span>
          </div>
        </div>

        {/* Footer Info */}
        <div className="text-center mt-6 pt-4 border-t border-slate-700/50">
          <p className="text-slate-500 text-xs font-mono">
            ✅ Real Market Data from usdcop_m5_ohlcv table •
            Volume Profile shows liquidity distribution •
            POC = Point of Control (highest traded volume) •
            {hasData ? `${volumeData.levels.length} price levels analyzed` : 'Awaiting data'}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
