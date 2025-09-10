'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { fetchLatestPipelineOutput } from '@/lib/services/pipeline';

interface Trade {
  id: string;
  timestamp: string;
  action: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  pnl: number;
  duration_minutes: number;
  confidence: number;
}

interface BacktestResults {
  summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    profit_factor: number;
  };
  equity_curve: { timestamp: string; equity: number; drawdown: number }[];
  trade_ledger: Trade[];
  monthly_returns: { month: string; return_pct: number }[];
}

export default function L6BacktestResults() {
  const [backtestData, setBacktestData] = useState<BacktestResults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchL6Data = async () => {
    try {
      setError(null);
      const pipelineData = await fetchLatestPipelineOutput('L6');
      
      // Mock backtest results
      const mockBacktestData: BacktestResults = {
        summary: {
          total_trades: 1247,
          winning_trades: 776,
          losing_trades: 471,
          win_rate: 0.622,
          total_return: 0.187,
          sharpe_ratio: 1.87,
          max_drawdown: 0.087,
          profit_factor: 1.45,
        },
        equity_curve: Array.from({ length: 30 }, (_, i) => ({
          timestamp: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
          equity: 10000 * (1 + (Math.random() - 0.3) * 0.05 + i * 0.006),
          drawdown: Math.random() * 0.05,
        })),
        trade_ledger: Array.from({ length: 10 }, (_, i) => ({
          id: `trade_${1247 - i}`,
          timestamp: new Date(Date.now() - i * 2 * 60 * 60 * 1000).toISOString(),
          action: i % 3 === 0 ? 'BUY' : 'SELL',
          price: 4200 + Math.random() * 200,
          quantity: Math.floor(Math.random() * 1000) + 100,
          pnl: (Math.random() - 0.4) * 500,
          duration_minutes: Math.floor(Math.random() * 300) + 30,
          confidence: 0.5 + Math.random() * 0.5,
        })),
        monthly_returns: [
          { month: '2025-06', return_pct: 3.2 },
          { month: '2025-07', return_pct: -1.8 },
          { month: '2025-08', return_pct: 5.7 },
        ],
      };
      
      setBacktestData(mockBacktestData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch L6 data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchL6Data();
    const interval = setInterval(fetchL6Data, 300000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return <div className="animate-pulse h-96 bg-gray-200 rounded"></div>;
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">L6 Backtest Results</h2>
      
      {error && <Alert><div className="text-red-600">Error: {error}</div></Alert>}

      {backtestData && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-4 text-center">
              <p className="text-sm text-gray-600">Total Return</p>
              <p className="text-2xl font-bold text-green-600">
                {(backtestData.summary.total_return * 100).toFixed(1)}%
              </p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-sm text-gray-600">Win Rate</p>
              <p className="text-2xl font-bold">{(backtestData.summary.win_rate * 100).toFixed(1)}%</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-sm text-gray-600">Sharpe Ratio</p>
              <p className="text-2xl font-bold">{backtestData.summary.sharpe_ratio.toFixed(2)}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-sm text-gray-600">Max Drawdown</p>
              <p className="text-2xl font-bold text-red-600">
                {(backtestData.summary.max_drawdown * 100).toFixed(1)}%
              </p>
            </Card>
          </div>

          {/* Equity Curve */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={backtestData.equity_curve}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
                  <YAxis />
                  <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} />
                  <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Recent Trades */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">Recent Trades</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Time</th>
                    <th className="text-left p-2">Action</th>
                    <th className="text-left p-2">Price</th>
                    <th className="text-left p-2">Quantity</th>
                    <th className="text-left p-2">P&L</th>
                    <th className="text-left p-2">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestData.trade_ledger.map((trade) => (
                    <tr key={trade.id} className="border-b hover:bg-gray-50">
                      <td className="p-2">{new Date(trade.timestamp).toLocaleString()}</td>
                      <td className="p-2">
                        <Badge className={trade.action === 'BUY' ? 'bg-green-500' : 'bg-red-500'}>
                          {trade.action}
                        </Badge>
                      </td>
                      <td className="p-2">{trade.price.toFixed(2)}</td>
                      <td className="p-2">{trade.quantity}</td>
                      <td className={`p-2 font-bold ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${trade.pnl.toFixed(2)}
                      </td>
                      <td className="p-2">{trade.duration_minutes}m</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}