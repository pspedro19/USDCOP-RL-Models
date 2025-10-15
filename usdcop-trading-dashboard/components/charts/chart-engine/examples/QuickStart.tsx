/**
 * Quick Start Example for ChartPro Integration
 * Shows how to integrate ChartPro into existing dashboard
 */

'use client';

import React, { useState, useEffect } from 'react';
import { ChartPro, ChartProData, createOptimalChartConfig } from '../index';
import { CandlestickData, HistogramData, Time } from 'lightweight-charts';

// Mock data generator for USDCOP
const generateUSDCOPData = (days: number = 30): ChartProData => {
  const candlesticks: CandlestickData[] = [];
  const volume: HistogramData[] = [];

  const startTime = Date.now() - (days * 24 * 60 * 60 * 1000);
  const startPrice = 4000; // USDCOP starting price

  let currentPrice = startPrice;

  for (let i = 0; i < days * 24 * 12; i++) { // 5-minute intervals
    const time = (startTime + i * 5 * 60 * 1000) / 1000 as Time;

    // Simulate USDCOP volatility
    const change = (Math.random() - 0.5) * 10; // ±5 COP movement
    currentPrice += change;

    const spread = 2; // 2 COP spread
    const open = currentPrice + (Math.random() - 0.5) * spread;
    const close = currentPrice + (Math.random() - 0.5) * spread;
    const high = Math.max(open, close) + Math.random() * spread;
    const low = Math.min(open, close) - Math.random() * spread;

    candlesticks.push({
      time,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2))
    });

    // Volume data
    volume.push({
      time,
      value: Math.floor(Math.random() * 5000000) + 1000000, // 1M-6M volume
      color: close >= open ? '#26a69a' : '#ef5350'
    });
  }

  return { candlesticks, volume };
};

const QuickStartExample: React.FC = () => {
  const [data, setData] = useState<ChartProData>(() => generateUSDCOPData());
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [isLive, setIsLive] = useState(false);

  // Optimal configuration for USDCOP trading
  const chartConfig = createOptimalChartConfig({
    layout: {
      background: { type: 'solid', color: '#0a0b0f' },
      textColor: '#e5e7eb'
    },
    performance: {
      enableWebGL: true,
      maxDataPoints: 25000,
      updateFrequency: 16
    },
    features: {
      enableDrawingTools: true,
      enableVolumeProfile: true,
      enableTechnicalIndicators: true
    }
  });

  // Simulate live data updates
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      setData(prevData => {
        const lastCandle = prevData.candlesticks[prevData.candlesticks.length - 1];
        const newTime = (lastCandle.time as number + 300) as Time; // 5 minutes later

        // Simulate price movement
        const change = (Math.random() - 0.5) * 8;
        const newPrice = lastCandle.close + change;

        const newCandle: CandlestickData = {
          time: newTime,
          open: lastCandle.close,
          high: Math.max(lastCandle.close, newPrice) + Math.random() * 2,
          low: Math.min(lastCandle.close, newPrice) - Math.random() * 2,
          close: parseFloat(newPrice.toFixed(2))
        };

        const newVolume: HistogramData = {
          time: newTime,
          value: Math.floor(Math.random() * 3000000) + 500000,
          color: newCandle.close >= newCandle.open ? '#26a69a' : '#ef5350'
        };

        return {
          candlesticks: [...prevData.candlesticks.slice(-1000), newCandle],
          volume: [...(prevData.volume || []).slice(-1000), newVolume]
        };
      });
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [isLive]);

  const handleCrosshairMove = (price: number | null, time: Time | null) => {
    setCurrentPrice(price);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between bg-slate-900 p-4 rounded-lg">
        <div>
          <h2 className="text-xl font-bold text-white">USD/COP</h2>
          <p className="text-slate-400">Colombian Peso Exchange Rate</p>
        </div>

        <div className="flex items-center gap-4">
          {currentPrice && (
            <div className="text-right">
              <div className="text-2xl font-bold text-white">
                {currentPrice.toFixed(2)}
              </div>
              <div className="text-sm text-slate-400">COP</div>
            </div>
          )}

          <button
            onClick={() => setIsLive(!isLive)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isLive
                ? 'bg-green-600 text-white hover:bg-green-700'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {isLive ? '● LIVE' : '○ START LIVE'}
          </button>
        </div>
      </div>

      {/* Chart */}
      <ChartPro
        data={data}
        height={600}
        theme="dark"
        config={chartConfig}
        enableDrawingTools={true}
        enableVolumeProfile={true}
        enableIndicators={true}
        enableExport={true}
        onCrosshairMove={handleCrosshairMove}
        className="rounded-lg overflow-hidden border border-slate-700"
      />

      {/* Quick Actions */}
      <div className="flex gap-2 flex-wrap">
        <button className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
          Add Trendline
        </button>
        <button className="px-3 py-1 bg-purple-600 text-white rounded text-sm hover:bg-purple-700">
          Fibonacci
        </button>
        <button className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700">
          Volume Profile
        </button>
        <button className="px-3 py-1 bg-orange-600 text-white rounded text-sm hover:bg-orange-700">
          RSI
        </button>
        <button className="px-3 py-1 bg-slate-600 text-white rounded text-sm hover:bg-slate-700">
          Export PNG
        </button>
      </div>
    </div>
  );
};

export default QuickStartExample;

// Usage in existing dashboard pages:
/*

// pages/trading/charts.tsx
import QuickStartExample from '@/components/charts/chart-engine/examples/QuickStart';

export default function ChartsPage() {
  return (
    <div className="container mx-auto p-6">
      <QuickStartExample />
    </div>
  );
}

// components/views/TradingTerminal.tsx
import { ChartPro, ChartProData } from '@/components/charts/chart-engine';

const TradingTerminal = ({ marketData }) => {
  const chartData: ChartProData = {
    candlesticks: marketData.candlesticks,
    volume: marketData.volume,
    indicators: {
      'SMA 20': calculateSMA(marketData.candlesticks, 20),
      'EMA 12': calculateEMA(marketData.candlesticks, 12)
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-9">
        <ChartPro
          data={chartData}
          height={500}
          theme="dark"
          enableDrawingTools={true}
          enableVolumeProfile={true}
          enableIndicators={true}
        />
      </div>
      <div className="col-span-3">
        // Market info sidebar
      </div>
    </div>
  );
};

*/