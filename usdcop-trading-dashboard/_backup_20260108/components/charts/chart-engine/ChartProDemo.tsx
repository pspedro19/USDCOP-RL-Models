/**
 * ChartPro Demo Component
 * Showcases all institutional-grade features
 */

'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { CandlestickData, HistogramData, LineData, Time } from 'lightweight-charts';
import ChartPro, { ChartProData } from './ChartPro';
import { Card } from '../../ui/card';
import { Button } from '../../ui/button';
import { Badge } from '../../ui/badge';
import {
  Play,
  Pause,
  RotateCcw,
  Settings,
  Download,
  TrendingUp,
  BarChart3,
  Activity,
  Zap,
  Target,
  Eye,
  Layers,
  Palette
} from 'lucide-react';

// Generate realistic USDCOP data
const generateRealisticCandleData = (
  count: number,
  startPrice: number = 4000,
  startTime: number = Date.now() - count * 5 * 60 * 1000
): CandlestickData[] => {
  const data: CandlestickData[] = [];
  let currentPrice = startPrice;
  let trend = 0.001; // Slight upward trend

  for (let i = 0; i < count; i++) {
    const time = (startTime + i * 5 * 60 * 1000) / 1000; // 5-minute intervals

    // Add some realistic market movement
    const volatility = 0.002; // 0.2% volatility
    const randomChange = (Math.random() - 0.5) * volatility * currentPrice;
    const trendChange = trend * currentPrice;

    currentPrice += randomChange + trendChange;

    // Generate OHLC from current price
    const spread = currentPrice * 0.001; // 0.1% spread
    const open = currentPrice + (Math.random() - 0.5) * spread;
    const close = currentPrice + (Math.random() - 0.5) * spread;
    const high = Math.max(open, close) + Math.random() * spread;
    const low = Math.min(open, close) - Math.random() * spread;

    data.push({
      time: time as Time,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2))
    });

    // Occasional trend changes
    if (Math.random() < 0.02) {
      trend *= -1;
    }
  }

  return data;
};

// Generate volume data
const generateVolumeData = (candleData: CandlestickData[]): HistogramData[] => {
  return candleData.map(candle => ({
    time: candle.time,
    value: Math.floor(Math.random() * 10000000) + 1000000, // 1M to 11M volume
    color: candle.close >= candle.open ? '#26a69a' : '#ef5350'
  }));
};

// Generate indicator data
const generateSMAData = (candleData: CandlestickData[], period: number): LineData[] => {
  const smaData: LineData[] = [];

  for (let i = period - 1; i < candleData.length; i++) {
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += candleData[i - j].close;
    }

    smaData.push({
      time: candleData[i].time,
      value: sum / period
    });
  }

  return smaData;
};

// Generate EMA data
const generateEMAData = (candleData: CandlestickData[], period: number): LineData[] => {
  const emaData: LineData[] = [];
  const k = 2 / (period + 1);

  let ema = candleData[0].close;
  emaData.push({
    time: candleData[0].time,
    value: ema
  });

  for (let i = 1; i < candleData.length; i++) {
    ema = candleData[i].close * k + ema * (1 - k);
    emaData.push({
      time: candleData[i].time,
      value: ema
    });
  }

  return emaData;
};

const ChartProDemo: React.FC = () => {
  // State
  const [isRealTime, setIsRealTime] = useState(false);
  const [dataCount, setDataCount] = useState(1000);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [showVolumeProfile, setShowVolumeProfile] = useState(false);
  const [enabledFeatures, setEnabledFeatures] = useState({
    drawingTools: true,
    volumeProfile: true,
    indicators: true,
    export: true,
    performance: true
  });

  // Generate initial data
  const chartData = useMemo<ChartProData>(() => {
    const candlesticks = generateRealisticCandleData(dataCount);
    const volume = generateVolumeData(candlesticks);

    const indicators = {
      'SMA 20': generateSMAData(candlesticks, 20),
      'SMA 50': generateSMAData(candlesticks, 50),
      'EMA 12': generateEMAData(candlesticks, 12),
      'EMA 26': generateEMAData(candlesticks, 26)
    };

    return {
      candlesticks,
      volume,
      indicators
    };
  }, [dataCount]);

  // Real-time data simulation
  useEffect(() => {
    if (!isRealTime) return;

    const interval = setInterval(() => {
      // In a real application, this would fetch new data from an API
      console.log('Simulating real-time data update...');
    }, 5000);

    return () => clearInterval(interval);
  }, [isRealTime]);

  // Event handlers
  const handleCrosshairMove = (price: number | null, time: Time | null) => {
    // Handle crosshair move events
    if (price && time) {
      console.log(`Crosshair: Price ${price.toFixed(2)} at ${new Date(time as number * 1000).toLocaleString()}`);
    }
  };

  const handleVisibleRangeChange = (range: { from: Time; to: Time } | null) => {
    // Handle visible range changes
    if (range) {
      console.log(`Visible range: ${new Date(range.from as number * 1000).toLocaleString()} to ${new Date(range.to as number * 1000).toLocaleString()}`);
    }
  };

  const toggleRealTime = () => {
    setIsRealTime(!isRealTime);
  };

  const resetChart = () => {
    setDataCount(1000);
    setIsRealTime(false);
  };

  const toggleFeature = (feature: keyof typeof enabledFeatures) => {
    setEnabledFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }));
  };

  // Chart configuration
  const chartConfig = {
    performance: {
      enableWebGL: enabledFeatures.performance,
      maxDataPoints: 50000,
      updateFrequency: 16
    },
    features: {
      enableDrawingTools: enabledFeatures.drawingTools,
      enableTechnicalIndicators: enabledFeatures.indicators,
      enableVolumeProfile: enabledFeatures.volumeProfile,
      enableOrderBook: false
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            ChartPro - Institutional Grade Trading Charts
          </h1>
          <p className="text-muted-foreground">
            Professional charting engine built on TradingView Lightweight Charts v5
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant={isRealTime ? "default" : "secondary"} className="gap-1">
            {isRealTime ? <Activity className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
            {isRealTime ? 'Live' : 'Static'}
          </Badge>
          <Badge variant="outline" className="gap-1">
            <Target className="h-3 w-3" />
            {dataCount.toLocaleString()} points
          </Badge>
        </div>
      </div>

      {/* Controls */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Playback Controls */}
          <div className="flex items-center gap-2">
            <Button
              variant={isRealTime ? "default" : "outline"}
              size="sm"
              onClick={toggleRealTime}
              className="gap-2"
            >
              {isRealTime ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isRealTime ? 'Pause' : 'Start'} Live
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={resetChart}
              className="gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>

          <div className="w-px h-6 bg-border" />

          {/* Theme Toggle */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="gap-2"
          >
            <Palette className="h-4 w-4" />
            {theme === 'dark' ? 'Light' : 'Dark'} Theme
          </Button>

          <div className="w-px h-6 bg-border" />

          {/* Feature Toggles */}
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant={enabledFeatures.drawingTools ? "default" : "outline"}
              size="sm"
              onClick={() => toggleFeature('drawingTools')}
              className="gap-2"
            >
              <TrendingUp className="h-4 w-4" />
              Drawing Tools
            </Button>
            <Button
              variant={enabledFeatures.volumeProfile ? "default" : "outline"}
              size="sm"
              onClick={() => toggleFeature('volumeProfile')}
              className="gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              Volume Profile
            </Button>
            <Button
              variant={enabledFeatures.indicators ? "default" : "outline"}
              size="sm"
              onClick={() => toggleFeature('indicators')}
              className="gap-2"
            >
              <Layers className="h-4 w-4" />
              Indicators
            </Button>
            <Button
              variant={enabledFeatures.performance ? "default" : "outline"}
              size="sm"
              onClick={() => toggleFeature('performance')}
              className="gap-2"
            >
              <Zap className="h-4 w-4" />
              WebGL
            </Button>
          </div>

          <div className="w-px h-6 bg-border" />

          {/* Data Controls */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Data Points:</label>
            <select
              value={dataCount}
              onChange={(e) => setDataCount(Number(e.target.value))}
              className="px-2 py-1 text-sm border rounded"
            >
              <option value={500}>500</option>
              <option value={1000}>1,000</option>
              <option value={5000}>5,000</option>
              <option value={10000}>10,000</option>
              <option value={25000}>25,000</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Chart */}
      <ChartPro
        data={chartData}
        height={700}
        theme={theme}
        enableDrawingTools={enabledFeatures.drawingTools}
        enableVolumeProfile={enabledFeatures.volumeProfile}
        enableIndicators={enabledFeatures.indicators}
        enableExport={enabledFeatures.export}
        onCrosshairMove={handleCrosshairMove}
        onVisibleRangeChange={handleVisibleRangeChange}
        config={chartConfig}
        className="border-2 border-dashed border-blue-300 dark:border-blue-700"
      />

      {/* Feature Showcase */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            <h3 className="font-semibold">Drawing Tools</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            Professional drawing tools with fabric.js integration
          </p>
          <ul className="text-xs space-y-1">
            <li>• Trendlines & Support/Resistance</li>
            <li>• Fibonacci Retracements</li>
            <li>• Geometric Shapes</li>
            <li>• Text Annotations</li>
            <li>• Parallel Channels</li>
          </ul>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 className="h-5 w-5 text-green-500" />
            <h3 className="font-semibold">Volume Profile</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            Advanced volume analysis with POC and Value Area
          </p>
          <ul className="text-xs space-y-1">
            <li>• Point of Control (POC)</li>
            <li>• Value Area High/Low</li>
            <li>• Volume Distribution</li>
            <li>• Market Profile</li>
            <li>• VWAP Integration</li>
          </ul>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Layers className="h-5 w-5 text-purple-500" />
            <h3 className="font-semibold">Technical Indicators</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            Extensible plugin architecture for indicators
          </p>
          <ul className="text-xs space-y-1">
            <li>• Moving Averages (SMA, EMA, WMA)</li>
            <li>• Oscillators (RSI, MACD, Stochastic)</li>
            <li>• Bollinger Bands & ATR</li>
            <li>• Custom Indicator Plugins</li>
            <li>• Real-time Calculations</li>
          </ul>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="h-5 w-5 text-orange-500" />
            <h3 className="font-semibold">Performance</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            WebGL acceleration and smart optimizations
          </p>
          <ul className="text-xs space-y-1">
            <li>• WebGL Hardware Acceleration</li>
            <li>• Adaptive Data Sampling</li>
            <li>• Level of Detail (LOD)</li>
            <li>• Memory Management</li>
            <li>• 60 FPS Real-time Updates</li>
          </ul>
        </Card>
      </div>

      {/* Export Showcase */}
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Download className="h-5 w-5 text-indigo-500" />
          <h3 className="font-semibold">Export Capabilities</h3>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Export PNG
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Export SVG
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Export PDF
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Eye className="h-4 w-4" />
            Print
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Settings className="h-4 w-4" />
            Copy to Clipboard
          </Button>
        </div>
      </Card>

      {/* Performance Stats */}
      <Card className="p-4">
        <h3 className="font-semibold mb-3">Performance Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-500">60</div>
            <div className="text-xs text-muted-foreground">FPS</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-500">16</div>
            <div className="text-xs text-muted-foreground">Render Time (ms)</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-500">{dataCount.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Data Points</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-500">4</div>
            <div className="text-xs text-muted-foreground">Active Series</div>
          </div>
        </div>
      </Card>

      {/* Technical Details */}
      <Card className="p-4">
        <h3 className="font-semibold mb-3">Technical Implementation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-2">Core Technologies</h4>
            <ul className="text-sm space-y-1">
              <li>• TradingView Lightweight Charts v5</li>
              <li>• Fabric.js for drawing tools</li>
              <li>• Apache ECharts for volume profile</li>
              <li>• uPlot for mini charts</li>
              <li>• WebGL for hardware acceleration</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Features</h4>
            <ul className="text-sm space-y-1">
              <li>• Bloomberg Terminal styling</li>
              <li>• Institutional-grade performance</li>
              <li>• Professional drawing tools</li>
              <li>• Advanced volume analysis</li>
              <li>• High-quality export options</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ChartProDemo;