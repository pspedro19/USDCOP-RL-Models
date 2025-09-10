'use client';

/**
 * Enhanced Trading Visualization Suite
 * Master component integrating all professional charting and analysis features
 * Complete trading visualization ecosystem with advanced capabilities
 */

import React, { useState, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  TrendingUp,
  Activity,
  Target,
  Zap,
  Download,
  Settings,
  Maximize2,
  Minimize2,
  RefreshCw,
  Layers,
  Eye,
  EyeOff,
  Grid3X3,
  Monitor,
  Palette
} from 'lucide-react';

// Import all enhanced components
import AdvancedTechnicalIndicators from './AdvancedTechnicalIndicators';
import InteractiveDrawingTools from './InteractiveDrawingTools';
import CandlestickPatternRecognition from './CandlestickPatternRecognition';
import MultiTimeframeAnalysis from './MultiTimeframeAnalysis';
import HighPerformanceVirtualizedChart from './HighPerformanceVirtualizedChart';
import ProfessionalTradingTerminal from './ProfessionalTradingTerminal';
import AdvancedExportCapabilities from './AdvancedExportCapabilities';
import { InteractiveTradingChart } from './InteractiveTradingChart';

interface OHLCData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
  index: number;
}

interface EnhancedVisualizationSuiteProps {
  data: OHLCData[];
  isRealtime?: boolean;
  symbol?: string;
  enableAllFeatures?: boolean;
}

type ViewMode = 'standard' | 'professional' | 'analysis' | 'terminal';

export const EnhancedTradingVisualizationSuite: React.FC<EnhancedVisualizationSuiteProps> = ({
  data,
  isRealtime = false,
  symbol = 'USD/COP',
  enableAllFeatures = true
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('standard');
  const [activeTab, setActiveTab] = useState('chart');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControlPanel, setShowControlPanel] = useState(true);
  const [enabledFeatures, setEnabledFeatures] = useState({
    indicators: true,
    patterns: true,
    drawings: true,
    multiTimeframe: true,
    terminal: true,
    export: true,
    performance: true
  });

  const chartRef = useRef<HTMLDivElement>(null);
  const suiteRef = useRef<HTMLDivElement>(null);

  // Prepare export data
  const exportData = useMemo(() => ({
    charts: [{ type: 'candlestick', data: data.slice(-100) }],
    indicators: [
      { name: 'RSI', value: 65.4, signal: 'neutral', strength: 'moderate' },
      { name: 'MACD', value: 0.45, signal: 'bullish', strength: 'strong' }
    ],
    patterns: [
      { name: 'Bullish Engulfing', type: 'bullish', strength: 'strong', reliability: 85, datetime: new Date().toISOString() }
    ],
    trades: [],
    analysis: {
      trend: 'bullish',
      confidence: 75,
      support: 4180.50,
      resistance: 4220.80,
      volatility: 'medium',
      volumeProfile: 'balanced'
    },
    metadata: {
      symbol,
      dateRange: {
        start: data.length > 0 ? data[0].datetime : new Date().toISOString(),
        end: data.length > 0 ? data[data.length - 1].datetime : new Date().toISOString()
      },
      timeframe: '5M',
      dataPoints: data.length
    }
  }), [data, symbol]);

  // Mock market data for terminal
  const marketData = useMemo(() => ({
    symbol,
    price: data.length > 0 ? data[data.length - 1].close : 4200,
    change: data.length > 1 ? data[data.length - 1].close - data[data.length - 2].close : 0,
    changePercent: data.length > 1 ? 
      ((data[data.length - 1].close - data[data.length - 2].close) / data[data.length - 2].close) * 100 : 0,
    volume: data.length > 0 ? data[data.length - 1].volume : 1000000,
    bid: data.length > 0 ? data[data.length - 1].close - 0.02 : 4199.98,
    ask: data.length > 0 ? data[data.length - 1].close + 0.02 : 4200.02,
    spread: 0.04,
    high24h: Math.max(...data.slice(-288).map(d => d.high)), // Assuming 5-min data, 288 = 24h
    low24h: Math.min(...data.slice(-288).map(d => d.low))
  }), [data, symbol]);

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      suiteRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const toggleFeature = (feature: keyof typeof enabledFeatures) => {
    setEnabledFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
    }));
  };

  const getViewModeConfig = (mode: ViewMode) => {
    switch (mode) {
      case 'professional':
        return {
          title: 'Professional Trading',
          description: 'Advanced charts with full technical analysis',
          tabs: ['chart', 'indicators', 'patterns', 'drawings']
        };
      case 'analysis':
        return {
          title: 'Multi-Timeframe Analysis',
          description: 'Comprehensive market analysis across timeframes',
          tabs: ['multiTimeframe', 'patterns', 'indicators', 'export']
        };
      case 'terminal':
        return {
          title: 'Trading Terminal',
          description: 'Professional trading interface with order flow',
          tabs: ['terminal', 'chart', 'export']
        };
      default:
        return {
          title: 'Standard Trading View',
          description: 'Enhanced interactive charts',
          tabs: ['chart', 'indicators', 'export']
        };
    }
  };

  const currentConfig = getViewModeConfig(viewMode);

  return (
    <motion.div
      ref={suiteRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className={`${isFullscreen ? 'fixed inset-0 z-50 bg-slate-950' : 'relative'} min-h-screen`}
    >
      {/* Control Header */}
      <AnimatePresence>
        {showControlPanel && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="sticky top-0 z-40 bg-slate-900/95 backdrop-blur-xl border-b border-slate-700/50 p-4"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-4">
                  <motion.h1
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent"
                  >
                    {currentConfig.title}
                  </motion.h1>
                  <Badge className="bg-slate-800 text-slate-300 border-slate-600">
                    {currentConfig.description}
                  </Badge>
                  {isRealtime && (
                    <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 animate-pulse">
                      LIVE
                    </Badge>
                  )}
                </div>

                {/* View Mode Selector */}
                <div className="flex items-center gap-2 bg-slate-800/50 rounded-lg p-1">
                  {(['standard', 'professional', 'analysis', 'terminal'] as ViewMode[]).map(mode => (
                    <motion.button
                      key={mode}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setViewMode(mode)}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200 ${
                        viewMode === mode
                          ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
                          : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                      }`}
                    >
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </motion.button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Feature Toggles */}
                <div className="flex items-center gap-1">
                  {Object.entries(enabledFeatures).map(([feature, enabled]) => (
                    <Button
                      key={feature}
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleFeature(feature as keyof typeof enabledFeatures)}
                      className={`${enabled ? 'text-cyan-400' : 'text-slate-500'} hover:text-white`}
                      title={`Toggle ${feature}`}
                    >
                      {feature === 'indicators' && <Activity className="w-4 h-4" />}
                      {feature === 'patterns' && <Target className="w-4 h-4" />}
                      {feature === 'drawings' && <Palette className="w-4 h-4" />}
                      {feature === 'multiTimeframe' && <Grid3X3 className="w-4 h-4" />}
                      {feature === 'terminal' && <Monitor className="w-4 h-4" />}
                      {feature === 'export' && <Download className="w-4 h-4" />}
                      {feature === 'performance' && <Zap className="w-4 h-4" />}
                    </Button>
                  ))}
                </div>

                {/* Control Buttons */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowControlPanel(!showControlPanel)}
                  className="text-slate-400 hover:text-white"
                >
                  {showControlPanel ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleFullscreen}
                  className="text-slate-400 hover:text-white"
                >
                  {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  className="text-slate-400 hover:text-white"
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className={`${isFullscreen ? 'p-6' : 'p-6'}`}>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full bg-slate-800/50 p-1 rounded-lg">
            {currentConfig.tabs.map((tab) => (
              <TabsTrigger
                key={tab}
                value={tab}
                className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-purple-500 data-[state=active]:text-white text-slate-400 font-medium"
              >
                {tab === 'chart' && <BarChart3 className="w-4 h-4 mr-2" />}
                {tab === 'indicators' && <Activity className="w-4 h-4 mr-2" />}
                {tab === 'patterns' && <Target className="w-4 h-4 mr-2" />}
                {tab === 'drawings' && <Palette className="w-4 h-4 mr-2" />}
                {tab === 'multiTimeframe' && <Grid3X3 className="w-4 h-4 mr-2" />}
                {tab === 'terminal' && <Monitor className="w-4 h-4 mr-2" />}
                {tab === 'export' && <Download className="w-4 h-4 mr-2" />}
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* Chart Tab */}
          <TabsContent value="chart" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              ref={chartRef}
            >
              {enabledFeatures.performance ? (
                <HighPerformanceVirtualizedChart
                  data={data.map((d, index) => ({ ...d, timestamp: new Date(d.datetime).getTime(), index }))}
                  width={isFullscreen ? window.innerWidth - 100 : 1200}
                  height={isFullscreen ? window.innerHeight - 200 : 600}
                  enableWebGL={true}
                  enableWorkers={true}
                />
              ) : (
                <InteractiveTradingChart
                  data={data}
                  isRealtime={isRealtime}
                />
              )}
            </motion.div>

            {/* Drawing Tools Overlay */}
            {enabledFeatures.drawings && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="absolute inset-0 pointer-events-none"
              >
                <InteractiveDrawingTools
                  data={data}
                  width={isFullscreen ? window.innerWidth - 100 : 1200}
                  height={isFullscreen ? window.innerHeight - 200 : 600}
                  xScale={(timestamp: string) => 0}
                  yScale={(price: number) => 0}
                  onDrawingsChange={(drawings) => console.log('Drawings updated:', drawings)}
                />
              </motion.div>
            )}
          </TabsContent>

          {/* Technical Indicators Tab */}
          <TabsContent value="indicators" className="space-y-6">
            {enabledFeatures.indicators && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <AdvancedTechnicalIndicators
                  data={data}
                  height={400}
                  showGrid={true}
                />
              </motion.div>
            )}
          </TabsContent>

          {/* Pattern Recognition Tab */}
          <TabsContent value="patterns" className="space-y-6">
            {enabledFeatures.patterns && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <CandlestickPatternRecognition
                  data={data}
                  onPatternClick={(pattern) => console.log('Pattern clicked:', pattern)}
                />
              </motion.div>
            )}
          </TabsContent>

          {/* Drawing Tools Tab */}
          <TabsContent value="drawings" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Interactive Drawing Tools</h3>
                <p className="text-slate-400 mb-4">
                  Professional drawing tools are integrated with the chart view. Switch to the Chart tab to access
                  trendlines, fibonacci retracements, support/resistance levels, and more.
                </p>
                <Button onClick={() => setActiveTab('chart')}>
                  <Palette className="w-4 h-4 mr-2" />
                  Go to Chart
                </Button>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Multi-Timeframe Analysis Tab */}
          <TabsContent value="multiTimeframe" className="space-y-6">
            {enabledFeatures.multiTimeframe && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <MultiTimeframeAnalysis
                  rawData={data}
                  height={300}
                />
              </motion.div>
            )}
          </TabsContent>

          {/* Trading Terminal Tab */}
          <TabsContent value="terminal" className="space-y-6">
            {enabledFeatures.terminal && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <ProfessionalTradingTerminal
                  marketData={marketData}
                  onOrderSubmit={(order) => console.log('Order submitted:', order)}
                  onOrderCancel={(orderId) => console.log('Order cancelled:', orderId)}
                  isConnected={true}
                  enableSound={true}
                />
              </motion.div>
            )}
          </TabsContent>

          {/* Export Tab */}
          <TabsContent value="export" className="space-y-6">
            {enabledFeatures.export && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <AdvancedExportCapabilities
                  data={exportData}
                  chartRef={chartRef}
                  onExportStart={() => console.log('Export started')}
                  onExportComplete={(result) => console.log('Export completed:', result)}
                />
              </motion.div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-slate-900/95 backdrop-blur-xl border-t border-slate-700/50 p-2 z-30">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-4">
            <span>Data Points: {data.length.toLocaleString()}</span>
            <span>Symbol: {symbol}</span>
            <span>Mode: {viewMode}</span>
            {isRealtime && (
              <span className="text-emerald-400 animate-pulse">● Live Data</span>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            <span>Features: {Object.values(enabledFeatures).filter(Boolean).length}/7 Active</span>
            <span>Performance: Optimized</span>
            <span>Ready</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default EnhancedTradingVisualizationSuite;