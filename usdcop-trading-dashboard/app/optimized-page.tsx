'use client';

import { Suspense, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import { NavigationProvider, useNavigation } from '@/lib/contexts/NavigationContext';
import { NavigationErrorBoundary } from '@/components/common/NavigationErrorBoundary';
import { AnimatedSidebar } from '@/components/ui/AnimatedSidebar';
import { EnhancedNavigationSidebar } from '@/components/ui/EnhancedNavigationSidebar';
import { Menu, X } from 'lucide-react';

// Enhanced loading component with better UX
const LoadingSpinner = ({ viewName }: { viewName?: string }) => (
  <div className="min-h-[60vh] flex items-center justify-center bg-slate-900/30 rounded-xl backdrop-blur-sm">
    <div className="flex flex-col items-center space-y-4">
      <motion.div
        className="relative w-16 h-16"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      >
        <div className="absolute inset-0 border-4 border-slate-700/30 rounded-full" />
        <div className="absolute inset-0 border-4 border-transparent border-t-cyan-400 rounded-full" />
        <motion.div
          className="absolute inset-2 border-2 border-transparent border-t-purple-400 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
        />
      </motion.div>
      <div className="text-center">
        <p className="text-slate-300 font-mono text-sm">Loading {viewName || 'view'}</p>
        <p className="text-slate-500 font-mono text-xs mt-1">Preparing interface...</p>
      </div>
    </div>
  </div>
);

// Optimized dynamic imports with better loading states and error boundaries
const componentImports = {
  'enhanced': () => import('@/components/views/EnhancedTradingDashboard'),
  'realtime': () => import('@/components/views/RealTimeChart'),
  'l5-serving': () => import('@/components/views/L5ModelDashboard'),
  'backtest': () => import('@/components/views/BacktestResults'),
};

// Create dynamic components with enhanced loading
const createDynamicComponent = (importFn: () => Promise<any>, name: string) => 
  dynamic(importFn, {
    ssr: false,
    loading: () => <LoadingSpinner viewName={name} />
  });

// Dynamic component registry
const DynamicComponents = {
  EnhancedTradingDashboard: createDynamicComponent(componentImports.enhanced, 'Trading Terminal'),
  RealTimeChart: createDynamicComponent(componentImports.realtime, 'Real-Time Chart'),
  L5ModelDashboard: createDynamicComponent(componentImports['l5-serving'], 'Model Dashboard'),
  BacktestResults: createDynamicComponent(componentImports.backtest, 'Backtest Results'),
};

// Enhanced placeholder components with better animations
const createPlaceholderComponent = (name: string, color: string, description: string, children: React.ReactNode) =>
  dynamic(() => Promise.resolve(() => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="p-8 bg-slate-900/50 rounded-2xl backdrop-blur-sm border border-slate-700/50 relative overflow-hidden"
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${color} opacity-5`} />
      <div className="relative z-10">
        <motion.h2 
          className={`text-2xl font-bold bg-gradient-to-r ${color} bg-clip-text text-transparent mb-4`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          {name}
        </motion.h2>
        <motion.div 
          className="text-slate-300 mb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          {description}
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          {children}
        </motion.div>
      </div>
    </motion.div>
  )), { ssr: false });

// View component mapping with optimized loading
const viewComponents = {
  'enhanced': DynamicComponents.EnhancedTradingDashboard,
  'realtime': DynamicComponents.RealTimeChart,
  'l5-serving': DynamicComponents.L5ModelDashboard,
  'backtest': DynamicComponents.BacktestResults,
  
  // Trading placeholders
  'signals': createPlaceholderComponent(
    'Trading Signals',
    'from-cyan-400 to-blue-400',
    'Advanced ML-powered trading signals with real-time market analysis',
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {['BUY Signal', 'SELL Signal', 'HOLD Signal'].map((signal, index) => (
        <motion.div
          key={signal}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 + index * 0.1 }}
          className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/30"
        >
          <h3 className="text-lg font-semibold mb-2 text-cyan-300">{signal}</h3>
          <p className="text-slate-400 text-sm">ML confidence: {(85 + Math.random() * 10).toFixed(1)}%</p>
        </motion.div>
      ))}
    </div>
  ),
  
  'risk': createPlaceholderComponent(
    'Risk Management',
    'from-emerald-400 to-green-400',
    'Advanced portfolio risk monitoring with real-time VaR calculations',
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {['VaR', 'CVaR', 'Sharpe Ratio', 'Max Drawdown'].map((metric, index) => (
        <motion.div
          key={metric}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 + index * 0.1 }}
          className="p-4 bg-slate-800/50 rounded-lg text-center"
        >
          <h3 className="text-sm font-mono text-slate-400 mb-1">{metric}</h3>
          <p className="text-lg font-bold text-emerald-400">{(Math.random() * 10).toFixed(2)}</p>
        </motion.div>
      ))}
    </div>
  ),
  
  'model': createPlaceholderComponent(
    'Model Performance',
    'from-purple-400 to-pink-400',
    'Real-time ML model performance analytics and prediction accuracy',
    <div className="space-y-4">
      {['Accuracy', 'Precision', 'Recall', 'F1-Score'].map((metric, index) => (
        <motion.div
          key={metric}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 + index * 0.1 }}
          className="flex justify-between items-center p-3 bg-slate-800/30 rounded-lg"
        >
          <span className="text-slate-300">{metric}</span>
          <span className="text-purple-400 font-bold">{(70 + Math.random() * 25).toFixed(1)}%</span>
        </motion.div>
      ))}
    </div>
  ),
  
  // Pipeline placeholders with enhanced visuals
  'l0-raw': createPlaceholderComponent(
    'L0 Raw Data Pipeline',
    'from-blue-400 to-indigo-400',
    'Real-time market data ingestion from TwelveData API with advanced monitoring',
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="p-4 bg-slate-800/50 rounded-lg border border-blue-500/20">
        <h3 className="text-lg font-semibold text-blue-300 mb-2">Data Ingestion Rate</h3>
        <p className="text-3xl font-bold text-white mb-2">1.2K/sec</p>
        <div className="w-full bg-slate-700 rounded-full h-2">
          <motion.div 
            className="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: '75%' }}
            transition={{ duration: 1.5 }}
          />
        </div>
      </div>
      <div className="p-4 bg-slate-800/50 rounded-lg border border-green-500/20">
        <h3 className="text-lg font-semibold text-green-300 mb-2">API Health</h3>
        <div className="flex items-center space-x-2">
          <motion.div 
            className="w-3 h-3 bg-green-400 rounded-full"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <p className="text-2xl font-bold text-green-400">Online</p>
        </div>
      </div>
    </div>
  ),
  
  // Add more pipeline components...
  'l1-features': createPlaceholderComponent(
    'L1 Feature Engineering',
    'from-green-400 to-emerald-400',
    'Technical indicators and feature statistics with real-time calculations',
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {['RSI', 'MACD', 'Bollinger Bands', 'Volume Profile', 'ATR', 'Stochastic'].map((indicator, index) => (
        <motion.div
          key={indicator}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 + index * 0.1 }}
          className="p-3 bg-slate-800/50 rounded-lg text-center border border-green-500/10"
        >
          <h3 className="text-sm font-mono text-green-300 mb-1">{indicator}</h3>
          <p className="text-lg font-bold text-white">{(Math.random() * 100).toFixed(1)}</p>
        </motion.div>
      ))}
    </div>
  ),
  
  // System monitoring components
  'pipeline-health': createPlaceholderComponent(
    'Pipeline Health Monitor',
    'from-yellow-400 to-orange-400',
    'Real-time monitoring of L0 through L5 pipeline stages with alerts',
    <div className="space-y-3">
      {['L0 Raw Data', 'L1 Features', 'L2 Standardized', 'L3 Correlations', 'L4 RL Ready', 'L5 Model'].map((stage, index) => (
        <motion.div
          key={stage}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 + index * 0.1 }}
          className="flex justify-between items-center p-3 bg-slate-800/30 rounded-lg border border-yellow-500/10"
        >
          <span className="text-slate-300">{stage}</span>
          <motion.div 
            className="w-3 h-3 bg-green-400 rounded-full"
            animate={{ scale: [1, 1.3, 1] }}
            transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
          />
        </motion.div>
      ))}
    </div>
  ),
  
  // Add remaining components with similar enhancements...
  'l3-correlation': createPlaceholderComponent('L3 Feature Correlations', 'from-orange-400 to-red-400', 'Feature correlation analysis and market regime detection', <div>Correlation Matrix Placeholder</div>),
  'l4-rl-ready': createPlaceholderComponent('L4 RL Training Data', 'from-indigo-400 to-purple-400', 'Reinforcement learning ready dataset preparation', <div>RL Data Placeholder</div>),
  'l6-backtest': createPlaceholderComponent('L6 Comprehensive Backtests', 'from-pink-400 to-red-400', 'Strategy performance analysis', <div>Backtest Placeholder</div>),
  'api-usage': createPlaceholderComponent('API Usage Monitor', 'from-red-400 to-pink-400', 'TwelveData API consumption monitoring', <div>API Usage Placeholder</div>),
  'pipeline-monitor': createPlaceholderComponent('Legacy Pipeline Tools', 'from-teal-400 to-cyan-400', 'Legacy pipeline monitoring utilities', <div>Legacy Tools Placeholder</div>),
};

function OptimizedDashboardContent() {
  const { state, toggleSidebar } = useNavigation();
  const mainContentRef = useRef<HTMLDivElement>(null);
  
  // Get the active component
  const ActiveComponent = viewComponents[state.activeView as keyof typeof viewComponents] || viewComponents.enhanced;
  
  // Enhanced view transition with better performance
  const handleViewTransition = useCallback(() => {
    if (mainContentRef.current) {
      mainContentRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100 font-mono relative overflow-hidden">
      
      {/* Enhanced background with better performance */}
      <div className="fixed inset-0 opacity-20 pointer-events-none">
        <motion.div 
          className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-emerald-500/10"
          animate={{ 
            background: [
              'linear-gradient(to right, rgba(6, 182, 212, 0.1), rgba(168, 85, 247, 0.1), rgba(16, 185, 129, 0.1))',
              'linear-gradient(to right, rgba(168, 85, 247, 0.1), rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1))',
              'linear-gradient(to right, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1), rgba(168, 85, 247, 0.1))'
            ]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"
          animate={{ 
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"
          animate={{ 
            x: [0, -100, 0],
            y: [0, -50, 0],
            scale: [1, 0.9, 1]
          }}
          transition={{ duration: 25, repeat: Infinity, ease: "easeInOut", delay: 5 }}
        />
      </div>

      {/* Mobile Menu Button */}
      <motion.button
        className="lg:hidden fixed top-4 left-4 z-50 p-3 bg-slate-900/90 backdrop-blur-sm border border-slate-700/50 rounded-xl text-cyan-400 hover:text-cyan-300 shadow-lg shadow-cyan-400/10"
        onClick={toggleSidebar}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        animate={{ 
          boxShadow: state.sidebarOpen 
            ? "0 10px 25px rgba(6, 182, 212, 0.3)" 
            : "0 10px 25px rgba(6, 182, 212, 0.1)"
        }}
        transition={{ duration: 0.2 }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={state.sidebarOpen ? 'close' : 'open'}
            initial={{ rotate: -90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={{ rotate: 90, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {state.sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </motion.div>
        </AnimatePresence>
      </motion.button>

      {/* Enhanced Sidebar Layout */}
      <motion.div 
        className={`fixed inset-y-0 left-0 z-40 transform transition-all duration-300 lg:translate-x-0 ${
          state.sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        animate={{ 
          x: state.sidebarOpen || window.innerWidth >= 1024 ? 0 : -100 
        }}
      >
        <div className="h-full flex shadow-2xl">
          {/* Controls Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <AnimatedSidebar
              onPlayPause={() => {}}
              onReset={() => {}}
              onAlignDataset={() => {}}
              isPlaying={state.isPlaying}
              isRealtime={state.isRealtime}
              dataSource={state.dataSource}
              onDataSourceChange={() => {}}
              marketStatus={state.marketData.marketStatus}
              currentPrice={state.marketData.currentPrice}
              priceChange={state.marketData.priceChange}
              priceChangePercent={state.marketData.priceChangePercent}
            />
          </motion.div>
          
          {/* Navigation Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <EnhancedNavigationSidebar />
          </motion.div>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className={`transition-all duration-300 ${
        state.sidebarExpanded 
          ? 'lg:ml-[660px]' // AnimatedSidebar (280px) + EnhancedNavigationSidebar (380px when expanded)
          : 'lg:ml-[296px]'  // AnimatedSidebar (280px) + EnhancedNavigationSidebar (16px when collapsed)
      } relative z-10`}>
        
        {/* Enhanced Header */}
        <motion.div 
          className="bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 p-4 sticky top-0 z-30"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex items-center justify-between">
            <motion.div 
              className="flex items-center space-x-4"
              key={state.activeView}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div>
                <h2 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  {/* Get view name from viewComponents mapping */}
                  {Object.entries(viewComponents).find(([key]) => key === state.activeView)?.[0] || 'Dashboard'}
                </h2>
                <p className="text-sm text-slate-400 font-normal">
                  Professional trading interface
                </p>
              </div>
              
              {/* Status badges */}
              <div className="flex items-center space-x-2">
                <motion.div
                  className="px-3 py-1 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-full border border-cyan-500/30"
                  whileHover={{ scale: 1.05 }}
                >
                  <span className="text-xs font-mono text-cyan-300">ACTIVE</span>
                </motion.div>
                
                <motion.div
                  className={`px-3 py-1 rounded-full border ${
                    state.isRealtime 
                      ? 'bg-green-500/20 border-green-500/30 text-green-300' 
                      : 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300'
                  }`}
                  animate={{ scale: state.isRealtime ? [1, 1.05, 1] : 1 }}
                  transition={{ duration: 2, repeat: state.isRealtime ? Infinity : 0 }}
                >
                  <span className="text-xs font-mono">
                    {state.isRealtime ? 'LIVE' : 'HISTORICAL'}
                  </span>
                </motion.div>
              </div>
            </motion.div>
            
            <div className="flex items-center gap-4">
              {/* Data source indicator */}
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <motion.div 
                  className={`w-3 h-3 rounded-full ${
                    state.dataSource === 'l0' ? 'bg-blue-400' :
                    state.dataSource === 'l1' ? 'bg-green-400' : 'bg-purple-400'
                  }`}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <span className="font-mono uppercase tracking-wider">{state.dataSource} DATA</span>
              </div>
              
              {/* Market status */}
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="font-mono uppercase tracking-wider">{state.marketData.marketStatus}</span>
              </div>
            </div>
          </div>
          
          {/* Loading progress bar */}
          <AnimatePresence>
            {state.isLoading && (
              <motion.div
                className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 opacity-75"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
              />
            )}
          </AnimatePresence>
        </motion.div>

        {/* Main Content Area */}
        <main ref={mainContentRef} className="p-6 relative min-h-screen">
          <AnimatePresence mode="wait" onExitComplete={handleViewTransition}>
            {state.isLoading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
              >
                <LoadingSpinner viewName="dashboard view" />
              </motion.div>
            ) : (
              <motion.div
                key={state.activeView}
                initial={{ opacity: 0, y: 30, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -30, scale: 0.96 }}
                transition={{ 
                  duration: 0.4, 
                  ease: [0.4, 0, 0.2, 1],
                  type: "spring",
                  damping: 25,
                  stiffness: 200
                }}
                className="relative"
              >
                <NavigationErrorBoundary
                >
                  <Suspense fallback={<LoadingSpinner viewName="component" />}>
                    <ActiveComponent />
                  </Suspense>
                </NavigationErrorBoundary>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {state.sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 bg-black/60 z-30 lg:hidden backdrop-blur-sm"
            onClick={toggleSidebar}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default function OptimizedTradingDashboard() {
  return (
    <NavigationProvider>
      <OptimizedDashboardContent />
    </NavigationProvider>
  );
}