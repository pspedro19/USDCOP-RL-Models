'use client';

import { useState, useCallback, useRef, useEffect, Suspense } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import { motionLibrary } from '@/lib/motion';
import { AnimatedSidebar } from '@/components/ui/AnimatedSidebar';
import { MobileControlsBar } from '@/components/ui/MobileControlsBar';
import { SidebarToggleButtons } from '@/components/ui/SidebarToggleButtons';
import { useSidebarState } from '@/hooks/useSidebarState';
import { useResponsiveLayout, useBreakpoints } from '@/hooks/useResponsiveLayout';

// Enhanced Loading component with professional animations
const LoadingSpinner = () => (
  <motion.div 
    className="min-h-screen flex items-center justify-center bg-slate-900"
    variants={motionLibrary.pages.fadeScale}
    initial="initial"
    animate="animate"
    exit="exit"
  >
    <div className="flex flex-col items-center space-y-6">
      {/* Professional spinner with glass effect */}
      <motion.div
        className="relative"
        variants={motionLibrary.loading.spinner}
        animate="animate"
      >
        <div className="w-20 h-20 border-4 border-cyan-400/20 border-t-cyan-400 rounded-full glass-surface-primary backdrop-blur-lg shadow-glass-lg" />
        <motion.div 
          className="absolute inset-2 border-2 border-purple-400/30 border-b-purple-400 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
      </motion.div>
      
      {/* Enhanced loading text with stagger animation */}
      <motion.div className="text-center space-y-2">
        <motion.p 
          className="text-slate-400 font-mono text-lg text-glow"
          variants={motionLibrary.micro.tooltip}
          initial="initial"
          animate="animate"
        >
          Loading view...
        </motion.p>
        <motion.div 
          className="flex space-x-1 justify-center"
          variants={motionLibrary.lists.container}
          initial="initial"
          animate="animate"
        >
          {[...Array(3)].map((_, i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-cyan-400 rounded-full"
              variants={motionLibrary.loading.pulse}
              animate="animate"
              style={{ animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </motion.div>
      </motion.div>
    </div>
  </motion.div>
);

// Enhanced Error boundary component with animations
const ErrorFallback = ({ error, resetError }: { error: Error; resetError: () => void }) => (
  <motion.div 
    className="min-h-screen flex items-center justify-center bg-slate-900 p-6"
    variants={motionLibrary.pages.glassEntrance}
    initial="initial"
    animate="animate"
  >
    <motion.div 
      className="glass-card bg-red-900/20 border border-red-500/50 p-8 max-w-md text-center backdrop-blur-lg"
      variants={motionLibrary.pages.modal}
      initial="initial"
      animate="animate"
    >
      <motion.div 
        className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center"
        variants={motionLibrary.micro.statusPulse}
        animate="animate"
      >
        <div className="w-8 h-8 border-2 border-red-400 rounded-full flex items-center justify-center">
          <div className="w-2 h-2 bg-red-400 rounded-full" />
        </div>
      </motion.div>
      
      <motion.h2 
        className="text-red-400 text-xl font-bold mb-4 text-glow"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        Something went wrong
      </motion.h2>
      
      <motion.p 
        className="text-slate-300 mb-6 font-mono text-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {error.message}
      </motion.p>
      
      <motion.button
        onClick={resetError}
        className="glass-button-primary px-6 py-3 text-white font-semibold rounded-xl"
        variants={motionLibrary.components.glassButton}
        initial="initial"
        whileHover="hover"
        whileTap="tap"
      >
        Try again
      </motion.button>
    </motion.div>
  </motion.div>
);

// Optimized dynamic imports with proper loading states
const EnhancedTradingDashboard = dynamic(
  () => import('@/components/views/EnhancedTradingDashboard'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const RealTimeChart = dynamic(
  () => import('@/components/views/RealTimeChart'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const L5ModelDashboard = dynamic(
  () => import('@/components/views/L5ModelDashboard'),
  { 
    ssr: false, 
    loading: () => <LoadingSpinner />
  }
);

const BacktestResults = dynamic(
  () => import('@/components/views/BacktestResults'),
  { 
    ssr: false, 
    loading: () => <LoadingSpinner />
  }
);

const PipelineHealthDashboard = dynamic(
  () => import('@/components/views/PipelineHealthDashboard'),
  { 
    ssr: false, 
    loading: () => <LoadingSpinner />
  }
);

// Enhanced placeholder components for missing views
const TradingSignals = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-cyan-400 mb-4">Trading Signals</h2>
    <div className="text-slate-300 mb-4">Advanced ML-powered trading signals with real-time market analysis</div>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {['BUY Signal', 'SELL Signal', 'HOLD Signal'].map((signal, index) => (
        <motion.div
          key={signal}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/30"
        >
          <h3 className="text-lg font-semibold mb-2 text-cyan-300">{signal}</h3>
          <p className="text-slate-400 text-sm">ML confidence: 89.5%</p>
        </motion.div>
      ))}
    </div>
  </div>
)), { ssr: false });

const RiskManagement = dynamic(
  () => import('@/components/views/RiskManagement'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const RealTimeRiskMonitor = dynamic(
  () => import('@/components/views/RealTimeRiskMonitor'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const PortfolioExposureAnalysis = dynamic(
  () => import('@/components/views/PortfolioExposureAnalysis'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const RiskAlertsCenter = dynamic(
  () => import('@/components/views/RiskAlertsCenter'),
  { 
    ssr: false,
    loading: () => <LoadingSpinner />
  }
);

const ModelPerformance = dynamic(() => import('@/components/ml-analytics/ModelPerformanceDashboard'), { 
  ssr: false,
  loading: () => <LoadingSpinner />
});

// Pipeline views with enhanced placeholders
const L0RawDataDashboard = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-blue-400 mb-4">L0 Raw Data Pipeline</h2>
    <div className="text-slate-300 mb-4">Real-time market data ingestion from TwelveData API</div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="p-4 bg-slate-800/50 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-300 mb-2">Data Ingestion Rate</h3>
        <p className="text-2xl font-bold text-white">1.2K/sec</p>
      </div>
      <div className="p-4 bg-slate-800/50 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-300 mb-2">API Health</h3>
        <p className="text-2xl font-bold text-green-400">Online</p>
      </div>
    </div>
  </div>
)), { ssr: false });

const L1FeatureStats = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-green-400 mb-4">L1 Feature Engineering</h2>
    <div className="text-slate-300 mb-4">Technical indicators and feature statistics</div>
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {['RSI', 'MACD', 'Bollinger Bands', 'Volume Profile', 'ATR', 'Stochastic'].map((indicator, index) => (
        <motion.div
          key={indicator}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="p-3 bg-slate-800/50 rounded-lg text-center"
        >
          <h3 className="text-sm font-mono text-green-300 mb-1">{indicator}</h3>
          <p className="text-lg font-bold text-white">72.3</p>
        </motion.div>
      ))}
    </div>
  </div>
)), { ssr: false });

const L3CorrelationMatrix = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-orange-400 mb-4">L3 Feature Correlations</h2>
    <div className="text-slate-300 mb-4">Feature correlation analysis and market regime detection</div>
    <div className="p-4 bg-slate-800/50 rounded-lg">
      <h3 className="text-lg font-semibold text-orange-300 mb-2">Correlation Matrix</h3>
      <p className="text-slate-400">Interactive correlation heatmap visualization</p>
    </div>
  </div>
)), { ssr: false });

const L4RLReadyData = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-indigo-400 mb-4">L4 RL Training Data</h2>
    <div className="text-slate-300 mb-4">Reinforcement learning ready dataset preparation</div>
    <div className="space-y-3">
      {['State Vector Dim', 'Action Space', 'Reward Function', 'Training Episodes'].map((item, index) => (
        <motion.div
          key={item}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="flex justify-between items-center p-3 bg-slate-800/30 rounded-lg"
        >
          <span className="text-slate-300">{item}</span>
          <span className="text-indigo-400 font-bold">
            {item === 'State Vector Dim' ? '42' : 
             item === 'Action Space' ? '3' :
             item === 'Reward Function' ? 'Sharpe' : '10K+'}
          </span>
        </motion.div>
      ))}
    </div>
  </div>
)), { ssr: false });

const L6BacktestResults = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-pink-400 mb-4">L6 Comprehensive Backtests</h2>
    <div className="text-slate-300 mb-4">Strategy performance analysis and backtesting results</div>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {['Total Return', 'Win Rate', 'Profit Factor', 'Max DD'].map((metric, index) => (
        <motion.div
          key={metric}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.1 }}
          className="p-4 bg-slate-800/50 rounded-lg text-center"
        >
          <h3 className="text-sm font-mono text-pink-300 mb-1">{metric}</h3>
          <p className="text-lg font-bold text-white">
            {metric === 'Total Return' ? '+24.5%' :
             metric === 'Win Rate' ? '68.3%' :
             metric === 'Profit Factor' ? '1.45' : '-5.2%'}
          </p>
        </motion.div>
      ))}
    </div>
  </div>
)), { ssr: false });

const PipelineHealthMonitor = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-yellow-400 mb-4">Pipeline Health Monitor</h2>
    <div className="text-slate-300 mb-4">Real-time monitoring of L0 through L5 pipeline stages</div>
    <div className="space-y-3">
      {['L0 Raw Data', 'L1 Features', 'L2 Standardized', 'L3 Correlations', 'L4 RL Ready', 'L5 Model'].map((stage, index) => (
        <motion.div
          key={stage}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="flex justify-between items-center p-3 bg-slate-800/30 rounded-lg"
        >
          <span className="text-slate-300">{stage}</span>
          <motion.div 
            className="w-3 h-3 bg-green-400 rounded-full"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
          />
        </motion.div>
      ))}
    </div>
  </div>
)), { ssr: false });

const APIUsagePanel = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-red-400 mb-4">API Usage Monitor</h2>
    <div className="text-slate-300 mb-4">TwelveData API consumption and rate limiting</div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="p-4 bg-slate-800/50 rounded-lg">
        <h3 className="text-lg font-semibold text-red-300 mb-2">Requests Today</h3>
        <p className="text-2xl font-bold text-white">2,847 / 8,000</p>
      </div>
      <div className="p-4 bg-slate-800/50 rounded-lg">
        <h3 className="text-lg font-semibold text-red-300 mb-2">Rate Limit</h3>
        <p className="text-2xl font-bold text-green-400">Normal</p>
      </div>
    </div>
  </div>
)), { ssr: false });

const PipelineMonitor = dynamic(() => Promise.resolve(() => (
  <div className="p-6 bg-slate-900/50 rounded-xl backdrop-blur-sm border border-slate-700/50">
    <h2 className="text-xl font-bold text-teal-400 mb-4">Legacy Pipeline Tools</h2>
    <div className="text-slate-300 mb-4">Legacy pipeline monitoring and maintenance utilities</div>
    <div className="p-4 bg-slate-800/50 rounded-lg">
      <h3 className="text-lg font-semibold text-teal-300 mb-2">System Status</h3>
      <p className="text-slate-400">All legacy components operational</p>
    </div>
  </div>
)), { ssr: false });

import { 
  LineChart, 
  Signal, 
  TrendingUp, 
  Database, 
  Shield, 
  Brain,
  Menu,
  X,
  Activity,
  BarChart3,
  GitBranch,
  Cpu,
  Target,
  PieChart,
  Zap,
  Key,
  ChevronRight,
  Sparkles,
  Gauge,
  Bell,
  Eye,
  EyeOff,
  LogOut,
  User
} from 'lucide-react';

const views = [
  // Enhanced Terminal (Primary)
  { id: 'enhanced', name: 'Trading Terminal', icon: Activity, component: EnhancedTradingDashboard, category: 'Trading', description: 'Advanced trading dashboard with replay controls' },
  
  // Original trading views
  { id: 'realtime', name: 'Real-Time Chart', icon: LineChart, component: RealTimeChart, category: 'Trading', description: 'Live market data visualization' },
  { id: 'signals', name: 'Trading Signals', icon: Signal, component: TradingSignals, category: 'Trading', description: 'ML-powered trading signals' },
  { id: 'backtest', name: 'Backtest Results', icon: TrendingUp, component: BacktestResults, category: 'Trading', description: 'Strategy backtesting analysis' },
  { id: 'ml-analytics', name: 'ML Analytics', icon: Brain, component: ModelPerformance, category: 'Trading', description: 'ML model performance dashboard' },
  
  // Risk Management views
  { id: 'risk', name: 'Risk Management', icon: Shield, component: RiskManagement, category: 'Risk', description: 'Portfolio risk monitoring & analytics' },
  { id: 'realtime-risk', name: 'Real-Time Risk Monitor', icon: Gauge, component: RealTimeRiskMonitor, category: 'Risk', description: 'Live risk metrics dashboard' },
  { id: 'exposure-analysis', name: 'Exposure Analysis', icon: Target, component: PortfolioExposureAnalysis, category: 'Risk', description: 'Multi-dimensional exposure breakdown' },
  { id: 'risk-alerts', name: 'Risk Alerts Center', icon: Bell, component: RiskAlertsCenter, category: 'Risk', description: 'Risk alert management & notifications' },
  
  // Pipeline data views
  { id: 'l0-raw', name: 'L0 Raw Data', icon: Database, component: L0RawDataDashboard, category: 'Pipeline', description: 'Raw market data ingestion' },
  { id: 'l1-features', name: 'L1 Feature Stats', icon: BarChart3, component: L1FeatureStats, category: 'Pipeline', description: 'Technical indicators' },
  { id: 'l3-correlation', name: 'L3 Correlations', icon: GitBranch, component: L3CorrelationMatrix, category: 'Pipeline', description: 'Feature correlation analysis' },
  { id: 'l4-rl-ready', name: 'L4 RL Data', icon: Cpu, component: L4RLReadyData, category: 'Pipeline', description: 'RL training data' },
  { id: 'l5-serving', name: 'L5 Model Serving', icon: Target, component: L5ModelDashboard, category: 'Pipeline', description: 'Model predictions' },
  { id: 'l6-backtest', name: 'L6 Backtests', icon: PieChart, component: L6BacktestResults, category: 'Pipeline', description: 'Comprehensive backtests' },
  
  // System monitoring
  { id: 'pipeline-health', name: 'Pipeline Health', icon: Zap, component: PipelineHealthMonitor, category: 'System', description: 'Pipeline monitoring' },
  { id: 'api-usage', name: 'API Usage', icon: Key, component: APIUsagePanel, category: 'System', description: 'API rate monitoring' },
  { id: 'pipeline-monitor', name: 'Legacy Pipeline', icon: Database, component: PipelineMonitor, category: 'System', description: 'Legacy tools' },
];

export default function TradingDashboard() {
  const router = useRouter();
  
  // Authentication check with fallback to localStorage
  useEffect(() => {
    // Try sessionStorage first, then localStorage as fallback
    const isAuthenticated = sessionStorage.getItem('isAuthenticated') || 
                          localStorage.getItem('isAuthenticated');
    const storedUsername = sessionStorage.getItem('username') || 
                          localStorage.getItem('username');
    
    if (!isAuthenticated) {
      router.push('/login');
    } else {
      setUsername(storedUsername || 'admin');
    }
  }, [router]);
  
  // Centralized sidebar state management
  const sidebarState = useSidebarState();
  const {
    // State
    leftSidebarCollapsed,
    navigationSidebarOpen: sidebarOpen,
    isDesktop,
    isMobile,
    viewportWidth,
    autoHideEnabled,
    // Actions
    toggleLeftSidebar,
    toggleNavigationSidebar,
    toggleBothSidebars,
    collapseLeftSidebar,
    expandLeftSidebar,
    openNavigationSidebar,
    closeNavigationSidebar,
    setAutoHide,
    resetToDefaults,
    forceSync,
    // Calculations
    leftSidebarWidth,
    navigationSidebarWidth,
    totalSidebarWidth,
    mainContentMarginLeft,
    isLeftSidebarVisible,
    isNavigationSidebarVisible,
  } = sidebarState;
  
  // Responsive layout calculations
  const responsiveLayout = useResponsiveLayout(sidebarState);
  const breakpoints = useBreakpoints(viewportWidth);
  
  // Navigation state
  const [activeView, setActiveView] = useState('enhanced');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(false);
  const [username, setUsername] = useState<string>('');
  
  // Market and data controls state
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRealtime, setIsRealtime] = useState(true);
  const [dataSource, setDataSource] = useState<'l0' | 'l1' | 'mock'>('l1');
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'pre-market' | 'after-hours'>('open');
  
  // Price state to avoid hydration errors
  const [currentPrice, setCurrentPrice] = useState(4150.25);
  const [priceChange, setPriceChange] = useState(15.75);
  const [priceChangePercent, setPriceChangePercent] = useState(0.38);
  
  // Navigation transition state
  const [isTransitioning, setIsTransitioning] = useState(false);
  const viewRef = useRef<HTMLDivElement>(null);
  const transitionTimeoutRef = useRef<NodeJS.Timeout>();

  const ActiveComponent = views.find(v => v.id === activeView)?.component || views[0].component;
  const activeViewData = views.find(v => v.id === activeView);

  const categories = ['Trading', 'Risk', 'Pipeline', 'System'];
  
  // Enhanced sidebar toggle handlers (using centralized state)
  const handleLeftSidebarToggle = useCallback(() => {
    toggleLeftSidebar();
  }, [toggleLeftSidebar]);

  const handleNavigationSidebarToggle = useCallback(() => {
    toggleNavigationSidebar();
  }, [toggleNavigationSidebar]);

  const handleRightSidebarToggle = useCallback(() => {
    setRightSidebarCollapsed(prev => !prev);
  }, []);

  const handleLogout = useCallback(() => {
    // Clear both sessionStorage and localStorage
    sessionStorage.removeItem('isAuthenticated');
    sessionStorage.removeItem('username');
    sessionStorage.removeItem('loginTime');
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    localStorage.removeItem('loginTime');
    router.push('/login');
  }, [router]);

  // Enhanced view switching with proper sidebar management
  const handleViewChange = useCallback(async (viewId: string) => {
    if (viewId === activeView || isTransitioning) return;
    
    try {
      setIsTransitioning(true);
      setIsLoading(true);
      setError(null);
      
      // Clear any existing transition timeout
      if (transitionTimeoutRef.current) {
        clearTimeout(transitionTimeoutRef.current);
      }
      
      // Close mobile sidebar using centralized state
      closeNavigationSidebar();
      
      // Simulate loading time for smooth UX
      transitionTimeoutRef.current = setTimeout(() => {
        setActiveView(viewId);
        setIsLoading(false);
        setIsTransitioning(false);
      }, 300);
      
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to change view'));
      setIsLoading(false);
      setIsTransitioning(false);
    }
  }, [activeView, isTransitioning]);
  
  // Control handlers
  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
    setIsRealtime(prev => !prev);
  }, []);
  
  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setIsRealtime(true);
    // Reset any view-specific state here
  }, []);
  
  const handleAlignDataset = useCallback(() => {
    setIsRealtime(true);
    setIsPlaying(false);
    // Trigger data alignment
  }, []);
  
  const handleDataSourceChange = useCallback((source: 'l0' | 'l1' | 'mock') => {
    setDataSource(source);
    // Trigger data refresh for new source
  }, []);
  
  // Simulate price updates after mount to avoid hydration errors
  useEffect(() => {
    const timer = setTimeout(() => {
      // Only update prices after component is mounted
      setCurrentPrice(4150.25 + (Math.random() - 0.5) * 20);
      setPriceChange(15.75 + (Math.random() - 0.5) * 10);
      setPriceChangePercent(0.38 + (Math.random() - 0.5) * 0.5);
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Market status simulation
  useEffect(() => {
    const updateMarketStatus = () => {
      const now = new Date();
      const hour = now.getHours();
      const day = now.getDay();
      
      if (day === 0 || day === 6) {
        setMarketStatus('closed'); // Weekend
      } else if (hour >= 9 && hour < 16) {
        setMarketStatus('open'); // Trading hours
      } else if (hour >= 4 && hour < 9) {
        setMarketStatus('pre-market');
      } else {
        setMarketStatus('after-hours');
      }
    };
    
    updateMarketStatus();
    const interval = setInterval(updateMarketStatus, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);
  
  // Keyboard navigation support
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // ESC to close mobile sidebar
      if (event.key === 'Escape' && sidebarOpen) {
        event.preventDefault();
        handleNavigationSidebarToggle();
      }
      
      // Ctrl/Cmd + B to toggle left sidebar (desktop only)
      if ((event.ctrlKey || event.metaKey) && event.key === 'b' && window.innerWidth >= 1024) {
        event.preventDefault();
        handleLeftSidebarToggle();
      }
      
      // Ctrl/Cmd + Shift + N to toggle navigation sidebar (mobile)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'N' && window.innerWidth < 1280) {
        event.preventDefault();
        handleNavigationSidebarToggle();
      }
      
      // Ctrl/Cmd + Shift + R to toggle right sidebar (desktop)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'R' && window.innerWidth >= 1024) {
        event.preventDefault();
        handleRightSidebarToggle();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (transitionTimeoutRef.current) {
        clearTimeout(transitionTimeoutRef.current);
      }
    };
  }, [sidebarOpen, handleNavigationSidebarToggle, handleLeftSidebarToggle, handleRightSidebarToggle]);

  // Error boundary
  if (error) {
    return <ErrorFallback error={error} resetError={() => setError(null)} />;
  }
  
  return (
    <div className="min-h-screen text-slate-100 font-mono relative overflow-hidden">
      {/* Enhanced Professional Background - matches new glassmorphism theme */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* Primary gradient overlay with enhanced glassmorphism */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950/95 via-slate-900/90 to-slate-950/98" />
        
        {/* Animated gradient orbs with professional opacity */}
        <div className="absolute inset-0 opacity-40">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/8 via-purple-500/6 to-emerald-500/8 glass-entrance" />
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl glass-float" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/8 rounded-full blur-3xl glass-float" style={{animationDelay: '3s'}} />
          <div className="absolute top-3/4 left-3/4 w-72 h-72 bg-emerald-500/6 rounded-full blur-2xl glass-float" style={{animationDelay: '6s'}} />
        </div>
        
        {/* Professional grid pattern with glassmorphism */}
        <div className="absolute inset-0 opacity-20">
          <div 
            className="w-full h-full"
            style={{
              backgroundImage: `
                radial-gradient(circle at 25% 25%, rgba(6, 182, 212, 0.12) 0%, transparent 25%),
                radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.08) 0%, transparent 25%),
                linear-gradient(rgba(6, 182, 212, 0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(6, 182, 212, 0.06) 1px, transparent 1px)
              `,
              backgroundSize: '600px 600px, 400px 400px, 50px 50px, 50px 50px'
            }}
          />
        </div>
      </div>

      {/* Enhanced Intelligent Sidebar Toggle Buttons */}
      <SidebarToggleButtons
        {...sidebarState}
        variant="floating"
        position="top-left"
        className="xl:hidden"
        showLabels={false}
      />
      
      {/* Desktop Advanced Toggle Controls */}
      {isDesktop && (
        <SidebarToggleButtons
          {...sidebarState}
          variant="floating"
          position="top-right"
          className="hidden xl:flex"
          showLabels={true}
        />
      )}

      {/* Left Sidebar Eye Toggle Button for Desktop */}
      {isDesktop && (
        <motion.button
          onClick={handleLeftSidebarToggle}
          className="fixed top-4 left-4 z-50 p-3 bg-slate-900/90 backdrop-blur-lg border border-slate-700/50 rounded-xl shadow-glass-lg hover:shadow-glass-xl transition-all duration-300 glass-button-primary hidden xl:flex items-center justify-center"
          variants={motionLibrary.components.glassButton}
          initial="initial"
          whileHover="hover"
          whileTap="tap"
          title={leftSidebarCollapsed ? "Show Left Sidebar" : "Hide Left Sidebar"}
        >
          <motion.div
            animate={{ rotate: leftSidebarCollapsed ? 0 : 180 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="text-cyan-400"
          >
            {leftSidebarCollapsed ? <Eye size={20} /> : <EyeOff size={20} />}
          </motion.div>
        </motion.button>
      )}

      {/* Right Sidebar Eye Toggle Button for Desktop */}
      {isDesktop && (
        <motion.button
          onClick={handleRightSidebarToggle}
          className="fixed top-4 right-4 z-50 p-3 bg-slate-900/90 backdrop-blur-lg border border-slate-700/50 rounded-xl shadow-glass-lg hover:shadow-glass-xl transition-all duration-300 glass-button-primary hidden xl:flex items-center justify-center"
          variants={motionLibrary.components.glassButton}
          initial="initial"
          whileHover="hover"
          whileTap="tap"
          title={rightSidebarCollapsed ? "Show Navigation Hub" : "Hide Navigation Hub"}
        >
          <motion.div
            animate={{ rotate: rightSidebarCollapsed ? 180 : 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="text-purple-400"
          >
            {rightSidebarCollapsed ? <Eye size={20} /> : <EyeOff size={20} />}
          </motion.div>
        </motion.button>
      )}

      {/* User Profile and Logout Button */}
      <div className="fixed top-4 right-20 z-50 flex items-center gap-3 hidden xl:flex">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="flex items-center gap-3 px-4 py-2 bg-slate-900/90 backdrop-blur-lg border border-slate-700/50 rounded-xl shadow-glass-lg"
        >
          <div className="flex items-center gap-2">
            <div className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
              <User className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm font-medium text-slate-300">{username}</span>
          </div>
          <div className="w-px h-6 bg-slate-700/50" />
          <motion.button
            onClick={handleLogout}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-all duration-200"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
            <span className="text-sm font-medium">Logout</span>
          </motion.button>
        </motion.div>
      </div>

      {/* Enhanced AnimatedSidebar Integration - Fully Responsive */}
      <div className={`fixed inset-y-0 left-0 z-40 transform transition-all duration-300 ease-in-out xl:translate-x-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="h-full flex">
          {/* AnimatedSidebar for controls - Fixed with proper handlers */}
          <div className="hidden lg:block">
            <AnimatedSidebar
              onPlayPause={handlePlayPause}
              onReset={handleReset}
              onAlignDataset={handleAlignDataset}
              isPlaying={isPlaying}
              isRealtime={isRealtime}
              dataSource={dataSource}
              onDataSourceChange={handleDataSourceChange}
              marketStatus={marketStatus}
              currentPrice={currentPrice}
              priceChange={priceChange}
              priceChangePercent={priceChangePercent}
              collapsed={leftSidebarCollapsed}
              onToggleCollapse={handleLeftSidebarToggle}
              isVisible={isLeftSidebarVisible}
              width={leftSidebarWidth}
            />
          </div>
          
          {/* Navigation Sidebar - Responsive */}
          <div className={`transition-all duration-300 ease-in-out bg-slate-900/95 backdrop-blur-xl border-r border-slate-700/50 h-full overflow-hidden ${
            rightSidebarCollapsed 
              ? 'w-0 opacity-0 pointer-events-none' 
              : 'w-80 sm:w-96 lg:w-80 opacity-100'
          }`}>
            {/* Header with gradient border - Responsive */}
            <div className="p-4 sm:p-6 border-b border-gradient-to-r from-cyan-500/20 to-purple-500/20">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg touch-manipulation">
                  <Sparkles className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                </div>
                <div className="min-w-0 flex-1">
                  <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent truncate">
                    Navigation Hub
                  </h1>
                  <p className="text-xs text-slate-400">16 Professional Views</p>
                </div>
              </div>
            </div>
            
            {/* Navigation */}
            <div className="flex-1 overflow-y-auto py-4 px-3">
              {categories.map((category) => (
                <div key={category} className="mb-6">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3 px-3">
                    {category}
                  </h3>
                  <div className="space-y-1">
                    {views.filter(view => view.category === category).map((view, index) => (
                      <motion.button
                        key={view.id}
                        onClick={() => handleViewChange(view.id)}
                        disabled={isTransitioning}
                        variants={motionLibrary.components.navItem}
                        initial="initial"
                        whileHover="hover"
                        whileTap="tap"
                        className={`w-full flex items-center gap-3 px-3 py-3 sm:py-4 rounded-xl transition-all duration-300 group disabled:opacity-50 touch-manipulation min-h-[56px] glass-card-interactive ${
                          activeView === view.id
                            ? 'glass-surface-primary border-l-2 border-cyan-400 text-cyan-300 shadow-glass-md'
                            : 'glass-surface-secondary text-slate-300 hover:text-cyan-300 hover:shadow-glass-sm'
                        }`}
                        transition={motionLibrary.utils.createTransition(0.2, motionLibrary.presets.easing.smooth, index * 0.05)}
                      >
                        <motion.div
                          className={`${
                            activeView === view.id ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'
                          } transition-colors duration-300`}
                          whileHover={{ scale: 1.1, rotate: 5 }}
                          transition={{ duration: 0.2 }}
                        >
                          <view.icon size={18} />
                        </motion.div>
                        <div className="flex-1 text-left min-w-0">
                          <motion.div 
                            className="text-sm font-medium truncate"
                            initial={{ x: 0 }}
                            whileHover={{ x: 2 }}
                            transition={{ duration: 0.2 }}
                          >
                            {view.name}
                          </motion.div>
                          <motion.div 
                            className="text-xs text-slate-500 group-hover:text-slate-400 line-clamp-1 sm:line-clamp-none transition-colors duration-300"
                            initial={{ opacity: 0.7 }}
                            whileHover={{ opacity: 1 }}
                          >
                            {view.description}
                          </motion.div>
                        </div>
                        <AnimatePresence>
                          {activeView === view.id && (
                            <motion.div
                              layoutId="activeViewIndicator"
                              className="w-3 h-3 bg-cyan-400 rounded-full shadow-glow-cyan"
                              variants={motionLibrary.micro.statusPulse}
                              animate="animate"
                              initial={{ scale: 0 }}
                              exit={{ scale: 0, opacity: 0 }}
                            />
                          )}
                        </AnimatePresence>
                        <AnimatePresence>
                          {isTransitioning && activeView === view.id && (
                            <motion.div 
                              className="w-4 h-4 border-2 border-cyan-400/20 border-t-cyan-400 rounded-full"
                              variants={motionLibrary.loading.spinner}
                              animate="animate"
                              initial={{ scale: 0, opacity: 0 }}
                              exit={{ scale: 0, opacity: 0 }}
                            />
                          )}
                        </AnimatePresence>
                      </motion.button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Enhanced System Status Footer */}
            <motion.div 
              className="border-t border-slate-700/50 p-4 glass-surface-secondary backdrop-blur-md"
              variants={motionLibrary.utils.createSlideAnimation('up', 10)}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.5 }}
            >
              <div className="flex items-center justify-between text-xs text-slate-400">
                <motion.div 
                  className="flex items-center gap-2"
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.2 }}
                >
                  <motion.div 
                    className="w-3 h-3 bg-green-400 rounded-full shadow-[0_0_8px_rgba(34,197,94,0.6)]"
                    variants={motionLibrary.micro.statusPulse}
                    animate="animate"
                  />
                  <span className="font-mono font-medium">All Systems Online</span>
                </motion.div>
                <motion.div 
                  className="text-slate-500 font-mono"
                  whileHover={{ color: '#64748b' }}
                  transition={{ duration: 0.2 }}
                >
                  v2.1.0
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Main Content - Intelligent Responsive Layout */}
      <div className={`relative z-10 transition-all duration-300 ease-in-out ${responsiveLayout.containerClasses} ${
        rightSidebarCollapsed ? 'xl:mr-0' : 'xl:mr-80'
      }`}
           style={{
             minWidth: responsiveLayout.isFullWidth ? '100%' : `${responsiveLayout.availableWidth}px`,
             maxWidth: responsiveLayout.isFullWidth ? 'none' : responsiveLayout.mainContentMaxWidth
           }}
      >
        {/* Enhanced Header Bar - Mobile optimized */}
        <motion.div 
          className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/50 p-3 sm:p-4"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center space-x-2 sm:space-x-4 min-w-0 flex-1">
              <div className="min-w-0 flex-1">
                <motion.h2 
                  className="text-base sm:text-lg font-semibold text-cyan-300 truncate"
                  key={activeView}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {activeViewData?.name}
                </motion.h2>
                <motion.p 
                  className="text-xs sm:text-sm text-slate-400 hidden sm:block"
                  key={`${activeView}-desc`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3, delay: 0.1 }}
                >
                  {activeViewData?.description}
                </motion.p>
              </div>
              
              {/* View category badge - Responsive */}
              <motion.div
                className="hidden sm:block px-3 py-1 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg border border-cyan-500/30"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <span className="text-xs font-mono text-cyan-300">{activeViewData?.category}</span>
              </motion.div>
            </div>
            
            <div className="flex items-center gap-2 sm:gap-4">
              {/* Data source indicator - Responsive */}
              <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
                <div className={`w-2 h-2 rounded-full ${
                  dataSource === 'l0' ? 'bg-blue-400' :
                  dataSource === 'l1' ? 'bg-green-400' : 'bg-purple-400'
                } animate-pulse`} />
                <span className="font-mono uppercase hidden lg:inline">{dataSource} DATA</span>
                <span className="font-mono uppercase lg:hidden">{dataSource}</span>
              </div>
              
              {/* Live status - Always visible but compact on mobile */}
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <motion.div 
                  className="w-2 h-2 bg-emerald-400 rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <span className="font-mono">{isRealtime ? 'LIVE' : 'HIST'}</span>
              </div>
            </div>
          </div>
          
          {/* Loading progress bar */}
          <AnimatePresence>
            {isLoading && (
              <motion.div
                className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              />
            )}
          </AnimatePresence>
        </motion.div>

        {/* Main Content Area with transitions - Smart Responsive Padding */}
        <main className={`${responsiveLayout.mainContentPaddingLeft} ${responsiveLayout.mainContentPaddingRight} relative`}
              style={{ paddingBottom: isMobile ? '8rem' : '1.5rem' }}> {/* Dynamic bottom padding for mobile controls */}
          <AnimatePresence mode="wait">
            {isLoading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2 }}
              >
                <LoadingSpinner />
              </motion.div>
            ) : (
              <motion.div
                key={activeView}
                ref={viewRef}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="relative"
              >
                <Suspense fallback={<LoadingSpinner />}>
                  <ActiveComponent />
                </Suspense>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Mobile Overlay - Enhanced with proper handler */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/60 z-30 xl:hidden backdrop-blur-sm touch-manipulation"
            onClick={handleNavigationSidebarToggle}
            onTouchEnd={handleNavigationSidebarToggle}
            aria-label="Close sidebar overlay"
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleNavigationSidebarToggle();
              }
            }}
          />
        )}
      </AnimatePresence>

      {/* Mobile Controls Bar - Trading controls for mobile */}
      <MobileControlsBar
        onPlayPause={handlePlayPause}
        onReset={handleReset}
        onAlignDataset={handleAlignDataset}
        isPlaying={isPlaying}
        isRealtime={isRealtime}
        dataSource={dataSource}
        onDataSourceChange={handleDataSourceChange}
        marketStatus={marketStatus}
        currentPrice={currentPrice}
        priceChange={priceChange}
        priceChangePercent={priceChangePercent}
      />
    </div>
  );
}