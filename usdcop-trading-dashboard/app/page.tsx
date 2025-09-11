'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import ViewRenderer from '../components/ViewRenderer';
import { 
  TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Shield, Settings, LogOut,
  Wifi, WifiOff, Clock, Target, AlertTriangle, Play, Pause, Square, Maximize2,
  Download, Share2, Camera, Volume2, Zap, Signal, Database, RefreshCw, MousePointer,
  Menu, X, ChevronLeft, ChevronRight, Home, LineChart, PieChart, Map, Users,
  Bell, Search, Filter, Globe, Layers, GitBranch, Eye, EyeOff, Lock, Key
} from 'lucide-react';

// Real-time market data simulation
const useMarketData = () => {
  const [data, setData] = useState({
    price: 4010.91,
    change: 63.47,
    changePercent: 1.58,
    volume: 1847329,
    high24h: 4165.50,
    low24h: 3890.25,
    spread: 0.08,
    liquidity: 98.7,
    volatility: 0.89,
    timestamp: new Date()
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => ({
        ...prev,
        price: prev.price + (Math.random() - 0.5) * 2,
        change: prev.price * 0.001 * (Math.random() - 0.5) * 10,
        timestamp: new Date()
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return data;
};

// Trading status hook
const useTradingStatus = () => {
  const [status, setStatus] = useState({
    isConnected: true,
    dataQuality: 'Premium',
    latency: 4,
    mode: 'Normal', // Normal, Contingency, Analysis, Simulation
    marketStatus: 'OPEN'
  });

  return status;
};

// Reorganized navigation items (max 5 groups, hierarchical)
const navigationItems = [
  {
    section: 'DASHBOARD',
    color: 'text-white',
    priority: 'always-visible',
    items: [
      { id: 'terminal', icon: Home, label: 'Dashboard Home', active: true, status: 'healthy', hotkey: 'Alt+1' }
    ]
  },
  {
    section: 'TRADING',
    color: 'text-cyan-400', 
    priority: 'critical',
    collapsed: false,
    items: [
      { id: 'live-terminal', icon: Activity, label: 'Terminal', status: 'healthy', badge: null, hotkey: 'Alt+2', kpi: 'P&L: +$4.2K' },
      { id: 'signals', icon: Zap, label: 'Signals', status: 'processing', badge: 3, hotkey: 'Alt+3', kpi: '3 señales activas' },
      { id: 'backtest', icon: GitBranch, label: 'Backtest', status: 'healthy', badge: null, hotkey: 'Alt+4', kpi: 'ROI: 12.5%' }
    ]
  },
  {
    section: 'RISK',
    color: 'text-red-400',
    priority: 'critical', 
    collapsed: false,
    items: [
      { id: 'realtime-risk', icon: AlertTriangle, label: 'Monitor', status: 'warning', badge: 2, kpi: 'VaR: $145K | DD: -2.3%' },
      { id: 'alerts', icon: Bell, label: 'Alerts', status: 'critical', badge: 5, kpi: '5 alertas críticas' }
    ]
  },
  {
    section: 'PIPELINE',
    color: 'text-purple-400',
    priority: 'secondary',
    collapsed: true,
    items: [
      { id: 'l0', icon: Database, label: 'L0 Raw Data', status: 'healthy', badge: null },
      { id: 'l1', icon: BarChart3, label: 'L1 Features', status: 'healthy', badge: null },
      { id: 'l2', icon: Shield, label: 'L2 Quality', status: 'warning', badge: 1 },
      { id: 'l3', icon: Users, label: 'L3 Correlations', status: 'healthy', badge: null },
      { id: 'l4', icon: Target, label: 'L4 RL Ready', status: 'healthy', badge: null },
      { id: 'l5', icon: Layers, label: 'L5 Serving', status: 'healthy', badge: null }
    ]
  },
  {
    section: 'EXECUTIVE',
    color: 'text-green-400',
    priority: 'secondary',
    collapsed: false,
    items: [
      { id: 'executive', icon: Globe, label: 'Overview', status: 'healthy', badge: null, kpi: 'Monthly: +8.7%' }
    ]
  }
];

export default function ProfessionalTradingDashboard() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [activeView, setActiveView] = useState('terminal');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDrawingTools, setShowDrawingTools] = useState(false);
  const [inactivityTimer, setInactivityTimer] = useState<NodeJS.Timeout | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);
  
  const marketData = useMarketData();
  const tradingStatus = useTradingStatus();

  // Keyboard shortcuts and auto-hide logic
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+M: Toggle menu
      if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        setSidebarCollapsed(!sidebarCollapsed);
      }
      // Ctrl+K: Command palette
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
      // Ctrl+H: Zen mode (hide sidebar)
      if (e.ctrlKey && e.key === 'h') {
        e.preventDefault();
        setSidebarHidden(!sidebarHidden);
      }
      // Esc: Close menus
      if (e.key === 'Escape') {
        setSidebarCollapsed(true);
        setShowCommandPalette(false);
      }
      // Alt+1-5: Direct section access
      if (e.altKey && ['1', '2', '3', '4', '5'].includes(e.key)) {
        e.preventDefault();
        const sectionIndex = parseInt(e.key) - 1;
        const section = navigationItems[sectionIndex];
        if (section && section.items.length > 0) {
          setActiveView(section.items[0].id);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [sidebarCollapsed, sidebarHidden]);

  // Auto-hide behavior
  useEffect(() => {
    const resetTimer = () => {
      if (inactivityTimer) clearTimeout(inactivityTimer);
      
      const timer = setTimeout(() => {
        if (!sidebarCollapsed) {
          setSidebarCollapsed(true);
        }
      }, 5000); // 5 seconds of inactivity
      
      setInactivityTimer(timer);
    };

    const handleActivity = () => {
      resetTimer();
    };

    // Reset timer on any interaction
    document.addEventListener('mousemove', handleActivity);
    document.addEventListener('keydown', handleActivity);
    document.addEventListener('click', handleActivity);

    return () => {
      if (inactivityTimer) clearTimeout(inactivityTimer);
      document.removeEventListener('mousemove', handleActivity);
      document.removeEventListener('keydown', handleActivity);
      document.removeEventListener('click', handleActivity);
    };
  }, [sidebarCollapsed, inactivityTimer]);

  // Load sidebar preferences from localStorage
  useEffect(() => {
    const savedCollapsed = localStorage.getItem('sidebarCollapsed');
    const savedHidden = localStorage.getItem('sidebarHidden');
    
    if (savedCollapsed !== null) {
      setSidebarCollapsed(savedCollapsed === 'true');
    }
    if (savedHidden !== null) {
      setSidebarHidden(savedHidden === 'true');
    }
  }, []);

  // Save sidebar state to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarCollapsed', sidebarCollapsed.toString());
  }, [sidebarCollapsed]);

  useEffect(() => {
    localStorage.setItem('sidebarHidden', sidebarHidden.toString());
  }, [sidebarHidden]);

  useEffect(() => {
    // Check authentication
    const authStatus = sessionStorage.getItem('isAuthenticated') || localStorage.getItem('isAuthenticated');
    if (!authStatus || authStatus !== 'true') {
      router.push('/login');
      return;
    }
    setIsAuthenticated(true);
    setIsLoading(false);
  }, [router]);

  const handleLogout = () => {
    sessionStorage.removeItem('isAuthenticated');
    localStorage.removeItem('isAuthenticated');
    sessionStorage.removeItem('username');
    localStorage.removeItem('username');
    router.push('/login');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
          <h2 className="text-2xl font-bold text-white mb-2">USD/COP Professional Trading Terminal</h2>
          <p className="text-slate-400">Initializing trading systems...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  const statusColor = tradingStatus.isConnected ? 'text-green-400' : 'text-red-400';
  const qualityBadgeColor = tradingStatus.dataQuality === 'Premium' ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400';

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}} />
          <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-green-500/10 rounded-full blur-2xl animate-pulse" style={{animationDelay: '4s'}} />
        </div>
      </div>

      {/* Top Status Bar */}
      <header className="relative z-50 bg-slate-900/95 backdrop-blur-xl border-b border-slate-700/50 px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Logo & Status */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg"
              >
                <Activity className="w-5 h-5 text-white" />
              </motion.div>
              <div>
                <h1 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  USD/COP Professional Trading Terminal
                </h1>
                <p className="text-xs text-slate-400">Dashboard Unificado</p>
              </div>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-lg border ${tradingStatus.isConnected ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                {tradingStatus.isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                <span className={`text-sm font-medium ${statusColor}`}>
                  {tradingStatus.isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>
              
              <div className={`px-2 py-1 rounded text-xs font-bold ${qualityBadgeColor}`}>
                {tradingStatus.dataQuality}
              </div>
            </div>
          </div>

          {/* Center: Market Status */}
          <div className="flex items-center gap-6">
            <div className="text-center">
              <div className="text-sm text-slate-400">Local Time</div>
              <div className="text-lg font-mono font-bold text-white">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-sm text-slate-400">Market</div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-lg font-bold text-green-400">{tradingStatus.marketStatus}</span>
              </div>
            </div>

            <div className="text-center">
              <div className="text-sm text-slate-400">Latency</div>
              <div className="text-lg font-bold text-cyan-400">&lt;{tradingStatus.latency}ms</div>
            </div>
          </div>

          {/* Right: Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-600/50 text-slate-300 transition-all"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
            
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-3 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg border border-slate-600/50 text-slate-300 transition-all"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden md:inline">Logout</span>
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Sidebar - Navigation Hub */}
        <motion.aside
          initial={false}
          animate={{ 
            width: sidebarHidden ? 0 : (sidebarCollapsed ? 48 : 280),
            opacity: sidebarHidden ? 0 : 1
          }}
          transition={{ duration: 0.15, ease: [0.4, 0, 0.2, 1] }}
          className="relative z-40 bg-slate-950/95 backdrop-blur-xl border-r border-slate-700/30 overflow-hidden group"
          onMouseEnter={() => {
            if (inactivityTimer) clearTimeout(inactivityTimer);
          }}
          onMouseLeave={() => {
            // Auto-collapse after delay
            const timer = setTimeout(() => setSidebarCollapsed(true), 3000);
            setInactivityTimer(timer);
          }}
        >
          <div className="p-2">
            {/* Hamburger menu when collapsed */}
            {sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex flex-col items-center space-y-4 mb-6"
              >
                <button
                  onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                  className="relative p-3 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 hover:from-cyan-500/30 hover:to-purple-500/30 rounded-xl border border-cyan-500/30 transition-all duration-300 group"
                >
                  <Menu className="w-6 h-6 text-cyan-400 group-hover:text-cyan-300" />
                  
                  {/* Floating tooltip */}
                  <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-slate-800/95 backdrop-blur-sm text-cyan-400 text-sm font-medium px-3 py-2 rounded-lg border border-cyan-500/30 opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap pointer-events-none">
                    Abrir Navegación
                    <div className="absolute top-1/2 -left-1 -translate-y-1/2 w-2 h-2 bg-slate-800 border-l border-t border-cyan-500/30 rotate-45"></div>
                  </div>
                </button>
                
                {/* Quick action indicators */}
                <div className="flex flex-col space-y-2">
                  <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" title="Trading Active"></div>
                  <div className="w-2 h-2 bg-green-500 rounded-full" title="Systems Healthy"></div>
                  <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" title="Processing"></div>
                </div>
              </motion.div>
            )}
            
            {/* Full toggle button when expanded */}
            {!sidebarCollapsed && (
              <button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="w-full flex items-center justify-center p-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg transition-all mb-4"
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
            )}

            {/* Collapsed state - icon-only navigation */}
            {sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col space-y-3"
              >
                {navigationItems.map((section) => (
                  <div key={section.section} className="space-y-2">
                    {section.items.map((item) => (
                      <div key={item.id} className="relative group">
                        <button
                          onClick={() => setActiveView(item.id)}
                          className={`w-full p-3 rounded-lg transition-all duration-200 flex items-center justify-center relative ${
                            item.active || activeView === item.id
                              ? 'bg-cyan-500/20 border border-cyan-500/30 text-cyan-400'
                              : 'hover:bg-slate-800/50 text-slate-300 hover:text-cyan-400'
                          }`}
                        >
                          <item.icon className="w-5 h-5" />
                          {item.status && (
                            <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-slate-900 ${
                              item.status === 'healthy' ? 'bg-green-500' :
                              item.status === 'warning' ? 'bg-yellow-500' :
                              item.status === 'processing' ? 'bg-blue-500 animate-pulse' : 'bg-red-500'
                            }`} />
                          )}
                        </button>
                        
                        {/* Floating tooltip */}
                        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 bg-slate-800/95 backdrop-blur-sm text-white text-sm font-medium px-3 py-2 rounded-lg border border-slate-600/50 opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap pointer-events-none z-50">
                          {item.label}
                          <div className="absolute top-1/2 -left-1 -translate-y-1/2 w-2 h-2 bg-slate-800 border-l border-t border-slate-600/50 rotate-45"></div>
                        </div>
                      </div>
                    ))}
                    
                    {/* Section divider */}
                    <div className="h-px bg-slate-700/50 my-2"></div>
                  </div>
                ))}
              </motion.div>
            )}
            
            {/* Expanded state - full navigation */}
            {!sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                {navigationItems.map((section, sectionIndex) => (
                  <div key={section.section}>
                    <h3 className={`text-xs font-bold ${section.color} mb-3 tracking-wide`}>
                      {section.section}
                    </h3>
                    <div className="space-y-1">
                      {section.items.map((item) => (
                        <button
                          key={item.id}
                          onClick={() => setActiveView(item.id)}
                          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left ${
                            item.active || activeView === item.id
                              ? 'bg-cyan-500/20 border border-cyan-500/30 text-cyan-400'
                              : 'hover:bg-slate-800/50 text-slate-300'
                          }`}
                        >
                          <item.icon className="w-4 h-4 flex-shrink-0" />
                          <span className="text-sm">{item.label}</span>
                          {item.status && (
                            <div className={`w-2 h-2 rounded-full ml-auto ${
                              item.status === 'healthy' ? 'bg-green-500' :
                              item.status === 'warning' ? 'bg-yellow-500' :
                              item.status === 'processing' ? 'bg-blue-500 animate-pulse' : 'bg-red-500'
                            }`} />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </div>
        </motion.aside>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* NIVEL 1: Precio Principal (Always Visible) */}
          <div className="bg-slate-900/60 backdrop-blur-xl border-b border-slate-700/50 px-6 py-5">
            <div className="flex items-center justify-between">
              {/* KPIs Críticos */}
              <div className="flex items-center gap-12">
                {/* Precio USD/COP */}
                <div className="flex items-center gap-6">
                  <div>
                    <div className="text-4xl font-bold text-white leading-none">
                      ${marketData.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-sm text-slate-400 mt-1">USD/COP</div>
                  </div>
                  
                  <div className={`flex items-center gap-3 ${marketData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {marketData.change >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                    <div>
                      <div className="text-xl font-bold">
                        {marketData.change >= 0 ? '+' : ''}{marketData.change.toFixed(2)}
                      </div>
                      <div className="text-sm opacity-80">
                        ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                </div>

                {/* P&L Session */}
                <div className="px-4 py-3 bg-slate-800/40 rounded-lg border border-slate-600/30">
                  <div className="text-slate-400 text-sm">P&L Sesión</div>
                  <div className="text-green-400 text-xl font-bold">+$1,247.85</div>
                </div>
              </div>

              {/* Time Controls */}
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-slate-800/50 rounded-lg p-1">
                  {['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'].map((timeframe) => (
                    <button
                      key={timeframe}
                      className={`px-3 py-1.5 rounded text-xs font-bold transition-all ${
                        timeframe === 'M5' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {timeframe}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* NIVEL 2: Métricas de Sesión (Cards Espaciadas) */}
          <div className="bg-slate-900/40 backdrop-blur-sm border-b border-slate-700/30 px-6 py-4">
            <div className="grid grid-cols-4 gap-8">
              {/* Trading Stats */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Volume 24H</div>
                <div className="text-white text-lg font-bold">
                  {(marketData.volume / 1000000).toFixed(2)}M
                </div>
                <div className="text-slate-500 text-xs mt-1">+12.5% vs avg</div>
              </div>
              
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Range 24H</div>
                <div className="flex items-center gap-2">
                  <span className="text-red-400 font-bold">{marketData.low24h.toFixed(2)}</span>
                  <span className="text-slate-500">-</span>
                  <span className="text-green-400 font-bold">{marketData.high24h.toFixed(2)}</span>
                </div>
                <div className="text-slate-500 text-xs mt-1">Rango: {(marketData.high24h - marketData.low24h).toFixed(0)} pips</div>
              </div>

              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Spread</div>
                <div className="text-cyan-400 text-lg font-bold">
                  {marketData.spread.toFixed(1)} bps
                </div>
                <div className="text-slate-500 text-xs mt-1">Target: &lt;21.5 bps</div>
              </div>

              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Liquidity</div>
                <div className="text-purple-400 text-lg font-bold">
                  {marketData.liquidity.toFixed(1)}%
                </div>
                <div className="text-slate-500 text-xs mt-1">Optimal: &gt;95%</div>
              </div>
            </div>
          </div>

          {/* Main Content Area - Dynamic Views */}
          <div className="flex-1 relative bg-slate-950/50">
            {/* View Container */}
            <div className="absolute inset-4 bg-slate-900/40 backdrop-blur-sm rounded-xl border border-slate-700/30 overflow-y-auto overflow-x-hidden">
              {/* Dynamic View Content */}
              <ViewRenderer activeView={activeView} />
            </div>
          </div>

          {/* Bottom Status Bar */}
          <div className="bg-slate-900/80 backdrop-blur-xl border-t border-slate-700/50 px-6 py-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-6">
                <span className="text-slate-400">Mode:</span>
                <span className="text-cyan-400 font-bold">{tradingStatus.mode}</span>
                
                <span className="text-slate-400">P&L 24H:</span>
                <span className="text-green-400 font-bold">+$4,250.75 (+1.06%)</span>
                
                <span className="text-slate-400">Drawdown:</span>
                <span className="text-yellow-400 font-bold">-2.3%</span>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400">Live Data Stream</span>
                </div>
                
                <div className="text-slate-400">
                  Last update: {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}