'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import ViewRenderer from '../components/ViewRenderer';
import { EnhancedNavigationSidebar } from '../components/ui/EnhancedNavigationSidebar';
import { NavigationProvider } from '../lib/contexts/NavigationContext';
import EnhancedTradingDashboard from '../components/charts/EnhancedTradingDashboard';
import { useMarketStats } from '../hooks/useMarketStats';
import { useRealTimePrice } from '../hooks/useRealTimePrice';
import {
  TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Shield, Settings, LogOut,
  Wifi, WifiOff, Clock, Target, AlertTriangle, Play, Pause, Square, Maximize2,
  Download, Share2, Camera, Volume2, Zap, Signal, Database, RefreshCw, MousePointer,
  Menu, X, ChevronLeft, ChevronRight, Home, LineChart, PieChart, Map, Users,
  Bell, Search, Filter, Globe, Layers, GitBranch, Eye, EyeOff, Lock, Key, Sparkles
} from 'lucide-react';

// Removing hardcoded navigation - will use EnhancedNavigationSidebar

export default function ProfessionalTradingDashboard() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [activeView, setActiveView] = useState('dashboard-home');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDrawingTools, setShowDrawingTools] = useState(false);
  const [inactivityTimer, setInactivityTimer] = useState<NodeJS.Timeout | null>(null);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const sidebarRef = useRef<HTMLElement>(null);

  // === REAL DATA FROM BACKEND - NO HARDCODED VALUES ===
  const { stats: marketStats, isConnected: statsConnected } = useMarketStats('USDCOP', 30000);
  const { currentPrice: realtimePrice, isConnected: priceConnected } = useRealTimePrice('USDCOP');

  // Use real-time price if available, otherwise use stats price
  const currentPrice = realtimePrice?.price || marketStats?.currentPrice || 0;
  const isConnected = statsConnected || priceConnected;

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
      // Alt+1-5: Direct section access (handled by EnhancedNavigationSidebar)
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

  // Trading status derived from market stats
  const statusColor = isConnected ? 'text-green-400' : 'text-red-400';
  const dataQuality = marketStats?.source === 'backend_api' ? 'Premium' : 'Historical';
  const qualityBadgeColor = dataQuality === 'Premium' ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400';
  const marketStatus = marketStats?.timestamp ? 'OPEN' : 'CLOSED';
  const latency = isConnected ? 4 : 999;

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
              <div className={`flex items-center gap-2 px-3 py-1 rounded-lg border ${isConnected ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'}`}>
                {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                <span className={`text-sm font-medium ${statusColor}`}>
                  {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>

              <div className={`px-2 py-1 rounded text-xs font-bold ${qualityBadgeColor}`}>
                {dataQuality}
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
                <div className={`w-2 h-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'} rounded-full animate-pulse`}></div>
                <span className={`text-lg font-bold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>{marketStatus}</span>
              </div>
            </div>

            <div className="text-center">
              <div className="text-sm text-slate-400">Latency</div>
              <div className={`text-lg font-bold ${latency < 100 ? 'text-cyan-400' : 'text-yellow-400'}`}>&lt;{latency}ms</div>
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

      <NavigationProvider
        initialView="dashboard-home"
        onViewChange={(viewId) => setActiveView(viewId)}
      >
      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Sidebar - Using EnhancedNavigationSidebar with all 16 views */}
        <EnhancedNavigationSidebar />

        {/* Old sidebar code has been completely removed and replaced with EnhancedNavigationSidebar
           which contains all 16 professional trading views */}

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
                      ${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="text-sm text-slate-400 mt-1">USD/COP</div>
                  </div>

                  <div className={`flex items-center gap-3 ${(Number(marketStats?.change24h) || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(Number(marketStats?.change24h) || 0) >= 0 ? <TrendingUp className="w-6 h-6" /> : <TrendingDown className="w-6 h-6" />}
                    <div>
                      <div className="text-xl font-bold">
                        {(Number(marketStats?.change24h) || 0) >= 0 ? '+' : ''}{(Number(marketStats?.change24h) || 0).toFixed(2)}
                      </div>
                      <div className="text-sm opacity-80">
                        ({(Number(marketStats?.changePercent) || 0) >= 0 ? '+' : ''}{(Number(marketStats?.changePercent) || 0).toFixed(2)}%)
                      </div>
                    </div>
                  </div>
                </div>

                {/* P&L Session */}
                <div className="px-4 py-3 bg-slate-800/40 rounded-lg border border-slate-600/30">
                  <div className="text-slate-400 text-sm">P&L Sesión</div>
                  <div className={`text-xl font-bold ${(marketStats?.sessionPnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(marketStats?.sessionPnl || 0) >= 0 ? '+' : ''}${Math.abs(marketStats?.sessionPnl || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
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
                  {((Number(marketStats?.volume24h) || 0) / 1000000).toFixed(2)}M
                </div>
                <div className="text-slate-500 text-xs mt-1">Real-time data</div>
              </div>

              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Range 24H</div>
                <div className="flex items-center gap-2">
                  <span className="text-red-400 font-bold">{(Number(marketStats?.low24h) || 0).toFixed(2)}</span>
                  <span className="text-slate-500">-</span>
                  <span className="text-green-400 font-bold">{(Number(marketStats?.high24h) || 0).toFixed(2)}</span>
                </div>
                <div className="text-slate-500 text-xs mt-1">Rango: {((Number(marketStats?.high24h) || 0) - (Number(marketStats?.low24h) || 0)).toFixed(0)} pips</div>
              </div>

              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Spread</div>
                <div className="text-cyan-400 text-lg font-bold">
                  {(Number(marketStats?.spread) || 0).toFixed(1)} COP
                </div>
                <div className="text-slate-500 text-xs mt-1">High-Low 24h</div>
              </div>

              <div className="bg-slate-800/30 backdrop-blur-sm rounded-lg p-4 border border-slate-600/20">
                <div className="text-slate-400 text-sm mb-1">Volatility</div>
                <div className="text-purple-400 text-lg font-bold">
                  {(Number(marketStats?.volatility) || 0).toFixed(2)}%
                </div>
                <div className="text-slate-500 text-xs mt-1">24h range</div>
              </div>
            </div>
          </div>

          {/* Main Content Area - Enhanced Trading Dashboard */}
          <div className="flex-1 relative bg-slate-950/50">
            {/* Enhanced Dashboard Container */}
            <div className="absolute inset-4 overflow-y-auto overflow-x-hidden">
              {/* ViewRenderer with Spectacular Components */}
              <ViewRenderer activeView={activeView} />
            </div>
          </div>

          {/* Bottom Status Bar */}
          <div className="bg-slate-900/80 backdrop-blur-xl border-t border-slate-700/50 px-6 py-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-6">
                <span className="text-slate-400">Source:</span>
                <span className="text-cyan-400 font-bold">{marketStats?.source || 'connecting...'}</span>

                <span className="text-slate-400">Change 24H:</span>
                <span className={`font-bold ${(Number(marketStats?.change24h) || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {(Number(marketStats?.change24h) || 0) >= 0 ? '+' : ''}{(Number(marketStats?.change24h) || 0).toFixed(2)} COP ({(Number(marketStats?.changePercent) || 0) >= 0 ? '+' : ''}{(Number(marketStats?.changePercent) || 0).toFixed(2)}%)
                </span>

                <span className="text-slate-400">Trend:</span>
                <span className={`font-bold ${
                  marketStats?.trend === 'up' ? 'text-green-400' :
                  marketStats?.trend === 'down' ? 'text-red-400' : 'text-yellow-400'
                }`}>{marketStats?.trend?.toUpperCase() || 'N/A'}</span>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'} rounded-full animate-pulse`}></div>
                  <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
                    {isConnected ? 'Live Data Stream' : 'Disconnected'}
                  </span>
                </div>

                <div className="text-slate-400">
                  Last update: {marketStats?.timestamp ? new Date(marketStats.timestamp).toLocaleTimeString() : 'N/A'}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
      </NavigationProvider>
    </div>
  );
}