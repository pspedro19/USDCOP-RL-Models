'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChartCandlestick, 
  Activity, 
  Settings, 
  Database, 
  Play, 
  Pause,
  RotateCcw,
  Zap,
  Calendar,
  Clock,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Layers,
  BarChart3,
  LineChart,
  Target,
  Shield,
  Eye,
  EyeOff
} from 'lucide-react';

interface SidebarProps {
  onPlayPause: () => void;
  onReset: () => void;
  onAlignDataset: () => void;
  isPlaying: boolean;
  isRealtime: boolean;
  dataSource: 'l0' | 'l1' | 'mock';
  onDataSourceChange: (source: 'l0' | 'l1' | 'mock') => void;
  marketStatus: 'open' | 'closed' | 'pre-market' | 'after-hours';
  currentPrice?: number;
  priceChange?: number;
  priceChangePercent?: number;
  collapsed?: boolean;
  onToggleCollapse: () => void;
  isVisible?: boolean; // New prop for visibility control
  width?: number; // New prop for dynamic width
}

export function AnimatedSidebar({ 
  onPlayPause, 
  onReset, 
  onAlignDataset, 
  isPlaying, 
  isRealtime,
  dataSource,
  onDataSourceChange,
  marketStatus,
  currentPrice = 4150.25,
  priceChange = 15.75,
  priceChangePercent = 0.38,
  collapsed = false,
  onToggleCollapse,
  isVisible = true,
  width
}: SidebarProps) {
  // Remove internal expanded state - use collapsed prop directly
  const expanded = !collapsed;
  const [activeSection, setActiveSection] = useState<string | null>(null);

  // Sync activeSection with current dataSource
  useEffect(() => {
    setActiveSection(dataSource);
  }, [dataSource]);

  const menuItems = [
    {
      id: 'chart',
      icon: ChartCandlestick,
      label: 'Chart View',
      description: 'Main trading chart',
      color: 'text-blue-400'
    },
    {
      id: 'indicators',
      icon: Activity,
      label: 'Indicators',
      description: 'Technical indicators',
      color: 'text-purple-400'
    },
    {
      id: 'analysis',
      icon: BarChart3,
      label: 'Analysis',
      description: 'Market analysis',
      color: 'text-green-400'
    },
    {
      id: 'settings',
      icon: Settings,
      label: 'Settings',
      description: 'Configuration',
      color: 'text-gray-400'
    }
  ];

  const controlButtons = [
    {
      id: 'play',
      icon: isPlaying ? Pause : Play,
      label: isPlaying ? 'Pause' : 'Play Historical',
      onClick: onPlayPause,
      color: isPlaying ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400',
      description: 'Replay from last trading day'
    },
    {
      id: 'reset',
      icon: RotateCcw,
      label: 'Reset View',
      onClick: onReset,
      color: 'bg-blue-500/20 text-blue-400',
      description: 'Reset to current data'
    },
    {
      id: 'align',
      icon: Zap,
      label: 'Align Dataset',
      onClick: onAlignDataset,
      color: isRealtime ? 'bg-purple-500/20 text-purple-400 animate-pulse' : 'bg-purple-500/20 text-purple-400',
      description: 'Start real-time updates'
    }
  ];

  // Fixed data sources to match actual system values
  const dataSources = [
    { value: 'l1', label: 'L1 Features', icon: ChartCandlestick },
    { value: 'l0', label: 'L0 Raw Data', icon: Database },
    { value: 'mock', label: 'Mock Data', icon: Activity }
  ];

  const marketStatusConfig = {
    'open': { color: 'text-green-400', bg: 'bg-green-500/20', label: 'Market Open' },
    'closed': { color: 'text-red-400', bg: 'bg-red-500/20', label: 'Market Closed' },
    'pre-market': { color: 'text-yellow-400', bg: 'bg-yellow-500/20', label: 'Pre-Market' },
    'after-hours': { color: 'text-purple-400', bg: 'bg-purple-500/20', label: 'After Hours' }
  };

  const status = marketStatusConfig[marketStatus];

  // Don't render if not visible
  if (!isVisible) {
    return null;
  }

  return (
    <motion.div
      initial={{ x: -100, opacity: 0 }}
      animate={{ 
        x: 0, 
        opacity: 1,
        width: width || (expanded ? 320 : 64)
      }}
      exit={{ x: -100, opacity: 0 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className={`bg-slate-900/95 backdrop-blur-xl border-r border-slate-700/50 h-full flex flex-col relative overflow-hidden touch-manipulation`}
      style={{
        width: width || (expanded ? '20rem' : '4rem'),
        minWidth: width || (expanded ? '20rem' : '4rem'),
        maxWidth: width || (expanded ? '20rem' : '4rem')
      }}
    >
      {/* Sidebar Glow Effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 to-purple-500/5" />
      <div className="absolute top-0 bottom-0 left-0 w-[1px] bg-gradient-to-b from-transparent via-blue-500/30 to-transparent" />
      
      {/* Header */}
      <div className="p-4 border-b border-slate-700/50 relative z-10">
        <motion.div 
          className="flex items-center justify-between cursor-pointer"
          onClick={onToggleCollapse}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          role="button"
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onToggleCollapse();
            }
          }}
        >
          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center gap-3"
              >
                <div className="relative">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/25">
                    <TrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-r from-emerald-400 to-green-500 rounded-full border border-slate-900 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-white font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">USD/COP</h2>
                  <p className="text-xs text-slate-400">Professional Trading</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            {expanded ? <EyeOff className="w-5 h-5 text-slate-400" /> : <Eye className="w-5 h-5 text-slate-400" />}
          </motion.div>
        </motion.div>
      </div>

      {/* Market Status */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-4 border-b border-slate-700/50 relative z-10"
          >
            <div className={`${status.bg} ${status.color} px-3 py-2 rounded-lg flex items-center justify-between mb-3`}>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">{status.label}</span>
              </div>
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="w-2 h-2 bg-current rounded-full"
              />
            </div>
            
            {/* Enhanced Price Display */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-4 border border-slate-700/30 relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5" />
              <div className="relative z-10">
              <div className="flex items-center justify-between mb-3">
                <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Current Price</span>
                {priceChange >= 0 ? (
                  <TrendingUp className="w-4 h-4 text-green-400" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-400" />
                )}
              </div>
              <motion.div
                key={currentPrice}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-2xl font-bold text-white mb-1"
              >
                ${currentPrice?.toFixed(3) || '0.000'}
              </motion.div>
              <div className={`flex items-center gap-2 text-sm ${(priceChange || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                <span className="font-semibold">{(priceChange || 0) >= 0 ? '+' : ''}{(priceChange || 0).toFixed(2)}</span>
                <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                  (priceChangePercent || 0) >= 0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'
                }`}>({(priceChangePercent || 0) >= 0 ? '+' : ''}{(priceChangePercent || 0).toFixed(2)}%)</span>
              </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Control Buttons */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-4 space-y-2 border-b border-slate-700/50 relative z-10"
          >
            {controlButtons.map((button, index) => (
              <motion.button
                key={button.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={button.onClick}
                whileHover={{ scale: 1.02, x: 5 }}
                whileTap={{ scale: 0.98 }}
                className={`w-full ${button.color} p-3 rounded-lg flex items-center gap-3 transition-all`}
              >
                <button.icon className="w-5 h-5" />
                <div className="text-left flex-1">
                  <p className="font-medium">{button.label}</p>
                  <p className="text-xs opacity-70">{button.description}</p>
                </div>
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Access Menu - Simplified */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex-1 p-4 space-y-2"
          >
            <p className="text-slate-400 text-xs mb-3 uppercase tracking-wider font-medium">Quick Access</p>
            {dataSources.map((item, index) => (
              <motion.button
                key={item.value}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => {
                  onDataSourceChange(item.value as 'l0' | 'l1' | 'mock');
                  setActiveSection(item.value);
                }}
                whileHover={{ scale: 1.02, x: 5 }}
                whileTap={{ scale: 0.98 }}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-all ${
                  dataSource === item.value
                    ? 'bg-slate-800 border border-slate-700 text-cyan-300' 
                    : 'hover:bg-slate-800/50 text-slate-300 hover:text-cyan-300'
                }`}
                role="button"
                aria-pressed={dataSource === item.value}
              >
                <item.icon className={`w-5 h-5 ${
                  dataSource === item.value ? 'text-cyan-400' : 'text-slate-400'
                }`} />
                <div className="text-left flex-1">
                  <p className="font-medium text-sm">{item.label}</p>
                  <p className="text-xs opacity-70">
                    {item.value === 'l1' ? 'Processed features' : 
                     item.value === 'l0' ? 'Raw market data' : 'Test data'}
                  </p>
                </div>
                {dataSource === item.value && (
                  <motion.div
                    layoutId="activeDataSource"
                    className="w-2 h-2 bg-cyan-400 rounded-full shadow-glow-cyan"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  />
                )}
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Collapsed Icons - Enhanced with tooltips */}
      {!expanded && (
        <motion.div 
          className="flex-1 p-2 space-y-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {controlButtons.map((button, index) => (
            <motion.button
              key={button.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.1, x: 5 }}
              whileTap={{ scale: 0.9 }}
              onClick={button.onClick}
              className={`w-full p-3 rounded-lg flex justify-center ${button.color} group relative`}
              title={button.label}
            >
              <button.icon className="w-5 h-5" />
              
              {/* Tooltip on hover */}
              <motion.div
                className="absolute left-full ml-2 px-2 py-1 bg-slate-800 text-slate-300 text-xs rounded-md opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50"
                initial={{ opacity: 0, x: -10 }}
                whileHover={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                {button.label}
              </motion.div>
            </motion.button>
          ))}
          
          {/* Collapsed Data Source Indicators */}
          <div className="mt-4 space-y-2">
            {dataSources.map((item, index) => (
              <motion.button
                key={item.value}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: (controlButtons.length + index) * 0.1 }}
                onClick={() => {
                  onDataSourceChange(item.value as 'l0' | 'l1' | 'mock');
                  setActiveSection(item.value);
                }}
                whileHover={{ scale: 1.1, x: 5 }}
                whileTap={{ scale: 0.9 }}
                className={`w-full p-2 rounded-lg flex justify-center transition-all group relative ${
                  dataSource === item.value
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' 
                    : 'hover:bg-slate-800/50 text-slate-400 hover:text-cyan-400'
                }`}
                title={item.label}
              >
                <item.icon className="w-4 h-4" />
                
                {/* Data source tooltip */}
                <motion.div
                  className="absolute left-full ml-2 px-2 py-1 bg-slate-800 text-slate-300 text-xs rounded-md opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50"
                  initial={{ opacity: 0, x: -10 }}
                  whileHover={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {item.label}
                </motion.div>
                
                {/* Active indicator */}
                {dataSource === item.value && (
                  <motion.div
                    className="absolute -right-1 top-1/2 w-2 h-2 bg-cyan-400 rounded-full transform -translate-y-1/2"
                    layoutId="collapsedActiveSource"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    exit={{ scale: 0 }}
                  />
                )}
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Footer - Enhanced for both states */}
      <AnimatePresence>
        {expanded ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="p-4 border-t border-slate-700/50 relative z-10"
          >
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span className="font-medium">Last Update</span>
              <span className="font-mono bg-slate-800/50 px-2 py-1 rounded" suppressHydrationWarning>
                {typeof window !== 'undefined' ? new Date().toLocaleTimeString() : '00:00:00'}
              </span>
            </div>
            
            {/* Connection status */}
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-700/30">
              <span className="flex items-center gap-1 text-xs text-slate-500">
                <motion.div 
                  className="w-2 h-2 bg-green-400 rounded-full"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                Connected
              </span>
              <span className="text-xs font-mono text-slate-600">
                v2.1.0
              </span>
            </div>
          </motion.div>
        ) : (
          // Collapsed footer with minimal status
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-2 border-t border-slate-700/50 relative z-10 flex justify-center"
          >
            <motion.div 
              className="w-3 h-3 bg-green-400 rounded-full"
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              title="System Online"
            />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}