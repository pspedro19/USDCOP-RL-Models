'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Play, 
  Pause,
  RotateCcw,
  Zap,
  Database,
  Shield,
  Eye,
  TrendingUp,
  TrendingDown,
  Clock,
  Activity
} from 'lucide-react';

interface MobileControlsBarProps {
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
}

export function MobileControlsBar({
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
  priceChangePercent = 0.38
}: MobileControlsBarProps) {
  const marketStatusConfig = {
    'open': { color: 'text-green-400', bg: 'bg-green-500/20', label: 'Open' },
    'closed': { color: 'text-red-400', bg: 'bg-red-500/20', label: 'Closed' },
    'pre-market': { color: 'text-yellow-400', bg: 'bg-yellow-500/20', label: 'Pre' },
    'after-hours': { color: 'text-purple-400', bg: 'bg-purple-500/20', label: 'AH' }
  };

  const status = marketStatusConfig[marketStatus];

  const controlButtons = [
    {
      id: 'play',
      icon: isPlaying ? Pause : Play,
      label: isPlaying ? 'Pause' : 'Play',
      onClick: onPlayPause,
      color: isPlaying ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400',
      active: isPlaying
    },
    {
      id: 'reset',
      icon: RotateCcw,
      label: 'Reset',
      onClick: onReset,
      color: 'bg-blue-500/20 text-blue-400',
      active: false
    },
    {
      id: 'align',
      icon: Zap,
      label: 'Live',
      onClick: onAlignDataset,
      color: isRealtime ? 'bg-purple-500/20 text-purple-400' : 'bg-slate-500/20 text-slate-400',
      active: isRealtime
    }
  ];

  const dataSources = [
    { value: 'l1', label: 'L1', icon: Shield, active: dataSource === 'l1' },
    { value: 'l0', label: 'L0', icon: Database, active: dataSource === 'l0' },
    { value: 'mock', label: 'Mock', icon: Eye, active: dataSource === 'mock' }
  ];

  return (
    <motion.div
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed bottom-0 left-0 right-0 z-50 xl:hidden bg-slate-900/95 backdrop-blur-xl border-t border-slate-700/50 safe-area-padding-bottom"
    >
      <div className="p-4 space-y-3">
        {/* Price Display - Compact */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="text-sm font-bold text-white">
                  ${currentPrice.toFixed(3)}
                </div>
                <div className={`text-xs flex items-center gap-1 ${priceChange >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {priceChange >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  <span>{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)</span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Market Status - Compact */}
          <div className={`${status.bg} ${status.color} px-2 py-1 rounded-md flex items-center gap-1 text-xs`}>
            <Clock className="w-3 h-3" />
            <span>{status.label}</span>
          </div>
        </div>

        {/* Controls Row */}
        <div className="flex items-center justify-between gap-2">
          {/* Control Buttons */}
          <div className="flex items-center gap-2 flex-1">
            {controlButtons.map((button) => (
              <motion.button
                key={button.id}
                onClick={button.onClick}
                whileTap={{ scale: 0.95 }}
                className={`${button.color} p-3 rounded-xl flex items-center gap-2 transition-all touch-manipulation min-h-[48px] ${
                  button.active ? 'ring-2 ring-current ring-opacity-50' : ''
                }`}
              >
                <button.icon className="w-4 h-4" />
                <span className="text-xs font-medium">{button.label}</span>
              </motion.button>
            ))}
          </div>

          {/* Data Source Selector */}
          <div className="flex items-center gap-1 bg-slate-800/50 rounded-xl p-1">
            {dataSources.map((source) => (
              <motion.button
                key={source.value}
                onClick={() => onDataSourceChange(source.value as 'l0' | 'l1' | 'mock')}
                whileTap={{ scale: 0.95 }}
                className={`p-2 rounded-lg flex items-center gap-1 transition-all touch-manipulation min-h-[40px] ${
                  source.active 
                    ? 'bg-blue-500/20 text-blue-400 ring-1 ring-blue-500/50' 
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                <source.icon className="w-3 h-3" />
                <span className="text-xs font-medium">{source.label}</span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center justify-center gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full ${
              dataSource === 'l0' ? 'bg-blue-400' :
              dataSource === 'l1' ? 'bg-green-400' : 'bg-purple-400'
            } animate-pulse`} />
            <span className="font-mono uppercase">{dataSource} DATA</span>
          </div>
          
          <div className="flex items-center gap-1">
            <motion.div 
              className="w-2 h-2 bg-emerald-400 rounded-full"
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            <span className="font-mono">{isRealtime ? 'LIVE' : 'HIST'}</span>
          </div>
        </div>
      </div>

      {/* Safe area padding for devices with home indicator */}
      <div className="h-safe-area-inset-bottom" />
    </motion.div>
  );
}