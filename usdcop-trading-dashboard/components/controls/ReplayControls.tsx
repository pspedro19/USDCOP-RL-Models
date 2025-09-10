/**
 * Replay Controls Component
 * Professional controls for historical data replay and aligned dataset loading
 * Enhanced with glassmorphism, glow effects, and smooth animations
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Play, Pause, RotateCcw, Download, Zap, Database, Cloud, Info } from 'lucide-react';
import { useMarketStore } from '@/lib/store/market-store';
import { EnhancedDataService } from '@/lib/services/enhanced-data-service';

interface ReplayControlsProps {
  onAlignedDatasetClick?: () => void;
  onRealtimeToggle?: (enabled: boolean) => void;
}

export const ReplayControls: React.FC<ReplayControlsProps> = ({
  onAlignedDatasetClick,
  onRealtimeToggle
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [dataStats, setDataStats] = useState({
    total: 0,
    minio: 0,
    twelvedata: 0,
    realtime: 0
  });
  const [isRealtimeEnabled, setIsRealtimeEnabled] = useState(false);
  
  const { 
    candles,
    setReplayState,
    connectionStatus,
    dataSource 
  } = useMarketStore();
  
  // Handle play/pause
  const handlePlayPause = useCallback(() => {
    const newState = !isPlaying;
    setIsPlaying(newState);
    setReplayState(newState, replaySpeed);
  }, [isPlaying, replaySpeed, setReplayState]);
  
  // Handle speed change
  const handleSpeedChange = useCallback((speed: number) => {
    setReplaySpeed(speed);
    if (isPlaying) {
      setReplayState(true, speed);
    }
  }, [isPlaying, setReplayState]);
  
  // Handle reset
  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setReplayState(false);
    // Reset to beginning of data
  }, [setReplayState]);
  
  // Handle aligned dataset loading
  const handleAlignedDataset = useCallback(async () => {
    setIsLoading(true);
    try {
      console.log('[ReplayControls] Loading aligned dataset...');
      
      // Initialize service
      const dataService = new EnhancedDataService();
      
      // Load complete history from MinIO
      const historicalData = await dataService.loadCompleteHistory();
      console.log(`[ReplayControls] Loaded ${historicalData.length} historical points`);
      
      // Get date range for gap filling
      const now = new Date();
      const lastHistoricalDate = historicalData.length > 0 
        ? new Date(historicalData[historicalData.length - 1].datetime)
        : new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
      
      // Fill gaps with TwelveData
      const alignedData = await dataService.createAlignedDataset(
        lastHistoricalDate,
        now
      );
      
      console.log(`[ReplayControls] Created aligned dataset with ${alignedData.length} total points`);
      
      // Update stats
      setDataStats({
        total: alignedData.length,
        minio: alignedData.filter(d => d.source === 'minio').length,
        twelvedata: alignedData.filter(d => d.source === 'twelvedata').length,
        realtime: alignedData.filter(d => d.source === 'realtime').length
      });
      
      // Trigger callback
      if (onAlignedDatasetClick) {
        onAlignedDatasetClick();
      }
      
    } catch (error) {
      console.error('[ReplayControls] Error loading aligned dataset:', error);
    } finally {
      setIsLoading(false);
    }
  }, [onAlignedDatasetClick]);
  
  // Handle realtime toggle
  const handleRealtimeToggle = useCallback(() => {
    const newState = !isRealtimeEnabled;
    setIsRealtimeEnabled(newState);
    if (onRealtimeToggle) {
      onRealtimeToggle(newState);
    }
  }, [isRealtimeEnabled, onRealtimeToggle]);
  
  // Update stats when candles change
  useEffect(() => {
    if (candles.length > 0) {
      setDataStats({
        total: candles.length,
        minio: candles.filter(d => d.source === 'minio').length,
        twelvedata: candles.filter(d => d.source === 'twelvedata').length,
        realtime: candles.filter(d => d.source === 'realtime').length
      });
    }
  }, [candles]);
  
  return (
    <div className="relative bg-slate-900/80 backdrop-blur-md border border-cyan-400/20 rounded-xl p-6 shadow-2xl shadow-cyan-400/10 hover:shadow-cyan-400/20 transition-all duration-500">
      {/* Glassmorphism overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 rounded-xl pointer-events-none" />
      
      {/* Animated border glow */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-400/20 via-emerald-400/20 to-purple-400/20 opacity-0 hover:opacity-100 transition-opacity duration-700 -z-10 blur-sm" />
      
      <div className="relative flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
            Replay Controls
          </h3>
          
          {/* Glassmorphism tooltip trigger */}
          <div className="group relative">
            <Info size={14} className="text-slate-400 hover:text-cyan-400 transition-colors cursor-help" />
            <div className="absolute left-0 top-6 invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-all duration-300 z-50">
              <div className="bg-slate-900/95 backdrop-blur-md border border-cyan-400/30 rounded-lg p-3 text-xs text-slate-200 whitespace-nowrap shadow-xl">
                Advanced replay controls with real-time data integration
                <div className="absolute -top-1 left-4 w-2 h-2 bg-slate-900 border-l border-t border-cyan-400/30 rotate-45"></div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Enhanced connection status */}
        <div className="flex items-center space-x-3 bg-slate-800/60 backdrop-blur-sm rounded-full px-3 py-1.5 border border-slate-700/50">
          <div className={`relative w-2.5 h-2.5 rounded-full ${
            connectionStatus === 'connected' ? 'bg-emerald-400' : 
            connectionStatus === 'reconnecting' ? 'bg-yellow-400' : 
            'bg-red-400'
          }`}>
            {connectionStatus === 'connected' && (
              <div className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-75"></div>
            )}
            {connectionStatus === 'reconnecting' && (
              <div className="absolute inset-0 rounded-full bg-yellow-400 animate-pulse opacity-60"></div>
            )}
          </div>
          <span className="text-slate-300 text-xs font-mono capitalize">
            {connectionStatus}
          </span>
        </div>
      </div>
      
      {/* Enhanced Control buttons */}
      <div className="flex items-center space-x-4 mb-6">
        {/* Play/Pause with glow */}
        <button
          onClick={handlePlayPause}
          disabled={isLoading}
          className={`group relative p-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
            isPlaying 
              ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50' 
              : 'bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 hover:text-white border border-slate-600/50 hover:border-cyan-400/50'
          } disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none`}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          {isPlaying && (
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 opacity-20 animate-pulse"></div>
          )}
        </button>
        
        {/* Reset with hover effect */}
        <button
          onClick={handleReset}
          disabled={isLoading}
          className="group relative p-3 rounded-xl bg-slate-800/60 hover:bg-red-500/20 text-slate-300 hover:text-red-400 border border-slate-600/50 hover:border-red-400/50 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
        >
          <RotateCcw size={18} className="group-hover:animate-spin" />
        </button>
        
        {/* Speed selector pills */}
        <div className="flex items-center space-x-1 bg-slate-800/40 backdrop-blur-sm rounded-full p-1 border border-slate-700/50">
          {[0.5, 1, 2, 5, 10].map((speed) => (
            <button
              key={speed}
              onClick={() => handleSpeedChange(speed)}
              disabled={isLoading}
              className={`px-3 py-1.5 rounded-full text-xs font-mono font-medium transition-all duration-300 transform hover:scale-105 ${
                replaySpeed === speed
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              } disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none`}
            >
              {speed}x
            </button>
          ))}
        </div>
        
        {/* Aligned Dataset button with loading animation */}
        <button
          onClick={handleAlignedDataset}
          disabled={isLoading}
          className="group relative px-4 py-2.5 rounded-xl bg-gradient-to-r from-slate-800 to-slate-700 hover:from-cyan-600 hover:to-blue-600 text-slate-300 hover:text-white border border-slate-600/50 hover:border-cyan-400/50 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
        >
          <div className="flex items-center space-x-2">
            <Database size={16} className={isLoading ? 'animate-spin' : 'group-hover:animate-pulse'} />
            <span className="text-xs font-medium">
              {isLoading ? 'Loading...' : 'Aligned Dataset'}
            </span>
          </div>
          {isLoading && (
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-500/20 to-blue-500/20 animate-pulse"></div>
          )}
        </button>
        
        {/* Realtime toggle with advanced styling */}
        <button
          onClick={handleRealtimeToggle}
          className={`group relative px-4 py-2.5 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg ${
            isRealtimeEnabled 
              ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 text-white shadow-emerald-500/30 hover:shadow-emerald-500/50' 
              : 'bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 hover:text-white border border-slate-600/50 hover:border-emerald-400/50'
          }`}
        >
          <div className="flex items-center space-x-2">
            <Zap size={16} className={isRealtimeEnabled ? 'animate-pulse' : ''} />
            <span className="text-xs font-medium">Real-time</span>
          </div>
          {isRealtimeEnabled && (
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 opacity-20 animate-ping"></div>
          )}
        </button>
      </div>
      
      {/* Enhanced Data statistics */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {/* Total stats card */}
        <div className="group relative bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-xl p-3 hover:border-cyan-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-400/10">
          <div className="text-slate-400 text-xs font-mono mb-1">Total</div>
          <div className="text-slate-100 font-bold text-lg font-mono">{dataStats.total.toLocaleString()}</div>
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
        
        {/* MinIO stats card */}
        <div className="group relative bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-xl p-3 hover:border-emerald-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-400/10">
          <div className="text-slate-400 text-xs font-mono mb-1 flex items-center space-x-1">
            <Database size={10} />
            <span>MinIO</span>
          </div>
          <div className="text-emerald-400 font-bold text-lg font-mono">{dataStats.minio.toLocaleString()}</div>
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-400/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
        
        {/* TwelveData stats card */}
        <div className="group relative bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-xl p-3 hover:border-purple-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-purple-400/10">
          <div className="text-slate-400 text-xs font-mono mb-1 flex items-center space-x-1">
            <Cloud size={10} />
            <span>TwelveData</span>
          </div>
          <div className="text-purple-400 font-bold text-lg font-mono">{dataStats.twelvedata.toLocaleString()}</div>
          <div className="absolute inset-0 bg-gradient-to-br from-purple-400/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
        
        {/* Real-time stats card */}
        <div className="group relative bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-xl p-3 hover:border-yellow-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-yellow-400/10">
          <div className="text-slate-400 text-xs font-mono mb-1 flex items-center space-x-1">
            <Zap size={10} className={dataStats.realtime > 0 ? 'animate-pulse text-yellow-400' : ''} />
            <span>Real-time</span>
          </div>
          <div className="text-yellow-400 font-bold text-lg font-mono">{dataStats.realtime.toLocaleString()}</div>
          <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/5 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          {dataStats.realtime > 0 && (
            <div className="absolute top-1 right-1 w-2 h-2 bg-yellow-400 rounded-full animate-ping"></div>
          )}
        </div>
      </div>
      
      {/* Enhanced animated progress bar */}
      {isPlaying && (
        <div className="space-y-3">
          <div className="relative">
            {/* Progress track */}
            <div className="h-2 bg-slate-800/60 rounded-full overflow-hidden backdrop-blur-sm border border-slate-700/30">
              {/* Animated progress fill with gradient and glow */}
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 relative transition-all duration-500 ease-out"
                style={{ width: '45%' }}
              >
                {/* Shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
                
                {/* Glow effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 blur-sm opacity-50"></div>
              </div>
            </div>
            
            {/* Progress thumb */}
            <div 
              className="absolute top-1/2 transform -translate-y-1/2 w-4 h-4 bg-gradient-to-r from-cyan-400 to-emerald-400 rounded-full shadow-lg shadow-cyan-400/50 border-2 border-white/20 transition-all duration-500 ease-out"
              style={{ left: 'calc(45% - 8px)' }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-emerald-400 rounded-full animate-ping opacity-60"></div>
            </div>
          </div>
          
          {/* Time labels with enhanced styling */}
          <div className="flex justify-between text-xs font-mono">
            <div className="flex flex-col">
              <span className="text-slate-400">Start</span>
              <span className="text-slate-300 font-medium">Aug 22, 2024</span>
            </div>
            <div className="flex flex-col text-center">
              <span className="text-cyan-400">Current</span>
              <span className="text-white font-medium">Nov 15, 2024</span>
            </div>
            <div className="flex flex-col text-right">
              <span className="text-slate-400">End</span>
              <span className="text-slate-300 font-medium">Today</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReplayControls;