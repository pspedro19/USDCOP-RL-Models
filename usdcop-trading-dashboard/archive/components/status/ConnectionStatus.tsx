/**
 * Connection Status Component
 * Displays connection status with latency-based coloring and semantic indicators
 */

import React, { useEffect, useState } from 'react';
import { useMarketStore } from '@/lib/store/market-store';
import { Activity, AlertCircle, CheckCircle, Clock, WifiOff, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ConnectionStatusProps {
  compact?: boolean;
  showLatency?: boolean;
  showDataRate?: boolean;
  className?: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  compact = false,
  showLatency = true,
  showDataRate = false,
  className
}) => {
  const { connectionStatus, dataSource } = useMarketStore();
  const [latency, setLatency] = useState<number>(0);
  const [dataRate, setDataRate] = useState<number>(0);
  const [lastPing, setLastPing] = useState<number>(Date.now());
  
  // Measure latency
  useEffect(() => {
    const measureLatency = async () => {
      const start = performance.now();
      try {
        // Ping the API endpoint
        const response = await fetch('/api/market/health', { 
          method: 'HEAD',
          cache: 'no-cache' 
        });
        if (response.ok) {
          const end = performance.now();
          setLatency(Math.round(end - start));
          setLastPing(Date.now());
        }
      } catch (error) {
        setLatency(-1); // Indicates error
      }
    };
    
    // Measure immediately and then every 5 seconds
    measureLatency();
    const interval = setInterval(measureLatency, 5000);
    
    return () => clearInterval(interval);
  }, [connectionStatus]);
  
  // Track data rate (messages per second)
  useEffect(() => {
    let messageCount = 0;
    const resetCount = () => {
      setDataRate(messageCount);
      messageCount = 0;
    };
    
    // Reset counter every second
    const interval = setInterval(resetCount, 1000);
    
    // Listen for market updates (you'd connect this to your WebSocket)
    const handleMessage = () => {
      messageCount++;
    };
    
    // Add event listener for market updates
    window.addEventListener('market-update', handleMessage);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('market-update', handleMessage);
    };
  }, []);
  
  // Determine status color and icon based on connection and latency
  const getStatusConfig = () => {
    if (connectionStatus === 'disconnected') {
      return {
        icon: WifiOff,
        color: 'text-negative',
        bgColor: 'bg-down-10',
        borderColor: 'border-negative',
        label: 'Offline',
        pulse: false
      };
    }
    
    if (connectionStatus === 'reconnecting') {
      return {
        icon: AlertCircle,
        color: 'text-status-delayed',
        bgColor: 'bg-neutral-bg',
        borderColor: 'border-status-delayed',
        label: 'Reconnecting',
        pulse: true
      };
    }
    
    // Connected - determine quality by latency
    if (latency < 0) {
      return {
        icon: AlertCircle,
        color: 'text-status-delayed',
        bgColor: 'bg-neutral-bg',
        borderColor: 'border-status-delayed',
        label: 'Degraded',
        pulse: false
      };
    }
    
    if (latency < 50) {
      return {
        icon: Zap,
        color: 'text-status-live',
        bgColor: 'bg-up-10',
        borderColor: 'border-status-live',
        label: 'Excellent',
        pulse: false
      };
    }
    
    if (latency < 150) {
      return {
        icon: CheckCircle,
        color: 'text-positive',
        bgColor: 'bg-positive-bg',
        borderColor: 'border-positive',
        label: 'Good',
        pulse: false
      };
    }
    
    if (latency < 300) {
      return {
        icon: Clock,
        color: 'text-status-delayed',
        bgColor: 'bg-neutral-bg',
        borderColor: 'border-status-delayed',
        label: 'Slow',
        pulse: false
      };
    }
    
    return {
      icon: AlertCircle,
      color: 'text-negative',
      bgColor: 'bg-negative-bg',
      borderColor: 'border-negative',
      label: 'Poor',
      pulse: true
    };
  };
  
  const config = getStatusConfig();
  const Icon = config.icon;
  
  // Format latency display
  const formatLatency = (ms: number) => {
    if (ms < 0) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };
  
  // Compact view for navbar/header
  if (compact) {
    return (
      <div className={cn(
        'flex items-center space-x-2 px-2 py-1 rounded-md border',
        config.bgColor,
        config.borderColor,
        config.pulse && 'animate-pulse',
        className
      )}>
        <Icon className={cn('w-4 h-4', config.color)} />
        {showLatency && latency > 0 && (
          <span className={cn('text-xs font-mono', config.color)}>
            {formatLatency(latency)}
          </span>
        )}
      </div>
    );
  }
  
  // Full detailed view
  return (
    <div className={cn(
      'bg-terminal-surface border border-terminal-border rounded-lg p-4',
      className
    )}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-terminal-accent font-mono text-sm uppercase tracking-wider">
          Connection Status
        </h3>
        <div className={cn(
          'flex items-center space-x-2 px-3 py-1 rounded-full border',
          config.bgColor,
          config.borderColor,
          config.pulse && 'animate-pulse'
        )}>
          <Icon className={cn('w-4 h-4', config.color)} />
          <span className={cn('text-xs font-semibold', config.color)}>
            {config.label}
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        {/* Data Source */}
        <div className="space-y-1">
          <div className="text-terminal-text-muted text-xs uppercase">Source</div>
          <div className="text-terminal-text font-mono text-sm">
            {dataSource.toUpperCase()}
          </div>
        </div>
        
        {/* Latency */}
        {showLatency && (
          <div className="space-y-1">
            <div className="text-terminal-text-muted text-xs uppercase">Latency</div>
            <div className={cn('font-mono text-sm font-bold', config.color)}>
              {formatLatency(latency)}
            </div>
          </div>
        )}
        
        {/* Data Rate */}
        {showDataRate && (
          <div className="space-y-1">
            <div className="text-terminal-text-muted text-xs uppercase">Data Rate</div>
            <div className="text-terminal-text font-mono text-sm flex items-center space-x-1">
              <Activity className="w-3 h-3" />
              <span>{dataRate} msg/s</span>
            </div>
          </div>
        )}
        
        {/* Last Update */}
        <div className="space-y-1">
          <div className="text-terminal-text-muted text-xs uppercase">Last Ping</div>
          <div className="text-terminal-text font-mono text-xs">
            {new Date(lastPing).toLocaleTimeString()}
          </div>
        </div>
      </div>
      
      {/* Latency Bar Visualization */}
      {showLatency && latency > 0 && (
        <div className="mt-4">
          <div className="w-full bg-terminal-surface-variant rounded-full h-2 overflow-hidden">
            <div 
              className={cn(
                'h-full transition-all duration-500',
                latency < 50 && 'bg-status-live',
                latency >= 50 && latency < 150 && 'bg-positive',
                latency >= 150 && latency < 300 && 'bg-status-delayed',
                latency >= 300 && 'bg-negative'
              )}
              style={{ 
                width: `${Math.min(100, (latency / 500) * 100)}%` 
              }}
            />
          </div>
          <div className="flex justify-between mt-1 text-xs text-terminal-text-muted font-mono">
            <span>0ms</span>
            <span>250ms</span>
            <span>500ms+</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConnectionStatus;