/**
 * Performance Status Component
 * Real-time display of system performance metrics
 */

import React from 'react';
import { usePerformanceMonitor } from '@/lib/metrics/performance-monitor';
import { Activity, Cpu, HardDrive, Zap, AlertTriangle } from 'lucide-react';

interface PerformanceStatusProps {
  compact?: boolean;
  showDetails?: boolean;
}

export const PerformanceStatus: React.FC<PerformanceStatusProps> = ({
  compact = false,
  showDetails = false
}) => {
  const metrics = usePerformanceMonitor();
  
  if (!metrics) {
    return compact ? null : (
      <div className="bg-terminal-surface border border-terminal-border rounded-lg p-4">
        <div className="animate-pulse flex items-center space-x-2">
          <Activity className="w-4 h-4 text-terminal-text-dim" />
          <span className="text-terminal-text-dim text-sm">Loading performance metrics...</span>
        </div>
      </div>
    );
  }
  
  // Determine status levels
  const getStatus = (metric: string, value: number) => {
    switch (metric) {
      case 'fps':
        if (value >= 55) return { level: 'excellent', color: 'text-positive' };
        if (value >= 30) return { level: 'good', color: 'text-terminal-accent' };
        if (value >= 20) return { level: 'poor', color: 'text-warning' };
        return { level: 'critical', color: 'text-negative' };
        
      case 'memory':
        if (value <= 100) return { level: 'excellent', color: 'text-positive' };
        if (value <= 300) return { level: 'good', color: 'text-terminal-accent' };
        if (value <= 500) return { level: 'poor', color: 'text-warning' };
        return { level: 'critical', color: 'text-negative' };
        
      default:
        return { level: 'unknown', color: 'text-terminal-text-dim' };
    }
  };
  
  const fpsStatus = metrics.fps ? getStatus('fps', metrics.fps.current) : null;
  const memoryStatus = metrics.memory ? getStatus('memory', metrics.memory.current) : null;
  
  // Compact view for header/navbar
  if (compact) {
    return (
      <div className="flex items-center space-x-3">
        {/* FPS Indicator */}
        {metrics.fps && (
          <div className="flex items-center space-x-1">
            <Activity className={`w-4 h-4 ${fpsStatus?.color}`} />
            <span className={`text-xs font-mono ${fpsStatus?.color}`}>
              {metrics.fps.current}fps
            </span>
          </div>
        )}
        
        {/* Memory Indicator */}
        {metrics.memory && (
          <div className="flex items-center space-x-1">
            <HardDrive className={`w-4 h-4 ${memoryStatus?.color}`} />
            <span className={`text-xs font-mono ${memoryStatus?.color}`}>
              {metrics.memory.current}MB
            </span>
          </div>
        )}
        
        {/* Warning indicator for critical issues */}
        {(fpsStatus?.level === 'critical' || memoryStatus?.level === 'critical') && (
          <AlertTriangle className="w-4 h-4 text-negative animate-pulse" />
        )}
      </div>
    );
  }
  
  // Full detailed view
  return (
    <div className="bg-terminal-surface border border-terminal-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-terminal-accent font-mono text-sm uppercase tracking-wider">
          Performance Monitor
        </h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-positive rounded-full animate-pulse" />
          <span className="text-xs text-terminal-text-dim">Live</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* FPS Metrics */}
        {metrics.fps && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Activity className={`w-4 h-4 ${fpsStatus?.color}`} />
                <span className="text-terminal-text text-sm font-medium">FPS</span>
              </div>
              <span className={`text-lg font-mono font-bold ${fpsStatus?.color}`}>
                {metrics.fps.current}
              </span>
            </div>
            
            {showDetails && (
              <div className="bg-terminal-surface-variant p-2 rounded text-xs font-mono space-y-1">
                <div className="flex justify-between">
                  <span className="text-terminal-text-muted">Average:</span>
                  <span className="text-terminal-text">{metrics.fps.average}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-text-muted">Min:</span>
                  <span className="text-terminal-text">{metrics.fps.min}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-text-muted">95th %:</span>
                  <span className="text-terminal-text">{metrics.fps.p95}</span>
                </div>
              </div>
            )}
            
            {/* FPS Bar */}
            <div className="w-full bg-terminal-surface-variant rounded-full h-2 overflow-hidden">
              <div 
                className={`h-full transition-all duration-300 ${
                  fpsStatus?.level === 'excellent' ? 'bg-positive' :
                  fpsStatus?.level === 'good' ? 'bg-terminal-accent' :
                  fpsStatus?.level === 'poor' ? 'bg-warning' : 'bg-negative'
                }`}
                style={{ width: `${Math.min(100, (metrics.fps.current / 60) * 100)}%` }}
              />
            </div>
          </div>
        )}
        
        {/* Memory Metrics */}
        {metrics.memory && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <HardDrive className={`w-4 h-4 ${memoryStatus?.color}`} />
                <span className="text-terminal-text text-sm font-medium">Memory</span>
              </div>
              <span className={`text-lg font-mono font-bold ${memoryStatus?.color}`}>
                {metrics.memory.current}MB
              </span>
            </div>
            
            {showDetails && (
              <div className="bg-terminal-surface-variant p-2 rounded text-xs font-mono space-y-1">
                <div className="flex justify-between">
                  <span className="text-terminal-text-muted">Average:</span>
                  <span className="text-terminal-text">{metrics.memory.average}MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-text-muted">Peak:</span>
                  <span className="text-terminal-text">{metrics.memory.max}MB</span>
                </div>
              </div>
            )}
            
            {/* Memory Bar */}
            <div className="w-full bg-terminal-surface-variant rounded-full h-2 overflow-hidden">
              <div 
                className={`h-full transition-all duration-300 ${
                  memoryStatus?.level === 'excellent' ? 'bg-positive' :
                  memoryStatus?.level === 'good' ? 'bg-terminal-accent' :
                  memoryStatus?.level === 'poor' ? 'bg-warning' : 'bg-negative'
                }`}
                style={{ width: `${Math.min(100, (metrics.memory.current / 1000) * 100)}%` }}
              />
            </div>
          </div>
        )}
        
        {/* Long Tasks */}
        {metrics.longTasks && showDetails && (
          <div className="space-y-2 md:col-span-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-warning" />
                <span className="text-terminal-text text-sm font-medium">Long Tasks</span>
              </div>
              <span className="text-lg font-mono font-bold text-terminal-text">
                {metrics.longTasks.count}
              </span>
            </div>
            
            <div className="bg-terminal-surface-variant p-2 rounded text-xs font-mono grid grid-cols-2 gap-4">
              <div className="flex justify-between">
                <span className="text-terminal-text-muted">Avg Duration:</span>
                <span className="text-terminal-text">{metrics.longTasks.averageDuration.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-terminal-text-muted">Max Duration:</span>
                <span className="text-terminal-text">{metrics.longTasks.maxDuration.toFixed(1)}ms</span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Status Summary */}
      {showDetails && (
        <div className="mt-4 p-3 bg-terminal-surface-variant rounded">
          <div className="flex items-center space-x-2 text-xs">
            <Cpu className="w-4 h-4 text-terminal-accent" />
            <span className="text-terminal-text-dim">
              System Status: 
            </span>
            <span className={`font-semibold ${
              (fpsStatus?.level === 'excellent' || fpsStatus?.level === 'good') &&
              (memoryStatus?.level === 'excellent' || memoryStatus?.level === 'good')
                ? 'text-positive' : 
              (fpsStatus?.level === 'critical' || memoryStatus?.level === 'critical')
                ? 'text-negative' : 'text-warning'
            }`}>
              {(fpsStatus?.level === 'excellent' || fpsStatus?.level === 'good') &&
               (memoryStatus?.level === 'excellent' || memoryStatus?.level === 'good')
                ? 'Optimal' :
               (fpsStatus?.level === 'critical' || memoryStatus?.level === 'critical')
                ? 'Critical' : 'Degraded'
              }
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceStatus;