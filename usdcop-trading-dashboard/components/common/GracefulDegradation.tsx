/**
 * Graceful Degradation Components
 * Fallback UI components for when main features fail
 */

'use client';

import React, { useState, useEffect, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  RefreshCw,
  Download,
  Table,
  LineChart,
  Activity,
  Wifi,
  WifiOff,
  Database,
  Clock,
  Info,
  Zap
} from 'lucide-react';

interface FallbackProps {
  children: ReactNode;
  fallback: ReactNode;
  errorMessage?: string;
}

// Base fallback wrapper
export const FallbackWrapper: React.FC<FallbackProps> = ({ 
  children, 
  fallback, 
  errorMessage = 'Feature temporarily unavailable' 
}) => {
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const handleError = () => setHasError(true);
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (hasError) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

// Chart fallback component
interface ChartFallbackProps {
  title?: string;
  error?: Error;
  onRetry?: () => void;
  showStaticData?: boolean;
  staticData?: Array<{ label: string; value: number; change?: number }>;
  height?: number;
}

export const ChartFallback: React.FC<ChartFallbackProps> = ({
  title = 'Chart',
  error,
  onRetry,
  showStaticData = false,
  staticData = [],
  height = 400
}) => {
  const [isRetrying, setIsRetrying] = useState(false);

  const handleRetry = async () => {
    if (onRetry && !isRetrying) {
      setIsRetrying(true);
      try {
        await onRetry();
      } finally {
        setTimeout(() => setIsRetrying(false), 1000);
      }
    }
  };

  return (
    <div 
      className="flex items-center justify-center bg-terminal-surface border border-terminal-border rounded-lg"
      style={{ height: `${height}px` }}
    >
      <div className="text-center p-6 max-w-md">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="space-y-4"
        >
          {/* Icon */}
          <div className="mx-auto w-16 h-16 bg-yellow-500/10 rounded-full flex items-center justify-center border border-yellow-500/30">
            <BarChart3 className="w-8 h-8 text-yellow-400" />
          </div>

          {/* Title and message */}
          <div>
            <h3 className="text-lg font-semibold text-terminal-text mb-2">
              {title} Unavailable
            </h3>
            <p className="text-terminal-text-dim text-sm mb-4">
              {error 
                ? `Chart failed to load: ${error.message}` 
                : 'Chart is temporarily unavailable. This may be due to network issues or high server load.'
              }
            </p>
          </div>

          {/* Static data preview */}
          {showStaticData && staticData.length > 0 && (
            <div className="bg-terminal-surface-variant p-3 rounded border border-terminal-border">
              <h4 className="text-sm font-medium text-terminal-text mb-2">Latest Data:</h4>
              <div className="space-y-1">
                {staticData.slice(0, 3).map((item, index) => (
                  <div key={index} className="flex justify-between items-center text-xs">
                    <span className="text-terminal-text-dim">{item.label}</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-terminal-text font-mono">
                        {typeof item.value === 'number' ? item.value.toFixed(2) : item.value}
                      </span>
                      {item.change !== undefined && (
                        <span className={`flex items-center ${
                          item.change > 0 ? 'text-green-400' : item.change < 0 ? 'text-red-400' : 'text-terminal-text-dim'
                        }`}>
                          <TrendingUp className={`w-3 h-3 ${item.change < 0 ? 'rotate-180' : ''}`} />
                          <span className="ml-1">{Math.abs(item.change).toFixed(1)}%</span>
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-col sm:flex-row gap-2 justify-center">
            {onRetry && (
              <button
                onClick={handleRetry}
                disabled={isRetrying}
                className="terminal-button px-4 py-2 rounded flex items-center justify-center space-x-2"
              >
                <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                <span>{isRetrying ? 'Loading...' : 'Retry Chart'}</span>
              </button>
            )}
            
            <button
              onClick={() => window.location.reload()}
              className="border border-terminal-border px-4 py-2 rounded text-terminal-text hover:bg-terminal-surface-variant transition-colors flex items-center justify-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh Page</span>
            </button>
          </div>

          {/* Alternative data access */}
          <div className="pt-2 border-t border-terminal-border">
            <p className="text-xs text-terminal-text-dim mb-2">
              Alternative data access:
            </p>
            <div className="flex justify-center space-x-4">
              <button className="text-terminal-accent hover:text-terminal-text text-xs flex items-center space-x-1">
                <Table className="w-3 h-3" />
                <span>Table View</span>
              </button>
              <button className="text-terminal-accent hover:text-terminal-text text-xs flex items-center space-x-1">
                <Download className="w-3 h-3" />
                <span>Export Data</span>
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

// Data loading fallback
interface DataFallbackProps {
  title?: string;
  description?: string;
  onRetry?: () => void;
  showSkeleton?: boolean;
  skeletonRows?: number;
}

export const DataFallback: React.FC<DataFallbackProps> = ({
  title = 'Data',
  description = 'Unable to load data at this time',
  onRetry,
  showSkeleton = true,
  skeletonRows = 3
}) => {
  const [isRetrying, setIsRetrying] = useState(false);

  const handleRetry = async () => {
    if (onRetry && !isRetrying) {
      setIsRetrying(true);
      try {
        await onRetry();
      } finally {
        setTimeout(() => setIsRetrying(false), 1000);
      }
    }
  };

  if (showSkeleton) {
    return (
      <div className="space-y-3">
        {Array.from({ length: skeletonRows }).map((_, index) => (
          <div key={index} className="animate-pulse">
            <div className="h-4 bg-terminal-surface-variant rounded w-full"></div>
            <div className="h-3 bg-terminal-surface-variant rounded w-3/4 mt-2"></div>
          </div>
        ))}
        {onRetry && (
          <button
            onClick={handleRetry}
            disabled={isRetrying}
            className="mt-4 terminal-button px-3 py-1 rounded text-sm flex items-center space-x-2"
          >
            <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
            <span>{isRetrying ? 'Loading...' : 'Retry'}</span>
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="terminal-card p-4 text-center">
      <Database className="w-8 h-8 text-terminal-text-dim mx-auto mb-3" />
      <h3 className="text-terminal-text font-medium mb-2">{title} Unavailable</h3>
      <p className="text-terminal-text-dim text-sm mb-4">{description}</p>
      {onRetry && (
        <button
          onClick={handleRetry}
          disabled={isRetrying}
          className="terminal-button px-4 py-2 rounded flex items-center space-x-2 mx-auto"
        >
          <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
          <span>{isRetrying ? 'Loading...' : 'Retry'}</span>
        </button>
      )}
    </div>
  );
};

// Network error fallback
interface NetworkFallbackProps {
  onRetry?: () => void;
  showOfflineMode?: boolean;
}

export const NetworkFallback: React.FC<NetworkFallbackProps> = ({
  onRetry,
  showOfflineMode = false
}) => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isRetrying, setIsRetrying] = useState(false);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRetry = async () => {
    if (onRetry && !isRetrying) {
      setIsRetrying(true);
      try {
        await onRetry();
      } finally {
        setTimeout(() => setIsRetrying(false), 1000);
      }
    }
  };

  return (
    <div className="terminal-card p-6 text-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="mx-auto w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center border border-red-500/30 mb-4">
          <WifiOff className="w-8 h-8 text-red-400" />
        </div>

        <h3 className="text-lg font-semibold text-terminal-text mb-2">
          Connection Problem
        </h3>
        
        <p className="text-terminal-text-dim mb-6">
          {isOnline 
            ? 'Server is unreachable. This may be due to high traffic or maintenance.'
            : 'You appear to be offline. Please check your internet connection.'
          }
        </p>

        <div className="space-y-3">
          <button
            onClick={handleRetry}
            disabled={isRetrying}
            className="w-full terminal-button py-3 rounded flex items-center justify-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
            <span>{isRetrying ? 'Connecting...' : 'Try Again'}</span>
          </button>

          {showOfflineMode && (
            <button className="w-full border border-terminal-border py-3 rounded text-terminal-text hover:bg-terminal-surface-variant transition-colors">
              Switch to Offline Mode
            </button>
          )}
        </div>

        <div className="mt-6 pt-4 border-t border-terminal-border">
          <div className="flex items-center justify-center space-x-2 text-sm text-terminal-text-dim">
            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span>{isOnline ? 'Device Online' : 'Device Offline'}</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

// Loading with timeout fallback
interface LoadingFallbackProps {
  title?: string;
  timeout?: number; // in milliseconds
  onTimeout?: () => void;
  showProgress?: boolean;
}

export const LoadingFallback: React.FC<LoadingFallbackProps> = ({
  title = 'Loading',
  timeout = 30000, // 30 seconds
  onTimeout,
  showProgress = true
}) => {
  const [progress, setProgress] = useState(0);
  const [hasTimedOut, setHasTimedOut] = useState(false);

  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.min((elapsed / timeout) * 100, 100);
      setProgress(newProgress);

      if (elapsed >= timeout) {
        setHasTimedOut(true);
        clearInterval(interval);
        onTimeout?.();
      }
    }, 100);

    return () => clearInterval(interval);
  }, [timeout, onTimeout]);

  if (hasTimedOut) {
    return (
      <div className="terminal-card p-6 text-center">
        <Clock className="w-8 h-8 text-yellow-400 mx-auto mb-3" />
        <h3 className="text-terminal-text font-medium mb-2">Loading Timeout</h3>
        <p className="text-terminal-text-dim text-sm mb-4">
          This is taking longer than expected. The server might be busy.
        </p>
        <button
          onClick={() => window.location.reload()}
          className="terminal-button px-4 py-2 rounded flex items-center space-x-2 mx-auto"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh Page</span>
        </button>
      </div>
    );
  }

  return (
    <div className="terminal-card p-6 text-center">
      <div className="mx-auto w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center border border-blue-500/30 mb-4">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
          <Activity className="w-8 h-8 text-blue-400" />
        </motion.div>
      </div>

      <h3 className="text-terminal-text font-medium mb-2">{title}</h3>
      <p className="text-terminal-text-dim text-sm mb-4">
        Please wait while we fetch the latest data...
      </p>

      {showProgress && (
        <div className="w-full bg-terminal-surface-variant rounded-full h-2 mb-2">
          <motion.div
            className="bg-blue-400 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>
      )}

      <div className="text-xs text-terminal-text-dim">
        {Math.round(progress)}% complete
      </div>
    </div>
  );
};

// Feature unavailable fallback
interface FeatureUnavailableProps {
  featureName: string;
  reason?: string;
  alternativeAction?: {
    label: string;
    action: () => void;
  };
}

export const FeatureUnavailable: React.FC<FeatureUnavailableProps> = ({
  featureName,
  reason = 'This feature is temporarily unavailable',
  alternativeAction
}) => {
  return (
    <div className="terminal-card p-6 text-center border-2 border-dashed border-terminal-border">
      <div className="mx-auto w-12 h-12 bg-slate-500/10 rounded-full flex items-center justify-center border border-slate-500/30 mb-4">
        <Zap className="w-6 h-6 text-slate-400" />
      </div>

      <h3 className="text-terminal-text font-medium mb-2">
        {featureName} Unavailable
      </h3>
      
      <p className="text-terminal-text-dim text-sm mb-4">
        {reason}
      </p>

      {alternativeAction && (
        <button
          onClick={alternativeAction.action}
          className="terminal-button px-4 py-2 rounded"
        >
          {alternativeAction.label}
        </button>
      )}

      <div className="mt-4 pt-4 border-t border-terminal-border">
        <p className="text-xs text-terminal-text-dim">
          Feature will be restored automatically when available
        </p>
      </div>
    </div>
  );
};

// Progressive enhancement wrapper
interface ProgressiveEnhancementProps {
  children: ReactNode;
  fallback: ReactNode;
  condition: boolean;
  loadingComponent?: ReactNode;
}

export const ProgressiveEnhancement: React.FC<ProgressiveEnhancementProps> = ({
  children,
  fallback,
  condition,
  loadingComponent
}) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 100);
    return () => clearTimeout(timer);
  }, []);

  if (isLoading && loadingComponent) {
    return <>{loadingComponent}</>;
  }

  return condition ? <>{children}</> : <>{fallback}</>;
};