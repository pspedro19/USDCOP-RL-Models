'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, RefreshCw, Home, Bug, Clock } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, errorInfo: ErrorInfo, retry: () => void) => ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
  retryCount: number;
  lastErrorTime: number;
}

export class NavigationErrorBoundary extends Component<Props, State> {
  private retryTimeoutId: NodeJS.Timeout | null = null;
  
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
      retryCount: 0,
      lastErrorTime: 0
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    const now = Date.now();
    const errorId = `error_${now}_${Math.random().toString(36).substring(7)}`;
    
    return {
      hasError: true,
      error,
      errorId,
      lastErrorTime: now
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details for debugging
    console.error('Navigation Error Boundary caught an error:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      errorId: this.state.errorId,
      retryCount: this.state.retryCount,
      timestamp: new Date().toISOString()
    });

    this.setState({ errorInfo });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Auto-retry after a delay (with exponential backoff)
    if (this.state.retryCount < 3) {
      const retryDelay = Math.min(1000 * Math.pow(2, this.state.retryCount), 10000);
      
      this.retryTimeoutId = setTimeout(() => {
        this.handleRetry();
      }, retryDelay);
    }
  }

  componentWillUnmount() {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
    }
  }

  handleRetry = () => {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
      this.retryTimeoutId = null;
    }

    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1
    }));
  };

  handleReset = () => {
    if (this.retryTimeoutId) {
      clearTimeout(this.retryTimeoutId);
      this.retryTimeoutId = null;
    }

    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      lastErrorTime: 0
    });
  };

  render() {
    if (this.state.hasError && this.state.error && this.state.errorInfo) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.state.errorInfo, this.handleRetry);
      }

      // Default error UI
      return <EnhancedErrorFallback 
        error={this.state.error}
        errorInfo={this.state.errorInfo}
        errorId={this.state.errorId}
        retryCount={this.state.retryCount}
        lastErrorTime={this.state.lastErrorTime}
        onRetry={this.handleRetry}
        onReset={this.handleReset}
      />;
    }

    return this.props.children;
  }
}

interface ErrorFallbackProps {
  error: Error;
  errorInfo: ErrorInfo;
  errorId: string;
  retryCount: number;
  lastErrorTime: number;
  onRetry: () => void;
  onReset: () => void;
}

function EnhancedErrorFallback({ 
  error, 
  errorInfo, 
  errorId, 
  retryCount, 
  lastErrorTime,
  onRetry, 
  onReset 
}: ErrorFallbackProps) {
  const isNetworkError = error.message.includes('fetch') || error.message.includes('network');
  const isChunkError = error.message.includes('ChunkLoadError') || error.message.includes('Loading chunk');
  const isRenderError = errorInfo.componentStack.length > 0;

  const errorType = isNetworkError ? 'Network Error' : 
                   isChunkError ? 'Resource Loading Error' : 
                   isRenderError ? 'Rendering Error' : 'Unknown Error';

  const timeSinceError = Date.now() - lastErrorTime;
  const formattedTime = new Date(lastErrorTime).toLocaleTimeString();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="max-w-2xl w-full bg-slate-900/90 backdrop-blur-xl border border-red-500/30 rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-red-500/20 to-orange-500/20 p-6 border-b border-red-500/30">
          <div className="flex items-center space-x-4">
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="p-3 bg-red-500/20 rounded-full"
            >
              <AlertTriangle className="w-8 h-8 text-red-400" />
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold text-red-400 mb-1">Navigation System Error</h1>
              <p className="text-slate-300 text-sm">{errorType} detected in the trading dashboard</p>
            </div>
          </div>
        </div>

        {/* Error Details */}
        <div className="p-6">
          <div className="space-y-4">
            {/* Error Message */}
            <div className="p-4 bg-red-950/30 border border-red-500/30 rounded-lg">
              <div className="flex items-start space-x-3">
                <Bug className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <h3 className="font-semibold text-red-400 mb-1">Error Details</h3>
                  <p className="text-slate-300 font-mono text-sm break-words">{error.message}</p>
                </div>
              </div>
            </div>

            {/* Error Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center space-x-2 mb-2">
                  <Clock className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-medium text-slate-400">Error Time</span>
                </div>
                <p className="text-white font-mono text-sm">{formattedTime}</p>
              </div>
              
              <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center space-x-2 mb-2">
                  <RefreshCw className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-medium text-slate-400">Retry Attempts</span>
                </div>
                <p className="text-white font-mono text-sm">{retryCount} / 3</p>
              </div>
            </div>

            {/* Suggestions */}
            <div className="p-4 bg-blue-950/30 border border-blue-500/30 rounded-lg">
              <h3 className="font-semibold text-blue-400 mb-3">Suggested Actions</h3>
              <ul className="space-y-2 text-sm text-slate-300">
                {isNetworkError && (
                  <li className="flex items-start space-x-2">
                    <span className="text-blue-400 mt-1">•</span>
                    <span>Check your internet connection and try again</span>
                  </li>
                )}
                {isChunkError && (
                  <>
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-400 mt-1">•</span>
                      <span>Try refreshing the page to reload missing resources</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-400 mt-1">•</span>
                      <span>Clear browser cache if the issue persists</span>
                    </li>
                  </>
                )}
                <li className="flex items-start space-x-2">
                  <span className="text-blue-400 mt-1">•</span>
                  <span>Report this error if it continues to occur</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 mt-6">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onRetry}
              disabled={retryCount >= 3}
              className={`flex-1 flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${
                retryCount < 3
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50 hover:bg-blue-500/30 hover:border-blue-500/70'
                  : 'bg-slate-700/50 text-slate-500 border border-slate-600/50 cursor-not-allowed'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${retryCount < 3 ? '' : 'opacity-50'}`} />
              <span>{retryCount < 3 ? 'Retry Navigation' : 'Max Retries Reached'}</span>
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onReset}
              className="flex-1 flex items-center justify-center space-x-2 px-6 py-3 bg-green-500/20 text-green-400 border border-green-500/50 rounded-lg font-medium hover:bg-green-500/30 hover:border-green-500/70 transition-all"
            >
              <Home className="w-4 h-4" />
              <span>Reset Dashboard</span>
            </motion.button>
          </div>

          {/* Error ID */}
          <div className="mt-4 pt-4 border-t border-slate-700/50">
            <p className="text-xs text-slate-500 font-mono">
              Error ID: {errorId}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default NavigationErrorBoundary;