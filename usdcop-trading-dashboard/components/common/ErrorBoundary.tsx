/**
 * Error Boundary Components
 * Graceful error handling with retry capabilities
 */
'use client';

import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
  errorId: string;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  maxRetries?: number;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  level?: 'chart' | 'component' | 'page';
  showDetails?: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private retryTimer?: number;
  
  constructor(props: ErrorBoundaryProps) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      errorId: this.generateErrorId()
    };
  }
  
  private generateErrorId(): string {
    return `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { 
      hasError: true, 
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const errorDetails = {
      error: error?.message || 'Unknown error',
      name: error?.name || 'Error',
      stack: error?.stack || 'No stack trace available',
      componentStack: errorInfo?.componentStack || 'No component stack available',
      errorId: this.state.errorId
    };
    
    console.error('[ErrorBoundary] Component error caught:', errorDetails);
    
    // Store error info
    this.setState({ errorInfo });
    
    // Call custom error handler
    this.props.onError?.(error, errorInfo);
    
    // Report to monitoring service (if available)
    this.reportError(error, errorInfo);
    
    // Auto-retry logic
    const maxRetries = this.props.maxRetries ?? 3;
    if (this.state.retryCount < maxRetries) {
      this.scheduleRetry();
    }
  }
  
  private reportError(error: Error, errorInfo: ErrorInfo) {
    // This would integrate with your monitoring service (e.g., Sentry, LogRocket)
    try {
      const errorReport = {
        errorId: this.state.errorId,
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        level: this.props.level || 'component'
      };
      
      // Store in localStorage for debugging
      const errors = JSON.parse(localStorage.getItem('error-reports') || '[]');
      errors.push(errorReport);
      localStorage.setItem('error-reports', JSON.stringify(errors.slice(-10))); // Keep last 10
      
      // TODO: Send to monitoring service
      // analytics.reportError(errorReport);
      
    } catch (reportingError) {
      console.error('[ErrorBoundary] Failed to report error:', reportingError);
    }
  }
  
  private scheduleRetry() {
    const { retryCount } = this.state;
    const delay = Math.min(2000 * Math.pow(2, retryCount), 10000); // Exponential backoff
    
    console.log(`[ErrorBoundary] Scheduling retry ${retryCount + 1} in ${delay}ms`);
    
    this.retryTimer = window.setTimeout(() => {
      this.setState(prevState => ({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: prevState.retryCount + 1,
        errorId: this.generateErrorId()
      }));
    }, delay);
  }
  
  private handleManualRetry = () => {
    console.log('[ErrorBoundary] Manual retry triggered');
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
      errorId: this.generateErrorId()
    });
  };
  
  private handleRefreshPage = () => {
    window.location.reload();
  };
  
  private handleGoHome = () => {
    window.location.href = '/';
  };
  
  componentWillUnmount() {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
    }
  }
  
  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      // Default error UI based on level
      return this.renderErrorUI();
    }
    
    return this.props.children;
  }
  
  private renderErrorUI() {
    const { error, retryCount, errorId } = this.state;
    const { maxRetries = 3, level = 'component', showDetails = false } = this.props;
    const isRetrying = retryCount > 0 && retryCount < maxRetries;
    const hasExhaustedRetries = retryCount >= maxRetries;
    
    // Chart-level error (minimal UI)
    if (level === 'chart') {
      return (
        <div className="flex items-center justify-center h-96 bg-terminal-surface border border-terminal-border rounded-lg">
          <div className="text-center p-6">
            <AlertTriangle className="w-12 h-12 text-warning mx-auto mb-3" />
            <h3 className="text-terminal-text font-semibold mb-2">Chart Error</h3>
            <p className="text-terminal-text-dim text-sm mb-4">
              Failed to render chart data
            </p>
            {!hasExhaustedRetries && (
              <button
                onClick={this.handleManualRetry}
                className="terminal-button px-4 py-2 rounded flex items-center space-x-2 mx-auto"
                disabled={isRetrying}
              >
                <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                <span>{isRetrying ? 'Retrying...' : 'Retry'}</span>
              </button>
            )}
          </div>
        </div>
      );
    }
    
    // Component-level error
    if (level === 'component') {
      return (
        <div className="terminal-card p-6">
          <div className="flex items-start space-x-4">
            <AlertTriangle className="w-8 h-8 text-warning flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-terminal-text mb-2">
                Component Error
              </h3>
              <p className="text-terminal-text-dim mb-4">
                A component has encountered an error and cannot be displayed.
              </p>
              
              {showDetails && error && (
                <details className="mb-4">
                  <summary className="cursor-pointer text-terminal-accent text-sm mb-2">
                    Error Details
                  </summary>
                  <div className="bg-terminal-surface-variant p-3 rounded font-mono text-xs text-terminal-text-dim">
                    <div><strong>Error:</strong> {error.message}</div>
                    <div><strong>ID:</strong> {errorId}</div>
                    {retryCount > 0 && <div><strong>Retries:</strong> {retryCount}</div>}
                  </div>
                </details>
              )}
              
              <div className="flex space-x-3">
                {!hasExhaustedRetries && (
                  <button
                    onClick={this.handleManualRetry}
                    className="terminal-button px-4 py-2 rounded flex items-center space-x-2"
                    disabled={isRetrying}
                  >
                    <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                    <span>{isRetrying ? `Retrying (${retryCount}/${maxRetries})...` : 'Try Again'}</span>
                  </button>
                )}
                
                <button
                  onClick={this.handleRefreshPage}
                  className="border border-terminal-border px-4 py-2 rounded text-terminal-text hover:bg-terminal-surface-variant"
                >
                  Refresh Page
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }
    
    // Page-level error (full screen)
    return (
      <div className="min-h-screen bg-terminal-bg flex items-center justify-center p-6">
        <div className="max-w-md w-full text-center">
          <AlertTriangle className="w-16 h-16 text-warning mx-auto mb-6" />
          <h1 className="text-2xl font-bold text-terminal-text mb-4">
            Something went wrong
          </h1>
          <p className="text-terminal-text-dim mb-6">
            The application encountered an unexpected error. This has been reported automatically.
          </p>
          
          {showDetails && error && (
            <details className="mb-6 text-left">
              <summary className="cursor-pointer text-terminal-accent text-sm mb-2">
                Technical Details
              </summary>
              <div className="bg-terminal-surface p-4 rounded font-mono text-xs text-terminal-text-dim">
                <div className="mb-2"><strong>Error ID:</strong> {errorId}</div>
                <div className="mb-2"><strong>Message:</strong> {error.message}</div>
                <div><strong>Stack:</strong></div>
                <pre className="whitespace-pre-wrap text-xs mt-1 max-h-32 overflow-y-auto">
                  {error.stack}
                </pre>
              </div>
            </details>
          )}
          
          <div className="space-y-3">
            {!hasExhaustedRetries && (
              <button
                onClick={this.handleManualRetry}
                className="w-full terminal-button py-3 rounded flex items-center justify-center space-x-2"
                disabled={isRetrying}
              >
                <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
                <span>{isRetrying ? `Retrying (${retryCount}/${maxRetries})...` : 'Try Again'}</span>
              </button>
            )}
            
            <button
              onClick={this.handleRefreshPage}
              className="w-full border border-terminal-border py-3 rounded text-terminal-text hover:bg-terminal-surface-variant"
            >
              Refresh Page
            </button>
            
            <button
              onClick={this.handleGoHome}
              className="w-full text-terminal-text-dim hover:text-terminal-text flex items-center justify-center space-x-2"
            >
              <Home className="w-4 h-4" />
              <span>Go Home</span>
            </button>
          </div>
        </div>
      </div>
    );
  }
}

// Specialized error boundary for charts
export const ChartErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary
    level="chart"
    maxRetries={2}
  >
    {children}
  </ErrorBoundary>
);

// Component error boundary with details
export const ComponentErrorBoundary: React.FC<{ 
  children: ReactNode;
  showDetails?: boolean;
}> = ({ children, showDetails = false }) => (
  <ErrorBoundary
    level="component"
    maxRetries={3}
    showDetails={showDetails}
  >
    {children}
  </ErrorBoundary>
);

// Page-level error boundary
export const PageErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => (
  <ErrorBoundary
    level="page"
    maxRetries={1}
    showDetails={true}
  >
    {children}
  </ErrorBoundary>
);

export default ErrorBoundary;