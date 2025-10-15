/**
 * Performance Monitoring React Hooks
 * Elite Trading Platform Performance Hooks
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getPerformanceMonitor } from '../PerformanceMonitor';
import type { PerformanceMetrics, PerformanceAlert, RenderMetrics } from '../../types';

export interface UsePerformanceOptions {
  enableRealTimeUpdates?: boolean;
  samplingInterval?: number;
  alertThresholds?: {
    readonly cpuUsage?: number;
    readonly memoryUsage?: number;
    readonly renderTime?: number;
    readonly fps?: number;
  };
}

export interface PerformanceHookResult {
  metrics: PerformanceMetrics | null;
  alerts: PerformanceAlert[];
  isMonitoring: boolean;
  startMonitoring: () => void;
  stopMonitoring: () => void;
  clearAlerts: () => void;
  measureRender: () => void;
}

/**
 * Main performance monitoring hook
 */
export function usePerformance(options: UsePerformanceOptions = {}): PerformanceHookResult {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  const performanceMonitor = getPerformanceMonitor(options.alertThresholds ? {
    alertThresholds: {
      cpuUsage: options.alertThresholds.cpuUsage || 80,
      memoryUsage: options.alertThresholds.memoryUsage || 85,
      renderTime: options.alertThresholds.renderTime || 16,
      fps: options.alertThresholds.fps || 50
    },
    samplingInterval: options.samplingInterval
  } : {
    samplingInterval: options.samplingInterval
  });

  useEffect(() => {
    const handleMetricsUpdate = (newMetrics: PerformanceMetrics) => {
      setMetrics(newMetrics);
    };

    const handleAlert = (alert: PerformanceAlert) => {
      setAlerts(prev => [...prev, alert]);
    };

    const handleMonitoringStart = () => {
      setIsMonitoring(true);
    };

    const handleMonitoringStop = () => {
      setIsMonitoring(false);
    };

    const handleAlertsCleared = () => {
      setAlerts([]);
    };

    if (options.enableRealTimeUpdates) {
      performanceMonitor.on('metrics.collected', handleMetricsUpdate);
      performanceMonitor.on('alert.created', handleAlert);
      performanceMonitor.on('monitoring.started', handleMonitoringStart);
      performanceMonitor.on('monitoring.stopped', handleMonitoringStop);
      performanceMonitor.on('alerts.cleared', handleAlertsCleared);
    }

    return () => {
      performanceMonitor.removeListener('metrics.collected', handleMetricsUpdate);
      performanceMonitor.removeListener('alert.created', handleAlert);
      performanceMonitor.removeListener('monitoring.started', handleMonitoringStart);
      performanceMonitor.removeListener('monitoring.stopped', handleMonitoringStop);
      performanceMonitor.removeListener('alerts.cleared', handleAlertsCleared);
    };
  }, [performanceMonitor, options.enableRealTimeUpdates]);

  const startMonitoring = useCallback(() => {
    performanceMonitor.start();
  }, [performanceMonitor]);

  const stopMonitoring = useCallback(() => {
    performanceMonitor.stop();
  }, [performanceMonitor]);

  const clearAlerts = useCallback(() => {
    performanceMonitor.clearAlerts();
  }, [performanceMonitor]);

  const measureRender = useCallback(() => {
    performanceMonitor.markRenderStart();
    // The render end should be called after component renders
    requestAnimationFrame(() => {
      performanceMonitor.markRenderEnd();
    });
  }, [performanceMonitor]);

  return {
    metrics,
    alerts,
    isMonitoring,
    startMonitoring,
    stopMonitoring,
    clearAlerts,
    measureRender
  };
}

/**
 * Hook for measuring function execution time
 */
export function useMeasureFunction() {
  const performanceMonitor = getPerformanceMonitor();

  const measure = useCallback(<T>(name: string, fn: () => T): T => {
    return performanceMonitor.measureFunction(name, fn);
  }, [performanceMonitor]);

  const measureAsync = useCallback(async <T>(name: string, fn: () => Promise<T>): Promise<T> => {
    return performanceMonitor.measureAsyncFunction(name, fn);
  }, [performanceMonitor]);

  return { measure, measureAsync };
}

/**
 * Hook for component render time measurement
 */
export function useRenderTime(componentName: string) {
  const renderStartTime = useRef<number>(0);
  const [renderTime, setRenderTime] = useState<number>(0);
  const performanceMonitor = getPerformanceMonitor();

  const startMeasurement = useCallback(() => {
    renderStartTime.current = performance.now();
    performanceMonitor.mark(`${componentName}-render-start`);
  }, [componentName, performanceMonitor]);

  const endMeasurement = useCallback(() => {
    if (renderStartTime.current > 0) {
      const duration = performance.now() - renderStartTime.current;
      setRenderTime(duration);
      performanceMonitor.mark(`${componentName}-render-end`);
      performanceMonitor.measure(
        `${componentName}-render-time`,
        `${componentName}-render-start`,
        `${componentName}-render-end`
      );
      renderStartTime.current = 0;
    }
  }, [componentName, performanceMonitor]);

  useEffect(() => {
    startMeasurement();
    return endMeasurement;
  });

  return renderTime;
}

/**
 * Hook for monitoring component re-renders
 */
export function useRenderCount(componentName: string) {
  const renderCount = useRef(0);
  const [count, setCount] = useState(0);

  useEffect(() => {
    renderCount.current += 1;
    setCount(renderCount.current);

    if (renderCount.current > 1) {
      console.log(`[${componentName}] Re-render #${renderCount.current}`);
    }
  });

  return count;
}

/**
 * Hook for FPS monitoring
 */
export function useFPS() {
  const [fps, setFPS] = useState<number>(60);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    const updateFPS = () => {
      const now = performance.now();
      frameCountRef.current++;

      if (lastTimeRef.current === 0) {
        lastTimeRef.current = now;
      }

      const elapsed = now - lastTimeRef.current;

      if (elapsed >= 1000) {
        const currentFPS = (frameCountRef.current * 1000) / elapsed;
        setFPS(Math.round(currentFPS));
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }

      requestAnimationFrame(updateFPS);
    };

    const animationId = requestAnimationFrame(updateFPS);

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, []);

  return fps;
}

/**
 * Hook for memory usage monitoring
 */
export function useMemoryUsage() {
  const [memoryUsage, setMemoryUsage] = useState<{
    used: number;
    total: number;
    percentage: number;
  } | null>(null);

  useEffect(() => {
    const updateMemoryUsage = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        setMemoryUsage({
          used: memory.usedJSHeapSize,
          total: memory.totalJSHeapSize,
          percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
        });
      }
    };

    const interval = setInterval(updateMemoryUsage, 1000);
    updateMemoryUsage(); // Initial call

    return () => clearInterval(interval);
  }, []);

  return memoryUsage;
}

/**
 * Hook for network latency monitoring
 */
export function useNetworkLatency() {
  const [latency, setLatency] = useState<number>(0);

  useEffect(() => {
    const measureLatency = async () => {
      try {
        const start = performance.now();
        await fetch('/api/health', { method: 'HEAD' });
        const end = performance.now();
        setLatency(end - start);
      } catch (error) {
        console.warn('Failed to measure network latency:', error);
      }
    };

    const interval = setInterval(measureLatency, 5000);
    measureLatency(); // Initial call

    return () => clearInterval(interval);
  }, []);

  return latency;
}

/**
 * Hook for performance alerts
 */
export function usePerformanceAlerts() {
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const performanceMonitor = getPerformanceMonitor();

  useEffect(() => {
    const handleAlert = (alert: PerformanceAlert) => {
      setAlerts(prev => [...prev, alert]);
    };

    const handleAlertsCleared = () => {
      setAlerts([]);
    };

    performanceMonitor.on('alert.created', handleAlert);
    performanceMonitor.on('alerts.cleared', handleAlertsCleared);

    // Load existing alerts
    setAlerts(performanceMonitor.getAlerts());

    return () => {
      performanceMonitor.removeListener('alert.created', handleAlert);
      performanceMonitor.removeListener('alerts.cleared', handleAlertsCleared);
    };
  }, [performanceMonitor]);

  const clearAlerts = useCallback(() => {
    performanceMonitor.clearAlerts();
  }, [performanceMonitor]);

  const getAlertsByType = useCallback((type: PerformanceAlert['type']) => {
    return alerts.filter(alert => alert.type === type);
  }, [alerts]);

  const getCriticalAlerts = useCallback(() => {
    return alerts.filter(alert => alert.severity === 'critical');
  }, [alerts]);

  return {
    alerts,
    clearAlerts,
    getAlertsByType,
    getCriticalAlerts
  };
}