/**
 * useBacktest Hook
 * ================
 * React hook for managing backtest state and operations.
 *
 * Features:
 * - SSE streaming for real-time progress
 * - Automatic state management
 * - Integration with replay mode
 * - Error handling with recovery
 *
 * SOLID Principles:
 * - Single Responsibility: Only backtest state management
 * - Dependency Inversion: Depends on abstractions (service)
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  BacktestState,
  BacktestStatus,
  BacktestProgress,
  BacktestResult,
  BacktestRequest,
  BacktestTradeEvent,
  createInitialBacktestState,
  createEmptyProgress,
  isBacktestRunning,
  canStartBacktest,
} from '@/lib/contracts/backtest.contract';
import {
  createBacktestRunner,
  checkBacktestStatus,
  BacktestRunner,
  BacktestStatusResponse,
} from '@/lib/services/backtest.service';

// ============================================================================
// Types
// ============================================================================

export interface UseBacktestOptions {
  /** Callback when backtest completes successfully */
  onComplete?: (result: BacktestResult) => void;
  /** Callback when backtest fails */
  onError?: (error: Error) => void;
  /** Callback when a trade is generated (for real-time equity curve updates) */
  onTrade?: (trade: BacktestTradeEvent) => void;
  /** Auto-activate replay mode on completion */
  autoActivateReplay?: boolean;
}

export interface UseBacktestReturn {
  /** Current backtest state */
  state: BacktestState;
  /** Start a new backtest */
  startBacktest: (request: BacktestRequest) => Promise<void>;
  /** Cancel running backtest */
  cancelBacktest: () => void;
  /** Reset state to idle */
  resetBacktest: () => void;
  /** Check if data exists for date range */
  checkExistingData: (startDate: string, endDate: string, modelId: string) => Promise<BacktestStatusResponse | null>;
  /** Whether backtest is currently running */
  isRunning: boolean;
  /** Whether a new backtest can be started */
  canStart: boolean;
  /** Progress percentage (0-100) */
  progressPercent: number;
  /** Elapsed time since start */
  elapsedTime: number;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useBacktest(options: UseBacktestOptions = {}): UseBacktestReturn {
  const { onComplete, onError, onTrade, autoActivateReplay = true } = options;

  // State
  const [state, setState] = useState<BacktestState>(createInitialBacktestState);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Refs
  const runnerRef = useRef<BacktestRunner | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ============================================================================
  // Timer Management
  // ============================================================================

  const startTimer = useCallback(() => {
    setElapsedTime(0);
    timerRef.current = setInterval(() => {
      setElapsedTime((prev) => prev + 1);
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTimer();
      if (runnerRef.current) {
        runnerRef.current.cancel();
      }
    };
  }, [stopTimer]);

  // ============================================================================
  // State Updaters
  // ============================================================================

  const updateStatus = useCallback((status: BacktestStatus) => {
    setState((prev) => ({ ...prev, status }));
  }, []);

  const updateProgress = useCallback((progress: BacktestProgress) => {
    setState((prev) => ({
      ...prev,
      status: mapProgressStatusToBacktestStatus(progress.status),
      progress,
    }));
  }, []);

  const updateResult = useCallback((result: BacktestResult) => {
    stopTimer();
    setState((prev) => ({
      ...prev,
      status: 'completed',
      result,
      completedAt: new Date(),
    }));
  }, [stopTimer]);

  const updateError = useCallback((error: Error) => {
    stopTimer();
    setState((prev) => ({
      ...prev,
      status: 'error',
      error: error.message,
      completedAt: new Date(),
    }));
  }, [stopTimer]);

  // ============================================================================
  // Actions
  // ============================================================================

  const startBacktest = useCallback(async (request: BacktestRequest): Promise<void> => {
    if (!canStartBacktest(state.status)) {
      console.warn('[useBacktest] Cannot start - backtest already running');
      return;
    }

    // Reset state
    setState({
      ...createInitialBacktestState(),
      status: 'connecting',
      startedAt: new Date(),
    });

    startTimer();

    // Create runner with event handlers
    const runner = createBacktestRunner(request, {
      onProgress: (progress) => {
        updateProgress(progress);
      },
      onTrade: (trade) => {
        // Real-time trade event for equity curve updates
        onTrade?.(trade);
      },
      onResult: (result) => {
        updateResult(result);
        onComplete?.(result);
      },
      onError: (error) => {
        updateError(error);
        onError?.(error);
      },
      onConnectionChange: (connected) => {
        if (connected) {
          updateStatus('loading');
        }
      },
    });

    runnerRef.current = runner;

    try {
      await runner.start();
    } catch (error) {
      // Error already handled by onError callback
      console.error('[useBacktest] Unexpected error:', error);
    }
  }, [state.status, startTimer, updateProgress, updateResult, updateError, updateStatus, onComplete, onError, onTrade]);

  const cancelBacktest = useCallback(() => {
    if (runnerRef.current) {
      runnerRef.current.cancel();
      runnerRef.current = null;
    }

    stopTimer();

    setState((prev) => ({
      ...prev,
      status: 'cancelled',
      completedAt: new Date(),
    }));
  }, [stopTimer]);

  const resetBacktest = useCallback(() => {
    if (runnerRef.current) {
      runnerRef.current.cancel();
      runnerRef.current = null;
    }

    stopTimer();
    setElapsedTime(0);
    setState(createInitialBacktestState());
  }, [stopTimer]);

  const checkExistingData = useCallback(async (
    startDate: string,
    endDate: string,
    modelId: string
  ): Promise<BacktestStatusResponse | null> => {
    return checkBacktestStatus(startDate, endDate, modelId);
  }, []);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const isRunning = isBacktestRunning(state.status);
  const canStart = canStartBacktest(state.status);
  const progressPercent = state.progress ? Math.round(state.progress.progress * 100) : 0;

  return {
    state,
    startBacktest,
    cancelBacktest,
    resetBacktest,
    checkExistingData,
    isRunning,
    canStart,
    progressPercent,
    elapsedTime,
  };
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Map progress status string to BacktestStatus
 */
function mapProgressStatusToBacktestStatus(progressStatus: string): BacktestStatus {
  const mapping: Record<string, BacktestStatus> = {
    starting: 'connecting',
    loading: 'loading',
    running: 'running',
    saving: 'saving',
    completed: 'completed',
    error: 'error',
  };

  return mapping[progressStatus] || 'running';
}

// ============================================================================
// Export Default
// ============================================================================

export default useBacktest;
