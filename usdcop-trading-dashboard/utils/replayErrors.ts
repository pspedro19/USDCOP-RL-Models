/**
 * Replay Error Handling
 *
 * Provides structured error types, error codes, recovery strategies,
 * and a React hook for managing replay errors.
 */

import { useState, useCallback } from 'react';

// ═══════════════════════════════════════════════════════════════════════════
// ERROR CODES
// ═══════════════════════════════════════════════════════════════════════════

export enum ReplayErrorCode {
  // Data errors
  INVALID_DATE_RANGE = 'INVALID_DATE_RANGE',
  DATA_LOAD_FAILED = 'DATA_LOAD_FAILED',
  DATA_VALIDATION_FAILED = 'DATA_VALIDATION_FAILED',
  NO_TRADES_IN_RANGE = 'NO_TRADES_IN_RANGE',
  NO_DATA_AVAILABLE = 'NO_DATA_AVAILABLE',

  // State errors
  INVALID_STATE_TRANSITION = 'INVALID_STATE_TRANSITION',
  ANIMATION_FRAME_ERROR = 'ANIMATION_FRAME_ERROR',

  // Performance errors
  TOO_MANY_DATA_POINTS = 'TOO_MANY_DATA_POINTS',
  RENDER_TIMEOUT = 'RENDER_TIMEOUT',

  // Network errors
  API_TIMEOUT = 'API_TIMEOUT',
  API_ERROR = 'API_ERROR',
  NETWORK_OFFLINE = 'NETWORK_OFFLINE',

  // Unknown
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR MESSAGES (Spanish for user-facing messages)
// ═══════════════════════════════════════════════════════════════════════════

export const ERROR_MESSAGES: Record<ReplayErrorCode, string> = {
  [ReplayErrorCode.INVALID_DATE_RANGE]:
    'El rango de fechas seleccionado no es valido. Debe estar entre validacion y test.',
  [ReplayErrorCode.DATA_LOAD_FAILED]:
    'No se pudieron cargar los datos del replay. Por favor, intente de nuevo.',
  [ReplayErrorCode.DATA_VALIDATION_FAILED]:
    'Los datos recibidos no tienen el formato esperado.',
  [ReplayErrorCode.NO_TRADES_IN_RANGE]:
    'No hay trades en el rango de fechas seleccionado.',
  [ReplayErrorCode.NO_DATA_AVAILABLE]:
    'No hay datos disponibles para el replay.',
  [ReplayErrorCode.INVALID_STATE_TRANSITION]:
    'Operacion no permitida en el estado actual del replay.',
  [ReplayErrorCode.ANIMATION_FRAME_ERROR]:
    'Error en la animacion. El replay sera pausado.',
  [ReplayErrorCode.TOO_MANY_DATA_POINTS]:
    'Demasiados datos para mostrar. Por favor, seleccione un rango mas pequeno.',
  [ReplayErrorCode.RENDER_TIMEOUT]:
    'El renderizado esta tardando demasiado. Reduciendo calidad visual.',
  [ReplayErrorCode.API_TIMEOUT]:
    'El servidor tardo demasiado en responder.',
  [ReplayErrorCode.API_ERROR]:
    'Error del servidor. Por favor, intente mas tarde.',
  [ReplayErrorCode.NETWORK_OFFLINE]:
    'Sin conexion a internet. Verifique su conexion.',
  [ReplayErrorCode.UNKNOWN_ERROR]:
    'Ocurrio un error inesperado.',
};

// ═══════════════════════════════════════════════════════════════════════════
// REPLAY ERROR CLASS
// ═══════════════════════════════════════════════════════════════════════════

export class ReplayError extends Error {
  public readonly code: ReplayErrorCode;
  public readonly recoverable: boolean;
  public readonly context?: Record<string, unknown>;
  public readonly timestamp: Date;

  constructor(
    code: ReplayErrorCode,
    message?: string,
    recoverable: boolean = true,
    context?: Record<string, unknown>
  ) {
    super(message || ERROR_MESSAGES[code]);
    this.name = 'ReplayError';
    this.code = code;
    this.recoverable = recoverable;
    this.context = context;
    this.timestamp = new Date();

    // Maintains proper stack trace for where our error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ReplayError);
    }
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      recoverable: this.recoverable,
      context: this.context,
      timestamp: this.timestamp.toISOString(),
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RECOVERY STRATEGIES
// ═══════════════════════════════════════════════════════════════════════════

export type RecoveryAction =
  | 'retry'
  | 'reduce_range'
  | 'reduce_speed'
  | 'pause'
  | 'reset'
  | 'none';

export interface RecoveryStrategy {
  action: RecoveryAction;
  delay_ms?: number;
  message?: string;
  autoExecute?: boolean;
}

export function getRecoveryStrategy(error: ReplayError): RecoveryStrategy {
  switch (error.code) {
    case ReplayErrorCode.API_TIMEOUT:
    case ReplayErrorCode.DATA_LOAD_FAILED:
      return {
        action: 'retry',
        delay_ms: 2000,
        message: 'Reintentando...',
        autoExecute: true,
      };

    case ReplayErrorCode.TOO_MANY_DATA_POINTS:
      return {
        action: 'reduce_range',
        message: 'Reduciendo rango de fechas automaticamente...',
        autoExecute: true,
      };

    case ReplayErrorCode.RENDER_TIMEOUT:
      return {
        action: 'reduce_speed',
        message: 'Reduciendo velocidad de replay...',
        autoExecute: true,
      };

    case ReplayErrorCode.ANIMATION_FRAME_ERROR:
      return {
        action: 'pause',
        message: 'Replay pausado debido a un error.',
        autoExecute: true,
      };

    case ReplayErrorCode.NETWORK_OFFLINE:
      return {
        action: 'none',
        message: 'Esperando conexion...',
        autoExecute: false,
      };

    case ReplayErrorCode.INVALID_DATE_RANGE:
    case ReplayErrorCode.DATA_VALIDATION_FAILED:
      return {
        action: 'reset',
        message: 'Por favor, seleccione un rango valido.',
        autoExecute: false,
      };

    case ReplayErrorCode.NO_TRADES_IN_RANGE:
    case ReplayErrorCode.NO_DATA_AVAILABLE:
      return {
        action: 'none',
        message: 'Intente con un rango de fechas diferente.',
        autoExecute: false,
      };

    default:
      return {
        action: 'reset',
        message: 'Reiniciando el replay...',
        autoExecute: false,
      };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR CONVERSION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert any error to a ReplayError
 */
export function toReplayError(err: unknown): ReplayError {
  if (err instanceof ReplayError) {
    return err;
  }

  if (err instanceof Error) {
    // Check for network-related errors
    if (
      err.message.includes('network') ||
      err.message.includes('fetch') ||
      err.message.includes('Failed to fetch') ||
      err.name === 'TypeError'
    ) {
      if (!navigator.onLine) {
        return new ReplayError(
          ReplayErrorCode.NETWORK_OFFLINE,
          ERROR_MESSAGES[ReplayErrorCode.NETWORK_OFFLINE],
          true
        );
      }
      return new ReplayError(
        ReplayErrorCode.API_ERROR,
        err.message,
        true,
        { originalError: err.name }
      );
    }

    // Check for timeout
    if (err.name === 'AbortError' || err.message.includes('timeout')) {
      return new ReplayError(
        ReplayErrorCode.API_TIMEOUT,
        ERROR_MESSAGES[ReplayErrorCode.API_TIMEOUT],
        true
      );
    }

    // Generic error
    return new ReplayError(
      ReplayErrorCode.UNKNOWN_ERROR,
      err.message,
      true,
      { originalError: err.name, stack: err.stack }
    );
  }

  // Unknown error type
  return new ReplayError(
    ReplayErrorCode.UNKNOWN_ERROR,
    String(err),
    true
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR BOUNDARY HOOK
// ═══════════════════════════════════════════════════════════════════════════

export interface UseReplayErrorReturn {
  error: ReplayError | null;
  hasError: boolean;
  recovery: RecoveryStrategy | null;
  setError: (error: ReplayError) => void;
  handleError: (err: unknown) => void;
  clearError: () => void;
  retryCount: number;
  incrementRetry: () => void;
  resetRetryCount: () => void;
}

export function useReplayError(_maxRetries: number = 3): UseReplayErrorReturn {
  // maxRetries available for future use with auto-retry logic
  void _maxRetries;

  const [error, setErrorState] = useState<ReplayError | null>(null);
  const [recovery, setRecovery] = useState<RecoveryStrategy | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const handleError = useCallback((err: unknown) => {
    const replayError = toReplayError(err);
    setErrorState(replayError);
    setRecovery(getRecoveryStrategy(replayError));

    // Log error for debugging
    console.error('[ReplayError]', {
      code: replayError.code,
      message: replayError.message,
      context: replayError.context,
      recoverable: replayError.recoverable,
      timestamp: replayError.timestamp,
    });
  }, []);

  const setError = useCallback((error: ReplayError) => {
    setErrorState(error);
    setRecovery(getRecoveryStrategy(error));
  }, []);

  const clearError = useCallback(() => {
    setErrorState(null);
    setRecovery(null);
  }, []);

  const incrementRetry = useCallback(() => {
    setRetryCount(prev => prev + 1);
  }, []);

  const resetRetryCount = useCallback(() => {
    setRetryCount(0);
  }, []);

  return {
    error,
    hasError: error !== null,
    recovery,
    setError,
    handleError,
    clearError,
    retryCount,
    incrementRetry,
    resetRetryCount,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR DISPLAY HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get error severity for styling
 */
export function getErrorSeverity(code: ReplayErrorCode): 'info' | 'warning' | 'error' {
  switch (code) {
    case ReplayErrorCode.NO_TRADES_IN_RANGE:
    case ReplayErrorCode.NO_DATA_AVAILABLE:
      return 'info';

    case ReplayErrorCode.NETWORK_OFFLINE:
    case ReplayErrorCode.RENDER_TIMEOUT:
    case ReplayErrorCode.API_TIMEOUT:
      return 'warning';

    default:
      return 'error';
  }
}

/**
 * Check if error should show a toast notification
 */
export function shouldShowToast(code: ReplayErrorCode): boolean {
  const silentErrors = [
    ReplayErrorCode.ANIMATION_FRAME_ERROR,
    ReplayErrorCode.RENDER_TIMEOUT,
  ];
  return !silentErrors.includes(code);
}
