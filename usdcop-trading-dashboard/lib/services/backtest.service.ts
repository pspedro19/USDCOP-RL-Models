/**
 * Backtest Service
 * ================
 * Service layer for backtest operations with SSE streaming support.
 *
 * SOLID Principles:
 * - Single Responsibility: Only backtest-related operations
 * - Open/Closed: Extensible via event handlers
 * - Dependency Inversion: Uses abstractions (callbacks)
 *
 * Design Patterns:
 * - Factory Pattern: createBacktestRunner creates configured instances
 * - Observer Pattern: Event callbacks for progress updates
 * - Strategy Pattern: Different handling for progress vs result events
 *
 * SSOT: Date ranges and presets are imported from ssot.contract.ts
 */

import {
  BacktestRequest,
  BacktestProgress,
  BacktestResult,
  BacktestProgressSchema,
  BacktestResultSchema,
  BacktestSSEEvent,
  BacktestTradeEvent,
} from '@/lib/contracts/backtest.contract';

import {
  PIPELINE_DATE_RANGES,
  BACKTEST_PRESETS,
  BACKTEST_PRESET_CONFIG,
  getPresetDateRange,
  getTestEndDate,
  type BacktestPreset,
} from '@/lib/contracts/ssot.contract';

// ============================================================================
// Configuration
// ============================================================================

export const BACKTEST_CONFIG = {
  /** Base URL for backtest API (client-side, uses exposed port 8003) */
  INFERENCE_API_URL: process.env.NEXT_PUBLIC_INFERENCE_API_URL || 'http://localhost:8003',
  /** Timeout for SSE connection (5 minutes) */
  CONNECTION_TIMEOUT_MS: 300000,
  /** Retry attempts for failed connections */
  MAX_RETRIES: 2,
  /** Delay between retries */
  RETRY_DELAY_MS: 2000,
} as const;

// ============================================================================
// Types
// ============================================================================

export interface BacktestEventHandlers {
  onProgress: (progress: BacktestProgress) => void;
  onResult: (result: BacktestResult) => void;
  onError: (error: Error) => void;
  onConnectionChange?: (connected: boolean) => void;
  /** Called for each trade as it's generated - enables real-time equity curve updates */
  onTrade?: (trade: BacktestTradeEvent) => void;
}

export interface BacktestRunner {
  start: () => Promise<void>;
  cancel: () => void;
  isRunning: () => boolean;
}

export interface BacktestServiceError extends Error {
  code: 'CONNECTION_FAILED' | 'PARSE_ERROR' | 'TIMEOUT' | 'CANCELLED' | 'SERVER_ERROR';
  recoverable: boolean;
}

// ============================================================================
// Error Factory
// ============================================================================

function createBacktestError(
  code: BacktestServiceError['code'],
  message: string,
  recoverable: boolean = false
): BacktestServiceError {
  const error = new Error(message) as BacktestServiceError;
  error.code = code;
  error.recoverable = recoverable;
  return error;
}

// ============================================================================
// SSE Event Parser
// ============================================================================

/**
 * Parse SSE data line into typed event
 */
function parseSSEEvent(data: string): BacktestSSEEvent | null {
  try {
    const parsed = JSON.parse(data);

    if (parsed.type === 'progress') {
      const validated = BacktestProgressSchema.safeParse(parsed.data);
      if (validated.success) {
        return { type: 'progress', data: validated.data };
      }
      console.warn('[BacktestService] Invalid progress data:', validated.error);
    }

    if (parsed.type === 'result') {
      const validated = BacktestResultSchema.safeParse(parsed.data);
      if (validated.success) {
        console.log(`[BacktestService] Backtest completed: ${validated.data.trade_count} trades`);
        return { type: 'result', data: validated.data };
      }
      console.error('[BacktestService] Result validation failed:', validated.error);
    }

    if (parsed.type === 'trade') {
      // Trade event for real-time equity curve updates
      console.log(`[BacktestService] Trade received: ${parsed.data.side} @ ${parsed.data.entry_price}`);
      return { type: 'trade', data: parsed.data as BacktestTradeEvent };
    }

    if (parsed.type === 'error') {
      console.error('[BacktestService] Error:', parsed.data);
      return { type: 'error', data: parsed.data };
    }

    return null;
  } catch (e) {
    console.error('[BacktestService] Failed to parse SSE event:', e);
    return null;
  }
}

// ============================================================================
// Factory: Create Backtest Runner
// ============================================================================

/**
 * Factory function to create a backtest runner instance.
 *
 * Usage:
 * ```typescript
 * const runner = createBacktestRunner(
 *   { startDate: '2025-01-01', endDate: '2025-06-30', modelId: 'ppo_v20' },
 *   {
 *     onProgress: (p) => console.log(`Progress: ${p.progress * 100}%`),
 *     onResult: (r) => console.log(`Done! ${r.trade_count} trades`),
 *     onError: (e) => console.error(e),
 *   }
 * );
 *
 * await runner.start();
 * // or runner.cancel() to abort
 * ```
 */
export function createBacktestRunner(
  request: BacktestRequest,
  handlers: BacktestEventHandlers
): BacktestRunner {
  let abortController: AbortController | null = null;
  let running = false;
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

  const start = async (): Promise<void> => {
    if (running) {
      console.warn('[BacktestService] Backtest already running');
      return;
    }

    running = true;
    abortController = new AbortController();

    // Use Next.js API proxy route (which forwards to inference API or generates fallback)
    const url = `/api/backtest/stream`;

    console.log(`[BacktestService] Starting backtest: ${request.startDate} to ${request.endDate} (model=${request.modelId})`);

    try {
      handlers.onConnectionChange?.(true);

      // Set up timeout
      const timeoutId = setTimeout(() => {
        if (abortController) {
          abortController.abort();
          handlers.onError(createBacktestError('TIMEOUT', 'Backtest timed out', true));
        }
      }, BACKTEST_CONFIG.CONNECTION_TIMEOUT_MS);

      // Make POST request with SSE response
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          start_date: request.startDate,
          end_date: request.endDate,
          model_id: request.modelId,
          force_regenerate: request.forceRegenerate ?? false,
        }),
        signal: abortController.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw createBacktestError(
          'SERVER_ERROR',
          `Server error (${response.status}): ${errorText}`,
          response.status >= 500
        );
      }

      // Read SSE stream
      if (!response.body) {
        throw createBacktestError('CONNECTION_FAILED', 'No response body', false);
      }

      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (running) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE events (data: {...}\n\n)
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;

          // Extract data from "data: {...}" format
          const match = line.match(/^data:\s*(.+)$/m);
          if (match) {
            const event = parseSSEEvent(match[1]);

            if (event) {
              switch (event.type) {
                case 'progress':
                  handlers.onProgress(event.data as BacktestProgress);
                  break;
                case 'trade':
                  // Real-time trade event for equity curve updates
                  handlers.onTrade?.(event.data as BacktestTradeEvent);
                  break;
                case 'result':
                  handlers.onResult(event.data as BacktestResult);
                  running = false;
                  break;
                case 'error':
                  handlers.onError(createBacktestError(
                    'SERVER_ERROR',
                    String(event.data),
                    false
                  ));
                  running = false;
                  break;
              }
            }
          }
        }
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        // Cancelled by user - don't report as error
        console.log('[BacktestService] Backtest cancelled by user');
      } else if (error instanceof Error && 'code' in error) {
        handlers.onError(error as BacktestServiceError);
      } else {
        handlers.onError(createBacktestError(
          'CONNECTION_FAILED',
          error instanceof Error ? error.message : 'Connection failed',
          true
        ));
      }
    } finally {
      running = false;
      handlers.onConnectionChange?.(false);

      if (reader) {
        try {
          await reader.cancel();
        } catch {
          // Ignore cancel errors
        }
        reader = null;
      }

      abortController = null;
    }
  };

  const cancel = (): void => {
    if (abortController) {
      abortController.abort();
      running = false;
      console.log('[BacktestService] Backtest cancelled');
    }
  };

  const isRunning = (): boolean => running;

  return { start, cancel, isRunning };
}

// ============================================================================
// Utility: Check Backtest Status
// ============================================================================

export interface BacktestStatusResponse {
  model_id: string;
  start_date: string;
  end_date: string;
  trade_count: number;
  has_data: boolean;
}

/**
 * Check if backtest data already exists for a date range
 */
export async function checkBacktestStatus(
  startDate: string,
  endDate: string,
  modelId: string
): Promise<BacktestStatusResponse | null> {
  try {
    const url = `/api/backtest/status/${modelId}?start_date=${startDate}&end_date=${endDate}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });

    if (!response.ok) {
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('[BacktestService] Failed to check status:', error);
    return null;
  }
}

// ============================================================================
// Utility: Run Simple Backtest (non-streaming)
// ============================================================================

/**
 * Run backtest without streaming (simpler, but no progress updates)
 */
export async function runSimpleBacktest(
  request: BacktestRequest
): Promise<BacktestResult> {
  const url = `/api/backtest`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      start_date: request.startDate,
      end_date: request.endDate,
      model_id: request.modelId,
      force_regenerate: request.forceRegenerate ?? false,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw createBacktestError(
      'SERVER_ERROR',
      `Server error (${response.status}): ${errorText}`,
      response.status >= 500
    );
  }

  const result = await response.json();
  const validated = BacktestResultSchema.safeParse(result);

  if (!validated.success) {
    throw createBacktestError(
      'PARSE_ERROR',
      `Invalid response: ${validated.error.message}`,
      false
    );
  }

  return validated.data;
}

// ============================================================================
// Date Range Presets
// ============================================================================

export interface DateRangePreset {
  id: string;
  label: string;
  description: string;
  startDate: string;
  endDate: string;
}

/**
 * Pipeline dates interface - matches API response
 */
export interface PipelineDates {
  model_version: string;
  dates: {
    data_start: string;
    data_end: string;
    training_start: string;
    training_end: string;
    validation_start: string;
    validation_end: string;
    test_start: string;
    test_end: string;
  };
  metadata: {
    config_source: string;
    last_updated: string;
  };
}

/**
 * Default date ranges (fallback if API fails)
 * Uses SSOT values from ssot.contract.ts
 *
 * @deprecated Use PIPELINE_DATE_RANGES from ssot.contract.ts directly
 */
export const BACKTEST_DATE_RANGES = PIPELINE_DATE_RANGES;

/**
 * Fetch pipeline dates from API (reads from official config)
 */
export async function fetchPipelineDates(): Promise<PipelineDates | null> {
  try {
    const response = await fetch('/api/pipeline/dates');
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error('[BacktestService] Failed to fetch pipeline dates:', error);
    return null;
  }
}

/**
 * Get formatted date range description
 */
export function formatDateRange(startDate: string, endDate: string): string {
  const start = new Date(startDate);
  const end = new Date(endDate);

  const formatMonth = (d: Date) => d.toLocaleDateString('es-ES', { month: 'short', year: 'numeric' });
  const formatFull = (d: Date) => d.toLocaleDateString('es-ES', { day: 'numeric', month: 'short', year: 'numeric' });

  // If same month, show full dates
  if (start.getMonth() === end.getMonth() && start.getFullYear() === end.getFullYear()) {
    return `${start.getDate()} - ${formatFull(end)}`;
  }

  return `${formatMonth(start)} - ${formatMonth(end)}`;
}

/**
 * Get predefined date range presets for backtest
 * Uses SSOT values from ssot.contract.ts
 * Simplified: Validación, Test, Ambos, Personalizado
 */
export function getDateRangePresets(): DateRangePreset[] {
  // Use SSOT preset list and config
  return BACKTEST_PRESETS.map((presetId) => {
    const config = BACKTEST_PRESET_CONFIG[presetId];
    const dateRange = getPresetDateRange(presetId);

    return {
      id: presetId,
      label: config.labelEs, // Use Spanish label
      description: config.descriptionEs,
      startDate: dateRange.startDate,
      endDate: dateRange.endDate,
    };
  });
}
