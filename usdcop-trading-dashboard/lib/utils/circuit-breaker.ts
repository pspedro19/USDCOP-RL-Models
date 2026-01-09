/**
 * Circuit Breaker Pattern Implementation
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Backend failing, reject requests immediately
 * - HALF_OPEN: Testing if backend recovered
 *
 * Prevents cascading failures and provides fast-fail behavior.
 *
 * @example
 * ```typescript
 * // Create a circuit breaker for an API endpoint
 * const breaker = getCircuitBreaker('trading-api', {
 *   failureThreshold: 5,
 *   resetTimeout: 30000,
 *   monitorInterval: 5000,
 * });
 *
 * // Wrap API calls with circuit breaker
 * try {
 *   const data = await breaker.execute(async () => {
 *     const response = await fetch('/api/trades');
 *     if (!response.ok) throw new Error('API request failed');
 *     return response.json();
 *   });
 *   console.log('Data:', data);
 * } catch (error) {
 *   if (error instanceof CircuitOpenError) {
 *     console.error('Circuit is open, backend is down');
 *     // Show cached data or fallback UI
 *   } else {
 *     console.error('Request failed:', error);
 *   }
 * }
 *
 * // Check circuit breaker status
 * const status = getCircuitBreakerStatus();
 * console.log('All circuits:', status);
 *
 * // Check specific circuit
 * if (breaker.isOpen()) {
 *   console.log('Circuit is open, using fallback');
 * }
 * ```
 */

type CircuitState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

interface CircuitBreakerConfig {
  failureThreshold: number;      // Failures before opening (default: 5)
  resetTimeout: number;          // ms before trying again (default: 30000)
  monitorInterval: number;       // ms between health checks (default: 5000)
}

interface CircuitBreakerState {
  state: CircuitState;
  failures: number;
  lastFailure: number | null;
  lastSuccess: number | null;
}

class CircuitBreaker {
  private state: CircuitBreakerState;
  private config: CircuitBreakerConfig;
  private name: string;

  constructor(name: string, config?: Partial<CircuitBreakerConfig>) {
    this.name = name;
    this.config = {
      failureThreshold: config?.failureThreshold ?? 5,
      resetTimeout: config?.resetTimeout ?? 30000,
      monitorInterval: config?.monitorInterval ?? 5000,
    };
    this.state = {
      state: 'CLOSED',
      failures: 0,
      lastFailure: null,
      lastSuccess: null,
    };
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state.state === 'OPEN') {
      if (this.shouldAttemptReset()) {
        this.state.state = 'HALF_OPEN';
      } else {
        throw new CircuitOpenError(this.name, this.state);
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private shouldAttemptReset(): boolean {
    return this.state.lastFailure !== null &&
      Date.now() - this.state.lastFailure >= this.config.resetTimeout;
  }

  private onSuccess(): void {
    this.state.failures = 0;
    this.state.lastSuccess = Date.now();
    this.state.state = 'CLOSED';
  }

  private onFailure(): void {
    this.state.failures++;
    this.state.lastFailure = Date.now();
    if (this.state.failures >= this.config.failureThreshold) {
      this.state.state = 'OPEN';
      console.warn(`[CircuitBreaker:${this.name}] Circuit OPENED after ${this.state.failures} failures`);
    }
  }

  getState(): CircuitBreakerState {
    return { ...this.state };
  }

  isOpen(): boolean {
    return this.state.state === 'OPEN';
  }
}

class CircuitOpenError extends Error {
  constructor(name: string, state: CircuitBreakerState) {
    super(`Circuit breaker '${name}' is OPEN. Last failure: ${state.lastFailure}`);
    this.name = 'CircuitOpenError';
  }
}

// Singleton registry for circuit breakers
const circuitBreakers = new Map<string, CircuitBreaker>();

export function getCircuitBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
  if (!circuitBreakers.has(name)) {
    circuitBreakers.set(name, new CircuitBreaker(name, config));
  }
  return circuitBreakers.get(name)!;
}

export function getCircuitBreakerStatus(): Record<string, CircuitBreakerState> {
  const status: Record<string, CircuitBreakerState> = {};
  circuitBreakers.forEach((cb, name) => {
    status[name] = cb.getState();
  });
  return status;
}

export { CircuitBreaker, CircuitOpenError };
export type { CircuitBreakerConfig, CircuitBreakerState, CircuitState };
