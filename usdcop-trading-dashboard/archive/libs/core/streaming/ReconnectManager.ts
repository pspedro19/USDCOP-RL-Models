/**
 * ReconnectManager - Enterprise-Grade Auto-Reconnection System
 * Smart reconnection with exponential backoff, circuit breaker, and quality monitoring
 */

import { EventEmitter } from 'eventemitter3';
import { BehaviorSubject, Subject, Observable, timer, interval } from 'rxjs';
import { takeUntil, switchMap, retry, delay, tap } from 'rxjs/operators';

import type {
  ReconnectConfig,
  ConnectionState,
  StreamSource,
  StreamError
} from '../types/streaming-types';

export interface ReconnectAttempt {
  readonly attemptNumber: number;
  readonly timestamp: number;
  readonly delay: number;
  readonly reason: string;
  readonly sourceId: string;
  readonly success: boolean;
  readonly errorMessage?: string;
  readonly duration?: number;
}

export interface CircuitBreakerState {
  readonly isOpen: boolean;
  readonly failureCount: number;
  readonly lastFailureTime: number;
  readonly nextRetryTime: number;
  readonly state: 'closed' | 'open' | 'half_open';
}

export interface ReconnectMetrics {
  readonly sourceId: string;
  readonly totalAttempts: number;
  readonly successfulReconnects: number;
  readonly failedReconnects: number;
  readonly averageReconnectTime: number;
  readonly currentStreak: number;
  readonly longestDowntime: number;
  readonly uptime: number;
  readonly reliability: number;
  readonly lastAttempt?: ReconnectAttempt;
  readonly circuitBreaker: CircuitBreakerState;
}

export interface ReconnectManagerConfig {
  readonly globalRetryLimit: number;
  readonly globalBackoffMultiplier: number;
  readonly circuitBreakerThreshold: number;
  readonly circuitBreakerTimeout: number;
  readonly healthCheckInterval: number;
  readonly enableAdaptiveBackoff: boolean;
  readonly enableJitter: boolean;
  readonly maxJitterPercent: number;
  readonly qualityThreshold: number;
}

export class ReconnectManager extends EventEmitter {
  private readonly config: ReconnectManagerConfig;
  private readonly sources = new Map<string, StreamSource>();
  private readonly reconnectStates = new Map<string, ReconnectState>();
  private readonly metrics = new Map<string, ReconnectMetrics>();
  private readonly circuitBreakers = new Map<string, CircuitBreaker>();

  private readonly connectionStates$ = new BehaviorSubject<Map<string, ConnectionState>>(new Map());
  private readonly reconnectAttempts$ = new Subject<ReconnectAttempt>();
  private readonly destroy$ = new Subject<void>();

  private healthCheckTimer?: NodeJS.Timeout;
  private metricsTimer?: NodeJS.Timeout;

  constructor(config: ReconnectManagerConfig) {
    super();
    this.config = config;
    this.initialize();
  }

  // ==========================================
  // PUBLIC API
  // ==========================================

  public registerSource(source: StreamSource): void {
    this.sources.set(source.id, source);

    // Initialize reconnect state
    const reconnectState = new ReconnectState(source.id, source.reconnectConfig);
    this.reconnectStates.set(source.id, reconnectState);

    // Initialize circuit breaker
    const circuitBreaker = new CircuitBreaker(
      source.id,
      this.config.circuitBreakerThreshold,
      this.config.circuitBreakerTimeout
    );
    this.circuitBreakers.set(source.id, circuitBreaker);

    // Initialize metrics
    this.initializeMetrics(source.id);

    this.emit('source_registered', { sourceId: source.id, source });
  }

  public unregisterSource(sourceId: string): void {
    const reconnectState = this.reconnectStates.get(sourceId);
    if (reconnectState) {
      reconnectState.destroy();
      this.reconnectStates.delete(sourceId);
    }

    this.sources.delete(sourceId);
    this.metrics.delete(sourceId);
    this.circuitBreakers.delete(sourceId);

    this.emit('source_unregistered', { sourceId });
  }

  public async attemptReconnect(
    sourceId: string,
    reason: string = 'manual',
    force: boolean = false
  ): Promise<boolean> {
    const source = this.sources.get(sourceId);
    const reconnectState = this.reconnectStates.get(sourceId);
    const circuitBreaker = this.circuitBreakers.get(sourceId);

    if (!source || !reconnectState) {
      throw new Error(`Source not found: ${sourceId}`);
    }

    // Check circuit breaker unless forced
    if (!force && circuitBreaker && circuitBreaker.isOpen()) {
      this.emit('reconnect_blocked', {
        sourceId,
        reason: 'circuit_breaker_open',
        nextRetryTime: circuitBreaker.getNextRetryTime()
      });
      return false;
    }

    // Check if already reconnecting
    if (reconnectState.isReconnecting && !force) {
      this.emit('reconnect_blocked', {
        sourceId,
        reason: 'already_reconnecting'
      });
      return false;
    }

    // Check attempt limits
    if (!force && reconnectState.attemptCount >= source.reconnectConfig.maxAttempts) {
      this.emit('reconnect_failed', {
        sourceId,
        reason: 'max_attempts_exceeded',
        attemptCount: reconnectState.attemptCount
      });
      return false;
    }

    return this.performReconnect(sourceId, reason, force);
  }

  public async performReconnect(
    sourceId: string,
    reason: string,
    force: boolean = false
  ): Promise<boolean> {
    const source = this.sources.get(sourceId);
    const reconnectState = this.reconnectStates.get(sourceId);
    const circuitBreaker = this.circuitBreakers.get(sourceId);

    if (!source || !reconnectState || !circuitBreaker) {
      return false;
    }

    const attemptNumber = reconnectState.attemptCount + 1;
    const startTime = Date.now();

    // Calculate delay
    const delay = this.calculateDelay(source.reconnectConfig, attemptNumber);

    const attempt: ReconnectAttempt = {
      attemptNumber,
      timestamp: startTime,
      delay,
      reason,
      sourceId,
      success: false
    };

    this.emit('reconnect_attempt_started', attempt);
    this.reconnectAttempts$.next(attempt);

    try {
      // Mark as reconnecting
      reconnectState.startReconnect();

      // Wait for calculated delay
      if (delay > 0) {
        await this.sleep(delay);
      }

      // Attempt to connect
      const success = await this.connectSource(sourceId);

      const duration = Date.now() - startTime;
      const completedAttempt = {
        ...attempt,
        success,
        duration
      };

      if (success) {
        // Reset state on successful reconnect
        reconnectState.onSuccess();
        circuitBreaker.onSuccess();

        this.updateMetrics(sourceId, completedAttempt);
        this.emit('reconnect_success', completedAttempt);

        return true;
      } else {
        throw new Error('Connection failed');
      }

    } catch (error) {
      const duration = Date.now() - startTime;
      const failedAttempt = {
        ...attempt,
        success: false,
        duration,
        errorMessage: error.message
      };

      // Update failure state
      reconnectState.onFailure();
      circuitBreaker.onFailure();

      this.updateMetrics(sourceId, failedAttempt);
      this.emit('reconnect_failed', failedAttempt);

      // Schedule next attempt if within limits
      if (!force && attemptNumber < source.reconnectConfig.maxAttempts) {
        this.scheduleNextAttempt(sourceId, reason);
      }

      return false;
    }
  }

  public getConnectionState(sourceId: string): ConnectionState {
    const reconnectState = this.reconnectStates.get(sourceId);
    if (!reconnectState) {
      return 'disconnected';
    }

    if (reconnectState.isReconnecting) {
      return 'reconnecting';
    }

    return reconnectState.isConnected ? 'connected' : 'disconnected';
  }

  public getMetrics(sourceId?: string): ReconnectMetrics[] {
    if (sourceId) {
      const metrics = this.metrics.get(sourceId);
      return metrics ? [metrics] : [];
    }

    return Array.from(this.metrics.values());
  }

  public getReconnectStream(): Observable<ReconnectAttempt> {
    return this.reconnectAttempts$.asObservable().pipe(
      takeUntil(this.destroy$)
    );
  }

  public getConnectionStateStream(): Observable<Map<string, ConnectionState>> {
    return this.connectionStates$.asObservable().pipe(
      takeUntil(this.destroy$)
    );
  }

  // ==========================================
  // CONNECTION MANAGEMENT
  // ==========================================

  private async connectSource(sourceId: string): Promise<boolean> {
    // This would integrate with the actual WebSocket connection logic
    // For now, simulate connection attempt
    return new Promise((resolve, reject) => {
      // Simulate connection delay
      setTimeout(() => {
        // Simulate 80% success rate for testing
        const success = Math.random() > 0.2;

        if (success) {
          this.onConnectionEstablished(sourceId);
          resolve(true);
        } else {
          reject(new Error('Connection timeout'));
        }
      }, 1000 + Math.random() * 2000);
    });
  }

  private onConnectionEstablished(sourceId: string): void {
    const reconnectState = this.reconnectStates.get(sourceId);
    if (reconnectState) {
      reconnectState.onConnectionEstablished();
    }

    this.updateConnectionState(sourceId, 'connected');
    this.emit('connection_established', { sourceId, timestamp: Date.now() });
  }

  private onConnectionLost(sourceId: string, reason: string): void {
    const reconnectState = this.reconnectStates.get(sourceId);
    if (reconnectState) {
      reconnectState.onConnectionLost();
    }

    this.updateConnectionState(sourceId, 'disconnected');
    this.emit('connection_lost', { sourceId, reason, timestamp: Date.now() });

    // Automatically attempt reconnect
    this.scheduleNextAttempt(sourceId, reason);
  }

  private updateConnectionState(sourceId: string, state: ConnectionState): void {
    const currentStates = this.connectionStates$.value;
    const newStates = new Map(currentStates);
    newStates.set(sourceId, state);
    this.connectionStates$.next(newStates);
  }

  // ==========================================
  // SCHEDULING & TIMING
  // ==========================================

  private calculateDelay(config: ReconnectConfig, attemptNumber: number): number {
    let delay = config.initialDelay * Math.pow(config.backoffMultiplier, attemptNumber - 1);

    // Apply max delay limit
    delay = Math.min(delay, config.maxDelay);

    // Apply jitter if enabled
    if (config.jitter && this.config.enableJitter) {
      const jitterRange = delay * (this.config.maxJitterPercent / 100);
      const jitter = (Math.random() - 0.5) * 2 * jitterRange;
      delay += jitter;
    }

    return Math.max(0, Math.round(delay));
  }

  private scheduleNextAttempt(sourceId: string, reason: string): void {
    const source = this.sources.get(sourceId);
    const reconnectState = this.reconnectStates.get(sourceId);

    if (!source || !reconnectState || !source.reconnectConfig.enabled) {
      return;
    }

    const nextAttemptNumber = reconnectState.attemptCount + 1;

    if (nextAttemptNumber > source.reconnectConfig.maxAttempts) {
      this.emit('max_attempts_exceeded', { sourceId, attemptCount: nextAttemptNumber - 1 });
      return;
    }

    const delay = this.calculateDelay(source.reconnectConfig, nextAttemptNumber);

    setTimeout(() => {
      this.attemptReconnect(sourceId, reason).catch(error => {
        this.emit('schedule_error', { sourceId, error });
      });
    }, delay);

    this.emit('next_attempt_scheduled', {
      sourceId,
      attemptNumber: nextAttemptNumber,
      delay,
      scheduledTime: Date.now() + delay
    });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ==========================================
  // METRICS & MONITORING
  // ==========================================

  private initializeMetrics(sourceId: string): void {
    const metrics: ReconnectMetrics = {
      sourceId,
      totalAttempts: 0,
      successfulReconnects: 0,
      failedReconnects: 0,
      averageReconnectTime: 0,
      currentStreak: 0,
      longestDowntime: 0,
      uptime: 0,
      reliability: 1.0,
      circuitBreaker: {
        isOpen: false,
        failureCount: 0,
        lastFailureTime: 0,
        nextRetryTime: 0,
        state: 'closed'
      }
    };

    this.metrics.set(sourceId, metrics);
  }

  private updateMetrics(sourceId: string, attempt: ReconnectAttempt): void {
    const metrics = this.metrics.get(sourceId);
    const circuitBreaker = this.circuitBreakers.get(sourceId);

    if (!metrics || !circuitBreaker) return;

    const updatedMetrics = { ...metrics };

    // Update attempt counts
    updatedMetrics.totalAttempts++;

    if (attempt.success) {
      updatedMetrics.successfulReconnects++;
      updatedMetrics.currentStreak++;

      // Update average reconnect time
      const totalTime = metrics.averageReconnectTime * (metrics.successfulReconnects - 1) + (attempt.duration || 0);
      updatedMetrics.averageReconnectTime = totalTime / metrics.successfulReconnects;
    } else {
      updatedMetrics.failedReconnects++;
      updatedMetrics.currentStreak = 0;
    }

    // Calculate reliability
    updatedMetrics.reliability = updatedMetrics.totalAttempts > 0 ?
      updatedMetrics.successfulReconnects / updatedMetrics.totalAttempts : 1.0;

    // Update circuit breaker state
    updatedMetrics.circuitBreaker = {
      isOpen: circuitBreaker.isOpen(),
      failureCount: circuitBreaker.getFailureCount(),
      lastFailureTime: circuitBreaker.getLastFailureTime(),
      nextRetryTime: circuitBreaker.getNextRetryTime(),
      state: circuitBreaker.getState()
    };

    // Update last attempt
    updatedMetrics.lastAttempt = attempt;

    this.metrics.set(sourceId, updatedMetrics);
    this.emit('metrics_updated', { sourceId, metrics: updatedMetrics });
  }

  private setupHealthCheck(): void {
    this.healthCheckTimer = setInterval(() => {
      this.performHealthCheck();
    }, this.config.healthCheckInterval);
  }

  private setupMetricsCollection(): void {
    this.metricsTimer = setInterval(() => {
      this.collectMetrics();
    }, 5000);
  }

  private performHealthCheck(): void {
    this.sources.forEach((source, sourceId) => {
      const state = this.getConnectionState(sourceId);
      const metrics = this.metrics.get(sourceId);

      if (state === 'connected' && metrics) {
        // Check connection quality
        const reliability = metrics.reliability;

        if (reliability < this.config.qualityThreshold) {
          this.emit('quality_degraded', {
            sourceId,
            reliability,
            threshold: this.config.qualityThreshold
          });
        }
      }

      // Update uptime
      if (metrics && state === 'connected') {
        const reconnectState = this.reconnectStates.get(sourceId);
        if (reconnectState) {
          const uptime = Date.now() - reconnectState.lastConnectTime;
          (metrics as any).uptime = uptime;
        }
      }
    });
  }

  private collectMetrics(): void {
    const allMetrics = Array.from(this.metrics.values());

    this.emit('metrics_collected', {
      timestamp: Date.now(),
      sources: allMetrics.length,
      totalAttempts: allMetrics.reduce((sum, m) => sum + m.totalAttempts, 0),
      successRate: allMetrics.length > 0 ?
        allMetrics.reduce((sum, m) => sum + m.reliability, 0) / allMetrics.length : 1,
      activeCircuitBreakers: allMetrics.filter(m => m.circuitBreaker.isOpen).length
    });
  }

  // ==========================================
  // INITIALIZATION & CLEANUP
  // ==========================================

  private initialize(): void {
    this.setupHealthCheck();
    this.setupMetricsCollection();

    // Setup global error handlers
    this.on('connection_lost', (event) => {
      this.onConnectionLost(event.sourceId, event.reason);
    });

    this.on('connection_error', (event) => {
      this.onConnectionLost(event.sourceId, 'error');
    });
  }

  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear timers
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
    }

    // Destroy all reconnect states
    this.reconnectStates.forEach(state => state.destroy());
    this.reconnectStates.clear();

    // Clear data
    this.sources.clear();
    this.metrics.clear();
    this.circuitBreakers.clear();

    // Complete observables
    this.connectionStates$.complete();
    this.reconnectAttempts$.complete();

    this.removeAllListeners();
  }
}

// ==========================================
// RECONNECT STATE MANAGEMENT
// ==========================================

class ReconnectState {
  public readonly sourceId: string;
  public readonly config: ReconnectConfig;

  public attemptCount = 0;
  public isReconnecting = false;
  public isConnected = false;
  public lastConnectTime = 0;
  public lastAttemptTime = 0;
  public failureStreak = 0;

  constructor(sourceId: string, config: ReconnectConfig) {
    this.sourceId = sourceId;
    this.config = config;
  }

  public startReconnect(): void {
    this.isReconnecting = true;
    this.attemptCount++;
    this.lastAttemptTime = Date.now();
  }

  public onSuccess(): void {
    this.isReconnecting = false;
    this.isConnected = true;
    this.attemptCount = 0;
    this.failureStreak = 0;
    this.lastConnectTime = Date.now();
  }

  public onFailure(): void {
    this.isReconnecting = false;
    this.isConnected = false;
    this.failureStreak++;
  }

  public onConnectionEstablished(): void {
    this.isConnected = true;
    this.lastConnectTime = Date.now();
  }

  public onConnectionLost(): void {
    this.isConnected = false;
  }

  public destroy(): void {
    // Cleanup if needed
  }
}

// ==========================================
// CIRCUIT BREAKER IMPLEMENTATION
// ==========================================

class CircuitBreaker {
  private readonly sourceId: string;
  private readonly failureThreshold: number;
  private readonly timeout: number;

  private failureCount = 0;
  private lastFailureTime = 0;
  private state: 'closed' | 'open' | 'half_open' = 'closed';

  constructor(sourceId: string, failureThreshold: number, timeout: number) {
    this.sourceId = sourceId;
    this.failureThreshold = failureThreshold;
    this.timeout = timeout;
  }

  public onSuccess(): void {
    this.failureCount = 0;
    this.state = 'closed';
  }

  public onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.failureThreshold) {
      this.state = 'open';
    }
  }

  public isOpen(): boolean {
    if (this.state === 'open') {
      // Check if timeout period has passed
      if (Date.now() - this.lastFailureTime >= this.timeout) {
        this.state = 'half_open';
        return false;
      }
      return true;
    }

    return false;
  }

  public getState(): 'closed' | 'open' | 'half_open' {
    // Update state based on timeout
    if (this.state === 'open' && Date.now() - this.lastFailureTime >= this.timeout) {
      this.state = 'half_open';
    }

    return this.state;
  }

  public getFailureCount(): number {
    return this.failureCount;
  }

  public getLastFailureTime(): number {
    return this.lastFailureTime;
  }

  public getNextRetryTime(): number {
    if (this.state === 'open') {
      return this.lastFailureTime + this.timeout;
    }
    return 0;
  }
}