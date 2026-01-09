/**
 * WebSocketManager - Elite Trading Platform WebSocket Foundation
 * High-performance, resilient WebSocket connection management
 */

import { EventEmitter } from 'eventemitter3';
import { Subject, BehaviorSubject, Observable, fromEvent, timer } from 'rxjs';
import { filter, map, retry, takeUntil, switchMap } from 'rxjs/operators';
import type {
  WebSocketMessage,
  SubscriptionRequest,
  StreamConfig,
  ConnectionStatus,
  TradingPlatformEvent
} from '../types';

export interface WebSocketManagerConfig {
  readonly url: string;
  readonly protocols?: string[];
  readonly maxReconnectAttempts: number;
  readonly reconnectDelay: number;
  readonly heartbeatInterval: number;
  readonly connectionTimeout: number;
  readonly messageTimeout: number;
  readonly maxMessageQueue: number;
  readonly enableCompression: boolean;
  readonly enableLogging: boolean;
  readonly autoReconnect: boolean;
}

export interface WebSocketConnection {
  readonly id: string;
  readonly url: string;
  readonly ws: WebSocket | null;
  readonly status: ConnectionStatus;
  readonly connectedAt?: number;
  readonly lastPing?: number;
  readonly latency: number;
  readonly messageCount: number;
  readonly errorCount: number;
  readonly subscriptions: Set<string>;
}

export interface ConnectionMetrics {
  readonly totalConnections: number;
  readonly activeConnections: number;
  readonly totalMessages: number;
  readonly messagesPerSecond: number;
  readonly averageLatency: number;
  readonly errorRate: number;
  readonly uptime: number;
  readonly reconnections: number;
}

export class WebSocketManager extends EventEmitter {
  private readonly config: WebSocketManagerConfig;
  private readonly connections = new Map<string, WebSocketConnection>();
  private readonly messageQueue: WebSocketMessage[] = [];
  private readonly subscriptions = new Map<string, SubscriptionRequest>();

  // RxJS Subjects for reactive streams
  private readonly connectionStatus$ = new BehaviorSubject<ConnectionStatus>('disconnected');
  private readonly messages$ = new Subject<WebSocketMessage>();
  private readonly errors$ = new Subject<Error>();
  private readonly destroy$ = new Subject<void>();

  // Performance tracking
  private readonly metrics: ConnectionMetrics = {
    totalConnections: 0,
    activeConnections: 0,
    totalMessages: 0,
    messagesPerSecond: 0,
    averageLatency: 0,
    errorRate: 0,
    uptime: 0,
    reconnections: 0
  };

  private heartbeatTimer?: NodeJS.Timeout;
  private metricsTimer?: NodeJS.Timeout;
  private startTime = Date.now();

  constructor(config: WebSocketManagerConfig) {
    super();

    this.config = config;
    this.initialize();
  }

  private initialize(): void {
    this.setupMetricsCollection();
    this.setupHeartbeat();
  }

  /**
   * Create a new WebSocket connection
   */
  public async connect(connectionId?: string): Promise<string> {
    const id = connectionId || this.generateId();

    if (this.connections.has(id)) {
      throw new Error(`Connection ${id} already exists`);
    }

    const connection: WebSocketConnection = {
      id,
      url: this.config.url,
      ws: null,
      status: 'disconnected',
      latency: 0,
      messageCount: 0,
      errorCount: 0,
      subscriptions: new Set()
    };

    this.connections.set(id, connection);

    try {
      await this.establishConnection(connection);
      (this.metrics as any).totalConnections++;
      this.log('info', `Connection established: ${id}`);
      return id;
    } catch (error) {
      this.connections.delete(id);
      throw error;
    }
  }

  /**
   * Disconnect a WebSocket connection
   */
  public disconnect(connectionId: string): void {
    const connection = this.connections.get(connectionId);
    if (!connection) {
      this.log('warn', `Connection not found: ${connectionId}`);
      return;
    }

    this.closeConnection(connection);
    this.connections.delete(connectionId);
    this.updateConnectionStatus();
    this.log('info', `Connection closed: ${connectionId}`);
  }

  /**
   * Send a message through a specific connection
   */
  public send(connectionId: string, message: WebSocketMessage): void {
    const connection = this.connections.get(connectionId);
    if (!connection || !connection.ws) {
      this.queueMessage(message);
      this.log('warn', `Message queued - connection unavailable: ${connectionId}`);
      return;
    }

    if (connection.ws.readyState !== WebSocket.OPEN) {
      this.queueMessage(message);
      this.log('warn', `Message queued - connection not ready: ${connectionId}`);
      return;
    }

    try {
      const data = JSON.stringify(message);
      connection.ws.send(data);
      (connection as any).messageCount++;
      (this.metrics as any).totalMessages++;

      this.emit('message.sent', { connectionId, message });
      this.log('debug', `Message sent to ${connectionId}`, message);

    } catch (error) {
      this.handleError(error as Error, { connectionId, message });
    }
  }

  /**
   * Subscribe to a channel through all active connections
   */
  public subscribe(request: SubscriptionRequest): void {
    this.subscriptions.set(request.id, request);

    // Send subscription to all active connections
    Array.from(this.connections.entries()).forEach(([connectionId, connection]) => {
      if (connection.status === 'connected') {
        connection.subscriptions.add(request.id);
        this.send(connectionId, {
          id: this.generateId(),
          type: 'subscribe',
          data: request,
          timestamp: Date.now()
        });
      }
    });

    this.emit('subscription.created', request);
    this.log('info', `Subscription created: ${request.id}`);
  }

  /**
   * Unsubscribe from a channel
   */
  public unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      this.log('warn', `Subscription not found: ${subscriptionId}`);
      return;
    }

    // Remove from all connections
    Array.from(this.connections.values()).forEach(connection => {
      connection.subscriptions.delete(subscriptionId);
      if (connection.status === 'connected') {
        this.send(connection.id, {
          id: this.generateId(),
          type: 'unsubscribe',
          data: { id: subscriptionId },
          timestamp: Date.now()
        });
      }
    });

    this.subscriptions.delete(subscriptionId);
    this.emit('subscription.removed', { id: subscriptionId });
    this.log('info', `Subscription removed: ${subscriptionId}`);
  }

  /**
   * Get reactive observable for messages
   */
  public getMessages$(): Observable<WebSocketMessage> {
    return this.messages$.asObservable();
  }

  /**
   * Get reactive observable for connection status
   */
  public getConnectionStatus$(): Observable<ConnectionStatus> {
    return this.connectionStatus$.asObservable();
  }

  /**
   * Get filtered message stream
   */
  public getFilteredMessages$(filterFn: {
    type?: string;
    channel?: string;
    connectionId?: string;
  }): Observable<WebSocketMessage> {
    return this.messages$.pipe(
      filter(message => {
        if (filterFn.type && message.type !== filterFn.type) return false;
        if (filterFn.channel && message.channel !== filterFn.channel) return false;
        return true;
      })
    );
  }

  /**
   * Get connection metrics
   */
  public getMetrics(): ConnectionMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }

  /**
   * Get connection information
   */
  public getConnection(connectionId: string): WebSocketConnection | null {
    return this.connections.get(connectionId) || null;
  }

  /**
   * Get all active connections
   */
  public getActiveConnections(): WebSocketConnection[] {
    return Array.from(this.connections.values()).filter(c => c.status === 'connected');
  }

  /**
   * Force reconnect all connections
   */
  public async reconnectAll(): Promise<void> {
    const reconnectPromises: Promise<void>[] = [];

    Array.from(this.connections.values()).forEach(connection => {
      reconnectPromises.push(this.reconnectConnection(connection));
    });

    await Promise.allSettled(reconnectPromises);
    this.log('info', 'All connections reconnected');
  }

  /**
   * Destroy WebSocket manager and cleanup
   */
  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear timers
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    if (this.metricsTimer) clearInterval(this.metricsTimer);

    // Close all connections
    Array.from(this.connections.values()).forEach(connection => {
      this.closeConnection(connection);
    });

    // Clean up
    this.connections.clear();
    this.subscriptions.clear();
    this.messageQueue.length = 0;

    // Complete observables
    this.connectionStatus$.complete();
    this.messages$.complete();
    this.errors$.complete();

    this.removeAllListeners();
    this.log('info', 'WebSocketManager destroyed');
  }

  // Private helper methods
  private async establishConnection(connection: WebSocketConnection): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(this.config.url, this.config.protocols);
        (connection as any).ws = ws;
        (connection as any).status = 'connecting';

        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('Connection timeout'));
        }, this.config.connectionTimeout);

        ws.onopen = () => {
          clearTimeout(timeout);
          (connection as any).status = 'connected';
          (connection as any).connectedAt = Date.now();
          this.setupConnectionEventHandlers(connection);
          this.updateConnectionStatus();
          this.resubscribeToChannels(connection);
          this.processMessageQueue(connection);
          resolve();
        };

        ws.onerror = (error) => {
          clearTimeout(timeout);
          (connection as any).errorCount++;
          this.handleConnectionError(connection, error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  private setupConnectionEventHandlers(connection: WebSocketConnection): void {
    if (!connection.ws) return;

    connection.ws.onmessage = (event) => {
      this.handleMessage(connection, event);
    };

    connection.ws.onclose = (event) => {
      this.handleConnectionClose(connection, event);
    };

    connection.ws.onerror = (error) => {
      this.handleConnectionError(connection, error);
    };
  }

  private handleMessage(connection: WebSocketConnection, event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      (message as any).timestamp = Date.now();

      // Update latency if this is a pong message
      if (message.type === 'pong' && connection.lastPing) {
        (connection as any).latency = Date.now() - connection.lastPing;
      }

      (connection as any).messageCount++;
      this.messages$.next(message);
      this.emit('message.received', { connectionId: connection.id, message });

      this.log('debug', `Message received from ${connection.id}`, message);

    } catch (error) {
      this.handleError(error as Error, { connectionId: connection.id, rawMessage: event.data });
    }
  }

  private handleConnectionClose(connection: WebSocketConnection, event: CloseEvent): void {
    (connection as any).status = 'disconnected';
    (connection as any).ws = null;

    this.updateConnectionStatus();
    this.emit('connection.closed', { connectionId: connection.id, code: event.code, reason: event.reason });

    this.log('warn', `Connection closed: ${connection.id} (${event.code}: ${event.reason})`);

    // Auto-reconnect if enabled
    if (this.config.autoReconnect) {
      this.scheduleReconnect(connection);
    }
  }

  private handleConnectionError(connection: WebSocketConnection, error: Event | Error): void {
    (connection as any).status = 'error';
    (connection as any).errorCount++;

    this.updateConnectionStatus();
    this.errors$.next(error instanceof Error ? error : new Error('WebSocket error'));
    this.emit('connection.error', { connectionId: connection.id, error });

    this.log('error', `Connection error: ${connection.id}`, error);
  }

  private async scheduleReconnect(connection: WebSocketConnection): Promise<void> {
    if (connection.errorCount >= this.config.maxReconnectAttempts) {
      this.log('error', `Max reconnect attempts reached for ${connection.id}`);
      return;
    }

    (connection as any).status = 'reconnecting';
    this.updateConnectionStatus();

    const delay = this.config.reconnectDelay * Math.pow(2, connection.errorCount);

    setTimeout(async () => {
      try {
        await this.reconnectConnection(connection);
        (this.metrics as any).reconnections++;
      } catch (error) {
        this.log('error', `Reconnection failed for ${connection.id}`, error);
      }
    }, delay);
  }

  private async reconnectConnection(connection: WebSocketConnection): Promise<void> {
    this.closeConnection(connection);
    await this.establishConnection(connection);
    this.log('info', `Connection reconnected: ${connection.id}`);
  }

  private closeConnection(connection: WebSocketConnection): void {
    if (connection.ws) {
      connection.ws.close();
      (connection as any).ws = null;
    }
    (connection as any).status = 'disconnected';
  }

  private resubscribeToChannels(connection: WebSocketConnection): void {
    Array.from(connection.subscriptions).forEach(subscriptionId => {
      const subscription = this.subscriptions.get(subscriptionId);
      if (subscription) {
        this.send(connection.id, {
          id: this.generateId(),
          type: 'subscribe',
          data: subscription,
          timestamp: Date.now()
        });
      }
    });
  }

  private processMessageQueue(connection: WebSocketConnection): void {
    const messages = this.messageQueue.splice(0);
    for (const message of messages) {
      this.send(connection.id, message);
    }
  }

  private queueMessage(message: WebSocketMessage): void {
    if (this.messageQueue.length >= this.config.maxMessageQueue) {
      this.messageQueue.shift(); // Remove oldest message
    }
    this.messageQueue.push(message);
  }

  private updateConnectionStatus(): void {
    const activeConnections = this.getActiveConnections();

    let status: ConnectionStatus = 'disconnected';
    if (activeConnections.length > 0) {
      status = 'connected';
    } else if (Array.from(this.connections.values()).some(c => c.status === 'reconnecting')) {
      status = 'reconnecting';
    }

    this.connectionStatus$.next(status);
    (this.metrics as any).activeConnections = activeConnections.length;
  }

  private setupHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
    }, this.config.heartbeatInterval);
  }

  private sendHeartbeat(): void {
    const now = Date.now();

    Array.from(this.connections.values()).forEach(connection => {
      if (connection.status === 'connected') {
        (connection as any).lastPing = now;
        this.send(connection.id, {
          id: this.generateId(),
          type: 'ping',
          data: { timestamp: now },
          timestamp: now
        });
      }
    });
  }

  private setupMetricsCollection(): void {
    this.metricsTimer = setInterval(() => {
      this.updateMetrics();
      this.emit('metrics.updated', this.getMetrics());
    }, 5000);
  }

  private updateMetrics(): void {
    const now = Date.now();
    (this.metrics as any).uptime = now - this.startTime;

    // Calculate messages per second
    const timeDiff = 1000; // 1 second interval
    const recentMessages = this.metrics.totalMessages; // Simplified calculation
    (this.metrics as any).messagesPerSecond = recentMessages / (timeDiff / 1000);

    // Calculate average latency
    const connections = Array.from(this.connections.values());
    const connectedConnections = connections.filter(c => c.status === 'connected');

    if (connectedConnections.length > 0) {
      const totalLatency = connectedConnections.reduce((sum, c) => sum + c.latency, 0);
      (this.metrics as any).averageLatency = totalLatency / connectedConnections.length;
    }

    // Calculate error rate
    const totalErrors = connections.reduce((sum, c) => sum + c.errorCount, 0);
    (this.metrics as any).errorRate = this.metrics.totalConnections > 0
      ? totalErrors / this.metrics.totalConnections
      : 0;
  }

  private handleError(error: Error, context?: any): void {
    this.errors$.next(error);
    this.emit('error', error, context);
    this.log('error', `WebSocketManager error: ${error.message}`, { error, context });
  }

  private log(level: string, message: string, data?: any): void {
    if (!this.config.enableLogging) return;

    const logFn = level === 'error' ? console.error :
                  level === 'warn' ? console.warn :
                  level === 'debug' ? console.debug :
                  console.log;

    logFn(`[WebSocketManager] ${message}`, data || '');
  }

  private generateId(): string {
    return `ws-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }
}

// Singleton instance
let wsManagerInstance: WebSocketManager | null = null;

export function getWebSocketManager(config?: WebSocketManagerConfig): WebSocketManager {
  if (!wsManagerInstance && config) {
    wsManagerInstance = new WebSocketManager(config);
  }
  return wsManagerInstance!;
}

export function resetWebSocketManager(): void {
  if (wsManagerInstance) {
    wsManagerInstance.destroy();
    wsManagerInstance = null;
  }
}