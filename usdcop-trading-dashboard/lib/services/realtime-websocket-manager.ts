/**
 * Enhanced Real-Time WebSocket Manager
 * Handles live data updates for both frontend and database synchronization
 * Provides seamless integration between historical and real-time data
 */

import { io, Socket } from 'socket.io-client';

interface RealTimeDataPoint {
  timestamp: string;
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  source: string;
  type: 'tick' | 'candlestick' | 'quote';
}

interface WebSocketConfig {
  reconnectAttempts: number;
  reconnectDelay: number;
  heartbeatInterval: number;
  maxRetries: number;
  backoffMultiplier: number;
}

interface ConnectionStatus {
  connected: boolean;
  lastHeartbeat: Date | null;
  reconnectAttempts: number;
  latency: number;
  quality: 'excellent' | 'good' | 'poor' | 'disconnected';
}

type DataCallback = (data: RealTimeDataPoint) => void;
type StatusCallback = (status: ConnectionStatus) => void;
type ErrorCallback = (error: Error) => void;

class RealTimeWebSocketManager {
  private socket: Socket | null = null;
  private wsConnection: WebSocket | null = null;
  private config: WebSocketConfig;
  private status: ConnectionStatus;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private latencyTimer: NodeJS.Timeout | null = null;

  // Callback arrays
  private dataCallbacks: DataCallback[] = [];
  private statusCallbacks: StatusCallback[] = [];
  private errorCallbacks: ErrorCallback[] = [];

  // Data management
  private dataBuffer: RealTimeDataPoint[] = [];
  private bufferSize = 1000;
  private lastDataTime: Date | null = null;

  // Market status
  private marketOpen = false;
  private tradingHours = {
    start: 8, // 8:00 AM COT
    end: 12.92 // 12:55 PM COT (12 + 55/60)
  };

  constructor(config?: Partial<WebSocketConfig>) {
    this.config = {
      reconnectAttempts: 0,
      reconnectDelay: 1000,
      heartbeatInterval: 30000, // 30 seconds
      maxRetries: 10,
      backoffMultiplier: 1.5,
      ...config
    };

    this.status = {
      connected: false,
      lastHeartbeat: null,
      reconnectAttempts: 0,
      latency: 0,
      quality: 'disconnected'
    };

    this.checkMarketStatus();
    this.startMarketStatusChecker();
  }

  /**
   * Connect to WebSocket with fallback options
   */
  async connect(): Promise<void> {
    try {
      // Try primary WebSocket connection first
      await this.connectPrimaryWebSocket();
    } catch (error) {
      console.warn('Primary WebSocket failed, trying Socket.IO fallback:', error);

      try {
        await this.connectSocketIOFallback();
      } catch (fallbackError) {
        console.warn('Socket.IO fallback failed, using HTTP polling:', fallbackError);
        this.startHttpPolling();
      }
    }
  }

  /**
   * Primary WebSocket connection
   */
  private async connectPrimaryWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Use localhost for WebSocket connection to avoid external IP issues
        const wsUrl = `ws://localhost:5000/api/proxy/ws`;
        this.wsConnection = new WebSocket(wsUrl);


        this.wsConnection.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleIncomingData(data);
          } catch (error) {
            console.error('Error parsing WebSocket data:', error);
          }
        };

        this.wsConnection.onclose = () => {
          console.log('Primary WebSocket disconnected');
          this.handleConnectionLoss();
        };

        this.wsConnection.onerror = (error) => {
          console.error('Primary WebSocket error:', error);
          reject(error);
        };

        // Connection timeout with better error handling
        const connectionTimeout = setTimeout(() => {
          if (this.wsConnection?.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket connection timeout after 10 seconds');
            this.wsConnection?.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000); // Increased timeout to 10 seconds

        // Clear timeout on successful connection
        this.wsConnection.onopen = () => {
          clearTimeout(connectionTimeout);
          console.log('Primary WebSocket connected');
          this.handleConnectionSuccess();
          resolve();
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Socket.IO fallback connection
   */
  private async connectSocketIOFallback(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(`/api/proxy/trading`, {
          transports: ['websocket', 'polling'],
          reconnection: true,
          reconnectionAttempts: this.config.maxRetries,
          reconnectionDelay: this.config.reconnectDelay,
          timeout: 20000, // 20 second timeout
          forceNew: true
        });

        this.socket.on('connect', () => {
          console.log('Socket.IO fallback connected');
          this.handleConnectionSuccess();

          // Subscribe to USDCOP updates
          this.socket?.emit('subscribe', { symbol: 'USDCOP', channels: ['quotes', 'trades'] });
          resolve();
        });

        this.socket.on('market_data', (data) => {
          this.handleIncomingData(data);
        });

        this.socket.on('disconnect', () => {
          console.log('Socket.IO disconnected');
          this.handleConnectionLoss();
        });

        this.socket.on('connect_error', (error) => {
          console.error('Socket.IO connection error:', error);
          reject(error);
        });

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * HTTP polling as last resort
   */
  private startHttpPolling(): void {
    console.log('Starting HTTP polling fallback');

    const pollInterval = this.marketOpen ? 5000 : 30000; // 5s when market open, 30s when closed

    const poll = async () => {
      try {
        const response = await fetch('/api/proxy/ws');
        const data = await response.json();

        if (data && data.price) {
          this.handleIncomingData({
            timestamp: data.timestamp || new Date().toISOString(),
            symbol: data.symbol || 'USDCOP',
            price: data.price,
            bid: data.bid,
            ask: data.ask,
            volume: data.volume,
            source: data.source || 'http_polling',
            type: 'quote'
          });
        }

        // Update connection status
        this.status.connected = true;
        this.status.lastHeartbeat = new Date();
        this.status.quality = 'poor'; // HTTP polling is poor quality
        this.notifyStatusChange();

      } catch (error) {
        console.error('HTTP polling error:', error);
        this.status.connected = false;
        this.status.quality = 'disconnected';
        this.notifyStatusChange();
      }

      // Schedule next poll
      setTimeout(poll, pollInterval);
    };

    poll();
  }

  /**
   * Handle successful connection
   */
  private handleConnectionSuccess(): void {
    this.status.connected = true;
    this.status.reconnectAttempts = 0;
    this.status.lastHeartbeat = new Date();
    this.status.quality = 'excellent';

    this.config.reconnectAttempts = 0;
    this.startHeartbeat();
    this.notifyStatusChange();
  }

  /**
   * Handle connection loss
   */
  private handleConnectionLoss(): void {
    this.status.connected = false;
    this.status.quality = 'disconnected';
    this.notifyStatusChange();

    this.stopHeartbeat();
    this.scheduleReconnect();
  }

  /**
   * Handle incoming real-time data
   */
  private handleIncomingData(rawData: any): void {
    try {
      const data: RealTimeDataPoint = {
        timestamp: rawData.timestamp || new Date().toISOString(),
        symbol: rawData.symbol || 'USDCOP',
        price: parseFloat(rawData.price || rawData.close || 0),
        bid: rawData.bid ? parseFloat(rawData.bid) : undefined,
        ask: rawData.ask ? parseFloat(rawData.ask) : undefined,
        volume: rawData.volume ? parseInt(rawData.volume) : undefined,
        source: rawData.source || 'websocket',
        type: rawData.type || 'quote'
      };

      // Validate data
      if (!data.price || data.price <= 0) {
        console.warn('Invalid price data received:', rawData);
        return;
      }

      // Update buffer
      this.dataBuffer.push(data);
      if (this.dataBuffer.length > this.bufferSize) {
        this.dataBuffer.shift();
      }

      this.lastDataTime = new Date();

      // Notify all callbacks
      this.dataCallbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in data callback:', error);
        }
      });

      // Update database via API
      this.updateDatabase(data);

    } catch (error) {
      console.error('Error processing incoming data:', error);
      this.notifyError(error as Error);
    }
  }

  /**
   * Update database with new data point
   */
  private async updateDatabase(data: RealTimeDataPoint): Promise<void> {
    if (!this.marketOpen) return; // Only update database during market hours

    try {
      const response = await fetch('/api/market/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          timestamp: data.timestamp,
          symbol: data.symbol,
          price: data.price,
          bid: data.bid,
          ask: data.ask,
          volume: data.volume,
          source: data.source
        })
      });

      if (!response.ok) {
        console.warn('Failed to update database:', response.statusText);
      }
    } catch (error) {
      console.warn('Error updating database:', error);
    }
  }

  /**
   * Check if market is currently open
   */
  private checkMarketStatus(): void {
    const now = new Date();
    const bogotaTime = new Date(now.toLocaleString("en-US", {timeZone: "America/Bogota"}));
    const currentHour = bogotaTime.getHours() + (bogotaTime.getMinutes() / 60);
    const dayOfWeek = bogotaTime.getDay();

    // Market is open Monday to Friday, 8:00 AM to 12:55 PM COT
    this.marketOpen = dayOfWeek >= 1 && dayOfWeek <= 5 &&
                     currentHour >= this.tradingHours.start &&
                     currentHour <= this.tradingHours.end;
  }

  /**
   * Start market status checker
   */
  private startMarketStatusChecker(): void {
    setInterval(() => {
      const wasOpen = this.marketOpen;
      this.checkMarketStatus();

      if (wasOpen !== this.marketOpen) {
        console.log(`Market status changed: ${this.marketOpen ? 'OPEN' : 'CLOSED'}`);

        if (this.marketOpen) {
          // Market just opened - ensure we have active connection
          if (!this.status.connected) {
            this.connect();
          }
        }
      }
    }, 60000); // Check every minute
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.wsConnection?.readyState === WebSocket.OPEN) {
        const startTime = Date.now();
        this.wsConnection.send(JSON.stringify({ type: 'ping', timestamp: startTime }));

        // Measure latency (simplified - would need pong response in real implementation)
        this.status.latency = Date.now() - startTime;
      } else if (this.socket?.connected) {
        const startTime = Date.now();
        this.socket.emit('ping', startTime);

        this.socket.once('pong', (timestamp) => {
          this.status.latency = Date.now() - timestamp;
        });
      }

      this.status.lastHeartbeat = new Date();
      this.updateConnectionQuality();
      this.notifyStatusChange();
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Update connection quality based on latency and data freshness
   */
  private updateConnectionQuality(): void {
    if (!this.status.connected) {
      this.status.quality = 'disconnected';
      return;
    }

    const now = new Date();
    const timeSinceLastData = this.lastDataTime ?
      now.getTime() - this.lastDataTime.getTime() : Infinity;

    if (this.status.latency < 100 && timeSinceLastData < 10000) {
      this.status.quality = 'excellent';
    } else if (this.status.latency < 500 && timeSinceLastData < 30000) {
      this.status.quality = 'good';
    } else {
      this.status.quality = 'poor';
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.config.reconnectAttempts >= this.config.maxRetries) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(this.config.backoffMultiplier, this.config.reconnectAttempts),
      30000 // Max 30 seconds
    );

    this.reconnectTimer = setTimeout(() => {
      console.log(`Reconnection attempt ${this.config.reconnectAttempts + 1}/${this.config.maxRetries}`);
      this.config.reconnectAttempts++;
      this.status.reconnectAttempts = this.config.reconnectAttempts;
      this.connect();
    }, delay);
  }

  /**
   * Subscribe to data updates
   */
  onData(callback: DataCallback): () => void {
    this.dataCallbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = this.dataCallbacks.indexOf(callback);
      if (index > -1) {
        this.dataCallbacks.splice(index, 1);
      }
    };
  }

  /**
   * Subscribe to status updates
   */
  onStatusChange(callback: StatusCallback): () => void {
    this.statusCallbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = this.statusCallbacks.indexOf(callback);
      if (index > -1) {
        this.statusCallbacks.splice(index, 1);
      }
    };
  }

  /**
   * Subscribe to error events
   */
  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = this.errorCallbacks.indexOf(callback);
      if (index > -1) {
        this.errorCallbacks.splice(index, 1);
      }
    };
  }

  /**
   * Notify status change
   */
  private notifyStatusChange(): void {
    this.statusCallbacks.forEach(callback => {
      try {
        callback(this.status);
      } catch (error) {
        console.error('Error in status callback:', error);
      }
    });
  }

  /**
   * Notify error
   */
  private notifyError(error: Error): void {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('Error in error callback:', callbackError);
      }
    });
  }

  /**
   * Get recent data from buffer
   */
  getRecentData(count: number = 100): RealTimeDataPoint[] {
    return this.dataBuffer.slice(-count);
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return { ...this.status };
  }

  /**
   * Check if market is open
   */
  isMarketOpen(): boolean {
    return this.marketOpen;
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.status.connected = false;
    this.status.quality = 'disconnected';
    this.notifyStatusChange();
  }

  /**
   * Force reconnection
   */
  forceReconnect(): void {
    this.disconnect();
    setTimeout(() => this.connect(), 1000);
  }
}

// Create and export singleton instance
export const realTimeWebSocketManager = new RealTimeWebSocketManager();
export default realTimeWebSocketManager;