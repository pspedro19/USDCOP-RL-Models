/**
 * SignalBridge WebSocket Hook
 * ===========================
 *
 * React hook for real-time SignalBridge updates.
 * Handles connection, reconnection, and message processing.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { signalBridgeService } from '@/lib/services/execution';
import {
  type WebSocketMessage,
  type BridgeExecutionResult,
  type TradingMode,
  validateWebSocketMessage,
} from '@/lib/contracts/execution/signal-bridge.contract';

// ============================================================================
// TYPES
// ============================================================================

interface UseSignalBridgeWebSocketOptions {
  /** User ID for subscribing to user-specific events */
  userId?: string;
  /** Auto-connect on mount */
  autoConnect?: boolean;
  /** Callback for execution events */
  onExecution?: (execution: BridgeExecutionResult) => void;
  /** Callback for kill switch events */
  onKillSwitch?: (active: boolean, reason?: string) => void;
  /** Callback for trading mode changes */
  onTradingModeChange?: (mode: TradingMode) => void;
  /** Callback for risk alerts */
  onRiskAlert?: (alert: { message: string; severity: string }) => void;
}

interface UseSignalBridgeWebSocketResult {
  /** Whether WebSocket is connected */
  isConnected: boolean;
  /** Latest execution event */
  latestExecution: BridgeExecutionResult | null;
  /** Recent messages (last 50) */
  recentMessages: WebSocketMessage[];
  /** Connect to WebSocket */
  connect: () => void;
  /** Disconnect from WebSocket */
  disconnect: () => void;
  /** Send a message */
  sendMessage: (message: unknown) => void;
  /** Connection error */
  error: string | null;
}

// ============================================================================
// HOOK
// ============================================================================

export function useSignalBridgeWebSocket(
  options: UseSignalBridgeWebSocketOptions = {}
): UseSignalBridgeWebSocketResult {
  const {
    userId,
    autoConnect = true,
    onExecution,
    onKillSwitch,
    onTradingModeChange,
    onRiskAlert,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestExecution, setLatestExecution] = useState<BridgeExecutionResult | null>(null);
  const [recentMessages, setRecentMessages] = useState<WebSocketMessage[]>([]);

  // WebSocket connection reference
  const wsRef = useRef<{
    connect: () => void;
    disconnect: () => void;
    send: (message: unknown) => void;
    isConnected: () => boolean;
  } | null>(null);

  // Process incoming WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    // Add to recent messages (keep last 50)
    setRecentMessages(prev => [message, ...prev].slice(0, 50));

    // Process message by type
    switch (message.type) {
      case 'connected':
        console.log('[SignalBridge WS] Connected');
        break;

      case 'execution_created':
      case 'execution_updated':
      case 'execution_filled':
      case 'execution_failed': {
        const execution = message.data as unknown as BridgeExecutionResult;
        setLatestExecution(execution);
        onExecution?.(execution);
        break;
      }

      case 'kill_switch': {
        const { active, reason } = message.data as { active: boolean; reason?: string };
        onKillSwitch?.(active, reason);
        break;
      }

      case 'trading_mode_changed': {
        const { mode } = message.data as { mode: TradingMode };
        onTradingModeChange?.(mode);
        break;
      }

      case 'risk_alert': {
        const alert = message.data as { message: string; severity: string };
        onRiskAlert?.(alert);
        break;
      }

      case 'heartbeat':
      case 'pong':
        // Heartbeat - no action needed
        break;

      case 'error': {
        const { message: errorMsg } = message.data as { message: string };
        setError(errorMsg);
        break;
      }

      default:
        console.log('[SignalBridge WS] Unknown message type:', message.type);
    }
  }, [onExecution, onKillSwitch, onTradingModeChange, onRiskAlert]);

  // Initialize WebSocket connection
  useEffect(() => {
    if (wsRef.current) {
      return; // Already initialized
    }

    wsRef.current = signalBridgeService.createWebSocket({
      userId,
      onMessage: handleMessage,
      onConnect: () => {
        setIsConnected(true);
        setError(null);
      },
      onDisconnect: () => {
        setIsConnected(false);
      },
      onError: (err) => {
        setError(err.message);
      },
    });

    if (autoConnect) {
      wsRef.current.connect();
    }

    return () => {
      wsRef.current?.disconnect();
      wsRef.current = null;
    };
  }, [userId, autoConnect, handleMessage]);

  // Public methods
  const connect = useCallback(() => {
    wsRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
  }, []);

  const sendMessage = useCallback((message: unknown) => {
    wsRef.current?.send(message);
  }, []);

  return {
    isConnected,
    latestExecution,
    recentMessages,
    connect,
    disconnect,
    sendMessage,
    error,
  };
}

// ============================================================================
// SINGLETON MANAGER (for sharing across components)
// ============================================================================

type MessageCallback = (message: WebSocketMessage) => void;

class SignalBridgeWebSocketManager {
  private static instance: SignalBridgeWebSocketManager | null = null;
  private ws: ReturnType<typeof signalBridgeService.createWebSocket> | null = null;
  private listeners: Set<MessageCallback> = new Set();
  private isConnectedState = false;

  static getInstance(): SignalBridgeWebSocketManager {
    if (!SignalBridgeWebSocketManager.instance) {
      SignalBridgeWebSocketManager.instance = new SignalBridgeWebSocketManager();
    }
    return SignalBridgeWebSocketManager.instance;
  }

  connect(userId?: string) {
    if (this.ws) {
      return;
    }

    this.ws = signalBridgeService.createWebSocket({
      userId,
      onMessage: (message) => {
        this.listeners.forEach(callback => callback(message));
      },
      onConnect: () => {
        this.isConnectedState = true;
      },
      onDisconnect: () => {
        this.isConnectedState = false;
      },
    });

    this.ws.connect();
  }

  disconnect() {
    this.ws?.disconnect();
    this.ws = null;
    this.isConnectedState = false;
  }

  subscribe(callback: MessageCallback): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  send(message: unknown) {
    this.ws?.send(message);
  }

  get isConnected(): boolean {
    return this.isConnectedState;
  }
}

export { SignalBridgeWebSocketManager };
export default useSignalBridgeWebSocket;
