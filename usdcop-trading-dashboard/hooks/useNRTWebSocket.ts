/**
 * NRT (Near Real-Time) WebSocket Hook
 * ====================================
 *
 * React hook for receiving real-time inference signals from L5.
 * Handles WebSocket connection, reconnection, and message processing.
 *
 * Contract: CTR-NRT-WS-001
 * Version: 1.0.0
 * Created: 2026-02-04
 *
 * Usage:
 *   const { signals, latestSignal, connected } = useNRTWebSocket();
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// TYPES
// ============================================================================

/** Trading signal from L5 inference */
export interface NRTSignal {
  type: 'nrt_inference';
  model_id: string;
  timestamp: string;
  signal: 'LONG' | 'SHORT' | 'HOLD';
  raw_action: number;
  confidence: number;
  price: number;
  position: number;
  unrealized_pnl: number;
  latency_ms: number;
}

/** Connection status */
export interface NRTConnectionStatus {
  connected: boolean;
  reconnecting: boolean;
  reconnectAttempts: number;
  lastMessageTime: Date | null;
  error: string | null;
}

/** Hook options */
export interface UseNRTWebSocketOptions {
  /** WebSocket URL (defaults to env var or /api/ws/nrt) */
  url?: string;
  /** Auto-connect on mount */
  autoConnect?: boolean;
  /** Maximum signals to keep in history */
  maxSignals?: number;
  /** Callback when signal received */
  onSignal?: (signal: NRTSignal) => void;
  /** Callback on connection change */
  onConnectionChange?: (connected: boolean) => void;
  /** Callback on error */
  onError?: (error: string) => void;
}

/** Hook return value */
export interface UseNRTWebSocketResult {
  /** All received signals (most recent first) */
  signals: NRTSignal[];
  /** Latest signal received */
  latestSignal: NRTSignal | null;
  /** Connection status */
  status: NRTConnectionStatus;
  /** Whether WebSocket is connected */
  connected: boolean;
  /** Connect to WebSocket */
  connect: () => void;
  /** Disconnect from WebSocket */
  disconnect: () => void;
  /** Clear signal history */
  clearSignals: () => void;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const DEFAULT_MAX_SIGNALS = 100;
const RECONNECT_INTERVAL_MS = 5000;
const MAX_RECONNECT_ATTEMPTS = 10;

// ============================================================================
// HOOK
// ============================================================================

export function useNRTWebSocket(
  options: UseNRTWebSocketOptions = {}
): UseNRTWebSocketResult {
  const {
    url = process.env.NEXT_PUBLIC_NRT_WS_URL || '/api/ws/nrt',
    autoConnect = true,
    maxSignals = DEFAULT_MAX_SIGNALS,
    onSignal,
    onConnectionChange,
    onError,
  } = options;

  // State
  const [signals, setSignals] = useState<NRTSignal[]>([]);
  const [latestSignal, setLatestSignal] = useState<NRTSignal | null>(null);
  const [status, setStatus] = useState<NRTConnectionStatus>({
    connected: false,
    reconnecting: false,
    reconnectAttempts: 0,
    lastMessageTime: null,
    error: null,
  });

  // Refs for WebSocket and reconnection
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  // Build full WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    // If URL is relative, build absolute URL
    if (url.startsWith('/')) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}${url}`;
    }
    // If URL doesn't have protocol, add ws://
    if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
      return `ws://${url}`;
    }
    return url;
  }, [url]);

  // Process incoming message
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);

      // Validate it's an NRT signal
      if (data.type !== 'nrt_inference') {
        return;
      }

      const signal = data as NRTSignal;

      // Update state
      setLatestSignal(signal);
      setSignals(prev => [signal, ...prev].slice(0, maxSignals));
      setStatus(prev => ({
        ...prev,
        lastMessageTime: new Date(),
        error: null,
      }));

      // Callback
      onSignal?.(signal);

    } catch (error) {
      console.error('[NRT WebSocket] Failed to parse message:', error);
    }
  }, [maxSignals, onSignal]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    // Don't connect if already connected or not mounted
    if (wsRef.current?.readyState === WebSocket.OPEN || !mountedRef.current) {
      return;
    }

    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      const wsUrl = getWebSocketUrl();
      console.log('[NRT WebSocket] Connecting to:', wsUrl);

      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('[NRT WebSocket] Connected');
        if (mountedRef.current) {
          setStatus(prev => ({
            ...prev,
            connected: true,
            reconnecting: false,
            reconnectAttempts: 0,
            error: null,
          }));
          onConnectionChange?.(true);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('[NRT WebSocket] Disconnected:', event.code, event.reason);
        if (mountedRef.current) {
          setStatus(prev => ({
            ...prev,
            connected: false,
          }));
          onConnectionChange?.(false);

          // Attempt reconnection if not a normal close
          if (event.code !== 1000) {
            scheduleReconnect();
          }
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('[NRT WebSocket] Error:', error);
        if (mountedRef.current) {
          const errorMsg = 'WebSocket connection error';
          setStatus(prev => ({
            ...prev,
            error: errorMsg,
          }));
          onError?.(errorMsg);
        }
      };

      wsRef.current.onmessage = handleMessage;

    } catch (error) {
      console.error('[NRT WebSocket] Failed to create connection:', error);
      if (mountedRef.current) {
        const errorMsg = error instanceof Error ? error.message : 'Connection failed';
        setStatus(prev => ({
          ...prev,
          error: errorMsg,
        }));
        onError?.(errorMsg);
        scheduleReconnect();
      }
    }
  }, [getWebSocketUrl, handleMessage, onConnectionChange, onError]);

  // Schedule reconnection attempt
  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;

    setStatus(prev => {
      if (prev.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        return {
          ...prev,
          reconnecting: false,
          error: 'Max reconnection attempts reached',
        };
      }

      return {
        ...prev,
        reconnecting: true,
        reconnectAttempts: prev.reconnectAttempts + 1,
      };
    });

    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, RECONNECT_INTERVAL_MS);
  }, [connect]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect');
      wsRef.current = null;
    }

    setStatus(prev => ({
      ...prev,
      connected: false,
      reconnecting: false,
      reconnectAttempts: 0,
    }));
  }, []);

  // Clear signal history
  const clearSignals = useCallback(() => {
    setSignals([]);
    setLatestSignal(null);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    mountedRef.current = true;

    if (autoConnect) {
      connect();
    }

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    signals,
    latestSignal,
    status,
    connected: status.connected,
    connect,
    disconnect,
    clearSignals,
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get color class for signal type
 */
export function getSignalColorClass(signal: 'LONG' | 'SHORT' | 'HOLD'): string {
  switch (signal) {
    case 'LONG':
      return 'text-green-400';
    case 'SHORT':
      return 'text-red-400';
    case 'HOLD':
    default:
      return 'text-gray-400';
  }
}

/**
 * Get background color class for signal type
 */
export function getSignalBgClass(signal: 'LONG' | 'SHORT' | 'HOLD'): string {
  switch (signal) {
    case 'LONG':
      return 'bg-green-500/20 border-green-500/30';
    case 'SHORT':
      return 'bg-red-500/20 border-red-500/30';
    case 'HOLD':
    default:
      return 'bg-gray-500/20 border-gray-500/30';
  }
}

/**
 * Format latency for display
 */
export function formatLatency(ms: number): string {
  if (ms < 1) {
    return '<1ms';
  }
  return `${ms.toFixed(1)}ms`;
}

/**
 * Format confidence as percentage
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(0)}%`;
}

export default useNRTWebSocket;
