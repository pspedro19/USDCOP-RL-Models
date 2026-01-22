/**
 * Signal Bridge Service - Real Integration
 * =========================================
 *
 * Service for interacting with SignalBridge API.
 * Handles bridge status, kill switch, executions, and WebSocket.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { api } from './api';
import {
  BridgeStatusSchema,
  BridgeHealthCheckSchema,
  BridgeStatisticsSchema,
  RiskCheckResultSchema,
  BridgeExecutionResultSchema,
  UserRiskLimitsSchema,
  UserRiskLimitsUpdateSchema,
  WebSocketMessageSchema,
  validateBridgeStatus,
  validateBridgeHealth,
  validateBridgeExecutionResult,
  type BridgeStatus,
  type BridgeHealthCheck,
  type BridgeStatistics,
  type RiskCheckResult,
  type BridgeExecutionResult,
  type UserRiskLimits,
  type UserRiskLimitsUpdate,
  type ManualSignalCreate,
  type KillSwitchRequest,
  type WebSocketMessage,
  type TradingMode,
  type InferenceAction,
} from '@/lib/contracts/execution/signal-bridge.contract';

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE = '/signal-bridge';
const WS_BASE = process.env.NEXT_PUBLIC_SIGNALBRIDGE_WS_URL || 'ws://localhost:8080/ws/executions';

// ============================================================================
// ERROR HANDLING
// ============================================================================

export class SignalBridgeError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string,
    public code?: string
  ) {
    super(message);
    this.name = 'SignalBridgeError';
  }
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

function validateData<T>(schema: z.ZodType<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);

  if (!result.success) {
    console.error(`[SignalBridge] Validation failed for ${context}:`, result.error.format());
    throw new SignalBridgeError(
      `Validation failed: ${result.error.issues[0]?.message || 'Unknown error'}`,
      undefined,
      undefined,
      'VALIDATION_ERROR'
    );
  }

  return result.data;
}

// ============================================================================
// PAGINATED RESPONSE
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

// ============================================================================
// SERVICE
// ============================================================================

export const signalBridgeService = {
  // ==========================================================================
  // STATUS & HEALTH
  // ==========================================================================

  /**
   * Get current bridge status
   */
  async getStatus(): Promise<BridgeStatus> {
    const { data } = await api.get<BridgeStatus>(`${API_BASE}/status`);
    return validateData(BridgeStatusSchema, data, 'bridge status');
  },

  /**
   * Get detailed health check
   */
  async getHealth(): Promise<BridgeHealthCheck> {
    const { data } = await api.get<BridgeHealthCheck>(`${API_BASE}/health`);
    return validateData(BridgeHealthCheckSchema, data, 'bridge health');
  },

  /**
   * Get bridge statistics for a period
   */
  async getStatistics(days: number = 7): Promise<BridgeStatistics> {
    const { data } = await api.get<BridgeStatistics>(`${API_BASE}/statistics?days=${days}`);
    return validateData(BridgeStatisticsSchema, data, 'bridge statistics');
  },

  // ==========================================================================
  // SIGNAL PROCESSING
  // ==========================================================================

  /**
   * Process a manual trading signal
   */
  async processSignal(signal: ManualSignalCreate): Promise<BridgeExecutionResult> {
    const { data } = await api.post<BridgeExecutionResult>(`${API_BASE}/process`, signal);
    return validateData(BridgeExecutionResultSchema, data, 'execution result');
  },

  /**
   * Validate a signal without executing
   */
  async validateSignal(signal: ManualSignalCreate): Promise<RiskCheckResult> {
    const { data } = await api.post<RiskCheckResult>(`${API_BASE}/validate`, signal);
    return validateData(RiskCheckResultSchema, data, 'risk check result');
  },

  // ==========================================================================
  // KILL SWITCH
  // ==========================================================================

  /**
   * Activate kill switch
   */
  async activateKillSwitch(reason: string): Promise<{ success: boolean; message: string }> {
    const request: KillSwitchRequest = { activate: true, reason, confirm: true };
    const { data } = await api.post<{ success: boolean; message: string }>(`${API_BASE}/kill-switch`, request);
    return data;
  },

  /**
   * Deactivate kill switch
   */
  async deactivateKillSwitch(): Promise<{ success: boolean; message: string }> {
    const request: KillSwitchRequest = { activate: false, reason: 'Manual deactivation', confirm: true };
    const { data } = await api.post<{ success: boolean; message: string }>(`${API_BASE}/kill-switch`, request);
    return data;
  },

  /**
   * Get kill switch status
   */
  async getKillSwitchStatus(): Promise<{ active: boolean; reason: string | null; trading_mode: TradingMode }> {
    const { data } = await api.get<{ active: boolean; reason: string | null; trading_mode: TradingMode }>(`${API_BASE}/kill-switch/status`);
    return data;
  },

  // ==========================================================================
  // HISTORY
  // ==========================================================================

  /**
   * Get execution history with filters
   */
  async getHistory(params: {
    exchange?: string;
    symbol?: string;
    status?: string;
    model_id?: string;
    since?: Date;
    until?: Date;
    page?: number;
    limit?: number;
  } = {}): Promise<PaginatedResponse<BridgeExecutionResult>> {
    const searchParams = new URLSearchParams();

    if (params.exchange) searchParams.set('exchange', params.exchange);
    if (params.symbol) searchParams.set('symbol', params.symbol);
    if (params.status) searchParams.set('status', params.status);
    if (params.model_id) searchParams.set('model_id', params.model_id);
    if (params.since) searchParams.set('since', params.since.toISOString());
    if (params.until) searchParams.set('until', params.until.toISOString());
    if (params.page) searchParams.set('page', params.page.toString());
    if (params.limit) searchParams.set('limit', params.limit.toString());

    const queryString = searchParams.toString();
    const url = queryString ? `${API_BASE}/history?${queryString}` : `${API_BASE}/history`;

    const { data } = await api.get<PaginatedResponse<BridgeExecutionResult>>(url);

    // Validate items
    const validatedItems = z.array(BridgeExecutionResultSchema).parse(data.items);

    return {
      ...data,
      items: validatedItems,
    };
  },

  // ==========================================================================
  // USER RISK LIMITS
  // ==========================================================================

  /**
   * Get current user's risk limits
   */
  async getUserLimits(userId: string): Promise<UserRiskLimits> {
    const { data } = await api.get<UserRiskLimits>(`${API_BASE}/user/${userId}/limits`);
    return validateData(UserRiskLimitsSchema, data, 'user risk limits');
  },

  /**
   * Update user's risk limits
   */
  async updateUserLimits(userId: string, limits: UserRiskLimitsUpdate): Promise<UserRiskLimits> {
    const validatedUpdate = validateData(UserRiskLimitsUpdateSchema, limits, 'limits update');
    const { data } = await api.put<UserRiskLimits>(`${API_BASE}/user/${userId}/limits`, validatedUpdate);
    return validateData(UserRiskLimitsSchema, data, 'updated limits');
  },

  /**
   * Get user trading state
   */
  async getUserState(userId: string): Promise<{
    user_id: string;
    is_trading_allowed: boolean;
    kill_switch_active: boolean;
    daily_blocked: boolean;
    trade_count_today: number;
    daily_pnl_pct: number;
  }> {
    const { data } = await api.get<{
      user_id: string;
      is_trading_allowed: boolean;
      kill_switch_active: boolean;
      daily_blocked: boolean;
      trade_count_today: number;
      daily_pnl_pct: number;
    }>(`${API_BASE}/user/${userId}/state`);
    return data;
  },

  /**
   * Reset user's risk state
   */
  async resetUserState(userId: string): Promise<{ success: boolean; message: string }> {
    const { data } = await api.post<{ success: boolean; message: string }>(`${API_BASE}/user/${userId}/reset`);
    return data;
  },

  // ==========================================================================
  // WEBSOCKET
  // ==========================================================================

  /**
   * Create WebSocket connection to receive real-time updates
   */
  createWebSocket(options: {
    onMessage: (message: WebSocketMessage) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Error) => void;
    userId?: string;
  }): {
    connect: () => void;
    disconnect: () => void;
    send: (message: unknown) => void;
    isConnected: () => boolean;
  } {
    let ws: WebSocket | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 3000;
    let pingInterval: NodeJS.Timeout | null = null;

    const connect = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        return;
      }

      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('auth-token') : null;
        const wsUrl = token ? `${WS_BASE}?token=${token}` : WS_BASE;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log('[SignalBridge WS] Connected');
          reconnectAttempts = 0;
          options.onConnect?.();

          // Start ping interval
          pingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'ping' }));
            }
          }, 30000);

          // Subscribe to user-specific events if userId provided
          if (options.userId && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              type: 'subscribe',
              data: { user_id: options.userId },
            }));
          }
        };

        ws.onmessage = (event) => {
          try {
            const rawData = JSON.parse(event.data);
            const validated = WebSocketMessageSchema.safeParse(rawData);

            if (validated.success) {
              options.onMessage(validated.data);
            } else {
              console.warn('[SignalBridge WS] Invalid message format:', rawData);
            }
          } catch (e) {
            console.error('[SignalBridge WS] Failed to parse message:', e);
          }
        };

        ws.onclose = (event) => {
          console.log('[SignalBridge WS] Disconnected:', event.code, event.reason);

          if (pingInterval) {
            clearInterval(pingInterval);
            pingInterval = null;
          }

          options.onDisconnect?.();

          // Auto-reconnect
          if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`[SignalBridge WS] Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`);
            setTimeout(connect, reconnectDelay * reconnectAttempts);
          }
        };

        ws.onerror = (error) => {
          console.error('[SignalBridge WS] Error:', error);
          options.onError?.(new Error('WebSocket connection error'));
        };
      } catch (error) {
        console.error('[SignalBridge WS] Failed to create connection:', error);
        options.onError?.(error instanceof Error ? error : new Error('Unknown error'));
      }
    };

    const disconnect = () => {
      reconnectAttempts = maxReconnectAttempts; // Prevent auto-reconnect

      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }

      if (ws) {
        ws.close(1000, 'Client disconnect');
        ws = null;
      }
    };

    const send = (message: unknown) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      } else {
        console.warn('[SignalBridge WS] Cannot send message, not connected');
      }
    };

    const isConnected = () => {
      return ws !== null && ws.readyState === WebSocket.OPEN;
    };

    return { connect, disconnect, send, isConnected };
  },
};

// Export as default
export default signalBridgeService;
