/**
 * Exchange Service - SignalBridge Integration
 * ============================================
 *
 * Service for managing exchange connections with Zod validation.
 * Follows dashboard API client patterns.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { sleep } from '@/lib/utils';
import {
  ConnectedExchangeSchema,
  ConnectExchangeRequestSchema,
  ValidationResultSchema,
  ExchangeBalancesSchema,
  type ConnectedExchange,
  type ConnectExchangeRequest,
  type ValidationResult,
  type ExchangeBalances,
  type SupportedExchange,
} from '@/lib/contracts/execution/exchange.contract';

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE = process.env.NEXT_PUBLIC_SIGNALBRIDGE_API_URL || '/api/execution';
const MOCK_MODE = process.env.NEXT_PUBLIC_MOCK_MODE === 'true';
const DEFAULT_TIMEOUT = 30000;

// ============================================================================
// ERROR HANDLING
// ============================================================================

class ExchangeServiceError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = 'ExchangeServiceError';
  }
}

// ============================================================================
// FETCH WRAPPER (follows dashboard pattern)
// ============================================================================

async function fetchWithTimeout<T>(
  url: string,
  options: RequestInit = {},
  timeout: number = DEFAULT_TIMEOUT
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ExchangeServiceError(
        errorData.error || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        url
      );
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof ExchangeServiceError) {
      throw error;
    }

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new ExchangeServiceError(`Request timeout after ${timeout}ms`, undefined, url);
      }
      throw new ExchangeServiceError(error.message, undefined, url);
    }

    throw new ExchangeServiceError('Unknown error occurred', undefined, url);
  }
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

function validateData<T>(schema: z.ZodType<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);

  if (!result.success) {
    console.error(`[Exchange Service] Validation failed for ${context}:`, result.error.format());
    throw new ExchangeServiceError(
      `Validation failed: ${result.error.issues[0]?.message || 'Unknown error'}`
    );
  }

  return result.data;
}

// ============================================================================
// MOCK DATA
// ============================================================================

const mockExchanges: ConnectedExchange[] = [
  {
    id: '550e8400-e29b-41d4-a716-446655440001',
    exchange: 'mexc',
    is_valid: true,
    connected_at: '2026-01-10T00:00:00Z',
    last_used_at: '2026-01-18T14:30:00Z',
    key_fingerprint: 'mx0v...3f2a',
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440002',
    exchange: 'binance',
    is_valid: true,
    connected_at: '2026-01-12T00:00:00Z',
    last_used_at: '2026-01-18T14:30:00Z',
    key_fingerprint: 'bnc1...8x9z',
  },
];

const mockBalances: Record<SupportedExchange, ExchangeBalances> = {
  mexc: {
    exchange: 'mexc',
    balances: [
      { asset: 'USDT', free: 1250.50, locked: 0, total: 1250.50 },
      { asset: 'BTC', free: 0.005, locked: 0, total: 0.005 },
    ],
    total_usd: 1450.50,
    updated_at: new Date().toISOString(),
  },
  binance: {
    exchange: 'binance',
    balances: [
      { asset: 'USDT', free: 1200.00, locked: 50, total: 1250.00 },
    ],
    total_usd: 1250.00,
    updated_at: new Date().toISOString(),
  },
};

// ============================================================================
// SERVICE
// ============================================================================

export const exchangeService = {
  /**
   * Get all connected exchanges
   */
  async getExchanges(): Promise<ConnectedExchange[]> {
    if (MOCK_MODE) {
      await sleep(300);
      return mockExchanges;
    }

    const response = await fetchWithTimeout<{ data: unknown[] }>(`${API_BASE}/exchanges`);
    return z.array(ConnectedExchangeSchema).parse(response.data);
  },

  /**
   * Connect a new exchange
   */
  async connectExchange(
    exchange: SupportedExchange,
    data: ConnectExchangeRequest
  ): Promise<ValidationResult> {
    // Validate request first
    const validatedRequest = validateData(ConnectExchangeRequestSchema, data, 'connect request');

    if (MOCK_MODE) {
      await sleep(1500); // Simulate API validation time

      // Simulate validation - reject if key contains 'withdraw'
      if (validatedRequest.api_key.toLowerCase().includes('withdraw')) {
        return {
          is_valid: false,
          exchange,
          permissions: ['spot', 'withdraw'],
          can_trade_spot: true,
          has_withdraw_permission: true,
          error_message: 'API key has WITHDRAW permission. Please create a new key without withdrawal access for security.',
        };
      }

      // Simulate invalid key
      if (validatedRequest.api_key.length < 20) {
        return {
          is_valid: false,
          exchange,
          permissions: [],
          can_trade_spot: false,
          has_withdraw_permission: false,
          error_message: 'Invalid API key format. Please check your credentials.',
        };
      }

      return {
        is_valid: true,
        exchange,
        permissions: ['spot', 'read'],
        can_trade_spot: true,
        has_withdraw_permission: false,
        balance_check: { USDT: 1250.50 },
      };
    }

    const response = await fetchWithTimeout<{ data: unknown }>(
      `${API_BASE}/exchanges/${exchange}/connect`,
      {
        method: 'POST',
        body: JSON.stringify(validatedRequest),
      }
    );

    return validateData(ValidationResultSchema, response.data, 'validation result');
  },

  /**
   * Disconnect an exchange
   */
  async disconnectExchange(exchange: SupportedExchange): Promise<void> {
    if (MOCK_MODE) {
      await sleep(300);
      const idx = mockExchanges.findIndex(e => e.exchange === exchange);
      if (idx !== -1) mockExchanges.splice(idx, 1);
      return;
    }

    await fetchWithTimeout<void>(`${API_BASE}/exchanges/${exchange}/disconnect`, {
      method: 'DELETE',
    });
  },

  /**
   * Test exchange connection
   */
  async testConnection(exchange: SupportedExchange): Promise<ValidationResult> {
    if (MOCK_MODE) {
      await sleep(800);
      return {
        is_valid: true,
        exchange,
        permissions: ['spot', 'read'],
        can_trade_spot: true,
        has_withdraw_permission: false,
        balance_check: { USDT: mockBalances[exchange]?.total_usd || 0 },
      };
    }

    const response = await fetchWithTimeout<{ data: unknown }>(
      `${API_BASE}/exchanges/${exchange}/validate`,
      { method: 'POST' }
    );

    return validateData(ValidationResultSchema, response.data, 'validation result');
  },

  /**
   * Get balances for a specific exchange
   */
  async getBalance(exchange: SupportedExchange): Promise<ExchangeBalances> {
    if (MOCK_MODE) {
      await sleep(300);
      return mockBalances[exchange] || {
        exchange,
        balances: [],
        total_usd: 0,
        updated_at: new Date().toISOString(),
      };
    }

    const response = await fetchWithTimeout<{ data: unknown }>(
      `${API_BASE}/exchanges/${exchange}/balance`
    );

    return validateData(ExchangeBalancesSchema, response.data, 'exchange balances');
  },

  /**
   * Get balances for all connected exchanges
   */
  async getAllBalances(): Promise<ExchangeBalances[]> {
    if (MOCK_MODE) {
      await sleep(400);
      return Object.values(mockBalances);
    }

    const response = await fetchWithTimeout<{ data: unknown[] }>(
      `${API_BASE}/exchanges/balances`
    );

    return z.array(ExchangeBalancesSchema).parse(response.data);
  },
};

// Export error class for external handling
export { ExchangeServiceError };
