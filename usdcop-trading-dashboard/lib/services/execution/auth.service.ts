/**
 * Auth Service - SignalBridge Integration
 * ========================================
 *
 * Authentication service with Zod validation.
 * Follows dashboard patterns.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { api } from './api';
import { sleep } from '@/lib/utils';
import { MOCK_MODE } from '@/lib/config/execution/constants';
import {
  LoginRequestSchema,
  AuthTokenSchema,
  UserSchema,
  AuthResponseSchema,
  type LoginRequest,
  type RegisterRequest,
  type AuthToken,
  type User,
  type AuthResponse,
} from '@/lib/contracts/execution/auth.contract';

// ============================================================================
// MOCK DATA
// ============================================================================

const mockUser: User = {
  id: '550e8400-e29b-41d4-a716-446655440000',
  email: 'pedro@example.com',
  subscription_tier: 'pro',
  risk_profile: 'moderate',
  max_daily_trades: 20,
  max_position_size_pct: 0.02,
  connected_exchanges: ['mexc', 'binance'],
  is_active: true,
  created_at: '2026-01-01T00:00:00Z',
};

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

function validateData<T>(schema: z.ZodType<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    console.error(`[Auth Service] Validation failed for ${context}:`, result.error.format());
    throw new Error(`Validation failed: ${result.error.issues[0]?.message || 'Unknown error'}`);
  }
  return result.data;
}

// ============================================================================
// SERVICE
// ============================================================================

export const authService = {
  /**
   * Login with email and password
   */
  async login(data: LoginRequest): Promise<AuthResponse> {
    // Validate request
    validateData(LoginRequestSchema, data, 'login request');

    // Audit A8-09: mock auth must NEVER ship a hardcoded bypass credential in the
    // client bundle. Mock mode is dev-only (NODE_ENV guard) and accepts no password —
    // it just returns the mock session for local UI work.
    if (MOCK_MODE && process.env.NODE_ENV !== 'production') {
      await sleep(500);
      return {
        user: { ...mockUser, email: data.email },
        tokens: {
          access_token: 'mock-jwt-token-' + Date.now(),
          token_type: 'bearer',
          expires_at: new Date(Date.now() + 3600000).toISOString(),
          refresh_token: 'mock-refresh-token',
        },
      };
    }

    const response = await api.post<AuthResponse>('/auth/login', data);
    return validateData(AuthResponseSchema, response.data, 'auth response');
  },

  /**
   * Register a new account
   */
  async register(data: Omit<RegisterRequest, 'confirmPassword' | 'acceptTerms'>): Promise<AuthResponse> {
    if (MOCK_MODE) {
      await sleep(500);
      return {
        user: { ...mockUser, email: data.email },
        tokens: {
          access_token: 'mock-jwt-token-' + Date.now(),
          token_type: 'bearer',
          expires_at: new Date(Date.now() + 3600000).toISOString(),
        },
      };
    }

    // Backend route is /auth/register (was incorrectly /auth/signup).
    const response = await api.post<AuthResponse>('/auth/register', data);
    return validateData(AuthResponseSchema, response.data, 'auth response');
  },

  /**
   * Refresh access token
   */
  async refreshToken(refreshToken: string): Promise<AuthToken> {
    if (MOCK_MODE) {
      await sleep(200);
      return {
        access_token: 'new-mock-jwt-token-' + Date.now(),
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 3600000).toISOString(),
      };
    }

    const response = await api.post<AuthToken>('/auth/refresh', { refresh_token: refreshToken });
    return validateData(AuthTokenSchema, response.data, 'auth token');
  },

  /**
   * Get current user profile
   */
  async getMe(): Promise<User> {
    if (MOCK_MODE) {
      await sleep(200);
      return mockUser;
    }

    const response = await api.get<User>('/users/me');
    return validateData(UserSchema, response.data, 'user');
  },

  /**
   * Update user profile
   */
  async updateProfile(data: Partial<User>): Promise<User> {
    if (MOCK_MODE) {
      await sleep(300);
      return { ...mockUser, ...data };
    }

    const response = await api.patch<User>('/users/me', data);
    return validateData(UserSchema, response.data, 'user');
  },

  /**
   * Request password reset
   */
  async forgotPassword(email: string): Promise<{ message: string }> {
    if (MOCK_MODE) {
      await sleep(500);
      return { message: 'Password reset email sent' };
    }

    const response = await api.post<{ message: string }>('/auth/forgot-password', { email });
    return response.data;
  },

  /**
   * Reset password with token
   */
  async resetPassword(token: string, password: string): Promise<{ message: string }> {
    if (MOCK_MODE) {
      await sleep(500);
      return { message: 'Password reset successful' };
    }

    const response = await api.post<{ message: string }>('/auth/reset-password', { token, password });
    return response.data;
  },

  /**
   * Logout current session
   */
  async logout(): Promise<void> {
    if (MOCK_MODE) {
      await sleep(100);
      return;
    }

    await api.post('/auth/logout');
  },
};
