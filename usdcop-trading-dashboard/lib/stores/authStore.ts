/**
 * Auth Store - SignalBridge Integration
 * ======================================
 *
 * Zustand store for authentication state.
 * Uses correct imports from dashboard structure.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authService } from '@/lib/services/execution/auth.service';
import type { User, AuthToken } from '@/lib/contracts/execution/auth.contract';

// ============================================================================
// TYPES
// ============================================================================

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshAuth: () => Promise<void>;
  updateUser: (data: Partial<User>) => void;
  clearError: () => void;
}

// ============================================================================
// STORE
// ============================================================================

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const { user, tokens } = await authService.login({ email, password });
          localStorage.setItem('auth-token', tokens.access_token);
          set({
            user,
            token: tokens.access_token,
            refreshToken: tokens.refresh_token || null,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : 'Login failed';
          set({
            error: message,
            isLoading: false,
          });
          throw error;
        }
      },

      register: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const { user, tokens } = await authService.register({ email, password });
          localStorage.setItem('auth-token', tokens.access_token);
          set({
            user,
            token: tokens.access_token,
            refreshToken: tokens.refresh_token || null,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : 'Registration failed';
          set({
            error: message,
            isLoading: false,
          });
          throw error;
        }
      },

      logout: () => {
        localStorage.removeItem('auth-token');
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        });
      },

      refreshAuth: async () => {
        const { refreshToken } = get();
        if (!refreshToken) {
          get().logout();
          return;
        }

        try {
          const tokens = await authService.refreshToken(refreshToken);
          localStorage.setItem('auth-token', tokens.access_token);
          set({
            token: tokens.access_token,
            refreshToken: tokens.refresh_token || null,
          });
        } catch {
          get().logout();
        }
      },

      updateUser: (data: Partial<User>) => {
        const { user } = get();
        if (user) {
          set({ user: { ...user, ...data } });
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
