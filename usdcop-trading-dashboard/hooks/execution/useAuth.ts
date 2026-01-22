/**
 * Auth Hook - SignalBridge Integration
 * =====================================
 *
 * React hook for authentication operations.
 * Uses Next.js router for navigation.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

'use client';

import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/stores/authStore';
import { toast } from '@/lib/stores/uiStore';
import { EXECUTION_ROUTES } from '@/lib/config/execution/constants';

export function useAuth() {
  const router = useRouter();
  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    login: storeLogin,
    register: storeRegister,
    logout: storeLogout,
    clearError,
  } = useAuthStore();

  const login = async (email: string, password: string) => {
    try {
      await storeLogin(email, password);
      toast.success('Welcome back!');
      router.push(EXECUTION_ROUTES.DASHBOARD);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed';
      toast.error(message);
      throw err;
    }
  };

  const register = async (email: string, password: string) => {
    try {
      await storeRegister(email, password);
      toast.success('Account created successfully!');
      router.push(EXECUTION_ROUTES.DASHBOARD);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Registration failed';
      toast.error(message);
      throw err;
    }
  };

  const logout = () => {
    storeLogout();
    toast.info('You have been logged out');
    router.push(EXECUTION_ROUTES.LOGIN);
  };

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    register,
    logout,
    clearError,
  };
}
