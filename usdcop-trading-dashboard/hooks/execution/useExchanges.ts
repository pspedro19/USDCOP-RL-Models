/**
 * Exchange Hooks - SignalBridge Integration
 * ==========================================
 *
 * React Query hooks for exchange operations.
 * Uses correct imports from dashboard structure.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { exchangeService } from '@/lib/services/execution/exchange.service';
import { toast } from '@/lib/stores/uiStore';
import {
  type ConnectExchangeRequest,
  type SupportedExchange,
  getExchangeDisplayName,
} from '@/lib/contracts/execution/exchange.contract';

// ============================================================================
// QUERY KEYS (SSOT for cache management)
// ============================================================================

export const exchangeKeys = {
  all: ['exchanges'] as const,
  lists: () => [...exchangeKeys.all, 'list'] as const,
  balances: () => [...exchangeKeys.all, 'balances'] as const,
  balance: (exchange: SupportedExchange) => [...exchangeKeys.balances(), exchange] as const,
};

// ============================================================================
// HOOKS
// ============================================================================

/**
 * Get all connected exchanges
 */
export function useExchanges() {
  return useQuery({
    queryKey: exchangeKeys.lists(),
    queryFn: exchangeService.getExchanges,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Connect a new exchange
 */
export function useConnectExchange() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ exchange, data }: { exchange: SupportedExchange; data: ConnectExchangeRequest }) =>
      exchangeService.connectExchange(exchange, data),
    onSuccess: (result, { exchange }) => {
      const displayName = getExchangeDisplayName(exchange);
      if (result.is_valid) {
        queryClient.invalidateQueries({ queryKey: exchangeKeys.all });
        toast.success(`${displayName} connected successfully!`);
      } else {
        toast.error(result.error_message || `Failed to connect ${displayName}`);
      }
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to connect exchange');
    },
  });
}

/**
 * Disconnect an exchange
 */
export function useDisconnectExchange() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (exchange: SupportedExchange) => exchangeService.disconnectExchange(exchange),
    onSuccess: (_, exchange) => {
      const displayName = getExchangeDisplayName(exchange);
      queryClient.invalidateQueries({ queryKey: exchangeKeys.all });
      toast.success(`${displayName} disconnected`);
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to disconnect exchange');
    },
  });
}

/**
 * Test exchange connection
 */
export function useTestConnection() {
  return useMutation({
    mutationFn: (exchange: SupportedExchange) => exchangeService.testConnection(exchange),
    onSuccess: (result, exchange) => {
      const displayName = getExchangeDisplayName(exchange);
      if (result.is_valid) {
        toast.success(`${displayName} connection is valid`);
      } else {
        toast.error(result.error_message || `${displayName} connection test failed`);
      }
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to test connection');
    },
  });
}

/**
 * Get balances for all connected exchanges
 */
export function useExchangeBalances() {
  return useQuery({
    queryKey: exchangeKeys.balances(),
    queryFn: exchangeService.getAllBalances,
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Get balance for a specific exchange
 */
export function useExchangeBalance(exchange: SupportedExchange) {
  return useQuery({
    queryKey: exchangeKeys.balance(exchange),
    queryFn: () => exchangeService.getBalance(exchange),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000, // 30 seconds
    enabled: !!exchange,
  });
}
