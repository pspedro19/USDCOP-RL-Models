'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  History,
  RefreshCw,
  Filter,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Search,
} from 'lucide-react';
import { signalBridgeService, type PaginatedResponse } from '@/lib/services/execution/signal-bridge.service';
import {
  type BridgeExecutionResult,
  INFERENCE_ACTION_LABELS,
  INFERENCE_ACTION_COLORS,
} from '@/lib/contracts/execution/signal-bridge.contract';
import {
  ORDER_STATUS_COLORS,
  ORDER_STATUS_NAMES,
  type OrderStatus,
} from '@/lib/contracts/execution/execution.contract';

export default function ExecutionsPage() {
  const [executions, setExecutions] = useState<BridgeExecutionResult[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit] = useState(20);
  const [hasMore, setHasMore] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [exchangeFilter, setExchangeFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  const fetchExecutions = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await signalBridgeService.getHistory({
        page,
        limit,
        status: statusFilter || undefined,
        exchange: exchangeFilter || undefined,
      });

      setExecutions(response.items);
      setTotal(response.total);
      setHasMore(response.has_more);
    } catch (err) {
      console.error('Failed to fetch executions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load executions');
    } finally {
      setIsLoading(false);
    }
  }, [page, limit, statusFilter, exchangeFilter]);

  useEffect(() => {
    fetchExecutions();
  }, [fetchExecutions]);

  const totalPages = Math.ceil(total / limit);

  const getStatusIcon = (status: OrderStatus) => {
    switch (status) {
      case 'filled':
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case 'failed':
      case 'rejected':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'pending':
      case 'submitted':
        return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'cancelled':
        return <AlertTriangle className="w-4 h-4 text-gray-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Execution History</h1>
          <p className="text-gray-400">View all trading executions and their status</p>
        </div>
        <button
          onClick={fetchExecutions}
          className="p-2 text-gray-400 hover:text-white transition-colors"
        >
          <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Filters */}
      <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-4">
        <div className="flex flex-wrap gap-4 items-center">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search by symbol or order ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
            />
          </div>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
            className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="">All Status</option>
            <option value="filled">Filled</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>

          {/* Exchange Filter */}
          <select
            value={exchangeFilter}
            onChange={(e) => {
              setExchangeFilter(e.target.value);
              setPage(1);
            }}
            className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="">All Exchanges</option>
            <option value="binance">Binance</option>
            <option value="mexc">MEXC</option>
          </select>

          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Filter className="w-4 h-4" />
            {total} executions
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading && executions.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin" />
        </div>
      ) : executions.length === 0 ? (
        /* Empty State */
        <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-12 text-center">
          <History className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-300 mb-2">No Executions Found</h3>
          <p className="text-gray-500">
            {statusFilter || exchangeFilter
              ? 'Try adjusting your filters'
              : 'Executions will appear here once trades are made'}
          </p>
        </div>
      ) : (
        /* Executions Table */
        <>
          <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800/50">
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Exchange</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Symbol</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Side</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">Quantity</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">Price</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">Exec Time</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/50">
                  {executions.map((execution, idx) => (
                    <motion.tr
                      key={execution.execution_id || idx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.02 }}
                      className="hover:bg-gray-800/30 transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(execution.status)}
                          <span className={ORDER_STATUS_COLORS[execution.status]?.split(' ')[0] || 'text-gray-400'}>
                            {ORDER_STATUS_NAMES[execution.status] || execution.status}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-white capitalize">{execution.exchange || '-'}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-white font-medium">{execution.symbol || 'USD/COP'}</span>
                      </td>
                      <td className="px-4 py-3">
                        {execution.side && (
                          <span className={`flex items-center gap-1 ${execution.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                            {execution.side === 'buy' ? (
                              <TrendingUp className="w-4 h-4" />
                            ) : (
                              <TrendingDown className="w-4 h-4" />
                            )}
                            {execution.side.toUpperCase()}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-white">
                          {execution.filled_quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                        </span>
                        {execution.requested_quantity > execution.filled_quantity && (
                          <span className="text-gray-500 text-xs ml-1">
                            /{execution.requested_quantity.toLocaleString()}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-white">
                          {execution.filled_price
                            ? `$${execution.filled_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                            : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-gray-400">
                          {execution.processing_time_ms
                            ? `${execution.processing_time_ms.toFixed(0)}ms`
                            : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-gray-400 text-sm">
                          {execution.created_at
                            ? new Date(execution.created_at).toLocaleString()
                            : '-'}
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">
                Showing {(page - 1) * limit + 1}-{Math.min(page * limit, total)} of {total}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="p-2 bg-gray-800 rounded-lg text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <span className="px-4 py-2 text-white">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={!hasMore}
                  className="p-2 bg-gray-800 rounded-lg text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
