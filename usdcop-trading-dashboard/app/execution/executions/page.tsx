'use client';

/**
 * /execution/executions — historial de ejecuciones, re-skin GM (prototipo Var B:
 * tabla "Ejecuciones recientes" con cifras en mono). Presentación + i18n
 * ÚNICAMENTE: paginación, filtros y fetch (signalBridgeService) quedan intactos.
 */
import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
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
import { GmPageHeader } from '@/components/gm';
import { GM, GMT, type GmTone, GM_TONE_TEXT } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { signalBridgeService } from '@/lib/services/execution/signal-bridge.service';
import { type BridgeExecutionResult } from '@/lib/contracts/execution/signal-bridge.contract';
import { type OrderStatus } from '@/lib/contracts/execution/execution.contract';
import { EXEC_DICT } from './../i18n';

/** Tono GM por estado de orden (presentacional; el enum viene del contrato). */
const STATUS_TONE: Record<OrderStatus, GmTone> = {
  pending: 'warn',
  submitted: 'warn',
  partial: 'info',
  filled: 'pos',
  cancelled: 'neutral',
  rejected: 'neg',
  failed: 'neg',
};

export default function ExecutionsPage() {
  const t = useGmT(EXEC_DICT);
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
      setError(err instanceof Error ? err.message : t('loadError'));
    } finally {
      setIsLoading(false);
    }
  }, [page, limit, statusFilter, exchangeFilter]);

  useEffect(() => {
    fetchExecutions();
  }, [fetchExecutions]);

  const totalPages = Math.ceil(total / limit);

  /** Nombre ES/EN por estado de orden (i18n local; el enum del contrato no cambia). */
  const statusName: Record<OrderStatus, string> = {
    pending: t('stPending'),
    submitted: t('stSubmitted'),
    partial: t('stPartial'),
    filled: t('stFilled'),
    cancelled: t('stCancelled'),
    rejected: t('stRejected'),
    failed: t('stFailed'),
  };

  const getStatusIcon = (status: OrderStatus) => {
    switch (status) {
      case 'filled':
        return <CheckCircle2 className={`w-4 h-4 ${GM.pos}`} aria-hidden />;
      case 'failed':
      case 'rejected':
        return <XCircle className={`w-4 h-4 ${GM.neg}`} aria-hidden />;
      case 'pending':
      case 'submitted':
        return <Clock className={`w-4 h-4 ${GM.warn}`} aria-hidden />;
      case 'cancelled':
        return <AlertTriangle className={`w-4 h-4 ${GM.textMuted}`} aria-hidden />;
      default:
        return <Clock className={`w-4 h-4 ${GM.textMuted}`} aria-hidden />;
    }
  };

  const selectClass = `${GM.input} ${GM.focus} h-11 px-4`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <GmPageHeader
        kicker={t('kicker')}
        title={t('execTitle')}
        subtitle={t('execSub')}
        actions={
          <button
            onClick={fetchExecutions}
            aria-label={t('refresh')}
            className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center`}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'motion-safe:animate-spin' : ''}`} aria-hidden />
          </button>
        }
      />

      {/* Filtros */}
      <div className={`${GM.panel} p-4`}>
        <div className="flex flex-wrap gap-4 items-center">
          {/* Búsqueda */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${GM.textMuted}`} aria-hidden />
            <input
              type="text"
              placeholder={t('searchPh')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`${GM.input} ${GM.focus} w-full h-11 pl-10 pr-4`}
            />
          </div>

          {/* Filtro de estado */}
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
            aria-label={t('thStatus')}
            className={selectClass}
          >
            <option value="">{t('allStatus')}</option>
            <option value="filled">{t('fFilled')}</option>
            <option value="pending">{t('fPending')}</option>
            <option value="failed">{t('fFailed')}</option>
            <option value="cancelled">{t('fCancelled')}</option>
          </select>

          {/* Filtro de exchange */}
          <select
            value={exchangeFilter}
            onChange={(e) => {
              setExchangeFilter(e.target.value);
              setPage(1);
            }}
            aria-label={t('thExchange')}
            className={selectClass}
          >
            <option value="">{t('allExchanges')}</option>
            <option value="binance">Binance</option>
            <option value="mexc">MEXC</option>
          </select>

          <div className={`flex items-center gap-2 ${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
            <Filter className="w-4 h-4" aria-hidden />
            {total} {t('countSuffix')}
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className={`${GM.negBadge} rounded-xl p-4 flex items-center gap-3`} role="alert">
          <AlertTriangle className={`w-5 h-5 ${GM.neg}`} aria-hidden />
          <span className={`${GMT.body} ${GM.neg}`}>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading && executions.length === 0 ? (
        <div className="flex items-center justify-center py-12" aria-busy>
          <RefreshCw className={`w-8 h-8 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
        </div>
      ) : executions.length === 0 ? (
        /* Empty State */
        <div className={`${GM.panel} p-12 text-center`}>
          <History className={`w-12 h-12 ${GM.textFaint} mx-auto mb-4`} aria-hidden />
          <h3 className={`text-lg font-medium ${GM.textStrong} mb-2`}>{t('noExecTitle')}</h3>
          <p className={`${GMT.body} ${GM.textMuted}`}>
            {statusFilter || exchangeFilter ? t('noExecFilter') : t('noExecBody')}
          </p>
        </div>
      ) : (
        /* Tabla de ejecuciones */
        <>
          <div className={`${GM.panel} overflow-hidden`}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[var(--gm-border)]">
                    <th className={`px-4 py-3 text-left ${GMT.label} ${GM.textMuted}`}>{t('thStatus')}</th>
                    <th className={`px-4 py-3 text-left ${GMT.label} ${GM.textMuted}`}>{t('thExchange')}</th>
                    <th className={`px-4 py-3 text-left ${GMT.label} ${GM.textMuted}`}>{t('thSymbol')}</th>
                    <th className={`px-4 py-3 text-left ${GMT.label} ${GM.textMuted}`}>{t('thSide')}</th>
                    <th className={`px-4 py-3 text-right ${GMT.label} ${GM.textMuted}`}>{t('thQty')}</th>
                    <th className={`px-4 py-3 text-right ${GMT.label} ${GM.textMuted}`}>{t('thPrice')}</th>
                    <th className={`px-4 py-3 text-right ${GMT.label} ${GM.textMuted}`}>{t('thPnl')}</th>
                    <th className={`px-4 py-3 text-right ${GMT.label} ${GM.textMuted}`}>{t('thExecTime')}</th>
                    <th className={`px-4 py-3 text-right ${GMT.label} ${GM.textMuted}`}>{t('thDate')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(148,163,184,.07)]">
                  {executions.map((execution, idx) => (
                    <motion.tr
                      key={execution.execution_id || idx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.02 }}
                      className={`${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(execution.status)}
                          <span className={`${GMT.meta} ${GM_TONE_TEXT[STATUS_TONE[execution.status] ?? 'neutral']}`}>
                            {statusName[execution.status] || execution.status}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`${GMT.body} ${GM.text} capitalize`}>{execution.exchange || '-'}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`${GMT.body} ${GM.text} font-medium ${GMT.mono}`}>
                          {execution.symbol || 'USD/COP'}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {execution.side && (
                          <span className={`flex items-center gap-1 ${GMT.meta} font-bold ${execution.side === 'buy' ? GM.pos : GM.neg}`}>
                            {execution.side === 'buy' ? (
                              <TrendingUp className="w-4 h-4" aria-hidden />
                            ) : (
                              <TrendingDown className="w-4 h-4" aria-hidden />
                            )}
                            {execution.side.toUpperCase()}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`${GMT.body} ${GM.text} ${GMT.mono}`}>
                          {execution.filled_quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                        </span>
                        {execution.requested_quantity > execution.filled_quantity && (
                          <span className={`${GMT.micro} ${GM.textMuted} ${GMT.mono} ml-1`}>
                            /{execution.requested_quantity.toLocaleString()}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`${GMT.body} ${GM.text} ${GMT.mono}`}>
                          {execution.filled_price
                            ? `$${execution.filled_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                            : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        {execution.pnl != null ? (
                          <span className={`${GMT.body} font-bold ${GMT.mono} ${execution.pnl > 0 ? GM.pos : execution.pnl < 0 ? GM.neg : GM.textMuted}`}>
                            {execution.pnl > 0 ? '+' : ''}${execution.pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </span>
                        ) : (
                          <span className={`${GMT.meta} ${GM.textFaint} ${GMT.mono}`}>—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
                          {execution.processing_time_ms
                            ? `${execution.processing_time_ms.toFixed(0)}ms`
                            : '-'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
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

          {/* Paginación */}
          {totalPages > 1 && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
                {t('showingLabel')} {(page - 1) * limit + 1}-{Math.min(page * limit, total)} {t('ofLabel')} {total}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  aria-label={t('prevPage')}
                  className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <ChevronLeft className="w-5 h-5" aria-hidden />
                </button>
                <span className={`px-4 py-2 ${GMT.body} ${GM.text} ${GMT.mono}`}>
                  {t('pageLabel')} {page} {t('ofLabel')} {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={!hasMore}
                  aria-label={t('nextPage')}
                  className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <ChevronRight className="w-5 h-5" aria-hidden />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
