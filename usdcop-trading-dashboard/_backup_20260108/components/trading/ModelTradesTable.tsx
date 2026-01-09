'use client';

/**
 * ModelTradesTable Component
 * ===========================
 * Displays trade history for the selected model.
 * Includes summary statistics and filtering options.
 */

import React, { useState } from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import { useModelTrades } from '@/hooks/useModelTrades';
import {
  getSignalBadgeProps,
  formatDuration,
  formatPnL,
  formatPct,
  getPnLColorClass,
  periodOptions,
  type PeriodOption,
} from '@/lib/config/models.config';
import { cn } from '@/lib/utils';

interface ModelTradesTableProps {
  className?: string;
  maxHeight?: string;
}

export function ModelTradesTable({
  className,
  maxHeight = '400px',
}: ModelTradesTableProps) {
  const { model, isLoading: isModelLoading } = useSelectedModel();
  const [period, setPeriod] = useState<PeriodOption>('today');

  const { trades, summary, openTrade, isLoading, error, refresh } =
    useModelTrades({ period });

  if (isModelLoading) {
    return (
      <div className={cn('animate-pulse rounded-lg bg-slate-900 p-4', className)}>
        <div className="h-8 w-48 rounded bg-slate-800" />
        <div className="mt-4 space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-12 rounded bg-slate-800" />
          ))}
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className={cn('rounded-lg bg-slate-900 p-4 text-center', className)}>
        <p className="text-slate-400">Selecciona un modelo para ver trades</p>
      </div>
    );
  }

  return (
    <div className={cn('rounded-lg bg-slate-900 p-4', className)}>
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-lg font-semibold">
          <span
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: model.color }}
          />
          TRADES - {model.name}
        </h3>

        <div className="flex items-center gap-3">
          {/* Period selector */}
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value as PeriodOption)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-300 focus:border-blue-500 focus:outline-none"
          >
            {periodOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          {/* Refresh button */}
          <button
            onClick={refresh}
            disabled={isLoading}
            className="rounded-lg bg-slate-800 p-1.5 text-slate-400 hover:bg-slate-700 hover:text-slate-300 disabled:opacity-50"
            title="Actualizar"
          >
            <svg
              className={cn('h-4 w-4', isLoading && 'animate-spin')}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="mb-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-400">
          Error: {error}
        </div>
      )}

      {/* Table */}
      <div
        className="overflow-auto"
        style={{ maxHeight }}
      >
        <table className="w-full text-sm">
          <thead className="sticky top-0 border-b border-slate-700 bg-slate-900 text-slate-400">
            <tr>
              <th className="py-2 text-left font-medium">Hora</th>
              <th className="py-2 text-left font-medium">Señal</th>
              <th className="py-2 text-right font-medium">Entrada</th>
              <th className="py-2 text-right font-medium">Salida</th>
              <th className="py-2 text-right font-medium">P&L</th>
              <th className="py-2 text-right font-medium">Duración</th>
              <th className="py-2 text-center font-medium">Estado</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-400">
                  Cargando trades...
                </td>
              </tr>
            ) : trades.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-400">
                  No hay trades en este período
                </td>
              </tr>
            ) : (
              trades.map((trade) => {
                const signalProps = getSignalBadgeProps(trade.signal);
                const isOpen = trade.status === 'OPEN';

                return (
                  <tr
                    key={trade.tradeId}
                    className="border-b border-slate-800 transition-colors hover:bg-slate-800/50"
                  >
                    {/* Time */}
                    <td className="py-3">
                      <div>
                        {new Date(trade.openTime).toLocaleTimeString('es-CO', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </div>
                      <div className="text-xs text-slate-500">
                        {new Date(trade.openTime).toLocaleDateString('es-CO', {
                          day: '2-digit',
                          month: 'short',
                        })}
                      </div>
                    </td>

                    {/* Signal */}
                    <td className="py-3">
                      <span
                        className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium"
                        style={{
                          color: signalProps.color,
                          backgroundColor: signalProps.bgColor,
                        }}
                      >
                        {signalProps.icon} {signalProps.label}
                      </span>
                      <div className="mt-0.5 text-xs text-slate-500">
                        {Math.round(trade.confidence * 100)}%
                      </div>
                    </td>

                    {/* Entry Price */}
                    <td className="py-3 text-right font-mono">
                      ${trade.entryPrice?.toLocaleString('en-US', {
                        minimumFractionDigits: 2,
                      })}
                    </td>

                    {/* Exit Price */}
                    <td className="py-3 text-right font-mono">
                      {trade.exitPrice
                        ? `$${trade.exitPrice.toLocaleString('en-US', {
                            minimumFractionDigits: 2,
                          })}`
                        : '—'}
                    </td>

                    {/* P&L */}
                    <td
                      className={cn(
                        'py-3 text-right font-mono',
                        getPnLColorClass(trade.pnl)
                      )}
                    >
                      {trade.pnl !== null ? (
                        <>
                          <div>{formatPnL(trade.pnl)}</div>
                          <div className="text-xs opacity-70">
                            {formatPct(trade.pnlPct)}
                          </div>
                        </>
                      ) : (
                        '—'
                      )}
                    </td>

                    {/* Duration */}
                    <td className="py-3 text-right">
                      {formatDuration(trade.durationMinutes)}
                    </td>

                    {/* Status */}
                    <td className="py-3 text-center">
                      {isOpen ? (
                        <span className="inline-flex items-center gap-1 text-amber-400">
                          <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
                          OPEN
                        </span>
                      ) : (
                        <span className="text-slate-400">
                          {trade.pnl !== null && trade.pnl > 0 ? '✅' : trade.pnl !== null && trade.pnl < 0 ? '❌' : '➖'}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      {!isLoading && trades.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-4 rounded-lg bg-slate-800 p-3 text-sm">
          <div className="flex flex-wrap gap-4">
            <span>
              Total: <b>{summary.total}</b>
            </span>
            <span className="text-green-400">
              Ganados: <b>{summary.wins}</b>
            </span>
            <span className="text-red-400">
              Perdidos: <b>{summary.losses}</b>
            </span>
            <span className="text-slate-400">
              Hold: <b>{summary.holds}</b>
            </span>
          </div>

          <div className="flex flex-wrap gap-4">
            <span>
              Win Rate: <b>{summary.winRate}%</b>
            </span>
            <span className={getPnLColorClass(summary.pnlTotal)}>
              P&L: <b>{formatPnL(summary.pnlTotal)}</b>
            </span>
            {summary.streak !== 0 && (
              <span
                className={
                  summary.streak > 0 ? 'text-green-400' : 'text-red-400'
                }
              >
                {summary.streak > 0 ? '🔥' : '❄️'} Racha:{' '}
                <b>
                  {summary.streak > 0 ? '+' : ''}
                  {summary.streak}
                </b>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelTradesTable;
