'use client';

/**
 * GuardrailsCard — Circuit breaker status and rolling metrics.
 *
 * Shows:
 *   - Circuit breaker indicator (red/green)
 *   - Cumulative PnL
 *   - Consecutive losses count
 *   - Rolling Sharpe 16w
 *   - Rolling DA SHORT 16w
 *   - Alert list
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ShieldCheck, ShieldAlert, AlertTriangle, TrendingDown, Activity, Target } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Guardrails } from '@/lib/contracts/production-monitor.contract';

interface GuardrailsCardProps {
  guardrails: Guardrails;
}

function MetricRow({
  label,
  value,
  icon,
  status,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  status: 'ok' | 'warn' | 'danger' | 'neutral';
}) {
  const colors = {
    ok: 'text-emerald-400',
    warn: 'text-amber-400',
    danger: 'text-red-400',
    neutral: 'text-slate-400',
  };

  return (
    <div className="flex items-center justify-between py-2 px-1">
      <div className="flex items-center gap-2 text-slate-400">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <span className={cn('text-sm font-mono font-semibold', colors[status])}>
        {value}
      </span>
    </div>
  );
}

export function GuardrailsCard({ guardrails }: GuardrailsCardProps) {
  const cbActive = guardrails.circuit_breaker_active;
  const consLosses = guardrails.consecutive_losses;
  const cumPnl = guardrails.cumulative_pnl_pct;

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn('p-2 rounded-lg', cbActive ? 'bg-red-500/10' : 'bg-emerald-500/10')}>
              {cbActive
                ? <ShieldAlert className="w-5 h-5 text-red-400" />
                : <ShieldCheck className="w-5 h-5 text-emerald-400" />
              }
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-base text-white">
                Guardrails
              </CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Protecciones del pipeline
              </p>
            </div>
          </div>
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] font-bold',
              cbActive
                ? 'bg-red-500/10 text-red-400 border-red-500/30'
                : guardrails.alerts.length > 0
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
            )}
          >
            {cbActive ? 'CB Activo' : guardrails.alerts.length > 0 ? 'Alerta' : 'Normal'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0 p-4 sm:p-6">
        <div className="divide-y divide-slate-800/50">
          <MetricRow
            label="PnL Acumulado"
            value={cumPnl != null ? `${cumPnl >= 0 ? '+' : ''}${cumPnl.toFixed(2)}%` : '-'}
            icon={<TrendingDown className="w-3.5 h-3.5" />}
            status={cumPnl == null ? 'neutral' : cumPnl >= 0 ? 'ok' : cumPnl > -5 ? 'warn' : 'danger'}
          />
          <MetricRow
            label="Perdidas Consecutivas"
            value={String(consLosses)}
            icon={<AlertTriangle className="w-3.5 h-3.5" />}
            status={consLosses >= 5 ? 'danger' : consLosses >= 3 ? 'warn' : 'ok'}
          />
          <MetricRow
            label="Sharpe 16 Sem."
            value={guardrails.rolling_sharpe_16w != null
              ? guardrails.rolling_sharpe_16w.toFixed(2)
              : '< 16 semanas'}
            icon={<Activity className="w-3.5 h-3.5" />}
            status={guardrails.rolling_sharpe_16w == null
              ? 'neutral'
              : guardrails.rolling_sharpe_16w >= 1.5 ? 'ok'
                : guardrails.rolling_sharpe_16w >= 0.5 ? 'warn'
                  : 'danger'}
          />
          <MetricRow
            label="DA SHORT 16 Sem."
            value={guardrails.rolling_da_short_16w != null
              ? `${guardrails.rolling_da_short_16w.toFixed(1)}%`
              : '< 16 semanas'}
            icon={<Target className="w-3.5 h-3.5" />}
            status={guardrails.rolling_da_short_16w == null
              ? 'neutral'
              : guardrails.rolling_da_short_16w >= 60 ? 'ok'
                : guardrails.rolling_da_short_16w >= 55 ? 'warn'
                  : 'danger'}
          />
        </div>

        {/* Alert list */}
        {guardrails.alerts.length > 0 && (
          <div className="mt-4 space-y-2">
            {guardrails.alerts.map((alert, i) => (
              <div
                key={i}
                className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-500/5 border border-amber-500/20"
              >
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <span className="text-xs text-amber-300">{alert}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
