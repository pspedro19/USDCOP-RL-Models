'use client';

/**
 * LivePositionCard — Shows current week's signal and active position.
 *
 * Three states:
 *   1. No signal → "Sin senal activa"
 *   2. Signal but no position → Signal details + "Esperando entrada" / "Trade omitido"
 *   3. Positioned → Entry, current price, unrealized PnL, TP/HS progress bars
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Target, ShieldAlert, Clock, Crosshair } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CurrentSignal, ActivePosition } from '@/lib/contracts/production-monitor.contract';

interface LivePositionCardProps {
  signal: CurrentSignal | null;
  position: ActivePosition | null;
  marketOpen: boolean;
}

function ProgressBar({
  label,
  pct,
  colorClass,
  icon,
}: {
  label: string;
  pct: number;
  colorClass: string;
  icon: React.ReactNode;
}) {
  const clamped = Math.min(100, Math.max(0, pct));
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-slate-400">
          {icon}
          {label}
        </span>
        <span className={cn('font-mono font-semibold', colorClass)}>
          {clamped.toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-700', colorClass.replace('text-', 'bg-'))}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function DirectionBadge({ direction }: { direction: number }) {
  const isShort = direction === -1;
  return (
    <Badge
      variant="outline"
      className={cn(
        'text-[10px] font-bold',
        isShort
          ? 'bg-red-500/10 text-red-400 border-red-500/30'
          : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
      )}
    >
      {isShort ? 'SHORT' : 'LONG'}
    </Badge>
  );
}

function ConfidenceBadge({ tier }: { tier: string | null }) {
  if (!tier) return null;
  const colors: Record<string, string> = {
    HIGH: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    MEDIUM: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    LOW: 'bg-slate-500/10 text-slate-400 border-slate-500/30',
  };
  return (
    <Badge variant="outline" className={cn('text-[10px] font-bold', colors[tier] ?? colors.LOW)}>
      {tier}
    </Badge>
  );
}

export function LivePositionCard({ signal, position, marketOpen }: LivePositionCardProps) {
  // State 1: No signal
  if (!signal) {
    return (
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader variant="minimal">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-slate-500/10">
              <Crosshair className="w-5 h-5 text-slate-400" />
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-base text-white">
                Esta Semana
              </CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">Sin senal activa</p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0 p-4 sm:p-6">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
            <Clock className="w-4 h-4 text-slate-500 shrink-0" />
            <p className="text-sm text-slate-400">
              Proxima senal: lunes 08:15 COT
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // State 2: Signal but skipped or no position yet
  if (signal.skip_trade || !position) {
    return (
      <Card className="bg-slate-900/50 border-slate-800">
        <CardHeader variant="minimal">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10">
                <Crosshair className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <CardTitle variant="default" gradient={false} className="text-base text-white">
                  Esta Semana
                </CardTitle>
                <p className="text-xs text-slate-500 mt-0.5">Senal del {signal.signal_date}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <DirectionBadge direction={signal.direction} />
              <ConfidenceBadge tier={signal.confidence_tier} />
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0 p-4 sm:p-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <div className="text-center p-2 rounded-lg bg-slate-800/50">
              <p className="text-[10px] text-slate-500 uppercase">Retorno Est.</p>
              <p className={cn('text-sm font-mono font-bold',
                signal.ensemble_return < 0 ? 'text-red-400' : 'text-emerald-400'
              )}>
                {signal.ensemble_return >= 0 ? '+' : ''}{(signal.ensemble_return * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-2 rounded-lg bg-slate-800/50">
              <p className="text-[10px] text-slate-500 uppercase">Leverage</p>
              <p className="text-sm font-mono font-bold text-white">
                {signal.adjusted_leverage?.toFixed(2) ?? '-'}x
              </p>
            </div>
            <div className="text-center p-2 rounded-lg bg-slate-800/50">
              <p className="text-[10px] text-slate-500 uppercase">Hard Stop</p>
              <p className="text-sm font-mono font-bold text-red-400/70">
                {signal.hard_stop_pct?.toFixed(1) ?? '-'}%
              </p>
            </div>
            <div className="text-center p-2 rounded-lg bg-slate-800/50">
              <p className="text-[10px] text-slate-500 uppercase">Take Profit</p>
              <p className="text-sm font-mono font-bold text-emerald-400/70">
                {signal.take_profit_pct?.toFixed(1) ?? '-'}%
              </p>
            </div>
          </div>
          <div className={cn(
            'flex items-center gap-3 p-3 rounded-lg border',
            signal.skip_trade
              ? 'bg-amber-500/5 border-amber-500/20'
              : 'bg-cyan-500/5 border-cyan-500/20'
          )}>
            {signal.skip_trade
              ? <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
              : <Clock className="w-4 h-4 text-cyan-400 shrink-0 animate-pulse" />
            }
            <p className={cn('text-sm font-medium',
              signal.skip_trade ? 'text-amber-300' : 'text-cyan-300'
            )}>
              {signal.skip_trade ? 'Trade omitido (LOW confidence LONG)' : 'Esperando entrada...'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // State 3: Positioned — show full position details with TP/HS bars
  const isShort = position.direction === -1;
  const pnlPositive = position.unrealized_pnl_pct >= 0;

  // Progress calculations
  const tpPct = signal.take_profit_pct;
  const hsPct = signal.hard_stop_pct;

  // For TP: how far price has moved in favorable direction relative to TP target
  let tpProgress = 0;
  if (tpPct && tpPct > 0) {
    const favorableMove = isShort
      ? (position.entry_price - position.current_price) / position.entry_price * 100
      : (position.current_price - position.entry_price) / position.entry_price * 100;
    tpProgress = Math.max(0, (favorableMove / tpPct) * 100);
  }

  // For HS: how far price has moved in adverse direction relative to HS target
  let hsProgress = 0;
  if (hsPct && hsPct > 0) {
    const adverseMove = isShort
      ? (position.current_price - position.entry_price) / position.entry_price * 100
      : (position.entry_price - position.current_price) / position.entry_price * 100;
    hsProgress = Math.max(0, (adverseMove / hsPct) * 100);
  }

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn('p-2 rounded-lg', pnlPositive ? 'bg-emerald-500/10' : 'bg-red-500/10')}>
              {pnlPositive
                ? <TrendingUp className="w-5 h-5 text-emerald-400" />
                : <TrendingDown className="w-5 h-5 text-red-400" />
              }
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-base text-white">
                Posicion Activa
              </CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                {signal.signal_date} | Bar #{position.bar_count}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <DirectionBadge direction={position.direction} />
            <ConfidenceBadge tier={signal.confidence_tier} />
            {marketOpen && (
              <Badge variant="outline" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-[9px] font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse mr-1 inline-block" />
                LIVE
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0 p-4 sm:p-6 space-y-5">
        {/* Price + PnL row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="text-center p-2.5 rounded-lg bg-slate-800/50">
            <p className="text-[10px] text-slate-500 uppercase">Entrada</p>
            <p className="text-sm font-mono font-bold text-white">
              ${position.entry_price.toLocaleString('en-US', { minimumFractionDigits: 1 })}
            </p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-slate-800/50">
            <p className="text-[10px] text-slate-500 uppercase">Actual</p>
            <p className="text-sm font-mono font-bold text-cyan-300">
              ${position.current_price.toLocaleString('en-US', { minimumFractionDigits: 1 })}
            </p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-slate-800/50">
            <p className="text-[10px] text-slate-500 uppercase">PnL No Realizado</p>
            <p className={cn('text-sm font-mono font-bold',
              pnlPositive ? 'text-emerald-400' : 'text-red-400'
            )}>
              {pnlPositive ? '+' : ''}{position.unrealized_pnl_pct.toFixed(2)}%
            </p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-slate-800/50">
            <p className="text-[10px] text-slate-500 uppercase">Leverage</p>
            <p className="text-sm font-mono font-bold text-white">
              {position.leverage.toFixed(2)}x
            </p>
          </div>
        </div>

        {/* TP / HS Progress Bars */}
        <div className="space-y-3">
          <ProgressBar
            label="Take Profit"
            pct={tpProgress}
            colorClass="text-emerald-400"
            icon={<Target className="w-3 h-3" />}
          />
          <ProgressBar
            label="Hard Stop"
            pct={hsProgress}
            colorClass="text-red-400"
            icon={<ShieldAlert className="w-3 h-3" />}
          />
        </div>

        {/* Peak price */}
        {position.peak_price != null && (
          <div className="flex items-center justify-between text-xs text-slate-500 px-1">
            <span>Peak: ${position.peak_price.toLocaleString('en-US', { minimumFractionDigits: 1 })}</span>
            <span>TP: {signal.take_profit_pct?.toFixed(1) ?? '-'}% | HS: {signal.hard_stop_pct?.toFixed(1) ?? '-'}%</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
