'use client';

/**
 * /execution/dashboard — SignalBridge panel, re-skin GM (prototipo Var B §Execution,
 * líneas 855–970 + view-model 2636–2718). Presentación + i18n ÚNICAMENTE:
 * fetch/polling 10s, handlers del kill switch y contratos quedan intactos.
 */
import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useSession } from 'next-auth/react';
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  Zap,
  Shield,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  Power,
} from 'lucide-react';
import { GmBadge, GmPageHeader, GmPanel } from '@/components/gm';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { signalBridgeService } from '@/lib/services/execution';
import {
  type BridgeStatus,
  type BridgeStatistics,
  type TradingMode,
  formatUptime,
  isBridgeOperational,
} from '@/lib/contracts/execution/signal-bridge.contract';
import { EXEC_DICT } from './../i18n';

/** Tono GM por modo de trading (presentacional; el enum viene del contrato). */
const MODE_TONE: Record<TradingMode, GmTone> = {
  KILLED: 'neg',
  DISABLED: 'neutral',
  SHADOW: 'info',
  PAPER: 'warn',
  STAGING: 'info',
  LIVE: 'pos',
};

/** Real per-user trading state (subset used here) resolved from SignalBridge. */
interface UserTradingState {
  trade_count_today: number;
  daily_pnl_pct: number;
}

export default function ExecutionDashboardPage() {
  const t = useGmT(EXEC_DICT);
  const { data: session } = useSession();
  const role = (session?.user as { role?: string } | undefined)?.role ?? 'free';
  const subtitle = role === 'admin' ? t('subAdmin') : t('subUser');

  const [status, setStatus] = useState<BridgeStatus | null>(null);
  const [statistics, setStatistics] = useState<BridgeStatistics | null>(null);
  const [statsDegraded, setStatsDegraded] = useState(false);
  const [userState, setUserState] = useState<UserTradingState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isKillSwitchLoading, setIsKillSwitchLoading] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [statusData, statsResult, userStateResult] = await Promise.all([
        signalBridgeService.getStatus(),
        signalBridgeService.getStatisticsWithHealth(7),
        // Per-user state is best-effort — never fail the whole panel over it.
        signalBridgeService.getUserState().catch(() => null),
      ]);
      setStatus(statusData);
      setStatistics(statsResult.statistics);
      setStatsDegraded(statsResult.degraded);
      setUserState(userStateResult);
    } catch (err) {
      console.error('Failed to fetch bridge data:', err);
      setError(err instanceof Error ? err.message : t('loadError'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleKillSwitch = async () => {
    if (!status) return;

    const confirmed = window.confirm(
      status.kill_switch_active ? t('killConfirmOff') : t('killConfirmOn')
    );

    if (!confirmed) return;

    setIsKillSwitchLoading(true);
    try {
      if (status.kill_switch_active) {
        await signalBridgeService.deactivateKillSwitch();
      } else {
        await signalBridgeService.activateKillSwitch('Manual activation from dashboard');
      }
      await fetchData();
    } catch (err) {
      console.error('Kill switch error:', err);
      alert(t('killError'));
    } finally {
      setIsKillSwitchLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center" aria-busy>
        <div className="flex flex-col items-center gap-4">
          <RefreshCw className={`w-8 h-8 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('loadingBridge')}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center" role="alert">
        <div className="text-center">
          <AlertTriangle className={`w-12 h-12 ${GM.neg} mx-auto mb-4`} aria-hidden />
          <h2 className={`text-xl font-bold ${GM.headline} mb-2`}>{t('connErrorTitle')}</h2>
          <p className={`${GMT.body} ${GM.textMuted} mb-4`}>{error}</p>
          <button
            onClick={fetchData}
            className={`${GM.ctaPrimary} ${GM.focus} h-11 px-5 text-[13.5px] inline-flex items-center gap-2`}
          >
            <RefreshCw className="w-4 h-4" aria-hidden />
            {t('retry')}
          </button>
        </div>
      </div>
    );
  }

  const isOperational = status ? isBridgeOperational(status) : false;
  const mode = status?.trading_mode;
  const modeTone: GmTone = mode ? MODE_TONE[mode] : 'neutral';

  return (
    <div className="space-y-6">
      {/* Header (título + subtítulo por rol como el prototipo `exSub`) */}
      <GmPageHeader
        kicker={t('kicker')}
        title={t('dashTitle')}
        subtitle={subtitle}
        actions={
          <>
            {mode && <GmBadge tone={modeTone}>{mode}</GmBadge>}
            <button
              onClick={fetchData}
              aria-label={t('refresh')}
              className={`${GM.ctaGhost} ${GM.focus} w-11 h-11 flex items-center justify-center`}
            >
              <RefreshCw className="w-4 h-4" aria-hidden />
            </button>
          </>
        }
      />

      {/* Alerta kill switch activo */}
      {status?.kill_switch_active && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`${GM.negBadge} rounded-xl p-4 flex flex-wrap items-center gap-4`}
          role="alert"
        >
          <AlertTriangle className={`w-6 h-6 ${GM.neg} flex-shrink-0`} aria-hidden />
          <div className="flex-1 min-w-[200px]">
            <h3 className={`font-bold ${GM.neg}`}>{t('killAlertTitle')}</h3>
            <p className={`${GMT.meta} ${GM.textSec}`}>
              {status.kill_switch_reason || t('killAlertFallback')}
            </p>
          </div>
          <button
            onClick={handleKillSwitch}
            disabled={isKillSwitchLoading}
            className={`${GM.posBadge} ${GM.focus} h-11 px-4 rounded-[11px] font-bold text-[13px] disabled:opacity-50`}
          >
            {isKillSwitchLoading ? t('processing') : t('killResume')}
          </button>
        </motion.div>
      )}

      {/* Aviso: estadísticas degradadas (backend inaccesible) — no fingir $0/0 */}
      {statsDegraded && (
        <div className={`${GM.warnBadge} rounded-xl p-4 flex items-center gap-3`} role="alert">
          <AlertTriangle className={`w-5 h-5 ${GM.warn} flex-shrink-0`} aria-hidden />
          <span className={`${GMT.body} ${GM.warn}`}>{t('statsDegraded')}</span>
        </div>
      )}

      {/* KPIs de estado */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Estado del bridge */}
        <div className={`${GM.panel} p-5`}>
          <div className="flex items-center justify-between mb-3">
            <span className={`${GMT.label} ${GM.textMuted}`}>{t('bridgeStatus')}</span>
            <Activity className={`w-5 h-5 ${isOperational ? GM.pos : GM.neg}`} aria-hidden />
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isOperational
                  ? 'bg-[var(--gm-pos)] motion-safe:animate-pulse'
                  : 'bg-[var(--gm-neg)]'
              }`}
              aria-hidden
            />
            <span className={`text-xl font-bold ${GM.headline}`}>
              {isOperational ? t('operational') : t('stoppedStatus')}
            </span>
          </div>
          {status && (
            <p className={`${GMT.micro} ${GM.textMuted} ${GMT.mono} mt-2`}>
              {t('uptime')}: {formatUptime(status.uptime_seconds)}
            </p>
          )}
        </div>

        {/* Modo de trading */}
        <div className={`${GM.panel} p-5`}>
          <div className="flex items-center justify-between mb-3">
            <span className={`${GMT.label} ${GM.textMuted}`}>{t('tradingMode')}</span>
            <Shield className={`w-5 h-5 ${GM.accent}`} aria-hidden />
          </div>
          {status && (
            <>
              <GmBadge tone={modeTone} className="text-[13px] px-3 py-1">
                {status.trading_mode}
              </GmBadge>
              <p className={`${GMT.micro} ${GM.textMuted} mt-2`}>
                {status.inference_ws_connected ? t('wsConnected') : t('wsDisconnected')}
              </p>
            </>
          )}
        </div>

        {/* Ejecuciones 7d */}
        <div className={`${GM.panel} p-5`}>
          <div className="flex items-center justify-between mb-3">
            <span className={`${GMT.label} ${GM.textMuted}`}>{t('executions7d')}</span>
            <Zap className={`w-5 h-5 ${GM.warn}`} aria-hidden />
          </div>
          <span className={`${GMT.kpi} ${GM.headline}`}>
            {statsDegraded ? '—' : statistics?.total_executions || 0}
          </span>
          <div className={`flex items-center gap-2 mt-2 ${GMT.micro} ${GMT.mono}`}>
            {statsDegraded ? (
              <span className={GM.textMuted}>{t('noData')}</span>
            ) : (
              <>
                <span className={GM.pos}>{statistics?.successful_executions || 0} {t('okSuffix')}</span>
                <span className={GM.textFaint} aria-hidden>|</span>
                <span className={GM.neg}>{statistics?.failed_executions || 0} {t('failSuffix')}</span>
              </>
            )}
          </div>
        </div>

        {/* P&L 7d */}
        <div className={`${GM.panel} p-5`}>
          <div className="flex items-center justify-between mb-3">
            <span className={`${GMT.label} ${GM.textMuted}`}>{t('pnl7d')}</span>
            <TrendingUp className={`w-5 h-5 ${GM.pos}`} aria-hidden />
          </div>
          {statsDegraded ? (
            <span className={`${GMT.kpi} ${GM.textMuted}`}>—</span>
          ) : (
            <span className={`${GMT.kpi} ${(statistics?.total_pnl_usd || 0) >= 0 ? GM.pos : GM.neg}`}>
              ${(statistics?.total_pnl_usd || 0).toFixed(2)}
            </span>
          )}
          <p className={`${GMT.micro} ${GM.textMuted} ${GMT.mono} mt-2`}>
            {statsDegraded
              ? t('noData')
              : `${t('volumeLabel')}: $${(statistics?.total_volume_usd || 0).toLocaleString()}`}
          </p>
        </div>
      </div>

      {/* Grid principal */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Columna kill switch + paper (prototipo, columna derecha) */}
        <div className="space-y-6">
          {/* Kill switch */}
          <section
            className="rounded-2xl bg-[rgba(251,113,133,.05)] border border-[rgba(251,113,133,.25)] p-[18px]"
            aria-labelledby="kill-switch-title"
          >
            <div className="flex items-center justify-between mb-1.5">
              <h2 id="kill-switch-title" className={`${GMT.panelTitle} ${GM.neg} flex items-center gap-2`}>
                <Power className="w-4 h-4" aria-hidden />
                {t('killTitle')}
              </h2>
              <GmBadge tone={status?.kill_switch_active ? 'neg' : 'pos'}>
                {status?.kill_switch_active ? t('killStateActive') : t('killStateInactive')}
              </GmBadge>
            </div>
            <p className={`${GMT.meta} ${GM.textSec} mb-4 leading-relaxed`}>{t('killDesc')}</p>
            <button
              onClick={handleKillSwitch}
              disabled={isKillSwitchLoading}
              className={`${GM.focus} w-full h-11 rounded-[11px] font-extrabold text-[13.5px] tracking-[.5px] transition-colors duration-[var(--gm-dur-fast)] disabled:opacity-50 ${
                status?.kill_switch_active
                  ? `${GM.posBadge} font-bold`
                  : `bg-[var(--gm-neg)] ${GM.onAccent} shadow-[0_6px_20px_rgba(251,113,133,.3)] hover:opacity-90`
              }`}
            >
              {isKillSwitchLoading
                ? t('processing')
                : status?.kill_switch_active
                  ? t('killResume')
                  : t('killStop')}
            </button>
            <p className={`${GMT.micro} ${GM.textMuted} mt-3 leading-relaxed`}>{t('killNote')}</p>
          </section>

          {/* Actividad real del usuario (trades de hoy) — datos de SB user state,
              NO el uptime del bridge. SB no expone una "semana de paper", así que
              mostramos el conteo real de trades y el P&L diario, etiquetado con honestidad. */}
          <GmPanel title={t('paperTitle')}>
            <div className="flex items-center justify-between mb-2">
              <span className={`${GMT.meta} ${GM.textSec}`}>{t('tradesTodayLabel')}</span>
              {mode && <GmBadge tone={modeTone}>{mode}</GmBadge>}
            </div>
            <div className="flex items-baseline gap-2">
              <span className={`${GMT.kpi} ${GM.headline}`}>
                {userState ? userState.trade_count_today : '—'}
              </span>
              {userState && (
                <span
                  className={`${GMT.micro} ${GMT.mono} ${
                    userState.daily_pnl_pct >= 0 ? GM.pos : GM.neg
                  }`}
                >
                  {userState.daily_pnl_pct >= 0 ? '+' : ''}
                  {userState.daily_pnl_pct.toFixed(2)}%
                </span>
              )}
            </div>
            <p className={`${GMT.micro} ${GM.textMuted} mt-3`}>{t('paperRealNote')}</p>
          </GmPanel>
        </div>

        {/* Estado del sistema */}
        <GmPanel
          title={
            <span className="flex items-center gap-2">
              <Clock className={`w-4 h-4 ${GM.accent}`} aria-hidden />
              {t('systemStatus')}
            </span>
          }
          className="lg:col-span-2"
        >
          <div className="space-y-3">
            {[
              {
                label: t('inferenceWs'),
                status: status?.inference_ws_connected,
                icon: Activity,
              },
              {
                label: t('connectedLabel'),
                status: (status?.connected_exchanges?.length || 0) > 0,
                detail: status?.connected_exchanges?.join(', ') || t('noneLabel'),
                icon: Link2Icon,
              },
              {
                label: t('pendingLabel'),
                status: true,
                detail: `${status?.pending_executions || 0} ${t('pendingSuffix')}`,
                icon: Clock,
              },
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className={`${GM.panelInner} flex items-center justify-between p-3`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-4 h-4 ${GM.textMuted}`} aria-hidden />
                    <span className={`${GMT.body} ${GM.textStrong}`}>{item.label}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {item.detail && (
                      <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>{item.detail}</span>
                    )}
                    {item.status ? (
                      <CheckCircle2 className={`w-5 h-5 ${GM.pos}`} aria-hidden />
                    ) : (
                      <XCircle className={`w-5 h-5 ${GM.neg}`} aria-hidden />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Resumen de stats (oculto si están degradadas para no mostrar ceros falsos) */}
          {statistics && !statsDegraded && (
            <div className="mt-6 pt-4 border-t border-[var(--gm-border)]">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className={`${GMT.kpi} ${GM.headline}`}>{statistics.total_signals_received}</p>
                  <p className={`${GMT.micro} ${GM.textMuted}`}>{t('signalsReceived')}</p>
                </div>
                <div>
                  <p className={`${GMT.kpi} ${GM.warn}`}>{statistics.blocked_by_risk}</p>
                  <p className={`${GMT.micro} ${GM.textMuted}`}>{t('blockedByRisk')}</p>
                </div>
                <div>
                  <p className={`${GMT.kpi} ${GM.accent}`}>
                    {statistics.avg_execution_time_ms.toFixed(0)}ms
                  </p>
                  <p className={`${GMT.micro} ${GM.textMuted}`}>{t('avgExecTime')}</p>
                </div>
              </div>
            </div>
          )}
        </GmPanel>
      </div>
    </div>
  );
}

// Helper component for the link icon
function Link2Icon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M9 17H7A5 5 0 0 1 7 7h2" />
      <path d="M15 7h2a5 5 0 1 1 0 10h-2" />
      <line x1="8" x2="16" y1="12" y2="12" />
    </svg>
  );
}
