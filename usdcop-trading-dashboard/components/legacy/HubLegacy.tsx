/* ARCHIVED pre-GlobalMarkets UI (CTR-GM-UI-001) — verbatim git-HEAD copy of app/hub/page.tsx.
   Reference only, mounted at /legacy/* (admin:all). Do not evolve; the GM view is the live one. */
'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import {
  BarChart3, TrendingUp, Activity, ChevronRight,
  Calendar, LineChart, Zap, Target, ArrowRight, Cpu, FlaskConical, FileText
} from 'lucide-react';
import { GlobalNavbar } from '@/components/navigation/GlobalNavbar';
import { roleHasPermission, type Permission, type Role } from '@/lib/contracts/rbac.contract';
import { MetricBadge } from '@/components/ui/MetricBadge';
import { canShowRatios } from '@/lib/contracts/ui.contract';

export default function HubPage() {
  const router = useRouter();
  // Cards render from the RBAC contract (CTR-RBAC-001): the matrix decides what each role
  // sees; the middleware is the real defense — this is reflection only.
  const { data: session, status } = useSession();
  const sessionLoading = status === 'loading';
  const role = ((session?.user as { role?: string } | undefined)?.role ?? 'free') as Role;
  const isSubscriber = role === 'subscriber';

  const menuOptionsAll: Array<{
    id: string; title: string; subtitle: string; description: string;
    icon: typeof BarChart3; gradient: string; glowColor: string; href: string;
    features: string[]; permission: Permission;
  }> = [
    {
      id: 'dashboard',
      title: 'Trading Dashboard',
      subtitle: 'Backtest y analisis',
      description: 'Visualiza precios, senales, metricas de rendimiento y el historial de operaciones de la estrategia Smart Simple (regime gate + TP/HS).',
      icon: BarChart3,
      gradient: 'from-cyan-500 to-blue-600',
      glowColor: 'cyan',
      href: '/dashboard',
      features: ['Backtest Interactivo', 'Replay por Version', 'Equity Curve', 'Historial de Trades'],
      permission: 'research:read',
    },
    {
      id: 'production',
      title: isSubscriber ? 'Señales' : 'Monitor de Produccion',
      subtitle: isSubscriber ? 'Señales de la estrategia activa' : 'Estrategia en tiempo real',
      description: isSubscriber
        ? 'Señales y desempeño de la estrategia activa en horario de mercado, con equity y P&L al dia.'
        : 'Visualiza la estrategia activa en produccion durante horario de mercado. Equity curve, posicion actual y P&L en vivo.',
      icon: Cpu,
      gradient: 'from-green-500 to-teal-600',
      glowColor: 'green',
      href: '/production',
      features: isSubscriber
        ? ['Señal Semanal', 'Equity al Dia', 'Posicion Actual', 'P&L en Vivo']
        : ['Estrategia Activa', 'Equity NRT', 'Posicion Actual', 'P&L en Vivo'],
      permission: 'signals:read',
    },
    {
      id: 'experiments',
      title: 'Experimentos',
      subtitle: 'Aprobacion de modelos',
      description: 'Revisa y aprueba experimentos propuestos por L4. Sistema de dos votos para promocion a produccion.',
      icon: FlaskConical,
      gradient: 'from-purple-500 to-pink-600',
      glowColor: 'purple',
      href: '/dashboard', // Experiments approval is integrated in Dashboard via FloatingExperimentPanel
      features: ['Propuestas L4', 'Metricas Backtest', 'Comparacion Baseline', 'Segundo Voto'],
      permission: 'research:read',
    },
    {
      id: 'forecasting',
      title: 'Forecasting Semanal',
      subtitle: 'Predicciones a mediano plazo',
      description: 'Analiza proyecciones semanales multi-activo (USD/COP, Oro, BTC) basadas en modelos de series de tiempo y reglas cuantitativas.',
      icon: Calendar,
      gradient: 'from-amber-500 to-orange-600',
      glowColor: 'amber',
      href: '/forecasting',
      features: ['Proyeccion Semanal', 'Intervalos de confianza', 'Tendencias macro', 'Multi-activo'],
      permission: 'forecast:read',
    },
    {
      id: 'analysis',
      title: 'Analisis Semanal',
      subtitle: 'Inteligencia de mercado',
      description: 'Analisis AI multi-activo con indicadores macro, señales de estrategias, noticias y calendario economico.',
      icon: FileText,
      gradient: 'from-indigo-500 to-violet-600',
      glowColor: 'indigo',
      href: '/analysis',
      features: ['Analisis AI Diario', 'Graficos Macro', 'Timeline Semanal', 'Chat Asistente'],
      permission: 'analysis:read',
    },
    {
      id: 'execution',
      title: 'SignalBridge',
      subtitle: 'Ejecucion automatizada',
      description: 'Conecta tus exchanges y ejecuta trades automaticamente basados en las senales de la estrategia activa.',
      icon: Zap,
      gradient: 'from-rose-500 to-red-600',
      glowColor: 'rose',
      href: '/execution/dashboard',
      features: ['Conexion Exchanges', 'Paper-first', 'Gestion de Riesgo', 'Kill Switch'],
      permission: 'execution:self',
    }
  ];
  // §3.1 ux-navigation: client roles see DENIED modules as locked teasers (conversion),
  // never as absence. Internal roles simply don't see what isn't theirs.
  const isClientRole = role === 'free' || role === 'subscriber';
  // Plan-level lock (#6): a subscriber whose PLAN lacks execution sees SignalBridge as a
  // locked teaser (upsell to Auto), even though the ROLE has execution:self.
  const [execEnabled, setExecEnabled] = useState<boolean | null>(null);
  useEffect(() => {
    if (role !== 'subscriber') return;
    fetch('/api/billing/me').then((r) => (r.ok ? r.json() : null))
      .then((e) => setExecEnabled(!!e?.execution?.enabled)).catch(() => setExecEnabled(false));
  }, [role]);
  const planLockedIds = role === 'subscriber' && execEnabled === false ? ['execution'] : [];
  const allowed = menuOptionsAll.filter((c) => roleHasPermission(role, c.permission)
    && !planLockedIds.includes(c.id));
  const locked = isClientRole
    ? menuOptionsAll.filter((c) => ((!roleHasPermission(role, c.permission)
        && (c.id === 'production' || c.id === 'execution'))
        || planLockedIds.includes(c.id)))
    : [];
  const menuOptions = allowed;

  const handleNavigate = (href: string) => {
    console.log('[HUB] handleNavigate called with:', href);
    router.push(href);
  };

  return (
    <div className="min-h-screen bg-black">
      <GlobalNavbar currentPage="hub" />

      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/30 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        </div>
      </div>

      {/* Main Content */}
      <main className="relative z-10 pt-20 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">

          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl"
              >
                <Activity className="w-8 h-8 text-white" />
              </motion.div>
              <h1 className="text-3xl sm:text-4xl font-bold text-white">
                Terminal USD/COP
              </h1>
            </div>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Selecciona el modulo al que deseas acceder
            </p>
          </motion.div>

          {/* Menu Cards — skeleton while the session hydrates (no 'free-flash' for admins) */}
          {sessionLoading && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1,2,3,4,5,6].map((i) => (
                <div key={i} className="h-56 rounded-2xl border border-slate-800/60 bg-slate-900/40 animate-pulse" />
              ))}
            </div>
          )}
          {!sessionLoading && (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...menuOptions].map((option, index) => {
              const Icon = option.icon;
              return (
                <motion.div
                  key={option.id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.15 }}
                >
                  <button
                    data-testid={`hub-card-${option.id}`}
                    onClick={() => handleNavigate(option.href)}
                    className="w-full text-left group"
                  >
                    <div className={`
                      relative overflow-hidden rounded-2xl
                      bg-gray-900/80 backdrop-blur-xl
                      border border-gray-800/50
                      p-6 sm:p-8
                      transition-all duration-300
                      hover:border-${option.glowColor}-500/50
                      hover:shadow-lg hover:shadow-${option.glowColor}-500/20
                      hover:scale-[1.02]
                      active:scale-[0.98]
                    `}>
                      {/* Gradient Overlay on Hover */}
                      <div className={`
                        absolute inset-0 opacity-0 group-hover:opacity-10
                        bg-gradient-to-br ${option.gradient}
                        transition-opacity duration-300
                      `} />

                      {/* Icon & Title */}
                      <div className="relative flex items-start gap-4 mb-4">
                        <div className={`
                          p-3 rounded-xl bg-gradient-to-br ${option.gradient}
                          shadow-lg
                        `}>
                          <Icon className="w-7 h-7 text-white" />
                        </div>
                        <div className="flex-1">
                          <h2 className="text-xl sm:text-2xl font-bold text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:${option.gradient} transition-all duration-300">
                            {option.title}
                          </h2>
                          <p className="text-sm text-gray-400 mt-1">
                            {option.subtitle}
                          </p>
                        </div>
                        <ChevronRight className="w-6 h-6 text-gray-500 group-hover:text-white group-hover:translate-x-1 transition-all duration-300" />
                      </div>

                      {/* Description */}
                      <p className="relative text-gray-300 mb-6 leading-relaxed">
                        {option.description}
                      </p>

                      {/* Features */}
                      <div className="relative grid grid-cols-2 gap-2">
                        {option.features.map((feature, i) => (
                          <div
                            key={i}
                            className="flex items-center gap-2 text-sm text-gray-400"
                          >
                            <div className={`w-1.5 h-1.5 rounded-full bg-gradient-to-r ${option.gradient}`} />
                            {feature}
                          </div>
                        ))}
                      </div>

                      {/* CTA */}
                      <div className="relative mt-6 pt-4 border-t border-gray-800/50">
                        <div className={`
                          flex items-center gap-2 text-sm font-medium
                          text-gray-400 group-hover:text-white
                          transition-colors duration-300
                        `}>
                          <span>Acceder al modulo</span>
                          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
                        </div>
                      </div>
                    </div>
                  </button>
                </motion.div>
              );
            })}
            {locked.map((option) => {
              const Icon = option.icon;
              return (
                <div key={`locked-${option.id}`} className="relative rounded-2xl border border-slate-700/60 bg-slate-900/40 p-6 overflow-hidden">
                  {/* real content behind the lock (blurred) — candado con propósito */}
                  <div className="blur-[6px] select-none pointer-events-none" aria-hidden>
                    <Icon className="w-8 h-8 text-slate-400 mb-3" />
                    <h3 className="text-lg font-semibold text-white">{option.title}</h3>
                    <p className="text-sm text-slate-400 mt-1">{option.description}</p>
                  </div>
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/40">
                    <span className="text-2xl" aria-hidden>🔒</span>
                    <p className="text-xs text-slate-300 text-center px-6">
                      {option.id === 'production'
                        ? 'La señal de esta semana ya fue publicada — actívala con el plan Signals.'
                        : 'Esta señal se habría ejecutado sola en tu exchange — plan Auto.'}
                    </p>
                    <a href="/pricing" className="mt-1 text-xs font-semibold text-cyan-400 hover:underline">
                      Ver planes →
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
          )}

          {/* Quick Stats — REAL figures from the published bundle (ux-navigation §6: no
              fabricated numbers, ever; ratios hidden under N<20). Replaces the stale
              "PPO v2.4 / 67.3%" hardcodes flagged by the audit. */}
          <LiveQuickStats />
        </div>
      </main>
    </div>
  );
}


/** Live quick-stats strip — bundle truth via /api/public/live-stats (fail-soft: hides). */
function LiveQuickStats() {
  const [s, setS] = useState<{ strategy_name?: string; year?: number; unavailable?: boolean;
    return_ytd_pct?: number | null; max_dd_pct?: number | null; n_trades?: number | null;
    sharpe?: number | null; bundle_date?: string | null } | null>(null);
  useEffect(() => {
    fetch('/api/public/live-stats').then(r => r.ok ? r.json() : null).then(setS).catch(() => null);
  }, []);
  if (!s || s.unavailable || s.return_ytd_pct == null) return null;
  const items: Array<[string, string]> = [
    ['Estrategia activa', s.strategy_name ?? '—'],
    [`Retorno ${s.year} YTD`, `${s.return_ytd_pct >= 0 ? '+' : ''}${s.return_ytd_pct.toFixed(2)}%`],
    ['Max Drawdown', s.max_dd_pct != null ? `−${Math.abs(s.max_dd_pct).toFixed(1)}%` : '—'],
    canShowRatios(s.n_trades) && s.sharpe != null
      ? ['Sharpe', s.sharpe.toFixed(2)]
      : ['Operaciones', String(s.n_trades ?? '—')],
  ];
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.5 }}
      className="mt-12">
      <div className="flex items-center gap-2 mb-3">
        <MetricBadge phase="live" provenance={{ strategyId: s.strategy_name ?? '',
          bundleDate: s.bundle_date ?? undefined }} />
        <span className="text-[11px] text-gray-500">cifras del bundle publicado — no recomputadas</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {items.map(([label, value]) => (
          <div key={label} className="bg-gray-900/50 backdrop-blur rounded-xl p-4 border border-gray-800/30">
            <div className="text-xs text-gray-500 mb-1">{label}</div>
            <span className="text-lg font-bold text-white tabular-nums">{value}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
