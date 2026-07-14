'use client';

/**
 * LandingView — public landing ('/') on the GlobalMarkets design system
 * (CTR-GM-UI-001, prototype Var B lines 1422-1495 + labels from support strings).
 *
 * Public page: PublicChrome header (NOT TerminalShell). Live numbers come from
 * GET /api/public/live-stats (published bundle, forward-production only — marketing
 * policy audit I-8/B.2). The terminal demo card's ticker/signal rows are STATIC
 * ILLUSTRATIVE content and are labeled as such; only the KPI tiles show live data.
 * Disclaimer persistente: rbac.md §9 via PublicFooter.
 */
import { useState } from 'react';
import Link from 'next/link';
import {
  ArrowRight, Bolt, LineChart, Loader2, Radio, ShieldCheck, TrendingDown, UserRound,
} from 'lucide-react';

import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, toneOf, GM_TONE_TEXT, GM_HEX } from '@/lib/ui/gm-tokens';
import { useGmQuery } from '@/components/gm';
import { PublicFooter, PublicHeader } from './PublicChrome';

// ─────────────────────────────────────────────── i18n (CTA invitado, prototipo l. 3075)

const LANDING_GUEST_DICT = defineGmDict({
  es: {
    guestCta: 'Explorar como invitado',
    guestBusy: 'Entrando…',
    guestError: 'El acceso de invitado no está disponible ahora — inicia sesión o crea tu cuenta.',
  },
  en: {
    guestCta: 'Explore as guest',
    guestBusy: 'Signing in…',
    guestError: 'Guest access is unavailable right now — sign in or create your account.',
  },
});

/** Sesión demo rol free (guest@demo.local, is_test) vía POST /api/auth/guest → /hub. */
function GuestButton({ testid, onError }: { testid: string; onError: (msg: string) => void }) {
  const t = useGmT(LANDING_GUEST_DICT);
  const [busy, setBusy] = useState(false);
  const enterAsGuest = async () => {
    setBusy(true);
    try {
      const res = await fetch('/api/auth/guest', { method: 'POST' });
      if (res.ok) {
        window.location.href = '/hub';
        return;
      }
      onError(t('guestError'));
    } catch {
      onError(t('guestError'));
    }
    setBusy(false);
  };
  return (
    <button
      type="button"
      data-testid={testid}
      onClick={enterAsGuest}
      disabled={busy}
      className={`${GM.ctaGhost} ${GM.focus} h-12 px-5 inline-flex items-center gap-2 text-[14.5px] font-bold disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      {busy
        ? <><Loader2 className="w-4 h-4 animate-spin" aria-hidden /> {t('guestBusy')}</>
        : <><UserRound className="w-4 h-4" aria-hidden /> {t('guestCta')}</>}
    </button>
  );
}

interface LiveStats {
  phase: 'live';
  unavailable?: boolean;
  strategy_name?: string;
  year?: number;
  return_ytd_pct?: number | null;
  max_dd_pct?: number | null;
  weeks_live?: number | null;
  n_trades?: number | null;
  sharpe?: number | null;
  bundle_date?: string | null;
}

const fmtPct = (v: number | null | undefined) =>
  v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;

/** Assets actually onboarded to the registry today (strategy-contract.md). */
const ASSETS = [
  { sym: 'USD/COP', cat: 'Forex', dot: 'bg-[var(--gm-accent)]' },
  { sym: 'XAU/USD', cat: 'Oro', dot: 'bg-[var(--gm-pos)]' },
  { sym: 'BTC/USDT', cat: 'Cripto', dot: 'bg-[var(--gm-warn)]' },
] as const;

/** Static illustrative ticker for the demo card — NOT market data. */
const DEMO_TICKER = [
  { sym: 'USD/COP', price: '4.180,50', chg: '-0,42%', neg: true },
  { sym: 'XAU/USD', price: '3.352,10', chg: '+0,31%', neg: false },
  { sym: 'BTC/USDT', price: '108.420', chg: '+1,08%', neg: false },
] as const;

const FEATURES = [
  {
    icon: LineChart, color: GM_HEX.accent,
    title: 'Backtest OOS con gates',
    desc: 'Sharpe, drawdown, significancia y conteo de trades. Nada llega a producción sin pasar los gates automáticos y la doble aprobación humana.',
  },
  {
    icon: Radio, color: GM_HEX.pos,
    title: 'Señales publicadas',
    desc: 'Dirección, niveles de salida y confianza del modelo, publicadas desde bundles inmutables — los mismos números que auditamos.',
  },
  {
    icon: Bolt, color: GM_HEX.neg,
    title: 'SignalBridge',
    desc: 'Conecta Binance/MEXC con tus propias llaves (sin permiso de retiro). 4 semanas en paper antes de dinero real y kill switch propio.',
  },
  {
    icon: ShieldCheck, color: GM_HEX.violet,
    title: 'Riesgo primero',
    desc: 'Gates de régimen, límites de exposición y circuit breakers. El sistema sabe cuándo NO operar — esa es la mitad del edge.',
  },
] as const;

const STEPS = [
  { n: '01', title: 'Crea tu cuenta', desc: 'Regístrate y elige tu plan. El registro pasa por aprobación del equipo.' },
  { n: '02', title: 'Elige tus activos', desc: 'USD/COP incluido; Oro y BTC como add-ons por activo.' },
  { n: '03', title: 'Recibe señales', desc: 'Análisis semanal, forecasting y señales según tu plan.' },
  { n: '04', title: 'Ejecuta', desc: 'Paper-first y luego automático en tu propio exchange.' },
] as const;

export function LandingView() {
  const live = useGmQuery<LiveStats>('/api/public/live-stats');
  const stats = live.data && !live.data.unavailable ? live.data : null;
  const [guestError, setGuestError] = useState('');

  return (
    <div className={`min-h-screen flex flex-col ${GM.page}`} data-testid="gm-landing">
      <PublicHeader />

      {/* flex justify-center centers the inner column reliably; `w-full mx-auto`
          on a flex-column child resolves its margins to 0 → content off-center. */}
      <main className="flex-1 w-full flex justify-center px-6">
       <div className="w-full max-w-[1120px]">
        {/* ── hero ─────────────────────────────────────────────── */}
        <section className="grid lg:grid-cols-[1.1fr_.9fr] gap-10 items-center pt-8 pb-11">
          <div>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${GM.accentBadge} text-[11.5px] font-bold tracking-[.3px] mb-4`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--gm-accent)]" aria-hidden />
              Terminal cuantitativa multi-activo
            </div>
            <h1 className={`m-0 text-[34px] sm:text-[42px] leading-[1.08] font-extrabold ${GM.headline} tracking-[-1px] [text-wrap:balance]`}>
              Señales cuantitativas con gestión de riesgo institucional
            </h1>
            <p className={`mt-4 mb-0 text-[15.5px] leading-[1.65] ${GM.textSec} max-w-[520px]`}>
              Modelos cuantitativos sobre USD/COP, Oro y Bitcoin. Backtests verificados fuera
              de muestra, señales publicadas desde bundles inmutables y ejecución paper-first
              en tu propio exchange.
            </p>
            <div className="flex flex-wrap gap-3 mt-7">
              <Link
                href="/register"
                data-testid="landing-cta-register"
                className={`${GM.ctaPrimary} ${GM.focus} h-12 px-6 inline-flex items-center text-[14.5px] shadow-[0_10px_30px_rgba(34,211,238,.25)]`}
              >
                Crear cuenta gratis
              </Link>
              <Link
                href="/pricing"
                data-testid="landing-cta-pricing"
                className={`${GM.ctaGhost} ${GM.focus} h-12 px-5 inline-flex items-center text-[14.5px] font-bold`}
              >
                Ver planes
              </Link>
              <GuestButton testid="landing-cta-guest" onError={setGuestError} />
            </div>
            {guestError && (
              <p className={`mt-3 mb-0 ${GMT.meta} ${GM.warn}`} role="alert">{guestError}</p>
            )}
            <Link
              href="/metodologia"
              className={`mt-4 inline-flex items-center gap-1.5 ${GM.accent} text-[13px] font-semibold ${GM.focus}`}
            >
              Conoce la metodología <ArrowRight className="w-4 h-4" aria-hidden />
            </Link>
          </div>

          {/* terminal demo card */}
          <div
            data-testid="landing-terminal-card"
            className="rounded-[18px] bg-gradient-to-br from-[rgba(14,21,38,.95)] to-[rgba(10,14,24,.95)] border border-[rgba(148,163,184,.14)] shadow-[0_30px_80px_rgba(0,0,0,.5)] overflow-hidden"
          >
            <div className="flex items-center gap-1.5 px-3.5 py-3 border-b border-[rgba(148,163,184,.10)]">
              <span className="w-2.5 h-2.5 rounded-full bg-[var(--gm-neg)]" aria-hidden />
              <span className="w-2.5 h-2.5 rounded-full bg-[var(--gm-warn)]" aria-hidden />
              <span className="w-2.5 h-2.5 rounded-full bg-[var(--gm-pos)]" aria-hidden />
              <span className={`ml-2 ${GMT.micro} ${GM.textMuted} font-mono`}>GlobalMarkets · terminal</span>
              <span className={`ml-auto ${GM.neutralBadge} rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[.5px]`}>
                Vista demo
              </span>
            </div>
            <div className="p-4">
              {/* illustrative ticker rows */}
              <div className="flex flex-col gap-2 mb-3" aria-label="ticker ilustrativo">
                {DEMO_TICKER.map((t) => (
                  <div key={t.sym} className={`flex items-center gap-2.5 px-3 py-2 ${GM.panelInner} rounded-[10px]`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${t.neg ? 'bg-[var(--gm-neg)]' : 'bg-[var(--gm-pos)]'}`} aria-hidden />
                    <span className={`text-[12px] font-bold ${GM.textStrong} font-mono flex-1`}>{t.sym}</span>
                    <span className={`text-[12px] ${GM.textSec} font-mono`}>{t.price}</span>
                    <span className={`text-[11.5px] font-bold font-mono ${t.neg ? GM.neg : GM.pos}`}>{t.chg}</span>
                  </div>
                ))}
              </div>
              {/* illustrative signal chip */}
              <div className={`flex items-center gap-2 px-3 py-2.5 rounded-[11px] ${GM.negBadge} mb-3`}>
                <TrendingDown className="w-4 h-4" aria-hidden />
                <span className="text-[12px] font-extrabold">SHORT USD/COP</span>
                <span className={`ml-auto ${GMT.micro} ${GM.textSec}`}>señal de ejemplo</span>
              </div>
              {/* KPI tiles — LIVE forward numbers from the published bundle */}
              <div className="flex gap-2.5">
                <div className={`flex-1 p-3 ${GM.panelInner} rounded-[11px]`}>
                  <div className={`text-[20px] font-extrabold font-mono tabular-nums ${GM_TONE_TEXT[toneOf(stats?.return_ytd_pct)]}`}>
                    {fmtPct(stats?.return_ytd_pct)}
                  </div>
                  <div className={`${GMT.micro} ${GM.textMuted} mt-0.5`}>
                    Retorno {stats?.year ?? 'YTD'} · en vivo
                  </div>
                </div>
                <div className={`flex-1 p-3 ${GM.panelInner} rounded-[11px]`}>
                  <div className={`text-[20px] font-extrabold font-mono tabular-nums ${GM.accent}`}>
                    {stats?.weeks_live ?? '—'}
                  </div>
                  <div className={`${GMT.micro} ${GM.textMuted} mt-0.5`}>Semanas en forward</div>
                </div>
              </div>
              {stats?.bundle_date && (
                <div className={`mt-2.5 text-right ${GMT.micro} ${GM.textFaint} font-mono`}>
                  bundle {stats.bundle_date}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── trust bar: onboarded assets ──────────────────────── */}
        <section className="text-center pb-9">
          <div className={`${GMT.label} ${GM.textMuted} mb-3.5`}>
            Disponibles hoy · más activos próximamente
          </div>
          <div className="flex flex-wrap justify-center gap-2.5">
            {ASSETS.map((a) => (
              <div key={a.sym} className={`flex items-center gap-2 px-4 py-2 ${GM.panelSoft}`}>
                <span className={`w-2 h-2 rounded-full ${a.dot}`} aria-hidden />
                <span className={`text-[13px] font-bold ${GM.text} font-mono`}>{a.sym}</span>
                <span className={`${GMT.micro} ${GM.textMuted}`}>{a.cat}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ── features ─────────────────────────────────────────── */}
        <h2 className={`text-center mt-2 mb-5 text-[26px] font-extrabold ${GM.headline} tracking-[-.5px]`}>
          Todo el pipeline cuantitativo, en una terminal
        </h2>
        <section className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3.5 mb-11">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title} className={`${GM.panel} p-5`}>
                <span
                  className="w-11 h-11 rounded-[12px] flex items-center justify-center"
                  style={{ background: `${f.color}1a`, border: `1px solid ${f.color}33` }}
                >
                  <Icon className="w-5 h-5" style={{ color: f.color }} aria-hidden />
                </span>
                <div className={`text-[15.5px] font-bold ${GM.text} mt-3.5 mb-1.5`}>{f.title}</div>
                <div className={`${GMT.meta} leading-[1.6] ${GM.textSec}`}>{f.desc}</div>
              </div>
            );
          })}
        </section>

        {/* ── how it works ─────────────────────────────────────── */}
        <section className={`${GM.panel} p-7 mb-11`}>
          <h2 className={`text-center m-0 mb-6 text-[22px] font-extrabold ${GM.headline}`}>Cómo funciona</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-[18px]">
            {STEPS.map((s) => (
              <div key={s.n}>
                <div className={`text-[26px] font-extrabold ${GM.accent} font-mono opacity-50`}>{s.n}</div>
                <div className={`text-[14.5px] font-bold ${GM.text} mt-2 mb-1`}>{s.title}</div>
                <div className={`${GMT.meta} leading-[1.55] ${GM.textSec}`}>{s.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── live metrics band (forward production only) ──────── */}
        {stats && (
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-3.5 mb-11" data-testid="landing-live-metrics">
            {[
              { v: fmtPct(stats.return_ytd_pct), l: `Retorno ${stats.year ?? ''} · en vivo`, tone: toneOf(stats.return_ytd_pct) },
              { v: stats.max_dd_pct != null ? `${stats.max_dd_pct.toFixed(2)}%` : '—', l: 'Máx. drawdown', tone: 'neutral' as const },
              { v: String(stats.weeks_live ?? '—'), l: 'Semanas en forward', tone: 'accent' as const },
              { v: String(ASSETS.length), l: 'Activos disponibles', tone: 'neutral' as const },
            ].map((m) => (
              <div key={m.l} className={`${GM.panel} text-center py-5 px-3`}>
                <div className={`text-[28px] font-extrabold font-mono tabular-nums ${m.tone === 'neutral' ? GM.headline : GM_TONE_TEXT[m.tone]}`}>
                  {m.v}
                </div>
                <div className={`${GMT.micro} ${GM.textMuted} mt-1`}>{m.l}</div>
              </div>
            ))}
          </section>
        )}

        {/* ── final CTA ────────────────────────────────────────── */}
        <section className="text-center rounded-[20px] bg-gradient-to-br from-[rgba(34,211,238,.12)] to-[rgba(139,92,246,.10)] border border-[rgba(34,211,238,.25)] px-6 py-10 mb-4">
          <h2 className={`m-0 text-[28px] font-extrabold ${GM.headline} tracking-[-.5px]`}>Empieza gratis hoy</h2>
          <p className={`mt-2.5 mb-5 text-[14px] ${GM.textSec}`}>
            Crea tu cuenta y conoce el sistema con el plan Free. Sin tarjeta.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Link
              href="/register"
              data-testid="landing-final-register"
              className={`${GM.ctaPrimary} ${GM.focus} h-12 px-6 inline-flex items-center text-[14.5px]`}
            >
              Crear cuenta gratis
            </Link>
            <Link
              href="/login"
              className={`${GM.ctaGhost} ${GM.focus} h-12 px-5 inline-flex items-center text-[14.5px] font-bold`}
            >
              Iniciar sesión
            </Link>
            <GuestButton testid="landing-cta-guest-final" onError={setGuestError} />
          </div>
          {guestError && (
            <p className={`mt-3 mb-0 ${GMT.meta} ${GM.warn}`} role="alert">{guestError}</p>
          )}
        </section>
       </div>
      </main>

      <PublicFooter />
    </div>
  );
}
