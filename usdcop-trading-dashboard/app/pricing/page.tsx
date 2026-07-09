'use client';

/**
 * /pricing — public tier comparison (CTR-RBAC-001 §B.5).
 *
 * Tiers derive from PLAN_DEFAULTS (the SSOT) — feature flags here NEVER grant access;
 * the middleware + entitlements do. Marketing metric policy (audit I-8/B.2): headline is
 * the PRODUCTION FORWARD number, never the backtest.
 */
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Check, X, Zap, TrendingUp, Bell, Shield } from 'lucide-react';

const TIERS = [
  {
    id: 'free' as const,
    name: 'Free',
    price: 'Gratis',
    tagline: 'Conoce el sistema',
    cta: 'Crear cuenta',
    highlight: false,
    features: [
      { label: 'Análisis semanal (resumen, 7 días de retraso)', included: true },
      { label: 'Forecasting semana anterior (1 activo)', included: true },
      { label: 'Señales en vivo', included: false },
      { label: 'Multi-activo (Oro / BTC)', included: false },
      { label: 'Ejecución automática (SignalBridge)', included: false },
      { label: 'Notificaciones', included: false },
    ],
  },
  {
    id: 'signals' as const,
    name: 'Señales Pro',
    price: 'Suscripción mensual',
    tagline: 'Señales y análisis al día',
    cta: 'Suscribirme',
    highlight: true,
    features: [
      { label: 'Análisis semanal completo, al día', included: true },
      { label: 'Forecasting semana actual', included: true },
      { label: 'Señales en vivo (USD/COP)', included: true },
      { label: 'Add-ons por activo (Oro / BTC)', included: true },
      { label: 'Ejecución automática (SignalBridge)', included: false },
      { label: 'Notificaciones email/telegram', included: true },
    ],
  },
  {
    id: 'auto' as const,
    name: 'Auto Premium',
    price: 'Suscripción mensual',
    tagline: 'Ejecución en tu propio exchange',
    cta: 'Suscribirme',
    highlight: false,
    features: [
      { label: 'Todo lo de Señales Pro', included: true },
      { label: 'SignalBridge propio: tus llaves, tus límites', included: true },
      { label: 'Paper trading primero (4 semanas)', included: true },
      { label: 'Kill switch personal', included: true },
      { label: 'Límites de riesgo configurables (techo del sistema)', included: true },
      { label: 'Multi-activo según add-ons', included: true },
    ],
  },
];

export default function PricingPage() {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);

  const subscribe = async (plan: 'free' | 'signals' | 'auto') => {
    if (plan === 'free') return router.push('/login');
    setBusy(plan);
    try {
      const res = await fetch('/api/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      });
      if (res.status === 401) return router.push(`/login?callbackUrl=/pricing`);
      const data = await res.json();
      if (res.ok && data.checkoutUrl) window.location.href = data.checkoutUrl;
      else alert(data.error ?? 'La pasarela de pagos aún no está configurada.');
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center bg-slate-950 text-slate-100 py-16 px-4">
      <div className="w-full max-w-6xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Planes y precios
          </h1>
          <p className="mt-3 text-slate-400 max-w-2xl mx-auto">
            Señales cuantitativas multi-activo con gestión de riesgo institucional.
            Las métricas publicadas corresponden al desempeño <strong>forward de producción</strong>.
          </p>
        </header>

        <div className="grid md:grid-cols-3 gap-6">
          {TIERS.map((t) => (
            <section
              key={t.id}
              className={`rounded-2xl border p-6 flex flex-col ${
                t.highlight
                  ? 'border-cyan-500/60 bg-cyan-500/5 shadow-lg shadow-cyan-500/10'
                  : 'border-slate-700/60 bg-slate-900/40'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                {t.id === 'auto' ? <Zap className="w-5 h-5 text-rose-400" />
                  : t.id === 'signals' ? <TrendingUp className="w-5 h-5 text-cyan-400" />
                  : <Shield className="w-5 h-5 text-slate-400" />}
                <h2 className="text-xl font-semibold">{t.name}</h2>
              </div>
              <p className="text-sm text-slate-400">{t.tagline}</p>
              <p className="mt-4 text-2xl font-bold">{t.price}</p>

              <ul className="mt-6 space-y-3 flex-1">
                {t.features.map((f) => (
                  <li key={f.label} className="flex items-start gap-2 text-sm">
                    {f.included
                      ? <Check className="w-4 h-4 mt-0.5 text-emerald-400 shrink-0" />
                      : <X className="w-4 h-4 mt-0.5 text-slate-600 shrink-0" />}
                    <span className={f.included ? 'text-slate-200' : 'text-slate-500'}>{f.label}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => subscribe(t.id)}
                disabled={busy === t.id}
                className={`mt-6 rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:opacity-50 ${
                  t.highlight
                    ? 'bg-cyan-600 hover:bg-cyan-500 text-white'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-100'
                }`}
              >
                {busy === t.id ? 'Redirigiendo…' : t.cta}
              </button>
            </section>
          ))}
        </div>

        <footer className="mt-10 text-center text-xs text-slate-500 max-w-3xl mx-auto space-y-2">
          <p className="flex items-center justify-center gap-1"><Bell className="w-3 h-3" />
            El tier Auto opera <strong>paper-first</strong>: 4 semanas simuladas antes de habilitar dinero real.
          </p>
          <p>
            Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos
            pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica
            riesgo de pérdida total del capital.
          </p>
        </footer>
      </div>
    </main>
  );
}
