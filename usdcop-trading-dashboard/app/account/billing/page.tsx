'use client';

/**
 * /account/billing — current plan + upgrade entry point (CTR-RBAC-001 R6).
 * Reads the user's effective entitlements from the server (the DB is the truth; the
 * session only names the user). Payments happen exclusively via the provider checkout.
 */
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CreditCard, Calendar, Layers, ArrowRight, ShieldCheck } from 'lucide-react';

interface PlanView {
  plan: string;
  assets: string[];
  signals_realtime: boolean;
  expires_at: string | null;
  execution?: { enabled: boolean; mode: string };
}

export default function BillingPage() {
  const router = useRouter();
  const [plan, setPlan] = useState<PlanView | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Session cookie rides along; the endpoint resolves entitlements server-side.
    fetch('/api/billing/me')
      .then((r) => (r.ok ? r.json() : null))
      .then(setPlan)
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="min-h-screen flex flex-col items-center bg-slate-950 text-slate-100 py-16 px-4">
      <div className="w-full max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <CreditCard className="w-6 h-6 text-cyan-400" /> Mi suscripción
        </h1>

        {loading ? (
          <p className="mt-6 text-slate-400">Cargando…</p>
        ) : (
          <section className="mt-6 rounded-2xl border border-slate-700/60 bg-slate-900/40 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-sm">Plan actual</span>
              <span className="text-lg font-semibold uppercase tracking-wide text-cyan-300">
                {plan?.plan ?? 'free'}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400 flex items-center gap-1"><Layers className="w-4 h-4" /> Activos</span>
              <span>{(plan?.assets ?? ['usdcop']).join(' · ').toUpperCase()}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400 flex items-center gap-1"><ShieldCheck className="w-4 h-4" /> Señales en vivo</span>
              <span>{plan?.signals_realtime ? 'Sí' : 'No (contenido con retraso)'}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400 flex items-center gap-1"><Calendar className="w-4 h-4" /> Vence</span>
              <span>
                {plan?.expires_at
                  ? new Date(plan.expires_at).toLocaleDateString('es-CO', { day: '2-digit', month: 'short', year: 'numeric' })
                  : 'Sin vencimiento'}
              </span>
            </div>
            {plan?.execution?.enabled && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Ejecución (SignalBridge)</span>
                <span className="uppercase">{plan.execution.mode}</span>
              </div>
            )}
          </section>
        )}

        <button
          onClick={() => router.push('/pricing')}
          className="mt-6 w-full rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold px-4 py-2.5 text-sm inline-flex items-center justify-center gap-2 transition"
        >
          Ver planes / mejorar <ArrowRight className="w-4 h-4" />
        </button>
        <p className="mt-4 text-xs text-slate-500 text-center">
          Los pagos se procesan por la pasarela; tu plan se actualiza automáticamente al
          confirmarse (webhook firma-verificada). Nunca gestionamos tu tarjeta directamente.
        </p>
      </div>
    </main>
  );
}
