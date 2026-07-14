/* ARCHIVED pre-GlobalMarkets UI (CTR-GM-UI-001) — verbatim git-HEAD copy of app/reset-password/page.tsx.
   Reference only, mounted at /legacy/* (admin:all). Do not evolve; the GM view is the live one. */
'use client';

/**
 * /reset-password — forced consumption of the admin-issued temporary password.
 *
 * Journey (QA-100 §F1++): the temp-pw login lands here (never /hub). The temp session
 * bearer was stashed in sessionStorage by /login; we send it to SignalBridge along with
 * the temporary password + a new one. On success we wipe every temp credential and send
 * the user back to /login to sign in cleanly with the new password.
 */
import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { KeyRound, Loader2, XCircle, CheckCircle2 } from 'lucide-react';

function rules(pw: string) {
  return { length: pw.length >= 8, upper: /[A-Z]/.test(pw), lower: /[a-z]/.test(pw), digit: /\d/.test(pw) };
}

export default function ResetPasswordPage() {
  const router = useRouter();
  const [bearer, setBearer] = useState<string | null>(null);
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [status, setStatus] = useState<'idle' | 'submitting' | 'done'>('idle');
  const [error, setError] = useState('');

  useEffect(() => {
    const t = sessionStorage.getItem('reset-token') || localStorage.getItem('auth-token');
    setBearer(t);
    const tmp = sessionStorage.getItem('reset-temp-pw');
    if (tmp) setCurrent(tmp);
  }, []);

  const pw = useMemo(() => rules(next), [next]);
  const pwOk = pw.length && pw.upper && pw.lower && pw.digit;
  const matchOk = next.length > 0 && next === confirm;
  const differs = next !== current;
  const canSubmit = !!bearer && current.length >= 8 && !!pwOk && matchOk && differs && status !== 'submitting';

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    if (!canSubmit) return;
    setStatus('submitting');
    try {
      const res = await fetch('/api/execution/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${bearer}` },
        body: JSON.stringify({ current_password: current, new_password: next }),
      });
      if (res.ok) {
        // Wipe every temporary credential — force a clean re-login with the new password.
        ['reset-token', 'reset-temp-pw'].forEach((k) => sessionStorage.removeItem(k));
        ['auth-token', 'refresh-token', 'isAuthenticated', 'username'].forEach((k) => localStorage.removeItem(k));
        setStatus('done');
        setTimeout(() => router.push('/login?reset=1'), 1400);
        return;
      }
      const body = await res.json().catch(() => ({}));
      setError(body?.message || body?.detail || `No se pudo actualizar la contraseña (HTTP ${res.status}).`);
      setStatus('idle');
    } catch {
      setError('No se pudo conectar con el servicio. Inténtalo más tarde.');
      setStatus('idle');
    }
  }

  return (
    <div className="min-h-screen bg-black text-slate-100 flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-md">
        {status === 'done' ? (
          <div data-testid="reset-done" className="rounded-2xl border border-emerald-500/30 bg-slate-900/70 p-8 text-center space-y-3">
            <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto" />
            <h1 className="text-xl font-bold">Contraseña actualizada</h1>
            <p className="text-sm text-slate-300">Redirigiendo al inicio de sesión…</p>
          </div>
        ) : (
          <form onSubmit={onSubmit} className="rounded-2xl border border-slate-700/60 bg-slate-900/70 p-8 space-y-5">
            <div className="text-center space-y-1.5">
              <div className="mx-auto w-12 h-12 rounded-xl bg-amber-500/15 flex items-center justify-center mb-2">
                <KeyRound className="w-6 h-6 text-amber-300" />
              </div>
              <h1 className="text-xl font-bold">Cambia tu contraseña temporal</h1>
              <p className="text-xs text-slate-400">Tu acceso fue aprobado. Define una contraseña propia para continuar.</p>
            </div>

            {!bearer && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-300">
                No encontramos una sesión temporal. Inicia sesión con la contraseña temporal del correo.
              </div>
            )}
            {error && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300 flex items-start gap-2">
                <XCircle className="w-4 h-4 shrink-0 mt-0.5" /> <span>{error}</span>
              </div>
            )}

            <label className="block">
              <span className="text-xs font-medium text-slate-300 block mb-1.5">Contraseña temporal</span>
              <input data-testid="rst-current" type="password" value={current} onChange={(e) => setCurrent(e.target.value)}
                placeholder="La del correo" className="w-full rounded-lg bg-slate-950/60 border border-slate-700/70 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70" />
            </label>

            <label className="block">
              <span className="text-xs font-medium text-slate-300 block mb-1.5">Nueva contraseña</span>
              <input data-testid="rst-new" type="password" value={next} onChange={(e) => setNext(e.target.value)}
                placeholder="••••••••" className="w-full rounded-lg bg-slate-950/60 border border-slate-700/70 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70" />
              <ul className="mt-2 grid grid-cols-2 gap-1 text-[11px]">
                {([['8+ caracteres', pw.length], ['Mayúscula', pw.upper], ['Minúscula', pw.lower], ['Número', pw.digit]] as const).map(([l, ok]) => (
                  <li key={l} className={`flex items-center gap-1 ${ok ? 'text-emerald-400' : 'text-slate-500'}`}>
                    <CheckCircle2 className={`w-3 h-3 ${ok ? 'opacity-100' : 'opacity-30'}`} /> {l}
                  </li>
                ))}
              </ul>
              {next && !differs && <p className="text-[11px] text-red-400 mt-1">Debe ser distinta a la temporal.</p>}
            </label>

            <label className="block">
              <span className="text-xs font-medium text-slate-300 block mb-1.5">Confirmar</span>
              <input data-testid="rst-confirm" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
                placeholder="••••••••" className="w-full rounded-lg bg-slate-950/60 border border-slate-700/70 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70" />
              {confirm && !matchOk && <p className="text-[11px] text-red-400 mt-1">No coincide.</p>}
            </label>

            <button data-testid="rst-submit" type="submit" disabled={!canSubmit}
              className="w-full rounded-lg bg-cyan-500 enabled:hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-black font-semibold py-2.5 inline-flex items-center justify-center gap-2">
              {status === 'submitting' ? <><Loader2 className="w-4 h-4 animate-spin" /> Guardando…</> : 'Guardar y continuar'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
