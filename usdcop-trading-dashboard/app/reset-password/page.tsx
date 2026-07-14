'use client';

/**
 * /reset-password — forced consumption of the admin-issued temporary password,
 * GlobalMarkets skin (CTR-GM-UI-001). Re-skin only — logic preserved verbatim from
 * components/legacy/ResetPasswordLegacy.tsx.
 *
 * Journey (QA-100 §F1++): the temp-pw login lands here (never /hub). The temp session
 * bearer was stashed in sessionStorage by /login; we send it to SignalBridge along with
 * the temporary password + a new one. On success we wipe every temp credential and send
 * the user back to /login to sign in cleanly with the new password (/login?reset=1).
 *
 * E2E contract (scripts/registration-qa.mjs): data-testids rst-current (pre-filled),
 * rst-new, rst-confirm, rst-submit, and panel reset-done after success.
 */
import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CheckCircle2, KeyRound, Loader2, XCircle } from 'lucide-react';

import { PublicHeader, PublicFooter } from '@/components/gm/views/PublicChrome';
import { GM, GMT } from '@/lib/ui/gm-tokens';

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
    <div className={`min-h-screen flex flex-col ${GM.page}`}>
      <PublicHeader />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-[420px]">
          {status === 'done' ? (
            <div
              data-testid="reset-done"
              className={`${GM.panel} p-8 text-center space-y-3 shadow-[0_20px_60px_rgba(0,0,0,.4)]`}
            >
              <div className={`mx-auto w-14 h-14 rounded-2xl ${GM.posBadge} flex items-center justify-center`}>
                <CheckCircle2 className="w-7 h-7" aria-hidden />
              </div>
              <h1 className={`text-[21px] font-extrabold ${GM.headline}`}>Contraseña actualizada</h1>
              <p className={`${GMT.body} ${GM.textStrong}`}>Redirigiendo al inicio de sesión…</p>
            </div>
          ) : (
            <form onSubmit={onSubmit} className={`${GM.panel} p-8 space-y-5 shadow-[0_20px_60px_rgba(0,0,0,.4)]`}>
              <div className="flex flex-col items-center text-center space-y-3">
                <span
                  className={`w-12 h-12 rounded-[13px] ${GM.warnBadge} flex items-center justify-center`}
                  aria-hidden
                >
                  <KeyRound className="w-6 h-6" />
                </span>
                <div className="space-y-1.5">
                  <h1 className={`text-[21px] font-extrabold ${GM.headline}`}>Cambia tu contraseña temporal</h1>
                  <p className={`${GMT.meta} ${GM.textSec}`}>
                    Tu acceso fue aprobado. Define una contraseña propia para continuar.
                  </p>
                </div>
              </div>

              {!bearer && (
                <div className={`${GM.warnBadge} rounded-xl p-3 ${GMT.meta}`}>
                  No encontramos una sesión temporal. Inicia sesión con la contraseña temporal del correo.
                </div>
              )}
              {error && (
                <div className={`${GM.negBadge} rounded-xl p-3 flex items-start gap-2 ${GMT.meta}`} role="alert">
                  <XCircle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden /> <span>{error}</span>
                </div>
              )}

              <label className="block">
                <span className={`${GMT.label} ${GM.textSec} block mb-1.5`}>Contraseña temporal</span>
                <input
                  data-testid="rst-current" type="password" value={current}
                  onChange={(e) => setCurrent(e.target.value)}
                  placeholder="La del correo" className={`${GM.input} ${GM.focus} w-full h-11`}
                />
              </label>

              <label className="block">
                <span className={`${GMT.label} ${GM.textSec} block mb-1.5`}>Nueva contraseña</span>
                <input
                  data-testid="rst-new" type="password" value={next}
                  onChange={(e) => setNext(e.target.value)}
                  placeholder="••••••••" className={`${GM.input} ${GM.focus} w-full h-11`}
                />
                <ul className={`mt-2 grid grid-cols-2 gap-1 ${GMT.micro}`}>
                  {([['8+ caracteres', pw.length], ['Mayúscula', pw.upper], ['Minúscula', pw.lower], ['Número', pw.digit]] as const).map(([l, ok]) => (
                    <li key={l} className={`flex items-center gap-1 ${ok ? GM.pos : GM.textMuted}`}>
                      <CheckCircle2 className={`w-3 h-3 ${ok ? 'opacity-100' : 'opacity-30'}`} aria-hidden /> {l}
                    </li>
                  ))}
                </ul>
                {next && !differs && <p className={`${GMT.micro} ${GM.neg} mt-1`}>Debe ser distinta a la temporal.</p>}
              </label>

              <label className="block">
                <span className={`${GMT.label} ${GM.textSec} block mb-1.5`}>Confirmar</span>
                <input
                  data-testid="rst-confirm" type="password" value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="••••••••" className={`${GM.input} ${GM.focus} w-full h-11`}
                />
                {confirm && !matchOk && <p className={`${GMT.micro} ${GM.neg} mt-1`}>No coincide.</p>}
              </label>

              <button
                data-testid="rst-submit" type="submit" disabled={!canSubmit}
                className={`${GM.ctaPrimary} ${GM.focus} w-full h-[46px] text-[14px] shadow-[0_8px_24px_rgba(34,211,238,.25)] disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2`}
              >
                {status === 'submitting'
                  ? <><Loader2 className="w-4 h-4 animate-spin" aria-hidden /> Guardando…</>
                  : 'Guardar y continuar'}
              </button>
            </form>
          )}
        </div>
      </main>

      <PublicFooter />
    </div>
  );
}
