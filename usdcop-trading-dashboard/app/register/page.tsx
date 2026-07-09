'use client';

/**
 * /register — self-serve account request (admin-approval flow).
 *
 * Product/UX contract (QA-100 §F1++):
 *   1. Applicant fills email + name + password (rules mirror SignalBridge RegisterRequest:
 *      ≥8 chars, upper+lower+digit; name ≥2; email must be a real TLD — reserved
 *      .local/.test/.localhost are rejected server-side by EmailStr).
 *   2. POST /api/execution/auth/register → 202 ACCEPTED, account created PENDING, NO tokens.
 *   3. We show a "pending approval" panel — the account is unusable until an admin approves
 *      on /admin, which emails a temporary password. This screen NEVER logs the user in.
 *
 * Deny-by-default note: this page is public (middleware PUBLIC_PREFIXES + RBAC PAGE_ROUTES),
 * exactly like /login. The register proxy lives under the public /api/execution/auth prefix.
 */
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { CheckCircle2, XCircle, Loader2, MailCheck, ShieldCheck, ArrowLeft } from 'lucide-react';

const RESERVED_TLD = /\.(local|localhost|test|example|invalid)$/i;

function rules(pw: string) {
  return {
    length: pw.length >= 8,
    upper: /[A-Z]/.test(pw),
    lower: /[a-z]/.test(pw),
    digit: /\d/.test(pw),
  };
}

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [status, setStatus] = useState<'idle' | 'submitting' | 'done'>('idle');
  const [error, setError] = useState('');
  // Anti-bot: server-issued signed challenge (see lib/auth/captcha.ts)
  const [captcha, setCaptcha] = useState<{ question: string; token: string } | null>(null);
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const loadCaptcha = async () => {
    setCaptchaAnswer('');
    try { setCaptcha(await (await fetch('/api/captcha')).json()); } catch { setCaptcha(null); }
  };
  useEffect(() => { loadCaptcha(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const pw = useMemo(() => rules(password), [password]);
  const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) && !RESERVED_TLD.test(email.trim());
  const nameOk = name.trim().length >= 2;
  const pwOk = pw.length && pw.upper && pw.lower && pw.digit;
  const matchOk = password.length > 0 && password === confirm;
  const captchaOk = captchaAnswer.trim().length > 0;
  const canSubmit = emailOk && nameOk && pwOk && matchOk && captchaOk && status !== 'submitting';

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    if (!canSubmit) return;
    setStatus('submitting');
    try {
      const res = await fetch('/api/execution/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: email.trim(), password, name: name.trim(),
          captcha_token: captcha?.token, captcha_answer: captchaAnswer.trim(),
        }),
      });
      if (res.status === 202 || res.status === 200 || res.status === 201) {
        setStatus('done');
        return;
      }
      if (res.status === 400) {
        const b = await res.clone().json().catch(() => ({}));
        if (b?.captcha) {
          setError('Verificación incorrecta o vencida. Resuelve la nueva operación.');
          await loadCaptcha();
          setStatus('idle');
          return;
        }
      }
      const body = await res.json().catch(() => ({}));
      if (res.status === 409) {
        setError('Ese correo ya está registrado. Si ya te registraste, espera la aprobación o inicia sesión.');
      } else if (res.status === 422) {
        setError('Datos inválidos. Revisa que el correo sea real y la contraseña cumpla los requisitos.');
      } else {
        setError(body?.message || body?.detail || `No se pudo completar el registro (HTTP ${res.status}).`);
      }
      setStatus('idle');
    } catch {
      setError('No se pudo conectar con el servicio de registro. Inténtalo más tarde.');
      setStatus('idle');
    }
  }

  return (
    <div className="min-h-screen bg-black text-slate-100 flex items-center justify-center px-4 py-10 relative overflow-hidden">
      {/* ambient background — matches /login */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.12]">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/25 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '3s' }} />
      </div>

      <div className="relative w-full max-w-md">
        <Link href="/login" className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-cyan-300 mb-4 transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" /> Volver a iniciar sesión
        </Link>

        {status === 'done' ? (
          <div data-testid="register-pending" className="rounded-2xl border border-cyan-500/30 bg-slate-900/70 backdrop-blur p-8 text-center space-y-4">
            <div className="mx-auto w-14 h-14 rounded-full bg-cyan-500/15 flex items-center justify-center">
              <MailCheck className="w-7 h-7 text-cyan-300" />
            </div>
            <h1 className="text-xl font-bold">Registro recibido</h1>
            <p className="text-sm text-slate-300 leading-relaxed">
              Tu solicitud quedó <span className="font-semibold text-cyan-300">pendiente de aprobación</span>.
              Un administrador la revisará y, al aprobarla, recibirás por correo una
              <span className="font-semibold"> contraseña temporal</span> para tu primer inicio de sesión.
            </p>
            <p className="text-xs text-slate-500">
              Enviamos la notificación a <span className="text-slate-300">{email}</span>.
            </p>
            <Link href="/login" className="inline-block w-full rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black font-semibold py-2.5 transition-colors">
              Ir a iniciar sesión
            </Link>
          </div>
        ) : (
          <form onSubmit={onSubmit} className="rounded-2xl border border-slate-700/60 bg-slate-900/70 backdrop-blur p-8 space-y-5">
            <div className="text-center space-y-1.5">
              <div className="mx-auto w-12 h-12 rounded-xl bg-cyan-500/15 flex items-center justify-center mb-2">
                <ShieldCheck className="w-6 h-6 text-cyan-300" />
              </div>
              <h1 className="text-xl font-bold">Crear cuenta</h1>
              <p className="text-xs text-slate-400">Acceso por aprobación — un administrador revisa cada solicitud.</p>
            </div>

            {error && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300 flex items-start gap-2">
                <XCircle className="w-4 h-4 shrink-0 mt-0.5" /> <span>{error}</span>
              </div>
            )}

            <Field label="Nombre" hint={nameOk || !name ? '' : 'Mínimo 2 caracteres'} bad={!!name && !nameOk}>
              <input
                data-testid="reg-name" name="name" type="text" autoComplete="name" value={name}
                onChange={(e) => setName(e.target.value)} placeholder="Tu nombre"
                className={inputCls(!!name, nameOk)}
              />
            </Field>

            <Field label="Correo" hint={email && !emailOk ? 'Usa un correo real (no .local/.test)' : ''} bad={!!email && !emailOk}>
              <input
                data-testid="reg-email" name="email" type="email" autoComplete="email" value={email}
                onChange={(e) => setEmail(e.target.value)} placeholder="tucorreo@dominio.com"
                className={inputCls(!!email, emailOk)}
              />
            </Field>

            <Field label="Contraseña">
              <input
                data-testid="reg-password" name="password" type="password" autoComplete="new-password" value={password}
                onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
                className={inputCls(!!password, !!pwOk)}
              />
              <ul className="mt-2 grid grid-cols-2 gap-1 text-[11px]">
                <Req ok={pw.length} label="8+ caracteres" />
                <Req ok={pw.upper} label="Mayúscula" />
                <Req ok={pw.lower} label="Minúscula" />
                <Req ok={pw.digit} label="Número" />
              </ul>
            </Field>

            <Field label="Confirmar contraseña" hint={confirm && !matchOk ? 'No coincide' : ''} bad={!!confirm && !matchOk}>
              <input
                data-testid="reg-confirm" name="confirm" type="password" autoComplete="new-password" value={confirm}
                onChange={(e) => setConfirm(e.target.value)} placeholder="••••••••"
                className={inputCls(!!confirm, matchOk)}
              />
            </Field>

            <Field label={`Verificación: ${captcha?.question ?? 'cargando…'}`} hint="" bad={false}>
              <div className="flex gap-2">
                <input
                  data-testid="reg-captcha" name="captcha" type="text" inputMode="numeric"
                  value={captchaAnswer} onChange={(e) => setCaptchaAnswer(e.target.value)}
                  placeholder="respuesta" className={inputCls(!!captchaAnswer, captchaOk)}
                />
                <button
                  type="button" onClick={loadCaptcha} title="Nueva operación"
                  className="shrink-0 rounded-lg border border-slate-700 px-3 text-slate-300 hover:bg-slate-800"
                >↻</button>
              </div>
            </Field>

            <button
              data-testid="reg-submit" type="submit" disabled={!canSubmit}
              className="w-full rounded-lg bg-cyan-500 enabled:hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-black font-semibold py-2.5 transition-colors inline-flex items-center justify-center gap-2"
            >
              {status === 'submitting' ? <><Loader2 className="w-4 h-4 animate-spin" /> Enviando…</> : 'Solicitar acceso'}
            </button>

            <p className="text-center text-xs text-slate-500">
              ¿Ya tienes cuenta?{' '}
              <Link href="/login" className="text-cyan-300 hover:underline">Inicia sesión</Link>
            </p>
          </form>
        )}
      </div>
    </div>
  );
}

function inputCls(touched: boolean, ok: boolean) {
  const base = 'w-full rounded-lg bg-slate-950/60 border px-3 py-2.5 text-sm outline-none transition-colors placeholder:text-slate-600 focus:border-cyan-500/70';
  const border = !touched ? 'border-slate-700/70' : ok ? 'border-emerald-500/50' : 'border-red-500/50';
  return `${base} ${border}`;
}

function Field({ label, hint, bad, children }: { label: string; hint?: string; bad?: boolean; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-300">{label}</span>
        {hint && <span className={`text-[11px] ${bad ? 'text-red-400' : 'text-slate-500'}`}>{hint}</span>}
      </div>
      {children}
    </label>
  );
}

function Req({ ok, label }: { ok: boolean; label: string }) {
  return (
    <li className={`flex items-center gap-1 ${ok ? 'text-emerald-400' : 'text-slate-500'}`}>
      <CheckCircle2 className={`w-3 h-3 ${ok ? 'opacity-100' : 'opacity-30'}`} /> {label}
    </li>
  );
}
