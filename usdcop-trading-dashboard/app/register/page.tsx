'use client';

/**
 * /register — self-serve account request (admin-approval flow), GlobalMarkets skin
 * (CTR-GM-UI-001, prototype Var B lines 1496-1524). Re-skin only — logic preserved
 * verbatim from components/legacy/RegisterLegacy.tsx.
 *
 * Product/UX contract (QA-100 §F1++):
 *   1. Applicant fills email + name + password (rules mirror SignalBridge RegisterRequest:
 *      ≥8 chars, upper+lower+digit; name ≥2; email must be a real TLD — reserved
 *      .local/.test/.localhost are rejected server-side by EmailStr).
 *   2. POST /api/execution/auth/register → 202 ACCEPTED, account created PENDING, NO tokens.
 *   3. We show a "pending approval" panel — the account is unusable until an admin approves
 *      on /admin, which emails a temporary password. This screen NEVER logs the user in.
 *
 * E2E contract (scripts/registration-qa.mjs): data-testids reg-name, reg-email,
 * reg-password, reg-confirm, reg-captcha (question "¿Cuánto es A + B?"), reg-submit,
 * and panel register-pending after success.
 *
 * Deny-by-default note: this page is public (middleware PUBLIC_PREFIXES + RBAC PAGE_ROUTES),
 * exactly like /login. The register proxy lives under the public /api/execution/auth prefix.
 */
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, CheckCircle2, Loader2, MailCheck, RefreshCw, ShieldCheck, XCircle } from 'lucide-react';

import { PublicHeader, PublicFooter } from '@/components/gm/views/PublicChrome';
import { GM, GMT } from '@/lib/ui/gm-tokens';

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
    <div className={`min-h-screen flex flex-col ${GM.page}`}>
      <PublicHeader />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-[420px]">
          <Link
            href="/login"
            className={`inline-flex items-center gap-1.5 ${GMT.meta} ${GM.textSec} hover:text-[var(--gm-accent)] mb-4 transition-colors ${GM.focus}`}
          >
            <ArrowLeft className="w-3.5 h-3.5" aria-hidden /> Volver a iniciar sesión
          </Link>

          {status === 'done' ? (
            <div
              data-testid="register-pending"
              className={`${GM.panel} p-8 text-center space-y-4 shadow-[0_20px_60px_rgba(0,0,0,.4)]`}
            >
              <div className={`mx-auto w-14 h-14 rounded-2xl ${GM.accentBadge} flex items-center justify-center`}>
                <MailCheck className="w-7 h-7" aria-hidden />
              </div>
              <h1 className={`text-[21px] font-extrabold ${GM.headline}`}>Registro recibido</h1>
              <p className={`${GMT.body} ${GM.textStrong} leading-relaxed`}>
                Tu solicitud quedó <span className={`font-semibold ${GM.accent}`}>pendiente de aprobación</span>.
                Un administrador la revisará y, al aprobarla, recibirás por correo una
                <span className={`font-semibold ${GM.text}`}> contraseña temporal</span> para tu primer inicio de sesión.
              </p>
              <p className={`${GMT.micro} ${GM.textMuted}`}>
                Enviamos la notificación a <span className={GM.textStrong}>{email}</span>.
              </p>
              <Link
                href="/login"
                className={`${GM.ctaPrimary} ${GM.focus} inline-flex items-center justify-center w-full h-[46px] text-[14px]`}
              >
                Ir a iniciar sesión
              </Link>
            </div>
          ) : (
            <form onSubmit={onSubmit} className={`${GM.panel} p-8 space-y-5 shadow-[0_20px_60px_rgba(0,0,0,.4)]`}>
              <div className="flex flex-col items-center text-center space-y-3">
                <span
                  className={`w-12 h-12 rounded-[13px] ${GM.brandGradient} flex items-center justify-center shadow-[0_8px_24px_rgba(34,211,238,.3)]`}
                  aria-hidden
                >
                  <ShieldCheck className="w-6 h-6 text-white" />
                </span>
                <div className="space-y-1.5">
                  <h1 className={`text-[21px] font-extrabold ${GM.headline}`}>Crear cuenta</h1>
                  <p className={`${GMT.meta} ${GM.textSec}`}>
                    Acceso por aprobación — un administrador revisa cada solicitud.
                  </p>
                </div>
              </div>

              {error && (
                <div className={`${GM.negBadge} rounded-xl p-3 flex items-start gap-2 ${GMT.meta}`} role="alert">
                  <XCircle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden /> <span>{error}</span>
                </div>
              )}

              <Field label="Nombre" hint={nameOk || !name ? '' : 'Mínimo 2 caracteres'} bad={!!name && !nameOk}>
                <input
                  data-testid="reg-name" name="name" type="text" autoComplete="name" value={name}
                  onChange={(e) => setName(e.target.value)} placeholder="Tu nombre"
                  className={`${GM.input} ${GM.focus} w-full h-11`}
                />
              </Field>

              <Field label="Correo" hint={email && !emailOk ? 'Usa un correo real (no .local/.test)' : ''} bad={!!email && !emailOk}>
                <input
                  data-testid="reg-email" name="email" type="email" autoComplete="email" value={email}
                  onChange={(e) => setEmail(e.target.value)} placeholder="tucorreo@dominio.com"
                  className={`${GM.input} ${GM.focus} w-full h-11`}
                />
              </Field>

              <Field label="Contraseña">
                <input
                  data-testid="reg-password" name="password" type="password" autoComplete="new-password" value={password}
                  onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
                  className={`${GM.input} ${GM.focus} w-full h-11`}
                />
                <ul className={`mt-2 grid grid-cols-2 gap-1 ${GMT.micro}`}>
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
                  className={`${GM.input} ${GM.focus} w-full h-11`}
                />
              </Field>

              <Field
                label={(
                  <>Verificación{' '}
                    <span className={`${GMT.meta} ${GM.textStrong} normal-case tracking-normal font-normal`}>
                      {captcha?.question ?? 'cargando…'}
                    </span>
                  </>
                )}
                hint="" bad={false}
              >
                <div className="flex gap-2">
                  <input
                    data-testid="reg-captcha" name="captcha" type="text" inputMode="numeric"
                    value={captchaAnswer} onChange={(e) => setCaptchaAnswer(e.target.value)}
                    placeholder="respuesta" className={`${GM.input} ${GM.focus} w-full h-11`}
                  />
                  <button
                    type="button" onClick={loadCaptcha} title="Nueva operación"
                    className={`${GM.ctaGhost} ${GM.focus} shrink-0 h-11 px-3.5 flex items-center justify-center`}
                    aria-label="Generar nueva operación"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </Field>

              <button
                data-testid="reg-submit" type="submit" disabled={!canSubmit}
                className={`${GM.ctaPrimary} ${GM.focus} w-full h-[46px] text-[14px] shadow-[0_8px_24px_rgba(34,211,238,.25)] disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2`}
              >
                {status === 'submitting'
                  ? <><Loader2 className="w-4 h-4 animate-spin" aria-hidden /> Enviando…</>
                  : 'Solicitar acceso'}
              </button>

              <p className={`text-center ${GMT.meta} ${GM.textMuted}`}>
                ¿Ya tienes cuenta?{' '}
                <Link href="/login" className={`${GM.accent} font-semibold hover:underline underline-offset-2 ${GM.focus}`}>
                  Inicia sesión
                </Link>
              </p>
            </form>
          )}
        </div>
      </main>

      <PublicFooter />
    </div>
  );
}

function Field({ label, hint, bad, children }: { label: React.ReactNode; hint?: string; bad?: boolean; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="flex items-center justify-between mb-1.5 gap-2">
        <span className={`${GMT.label} ${GM.textSec}`}>{label}</span>
        {hint && <span className={`${GMT.micro} ${bad ? GM.neg : GM.textMuted}`}>{hint}</span>}
      </div>
      {children}
    </label>
  );
}

function Req({ ok, label }: { ok: boolean; label: string }) {
  return (
    <li className={`flex items-center gap-1 ${ok ? GM.pos : GM.textMuted}`}>
      <CheckCircle2 className={`w-3 h-3 ${ok ? 'opacity-100' : 'opacity-30'}`} aria-hidden /> {label}
    </li>
  );
}
