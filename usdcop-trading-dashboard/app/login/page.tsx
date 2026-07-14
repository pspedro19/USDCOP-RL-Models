'use client';

/**
 * /login — GlobalMarkets Terminal skin (CTR-GM-UI-001, prototype Var B lines 1527-1552).
 *
 * Re-skin ONLY — the auth flow is preserved verbatim from the legacy page
 * (components/legacy/LoginLegacy.tsx): SignalBridge proxy login → NextAuth cookie is
 * minted server-side by the proxy route → tokens to localStorage → must_reset_password
 * forces /reset-password → 429 lockout copy → NextAuth credentials fallback.
 *
 * E2E contract (scripts/registration-qa.mjs — DO NOT break):
 *   - username input matches `input[name="username"], input[type="text"]` and is the
 *     FIRST text input in the DOM (the captcha input comes after it).
 *   - single `input[type="password"]`.
 *   - captcha question text "¿Cuánto es A + B?" visible + `input[placeholder="respuesta"]`.
 *   - `button[type="submit"]` submits the form; link `data-testid="login-to-register"`.
 *   - `?callbackUrl=` / `?next=` destinations and `?reset=1` notice are respected.
 */
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signIn } from 'next-auth/react';
import { CheckCircle2, Eye, EyeOff, Loader2, LogIn, RefreshCw, ShieldCheck, Sparkles, UserRound, XCircle } from 'lucide-react';

import { PublicHeader, PublicFooter } from '@/components/gm/views/PublicChrome';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

// CTA invitado (prototipo Var B: login('free','Invitado'), l. 3151-3160).
const LOGIN_GUEST_DICT = defineGmDict({
  es: {
    guestCta: 'Explorar como invitado',
    guestBusy: 'Entrando…',
    guestError: 'El acceso de invitado no está disponible ahora.',
  },
  en: {
    guestCta: 'Explore as guest',
    guestBusy: 'Signing in…',
    guestError: 'Guest access is unavailable right now.',
  },
});

export default function LoginPage() {
  const router = useRouter();
  const tGuest = useGmT(LOGIN_GUEST_DICT);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [guestLoading, setGuestLoading] = useState(false);
  // `?reset=1` — the reset-password page just completed; invite a clean sign-in.
  const [justReset, setJustReset] = useState(false);

  // Anti-bot verification: server-issued signed challenge (lib/auth/captcha.ts).
  const [captcha, setCaptcha] = useState<{ question: string; token: string } | null>(null);
  const [captchaAnswer, setCaptchaAnswer] = useState('');
  const loadCaptcha = async () => {
    setCaptchaAnswer('');
    try { setCaptcha(await (await fetch('/api/captcha')).json()); } catch { setCaptcha(null); }
  };
  useEffect(() => { loadCaptcha(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setJustReset(new URLSearchParams(window.location.search).get('reset') === '1');
    }
  }, []);

  // Prevent Web3 wallet injections on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const blockList = ['ethereum', 'web3', 'tronWeb', 'solana', 'phantom'];
      blockList.forEach(prop => {
        try {
          const descriptor = Object.getOwnPropertyDescriptor(window, prop);
          if (descriptor && descriptor.configurable) {
            delete (window as any)[prop];
          } else if ((window as any)[prop]) {
            const wallet = (window as any)[prop];
            if (wallet && typeof wallet === 'object') {
              if (wallet.autoRefreshOnNetworkChange !== undefined) {
                wallet.autoRefreshOnNetworkChange = false;
              }
              if (wallet.isMetaMask !== undefined) {
                wallet.isMetaMask = false;
              }
              if (wallet.request) {
                wallet.request = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
              if (wallet.enable) {
                wallet.enable = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
              if (wallet.connect) {
                wallet.connect = () => Promise.reject(new Error('Web3 disabled on login page'));
              }
            }
          }
        } catch (e) {
          console.log(`Could not modify ${prop}:`, (e as Error).message);
        }
      });
    }
  }, []);

  // Post-login destination: middleware/legacy paths emit ?callbackUrl=, the GM api
  // client (lib/api/gm-client.ts) emits ?next=. Both are respected; /hub is the default.
  const loginDestination = () => {
    const params = new URLSearchParams(window.location.search);
    return params.get('callbackUrl') || params.get('next') || '/hub';
  };

  // Sesión demo rol free (guest@demo.local, is_test) — el endpoint hace el login
  // SERVER-SIDE con las credenciales GUEST_BOOTSTRAP_* y mintea las mismas cookies
  // que el proxy de login (next-auth.session-token + sb-token). Sin captcha: no
  // viajan credenciales del usuario (ver app/api/auth/guest/route.ts).
  const handleGuest = async () => {
    setError('');
    setGuestLoading(true);
    try {
      const res = await fetch('/api/auth/guest', { method: 'POST' });
      if (res.ok) {
        window.location.href = loginDestination();
        return;
      }
      setError(tGuest('guestError'));
    } catch {
      setError(tGuest('guestError'));
    }
    setGuestLoading(false);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    // Identity is an email; accept a bare username by mapping it to the
    // SignalBridge tenant domain. No hardcoded credentials, no password logging.
    const email = username.includes('@')
      ? username.trim()
      : `${username.trim()}@trading.usdcop.com`;

    try {
      // Primary auth: SignalBridge backend (SSOT) via the dashboard proxy.
      const sbLoginResponse = await fetch('/api/execution/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email, password,
          captcha_token: captcha?.token, captcha_answer: captchaAnswer.trim(),
        }),
      });

      if (sbLoginResponse.status === 400) {
        const b = await sbLoginResponse.clone().json().catch(() => ({} as { captcha?: boolean }));
        if (b?.captcha) {
          setError('Verificación incorrecta o vencida. Resuelve la nueva operación.');
          await loadCaptcha();
          setIsLoading(false);
          return;
        }
      }

      if (sbLoginResponse.ok) {
        const sbData = await sbLoginResponse.json();
        // Forced first-login reset: account still holds the admin-issued temp password.
        // Route to /reset-password with the temp bearer; never grant app access yet.
        if (sbData.must_reset_password) {
          if (sbData.access_token) sessionStorage.setItem('reset-token', sbData.access_token);
          sessionStorage.setItem('reset-temp-pw', password);
          window.location.href = '/reset-password';
          return;
        }
        if (sbData.access_token) {
          localStorage.setItem('auth-token', sbData.access_token);
          if (sbData.refresh_token) {
            localStorage.setItem('refresh-token', sbData.refresh_token);
          }
        }
        localStorage.setItem('isAuthenticated', 'true');
        sessionStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('username', username);
        sessionStorage.setItem('username', username);

        const callbackUrl = loginDestination();
        setTimeout(() => {
          window.location.href = callbackUrl;
        }, 100);
        return;
      }

      if (sbLoginResponse.status === 429) {
        setError('Demasiados intentos fallidos. Espera unos minutos e inténtalo de nuevo.');
        setIsLoading(false);
        return;
      }

      // Secondary path: NextAuth credentials provider (if configured).
      const result = await signIn('credentials', {
        identifier: email,
        password,
        redirect: false,
      });

      if (result?.ok) {
        localStorage.setItem('isAuthenticated', 'true');
        sessionStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('username', username);
        sessionStorage.setItem('username', username);

        const callbackUrl = loginDestination();
        router.push(callbackUrl);
        return;
      }

      setError('Credenciales inválidas.');
      setIsLoading(false);
    } catch {
      setError('No se pudo conectar con el servicio de autenticación. Inténtalo más tarde.');
      setIsLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex flex-col ${GM.page}`}>
      <PublicHeader />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-[420px]">
          <form
            onSubmit={handleLogin}
            className={`${GM.panel} p-8 space-y-5 shadow-[0_20px_60px_rgba(0,0,0,.4)]`}
          >
            {/* Brand + title */}
            <div className="flex flex-col items-center text-center space-y-3">
              <span
                className={`w-[52px] h-[52px] rounded-[14px] ${GM.brandGradient} flex items-center justify-center shadow-[0_8px_24px_rgba(34,211,238,.3)]`}
                aria-hidden
              >
                <Sparkles className="w-6 h-6 text-white" />
              </span>
              <div className="space-y-1.5">
                <h1 className={`text-[22px] font-extrabold ${GM.headline}`}>Inicia sesión</h1>
                <p className={`${GMT.meta} ${GM.textSec}`}>
                  Terminal de trading cuantitativo · acceso para cuentas aprobadas
                </p>
              </div>
            </div>

            {/* Post-reset notice (?reset=1) */}
            {justReset && !error && (
              <div className={`${GM.posBadge} rounded-xl p-3 flex items-start gap-2 ${GMT.meta}`}>
                <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" aria-hidden />
                <span>Contraseña actualizada. Inicia sesión con tu nueva contraseña.</span>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className={`${GM.negBadge} rounded-xl p-3 flex items-start gap-2 ${GMT.meta}`} role="alert">
                <XCircle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden />
                <span>{error}</span>
              </div>
            )}

            {/* Usuario / email — FIRST text input in the DOM (E2E selector contract) */}
            <label className="block">
              <span className={`${GMT.label} ${GM.textSec} block mb-1.5`}>Usuario o correo</span>
              <input
                type="text"
                name="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className={`${GM.input} ${GM.focus} w-full h-11`}
                placeholder="usuario o tu@correo.com"
                required
                autoFocus
                autoComplete="username"
              />
            </label>

            {/* Contraseña */}
            <label className="block">
              <span className={`${GMT.label} ${GM.textSec} block mb-1.5`}>Contraseña</span>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={`${GM.input} ${GM.focus} w-full h-11 pr-11`}
                  placeholder="••••••••"
                  required
                  minLength={8}
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={`absolute right-3 top-1/2 -translate-y-1/2 ${GM.textMuted} hover:text-[var(--gm-text)] ${GM.focus}`}
                  aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </label>

            {/* Verificación anti-bot (server-signed challenge) */}
            <label className="block">
              <span className={`${GMT.label} ${GM.textSec} mb-1.5 flex items-center gap-1.5`}>
                <ShieldCheck className={`w-3.5 h-3.5 ${GM.accent}`} aria-hidden />
                Verificación
                <span className={`${GMT.meta} ${GM.textStrong} normal-case tracking-normal font-normal`}>
                  {captcha?.question ?? 'cargando…'}
                </span>
              </span>
              <div className="flex gap-2">
                <input
                  type="text"
                  inputMode="numeric"
                  value={captchaAnswer}
                  onChange={(e) => setCaptchaAnswer(e.target.value)}
                  className={`${GM.input} ${GM.focus} w-full h-11`}
                  placeholder="respuesta"
                  required
                  aria-label="Respuesta de verificación"
                />
                <button
                  type="button"
                  onClick={loadCaptcha}
                  title="Nueva operación"
                  className={`${GM.ctaGhost} ${GM.focus} shrink-0 h-11 px-3.5 flex items-center justify-center`}
                  aria-label="Generar nueva operación"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </label>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading || !username || !password}
              className={`${GM.ctaPrimary} ${GM.focus} w-full h-[46px] text-[14px] shadow-[0_8px_24px_rgba(34,211,238,.25)] disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2`}
            >
              {isLoading ? (
                <><Loader2 className="w-4 h-4 animate-spin" aria-hidden /> Verificando acceso…</>
              ) : (
                <><LogIn className="w-4 h-4" aria-hidden /> Entrar a la terminal</>
              )}
            </button>

            {/* Crear cuenta */}
            <div className="border-t border-[rgba(148,163,184,.12)] pt-4 text-center">
              <p className={`${GMT.meta} ${GM.textMuted}`}>
                ¿No tienes cuenta?{' '}
                <a
                  href="/register"
                  data-testid="login-to-register"
                  className={`${GM.accent} font-semibold hover:underline underline-offset-2 ${GM.focus}`}
                >
                  Solicitar acceso
                </a>
              </p>
              {/* Invitado — botón real (prototipo: el único quick-access que se replica) */}
              <button
                type="button"
                data-testid="login-guest"
                onClick={handleGuest}
                disabled={guestLoading || isLoading}
                className={`${GM.ctaGhost} ${GM.focus} mt-3 w-full h-11 text-[13px] font-bold inline-flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {guestLoading
                  ? <><Loader2 className="w-4 h-4 animate-spin" aria-hidden /> {tGuest('guestBusy')}</>
                  : <><UserRound className="w-4 h-4" aria-hidden /> {tGuest('guestCta')}</>}
              </button>
            </div>
          </form>

          {/* Security note */}
          <p className={`mt-4 text-center ${GMT.micro} ${GM.textFaint}`}>
            Sesión protegida · actividad registrada en audit log
          </p>
        </div>
      </main>

      <PublicFooter />
    </div>
  );
}
