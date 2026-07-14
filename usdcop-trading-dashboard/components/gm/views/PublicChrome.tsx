'use client';

/**
 * Public chrome for unauthenticated GM pages (CTR-GM-UI-001, prototype Var B
 * lines 93-105) — the landing/pricing header + sober disclaimer footer.
 *
 * NOT TerminalShell: that chrome is for authenticated sections. This header is the
 * public one (brand · Planes · idioma · Iniciar sesión · Crear cuenta).
 * All palette via lib/ui/gm-tokens.ts; all strings via GM_DICT (lib/i18n/gm.ts)
 * with the live ES/EN toggle (setGmLang/useGmLang).
 */
import Link from 'next/link';
import { Languages, Sparkles } from 'lucide-react';

import { GM_DICT } from '@/lib/i18n/gm';
import { setGmLang, useGmLang, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

export function PublicHeader() {
  const lang = useGmLang();
  const t = useGmT(GM_DICT);

  return (
    <header
      data-testid="public-header"
      className={`sticky top-0 z-[var(--z-sticky)] h-[60px] flex items-center gap-3.5 px-5 ${GM.headerBar}`}
    >
      <Link href="/" className={`flex items-center gap-2.5 ${GM.focus}`} aria-label={t('brandHome')}>
        <span className={`w-8 h-8 rounded-[10px] ${GM.brandGradient} flex items-center justify-center shadow-[0_4px_16px_rgba(34,211,238,.35)]`}>
          <Sparkles className="w-4 h-4 text-white" aria-hidden />
        </span>
        <span className={`text-[15px] font-extrabold ${GM.headline} tracking-[-.2px]`}>{t('brand')}</span>
      </Link>

      <div className="flex-1" />

      <Link
        href="/pricing"
        data-testid="public-nav-pricing"
        className={`hidden sm:flex items-center h-11 px-1.5 text-[13px] font-semibold ${GM.textStrong} hover:text-[var(--gm-headline)] ${GM.focus}`}
      >
        {t('plans')}
      </Link>
      <button
        onClick={() => setGmLang(lang === 'es' ? 'en' : 'es')}
        title={t('language')}
        aria-label={t('language')}
        className={`hidden md:flex items-center gap-1.5 h-11 px-3 ${GM.ctaGhost} ${GM.focus} text-[12px] font-semibold font-mono`}
      >
        <Languages className="w-4 h-4" aria-hidden /> {lang.toUpperCase()}
      </button>
      <Link
        href="/login"
        data-testid="public-nav-login"
        className={`${GM.ctaGhost} ${GM.focus} flex items-center h-11 px-3.5 text-[13px] font-bold`}
      >
        {t('signIn')}
      </Link>
      <Link
        href="/register"
        data-testid="public-nav-register"
        className={`${GM.ctaPrimary} ${GM.focus} flex items-center h-11 px-4 text-[13px]`}
      >
        {t('signUp')}
      </Link>
    </header>
  );
}

/**
 * Sober public footer. The risk disclaimer is MANDATORY on every surface with
 * signals (rbac.md §9 — persistent disclaimer); do not remove it.
 */
export function PublicFooter() {
  const t = useGmT(GM_DICT);

  return (
    <footer data-testid="public-footer" className="mt-14 border-t border-[rgba(148,163,184,.10)]">
      <div className="max-w-[1120px] mx-auto px-6 py-8 text-center space-y-3">
        <p className={`${GMT.meta} ${GM.textMuted} max-w-3xl mx-auto leading-relaxed`}>
          {t('disclaimer')}
        </p>
        <nav className={`flex items-center justify-center gap-4 ${GMT.micro} ${GM.textSec}`} aria-label={t('legal')}>
          <Link href="/metodologia" className={`hover:text-[var(--gm-text)] ${GM.focus}`}>{t('methodology')}</Link>
          <span aria-hidden>·</span>
          <Link href="/legal/terminos" className={`hover:text-[var(--gm-text)] ${GM.focus}`}>{t('terms')}</Link>
          <span aria-hidden>·</span>
          <Link href="/pricing" className={`hover:text-[var(--gm-text)] ${GM.focus}`}>{t('plans')}</Link>
        </nav>
        <p className={`${GMT.micro} ${GM.textFaint}`}>{t('copyright')}</p>
      </div>
    </footer>
  );
}
