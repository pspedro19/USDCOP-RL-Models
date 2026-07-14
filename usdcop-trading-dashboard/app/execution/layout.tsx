'use client';

/**
 * /execution layout — SignalBridge pages inside the GlobalMarkets
 * <TerminalShell active="signalbridge"> chrome (CTR-GM-UI-001).
 *
 * The previous standalone chrome (own sidebar + mobile navbar) is archived at
 * `components/legacy/execution/ExecutionLayoutLegacy.tsx`.
 *
 * Preserved behavior:
 * - localStorage/sessionStorage auth check (SB `auth-token` OR main-app flag)
 *   with redirect to /login?callbackUrl=… — unchanged.
 * - /execution/login and /execution/register render WITHOUT any chrome
 *   (standalone pages), same as before.
 * - Section nav (Dashboard / Exchanges / Executions / Settings) survives as a
 *   GM-styled tab strip at the top of the content column. Logout now lives in
 *   the shell sidebar (it clears the SB token too).
 */

import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { History, LayoutDashboard, Link2, Settings } from 'lucide-react';

import { TerminalShell } from '@/components/gm';
import { GM } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { EXECUTION_ROUTES } from '@/lib/config/execution/constants';
import { EXEC_DICT } from './i18n';

const navItems = [
  { href: EXECUTION_ROUTES.DASHBOARD, labelKey: 'navDashboard' as const, icon: LayoutDashboard },
  { href: EXECUTION_ROUTES.EXCHANGES, labelKey: 'navExchanges' as const, icon: Link2 },
  { href: EXECUTION_ROUTES.EXECUTIONS, labelKey: 'navExecutions' as const, icon: History },
  { href: EXECUTION_ROUTES.SETTINGS, labelKey: 'navSettings' as const, icon: Settings },
];

export default function ExecutionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const t = useGmT(EXEC_DICT);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const isAuthPage =
    pathname === EXECUTION_ROUTES.LOGIN || pathname === EXECUTION_ROUTES.REGISTER;

  // Check authentication - supports both main app auth and execution module auth
  useEffect(() => {
    if (isAuthPage) return;
    const executionToken = localStorage.getItem('auth-token');
    const mainAppAuth =
      localStorage.getItem('isAuthenticated') === 'true' ||
      sessionStorage.getItem('isAuthenticated') === 'true';

    if (executionToken || mainAppAuth) {
      setIsAuthenticated(true);
    } else {
      // Not authenticated, redirect to main login (not execution login)
      router.push('/login?callbackUrl=' + encodeURIComponent(pathname));
    }
  }, [pathname, router, isAuthPage]);

  // Login/register pages keep their own standalone chrome (outside the shell)
  if (isAuthPage) {
    return <>{children}</>;
  }

  // Don't render until auth check completes
  if (!isAuthenticated) {
    return (
      <div className={`min-h-screen ${GM.page} flex items-center justify-center`}>
        <div
          className="motion-safe:animate-spin w-8 h-8 border-2 border-[var(--gm-accent)] border-t-transparent rounded-full"
          role="status"
          aria-label={t('loadingBridge')}
        />
      </div>
    );
  }

  return (
    <TerminalShell active="signalbridge">
      {/* SignalBridge section tabs (replaces the legacy sidebar nav) */}
      <nav aria-label="SignalBridge" className="flex flex-wrap items-center gap-1.5 mb-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
          return (
            <button
              key={item.href}
              onClick={() => router.push(item.href)}
              aria-current={isActive ? 'page' : undefined}
              className={`flex items-center gap-2 h-11 px-4 rounded-[10px] text-[12.5px] font-semibold ${GM.focus} ${
                isActive ? GM.navActive : GM.navIdle
              }`}
            >
              <Icon className="w-4 h-4" aria-hidden />
              {t(item.labelKey)}
            </button>
          );
        })}
      </nav>
      {children}
    </TerminalShell>
  );
}
