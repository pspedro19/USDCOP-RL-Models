'use client';

/**
 * /execution/login — redirect shim (audit A8-08).
 *
 * The old page called `authService.login(email, password)` positionally against a
 * `login(data)` signature and read non-existent `result.success/token` — it could
 * never log anyone in. The platform's single sign-in surface is /login (SignalBridge
 * -backed: mints the NextAuth session + SB token). This page now just redirects,
 * preserving the deep-link into the execution dashboard.
 */
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

import { GM, GMT } from '@/lib/ui/gm-tokens';
import { useGmT } from '@/lib/i18n/gm-core';
import { EXEC_DICT } from './../i18n';

export default function ExecutionLoginRedirect() {
  const router = useRouter();
  const t = useGmT(EXEC_DICT);
  useEffect(() => {
    router.replace('/login?callbackUrl=%2Fexecution%2Fdashboard');
  }, [router]);
  return (
    <div className={`min-h-screen ${GM.page} flex items-center justify-center`}>
      <p className={`${GMT.body} ${GM.textSec}`}>{t('redirecting')}</p>
    </div>
  );
}
