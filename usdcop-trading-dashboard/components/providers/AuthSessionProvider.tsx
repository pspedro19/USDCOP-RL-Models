'use client';

/**
 * NextAuth SessionProvider wrapper (client boundary) — required by the RBAC-driven UI
 * (`useSession` in GlobalNavbar / hub, CTR-RBAC-001 R4). Session is refetched on window
 * focus so role/plan changes propagate without a full reload.
 */
import { SessionProvider } from 'next-auth/react';
import type { ReactNode } from 'react';

export function AuthSessionProvider({ children }: { children: ReactNode }) {
  return <SessionProvider refetchOnWindowFocus>{children}</SessionProvider>;
}
