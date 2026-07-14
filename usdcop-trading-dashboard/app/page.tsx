'use client';

/**
 * '/' — public landing on the GlobalMarkets design system (CTR-GM-UI-001).
 * Public per PAGE_ROUTES ('/', exact). Previous landing archived at /legacy/landing
 * (components/legacy/LandingLegacy.tsx, admin-only).
 */
import { LandingView } from '@/components/gm/views/LandingView';

export default function LandingPage() {
  return <LandingView />;
}
