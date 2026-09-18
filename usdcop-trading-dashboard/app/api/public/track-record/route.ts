/**
 * GET /api/public/track-record — the FULL public track record (no session required).
 *
 * WHY THIS EXISTS: every performance surface sat behind a login. `/production` needs
 * `signals:read` and the `/data/**` artifacts need a session at the edge, so a prospective
 * client could not see a single number without registering first — the exact friction that
 * kills a sale for a product whose credibility IS the disclosure.
 *
 * The composition lives in `lib/public/track-record` and is shared with the `/track-record`
 * page, which reads it directly instead of calling this route over HTTP. What is public,
 * what is withheld, and the honesty invariants are all documented there.
 */
import { NextResponse } from 'next/server';

import { buildTrackRecord } from '@/lib/public/track-record';

export const revalidate = 300;

export async function GET() {
  return NextResponse.json(await buildTrackRecord());
}
