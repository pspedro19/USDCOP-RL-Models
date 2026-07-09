/**
 * GET /api/captcha — issue a signed arithmetic challenge for the public auth forms
 * (register + login). Public by design; the answer is never in the response, only
 * an HMAC-signed expiring one-time token (lib/auth/captcha.ts).
 */
import { NextResponse } from 'next/server';

import { issueCaptcha } from '@/lib/auth/captcha';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(issueCaptcha(), {
    headers: { 'Cache-Control': 'no-store' },
  });
}
