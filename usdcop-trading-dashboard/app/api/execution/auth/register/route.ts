/**
 * SignalBridge Auth Register API Route
 * =====================================
 *
 * Proxy route to SignalBridge backend /auth/register
 */

import { NextRequest, NextResponse } from 'next/server';

import { verifyCaptcha } from '@/lib/auth/captcha';

import { SIGNALBRIDGE_BACKEND_URL as BACKEND_URL } from '@/lib/services/execution/bff';

export async function POST(request: NextRequest) {
  try {
    const raw = await request.json();
    // CAPTCHA gate (public endpoint): signed one-time challenge must verify before
    // anything reaches SignalBridge. Captcha fields are stripped from the forward.
    const { captcha_token, captcha_answer, ...body } = raw ?? {};
    if (!verifyCaptcha(captcha_token, captcha_answer)) {
      return NextResponse.json(
        { error: 'captcha inválido o vencido', captcha: true },
        { status: 400 },
      );
    }

    const response = await fetch(`${BACKEND_URL}/api/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] SignalBridge auth register error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
