/**
 * Exchange Connect API Route (create → validate → withdraw gate)
 * ==============================================================
 *
 * POST `/api/execution/exchanges/{exchange}/connect`
 *
 * 1. Creates the credential via SignalBridge `POST /api/exchanges/credentials`.
 * 2. Validates it via `POST /api/exchanges/credentials/{id}/validate`.
 * 3. SECURITY GATE (rbac rule 5): if the key carries WITHDRAW permission, the
 *    just-created credential is DELETED and a rejection is returned. The system
 *    never stores a withdraw-capable key.
 * 4. Returns a proper `ValidationResult` (is_valid, has_withdraw_permission,
 *    can_trade_spot, permissions, balance_check) — the shape the UI branches on.
 *    Previously this route returned the raw credential record, so the UI's
 *    withdraw/balance checks read `undefined` → falsy → silently passed.
 *
 * NOTE (backend limitation, honest): SignalBridge's `/validate` endpoint
 * currently reports a coarse permission set and does not yet surface a distinct
 * withdraw flag for MEXC (its API can't be queried for it). The gate below is
 * wired correctly and fires the moment the backend reports a withdraw
 * permission; until then it relies on `is_valid` + whatever permissions the
 * backend returns.
 */

import { NextRequest, NextResponse } from 'next/server';

import { resolveExecutionIdentity, sbFetch } from '@/lib/services/execution/bff';

interface Params {
  params: Promise<{ exchange: string }>;
}

interface ValidationResult {
  is_valid: boolean;
  exchange: string;
  permissions: string[];
  can_trade_spot: boolean;
  has_withdraw_permission: boolean;
  balance_check?: Record<string, number>;
  error_message?: string;
}

const isWithdrawPerm = (perm: string) => /withdraw/i.test(perm);
const isSpotPerm = (perm: string) => /spot|trad/i.test(perm);

/** Best-effort: fetch balances for a validated credential → `{ asset: total }`. */
async function fetchBalanceCheck(
  credentialId: string,
  authHeader: string | null,
): Promise<Record<string, number> | undefined> {
  try {
    const res = await sbFetch(`/api/exchanges/credentials/${credentialId}/balances`, {
      authHeader,
    });
    if (!res.ok) return undefined;
    const balances = (await res.json()) as Array<{ asset: string; total: number }>;
    const out: Record<string, number> = {};
    for (const b of balances || []) {
      if (b?.asset && typeof b.total === 'number') out[b.asset] = b.total;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  } catch {
    return undefined;
  }
}

async function deleteCredential(credentialId: string, authHeader: string | null) {
  try {
    await sbFetch(`/api/exchanges/credentials/${credentialId}`, {
      method: 'DELETE',
      authHeader,
    });
  } catch (e) {
    console.error('[API] Failed to roll back credential', credentialId, e);
  }
}

export async function POST(request: NextRequest, { params }: Params) {
  try {
    const { exchange } = await params;
    const { userId, authHeader } = await resolveExecutionIdentity(request);
    if (!userId) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const body = await request.json();
    const apiKey: string | undefined = body?.api_key;
    const apiSecret: string | undefined = body?.api_secret;
    if (!apiKey || !apiSecret) {
      return NextResponse.json(
        { error: 'api_key and api_secret are required' },
        { status: 400 },
      );
    }

    // 1. Create the credential (SignalBridge requires exchange + label).
    const createResponse = await sbFetch('/api/exchanges/credentials', {
      method: 'POST',
      authHeader,
      body: {
        exchange,
        label: `${exchange} (${apiKey.slice(0, 4)}…)`,
        is_testnet: body?.is_testnet ?? false,
        api_key: apiKey,
        api_secret: apiSecret,
        ...(body?.passphrase ? { passphrase: body.passphrase } : {}),
      },
    });

    if (!createResponse.ok) {
      const error = await createResponse.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to connect exchange' },
        { status: createResponse.status },
      );
    }

    const credential = (await createResponse.json()) as { id: string };

    // 2. Validate the freshly-stored credential.
    const validateResponse = await sbFetch(
      `/api/exchanges/credentials/${credential.id}/validate`,
      { method: 'POST', authHeader },
    );

    if (!validateResponse.ok) {
      // Could not validate — do not keep an unverified key.
      await deleteCredential(credential.id, authHeader);
      const error = await validateResponse.json().catch(() => ({}));
      const result: ValidationResult = {
        is_valid: false,
        exchange,
        permissions: [],
        can_trade_spot: false,
        has_withdraw_permission: false,
        error_message: error.detail || 'Failed to validate API key',
      };
      return NextResponse.json({ data: result });
    }

    const validation = (await validateResponse.json()) as {
      is_valid: boolean;
      permissions?: string[];
      error_message?: string | null;
    };
    const permissions = validation.permissions ?? [];
    const hasWithdraw = permissions.some(isWithdrawPerm);
    const canTradeSpot = permissions.some(isSpotPerm) || validation.is_valid;

    // 2a. Invalid key → roll back, surface the reason.
    if (!validation.is_valid) {
      await deleteCredential(credential.id, authHeader);
      const result: ValidationResult = {
        is_valid: false,
        exchange,
        permissions,
        can_trade_spot: false,
        has_withdraw_permission: hasWithdraw,
        error_message: validation.error_message || 'API key validation failed',
      };
      return NextResponse.json({ data: result });
    }

    // 3. SECURITY GATE — reject withdraw-capable keys (never store them).
    if (hasWithdraw) {
      await deleteCredential(credential.id, authHeader);
      const result: ValidationResult = {
        is_valid: true,
        exchange,
        permissions,
        can_trade_spot: canTradeSpot,
        has_withdraw_permission: true,
        error_message:
          'API key has WITHDRAW permission and was rejected. Create a trading-only key.',
      };
      return NextResponse.json({ data: result });
    }

    // 4. Valid, trading-only key → keep it. Attach balances best-effort.
    const balanceCheck = await fetchBalanceCheck(credential.id, authHeader);
    const result: ValidationResult = {
      is_valid: true,
      exchange,
      permissions,
      can_trade_spot: canTradeSpot,
      has_withdraw_permission: false,
      ...(balanceCheck ? { balance_check: balanceCheck } : {}),
    };
    return NextResponse.json({ data: result });
  } catch (error) {
    console.error('[API] Exchange connect error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
