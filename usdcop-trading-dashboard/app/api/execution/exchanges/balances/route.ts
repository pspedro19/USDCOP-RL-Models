/**
 * Exchange Balances API Route
 * ===========================
 *
 * Returns balances for all connected exchanges.
 * Aggregates balances from all credentials.
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');

    // First get all credentials
    const credentialsResponse = await fetch(`${BACKEND_URL}/api/exchanges/credentials`, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!credentialsResponse.ok) {
      // If credentials fail, return empty balances
      return NextResponse.json({ data: [] });
    }

    const credentials = await credentialsResponse.json();

    // If no credentials, return empty balances
    if (!credentials || credentials.length === 0) {
      return NextResponse.json({ data: [] });
    }

    // Get balances for each credential
    const balancesPromises = credentials.map(async (cred: { id: string; exchange: string }) => {
      try {
        const balanceResponse = await fetch(
          `${BACKEND_URL}/api/exchanges/credentials/${cred.id}/balances`,
          {
            headers: {
              'Content-Type': 'application/json',
              ...(authHeader && { Authorization: authHeader }),
            },
          }
        );

        if (!balanceResponse.ok) {
          return {
            exchange: cred.exchange,
            balances: [],
            total_usd: 0,
            updated_at: new Date().toISOString(),
          };
        }

        const balances = await balanceResponse.json();

        // Calculate total USD
        const totalUsd = balances.reduce((acc: number, b: { usd_value?: number }) =>
          acc + (b.usd_value || 0), 0
        );

        return {
          exchange: cred.exchange,
          balances: balances,
          total_usd: totalUsd,
          updated_at: new Date().toISOString(),
        };
      } catch (e) {
        console.error(`[API] Failed to get balances for ${cred.exchange}:`, e);
        return {
          exchange: cred.exchange,
          balances: [],
          total_usd: 0,
          updated_at: new Date().toISOString(),
        };
      }
    });

    const allBalances = await Promise.all(balancesPromises);
    return NextResponse.json({ data: allBalances });
  } catch (error) {
    console.error('[API] Exchange balances error:', error);
    // Return empty balances on error to prevent UI from breaking
    return NextResponse.json({ data: [] });
  }
}
