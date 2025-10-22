// Trading API Proxy
// Proxies all requests to Trading API backend (http://localhost:8000)

import { NextRequest, NextResponse } from 'next/server';

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8000';

// Disable caching for this route - always fetch fresh data
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const url = `${TRADING_API_URL}/${path}${searchParams ? `?${searchParams}` : ''}`;

    console.log('[Trading Proxy] Forwarding GET request to:', url);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Trading API', details: String(error) },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const body = await request.json();
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding POST request to:', url);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Trading API', details: String(error) },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const body = await request.json();
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding PUT request to:', url);

    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Trading API', details: String(error) },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding DELETE request to:', url);

    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Trading API', details: String(error) },
      { status: 500 }
    );
  }
}
