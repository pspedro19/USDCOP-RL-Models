// Analytics API Proxy
// Proxies all requests to Analytics API backend (http://localhost:8001)

import { NextRequest, NextResponse } from 'next/server';

const ANALYTICS_API_URL = process.env.ANALYTICS_API_URL || 'http://localhost:8001';

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
    const url = `${ANALYTICS_API_URL}/${path}${searchParams ? `?${searchParams}` : ''}`;

    console.log('[Analytics Proxy] Forwarding GET request to:', url);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Analytics Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Analytics API', details: String(error) },
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
    const url = `${ANALYTICS_API_URL}/${path}`;

    console.log('[Analytics Proxy] Forwarding POST request to:', url);

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
    console.error('[Analytics Proxy] Error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from Analytics API', details: String(error) },
      { status: 500 }
    );
  }
}
