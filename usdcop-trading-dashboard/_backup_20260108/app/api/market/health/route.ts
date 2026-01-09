import { NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';

export async function GET() {
  return NextResponse.json(
    createApiResponse(true, 'live', {
      data: {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        market: {
          status: 'open',
          lastUpdate: new Date().toISOString()
        }
      }
    })
  );
}

export async function HEAD() {
  return new NextResponse(null, { status: 200 });
}